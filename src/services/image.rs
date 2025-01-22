use crate::models::{
    GenerationConfig, GenerationResult, ImageFormat, NFTError, PerformanceMetrics, Result,
};
use image::imageops::FilterType;
use image::{Rgb, RgbImage};
use ndarray::{s, Array, Array0, Array3, Array4, ArrayD, Axis, CowArray, IxDyn};
use ort::{
    tensor::OrtOwnedTensor, Environment, ExecutionProvider, GraphOptimizationLevel, Session,
    SessionBuilder, Value,
};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;

const LATENT_CHANNELS: usize = 4;
const LATENT_HEIGHT: usize = 64;
const LATENT_WIDTH: usize = 64;
const MAX_TEXT_LENGTH: usize = 77;

struct MemoryTracker {
    start_memory: f64,
    peak_memory: f64,
}

impl MemoryTracker {
    fn new() -> Self {
        Self {
            start_memory: 0.0,
            peak_memory: 0.0,
        }
    }

    fn get_peak_memory(&self) -> f64 {
        self.peak_memory
    }

    fn get_device_utilization(&self) -> f32 {
        0.0 // TODO: Implement actual GPU utilization tracking
    }
}

struct PNDMScheduler {
    timesteps: Vec<i64>,
    alphas: Vec<f32>,
    alphas_cumprod: Vec<f32>,
    final_alpha_cumprod: f32,
    num_inference_steps: usize,
    num_train_timesteps: i64,
}

impl PNDMScheduler {
    fn new(num_inference_steps: usize) -> Self {
        let num_train_timesteps = 1000i64;
        let beta_start = 0.00085f32;
        let beta_end = 0.012f32;

        // Calculate betas using scaled linear schedule
        let mut betas = Vec::with_capacity(num_train_timesteps as usize);
        for i in 0..num_train_timesteps {
            let t = i as f32 / (num_train_timesteps - 1) as f32;
            let scaled_beta = beta_start + t * (beta_end - beta_start);
            betas.push(scaled_beta);
        }

        // Calculate alphas and cumulative products
        let alphas: Vec<f32> = betas.iter().map(|beta| 1.0 - beta).collect();
        let mut alphas_cumprod = Vec::with_capacity(num_train_timesteps as usize);
        let mut cumprod = 1.0f32;
        for alpha in alphas.iter() {
            cumprod *= alpha;
            alphas_cumprod.push(cumprod);
        }

        // Calculate PNDM specific timesteps
        let mut timesteps = Vec::new();
        let step_size = num_train_timesteps / (num_inference_steps as i64);

        // Start from the highest timestep
        for i in (0..=num_inference_steps).rev() {
            timesteps.push(i as i64 * step_size);
        }
        let mut base_timesteps: Vec<i64> = Vec::new();
        // Generate base timesteps (half of target since we'll duplicate)
        for i in (0..=num_inference_steps / 2).rev() {
            base_timesteps.push((i as i64 * step_size * 2).min(num_train_timesteps - 1));
        }

        // Create final timesteps with duplicates and ensure we end at 0
        let mut final_timesteps = Vec::new();
        for &t in &base_timesteps[..base_timesteps.len() - 1] {
            final_timesteps.push(t);
            final_timesteps.push(t);
        }
        final_timesteps.push(0); // Ensure we end at 0
        let copied_cumprod = alphas_cumprod.clone();
        Self {
            timesteps: final_timesteps,
            alphas,
            alphas_cumprod: copied_cumprod,
            final_alpha_cumprod: alphas_cumprod[alphas_cumprod.len() - 1],
            num_inference_steps,
            num_train_timesteps,
        }
    }

    fn timesteps(&self) -> &[i64] {
        &self.timesteps
    }

    fn step(
        &self,
        noise_pred: Array4<f32>,
        timestep: i64,
        latents: &Array4<f32>,
        guidance_scale: f32,
    ) -> Result<Array4<f32>> {
        // Split noise predictions
        let noise_pred_uncond = noise_pred.slice(s![0..1, .., .., ..]).to_owned();
        let noise_pred_text = noise_pred.slice(s![1..2, .., .., ..]).to_owned();

        // Apply classifier-free guidance
        let noise_pred =
            &noise_pred_uncond + guidance_scale * (&noise_pred_text - &noise_pred_uncond);

        // Convert timestep to index in our schedule
        let timestep_idx = if timestep == 0 {
            0
        } else {
            ((timestep - 1) / (self.num_train_timesteps / 1000)) as usize
        };
        let timestep_idx = timestep_idx.min(self.alphas_cumprod.len() - 1);

        // Get alpha values
        let alpha_prod_t = self.alphas_cumprod[timestep_idx];
        let alpha_prod_t_prev = if timestep_idx > 0 {
            self.alphas_cumprod[timestep_idx - 1]
        } else {
            1.0
        };

        let beta_prod_t = 1.0 - alpha_prod_t;
        let beta_prod_t_prev = 1.0 - alpha_prod_t_prev;

        // Compute predicted original sample from predicted noise
        let pred_original_sample =
            (latents - (beta_prod_t.sqrt() * &noise_pred)) / alpha_prod_t.sqrt();

        // Direction pointing to x_t
        let dir_xt = (beta_prod_t_prev.sqrt()) * noise_pred;

        // Random noise for stochastic sampling
        let noise_shape = latents.raw_dim();
        let noise = match rand_distr::Normal::new(0.0f32, 1.0) {
            Ok(dist) => {
                let mut rng = rand::thread_rng();
                Array4::from_shape_simple_fn(noise_shape, || dist.sample(&mut rng))
            }
            Err(_) => {
                return Err(NFTError::ProcessingError(
                    "Failed to create noise distribution".to_string(),
                ))
            }
        };

        // Add noise scaled by the variance
        let variance =
            ((beta_prod_t_prev / beta_prod_t) * (1.0 - alpha_prod_t / alpha_prod_t_prev)).sqrt();

        let prev_sample =
            alpha_prod_t_prev.sqrt() * pred_original_sample + dir_xt + variance * noise;

        Ok(prev_sample)
    }
}

pub struct ImageService {
    text_encoder: Session,
    unet: Session,
    vae_decoder: Session,
    tokenizer: Tokenizer,
    output_path: PathBuf,
}

impl ImageService {
    fn unet_inference(
        &self,
        latent_input: &Array4<f32>,
        text_embeddings: &Array3<f32>,
        timestep: i64,
    ) -> Result<Array4<f32>> {
        let text_embeddings_dyn = text_embeddings.clone().into_dyn();
        let latent_input_dyn = latent_input.clone().into_dyn();

        // Create timestep array
        let timestep_array =
            Array::from_shape_vec(ndarray::IxDyn(&[1]), vec![timestep]).map_err(|e| {
                NFTError::ProcessingError(format!("Failed to create timestep tensor: {}", e))
            })?;

        // Create CowArrays first with let bindings
        let latent_cow = CowArray::from(&latent_input_dyn);
        let timestep_cow = CowArray::from(&timestep_array);
        let text_cow = CowArray::from(&text_embeddings_dyn);

        // Then create the tensors
        let latent_tensor = Value::from_array(self.unet.allocator(), &latent_cow).map_err(|e| {
            NFTError::ProcessingError(format!("Failed to create sample tensor: {}", e))
        })?;

        let timestep_tensor =
            Value::from_array(self.unet.allocator(), &timestep_cow).map_err(|e| {
                NFTError::ProcessingError(format!("Failed to create timestep tensor: {}", e))
            })?;

        let text_tensor = Value::from_array(self.unet.allocator(), &text_cow).map_err(|e| {
            NFTError::ProcessingError(format!(
                "Failed to create encoder_hidden_states tensor: {}",
                e
            ))
        })?;

        // Run UNet inference - order must match ONNX input order
        let outputs = self
            .unet
            .run(vec![latent_tensor, timestep_tensor, text_tensor])
            .map_err(|e| NFTError::ProcessingError(format!("UNet inference failed: {}", e)))?;

        let extracted_tensor: OrtOwnedTensor<f32, _> = outputs[0].try_extract().map_err(|e| {
            NFTError::ProcessingError(format!("Failed to extract out_sample: {}", e))
        })?;

        let noise_pred_array = Array::from_iter(extracted_tensor.view().iter().copied())
            .into_shape(latent_input.raw_dim())
            .map_err(|e| {
                NFTError::ProcessingError(format!("Failed to reshape UNet output: {}", e))
            })?;

        Ok(noise_pred_array)
    }

    pub fn new(output_path: PathBuf) -> Result<Self> {
        let env = Arc::new(
            Environment::builder()
                .with_name("stable_diffusion")
                .with_execution_providers([
                    ExecutionProvider::CUDA(Default::default()),
                    ExecutionProvider::CPU(Default::default()),
                ])
                .build()
                .map_err(|e| NFTError::ModelLoadError(e.to_string().into()))?,
        );

        // Use the onnx_sd directory that we created with the Python script
        let base_path = PathBuf::from("onnx_sd");
        let text_encoder_path = base_path.join("text_encoder").join("model.onnx");
        let unet_path = base_path.join("unet").join("model.onnx");
        let vae_decoder_path = base_path.join("vae_decoder").join("model.onnx");

        // Use the CLIP tokenizer from the converted model
        let tokenizer_path = base_path.join("tokenizer").join("tokenizer.json");

        // Verify all required files exist
        if !text_encoder_path.exists() {
            return Err(NFTError::ModelLoadError(
                format!("Text encoder not found at: {}", text_encoder_path.display()).into(),
            ));
        }
        if !unet_path.exists() {
            return Err(NFTError::ModelLoadError(
                format!("UNet not found at: {}", unet_path.display()).into(),
            ));
        }
        if !vae_decoder_path.exists() {
            return Err(NFTError::ModelLoadError(
                format!("VAE decoder not found at: {}", vae_decoder_path.display()).into(),
            ));
        }
        if !tokenizer_path.exists() {
            return Err(NFTError::ModelLoadError(
                format!("Tokenizer not found at: {}", tokenizer_path.display()).into(),
            ));
        }

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            NFTError::ModelLoadError(format!("Failed to load tokenizer: {}", e).into())
        })?;

        // Load model components
        let text_encoder = SessionBuilder::new(&env)
            .map_err(|e| NFTError::ModelLoadError(e.to_string().into()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| NFTError::ModelLoadError(e.to_string().into()))?
            .with_intra_threads(1)
            .map_err(|e| NFTError::ModelLoadError(e.to_string().into()))?
            .with_model_from_file(&text_encoder_path)
            .map_err(|e| NFTError::ModelLoadError(e.to_string().into()))?;

        let unet = SessionBuilder::new(&env)
            .map_err(|e| NFTError::ModelLoadError(e.to_string().into()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| NFTError::ModelLoadError(e.to_string().into()))?
            .with_intra_threads(1)
            .map_err(|e| NFTError::ModelLoadError(e.to_string().into()))?
            .with_model_from_file(&unet_path)
            .map_err(|e| NFTError::ModelLoadError(e.to_string().into()))?;

        let vae_decoder = SessionBuilder::new(&env)
            .map_err(|e| NFTError::ModelLoadError(e.to_string().into()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| NFTError::ModelLoadError(e.to_string().into()))?
            .with_intra_threads(1)
            .map_err(|e| NFTError::ModelLoadError(e.to_string().into()))?
            .with_model_from_file(&vae_decoder_path)
            .map_err(|e| NFTError::ModelLoadError(e.to_string().into()))?;

        Ok(Self {
            text_encoder,
            unet,
            vae_decoder,
            tokenizer,
            output_path,
        })
    }

    pub fn generate(&self, config: &GenerationConfig) -> Result<GenerationResult> {
        let start_time = Instant::now();
        let memory_tracker = MemoryTracker::new();

        // Encode text prompt
        let text_embeddings = self.encode_prompt(
            &config.parameters.prompt,
            config.parameters.negative_prompt.as_deref().unwrap_or(""),
        )?;

        let batch_size = 1;
        let mut latents = self.initialize_latents(batch_size, config.parameters.seed)?;

        // Setup scheduler
        let scheduler = PNDMScheduler::new(config.parameters.num_inference_steps as usize);

        // Diffusion process
        for (step_idx, &timestep) in scheduler.timesteps().iter().enumerate() {
            println!(
                "Processing timestep {}/{}: {}",
                step_idx + 1,
                scheduler.timesteps().len(),
                timestep
            );

            // Expand latents for classifier-free guidance
            let latent_input = self.prepare_latent_input(&latents)?;

            // Run UNet inference
            let noise_pred = self.unet_inference(&latent_input, &text_embeddings, timestep)?;

            // Scheduler step
            latents = scheduler.step(
                noise_pred,
                timestep,
                &latents,
                config.parameters.guidance_scale,
            )?;

            println!(
                "Latents range: {} to {}",
                latents.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
                latents.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
            );
        }

        // Decode latents to image
        let image = self.decode_latents(&latents)?;

        // Save image
        let output_path = self.save_image(&image, config.output_config.format.clone())?;

        Ok(GenerationResult {
            image_path: output_path,
            generation_params: config.parameters.clone(),
            performance_metrics: PerformanceMetrics {
                generation_time_ms: start_time.elapsed().as_millis() as u64,
                memory_used_mb: memory_tracker.get_peak_memory(),
                device_utilization: memory_tracker.get_device_utilization(),
            },
        })
    }

    fn encode_prompt(&self, prompt: &str, negative_prompt: &str) -> Result<Array3<f32>> {
        // Tokenize prompts
        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| NFTError::ProcessingError(format!("Tokenization failed: {:?}", e)))?;
        let neg_tokens = self.tokenizer.encode(negative_prompt, true).map_err(|e| {
            NFTError::ProcessingError(format!("Negative tokenization failed: {:?}", e))
        })?;

        let token_ids = tokens.get_ids();
        let neg_token_ids = neg_tokens.get_ids();

        // Create input array with shape [2, MAX_TEXT_LENGTH]
        let mut input_ids = vec![0i32; MAX_TEXT_LENGTH * 2];
        let token_len = token_ids.len().min(MAX_TEXT_LENGTH);
        let neg_token_len = neg_token_ids.len().min(MAX_TEXT_LENGTH);

        // Convert to i32 and copy to input array
        input_ids[..token_len].copy_from_slice(
            &token_ids[..token_len]
                .iter()
                .map(|&x| x as i32)
                .collect::<Vec<_>>(),
        );
        input_ids[MAX_TEXT_LENGTH..MAX_TEXT_LENGTH + neg_token_len].copy_from_slice(
            &neg_token_ids[..neg_token_len]
                .iter()
                .map(|&x| x as i32)
                .collect::<Vec<_>>(),
        );

        let input_tensor = ndarray::Array2::from_shape_vec((2, MAX_TEXT_LENGTH), input_ids)
            .map_err(|e| {
                NFTError::ProcessingError(format!("Failed to create input tensor: {:?}", e))
            })?;

        let input_tensor_dyn = input_tensor.into_dyn();
        let binding = CowArray::from(&input_tensor_dyn);
        let input = Value::from_array(self.text_encoder.allocator(), &binding).map_err(|e| {
            NFTError::ProcessingError(format!("Failed to create input value: {:?}", e))
        })?;

        // Output name is "last_hidden_state"
        let outputs = self.text_encoder.run(vec![input]).map_err(|e| {
            NFTError::ProcessingError(format!("Text encoder inference failed: {:?}", e))
        })?;

        let extracted_tensor: OrtOwnedTensor<f32, _> = outputs[0].try_extract().map_err(|e| {
            NFTError::ProcessingError(format!("Failed to extract last_hidden_state: {:?}", e))
        })?;

        // Shape to [2, 77, 768]
        let embeddings = Array::from_iter(extracted_tensor.view().iter().copied())
            .into_shape((2, MAX_TEXT_LENGTH, 768))
            .map_err(|e| {
                NFTError::ProcessingError(format!("Failed to reshape text embeddings: {:?}", e))
            })?;

        Ok(embeddings)
    }

    fn initialize_latents(&self, batch_size: usize, seed: Option<u64>) -> Result<Array4<f32>> {
        let mut rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };

        let normal = Normal::new(0.0, 1.0).map_err(|_e| {
            NFTError::ProcessingError("Failed to create normal distribution".to_string())
        })?;

        let shape = [1, LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH];
        let mut latents = Array4::<f32>::zeros(shape);
        for item in latents.iter_mut() {
            *item = normal.sample(&mut rng) as f32;
        }
        latents = latents.mapv(|x| x / 0.18215);
        Ok(latents)
    }

    fn prepare_latent_input(&self, latents: &Array4<f32>) -> Result<Array4<f32>> {
        let mut duplicated =
            Array4::<f32>::zeros([2, LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH]);
        duplicated.slice_mut(s![0..1, .., .., ..]).assign(&latents);
        duplicated.slice_mut(s![1..2, .., .., ..]).assign(&latents);
        Ok(duplicated)
    }

    fn decode_latents(&self, latents: &Array4<f32>) -> Result<RgbImage> {
        // Scale latents for VAE
        let scaled_latents = latents.mapv(|x| x / 0.18215);
        let latents_slice = scaled_latents.slice(s![0..1, .., .., ..]).to_owned();

        let latents_dyn = latents_slice.into_dyn();
        let latents_cow = CowArray::from(&latents_dyn);

        // Input name is "latent_sample"
        let vae_input =
            Value::from_array(self.vae_decoder.allocator(), &latents_cow).map_err(|_e| {
                NFTError::ProcessingError("Failed to create latent_sample tensor".to_string())
            })?;

        // Output name is "sample"
        let outputs = self
            .vae_decoder
            .run(vec![vae_input])
            .map_err(|_e| NFTError::ProcessingError("VAE decoding failed".to_string()))?;

        let extracted_tensor: OrtOwnedTensor<f32, _> = outputs[0]
            .try_extract()
            .map_err(|_e| NFTError::ProcessingError("Failed to extract sample".to_string()))?;

        // Shape to [1, 3, 512, 512]
        let shape = (1, 3, 512, 512);
        let image_array = Array::from_iter(extracted_tensor.view().iter().copied())
            .into_shape(shape)
            .map_err(|e| {
                NFTError::ProcessingError(format!("Failed to reshape decoded image: {:?}", e))
            })?;

        self.array_to_image(&image_array)
    }

    fn array_to_image(&self, array: &Array4<f32>) -> Result<RgbImage> {
        let (batch, channels, height, width) = array.dim();
        assert_eq!(channels, 3, "Expected RGB image with 3 channels");
        assert_eq!(batch, 1, "Expected single batch");

        let mut img = RgbImage::new(width as u32, height as u32);

        for y in 0..height {
            for x in 0..width {
                let r = ((array[[0, 0, y, x]] + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0;
                let g = ((array[[0, 1, y, x]] + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0;
                let b = ((array[[0, 2, y, x]] + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0;

                img.put_pixel(x as u32, y as u32, Rgb([r as u8, g as u8, b as u8]));
            }
        }

        Ok(img)
    }

    fn save_image(&self, image: &RgbImage, format: ImageFormat) -> Result<PathBuf> {
        // Create output directory if it doesn't exist
        std::fs::create_dir_all(&self.output_path).map_err(|e| NFTError::FileSystemError(e))?;

        // Generate filename
        let filename = format!(
            "nft_{}.{}",
            uuid::Uuid::new_v4(),
            match format {
                ImageFormat::PNG => "png",
                ImageFormat::JPEG => "jpg",
                ImageFormat::WEBP => "webp",
            }
        );
        let output_path = self.output_path.join(filename);

        // Save with specified format
        match format {
            ImageFormat::PNG => {
                image
                    .save_with_format(&output_path, image::ImageFormat::Png)
                    .map_err(|e| {
                        NFTError::FileSystemError(std::io::Error::new(std::io::ErrorKind::Other, e))
                    })?;
            }
            ImageFormat::JPEG => {
                image
                    .save_with_format(&output_path, image::ImageFormat::Jpeg)
                    .map_err(|e| {
                        NFTError::FileSystemError(std::io::Error::new(std::io::ErrorKind::Other, e))
                    })?;
            }
            ImageFormat::WEBP => {
                image
                    .save_with_format(&output_path, image::ImageFormat::WebP)
                    .map_err(|e| {
                        NFTError::FileSystemError(std::io::Error::new(std::io::ErrorKind::Other, e))
                    })?;
            }
        }

        Ok(output_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialize_latents() {
        let service = create_test_service();
        let latents = service.initialize_latents(1, Some(42)).unwrap();
        assert_eq!(
            latents.shape(),
            &[1, LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH]
        );
    }

    #[test]
    fn test_prepare_latent_input() {
        let service = create_test_service();
        let input = Array4::<f32>::zeros((1, LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH));
        let scaled = service.prepare_latent_input(&input).unwrap();
        assert_eq!(scaled.shape(), input.shape());
    }

    #[test]
    fn test_array_to_image() {
        let service = create_test_service();
        let array = Array4::<f32>::zeros((1, 3, 512, 512));
        let image = service.array_to_image(&array).unwrap();
        assert_eq!(image.dimensions(), (512, 512));
    }

    // Helper function to create a test instance
    fn create_test_service() -> ImageService {
        ImageService::new(PathBuf::from("test_output")).unwrap()
    }
}
