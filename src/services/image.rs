use crate::models::{
    GenerationConfig, GenerationResult, ImageFormat, NFTError, PerformanceMetrics, Result
};
use std::{path::PathBuf};
use std::time::Instant;
use ort::{tensor::OrtOwnedTensor, Environment, ExecutionProvider, GraphOptimizationLevel, Session, SessionBuilder, Value};
use tokenizers::Tokenizer;
use ndarray::{s, Array, Array0, Array3, Array4, ArrayD, Axis, CowArray, IxDyn};
use std::sync::Arc;
use image::{Rgb, RgbImage};
use rand::{SeedableRng};
use rand_distr::{Normal, Distribution};

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

struct DiffusionScheduler {
    timesteps: Vec<f32>,
    alphas: Vec<f32>,
    alphas_cumprod: Vec<f32>,
}

impl DiffusionScheduler {
    fn new(num_inference_steps: usize) -> Self {
        // Initialize scheduler parameters
        let beta_start = 0.00085f32;
        let beta_end = 0.012f32;
        let num_train_timesteps = 1000;

        let mut betas = Vec::with_capacity(num_train_timesteps);
        for i in 0..num_train_timesteps {
            let beta = beta_start + (beta_end - beta_start) * (i as f32 / num_train_timesteps as f32);
            betas.push(beta);
        }

        let alphas: Vec<f32> = betas.iter().map(|beta| 1.0 - beta).collect();
        let mut alphas_cumprod = Vec::with_capacity(num_train_timesteps);
        let mut cumprod = 1.0;
        for alpha in alphas.iter() {
            cumprod *= alpha;
            alphas_cumprod.push(cumprod);
        }

        // Calculate timesteps
        let step_size = num_train_timesteps / num_inference_steps;
        let timesteps: Vec<f32> = (0..num_inference_steps)
            .map(|i| (num_train_timesteps - (i + 1) * step_size) as f32)
            .collect();

        Self {
            timesteps,
            alphas,
            alphas_cumprod,
        }
    }

    fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    fn step(
        &self,
        noise_pred: Array4<f32>,
        timestep: f32,
        latents: &Array4<f32>,
        guidance_scale: f32,
    ) -> Result<Array4<f32>> {
        // Let's add debug prints
        println!("Step input shapes:");
        println!("noise_pred: {:?}", noise_pred.shape());
        println!("latents: {:?}", latents.shape());
    
        // Split noise prediction for guidance - this is where it might be failing
        let (noise_pred_uncond, noise_pred_text) = noise_pred.view().split_at(Axis(0), 1);
        
        println!("Split shapes:");
        println!("uncond: {:?}", noise_pred_uncond.shape());
        println!("text: {:?}", noise_pred_text.shape());
        
        // Try a different approach to the guidance calculation
        let mut noise_pred_combined = noise_pred_uncond.to_owned();
        noise_pred_combined = &noise_pred_combined + 
            (guidance_scale * (&noise_pred_text - &noise_pred_uncond));
    
        // Get timestep index
        let timestep_index = self.timesteps.iter()
            .position(|&t| t == timestep)
            .ok_or_else(|| NFTError::ProcessingError("Invalid timestep".to_string()))?;
        
        let alpha = self.alphas[timestep_index];
        let alpha_prod = self.alphas_cumprod[timestep_index];
        let beta = 1.0 - alpha;
    
        let prev_timestep_index = if timestep_index > 0 { timestep_index - 1 } else { 0 };
        let alpha_prod_prev = self.alphas_cumprod[prev_timestep_index];
    
        let variance = beta * (1.0 - alpha_prod_prev) / (1.0 - alpha_prod);
        
        // Create the denoised sample
        let pred_original_sample = (latents - variance.sqrt() * &noise_pred_combined) / alpha_prod.sqrt();
        
        // Calculate previous sample
        let mut prev_sample = alpha_prod_prev.sqrt() * &pred_original_sample +
            variance.sqrt() * &noise_pred_combined;
    
        if timestep_index > 0 {
            let noise_shape = prev_sample.raw_dim();
            let normal = Normal::new(0.0, 1.0)
                .map_err(|_| NFTError::ProcessingError("Failed to create normal distribution".to_string()))?;
            
            let mut rng = rand::thread_rng();
            let noise = Array4::from_shape_simple_fn(noise_shape, || normal.sample(&mut rng) as f32);
            prev_sample = prev_sample + variance.sqrt() * noise;
        }
    
        println!("Output shape: {:?}", prev_sample.shape());
        
        Ok(prev_sample)
    }
}

pub struct ImageService {
    env: Arc<Environment>,
    text_encoder: Session,
    unet: Session,
    vae_decoder: Session,
    tokenizer: Tokenizer,
    output_path: PathBuf, 
    timestep_tensor_owned: Option<Array<f32, IxDyn>>, 
}

impl ImageService {
    fn unet_inference(
        &self,
        latent_input: &Array4<f32>,
        text_embeddings: &Array3<f32>,
        timestep: f32,
    ) -> Result<Array4<f32>> {
        // Print original shapes
        println!("Original shapes:");
        println!("Latent input: {:?}", latent_input.shape());
        println!("Text embeddings: {:?}", text_embeddings.shape());
    
        // Ensure text embeddings are in batch size 1 for UNet
        let batch_size = 1;
        let seq_length = text_embeddings.shape()[1];
        let hidden_size = text_embeddings.shape()[2];
        
        let text_embeddings_reshaped = text_embeddings
    .slice(s![..1, .., ..])
    .to_owned()
    .into_shape((batch_size, seq_length, hidden_size))
    .map_err(|e| NFTError::ProcessingError(format!("Failed to reshape text embeddings: {}", e)))?;


println!("Reshaped embeddings shape: {:?}", text_embeddings_reshaped.shape());
    
        // Convert to dynamic arrays
        let latent_input_dyn = latent_input.clone().into_dyn();
        let text_embeddings_dyn = text_embeddings_reshaped.into_dyn();
        
        // Create timestep array
        let timestep_array = Array::from_shape_vec(
            ndarray::IxDyn(&[1]),
            vec![timestep]
        ).map_err(|e| NFTError::ProcessingError(format!("Failed to create timestep tensor: {}", e)))?;
    
        // Create CowArrays
        let latent_cow = CowArray::from(&latent_input_dyn);
        let text_cow = CowArray::from(&text_embeddings_dyn);
        let timestep_cow = CowArray::from(&timestep_array);
    
        println!("Transformed shapes:");
        println!("Latent input: {:?}", latent_cow.shape());
        println!("Text embeddings: {:?}", text_cow.shape());
        println!("Timestep: {:?}", timestep_cow.shape());
    
        // Create ONNX tensors
        let latent_tensor = Value::from_array(self.unet.allocator(), &latent_cow)
            .map_err(|e| NFTError::ProcessingError(format!("Failed to create UNet input tensor: {}", e)))?;
        
        let timestep_tensor = Value::from_array(self.unet.allocator(), &timestep_cow)
            .map_err(|e| NFTError::ProcessingError(format!("Failed to create UNet timestep tensor: {}", e)))?;
        
        let text_tensor = Value::from_array(self.unet.allocator(), &text_cow)
            .map_err(|e| NFTError::ProcessingError(format!("Failed to create UNet text input tensor: {}", e)))?;
    
        // Run UNet with correct input order
        let outputs = self.unet.run(vec![latent_tensor, timestep_tensor, text_tensor])
            .map_err(|e| NFTError::ProcessingError(format!("UNet inference failed: {}", e)))?;
    
            let extracted_tensor: OrtOwnedTensor<f32, _> = outputs[0]
            .try_extract()
            .map_err(|e| NFTError::ProcessingError(format!("Failed to extract UNet output: {}", e)))?;
    
        let view = extracted_tensor.view();
        let mut noise_pred_array = Array4::<f32>::zeros((1, 4, 64, 64));
        
        // Copy data using nested iteration
        for n in 0..1 {
            for c in 0..4 {
                for h in 0..64 {
                    for w in 0..64 {
                        let idx = n * 16384 + c * 4096 + h * 64 + w;
                        if let Some(&val) = view.as_slice().and_then(|s| s.get(idx)) {
                            noise_pred_array[[n, c, h, w]] = val;
                        }
                    }
                }
            }
        }
        
        Ok(noise_pred_array)
    
    }
    
    pub fn new(model_path: PathBuf, output_path: PathBuf) -> Result<Self> {

    let env = Arc::new(Environment::builder()
    .with_name("stable_diffusion")
    .with_execution_providers([
        ExecutionProvider::CUDA(Default::default()),
        ExecutionProvider::CPU(Default::default()),
    ])
    .build()
    .map_err(|e| NFTError::ModelLoadError(e.to_string().into()))?);

   
        let text_encoder_path = model_path.join("text_encoder").join("model.onnx");
        let unet_path = model_path.join("unet").join("model.onnx");
        let vae_decoder_path = model_path.join("vae_decoder").join("model.onnx");
        let mut tokenizer_path = PathBuf::from("models/tokenizer/tokenizer.json");
        if !tokenizer_path.exists() {
            let alt_path = PathBuf::from("models/tokenizer/vocab.json");
            if alt_path.exists() {
                tokenizer_path = alt_path;
            } else {
                return Err(NFTError::ModelLoadError("Tokenizer not found!".to_string().into()));
            }
        }

        // Verify all required files exist
        if !text_encoder_path.exists() {
            return Err(NFTError::ModelLoadError(format!(
                "Text encoder not found at: {}",
                text_encoder_path.display()
            ).into()));
        }
        if !unet_path.exists() {
            return Err(NFTError::ModelLoadError(format!(
                "UNet not found at: {}",
                unet_path.display()
            ).into()));
        }
        if !vae_decoder_path.exists() {
            return Err(NFTError::ModelLoadError(format!(
                "VAE decoder not found at: {}",
                vae_decoder_path.display()
            ).into()));
        }
        if !tokenizer_path.exists() {
            return Err(NFTError::ModelLoadError(format!(
                "Tokenizer not found at: {}",
                tokenizer_path.display()
            ).into()));
        }

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| NFTError::ModelLoadError(format!("Failed to load tokenizer: {}", e).into()))?;

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
            env,
            text_encoder,
            unet,
            vae_decoder,
            tokenizer,
            output_path,  
            timestep_tensor_owned: None,
        })
    }

    pub fn generate(&self, config: &GenerationConfig) -> Result<GenerationResult> {
        let start_time = Instant::now();
        let memory_tracker = MemoryTracker::new();

        // Encode text prompt
        let text_embeddings = self.encode_prompt(
            &config.parameters.prompt,
            config.parameters.negative_prompt.as_deref().unwrap_or("")
        )?;

        // Initialize latents
        let mut latents = self.initialize_latents(
            1,
            config.parameters.seed
        )?;

        // Setup scheduler
        let scheduler = DiffusionScheduler::new(
            config.parameters.num_inference_steps as usize
        );

        // Diffusion process
        for (_timestep_index, timestep) in scheduler.timesteps().iter().enumerate() {
            // Prepare latent input
            let latent_input = self.prepare_latent_input(&latents, *timestep)?;

            // Run UNet inference
            let noise_pred = self.unet_inference(
                &latent_input,
                &text_embeddings,
                *timestep
            )?;

            // Scheduler step
            latents = scheduler.step(
                noise_pred,
                *timestep,
                &latents,
                config.parameters.guidance_scale
            )?;
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
        let tokens = self.tokenizer.encode(prompt, true)
            .map_err(|e| NFTError::ProcessingError(format!("Tokenization failed: {:?}", e)))?;
        let neg_tokens = self.tokenizer.encode(negative_prompt, true)
            .map_err(|e| NFTError::ProcessingError(format!("Negative tokenization failed: {:?}", e)))?;
    
        println!("Token lengths - Prompt: {}, Negative: {}", tokens.get_ids().len(), neg_tokens.get_ids().len());
    
        let token_ids = tokens.get_ids();
        let neg_token_ids = neg_tokens.get_ids();
    
        // Ensure correct shape [2, 77]
        let mut input_ids = vec![0i64; MAX_TEXT_LENGTH];
        let mut neg_input_ids = vec![0i64; MAX_TEXT_LENGTH];
    
        let token_len = token_ids.len().min(MAX_TEXT_LENGTH);
        let neg_token_len = neg_token_ids.len().min(MAX_TEXT_LENGTH);
    
        input_ids[..token_len].copy_from_slice(&token_ids[..token_len].iter().map(|&x| x as i64).collect::<Vec<_>>());
        neg_input_ids[..neg_token_len].copy_from_slice(&neg_token_ids[..neg_token_len].iter().map(|&x| x as i64).collect::<Vec<_>>());
    
        let input_tensor = ndarray::Array2::from_shape_vec(
            (2, MAX_TEXT_LENGTH),
            neg_input_ids.into_iter().chain(input_ids.into_iter()).collect()
        ).map_err(|e| NFTError::ProcessingError(format!("Failed to create input tensor: {:?}", e)))?;
    
        let input_tensor_dyn = input_tensor.into_dyn();
        let input_tensor_cow = CowArray::from(&input_tensor_dyn);
        
        let input = Value::from_array(self.text_encoder.allocator(), &input_tensor_cow)
            .map_err(|e| NFTError::ProcessingError(format!("Failed to create input value: {:?}", e)))?;
    
        println!("Running text encoder...");
        let outputs = self.text_encoder.run(vec![input])
            .map_err(|e| NFTError::ProcessingError(format!("Text encoder inference failed: {:?}", e)))?;
    
        println!("Text encoder run completed");
        let extracted_tensor: OrtOwnedTensor<f32, _> = outputs[0]
            .try_extract()
            .map_err(|e| NFTError::ProcessingError(format!("Failed to extract embeddings: {:?}", e)))?;
    
        // Ensure correct projection to match UNet expectations
        let actual_hidden_dim = 768;  // Current shape
        let expected_hidden_dim = 768;  // Target shape
    
        println!("⚠️ Projecting text embeddings from {} → {}", actual_hidden_dim, expected_hidden_dim);
        
        let mut projected_embeddings = ndarray::Array3::<f32>::zeros((2, 77, expected_hidden_dim));
        

        for i in 0..actual_hidden_dim {
            for j in 0..expected_hidden_dim {
                projected_embeddings.slice_mut(s![.., .., j]).assign(&extracted_tensor.view().slice(s![.., .., i]));
            }
        }
    
        println!("Projected text embeddings shape: {:?}", projected_embeddings.shape());
    
        Ok(projected_embeddings)
    }
    
    fn initialize_latents(&self, batch_size: usize, seed: Option<u64>) -> Result<Array4<f32>> {
        let mut rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };

        let normal = Normal::new(0.0, 1.0)
            .map_err(|_e| NFTError::ProcessingError("Failed to create normal distribution".to_string()))?;

        let shape = [
            batch_size,
            LATENT_CHANNELS,
            LATENT_HEIGHT,
            LATENT_WIDTH
        ];

        let mut latents = Array4::<f32>::zeros(shape);
        for item in latents.iter_mut() {
            *item = normal.sample(&mut rng) as f32;
        }

        Ok(latents)
    }

    fn prepare_latent_input(&self, latents: &Array4<f32>, _timestep: f32) -> Result<Array4<f32>> {
        let mut input = latents.clone();
        input.mapv_inplace(|x| x / 0.18215);
        Ok(input)
    }
    fn decode_latents(&self, latents: &Array4<f32>) -> Result<RgbImage> {
        // Scale and decode
        let mut scaled_latents = latents.clone();
        scaled_latents.mapv_inplace(|x| 1.0 / 0.18215 * x);
    
        // Convert to dynamic dimensions and create CowArray
        let scaled_latents_dyn = scaled_latents.into_dyn();
        let scaled_latents_cow = CowArray::from(&scaled_latents_dyn);
    
        // Prepare VAE input
        let vae_input = Value::from_array(self.vae_decoder.allocator(), &scaled_latents_cow)
            .map_err(|_e| NFTError::ProcessingError("Failed to create VAE input".to_string()))?;
    
        // Run VAE decoder
        let outputs = self.vae_decoder.run(vec![vae_input])
            .map_err(|_e| NFTError::ProcessingError("VAE decoding failed".to_string()))?;
    
        // Extract image
        let extracted_tensor: OrtOwnedTensor<f32, _> = outputs[0]
            .try_extract()
            .map_err(|_e| NFTError::ProcessingError("Failed to extract decoded image".to_string()))?;
    
        // Convert to Array4 with correct shape
        let shape = (1, 3, 512, 512); 
        let image_array = Array::from_iter(extracted_tensor.view().iter().copied())
            .into_shape(shape)
            .map_err(|_e| NFTError::ProcessingError("Failed to reshape decoded image".to_string()))?;
    
        // Convert to RGB image
        self.array_to_image(&image_array)
    }

    fn array_to_image(&self, array: &Array4<f32>) -> Result<RgbImage> {
        let (_, channels, height, width) = array.dim();
        assert_eq!(channels, 3, "Expected RGB image with 3 channels");

        let mut img = RgbImage::new(width as u32, height as u32);

        for y in 0..height {
            for x in 0..width {
                let r = (array[[0, 0, y, x]].clamp(0.0, 1.0) * 255.0) as u8;
                let g = (array[[0, 1, y, x]].clamp(0.0, 1.0) * 255.0) as u8;
                let b = (array[[0, 2, y, x]].clamp(0.0, 1.0) * 255.0) as u8;
                img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
            }
        }

        Ok(img)
    }

    fn save_image(&self, image: &RgbImage, format: ImageFormat) -> Result<PathBuf> {
        // Create output directory if it doesn't exist
        std::fs::create_dir_all(&self.output_path)
            .map_err(|e| NFTError::FileSystemError(e))?;

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
                image.save_with_format(&output_path, image::ImageFormat::Png)
                    .map_err(|e| NFTError::FileSystemError(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
            },
            ImageFormat::JPEG => {
                image.save_with_format(&output_path, image::ImageFormat::Jpeg)
                    .map_err(|e| NFTError::FileSystemError(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
            },
            ImageFormat::WEBP => {
                image.save_with_format(&output_path, image::ImageFormat::WebP)
                    .map_err(|e| NFTError::FileSystemError(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
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
        assert_eq!(latents.shape(), &[1, LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH]);
    }

    #[test]
    fn test_prepare_latent_input() {
        let service = create_test_service();
        let input = Array4::<f32>::zeros((1, LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH));
        let scaled = service.prepare_latent_input(&input, 0.0).unwrap();
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
        ImageService::new(
            PathBuf::from("test_models"),
            PathBuf::from("test_output")
        ).unwrap()
    }
}