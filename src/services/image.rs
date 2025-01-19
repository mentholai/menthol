use crate::models::{
    GenerationConfig, GenerationResult, NFTError, PerformanceMetrics, Result
};
use std::path::PathBuf;
use std::time::Instant;
use ort::{Environment, ExecutionProvider, GraphOptimizationLevel, Session, SessionBuilder, Value};
use tokenizers::Tokenizer;
use ndarray::{Array4, Array3, Axis};
use image::{ImageBuffer, Rgb, RgbImage};
use rand::{Rng, SeedableRng};
use crate::models::config::ImageFormat;
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
        // Split noise prediction for guidance
        let (noise_pred_uncond, noise_pred_text) = noise_pred.view().split_at(Axis(0), 1);
        
        // Perform guidance
        let noise_pred = &noise_pred_uncond.to_owned() + 
            guidance_scale * (&noise_pred_text.to_owned() - &noise_pred_uncond.to_owned());

        // Get alpha and beta for current timestep
        let timestep_index = self.timesteps.iter()
            .position(|&t| t == timestep)
            .ok_or_else(|| NFTError::ProcessingError("Invalid timestep".to_string()))?;
        
        let alpha = self.alphas[timestep_index];
        let alpha_prod = self.alphas_cumprod[timestep_index];
        let beta = 1.0 - alpha;

        // Previous alpha for variance calculation
        let prev_timestep_index = if timestep_index > 0 { timestep_index - 1 } else { 0 };
        let alpha_prod_prev = self.alphas_cumprod[prev_timestep_index];

        // Calculate variance
        let variance = beta * (1.0 - alpha_prod_prev) / (1.0 - alpha_prod);
        
        // Predict the mean
        let pred_original_sample = (latents - variance.sqrt() * &noise_pred) / alpha_prod.sqrt();
        
        // Calculate x_t-1
        let mut prev_sample = alpha_prod_prev.sqrt() * pred_original_sample +
            variance.sqrt() * &noise_pred;

        if timestep_index > 0 {
            let mut rng = rand::thread_rng();
            let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
            let noise = Array4::from_shape_fn(
                prev_sample.raw_dim(),
                |_| normal.sample(&mut rng) as f32
            );
            prev_sample = prev_sample + variance.sqrt() * noise;
        }

        Ok(prev_sample)
    }
}

pub struct ImageService {
    env: Environment,
    text_encoder: Session,
    unet: Session,
    vae_decoder: Session,
    tokenizer: Tokenizer,
    output_path: PathBuf,
}

impl ImageService {
    pub fn new(model_path: PathBuf, output_path: PathBuf) -> Result<Self> {
        // Initialize ONNX Runtime with CUDA support
      // Initialize ONNX Runtime with CUDA support
// Initialize ONNX Runtime with CUDA support
    let env = Environment::builder()
    .with_name("stable_diffusion")
    .with_execution_providers([
        ExecutionProvider::CUDA(Default::default()),
        ExecutionProvider::CPU(Default::default()),
    ])
    .build()
    .map_err(|e| NFTError::ModelLoadError(e.to_string().into()))?;

        // Setup paths for each model component
        let text_encoder_path = model_path.join("text_encoder").join("model.onnx");
        let unet_path = model_path.join("unet").join("model.onnx");
        let vae_decoder_path = model_path.join("vae_decoder").join("model.onnx");
        let tokenizer_path = model_path.join("tokenizer").join("tokenizer.json");

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
        let text_encoder = SessionBuilder::new(&env)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .with_model_from_file(&text_encoder_path)
        .map_err(|e| NFTError::ModelLoadError(e.to_string().into()))?;
    
    let unet = SessionBuilder::new(&env)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .with_model_from_file(&unet_path)
        .map_err(|e| NFTError::ModelLoadError(e.to_string().into()))?;
    
    let vae_decoder = SessionBuilder::new(&env)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .with_model_from_file(&vae_decoder_path)
        .map_err(|e| NFTError::ModelLoadError(e.to_string().into()))?;

        Ok(Self {
            env,
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
            config.parameters.negative_prompt.as_deref().unwrap_or("")
        )?;

        // Initialize latents
        let mut latents = self.initialize_latents(
            config.parameters.batch_size,
            config.parameters.seed
        )?;

        // Setup scheduler
        let mut scheduler = DiffusionScheduler::new(
            config.parameters.num_inference_steps as usize
        );

        // Diffusion process
        for (timestep_index, timestep) in scheduler.timesteps().iter().enumerate() {
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
        let output_path = self.save_image(&image, config.output_config.format)?;

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

    fn encode_prompt(&self, prompt: &str, negative_prompt: &str) -> Result<Array4<f32>> {
        // Tokenize prompts
        let tokens = self.tokenizer.encode(prompt, true)
            .map_err(|e| NFTError::ProcessingError("Tokenization failed".to_string()))?;
        let neg_tokens = self.tokenizer.encode(negative_prompt, true)
            .map_err(|e| NFTError::ProcessingError("Negative tokenization failed".to_string()))?;

        // Convert to tensors
        let mut input_ids = vec![0; MAX_TEXT_LENGTH];
        let mut neg_input_ids = vec![0; MAX_TEXT_LENGTH];

        // Copy and pad tokens
        input_ids[..tokens.get_ids().len()].copy_from_slice(tokens.get_ids());
        neg_input_ids[..neg_tokens.get_ids().len()].copy_from_slice(neg_tokens.get_ids());

        // Create input tensor
        let input_tensor = ndarray::Array2::from_shape_vec(
            (2, MAX_TEXT_LENGTH),
            [&neg_input_ids[..], &input_ids[..]].concat()
        ).map_err(|e| NFTError::ProcessingError("Failed to create input tensor".to_string()))?;

        // Run text encoder
        let input = Value::from_array(self.text_encoder.allocator(), &input_tensor)
            .map_err(|e| NFTError::ProcessingError("Failed to create input value".to_string()))?;

        let outputs = self.text_encoder.run(vec![input])
            .map_err(|e| NFTError::ProcessingError("Text encoder inference failed".to_string()))?;

        // Convert output to ndarray
        let embeddings = outputs[0].try_extract::<f32>()
            .map_err(|e| NFTError::ProcessingError("Failed to extract embeddings".to_string()))?;

        Ok(embeddings.into_dimensionality()
            .map_err(|e| NFTError::ProcessingError("Failed to reshape embeddings".to_string()))?)
    }

    fn initialize_latents(&self, batch_size: usize, seed: Option<u64>) -> Result<Array4<f32>> {
        let mut rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };

        let normal = Normal::new(0.0, 1.0)
            .map_err(|e| NFTError::ProcessingError("Failed to create normal distribution".to_string()))?;

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

    fn prepare_latent_input(&self, latents: &Array4<f32>, timestep: f32) -> Result<Array4<f32>> {
        let mut input = latents.clone();
        input.mapv_inplace(|x| x / 0.18215);
        Ok(input)
    }
    fn decode_latents(&self, latents: &Array4<f32>) -> Result<RgbImage> {
        // Scale and decode
        let mut scaled_latents = latents.clone();
        scaled_latents.mapv_inplace(|x| 1.0 / 0.18215 * x);

        // Prepare VAE input
        let vae_input = Value::from_array(self.vae_decoder.allocator(), &scaled_latents)
            .map_err(|e| NFTError::ProcessingError("Failed to create VAE input".to_string()))?;

        // Run VAE decoder
        let outputs = self.vae_decoder.run(vec![vae_input])
            .map_err(|e| NFTError::ProcessingError("VAE decoding failed".to_string()))?;

        // Extract image
        let image_array = outputs[0].try_extract::<f32>()
            .map_err(|e| NFTError::ProcessingError("Failed to extract decoded image".to_string()))?;

        // Convert to RGB image
        self.array_to_image(&image_array.into_dimensionality()
            .map_err(|e| NFTError::ProcessingError("Failed to reshape decoded image".to_string()))?)
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
                    .map_err(|e| NFTError::FileSystemError(e))?;
            },
            ImageFormat::JPEG => {
                image.save_with_format(&output_path, image::ImageFormat::Jpeg)
                    .map_err(|e| NFTError::FileSystemError(e))?;
            },
            ImageFormat::WEBP => {
                image.save_with_format(&output_path, image::ImageFormat::WebP)
                    .map_err(|e| NFTError::FileSystemError(e))?;
            }
        }

        Ok(output_path)
    }
}