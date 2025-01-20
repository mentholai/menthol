use crate::{
    models::{GenerationConfig, ImageFormat, Result},
    services::ImageService,
};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;

///////////////BROKEN PLEASE FIX ME FOR IMAGE GEN TESTING/////////////
#[derive(Parser)]
pub struct GenerateArgs {
    /// The prompt to generate an image from
    #[arg(short, long)]
    prompt: String,

    /// Optional negative prompt
    #[arg(short, long)]
    negative_prompt: Option<String>,

    /// Number of inference steps (default: 50)
    #[arg(short, long, default_value = "50")]
    steps: u32,

    /// Guidance scale (default: 7.5)
    #[arg(short = 'g', long, default_value = "7.5")]
    guidance_scale: f32,

    /// Image width (default: 512)
    #[arg(short = 'W', long, default_value = "512")]
    width: u32,

    /// Image height (default: 512)
    #[arg(short = 'H', long, default_value = "512")]
    height: u32,

    /// Random seed (optional)
    #[arg(short, long)]
    seed: Option<u64>,

    /// Output directory (default: ./output)
    #[arg(short, long, default_value = "./output")]
    output_dir: PathBuf,

    /// Model directory containing ONNX models
    #[arg(short, long, default_value = "./models")]
    model_dir: PathBuf,
}

impl GenerateArgs {
    pub fn execute(&self) -> Result<()> {
        // Initialize progress bar
        let pb = ProgressBar::new(100);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
                .unwrap()
        );
        pb.set_message("Initializing...");

        // Initialize image service
        let image_service = ImageService::new(
            self.model_dir.clone(),
            self.output_dir.clone(),
        )?;

        pb.set_message("Creating generation config...");
        pb.inc(10);

        // Create generation config
        fn create_config(&self) -> GenerationConfig {
            GenerationConfig {
                model_path: self.model_dir.clone(),
                device: ComputeDevice::CUDA(0),
                parameters: self.create_generation_parameters(),
                output_config: self.create_output_config(),
            }
        }

        pb.set_message("Generating image...");
        pb.inc(10);

        // Generate the image
        let result = image_service.generate(&config)?;

        pb.set_message("Image generation complete!");
        pb.finish_with_message(format!(
            "Generated image saved to: {}",
            result.image_path.display()
        ));

        // Print performance metrics
        println!("\nPerformance Metrics:");
        println!("Generation Time: {}ms", result.performance_metrics.generation_time_ms);
        println!("Memory Used: {:.2}MB", result.performance_metrics.memory_used_mb);
        println!("Device Utilization: {:.1}%", result.performance_metrics.device_utilization * 100.0);

        Ok(())
    }

    fn create_generation_parameters(&self) -> GenerationParameters {
        GenerationParameters {
            prompt: self.prompt.clone(),
            negative_prompt: self.negative_prompt.clone(),
            width: self.width,
            height: self.height,
            num_inference_steps: self.steps,
            guidance_scale: self.guidance_scale,
            seed: self.seed,
            batch_size: 1,
        }
    }

    fn create_output_config(&self) -> OutputConfig {
        OutputConfig {
            output_dir: self.output_dir.clone(),
            file_prefix: "nft".to_string(),
            format: ImageFormat::PNG,
        }
    }
}