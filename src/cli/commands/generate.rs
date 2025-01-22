use crate::{
    brain::master_brain::MasterBrain,
    models::{NFTError, Result, SystemConfig},
};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs;
use std::path::PathBuf;

/// Command-line arguments for generating an image
#[derive(Parser)]
pub struct GenerateArgs {
    /// The prompt to generate an image from
    #[arg(short, long)]
    prompt: String,

    /// Output directory (default: ./output)
    #[arg(short, long, default_value = "./output")]
    output_dir: PathBuf,

    /// Config file path (default: ./config.json)
    #[arg(short, long, default_value = "./config.json")]
    config_path: PathBuf,
}

impl GenerateArgs {
    pub async fn execute(&self, master_brain: &mut MasterBrain) -> Result<()> {
        // Initialize progress bar
        let pb = ProgressBar::new(100);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
                .unwrap(),
        );
        pb.set_message("Loading system configuration...");

        // Load system configuration
        let config = SystemConfig::load(self.config_path.to_str().unwrap())
            .map_err(|e| NFTError::ConfigurationError(format!("Failed to load config: {}", e)))?;

        pb.inc(10);
        pb.set_message("Initializing MasterBrain...");

        // Initialize MasterBrain
        let mut master_brain = MasterBrain::new(config)?;

        pb.inc(10);
        pb.set_message("Starting image generation...");

        // Generate NFT (image + metadata)
        let result = master_brain
            .generate_nft(self.prompt.clone(), Some(pb.clone()))
            .await?;

        pb.set_message("Saving generated image...");
        fs::create_dir_all(&self.output_dir).map_err(|e| NFTError::FileSystemError(e))?;

        let output_path = self.output_dir.join("generated_nft.png");
        fs::copy(&result.image_path, &output_path).map_err(|e| NFTError::FileSystemError(e))?;

        pb.finish_with_message(format!(
            "✅ Image successfully generated: {}",
            output_path.display()
        ));

        Ok(())
    }
}
