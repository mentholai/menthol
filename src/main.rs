#![recursion_limit = "16363"]
mod brain;
mod cli;
mod models;
mod services;

use brain::MasterBrain;
use clap::Parser;
use cli::{Cli, Commands};
use models::SystemConfig;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let config_path = "./config.json";
    let system_config = SystemConfig::load(config_path)
        .map_err(|e| format!("Failed to load config: {:?}", e))?;
    let mut master_brain = MasterBrain::new(system_config)?;

    match cli.command {
        Commands::Generate(args) => {
            args.execute(&mut master_brain)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
        }
    }
    Ok(())
}