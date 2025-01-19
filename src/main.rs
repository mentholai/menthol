mod brain;
mod models;
mod services;
mod cli;

use clap::Parser;
use cli::{Cli, Commands};

fn main() -> Result<(), Box<dyn std::error::Error>> {

    let cli = Cli::parse();

    // Execute command
    match cli.command {
        Commands::Generate(args) => args.execute()?,
    }

    Ok(())
}