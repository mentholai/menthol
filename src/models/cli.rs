use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliConfig {
    pub working_dir: PathBuf,
    pub model_configs: ModelConfigs,
    pub default_output_dir: PathBuf,
    pub compute_preferences: ComputePreferences,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfigs {
    pub stable_diffusion_path: PathBuf,
    pub lora_paths: Vec<PathBuf>,
    pub textual_inversion_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputePreferences {
    pub preferred_device: PreferredDevice,
    pub max_memory_usage: Option<usize>,
    pub batch_size: usize,
    pub thread_count: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PreferredDevice {
    Auto,
    CPU,
    GPU(Vec<usize>), // List of GPU indices to use
    Metal,           // For MacOS
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationJob {
    pub prompt: String,
    pub config_overrides: Option<GenerationConfigOverrides>,
    pub output_preferences: OutputPreferences,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfigOverrides {
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub steps: Option<u32>,
    pub guidance_scale: Option<f32>,
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputPreferences {
    pub output_dir: Option<PathBuf>,
    pub filename_prefix: Option<String>,
    pub save_metadata: bool,
    pub save_generation_params: bool,
}
