use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    pub resources: ResourceConfig,
    pub models: ModelPaths,
    pub storage: StorageConfig,
    pub generation: GenerationDefaults,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    pub compute: ComputeConfig,
    pub memory: MemoryConfig,
    pub gpu_settings: Option<GPUSettings>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeConfig {
    pub preferred_device: String,
    pub cpu_threads: Option<usize>,
    pub gpu_indices: Option<Vec<usize>>,
    pub metal_enabled: bool,
    pub cuda_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub max_memory_gb: f32,
    pub cache_size_mb: usize,
    pub clear_cache_after_generation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUSettings {
    pub batch_size: usize,
    pub half_precision: bool,
    pub memory_split: Vec<f32>,
    pub attention_slicing: bool,
    pub vae_slicing: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPaths {
    pub stable_diffusion: StableDiffusionConfig,
    pub custom_models: Vec<CustomModelConfig>,
    pub embeddings: Vec<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StableDiffusionConfig {
    pub base_model_path: PathBuf,
    pub vae_path: Option<PathBuf>,
    pub lora_paths: Vec<PathBuf>,
    pub textual_inversions: Vec<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomModelConfig {
    pub name: String,
    pub path: PathBuf,
    pub model_type: ModelType,
    pub config_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    StableDiffusion,
    LoRA,
    TextualInversion,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub output_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub temp_dir: PathBuf,
    pub metadata_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationDefaults {
    pub image: ImageDefaults,
    pub prompt: PromptDefaults,
    pub nft: NFTDefaults,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageDefaults {
    pub width: u32,
    pub height: u32,
    pub format: ImageFormat,
    pub quality: u8,
    pub upscale: bool,
    pub upscale_factor: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptDefaults {
    pub negative_prompt: String,
    pub steps: u32,
    pub cfg_scale: f32,
    pub seed: Option<u64>,
    pub style_preset: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NFTDefaults {
    pub collection_name: String,
    pub symbol: String,
    pub seller_fee_basis_points: u16,
    pub creator_address: String,
    pub creator_share: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImageFormat {
    PNG,
    JPEG,
    WEBP,
}

impl SystemConfig {
    pub fn load(path: &str) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let config = serde_json::from_str(&content)?;
        Ok(config)
    }

    pub fn save(&self, path: &str) -> Result<(), ConfigError> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        // Check if paths exist
        if !self.models.stable_diffusion.base_model_path.exists() {
            return Err(ConfigError::InvalidPath(
                "Base model path does not exist".to_string(),
            ));
        }

        // Validate GPU settings if enabled
        if let Some(gpu_settings) = &self.resources.gpu_settings {
            if gpu_settings.memory_split.iter().sum::<f32>() != 1.0 {
                return Err(ConfigError::InvalidValue(
                    "GPU memory split must sum to 1.0".to_string(),
                ));
            }
        }

        // Validate memory settings
        if self.resources.memory.max_memory_gb <= 0.0 {
            return Err(ConfigError::InvalidValue(
                "Maximum memory must be positive".to_string(),
            ));
        }

        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Invalid path: {0}")]
    InvalidPath(String),

    #[error("Invalid value: {0}")]
    InvalidValue(String),
}
