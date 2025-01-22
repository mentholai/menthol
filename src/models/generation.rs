use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use super::ImageFormat;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub model_path: PathBuf,
    pub device: ComputeDevice,
    pub parameters: GenerationParameters,
    pub output_config: OutputConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComputeDevice {
    CPU,
    CUDA(usize), // GPU index
    Metal,       // For MacOS
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationParameters {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub width: u32,
    pub height: u32,
    pub num_inference_steps: u32,
    pub guidance_scale: f32,
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    pub output_dir: PathBuf,
    pub file_prefix: String,
    pub format: ImageFormat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImageType {
    PNG,
    JPEG(u8), // quality
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    pub image_path: PathBuf,
    pub generation_params: GenerationParameters,
    pub performance_metrics: PerformanceMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub generation_time_ms: u64,
    pub memory_used_mb: f64,
    pub device_utilization: f32,
}
