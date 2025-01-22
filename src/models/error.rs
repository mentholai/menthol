use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum NFTError {
    #[error("Failed to generate image: {0}")]
    ImageGenerationError(String),

    #[error("Failed to load model at path: {0}")]
    ModelLoadError(PathBuf),

    #[error("Device not available: {0}")]
    DeviceError(String),

    #[error("Invalid configuration: {0}")]
    ConfigurationError(String),

    #[error("File system error: {0}")]
    FileSystemError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("GPU error: {0}")]
    GPUError(String),

    #[error("Memory error: {0}")]
    MemoryError(String),

    #[error("Invalid prompt: {0}")]
    PromptError(String),

    #[error("Processing error: {0}")]
    ProcessingError(String),
}

pub type Result<T> = std::result::Result<T, NFTError>;
