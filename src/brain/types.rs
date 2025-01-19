use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub type NeuralWeight = f64;
pub type SynapticResponse = Vec<f64>;
pub type ThoughtVector = Vec<f64>;
pub type ConsciousnessLevel = f64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralArchitecture {
    pub layers: Vec<SynapticLayer>,
    pub consciousness_matrix: ConsciousnessMatrix,
    pub attention_weights: AttentionWeights,
}



#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapticLayer {
    pub weights: Vec<NeuralWeight>,
    pub bias: Vec<NeuralWeight>,
    pub activation_function: ActivationFunction,
    pub dropout_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessMatrix {
    pub values: Vec<Vec<f64>>,
    pub eigenvalues: Vec<f64>,
    pub stability_index: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionWeights {
    pub temporal: Vec<f64>,
    pub spatial: Vec<f64>,
    pub semantic: Vec<f64>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    QuantumActivation,
}

// Image generation related types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationParams {
    pub style_vector: Vec<f64>,
    pub composition_weights: Vec<f64>,
    pub color_palette: ColorPalette,
    pub noise_parameters: NoiseParameters,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorPalette {
    pub primary_colors: Vec<RGB>,
    pub accent_colors: Vec<RGB>,
    pub harmony_matrix: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RGB {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseParameters {
    pub frequency: f64,
    pub amplitude: f64,
    pub octaves: u32,
    pub persistence: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct BrainResponse {
    pub thought_vector: ThoughtVector,
    pub consciousness_level: ConsciousnessLevel,
    pub image_params: ImageGenerationParams,
    pub quantum_state: QuantumStateMetrics,
}

#[derive(Debug, Clone, Serialize)]
pub struct QuantumStateMetrics {
    pub entanglement_degree: f64,
    pub coherence_level: f64,
    pub collapse_probability: f64,
}

#[derive(Debug, Clone)]
pub enum ComputeDevice {
    CPU,
    CUDA(usize),  // GPU index
    Metal,
}

impl ComputeDevice {
    pub fn from_string(device: &str) -> Self {
        match device.to_lowercase().as_str() {
            "cpu" => ComputeDevice::CPU,
            "metal" => ComputeDevice::Metal,
            s if s.starts_with("cuda") => {
                // Parse CUDA device index if provided (e.g., "cuda:0")
                let index = s.split(':')
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                ComputeDevice::CUDA(index)
            }
            _ => ComputeDevice::CPU,  // Default to CPU
        }
    }
}

// If we need string conversion
impl ToString for ComputeDevice {
    fn to_string(&self) -> String {
        match self {
            ComputeDevice::CPU => "cpu".to_string(),
            ComputeDevice::CUDA(index) => format!("cuda:{}", index),
            ComputeDevice::Metal => "metal".to_string(),
        }
    }
}