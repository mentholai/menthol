use serde::{Deserialize, Serialize};

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

pub struct TextProcessor {
    embedding_dimension: usize,
    attention_weights: Vec<f64>,
}

impl TextProcessor {
    pub fn new(embedding_dimension: usize) -> Self {
        Self {
            embedding_dimension,
            attention_weights: vec![1.0; embedding_dimension],
        }
    }

    pub fn process_input(&mut self, input: &str) -> ThoughtVector {
        let mut thought = vec![0.0; self.embedding_dimension];
        let chars: Vec<char> = input.chars().collect();

        // Create pseudo-embeddings from character codes
        for (i, &c) in chars.iter().enumerate() {
            let pos = i % self.embedding_dimension;
            thought[pos] += (c as u32 as f64) / 1000.0;
        }

        // Apply attention weights
        for i in 0..self.embedding_dimension {
            thought[i] *= self.attention_weights[i];
        }

        // Normalize
        let sum: f64 = thought.iter().sum();
        if sum != 0.0 {
            for value in thought.iter_mut() {
                *value /= sum;
            }
        }

        thought
    }

    pub fn update_attention(&mut self, thought: &ThoughtVector) {
        for (weight, &thought_val) in self.attention_weights.iter_mut().zip(thought.iter()) {
            *weight = (*weight + thought_val.abs()) / 2.0;
        }
    }
}

pub struct ConsciousnessProcessor {
    threshold: f64,
    stability_factor: f64,
    history: Vec<ConsciousnessLevel>,
}

impl ConsciousnessProcessor {
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            stability_factor: 0.95,
            history: Vec::with_capacity(100),
        }
    }

    pub fn evaluate_consciousness(&mut self, thought: &ThoughtVector) -> ConsciousnessLevel {
        let raw_consciousness = self.calculate_raw_consciousness(thought);
        let stable_consciousness = self.stabilize_consciousness(raw_consciousness);

        self.history.push(stable_consciousness);
        if self.history.len() > 100 {
            self.history.remove(0);
        }

        stable_consciousness
    }

    fn calculate_raw_consciousness(&self, thought: &ThoughtVector) -> f64 {
        let magnitude: f64 = thought.iter().map(|x| x * x).sum::<f64>().sqrt();
        let complexity: f64 =
            thought.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f64>() / thought.len() as f64;

        (magnitude * complexity).min(1.0)
    }

    fn stabilize_consciousness(&self, raw: f64) -> f64 {
        if self.history.is_empty() {
            return raw;
        }

        let previous = self.history[self.history.len() - 1];
        previous * self.stability_factor + raw * (1.0 - self.stability_factor)
    }
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
    CUDA(usize), // GPU index
    Metal,
}

impl ComputeDevice {
    pub fn from_string(device: &str) -> Self {
        match device.to_lowercase().as_str() {
            "cpu" => ComputeDevice::CPU,
            "metal" => ComputeDevice::Metal,
            s if s.starts_with("cuda") => {
                // Parse CUDA device index if provided (e.g., "cuda:0")
                let index = s
                    .split(':')
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                ComputeDevice::CUDA(index)
            }
            _ => ComputeDevice::CPU, // Default to CPU
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
