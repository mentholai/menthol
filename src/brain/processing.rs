use crate::models::ComputeDevice;

use super::types::*;
use rand::Rng;
use std::sync::Arc;

pub struct ImageProcessor {
    style_params: ImageGenerationParams,
    noise_generator: NoiseGenerator,
}

impl ImageProcessor {
    pub fn new_with_device(device: ComputeDevice) -> Self {
        Self {
            style_params: Self::initialize_style_params(),
            noise_generator: NoiseGenerator::new(),
        }
    }

    pub fn new() -> Self {
        Self {
            style_params: Self::initialize_style_params(),
            noise_generator: NoiseGenerator::new(),
        }
    }

    fn initialize_style_params() -> ImageGenerationParams {
        let mut rng = rand::thread_rng();
        
        ImageGenerationParams {
            style_vector: vec![
                0.8,  // Cyberpunk intensity
                0.7,  // Neon glow
                0.9,  // Urban decay
                0.6   // Tech complexity
            ],
            composition_weights: vec![
                0.75, // Foreground emphasis
                0.60, // Background complexity
                0.85, // Contrast ratio
                0.70  // Detail density
            ],
            color_palette: ColorPalette {
                primary_colors: vec![
                    RGB { r: 0, g: 255, b: 196, weight: 0.8 },   // Cyber green
                    RGB { r: 255, g: 0, b: 128, weight: 0.7 },   // Neon pink
                    RGB { r: 0, g: 196, b: 255, weight: 0.6 },   // Electric blue
                ],
                accent_colors: vec![
                    RGB { r: 255, g: 128, b: 0, weight: 0.5 },   // Neon orange
                    RGB { r: 128, g: 0, b: 255, weight: 0.4 },   // Deep purple
                ],
                harmony_matrix: vec![
                    vec![1.0, 0.8, 0.6],
                    vec![0.8, 1.0, 0.7],
                    vec![0.6, 0.7, 1.0],
                ],
            },
            noise_parameters: NoiseParameters {
                frequency: 0.01,
                amplitude: 0.5,
                octaves: 4,
                persistence: 0.5,
            },
        }
    }

    pub fn process_thought_vector(&mut self, thought: &ThoughtVector) -> ImageGenerationParams {
        let mut params = self.style_params.clone();
        
        // Modify style based on thought vector
        for (i, &value) in thought.iter().take(4).enumerate() {
            params.style_vector[i] = (params.style_vector[i] + value) / 2.0;
        }
        
        // Generate noise variations
        self.noise_generator.apply_noise(&mut params.noise_parameters);
        
        // Adjust color weights based on thought intensity
        let intensity = thought.iter().sum::<f64>() / thought.len() as f64;
        self.adjust_colors(&mut params.color_palette, intensity);
        
        params
    }

    fn adjust_colors(&self, palette: &mut ColorPalette, intensity: f64) {
        for color in palette.primary_colors.iter_mut() {
            color.weight *= intensity.max(0.2);
        }
        for color in palette.accent_colors.iter_mut() {
            color.weight *= (1.0 - intensity).max(0.2);
        }
    }
}

struct NoiseGenerator {
    seed: u64,
}

impl NoiseGenerator {
    fn new() -> Self {
        Self {
            seed: rand::random(),
        }
    }

    fn apply_noise(&self, params: &mut NoiseParameters) {
        let mut rng = rand::thread_rng();
        
        params.frequency *= 1.0 + (rng.gen::<f64>() - 0.5) * 0.1;
        params.amplitude *= 1.0 + (rng.gen::<f64>() - 0.5) * 0.1;
        params.persistence = (params.persistence + rng.gen::<f64>()) / 2.0;
    }
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
        // Update attention weights based on thought vector
        for (weight, &thought_val) in self.attention_weights.iter_mut()
            .zip(thought.iter()) {
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
        let complexity: f64 = thought.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .sum::<f64>() / thought.len() as f64;
        
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