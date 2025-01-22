use super::types::*;
use crate::models::{
    ComputeDevice, GenerationConfig, GenerationParameters, GenerationResult, ImageFormat,
    OutputConfig, Result,
};
use crate::services::ImageService;
use rand::Rng;
use std::path::PathBuf;

struct NoiseGenerator {
    seed: u64,
}

pub struct ImageProcessor {
    style_params: ImageGenerationParams,
    noise_generator: NoiseGenerator,
    image_service: ImageService,
    output_path: PathBuf,
    device: ComputeDevice,
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

impl ImageProcessor {
    pub fn new_with_device(device: ComputeDevice) -> Result<Self> {
        let output_path = PathBuf::from("output"); // This could be configurable
        Ok(Self {
            style_params: Self::initialize_style_params(),
            noise_generator: NoiseGenerator::new(),
            image_service: ImageService::new(output_path.clone())?,
            output_path,
            device,
        })
    }

    pub fn generate(&mut self, thought_vector: &ThoughtVector) -> Result<GenerationResult> {
        let params = self.process_thought_vector(thought_vector);

        // Create generation config
        let config = GenerationConfig {
            model_path: PathBuf::from("models"),
            device: self.device.clone(), // Use stored device
            parameters: params.into_generation_parameters(),
            output_config: OutputConfig {
                output_dir: self.output_path.clone(),
                file_prefix: "nft".to_string(),
                format: ImageFormat::PNG,
            },
        };

        self.image_service.generate(&config)
    }

    fn initialize_style_params() -> ImageGenerationParams {
        let _rng = rand::thread_rng();

        ImageGenerationParams {
            style_vector: vec![
                0.8, // Cyberpunk intensity
                0.7, // Neon glow
                0.9, // Urban decay
                0.6, // Tech complexity
            ],
            composition_weights: vec![
                0.75, // Foreground emphasis
                0.60, // Background complexity
                0.85, // Contrast ratio
                0.70, // Detail density
            ],
            color_palette: ColorPalette {
                primary_colors: vec![
                    RGB {
                        r: 0,
                        g: 255,
                        b: 196,
                        weight: 0.8,
                    }, // Cyber green
                    RGB {
                        r: 255,
                        g: 0,
                        b: 128,
                        weight: 0.7,
                    }, // Neon pink
                    RGB {
                        r: 0,
                        g: 196,
                        b: 255,
                        weight: 0.6,
                    }, // Electric blue
                ],
                accent_colors: vec![
                    RGB {
                        r: 255,
                        g: 128,
                        b: 0,
                        weight: 0.5,
                    }, // Neon orange
                    RGB {
                        r: 128,
                        g: 0,
                        b: 255,
                        weight: 0.4,
                    }, // Deep purple
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
        self.noise_generator
            .apply_noise(&mut params.noise_parameters);

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

impl ImageGenerationParams {
    fn into_generation_parameters(self) -> GenerationParameters {
        GenerationParameters {
            prompt: format!(
                "A cyberpunk scene with intensity {}, neon glow {}, urban decay {}, tech complexity {}",
                self.style_vector[0],
                self.style_vector[1],
                self.style_vector[2],
                self.style_vector[3]
            ),
            negative_prompt: Some("blurry, low quality, distorted".to_string()),
            width: 512,
            height: 512,
            num_inference_steps: 25,
            guidance_scale: 7.5,
            seed: Some(rand::random())
        }
    }
}
