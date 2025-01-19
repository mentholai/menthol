use super::{
    neural::AdvancedNeuralProcessor, processing::{ConsciousnessProcessor, ImageProcessor, TextProcessor}, quantum::QuantumProcessor
};
use crate::models::{
    ComputeConfig, ComputeDevice, GenerationParameters, GenerationResult, NFTAttribute, NFTCollection, NFTCreator, NFTDefaults, NFTError, NFTFile, NFTMetadata, NFTProperties, PerformanceMetrics, PreferredDevice, Result, SystemConfig, NFT
};
use std::sync::Arc;
use std::path::PathBuf;
use indicatif::{ProgressBar, ProgressStyle};

pub struct MasterBrain {
    config: SystemConfig,
    neural_processor: AdvancedNeuralProcessor,
    quantum_processor: QuantumProcessor,
    image_processor: ImageProcessor,
    text_processor: TextProcessor,
    consciousness_processor: ConsciousnessProcessor,
}

impl MasterBrain {
    pub fn new(config: SystemConfig) -> Self {
        // Initialize with specific device
        let device = Self::determine_compute_device(&config.resources.compute);

        Self {
            neural_processor: AdvancedNeuralProcessor::new_with_device(device.clone()),
            quantum_processor: QuantumProcessor::new_with_device(device.clone()),
            image_processor: ImageProcessor::new_with_device(device.clone()),
            text_processor: TextProcessor::new(512),
            consciousness_processor: ConsciousnessProcessor::new(0.7),
            config,
        }
    }
    

    fn determine_compute_device(compute_config: &ComputeConfig) -> ComputeDevice {
        if compute_config.cuda_enabled && compute_config.gpu_indices.is_some() {
            let index = compute_config.gpu_indices.as_ref().unwrap()[0];
            ComputeDevice::CUDA(index)
        } else if compute_config.metal_enabled {
            ComputeDevice::Metal
        } else {
            ComputeDevice::CPU
        }
    }

    pub fn generate_nft(
        &mut self,
        prompt: String,
        progress: Option<ProgressBar>
    ) -> Result<GenerationResult> {
        // Set up progress reporting
        let progress = progress.unwrap_or_else(|| {
            let pb = ProgressBar::new(100);
            pb.set_style(ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
                .unwrap());
            pb
        });

        // Process text input
        progress.set_message("Processing prompt...");
        let thought_vector = self.text_processor.process_input(&prompt);
        progress.inc(10);

        // Apply quantum transformations
        progress.set_message("Applying quantum transformations...");
        self.quantum_processor.transform_vector(&mut thought_vector.clone());
        progress.inc(20);

        // Generate image
        progress.set_message("Generating image...");
        let image_result = self.image_processor.generate(&thought_vector)?;
        progress.inc(40);

        // Generate metadata
        progress.set_message("Creating NFT metadata...");
        let metadata = self.generate_metadata(&prompt, &image_result)?;
        progress.inc(20);

        // Save results
        progress.set_message("Saving results...");
        let result = self.save_generation(image_result, metadata)?;
        progress.inc(10);

        progress.finish_with_message("NFT generation complete!");

        Ok(result)
    }

    fn generate_metadata(&self, prompt: &str, image_result: &ImageResult) -> Result<NFT> {
        // Create NFT metadata using config defaults
        let defaults = &self.config.generation.nft;
        
        Ok(NFT {
            name: self.generate_name(prompt)?,
            symbol: defaults.symbol.clone(),
            description: self.generate_description(prompt)?,
            image_uri: image_result.path.to_string_lossy().to_string(),
            attributes: self.generate_attributes(prompt)?,
            collection: NFTCollection {
                name: defaults.collection_name.clone(),
                family: defaults.collection_name.clone(),
                collection_id: uuid::Uuid::new_v4().to_string(),
            },
            metadata: self.generate_technical_metadata(image_result, defaults)?,
        })
    }

    fn generate_name(&self, prompt: &str) -> Result<String> {
        // TODO: Implement name generation
        Ok(format!("NFT #{}", uuid::Uuid::new_v4().to_string().split('-').next().unwrap()))
    }

    fn generate_description(&self, prompt: &str) -> Result<String> {
        // TODO: Implement description generation
        Ok(prompt.to_string())
    }

    fn generate_attributes(&self, prompt: &str) -> Result<Vec<NFTAttribute>> {
        // TODO: Implement attribute generation
        Ok(vec![])
    }

    fn generate_technical_metadata(&self, image_result: &ImageResult, defaults: &NFTDefaults) -> Result<NFTMetadata> {
        Ok(NFTMetadata {
            seller_fee_basis_points: defaults.seller_fee_basis_points,
            creation_date: chrono::Utc::now().to_rfc3339(),
            files: vec![NFTFile {
                uri: image_result.path.to_string_lossy().to_string(),
                file_type: "image/png".to_string(),
            }],
            properties: NFTProperties {
                category: "image".to_string(),
                creators: vec![NFTCreator {
                    address: defaults.creator_address.clone(),
                    share: defaults.creator_share,
                }],
            },
        })
    }

    fn save_generation(&self, image_result: ImageResult, metadata: NFT) -> Result<GenerationResult> {
        // Save image and metadata to configured paths
        let output_dir = &self.config.storage.output_dir;
        std::fs::create_dir_all(output_dir)
            .map_err(|e| NFTError::FileSystemError(e))?;

        let result = GenerationResult {
            image_path: image_result.path,
            generation_params: image_result.params,
            performance_metrics: self.collect_performance_metrics(),
        };

        // Save metadata
        let metadata_path = output_dir.join("metadata.json");
        std::fs::write(
            &metadata_path,
            serde_json::to_string_pretty(&metadata)
                .map_err(|e| NFTError::SerializationError(e))?
        ).map_err(|e| NFTError::FileSystemError(e))?;

        Ok(result)
    }

    fn collect_performance_metrics(&self) -> PerformanceMetrics {
        PerformanceMetrics {
            generation_time_ms: 0,
            memory_used_mb: 0.0,
            device_utilization: 0.0
        }
    }
}

// Helper struct for internal image generation results
#[derive(Clone)]
struct ImageResult {
    path: PathBuf,
    params: GenerationParameters,
}