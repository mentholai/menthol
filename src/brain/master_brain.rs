use super::{
    neural::AdvancedNeuralProcessor, processing::ImageProcessor, quantum::QuantumProcessor,
    ConsciousnessProcessor, NeuralProcessor, TextProcessor,
};
use crate::models::{
    ComputeConfig, ComputeDevice, GenerationParameters, GenerationResult, ImageAnalyzer,
    LoreGenerator, NFTAttribute, NFTCollection, NFTCreator, NFTDefaults, NFTError, NFTFile,
    NFTMetadata, NFTProperties, PerformanceMetrics, Result, SystemConfig, NFT,
};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;

pub struct MasterBrain {
    config: SystemConfig,
    neural_processor: AdvancedNeuralProcessor,
    quantum_processor: QuantumProcessor,
    image_processor: ImageProcessor,
    text_processor: TextProcessor,
    consciousness_processor: ConsciousnessProcessor,
    lore_generator: LoreGenerator,
    image_analyzer: ImageAnalyzer,
}

impl MasterBrain {
    pub fn new(config: SystemConfig) -> Result<Self> {
        // Initialize with specific device
        let device = Self::determine_compute_device(&config.resources.compute);
        let lore_model_path = PathBuf::from("models/mistral.gguf");
        let image_model_path = PathBuf::from("models/clip.onnx");
        Ok(Self {
            neural_processor: AdvancedNeuralProcessor::new_with_device(device.clone()),
            quantum_processor: QuantumProcessor::new_with_device(device.clone()),
            image_processor: ImageProcessor::new_with_device(device.clone())?,
            text_processor: TextProcessor::new(512),
            consciousness_processor: ConsciousnessProcessor::new(0.7),
            lore_generator: LoreGenerator::new(lore_model_path),
            image_analyzer: ImageAnalyzer::new(image_model_path),
            config,
        })
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

    pub async fn generate_nft(
        &mut self,
        prompt: String,
        progress: Option<ProgressBar>,
    ) -> Result<GenerationResult> {
        let progress = progress.unwrap_or_else(|| {
            let pb = ProgressBar::new(100);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
                    .unwrap(),
            );
            pb
        });

        // Process text input
        progress.set_message("Processing prompt...");
        let thought_vector = self.text_processor.process_input(&prompt);
        progress.inc(10);

        // Apply quantum transformations
        progress.set_message("Applying quantum transformations...");
        self.quantum_processor
            .transform_vector(&mut thought_vector.clone());
        progress.inc(20);

        // Generate image
        progress.set_message("Generating image...");
        let image_result = self.image_processor.generate(&thought_vector)?;
        progress.inc(40);

        // 🔥 Analyze the image and generate lore
        progress.set_message("Analyzing image and generating lore...");
        let metadata = self.generate_metadata(&image_result).await?;
        progress.inc(20);

        // Save results
        progress.set_message("Saving results...");
        let result = self.save_generation(image_result, metadata)?;
        progress.inc(10);

        progress.finish_with_message("NFT generation complete!");

        Ok(result)
    }

    async fn generate_metadata(&mut self, result: &GenerationResult) -> Result<NFT> {
        // Analyze image to get a description
        let image_description = self.image_analyzer.analyze(&result.image_path);

        let name = self.generate_name(&image_description).await?;
        let lore = self.generate_description(&image_description).await?;

        let defaults = &self.config.generation.nft;

        Ok(NFT {
            name,
            symbol: defaults.symbol.clone(),
            description: lore, // Use generated lore as description
            image_uri: result.image_path.to_string_lossy().to_string(),
            attributes: [].into(),
            collection: NFTCollection {
                name: defaults.collection_name.clone(),
                family: defaults.collection_name.clone(),
                collection_id: uuid::Uuid::new_v4().to_string(),
            },
            metadata: self.generate_technical_metadata(result, defaults)?,
        })
    }

    async fn generate_name(&mut self, prompt: &str) -> Result<String> {
        let name = self.lore_generator.generate_name(prompt);

        let mut name_vector = self.neural_processor.process_thought(&name).await;

        self.quantum_processor
            .transform_vector(&mut name_vector)
            .await?;

        let name = name_vector
            .iter()
            .map(|&x| (x * 255.0).clamp(0.0, 255.0) as u8 as char)
            .collect::<String>();

        Ok(name)
    }

    async fn generate_description(&mut self, prompt: &str) -> Result<String> {
        let description = self.lore_generator.generate_lore(prompt);

        let mut thought_vector = self.neural_processor.process_thought(&description).await;
        self.quantum_processor
            .transform_vector(&mut thought_vector)
            .await?;

        let description = thought_vector
            .iter()
            .map(|&x| (x * 255.0).clamp(0.0, 255.0) as u8 as char)
            .collect::<String>();

        Ok(description)
    }
    fn generate_technical_metadata(
        &self,
        result: &GenerationResult,
        defaults: &NFTDefaults,
    ) -> Result<NFTMetadata> {
        Ok(NFTMetadata {
            seller_fee_basis_points: defaults.seller_fee_basis_points,
            creation_date: chrono::Utc::now().to_rfc3339(),
            files: vec![NFTFile {
                uri: result.image_path.to_string_lossy().to_string(),
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

    fn save_generation(&self, result: GenerationResult, metadata: NFT) -> Result<GenerationResult> {
        // Save image and metadata to configured paths
        let output_dir = &self.config.storage.output_dir;
        std::fs::create_dir_all(output_dir).map_err(|e| NFTError::FileSystemError(e))?;

        // Save metadata
        let metadata_path = output_dir.join("metadata.json");
        std::fs::write(
            &metadata_path,
            serde_json::to_string_pretty(&metadata).map_err(|e| NFTError::SerializationError(e))?,
        )
        .map_err(|e| NFTError::FileSystemError(e))?;

        Ok(result)
    }

    fn collect_performance_metrics(&self) -> PerformanceMetrics {
        PerformanceMetrics {
            generation_time_ms: 0,
            memory_used_mb: 0.0,
            device_utilization: 0.0,
        }
    }
}

// Helper struct for internal image generation results
#[derive(Clone)]
struct ImageResult {
    path: PathBuf,
    params: GenerationParameters,
}
