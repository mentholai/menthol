use crate::models::image_analysis::ImageAnalyzer;
use crate::models::lore_model::LoreGenerator;
use crate::models::{NFTAttribute, Result};
use std::path::PathBuf;

pub struct LoreService {
    image_analyzer: ImageAnalyzer,
    lore_generator: LoreGenerator,
}

impl LoreService {
    pub fn new(
        model_path: PathBuf,
        tokenizer_path: PathBuf,
        image_model_path: PathBuf,
    ) -> Result<Self> {
        let image_analyzer = ImageAnalyzer::new(image_model_path);
        let lore_generator = LoreGenerator::new(model_path);

        Ok(Self {
            image_analyzer,
            lore_generator,
        })
    }

    /// **Analyze image and generate character name**
    pub fn generate_name(&self, image_path: &PathBuf) -> Result<String> {
        let image_description = self.image_analyzer.analyze(image_path);
        let name = self.lore_generator.generate_name(&image_description);
        Ok(name)
    }

    /// **Generate full lore for an NFT based on image**
    pub fn generate_lore(&self, image_path: &PathBuf) -> Result<String> {
        let image_description = self.image_analyzer.analyze(image_path);
        let lore = self.lore_generator.generate_lore(&image_description);
        Ok(lore)
    }

    /// **Analyze image for description**
    pub fn analyze_image(&self, image_path: &PathBuf) -> Result<String> {
        Ok(self.image_analyzer.analyze(image_path))
    }
}
