use crate::models::{ModelType, NFTError, Result};
use std::collections::HashMap;
use std::path::PathBuf;

pub struct ModelService {
    models: HashMap<String, LoadedModel>,
    cache_dir: PathBuf,
}

struct LoadedModel {
    path: PathBuf,
    model_type: ModelType,
    last_used: std::time::SystemTime,
    memory_usage: usize,
}

impl ModelService {
    pub fn new(cache_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&cache_dir).map_err(|e| NFTError::FileSystemError(e))?;

        Ok(Self {
            models: HashMap::new(),
            cache_dir,
        })
    }

    pub fn load_model(&mut self, path: PathBuf, model_type: ModelType) -> Result<()> {
        let model_name = path
            .file_name()
            .ok_or_else(|| NFTError::ConfigurationError("Invalid model path".to_string()))?
            .to_string_lossy()
            .to_string();

        // Check if we need to free up memory
        self.manage_memory()?;

        // Load the model
        let model = LoadedModel {
            path: path.clone(),
            model_type,
            last_used: std::time::SystemTime::now(),
            memory_usage: 0, // TODO: Calculate actual memory usage
        };

        self.models.insert(model_name, model);
        Ok(())
    }

    pub fn get_model(&mut self, name: &str) -> Result<&LoadedModel> {
        if let Some(model) = self.models.get_mut(name) {
            model.last_used = std::time::SystemTime::now();
            Ok(model)
        } else {
            Err(NFTError::ConfigurationError(format!(
                "Model {} not loaded",
                name
            )))
        }
    }

    fn manage_memory(&mut self) -> Result<()> {
        const MAX_MEMORY_USAGE: usize = 8 * 1024 * 1024 * 1024; // 8GB

        // First collect total usage
        let total_usage: usize = self.models.values().map(|m| m.memory_usage).sum();

        if total_usage > MAX_MEMORY_USAGE {
            let mut models: Vec<(String, std::time::SystemTime)> = self
                .models
                .iter()
                .map(|(name, model)| (name.clone(), model.last_used))
                .collect();

            models.sort_by_key(|(_, last_used)| *last_used);

            for (name, _) in models {
                let current_usage: usize = self.models.values().map(|m| m.memory_usage).sum();
                if current_usage <= MAX_MEMORY_USAGE {
                    break;
                }
                self.models.remove(&name);
            }
        }

        Ok(())
    }

    pub fn unload_model(&mut self, name: &str) -> Result<()> {
        self.models
            .remove(name)
            .ok_or_else(|| NFTError::ConfigurationError(format!("Model {} not loaded", name)))?;
        Ok(())
    }

    pub fn list_loaded_models(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }
}
