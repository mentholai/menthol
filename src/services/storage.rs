use crate::models::{NFTError, Result, StorageConfig, NFT};
use chrono::Utc;
use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;

pub struct StorageService {
    config: StorageConfig,
}

impl StorageService {
    pub fn new(config: StorageConfig) -> Result<Self> {
        // Create all necessary directories
        fs::create_dir_all(&config.output_dir).map_err(|e| NFTError::FileSystemError(e))?;
        fs::create_dir_all(&config.cache_dir).map_err(|e| NFTError::FileSystemError(e))?;
        fs::create_dir_all(&config.temp_dir).map_err(|e| NFTError::FileSystemError(e))?;
        fs::create_dir_all(&config.metadata_dir).map_err(|e| NFTError::FileSystemError(e))?;

        Ok(Self { config })
    }

    // Save generated image
    pub fn save_image(&self, image_data: &[u8], extension: &str) -> Result<PathBuf> {
        let filename = format!("{}.{}", Uuid::new_v4(), extension);
        let path = self.config.output_dir.join(filename);

        fs::write(&path, image_data).map_err(|e| NFTError::FileSystemError(e))?;

        Ok(path)
    }

    // Save NFT metadata
    pub fn save_metadata(&self, nft: &NFT) -> Result<PathBuf> {
        let filename = format!(
            "{}_metadata.json",
            nft.name.to_lowercase().replace(" ", "_")
        );
        let path = self.config.metadata_dir.join(filename);

        let metadata_json =
            serde_json::to_string_pretty(nft).map_err(|e| NFTError::SerializationError(e))?;

        fs::write(&path, metadata_json).map_err(|e| NFTError::FileSystemError(e))?;

        Ok(path)
    }

    // Create temporary file
    pub fn create_temp_file(&self, prefix: &str) -> Result<PathBuf> {
        let filename = format!("{}_{}", prefix, Uuid::new_v4());
        let path = self.config.temp_dir.join(filename);

        // Create empty file to reserve the path
        fs::File::create(&path).map_err(|e| NFTError::FileSystemError(e))?;

        Ok(path)
    }

    // Cache management
    pub fn cache_file(&self, data: &[u8], key: &str) -> Result<PathBuf> {
        let cache_path = self.config.cache_dir.join(key);

        fs::write(&cache_path, data).map_err(|e| NFTError::FileSystemError(e))?;

        Ok(cache_path)
    }

    pub fn get_cached_file(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let cache_path = self.config.cache_dir.join(key);

        if cache_path.exists() {
            let data = fs::read(&cache_path).map_err(|e| NFTError::FileSystemError(e))?;
            Ok(Some(data))
        } else {
            Ok(None)
        }
    }

    // Cleanup functions
    pub fn cleanup_temp_files(&self) -> Result<()> {
        self.cleanup_directory(&self.config.temp_dir)
    }

    pub fn cleanup_old_cache(&self, max_age_days: u64) -> Result<()> {
        let now = Utc::now();

        for entry in
            fs::read_dir(&self.config.cache_dir).map_err(|e| NFTError::FileSystemError(e))?
        {
            let entry = entry.map_err(|e| NFTError::FileSystemError(e))?;
            let metadata = entry.metadata().map_err(|e| NFTError::FileSystemError(e))?;

            if let Ok(modified) = metadata.modified() {
                let modified = chrono::DateTime::<Utc>::from(modified);
                let age = now.signed_duration_since(modified);

                if age.num_days() > max_age_days as i64 {
                    fs::remove_file(entry.path()).map_err(|e| NFTError::FileSystemError(e))?;
                }
            }
        }

        Ok(())
    }

    // Helper functions
    fn cleanup_directory(&self, dir: &Path) -> Result<()> {
        for entry in fs::read_dir(dir).map_err(|e| NFTError::FileSystemError(e))? {
            let entry = entry.map_err(|e| NFTError::FileSystemError(e))?;
            fs::remove_file(entry.path()).map_err(|e| NFTError::FileSystemError(e))?;
        }

        Ok(())
    }

    pub fn ensure_directory_exists(&self, path: &Path) -> Result<()> {
        if !path.exists() {
            fs::create_dir_all(path).map_err(|e| NFTError::FileSystemError(e))?;
        }
        Ok(())
    }

    // Utility functions for managing disk space
    pub fn get_directory_size(&self, path: &Path) -> Result<u64> {
        let mut total_size = 0;

        if path.is_dir() {
            for entry in fs::read_dir(path).map_err(|e| NFTError::FileSystemError(e))? {
                let entry = entry.map_err(|e| NFTError::FileSystemError(e))?;
                let metadata = entry.metadata().map_err(|e| NFTError::FileSystemError(e))?;

                total_size += metadata.len();
            }
        }

        Ok(total_size)
    }

    pub fn check_available_space(&self, path: &Path) -> Result<u64> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let metadata = fs::metadata(path).map_err(|e| NFTError::FileSystemError(e))?;
            Ok(metadata.blocks() * 512)
        }
        #[cfg(windows)]
        {
            Ok(0) // TODO: Implement for Windows
        }
    }
}
