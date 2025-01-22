use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NFT {
    pub name: String,
    pub symbol: String,
    pub description: String,
    pub image_uri: String,
    pub attributes: Vec<NFTAttribute>,
    pub collection: NFTCollection,
    pub metadata: NFTMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NFTAttribute {
    pub trait_type: String,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NFTCollection {
    pub name: String,
    pub family: String,
    pub collection_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NFTMetadata {
    pub seller_fee_basis_points: u16,
    pub creation_date: String,
    pub files: Vec<NFTFile>,
    pub properties: NFTProperties,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NFTFile {
    pub uri: String,
    pub file_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NFTProperties {
    pub category: String,
    pub creators: Vec<NFTCreator>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NFTCreator {
    pub address: String,
    pub share: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalNFTStorage {
    pub image_path: PathBuf,
    pub metadata_path: PathBuf,
    pub arweave_paths: Option<ArweavePaths>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArweavePaths {
    pub image_uri: String,
    pub metadata_uri: String,
}
