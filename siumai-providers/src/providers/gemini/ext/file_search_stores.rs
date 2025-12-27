//! Gemini File Search Stores (extension API).

pub use super::super::file_search_stores::GeminiFileSearchStores;
pub use super::super::types::{
    ChunkingConfig, FileSearchOperation, FileSearchStore, FileSearchStoresList,
    FileSearchUploadConfig, WhiteSpaceChunkingConfig,
};

/// Convenience: get a File Search Stores client from a `GeminiClient`.
pub fn stores(client: &super::super::GeminiClient) -> GeminiFileSearchStores {
    client.file_search_stores()
}
