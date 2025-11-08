use serde::{Deserialize, Serialize};

/// File Search Store resource
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSearchStore {
    /// Resource name, e.g., "fileSearchStores/abc123"
    pub name: String,
    /// Optional display name
    #[serde(skip_serializing_if = "Option::is_none", rename = "displayName")]
    pub display_name: Option<String>,
    /// Create time (RFC 3339)
    #[serde(skip_serializing_if = "Option::is_none", rename = "createTime")]
    pub create_time: Option<String>,
}

/// Long-running operation structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSearchOperation {
    /// Operation resource name, e.g., "operations/xyz"
    pub name: String,
    /// Whether the operation is done
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done: Option<bool>,
    /// Successful response (schema varies by operation)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<serde_json::Value>,
    /// Error content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<OperationError>,
}

/// Operation error payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationError {
    pub code: Option<i32>,
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Vec<serde_json::Value>>,
}

/// List response for File Search Stores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSearchStoresList {
    /// Stores
    #[serde(default, rename = "fileSearchStores")]
    pub file_search_stores: Vec<FileSearchStore>,
    /// Next page token
    #[serde(skip_serializing_if = "Option::is_none", rename = "nextPageToken")]
    pub next_page_token: Option<String>,
}

/// White-space based chunking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhiteSpaceChunkingConfig {
    #[serde(rename = "max_tokens_per_chunk")]
    pub max_tokens_per_chunk: u32,
    #[serde(rename = "max_overlap_tokens")]
    pub max_overlap_tokens: u32,
}

/// Chunking configuration container
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChunkingConfig {
    /// White-space chunking config; other modes can be added in the future
    #[serde(skip_serializing_if = "Option::is_none", rename = "white_space_config")]
    pub white_space_config: Option<WhiteSpaceChunkingConfig>,
}

/// Upload configuration wrapper for File Search operations
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FileSearchUploadConfig {
    /// Chunking configuration
    #[serde(skip_serializing_if = "Option::is_none", rename = "chunking_config")]
    pub chunking_config: Option<ChunkingConfig>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize_upload_config_whitespace() {
        let cfg = FileSearchUploadConfig {
            chunking_config: Some(ChunkingConfig {
                white_space_config: Some(WhiteSpaceChunkingConfig {
                    max_tokens_per_chunk: 200,
                    max_overlap_tokens: 20,
                }),
            }),
        };
        let v = serde_json::to_value(&cfg).expect("serialize");
        assert!(v.get("chunking_config").is_some());
        let ws = &v["chunking_config"]["white_space_config"];
        assert_eq!(ws["max_tokens_per_chunk"].as_u64(), Some(200));
        assert_eq!(ws["max_overlap_tokens"].as_u64(), Some(20));
    }
}
