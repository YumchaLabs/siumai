use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::OpenAiMetadata;

/// Static chunking settings accepted by OpenAI vector stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiStaticChunkingSettings {
    pub max_chunk_size_tokens: u16,
    pub chunk_overlap_tokens: u16,
}

/// Chunking policy for newly attached vector-store files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAiChunkingStrategy {
    Auto,
    Static {
        #[serde(rename = "static")]
        settings: OpenAiStaticChunkingSettings,
    },
}

/// Vector-store expiration relative to the last active timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiVectorStoreExpiration {
    pub anchor: OpenAiVectorStoreExpirationAnchor,
    pub days: u32,
}

impl OpenAiVectorStoreExpiration {
    pub const fn after_last_active(days: u32) -> Self {
        Self {
            anchor: OpenAiVectorStoreExpirationAnchor::LastActiveAt,
            days,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiVectorStoreExpirationAnchor {
    LastActiveAt,
}

/// Request body for creating a vector store.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OpenAiVectorStoreCreateRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_after: Option<OpenAiVectorStoreExpiration>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub file_ids: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunking_strategy: Option<OpenAiChunkingStrategy>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: OpenAiMetadata,
}

impl OpenAiVectorStoreCreateRequest {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Request body for updating mutable vector-store fields.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OpenAiVectorStoreUpdateRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_after: Option<OpenAiVectorStoreExpiration>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<OpenAiMetadata>,
}

/// File-count summary for a vector store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiVectorStoreFileCounts {
    pub cancelled: u64,
    pub completed: u64,
    pub failed: u64,
    pub in_progress: u64,
    pub total: u64,
}

/// OpenAI vector-store metadata and lifecycle state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAiVectorStore {
    pub id: String,
    pub object: String,
    pub created_at: i64,
    pub file_counts: OpenAiVectorStoreFileCounts,
    #[serde(default)]
    pub last_active_at: Option<i64>,
    #[serde(default)]
    pub metadata: OpenAiMetadata,
    pub name: String,
    pub status: String,
    pub usage_bytes: u64,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub expires_after: Option<OpenAiVectorStoreExpiration>,
    #[serde(default)]
    pub expires_at: Option<i64>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Deletion acknowledgement for a vector store.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAiVectorStoreDeleted {
    pub id: String,
    pub object: String,
    pub deleted: bool,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Request body for attaching one uploaded file to a vector store.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAiVectorStoreFileAttachRequest {
    pub file_id: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub attributes: BTreeMap<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunking_strategy: Option<OpenAiChunkingStrategy>,
}

impl OpenAiVectorStoreFileAttachRequest {
    pub fn new(file_id: impl Into<String>) -> Self {
        Self {
            file_id: file_id.into(),
            attributes: BTreeMap::new(),
            chunking_strategy: None,
        }
    }
}

/// Last ingestion error associated with a vector-store file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAiVectorStoreFileError {
    pub code: String,
    pub message: String,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// One file attachment and ingestion state inside a vector store.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAiVectorStoreFile {
    pub id: String,
    pub object: String,
    pub created_at: i64,
    #[serde(default)]
    pub last_error: Option<OpenAiVectorStoreFileError>,
    pub status: String,
    pub usage_bytes: u64,
    pub vector_store_id: String,
    #[serde(default)]
    pub attributes: BTreeMap<String, Value>,
    #[serde(default)]
    pub chunking_strategy: Option<Value>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Detachment acknowledgement for a vector-store file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAiVectorStoreFileDeleted {
    pub id: String,
    pub object: String,
    pub deleted: bool,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn static_chunking_uses_the_provider_wire_shape() {
        let request = OpenAiVectorStoreFileAttachRequest {
            file_id: "file_123".to_string(),
            attributes: BTreeMap::new(),
            chunking_strategy: Some(OpenAiChunkingStrategy::Static {
                settings: OpenAiStaticChunkingSettings {
                    max_chunk_size_tokens: 800,
                    chunk_overlap_tokens: 400,
                },
            }),
        };
        assert_eq!(
            serde_json::to_value(request).unwrap(),
            json!({
                "file_id": "file_123",
                "chunking_strategy": {
                    "type": "static",
                    "static": {
                        "max_chunk_size_tokens": 800,
                        "chunk_overlap_tokens": 400
                    }
                }
            })
        );
    }
}
