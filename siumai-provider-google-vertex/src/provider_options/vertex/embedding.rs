//! Vertex AI embedding provider options (Vercel-aligned).

use serde::{Deserialize, Serialize};

/// Provider options for Vertex text embedding models.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VertexEmbeddingOptions {
    /// Optional reduced dimension for the output embedding.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_dimensionality: Option<u32>,

    /// Optional task type for generating embeddings (Vertex API uses SCREAMING_SNAKE_CASE values).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_type: Option<crate::types::EmbeddingTaskType>,

    /// Optional title (typically valid for retrieval-document tasks).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,

    /// Optional auto-truncate toggle.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auto_truncate: Option<bool>,
}
