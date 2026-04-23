//! Vertex AI embedding provider options (Vercel-aligned).

use serde::{Deserialize, Serialize};

/// Provider options for Vertex text embedding models.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
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

impl VertexEmbeddingOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub const fn with_output_dimensionality(mut self, output_dimensionality: u32) -> Self {
        self.output_dimensionality = Some(output_dimensionality);
        self
    }

    pub fn with_task_type(mut self, task_type: crate::types::EmbeddingTaskType) -> Self {
        self.task_type = Some(task_type);
        self
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub const fn with_auto_truncate(mut self, auto_truncate: bool) -> Self {
        self.auto_truncate = Some(auto_truncate);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_options_builder_serializes_to_ai_sdk_shape() {
        let value = serde_json::to_value(
            VertexEmbeddingOptions::new()
                .with_output_dimensionality(256)
                .with_task_type(crate::types::EmbeddingTaskType::RetrievalDocument)
                .with_title("vertex-doc")
                .with_auto_truncate(true),
        )
        .expect("serialize VertexEmbeddingOptions");

        assert_eq!(
            value,
            serde_json::json!({
                "outputDimensionality": 256,
                "taskType": "RETRIEVAL_DOCUMENT",
                "title": "vertex-doc",
                "autoTruncate": true
            })
        );
    }
}
