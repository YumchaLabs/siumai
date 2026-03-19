//! Vertex-specific response metadata.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub use siumai_protocol_gemini::standards::gemini::GeminiSource as VertexSource;
pub use siumai_protocol_gemini::standards::gemini::types::{
    GroundingMetadata as VertexGroundingMetadata, LogprobsResult as VertexLogprobsResult,
    PromptFeedback as VertexPromptFeedback, SafetyRating as VertexSafetyRating,
    UrlContextMetadata as VertexUrlContextMetadata, UsageMetadata as VertexUsageMetadata,
};

/// Vertex-specific metadata from chat responses.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VertexMetadata {
    /// Per-part or response-level thought signature metadata.
    #[serde(skip_serializing_if = "Option::is_none", rename = "thoughtSignature")]
    pub thought_signature: Option<String>,

    /// Grounding metadata (for search-grounded responses).
    #[serde(skip_serializing_if = "Option::is_none", rename = "groundingMetadata")]
    pub grounding_metadata: Option<VertexGroundingMetadata>,

    /// Sources extracted from provider-hosted grounding chunks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sources: Option<Vec<VertexSource>>,

    /// URL context metadata.
    #[serde(skip_serializing_if = "Option::is_none", rename = "urlContextMetadata")]
    pub url_context_metadata: Option<VertexUrlContextMetadata>,

    /// Safety ratings.
    #[serde(skip_serializing_if = "Option::is_none", rename = "safetyRatings")]
    pub safety_ratings: Option<Vec<VertexSafetyRating>>,

    /// Prompt feedback metadata.
    #[serde(skip_serializing_if = "Option::is_none", rename = "promptFeedback")]
    pub prompt_feedback: Option<VertexPromptFeedback>,

    /// Average log probability across all tokens in the candidate.
    #[serde(skip_serializing_if = "Option::is_none", rename = "avgLogprobs")]
    pub avg_logprobs: Option<f64>,

    /// Logprobs result payload.
    #[serde(skip_serializing_if = "Option::is_none", rename = "logprobsResult")]
    pub logprobs_result: Option<VertexLogprobsResult>,

    /// Provider-side usage metadata.
    #[serde(skip_serializing_if = "Option::is_none", rename = "usageMetadata")]
    pub usage_metadata: Option<VertexUsageMetadata>,

    /// Preserve unknown fields for forward compatibility.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl crate::types::provider_metadata::FromMetadata for VertexMetadata {
    fn from_metadata(
        metadata: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Option<Self> {
        serde_json::from_value(serde_json::to_value(metadata).ok()?).ok()
    }
}

/// Typed helper for Vertex metadata extraction from `ChatResponse`.
pub trait VertexChatResponseExt {
    fn vertex_metadata(&self) -> Option<VertexMetadata>;
}

impl VertexChatResponseExt for crate::types::ChatResponse {
    fn vertex_metadata(&self) -> Option<VertexMetadata> {
        use crate::types::provider_metadata::FromMetadata;
        let meta = self.provider_metadata.as_ref()?.get("vertex")?;
        VertexMetadata::from_metadata(meta)
    }
}

/// Typed helper for Vertex metadata extraction from `ContentPart`.
pub trait VertexContentPartExt {
    fn vertex_metadata(&self) -> Option<VertexMetadata>;
}

impl VertexContentPartExt for crate::types::ContentPart {
    fn vertex_metadata(&self) -> Option<VertexMetadata> {
        use crate::types::ContentPart;

        let provider_metadata = match self {
            ContentPart::Text {
                provider_metadata, ..
            }
            | ContentPart::Image {
                provider_metadata, ..
            }
            | ContentPart::Audio {
                provider_metadata, ..
            }
            | ContentPart::File {
                provider_metadata, ..
            }
            | ContentPart::ToolCall {
                provider_metadata, ..
            }
            | ContentPart::ToolResult {
                provider_metadata, ..
            }
            | ContentPart::Reasoning {
                provider_metadata, ..
            } => provider_metadata.as_ref()?,
            ContentPart::Source { .. }
            | ContentPart::ToolApprovalResponse { .. }
            | ContentPart::ToolApprovalRequest { .. } => return None,
        };

        let inner = provider_metadata.get("vertex")?;
        serde_json::from_value(inner.clone()).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_metadata_reads_vertex_namespace_only() {
        let mut response =
            crate::types::ChatResponse::new(crate::types::MessageContent::Text("ok".to_string()));

        let mut google_inner = HashMap::new();
        google_inner.insert(
            "thoughtSignature".to_string(),
            serde_json::json!("google-sig"),
        );
        let mut vertex_inner = HashMap::new();
        vertex_inner.insert(
            "thoughtSignature".to_string(),
            serde_json::json!("vertex-sig"),
        );
        vertex_inner.insert(
            "usageMetadata".to_string(),
            serde_json::json!({
                "promptTokenCount": 7,
                "totalTokenCount": 11
            }),
        );
        vertex_inner.insert(
            "safetyRatings".to_string(),
            serde_json::json!([
                {
                    "category": "HARM_CATEGORY_DEROGATORY",
                    "probability": "NEGLIGIBLE"
                }
            ]),
        );

        let mut outer = HashMap::new();
        outer.insert("google".to_string(), google_inner);
        outer.insert("vertex".to_string(), vertex_inner);
        response.provider_metadata = Some(outer);

        let parsed = response.vertex_metadata().expect("vertex metadata");
        assert_eq!(parsed.thought_signature.as_deref(), Some("vertex-sig"));
        assert_eq!(
            parsed
                .usage_metadata
                .as_ref()
                .and_then(|usage| usage.total_token_count),
            Some(11)
        );
        let first_rating = parsed
            .safety_ratings
            .as_ref()
            .and_then(|ratings| ratings.first())
            .expect("expected first safety rating");
        assert_eq!(
            serde_json::to_value(first_rating)
                .expect("serialize safety rating")
                .get("category")
                .cloned(),
            Some(serde_json::json!("HARM_CATEGORY_DEROGATORY"))
        );
    }

    #[test]
    fn vertex_metadata_returns_none_when_only_google_namespace_exists() {
        let mut response =
            crate::types::ChatResponse::new(crate::types::MessageContent::Text("ok".to_string()));
        let mut google_inner = HashMap::new();
        google_inner.insert(
            "thoughtSignature".to_string(),
            serde_json::json!("google-sig"),
        );
        let mut outer = HashMap::new();
        outer.insert("google".to_string(), google_inner);
        response.provider_metadata = Some(outer);

        assert!(response.vertex_metadata().is_none());
    }

    #[test]
    fn vertex_content_part_metadata_reads_thought_signature() {
        let part = crate::types::ContentPart::Reasoning {
            text: "thinking".to_string(),
            provider_metadata: Some(HashMap::from([(
                "vertex".to_string(),
                serde_json::json!({
                    "thoughtSignature": "sig_reasoning"
                }),
            )])),
        };

        let parsed = part.vertex_metadata().expect("vertex part metadata");
        assert_eq!(parsed.thought_signature.as_deref(), Some("sig_reasoning"));
    }
}
