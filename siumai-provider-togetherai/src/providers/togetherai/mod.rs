//! `TogetherAI` provider module.
//!
//! This module exposes a provider-owned config/client/builder surface for the
//! TogetherAI native surfaces that currently back reranking plus provider-owned
//! request option helpers used by the unified registry facade.

pub mod builder;
pub mod client;
pub mod config;
pub mod ext;
pub mod models;

pub use builder::TogetherAiBuilder;
pub use client::TogetherAiClient;
pub use config::TogetherAiConfig;
pub use ext::{TogetherAiImageRequestExt, TogetherAiRerankRequestExt};

/// AI SDK-aligned TogetherAI error envelope.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TogetherAIErrorData {
    pub error: TogetherAIErrorPayload,
}

/// AI SDK-aligned TogetherAI error payload.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TogetherAIErrorPayload {
    pub message: String,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub error_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::TogetherAIErrorData;

    #[test]
    fn togetherai_error_data_deserializes_ai_sdk_shape() {
        let data: TogetherAIErrorData = serde_json::from_value(serde_json::json!({
            "error": {
                "message": "rate limit exceeded",
                "type": "rate_limit_error",
                "code": "too_many_requests"
            }
        }))
        .expect("togetherai error data should deserialize");

        assert_eq!(data.error.message, "rate limit exceeded");
        assert_eq!(data.error.error_type.as_deref(), Some("rate_limit_error"));
        assert_eq!(
            data.error.code,
            Some(serde_json::json!("too_many_requests"))
        );
    }
}
