//! Typed provider options for Amazon Bedrock.

use serde::{Deserialize, Serialize};

/// Typed chat options stored under `provider_options_map["bedrock"]`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BedrockChatOptions {
    /// Additional provider-native inference fields passed through to Bedrock.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "additionalModelRequestFields"
    )]
    pub additional_model_request_fields: Option<serde_json::Value>,
}

impl BedrockChatOptions {
    /// Create empty Bedrock chat options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set `additionalModelRequestFields`.
    pub fn with_additional_model_request_fields(
        mut self,
        additional_model_request_fields: serde_json::Value,
    ) -> Self {
        self.additional_model_request_fields = Some(additional_model_request_fields);
        self
    }
}

/// Typed rerank options stored under `provider_options_map["bedrock"]`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BedrockRerankOptions {
    /// AWS region used to build the foundation model ARN.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// Pagination token.
    #[serde(skip_serializing_if = "Option::is_none", rename = "nextToken")]
    pub next_token: Option<String>,
    /// Additional provider-native model fields passed through to Bedrock.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "additionalModelRequestFields"
    )]
    pub additional_model_request_fields: Option<serde_json::Value>,
}

impl BedrockRerankOptions {
    /// Create empty Bedrock rerank options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set `region`.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Set `nextToken`.
    pub fn with_next_token(mut self, next_token: impl Into<String>) -> Self {
        self.next_token = Some(next_token.into());
        self
    }

    /// Set `additionalModelRequestFields`.
    pub fn with_additional_model_request_fields(
        mut self,
        additional_model_request_fields: serde_json::Value,
    ) -> Self {
        self.additional_model_request_fields = Some(additional_model_request_fields);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_options_serialize_expected_shape() {
        let value = serde_json::to_value(
            BedrockChatOptions::new()
                .with_additional_model_request_fields(serde_json::json!({ "topK": 32 })),
        )
        .expect("serialize options");

        assert_eq!(
            value,
            serde_json::json!({
                "additionalModelRequestFields": { "topK": 32 }
            })
        );
    }

    #[test]
    fn rerank_options_serialize_expected_shape() {
        let value = serde_json::to_value(
            BedrockRerankOptions::new()
                .with_region("us-east-1")
                .with_next_token("token-1")
                .with_additional_model_request_fields(serde_json::json!({ "topK": 4 })),
        )
        .expect("serialize options");

        assert_eq!(
            value,
            serde_json::json!({
                "region": "us-east-1",
                "nextToken": "token-1",
                "additionalModelRequestFields": { "topK": 4 }
            })
        );
    }
}
