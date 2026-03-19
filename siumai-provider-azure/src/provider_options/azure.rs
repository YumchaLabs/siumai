//! Azure OpenAI provider options.
//!
//! These typed option structs are owned by the Azure provider crate and are serialized into
//! `providerOptions["azure"]`.

use serde::{Deserialize, Serialize};

/// Azure reasoning effort hints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AzureReasoningEffort {
    Low,
    Medium,
    High,
}

/// Azure `responses_api` options currently consumed by the provider spec.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AzureResponsesApiConfig {
    /// Azure/OpenAI Responses reasoning summary mode.
    #[serde(skip_serializing_if = "Option::is_none", alias = "reasoningSummary")]
    pub reasoning_summary: Option<String>,
}

impl AzureResponsesApiConfig {
    /// Create new Azure Responses options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set reasoning summary mode.
    pub fn with_reasoning_summary(mut self, summary: impl Into<String>) -> Self {
        self.reasoning_summary = Some(summary.into());
        self
    }
}

/// Azure-specific options for chat requests.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AzureOpenAiOptions {
    /// Force reasoning-mode request shaping even when the deployment id does not look like a
    /// reasoning model.
    #[serde(skip_serializing_if = "Option::is_none", alias = "forceReasoning")]
    pub force_reasoning: Option<bool>,

    /// Azure/OpenAI reasoning effort hint.
    #[serde(skip_serializing_if = "Option::is_none", alias = "reasoningEffort")]
    pub reasoning_effort: Option<AzureReasoningEffort>,

    /// Structured-output strictness fallback used when `response_format.strict` is unset.
    #[serde(skip_serializing_if = "Option::is_none", alias = "strictJsonSchema")]
    pub strict_json_schema: Option<bool>,

    /// Azure/OpenAI Responses options.
    #[serde(skip_serializing_if = "Option::is_none", alias = "responsesApi")]
    pub responses_api: Option<AzureResponsesApiConfig>,
}

impl AzureOpenAiOptions {
    /// Create new Azure options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Force or disable reasoning-mode request shaping.
    pub fn with_force_reasoning(mut self, enabled: bool) -> Self {
        self.force_reasoning = Some(enabled);
        self
    }

    /// Set reasoning effort.
    pub fn with_reasoning_effort(mut self, effort: AzureReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }

    /// Set strict JSON schema fallback.
    pub fn with_strict_json_schema(mut self, strict: bool) -> Self {
        self.strict_json_schema = Some(strict);
        self
    }

    /// Attach nested Responses options.
    pub fn with_responses_api(mut self, config: AzureResponsesApiConfig) -> Self {
        self.responses_api = Some(config);
        self
    }

    /// Convenience helper for nested `responses_api.reasoning_summary`.
    pub fn with_responses_reasoning_summary(mut self, summary: impl Into<String>) -> Self {
        let config = self
            .responses_api
            .take()
            .unwrap_or_default()
            .with_reasoning_summary(summary);
        self.responses_api = Some(config);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn azure_options_serialize_sparse_snake_case_shape() {
        let value = serde_json::to_value(
            AzureOpenAiOptions::new()
                .with_force_reasoning(true)
                .with_reasoning_effort(AzureReasoningEffort::High)
                .with_strict_json_schema(false)
                .with_responses_reasoning_summary("detailed"),
        )
        .expect("serialize azure options");

        assert_eq!(
            value,
            serde_json::json!({
                "force_reasoning": true,
                "reasoning_effort": "high",
                "strict_json_schema": false,
                "responses_api": {
                    "reasoning_summary": "detailed"
                }
            })
        );
    }

    #[test]
    fn azure_options_deserialize_camel_case_aliases() {
        let options: AzureOpenAiOptions = serde_json::from_value(serde_json::json!({
            "forceReasoning": true,
            "reasoningEffort": "medium",
            "strictJsonSchema": true,
            "responsesApi": {
                "reasoningSummary": "auto"
            }
        }))
        .expect("deserialize azure options");

        assert_eq!(options.force_reasoning, Some(true));
        assert_eq!(options.reasoning_effort, Some(AzureReasoningEffort::Medium));
        assert_eq!(options.strict_json_schema, Some(true));
        assert_eq!(
            options
                .responses_api
                .and_then(|config| config.reasoning_summary),
            Some("auto".to_string())
        );
    }
}
