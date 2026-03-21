//! Provider-native request normalization helpers for gateway adapters.
//!
//! English-only comments in code as requested.

use std::fmt;
use std::sync::Arc;

#[cfg(feature = "anthropic")]
use siumai::experimental::bridge::bridge_anthropic_messages_json_to_chat_request_with_options;
#[cfg(feature = "google")]
use siumai::experimental::bridge::bridge_gemini_generate_content_json_to_chat_request_with_options;
use siumai::experimental::bridge::{
    BridgeCustomization, BridgeMode, BridgeOptions, BridgeOptionsOverride, BridgeResult,
};
#[cfg(feature = "openai")]
use siumai::experimental::bridge::{
    bridge_openai_chat_completions_json_to_chat_request_with_options,
    bridge_openai_responses_json_to_chat_request_with_options,
};
use siumai::prelude::unified::{ChatRequest, LlmError};

use crate::server::{GatewayBridgePolicy, resolve_gateway_bridge_options};

/// Source request wire format for provider-native request normalization helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceRequestFormat {
    /// OpenAI Responses request JSON.
    OpenAiResponses,
    /// OpenAI Chat Completions request JSON.
    OpenAiChatCompletions,
    /// Anthropic Messages request JSON.
    AnthropicMessages,
    /// Gemini/Vertex GenerateContent request JSON.
    GeminiGenerateContent,
}

/// Options for normalizing provider-native request JSON into `ChatRequest`.
#[derive(Clone, Default)]
pub struct NormalizeRequestOptions {
    /// Optional bridge customization applied after provider-owned parsing.
    pub bridge_options: Option<BridgeOptions>,
    /// Optional partial bridge override applied on top of route/policy defaults.
    pub bridge_options_override: Option<BridgeOptionsOverride>,
    /// Optional gateway bridge policy applied by the helper.
    pub policy: Option<GatewayBridgePolicy>,
}

impl fmt::Debug for NormalizeRequestOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NormalizeRequestOptions")
            .field("has_bridge_options", &self.bridge_options.is_some())
            .field(
                "has_bridge_options_override",
                &self.bridge_options_override.is_some(),
            )
            .field("has_policy", &self.policy.is_some())
            .finish()
    }
}

impl NormalizeRequestOptions {
    /// Attach bridge customization options to the request normalization helper.
    pub fn with_bridge_options(mut self, bridge_options: BridgeOptions) -> Self {
        self.bridge_options = Some(bridge_options);
        self
    }

    /// Attach a unified bridge customization object to the request normalization helper.
    pub fn with_bridge_customization(
        mut self,
        customization: Arc<dyn BridgeCustomization>,
    ) -> Self {
        self.bridge_options = Some(
            self.bridge_options
                .take()
                .unwrap_or_else(|| BridgeOptions::new(BridgeMode::BestEffort))
                .with_customization(customization),
        );
        self
    }

    /// Attach a partial bridge override to the request normalization helper.
    pub fn with_bridge_options_override(
        mut self,
        bridge_options_override: BridgeOptionsOverride,
    ) -> Self {
        self.bridge_options_override = Some(bridge_options_override);
        self
    }

    /// Override only the effective bridge mode used by the request normalization helper.
    pub fn with_bridge_mode_override(mut self, mode: BridgeMode) -> Self {
        let override_options = self
            .bridge_options_override
            .take()
            .unwrap_or_default()
            .with_mode(mode);
        self.bridge_options_override = Some(override_options);
        self
    }

    /// Attach a gateway bridge policy to the request normalization helper.
    pub fn with_policy(mut self, policy: GatewayBridgePolicy) -> Self {
        self.policy = Some(policy);
        self
    }
}

/// Normalize provider-native request JSON into `ChatRequest` (best-effort).
pub fn normalize_request_json(
    body: &serde_json::Value,
    source: SourceRequestFormat,
) -> Result<BridgeResult<ChatRequest>, LlmError> {
    normalize_request_json_with_options(body, source, &NormalizeRequestOptions::default())
}

/// Normalize provider-native request JSON into `ChatRequest` with explicit bridge options.
pub fn normalize_request_json_with_options(
    body: &serde_json::Value,
    source: SourceRequestFormat,
    opts: &NormalizeRequestOptions,
) -> Result<BridgeResult<ChatRequest>, LlmError> {
    let bridge_options = resolve_gateway_bridge_options(
        opts.policy.as_ref(),
        opts.bridge_options.clone(),
        opts.bridge_options_override.clone(),
    );

    match source {
        SourceRequestFormat::OpenAiResponses => {
            #[cfg(feature = "openai")]
            {
                bridge_openai_responses_json_to_chat_request_with_options(body, bridge_options)
            }
            #[cfg(not(feature = "openai"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "openai feature is disabled".to_string(),
                ))
            }
        }
        SourceRequestFormat::OpenAiChatCompletions => {
            #[cfg(feature = "openai")]
            {
                bridge_openai_chat_completions_json_to_chat_request_with_options(
                    body,
                    bridge_options,
                )
            }
            #[cfg(not(feature = "openai"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "openai feature is disabled".to_string(),
                ))
            }
        }
        SourceRequestFormat::AnthropicMessages => {
            #[cfg(feature = "anthropic")]
            {
                bridge_anthropic_messages_json_to_chat_request_with_options(body, bridge_options)
            }
            #[cfg(not(feature = "anthropic"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "anthropic feature is disabled".to_string(),
                ))
            }
        }
        SourceRequestFormat::GeminiGenerateContent => {
            #[cfg(feature = "google")]
            {
                bridge_gemini_generate_content_json_to_chat_request_with_options(
                    body,
                    bridge_options,
                )
            }
            #[cfg(not(feature = "google"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "google feature is disabled".to_string(),
                ))
            }
        }
    }
}

#[cfg(test)]
mod request_normalize_tests {
    use std::sync::Arc;

    use serde_json::json;
    use siumai::prelude::unified::MessageRole;

    use crate::bridge::ClosureBridgeCustomization;

    use super::*;

    #[test]
    #[cfg(feature = "openai")]
    fn openai_responses_request_normalize_helper_restores_basic_request() {
        let body = json!({
            "model": "gpt-4o-mini",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Explain ownership in one sentence.",
                        }
                    ]
                }
            ]
        });

        let bridged = normalize_request_json(&body, SourceRequestFormat::OpenAiResponses)
            .expect("normalize openai responses request");
        let (request, report) = bridged.into_result().expect("accepted");

        assert_eq!(request.common_params.model, "gpt-4o-mini");
        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.messages[0].role, MessageRole::User);
        assert!(!report.is_rejected());
    }

    #[test]
    #[cfg(feature = "openai")]
    fn request_normalize_helper_applies_bridge_customization() {
        let body = json!({
            "model": "gpt-4o-mini",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "hello",
                        }
                    ]
                }
            ]
        });

        let options = NormalizeRequestOptions::default().with_bridge_customization(Arc::new(
            ClosureBridgeCustomization::default().with_request(|_, request, report| {
                request.messages[0]
                    .metadata
                    .custom
                    .insert("route".to_string(), json!("gateway"));
                report
                    .lossy_fields
                    .push("request.metadata.route".to_string());
                Ok(())
            }),
        ));

        let bridged = normalize_request_json_with_options(
            &body,
            SourceRequestFormat::OpenAiResponses,
            &options,
        )
        .expect("normalize openai responses request");
        let (request, report) = bridged.into_result().expect("accepted");

        assert_eq!(
            request.messages[0].metadata.custom.get("route"),
            Some(&json!("gateway"))
        );
        assert_eq!(
            report.lossy_fields,
            vec!["request.metadata.route".to_string()]
        );
    }
}
