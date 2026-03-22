//! High-value direct request bridge pairs.
//!
//! These pair modules host the request bridge implementations that are worth
//! maintaining directly instead of routing through the normalized backbone.

#[cfg(all(feature = "anthropic", feature = "openai"))]
mod anthropic_messages_to_openai_responses;
#[cfg(all(feature = "anthropic", feature = "openai"))]
mod mcp;
#[cfg(all(feature = "anthropic", feature = "openai"))]
mod openai_responses_to_anthropic_messages;
#[cfg(all(feature = "anthropic", feature = "openai"))]
mod tool_rules;

use siumai_core::LlmError;
use siumai_core::bridge::BridgeReport;
use siumai_core::bridge::BridgeTarget;
use siumai_core::types::ChatRequest;

/// Explicit direct request bridge pairs worth implementing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DirectRequestBridgePair {
    AnthropicMessagesToOpenAiResponses,
    OpenAiResponsesToAnthropicMessages,
}

impl DirectRequestBridgePair {
    pub const fn source(self) -> BridgeTarget {
        match self {
            Self::AnthropicMessagesToOpenAiResponses => BridgeTarget::AnthropicMessages,
            Self::OpenAiResponsesToAnthropicMessages => BridgeTarget::OpenAiResponses,
        }
    }

    pub const fn target(self) -> BridgeTarget {
        match self {
            Self::AnthropicMessagesToOpenAiResponses => BridgeTarget::OpenAiResponses,
            Self::OpenAiResponsesToAnthropicMessages => BridgeTarget::AnthropicMessages,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AnthropicMessagesToOpenAiResponses => "anthropic-messages-to-openai-responses",
            Self::OpenAiResponsesToAnthropicMessages => "openai-responses-to-anthropic-messages",
        }
    }
}

/// Return a direct bridge pair for the given source/target combination when one exists.
pub const fn direct_request_bridge_pair(
    source: BridgeTarget,
    target: BridgeTarget,
) -> Option<DirectRequestBridgePair> {
    match (source, target) {
        (BridgeTarget::AnthropicMessages, BridgeTarget::OpenAiResponses) => {
            Some(DirectRequestBridgePair::AnthropicMessagesToOpenAiResponses)
        }
        (BridgeTarget::OpenAiResponses, BridgeTarget::AnthropicMessages) => {
            Some(DirectRequestBridgePair::OpenAiResponsesToAnthropicMessages)
        }
        _ => None,
    }
}

/// Serialize a request via a direct pair bridge when implemented.
pub(crate) fn serialize_direct_request_bridge_pair(
    pair: DirectRequestBridgePair,
    request: &ChatRequest,
    report: &mut BridgeReport,
) -> Result<serde_json::Value, LlmError> {
    match pair {
        DirectRequestBridgePair::AnthropicMessagesToOpenAiResponses => {
            #[cfg(all(feature = "anthropic", feature = "openai"))]
            {
                anthropic_messages_to_openai_responses::serialize_anthropic_messages_to_openai_responses(
                    request, report,
                )
            }
            #[cfg(not(all(feature = "anthropic", feature = "openai")))]
            {
                let _ = (request, report);
                Err(LlmError::UnsupportedOperation(
                    "anthropic/openai direct bridge requires both features".to_string(),
                ))
            }
        }
        DirectRequestBridgePair::OpenAiResponsesToAnthropicMessages => {
            #[cfg(all(feature = "anthropic", feature = "openai"))]
            {
                openai_responses_to_anthropic_messages::serialize_openai_responses_to_anthropic_messages(
                    request, report,
                )
            }
            #[cfg(not(all(feature = "anthropic", feature = "openai")))]
            {
                let _ = (request, report);
                Err(LlmError::UnsupportedOperation(
                    "anthropic/openai direct bridge requires both features".to_string(),
                ))
            }
        }
    }
}
