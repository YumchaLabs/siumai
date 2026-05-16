//! Anthropic extended thinking helpers (extension API).

use crate::error::LlmError;
use crate::provider_metadata::anthropic::{AnthropicChatResponseExt, AnthropicContentPartExt};
use crate::provider_options::anthropic::{AnthropicOptions, ThinkingModeConfig};
use crate::providers::anthropic::ext::AnthropicChatRequestExt;
use crate::traits::ChatCapability;
use crate::types::{ChatMessage, ChatRequest, ChatResponse, ContentPart, MessageContent};

/// Execute a chat request with Anthropic Thinking Mode enabled (explicit extension API).
pub async fn chat_with_thinking<C>(
    client: &C,
    mut request: ChatRequest,
    config: ThinkingModeConfig,
) -> Result<ChatResponse, LlmError>
where
    C: ChatCapability + ?Sized,
{
    request = request.with_anthropic_options(AnthropicOptions::new().with_thinking_mode(config));
    client.chat_request(request).await
}

fn merge_anthropic_reasoning_options(
    part: &mut ContentPart,
    key: &str,
    value: serde_json::Value,
) -> bool {
    let Some(provider_options) = part.provider_options_mut() else {
        return false;
    };

    match provider_options.0.get_mut("anthropic") {
        Some(existing) if existing.is_object() => {
            if let Some(obj) = existing.as_object_mut() {
                obj.insert(key.to_string(), value);
            }
        }
        _ => {
            provider_options.insert("anthropic", serde_json::json!({ key: value }));
        }
    }

    true
}

/// Convert an Anthropic response into a replayable assistant message with thinking metadata.
///
/// Vercel AI SDK can replay Anthropic thinking blocks by attaching `signature` or `redactedData`
/// to reasoning parts. This helper carries response-side Anthropic reasoning metadata into
/// request-side `provider_options["anthropic"]` on the corresponding reasoning parts.
///
/// Notes:
/// - This is an Anthropic-specific escape hatch (not part of the unified surface).
/// - If `redacted_thinking_data` exists but the response has no reasoning parts, a placeholder
///   reasoning part is appended so that the request converter can emit `redacted_thinking`.
pub fn assistant_message_with_thinking_metadata(response: &ChatResponse) -> ChatMessage {
    let mut msg = response.to_assistant_message();

    let Some(anthropic) = response.anthropic_metadata() else {
        return msg;
    };

    let mut assigned_signature = false;
    let mut assigned_redacted = false;

    if let MessageContent::MultiModal(parts) = &mut msg.content {
        for part in parts.iter_mut() {
            let Some(reasoning_meta) = part.anthropic_reasoning_metadata() else {
                continue;
            };

            if let Some(signature) = reasoning_meta.signature {
                assigned_signature = merge_anthropic_reasoning_options(
                    part,
                    "signature",
                    serde_json::json!(signature),
                ) || assigned_signature;
            }
            if let Some(redacted_data) = reasoning_meta.redacted_data {
                assigned_redacted = merge_anthropic_reasoning_options(
                    part,
                    "redactedData",
                    serde_json::json!(redacted_data),
                ) || assigned_redacted;
            }
        }
    }

    if !assigned_signature
        && let Some(signature) = anthropic.thinking_signature.as_deref()
        && let MessageContent::MultiModal(parts) = &mut msg.content
    {
        for part in parts.iter_mut() {
            if !matches!(part, ContentPart::Reasoning { .. }) {
                continue;
            }
            assigned_signature =
                merge_anthropic_reasoning_options(part, "signature", serde_json::json!(signature));
            if assigned_signature {
                break;
            }
        }
    }

    if !assigned_redacted && let Some(redacted_data) = anthropic.redacted_thinking_data.as_deref() {
        if !msg.has_reasoning() {
            msg.content = match msg.content {
                MessageContent::Text(t) if !t.is_empty() => MessageContent::MultiModal(vec![
                    ContentPart::text(t),
                    ContentPart::reasoning(""),
                ]),
                MessageContent::MultiModal(mut parts) => {
                    parts.push(ContentPart::reasoning(""));
                    MessageContent::MultiModal(parts)
                }
                _ => MessageContent::MultiModal(vec![ContentPart::reasoning("")]),
            };
        }

        if let MessageContent::MultiModal(parts) = &mut msg.content {
            for part in parts.iter_mut() {
                if !matches!(part, ContentPart::Reasoning { .. }) {
                    continue;
                }
                assigned_redacted = merge_anthropic_reasoning_options(
                    part,
                    "redactedData",
                    serde_json::json!(redacted_data),
                );
                if assigned_redacted {
                    break;
                }
            }
        }
    }

    msg
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source_section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
        let start_index = source.find(start).expect("section start marker");
        let end_index = source[start_index..]
            .find(end)
            .map(|offset| start_index + offset)
            .expect("section end marker");
        &source[start_index..end_index]
    }

    #[test]
    fn thinking_request_extension_source_does_not_read_response_metadata() {
        let source = include_str!("thinking.rs");
        let request_source = source_section(source, "pub async fn chat_with_thinking", "fn merge_");

        for disallowed in [
            "provider_metadata",
            "anthropic_metadata",
            "anthropic_reasoning_metadata",
        ] {
            assert!(
                !request_source.contains(disallowed),
                "Anthropic thinking request extension must stay request-only"
            );
        }
    }

    #[test]
    fn thinking_replay_bridge_stays_explicit_cross_step_exception() {
        let source = include_str!("thinking.rs");
        let replay_source = source_section(
            source,
            "pub fn assistant_message_with_thinking_metadata",
            "#[cfg(test)]",
        );

        for required in [
            "response.anthropic_metadata()",
            "part.anthropic_reasoning_metadata()",
            "merge_anthropic_reasoning_options",
            "\"signature\"",
            "\"redactedData\"",
        ] {
            assert!(
                replay_source.contains(required),
                "Anthropic thinking replay bridge must remain an explicit reasoning metadata replay helper"
            );
        }

        assert!(
            !replay_source.contains("provider_options_map"),
            "Anthropic thinking replay bridge must not read generic request provider option maps"
        );
    }

    #[test]
    fn assistant_message_with_thinking_metadata_uses_typed_metadata_fields() {
        let mut response = ChatResponse::new(MessageContent::MultiModal(vec![
            ContentPart::text("visible"),
            ContentPart::Reasoning {
                text: "internal".to_string(),
                provider_options: crate::types::ProviderOptionsMap::default(),
                provider_metadata: Some(std::collections::HashMap::from([(
                    "anthropic".to_string(),
                    serde_json::json!({
                        "signature": "sig-1"
                    }),
                )])),
            },
        ]));
        response.provider_metadata = Some(std::collections::HashMap::from([(
            "anthropic".to_string(),
            serde_json::json!({
                "redacted_thinking_data": "redacted-blob"
            }),
        )]));

        let msg = assistant_message_with_thinking_metadata(&response);
        let MessageContent::MultiModal(parts) = msg.content else {
            panic!("expected multimodal content");
        };
        let reasoning = parts
            .iter()
            .find(|part| matches!(part, ContentPart::Reasoning { .. }))
            .expect("reasoning part");
        let anthropic_options = reasoning
            .provider_options()
            .and_then(|provider_options| provider_options.get_object("anthropic"))
            .expect("anthropic provider options");

        assert_eq!(
            anthropic_options.get("signature"),
            Some(&serde_json::json!("sig-1"))
        );
        assert_eq!(
            anthropic_options.get("redactedData"),
            Some(&serde_json::json!("redacted-blob"))
        );
    }

    #[test]
    fn assistant_message_with_thinking_metadata_adds_placeholder_reasoning_for_redacted_data() {
        let mut response = ChatResponse::new(MessageContent::Text("visible".to_string()));
        response.provider_metadata = Some(std::collections::HashMap::from([(
            "anthropic".to_string(),
            serde_json::json!({
                "redacted_thinking_data": "redacted-blob"
            }),
        )]));

        let msg = assistant_message_with_thinking_metadata(&response);
        let MessageContent::MultiModal(parts) = msg.content else {
            panic!("expected multimodal content");
        };
        let reasoning = parts
            .iter()
            .find(|part| matches!(part, ContentPart::Reasoning { .. }))
            .expect("reasoning part");
        let anthropic_options = reasoning
            .provider_options()
            .and_then(|provider_options| provider_options.get_object("anthropic"))
            .expect("anthropic provider options");

        assert_eq!(
            anthropic_options.get("redactedData"),
            Some(&serde_json::json!("redacted-blob"))
        );
    }
}
