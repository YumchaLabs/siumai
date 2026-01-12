//! Anthropic extended thinking helpers (extension API).

use crate::error::LlmError;
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

/// Convert an Anthropic response into a replayable assistant message with thinking metadata.
///
/// Vercel AI SDK can replay Anthropic thinking blocks by attaching `signature` or `redactedData`
/// to reasoning parts. In Siumai we store these fields in `response.provider_metadata["anthropic"]`
/// and provide this helper to carry them into `ChatMessage.metadata.custom` for the next turn.
///
/// Notes:
/// - This is an Anthropic-specific escape hatch (not part of the unified surface).
/// - If `redacted_thinking_data` exists but the response has no reasoning parts, a placeholder
///   reasoning part is appended so that the request converter can emit `redacted_thinking`.
pub fn assistant_message_with_thinking_metadata(response: &ChatResponse) -> ChatMessage {
    let mut msg = response.to_assistant_message();

    let Some(provider_meta) = response.provider_metadata.as_ref() else {
        return msg;
    };
    let Some(anthropic) = provider_meta.get("anthropic") else {
        return msg;
    };

    if let Some(sig) = anthropic.get("thinking_signature").and_then(|v| v.as_str()) {
        msg.metadata.custom.insert(
            "anthropic_thinking_signature".to_string(),
            serde_json::json!(sig),
        );
    }

    if let Some(data) = anthropic
        .get("redacted_thinking_data")
        .and_then(|v| v.as_str())
    {
        msg.metadata.custom.insert(
            "anthropic_redacted_thinking_data".to_string(),
            serde_json::json!(data),
        );

        // Ensure the message contains a reasoning part so the converter can emit redacted_thinking.
        if !msg.has_reasoning() {
            msg.content = match msg.content {
                MessageContent::Text(t) if !t.is_empty() => MessageContent::MultiModal(vec![
                    ContentPart::text(t),
                    ContentPart::reasoning("redacted"),
                ]),
                MessageContent::MultiModal(mut parts) => {
                    parts.push(ContentPart::reasoning("redacted"));
                    MessageContent::MultiModal(parts)
                }
                _ => MessageContent::MultiModal(vec![ContentPart::reasoning("redacted")]),
            };
        }
    }

    msg
}
