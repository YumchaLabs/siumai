//! MiniMax thinking helpers (extension API).

use crate::error::LlmError;
use crate::provider_options::{MinimaxOptions, MinimaxThinking};
use crate::providers::minimax::ext::MinimaxChatRequestExt;
use crate::traits::ChatCapability;
use crate::types::{ChatRequest, ChatResponse};

/// Execute a chat request with MiniMax thinking mode enabled (explicit extension API).
pub async fn chat_with_thinking<C>(
    client: &C,
    mut request: ChatRequest,
    thinking: MinimaxThinking,
) -> Result<ChatResponse, LlmError>
where
    C: ChatCapability + ?Sized,
{
    request = request.with_minimax_options(MinimaxOptions::new().with_thinking(thinking));
    client.chat_request(request).await
}

#[cfg(test)]
mod tests {
    fn source_section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
        let start_index = source.find(start).expect("section start marker");
        let end_index = source[start_index..]
            .find(end)
            .map(|offset| start_index + offset)
            .expect("section end marker");
        &source[start_index..end_index]
    }

    #[test]
    fn minimax_thinking_extension_source_does_not_read_response_metadata() {
        let source = include_str!("thinking.rs");
        let request_source =
            source_section(source, "pub async fn chat_with_thinking", "#[cfg(test)]");

        for disallowed in ["provider_metadata", "ProviderMetadata", "ContentPart::"] {
            assert!(
                !request_source.contains(disallowed),
                "MiniMax thinking helper must stay request-only"
            );
        }
    }
}
