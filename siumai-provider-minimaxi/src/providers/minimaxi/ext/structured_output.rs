//! MiniMaxi structured output helpers (extension API).

use crate::error::LlmError;
use crate::provider_options::MinimaxiOptions;
use crate::providers::minimaxi::ext::MinimaxiChatRequestExt;
use crate::traits::ChatCapability;
use crate::types::{ChatRequest, ChatResponse};

/// Execute a chat request requesting a JSON object response (explicit extension API).
pub async fn chat_with_json_object<C>(
    client: &C,
    mut request: ChatRequest,
) -> Result<ChatResponse, LlmError>
where
    C: ChatCapability + ?Sized,
{
    request = request.with_minimaxi_options(MinimaxiOptions::new().with_json_object());
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
    fn minimaxi_structured_output_extension_source_does_not_read_response_metadata() {
        let source = include_str!("structured_output.rs");
        let request_source =
            source_section(source, "pub async fn chat_with_json_object", "#[cfg(test)]");

        for disallowed in ["provider_metadata", "ProviderMetadata", "ContentPart::"] {
            assert!(
                !request_source.contains(disallowed),
                "MiniMaxi structured output helper must stay request-only"
            );
        }
    }
}
