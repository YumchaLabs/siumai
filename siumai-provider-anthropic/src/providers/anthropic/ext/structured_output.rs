//! Anthropic structured output helpers (extension API).

use crate::error::LlmError;
use crate::traits::ChatCapability;
use crate::provider_options::anthropic::AnthropicOptions;
use crate::types::{ChatRequest, ChatResponse};

/// Execute a chat request requesting a JSON object response (explicit extension API).
pub async fn chat_with_json_object<C>(
    client: &C,
    mut request: ChatRequest,
) -> Result<ChatResponse, LlmError>
where
    C: ChatCapability + ?Sized,
{
    request = request.with_anthropic_options(AnthropicOptions::new().with_json_object());
    client.chat_request(request).await
}
