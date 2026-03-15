//! MiniMaxi thinking helpers (extension API).

use crate::error::LlmError;
use crate::provider_options::{MinimaxiOptions, MinimaxiThinkingModeConfig};
use crate::providers::minimaxi::ext::MinimaxiChatRequestExt;
use crate::traits::ChatCapability;
use crate::types::{ChatRequest, ChatResponse};

/// Execute a chat request with MiniMaxi thinking mode enabled (explicit extension API).
pub async fn chat_with_thinking<C>(
    client: &C,
    mut request: ChatRequest,
    config: MinimaxiThinkingModeConfig,
) -> Result<ChatResponse, LlmError>
where
    C: ChatCapability + ?Sized,
{
    request = request.with_minimaxi_options(MinimaxiOptions::new().with_thinking_mode(config));
    client.chat_request(request).await
}
