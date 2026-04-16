//! Completion capability trait.

use crate::error::LlmError;
use crate::streaming::{ChatStream, ChatStreamHandle};
use crate::types::{CompletionRequest, CompletionResponse};
use async_trait::async_trait;

#[async_trait]
pub trait CompletionCapability: Send + Sync {
    /// Execute a non-streaming completion request.
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError>;

    /// Execute a streaming completion request.
    ///
    /// Completion-family streaming reuses the shared language-model runtime stream carrier
    /// (`ChatStreamEvent`) so downstream stream consumers can continue to work on one semantic
    /// stream model.
    async fn complete_stream(&self, request: CompletionRequest) -> Result<ChatStream, LlmError>;

    /// Execute a streaming completion request with a first-class cancellation handle.
    async fn complete_stream_with_cancel(
        &self,
        request: CompletionRequest,
    ) -> Result<ChatStreamHandle, LlmError> {
        let stream = self.complete_stream(request).await?;
        let (cancellable, cancel) = crate::utils::cancel::make_cancellable_stream(stream);
        Ok(ChatStreamHandle {
            stream: cancellable,
            cancel,
        })
    }
}
