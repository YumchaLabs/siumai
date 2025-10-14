//! Completion capability trait

use crate::error::LlmError;
use crate::types::streaming::CompletionStream;
use crate::types::{CompletionRequest, CompletionResponse};
use async_trait::async_trait;

#[async_trait]
pub trait CompletionCapability: Send + Sync {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError>;

    async fn complete_stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionStream, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Streaming completion not supported by this provider".to_string(),
        ))
    }
}
