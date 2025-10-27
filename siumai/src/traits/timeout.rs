//! Application-level timeout wrappers for chat

use crate::error::LlmError;
use crate::streaming::ChatStream;
use crate::traits::chat::ChatCapability;
use crate::types::{ChatMessage, ChatResponse, Tool};
use async_trait::async_trait;

#[async_trait]
pub trait TimeoutCapability: ChatCapability + Send + Sync {
    async fn chat_with_timeout(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
        timeout: std::time::Duration,
    ) -> Result<ChatResponse, LlmError> {
        tokio::time::timeout(timeout, self.chat_with_tools(messages, tools))
            .await
            .map_err(|_| {
                LlmError::TimeoutError(format!(
                    "Operation timed out after {:?} (including retries)",
                    timeout
                ))
            })?
    }

    async fn chat_stream_with_timeout(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
        timeout: std::time::Duration,
    ) -> Result<ChatStream, LlmError> {
        tokio::time::timeout(timeout, self.chat_stream(messages, tools))
            .await
            .map_err(|_| {
                LlmError::TimeoutError(format!(
                    "Stream initialization timed out after {:?}",
                    timeout
                ))
            })?
    }
}

impl<T> TimeoutCapability for T where T: ChatCapability + Send + Sync {}
