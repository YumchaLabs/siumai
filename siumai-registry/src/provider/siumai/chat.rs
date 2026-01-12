use super::Siumai;
use crate::error::LlmError;
use crate::streaming::ChatStream;
use crate::traits::ChatCapability;
use crate::types::*;

#[async_trait::async_trait]
impl ChatCapability for Siumai {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        if let Some(opts) = &self.retry_options {
            crate::retry_api::retry_with(
                || {
                    let m = messages.clone();
                    let t = tools.clone();
                    async move { self.client.chat_with_tools(m, t).await }
                },
                opts.clone(),
            )
            .await
        } else {
            self.client.chat_with_tools(messages, tools).await
        }
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        if let Some(opts) = &self.retry_options {
            crate::retry_api::retry_with(
                || {
                    let m = messages.clone();
                    let t = tools.clone();
                    async move { self.client.chat_stream(m, t).await }
                },
                opts.clone(),
            )
            .await
        } else {
            self.client.chat_stream(messages, tools).await
        }
    }
}
