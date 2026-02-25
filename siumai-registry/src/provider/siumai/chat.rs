use super::Siumai;
use crate::error::LlmError;
use crate::streaming::{ChatStream, ChatStreamHandle};
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

    async fn chat_stream_with_cancel(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStreamHandle, LlmError> {
        let this = self.clone();
        Ok(
            crate::utils::cancel::make_cancellable_stream_handle_from_handle_future(async move {
                if let Some(opts) = this.retry_options.clone() {
                    crate::retry_api::retry_with(
                        || {
                            let m = messages.clone();
                            let t = tools.clone();
                            let client = std::sync::Arc::clone(&this.client);
                            async move { client.chat_stream_with_cancel(m, t).await }
                        },
                        opts,
                    )
                    .await
                } else {
                    this.client.chat_stream_with_cancel(messages, tools).await
                }
            }),
        )
    }

    async fn chat_stream_request_with_cancel(
        &self,
        request: ChatRequest,
    ) -> Result<ChatStreamHandle, LlmError> {
        let this = self.clone();
        Ok(
            crate::utils::cancel::make_cancellable_stream_handle_from_handle_future(async move {
                if let Some(opts) = this.retry_options.clone() {
                    crate::retry_api::retry_with(
                        || {
                            let req = request.clone();
                            let client = std::sync::Arc::clone(&this.client);
                            async move { client.chat_stream_request_with_cancel(req).await }
                        },
                        opts,
                    )
                    .await
                } else {
                    this.client.chat_stream_request_with_cancel(request).await
                }
            }),
        )
    }
}
