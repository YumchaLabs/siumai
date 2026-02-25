use super::OpenAiClient;
use crate::error::LlmError;
use crate::streaming::{ChatStream, ChatStreamHandle};
use crate::traits::ChatCapability;
use crate::types::{ChatMessage, ChatRequest, ChatResponse, Tool};
use async_trait::async_trait;

impl OpenAiClient {
    async fn chat_with_tools_inner(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        use crate::execution::executors::chat::ChatExecutor;

        let mut builder = ChatRequest::builder()
            .messages(messages)
            .common_params(self.common_params.clone());
        if let Some(ts) = tools {
            builder = builder.tools(ts);
        }
        let request = builder.build();

        let mut request = request;
        self.merge_default_provider_options_map(&mut request.provider_options_map);

        let exec = self.build_chat_executor(&request);
        ChatExecutor::execute(&*exec, request).await
    }

    // Stream chat via ProviderSpec (unified path)
    pub(super) async fn chat_stream_via_spec(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        use crate::execution::executors::chat::ChatExecutor;

        let mut builder = ChatRequest::builder()
            .messages(messages)
            .common_params(self.common_params.clone())
            .stream(true);
        if let Some(ts) = tools {
            builder = builder.tools(ts);
        }
        let request = builder.build();

        let mut request = request;
        self.merge_default_provider_options_map(&mut request.provider_options_map);

        let exec = self.build_chat_executor(&request);
        ChatExecutor::execute_stream(&*exec, request).await
    }

    // Execute chat (non-stream) via ProviderSpec with a fully-formed ChatRequest
    pub(super) async fn chat_request_via_spec(
        &self,
        request: ChatRequest,
    ) -> Result<ChatResponse, LlmError> {
        use crate::execution::executors::chat::ChatExecutor;
        let mut request = request;
        self.merge_default_provider_options_map(&mut request.provider_options_map);
        let exec = self.build_chat_executor(&request);
        ChatExecutor::execute(&*exec, request).await
    }

    // Execute chat (stream) via ProviderSpec with a fully-formed ChatRequest
    pub(super) async fn chat_stream_request_via_spec(
        &self,
        request: ChatRequest,
    ) -> Result<ChatStream, LlmError> {
        use crate::execution::executors::chat::ChatExecutor;
        let mut request = request;
        self.merge_default_provider_options_map(&mut request.provider_options_map);
        let exec = self.build_chat_executor(&request);
        ChatExecutor::execute_stream(&*exec, request).await
    }
}

#[async_trait]
impl ChatCapability for OpenAiClient {
    /// Chat with tools implementation
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        self.chat_with_tools_inner(messages, tools).await
    }

    /// Streaming chat with tools
    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        self.chat_stream_via_spec(messages, tools).await
    }

    async fn chat_stream_with_cancel(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStreamHandle, LlmError> {
        let this = self.clone();
        Ok(
            crate::utils::cancel::make_cancellable_stream_handle_from_future(async move {
                this.chat_stream_via_spec(messages, tools).await
            }),
        )
    }

    async fn chat_request(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        self.chat_request_via_spec(request).await
    }

    async fn chat_stream_request(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        self.chat_stream_request_via_spec(request).await
    }

    async fn chat_stream_request_with_cancel(
        &self,
        request: ChatRequest,
    ) -> Result<ChatStreamHandle, LlmError> {
        let this = self.clone();
        Ok(
            crate::utils::cancel::make_cancellable_stream_handle_from_future(async move {
                this.chat_stream_request_via_spec(request).await
            }),
        )
    }
}
