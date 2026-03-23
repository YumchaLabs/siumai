use super::AnthropicClient;
use crate::error::LlmError;
use crate::streaming::ChatStream;
use crate::traits::ChatCapability;
use crate::types::{ChatMessage, ChatRequest, ChatResponse, Tool};
use async_trait::async_trait;

impl AnthropicClient {
    fn prepare_chat_request(
        &self,
        request: ChatRequest,
        stream: bool,
    ) -> Result<ChatRequest, LlmError> {
        let request = crate::utils::chat_request::normalize_chat_request(
            request,
            crate::utils::chat_request::ChatRequestDefaults::new(&self.common_params),
            stream,
        );
        if request.common_params.model.trim().is_empty() {
            return Err(LlmError::InvalidParameter(
                "Model must be specified".to_string(),
            ));
        }
        Ok(request)
    }

    async fn chat_request_via_spec(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        use crate::execution::executors::chat::ChatExecutor;
        let request = self.prepare_chat_request(request, false)?;
        let exec = self.build_chat_executor(&request);
        ChatExecutor::execute(&*exec, request).await
    }

    async fn chat_stream_request_via_spec(
        &self,
        request: ChatRequest,
    ) -> Result<ChatStream, LlmError> {
        use crate::execution::executors::chat::ChatExecutor;
        let request = self.prepare_chat_request(request, true)?;
        let exec = self.build_chat_executor(&request);
        ChatExecutor::execute_stream(&*exec, request).await
    }

    async fn chat_with_tools_inner(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        let mut builder = ChatRequest::builder()
            .messages(messages)
            .common_params(self.common_params.clone());
        if let Some(ts) = tools {
            builder = builder.tools(ts);
        }
        let request = builder.build();
        self.chat_request_via_spec(request).await
    }
}

#[async_trait]
impl ChatCapability for AnthropicClient {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        self.chat_with_tools_inner(messages, tools).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        let mut builder = ChatRequest::builder()
            .messages(messages)
            .common_params(self.common_params.clone())
            .stream(true);
        if let Some(ts) = tools {
            builder = builder.tools(ts);
        }
        let request = builder.build();
        self.chat_stream_request_via_spec(request).await
    }

    async fn chat_request(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        self.chat_request_via_spec(request).await
    }

    async fn chat_stream_request(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        self.chat_stream_request_via_spec(request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepare_chat_request_for_stream_fills_missing_common_params_defaults() {
        let cfg = crate::providers::anthropic::AnthropicConfig::new("test-key")
            .with_model("claude-default")
            .with_temperature(0.2)
            .with_max_tokens(256)
            .with_top_p(0.9);
        let client =
            crate::providers::anthropic::AnthropicClient::from_config(cfg).expect("from_config ok");

        let request = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .build();

        let prepared = client
            .prepare_chat_request(request, true)
            .expect("prepare stream request");

        assert!(prepared.stream);
        assert_eq!(prepared.common_params.model, "claude-default");
        assert_eq!(prepared.common_params.temperature, Some(0.2));
        assert_eq!(prepared.common_params.max_tokens, Some(256));
        assert_eq!(prepared.common_params.top_p, Some(0.9));
    }

    #[test]
    fn prepare_chat_request_for_non_stream_preserves_explicit_common_params() {
        let cfg = crate::providers::anthropic::AnthropicConfig::new("test-key")
            .with_model("claude-default")
            .with_temperature(0.2)
            .with_max_tokens(256)
            .with_top_p(0.9);
        let client =
            crate::providers::anthropic::AnthropicClient::from_config(cfg).expect("from_config ok");

        let request = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .common_params(crate::types::CommonParams {
                model: "claude-explicit".to_string(),
                temperature: Some(0.7),
                max_tokens: Some(64),
                top_p: Some(0.5),
                ..Default::default()
            })
            .build();

        let prepared = client
            .prepare_chat_request(request, false)
            .expect("prepare non-stream request");

        assert!(!prepared.stream);
        assert_eq!(prepared.common_params.model, "claude-explicit");
        assert_eq!(prepared.common_params.temperature, Some(0.7));
        assert_eq!(prepared.common_params.max_tokens, Some(64));
        assert_eq!(prepared.common_params.top_p, Some(0.5));
    }
}
