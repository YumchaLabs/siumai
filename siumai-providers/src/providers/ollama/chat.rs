//! Ollama Chat Capability Implementation
//!
//! Implements the `ChatCapability` trait for Ollama using the /api/chat endpoint.

use async_trait::async_trait;

use crate::error::LlmError;
use crate::streaming::ChatStream;
use crate::traits::ChatCapability;
use crate::types::*;

use super::config::OllamaParams;
use super::types::*;
use super::utils::*;

/// Ollama Chat Capability Implementation
#[derive(Clone)]
pub struct OllamaChatCapability {
    pub base_url: String,
    pub http_client: reqwest::Client,
    pub http_config: HttpConfig,
    pub ollama_params: OllamaParams,
}

impl OllamaChatCapability {
    /// Creates a new Ollama chat capability
    pub const fn new(
        base_url: String,
        http_client: reqwest::Client,
        http_config: HttpConfig,
        ollama_params: OllamaParams,
    ) -> Self {
        Self {
            base_url,
            http_client,
            http_config,
            ollama_params,
        }
    }

    /// Build chat request body
    pub fn build_chat_request_body(
        &self,
        request: &ChatRequest,
    ) -> Result<OllamaChatRequest, LlmError> {
        build_chat_request(request, &self.ollama_params)
    }

    /// Parse chat response
    fn parse_chat_response(&self, response: OllamaChatResponse) -> ChatResponse {
        convert_chat_response(response)
    }
}

#[async_trait]
impl ChatCapability for OllamaChatCapability {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        // Create a default ChatRequest with empty common_params
        // This allows the capability to work independently
        let mut builder = ChatRequest::builder()
            .messages(messages)
            .common_params(CommonParams {
                model: "llama3.2".to_string(),
                ..Default::default()
            });
        if let Some(ts) = tools {
            builder = builder.tools(ts);
        }
        let request = builder.build();

        self.chat(request).await
    }

    async fn chat_stream(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        // This method is deprecated. The OllamaClient now uses HttpChatExecutor for streaming.
        // This implementation is kept for backward compatibility but should not be used directly.
        Err(LlmError::ConfigurationError(
            "OllamaChatCapability::chat_stream is deprecated. Use OllamaClient::chat_stream instead.".to_string()
        ))
    }
}

impl OllamaChatCapability {
    /// Chat implementation (internal)
    pub async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        use crate::core::ProviderSpec;

        let ctx = crate::core::ProviderContext::new(
            "ollama",
            self.base_url.clone(),
            None,
            self.http_config.headers.clone(),
        );
        let spec = std::sync::Arc::new(super::spec::OllamaSpec::new(self.ollama_params.clone()));
        let url = spec.chat_url(false, &request, &ctx);

        let body = self.build_chat_request_body(&request)?;
        let body_json = serde_json::to_value(&body)?;

        let config = crate::execution::executors::common::HttpExecutionConfig {
            provider_id: "ollama".to_string(),
            http_client: self.http_client.clone(),
            provider_spec: spec,
            provider_context: ctx,
            interceptors: Vec::new(),
            retry_options: None,
        };
        let res = crate::execution::executors::common::execute_json_request(
            &config,
            &url,
            crate::execution::executors::common::HttpBody::Json(body_json),
            None,
            false,
        )
        .await?;

        let ollama_response: OllamaChatResponse =
            serde_json::from_value(res.json).map_err(|e| {
                LlmError::ParseError(format!("Failed to parse Ollama chat response: {e}"))
            })?;
        Ok(self.parse_chat_response(ollama_response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CommonParams;

    #[test]
    fn test_build_chat_request_body() {
        let capability = OllamaChatCapability::new(
            "http://localhost:11434".to_string(),
            reqwest::Client::new(),
            HttpConfig::default(),
            OllamaParams::default(),
        );

        let common_params = CommonParams {
            model: "llama3.2".to_string(),
            temperature: Some(0.7),
            ..Default::default()
        };

        let request = ChatRequest::builder()
            .messages(vec![ChatMessage::user("Hello").build()])
            .common_params(common_params)
            .build();

        let body = capability.build_chat_request_body(&request).unwrap();
        assert_eq!(body.model, "llama3.2");
        assert_eq!(body.messages.len(), 1);
        assert_eq!(body.messages[0].content, "Hello");
        assert_eq!(body.stream, Some(false));
    }

    #[test]
    fn test_parse_chat_response() {
        let capability = OllamaChatCapability::new(
            "http://localhost:11434".to_string(),
            reqwest::Client::new(),
            HttpConfig::default(),
            OllamaParams::default(),
        );

        let ollama_response = OllamaChatResponse {
            model: "llama3.2".to_string(),
            created_at: "2023-01-01T00:00:00Z".to_string(),
            message: OllamaChatMessage {
                role: "assistant".to_string(),
                content: "Hello there!".to_string(),
                images: None,
                tool_calls: None,
                thinking: None,
            },
            done: true,
            done_reason: Some("stop".to_string()),
            total_duration: Some(1_000_000_000),
            load_duration: Some(100_000_000),
            prompt_eval_count: Some(10),
            prompt_eval_duration: Some(200_000_000),
            eval_count: Some(20),
            eval_duration: Some(700_000_000),
        };

        let response = capability.parse_chat_response(ollama_response);
        assert_eq!(response.model, Some("llama3.2".to_string()));
        assert_eq!(
            response.content,
            crate::types::MessageContent::Text("Hello there!".to_string())
        );
        assert_eq!(
            response.finish_reason,
            Some(crate::types::FinishReason::Stop)
        );
        assert!(response.usage.is_some());
        assert!(
            response
                .get_metadata("ollama", "total_duration_ms")
                .is_some()
        );
    }

    // Test for structured_output via provider_params has been removed
    // as this functionality is now handled via provider_options
}
