//! Ollama Chat Capability Implementation
//!
//! Implements the `ChatCapability` trait for Ollama using the /api/chat endpoint.

use async_trait::async_trait;
use std::time::Instant;

use crate::error::LlmError;
use crate::observability::tracing::ProviderTracer;
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
        // Get model from request
        let model = request.common_params.model.clone();
        if model.is_empty() {
            return Err(LlmError::ConfigurationError(
                "Model is required".to_string(),
            ));
        }

        validate_model_name(&model)?;

        // Convert messages
        let messages: Vec<OllamaChatMessage> =
            request.messages.iter().map(convert_chat_message).collect();

        // Convert tools if present
        let tools = request
            .tools
            .as_ref()
            .map(|tools| tools.iter().filter_map(convert_tool).collect());

        // Build model options
        let options = build_model_options(
            request.common_params.temperature,
            request.common_params.max_tokens,
            request.common_params.top_p,
            None, // frequency_penalty not in CommonParams
            None, // presence_penalty not in CommonParams
            self.ollama_params.options.as_ref(),
        );

        // Build format from client params (provider_params is deprecated)
        let format = if let Some(format_str) = &self.ollama_params.format {
            if format_str == "json" {
                Some(serde_json::Value::String("json".to_string()))
            } else {
                match serde_json::from_str(format_str) {
                    Ok(schema) => Some(schema),
                    Err(_) => Some(serde_json::Value::String(format_str.clone())),
                }
            }
        } else {
            None
        };

        // Determine thinking behavior
        let think = self.ollama_params.think.or_else(|| {
            // Check if this is a thinking model based on model name
            if model.contains("deepseek-r1") || model.contains("qwen3") {
                Some(true) // Enable thinking by default for thinking models
            } else {
                None
            }
        });

        Ok(OllamaChatRequest {
            model,
            messages,
            tools,
            stream: Some(request.stream),
            format,
            options: if options.is_empty() {
                None
            } else {
                Some(options)
            },
            keep_alive: self.ollama_params.keep_alive.clone(),
            think,
        })
    }

    /// Parse chat response
    fn parse_chat_response(&self, response: OllamaChatResponse) -> ChatResponse {
        let message = convert_from_ollama_message(&response.message);

        // Calculate usage if metrics are available
        let usage = if response.prompt_eval_count.is_some() || response.eval_count.is_some() {
            let prompt = response.prompt_eval_count.unwrap_or(0);
            let completion = response.eval_count.unwrap_or(0);
            Some(
                Usage::builder()
                    .prompt_tokens(prompt)
                    .completion_tokens(completion)
                    .total_tokens(prompt + completion)
                    .build(),
            )
        } else {
            None
        };

        // Parse finish reason
        let finish_reason = response
            .done_reason
            .as_deref()
            .map(|reason| match reason {
                "stop" => FinishReason::Stop,
                "length" => FinishReason::Length,
                _ => FinishReason::Other(reason.to_string()),
            })
            .or({
                if response.done {
                    Some(FinishReason::Stop)
                } else {
                    None
                }
            });

        // Create metadata with performance metrics
        let mut ollama_metadata = std::collections::HashMap::new();
        if let Some(tokens_per_second) =
            calculate_tokens_per_second(response.eval_count, response.eval_duration)
        {
            ollama_metadata.insert(
                "tokens_per_second".to_string(),
                serde_json::Value::Number(
                    serde_json::Number::from_f64(tokens_per_second)
                        .unwrap_or_else(|| serde_json::Number::from(0)),
                ),
            );
        }
        if let Some(total_duration) = response.total_duration {
            ollama_metadata.insert(
                "total_duration_ms".to_string(),
                serde_json::Value::Number(serde_json::Number::from(total_duration / 1_000_000)),
            );
        }

        // Wrap in provider_metadata structure
        let provider_metadata = if !ollama_metadata.is_empty() {
            let mut meta = std::collections::HashMap::new();
            meta.insert("ollama".to_string(), ollama_metadata);
            Some(meta)
        } else {
            None
        };

        ChatResponse {
            id: Some(format!("ollama-{}", chrono::Utc::now().timestamp_millis())),
            content: message.content,
            model: Some(response.model),
            usage,
            finish_reason,
            audio: None,
            system_fingerprint: None,
            service_tier: None,
            warnings: None,
            provider_metadata,
        }
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
        let request = ChatRequest {
            messages,
            tools,
            common_params: CommonParams {
                model: "llama3.2".to_string(), // Default model
                ..Default::default()
            },
            ..Default::default()
        };

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
        let start_time = Instant::now();

        // Extract model name for tracing
        let model = request.common_params.model.clone();
        let tracer = ProviderTracer::new("ollama").with_model(model);

        let headers = build_headers(&self.http_config.headers)?;
        let body = self.build_chat_request_body(&request)?;
        let url = crate::utils::url::join_url(&self.base_url, "api/chat");

        tracer.trace_request_start("POST", &url);

        // Convert OllamaChatRequest to JSON for tracing
        let body_json = serde_json::to_value(&body)?;
        tracer.trace_request_details(&headers, &body_json);

        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            tracer.trace_request_error(status.as_u16(), &error_text, start_time);
            return Err(LlmError::HttpError(format!(
                "Chat request failed: {status} - {error_text}"
            )));
        }

        tracer.trace_response_success(status.as_u16(), start_time, response.headers());

        // Get response body as text first for logging
        let response_text = response.text().await?;
        tracer.trace_response_body(&response_text);

        let ollama_response: OllamaChatResponse = serde_json::from_str(&response_text)?;
        let chat_response = self.parse_chat_response(ollama_response);

        tracer.trace_request_complete(start_time, chat_response.content.all_text().len());

        Ok(chat_response)
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

        let request = ChatRequest {
            messages: vec![ChatMessage {
                role: crate::types::MessageRole::User,
                content: crate::types::MessageContent::Text("Hello".to_string()),
                metadata: crate::types::MessageMetadata::default(),
            }],
            tools: None,
            common_params,
            ..Default::default()
        };

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
