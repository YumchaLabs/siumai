//! Ollama streaming implementation using eventsource-stream
//!
//! This module provides Ollama streaming functionality using the
//! eventsource-stream infrastructure for JSON streaming.

use crate::error::LlmError;
use crate::streaming::{ChatStream, ChatStreamEvent};
use crate::streaming::{JsonEventConverter, StreamFactory};
use crate::types::{ResponseMetadata, Usage};
use serde::Deserialize;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Ollama stream response structure
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
// Ollama may include fields we don't currently surface; keep them to remain parse-compatible
struct OllamaStreamResponse {
    model: Option<String>,
    message: Option<OllamaMessage>,
    done: Option<bool>,
    total_duration: Option<u64>,
    load_duration: Option<u64>,
    prompt_eval_count: Option<u32>,
    eval_count: Option<u32>,
}

/// Ollama message structure
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
// Message fields are retained for serde compatibility; converter extracts only the parts we emit
struct OllamaMessage {
    role: Option<String>,
    content: Option<String>,
    tool_calls: Option<Vec<super::types::OllamaToolCall>>,
    thinking: Option<String>,
}

/// Ollama event converter
#[derive(Clone)]
pub struct OllamaEventConverter {
    /// Track if StreamStart has been emitted
    stream_started: Arc<Mutex<bool>>,
}

impl Default for OllamaEventConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl OllamaEventConverter {
    pub fn new() -> Self {
        Self {
            stream_started: Arc::new(Mutex::new(false)),
        }
    }

    /// Convert Ollama stream response to multiple ChatStreamEvents
    async fn convert_ollama_response_async(
        &self,
        response: OllamaStreamResponse,
    ) -> Vec<ChatStreamEvent> {
        use crate::streaming::EventBuilder;

        let mut builder = EventBuilder::new();

        // Check if we need to emit StreamStart first
        if self.needs_stream_start().await {
            let metadata = self.create_stream_start_metadata(&response);
            builder = builder.add_stream_start(metadata);
        }

        // Process content - NO MORE CONTENT LOSS!
        if let Some(content) = self.extract_content(&response) {
            builder = builder.add_content_delta(content, None);
        }

        // Process usage updates
        if let Some(usage) = self.extract_usage(&response) {
            builder = builder.add_usage_update(usage);
        }

        builder.build()
    }

    /// Check if StreamStart event needs to be emitted
    async fn needs_stream_start(&self) -> bool {
        let mut started = self.stream_started.lock().await;
        if !*started {
            *started = true;
            true
        } else {
            false
        }
    }

    /// Extract content from Ollama response
    fn extract_content(&self, response: &OllamaStreamResponse) -> Option<String> {
        response
            .message
            .as_ref()?
            .content
            .as_ref()
            .filter(|content| !content.is_empty())
            .cloned()
    }

    /// Extract usage information
    fn extract_usage(&self, response: &OllamaStreamResponse) -> Option<Usage> {
        if response.done == Some(true)
            && let (Some(prompt_tokens), Some(completion_tokens)) =
                (response.prompt_eval_count, response.eval_count)
        {
            return Some(Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
                cached_tokens: None,
                reasoning_tokens: None,
            });
        }
        None
    }

    /// Create StreamStart metadata from Ollama response
    fn create_stream_start_metadata(&self, response: &OllamaStreamResponse) -> ResponseMetadata {
        ResponseMetadata {
            id: None, // Ollama doesn't provide ID in stream events
            model: response.model.clone(),
            created: Some(chrono::Utc::now()),
            provider: "ollama".to_string(),
            request_id: None,
        }
    }
}

impl JsonEventConverter for OllamaEventConverter {
    fn convert_json<'a>(
        &'a self,
        json_data: &'a str,
    ) -> Pin<Box<dyn Future<Output = Vec<Result<ChatStreamEvent, LlmError>>> + Send + Sync + 'a>>
    {
        Box::pin(async move {
            match serde_json::from_str::<OllamaStreamResponse>(json_data) {
                Ok(ollama_response) => self
                    .convert_ollama_response_async(ollama_response)
                    .await
                    .into_iter()
                    .map(Ok)
                    .collect(),
                Err(e) => {
                    vec![Err(LlmError::ParseError(format!(
                        "Failed to parse Ollama JSON: {e}"
                    )))]
                }
            }
        })
    }
}

/// Ollama streaming client
#[derive(Clone)]
pub struct OllamaStreaming {
    http_client: reqwest::Client,
}

impl OllamaStreaming {
    /// Create a new Ollama streaming client
    pub fn new(http_client: reqwest::Client) -> Self {
        Self { http_client }
    }

    /// Create a chat stream from URL, headers, and body
    pub async fn create_chat_stream(
        self,
        url: String,
        headers: reqwest::header::HeaderMap,
        body: crate::providers::ollama::types::OllamaChatRequest,
    ) -> Result<ChatStream, LlmError> {
        // Make the HTTP request
        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("Request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!("Ollama API error {status}: {error_text}"),
                details: None,
            });
        }

        // Create the stream using our new infrastructure
        let converter = OllamaEventConverter::new();
        StreamFactory::create_json_stream(response, converter).await
    }

    /// Create a completion stream from URL, headers, and body
    pub async fn create_completion_stream(
        self,
        url: String,
        headers: reqwest::header::HeaderMap,
        body: crate::providers::ollama::types::OllamaGenerateRequest,
    ) -> Result<ChatStream, LlmError> {
        // Make the HTTP request
        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("Request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!("Ollama API error {status}: {error_text}"),
                details: None,
            });
        }

        // Create the stream using our new infrastructure
        let converter = OllamaEventConverter::new();
        StreamFactory::create_json_stream(response, converter).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ollama_streaming_conversion() {
        let converter = OllamaEventConverter::new();

        // Test content delta conversion
        let json_data =
            r#"{"model":"llama2","message":{"role":"assistant","content":"Hello"},"done":false}"#;

        let result = converter.convert_json(json_data).await;
        assert!(!result.is_empty());

        // In the new architecture, we might get StreamStart + ContentDelta
        let content_event = result
            .iter()
            .find(|event| matches!(event, Ok(ChatStreamEvent::ContentDelta { .. })));

        if let Some(Ok(ChatStreamEvent::ContentDelta { delta, .. })) = content_event {
            assert_eq!(delta, "Hello");
        } else {
            panic!("Expected ContentDelta event in results: {:?}", result);
        }
    }

    #[tokio::test]
    async fn test_ollama_stream_end() {
        let converter = OllamaEventConverter::new();

        // Test stream end conversion
        let json_data = r#"{"model":"llama2","done":true,"prompt_eval_count":10,"eval_count":20}"#;

        let result = converter.convert_json(json_data).await;
        assert!(!result.is_empty());

        // In the new architecture, we might get StreamStart + UsageUpdate
        let usage_event = result
            .iter()
            .find(|event| matches!(event, Ok(ChatStreamEvent::UsageUpdate { .. })));

        if let Some(Ok(ChatStreamEvent::UsageUpdate { usage })) = usage_event {
            assert_eq!(usage.prompt_tokens, 10);
            assert_eq!(usage.completion_tokens, 20);
        } else {
            panic!("Expected UsageUpdate event in results: {:?}", result);
        }
    }
}
