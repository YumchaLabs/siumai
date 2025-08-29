//! Gemini streaming implementation using eventsource-stream
//!
//! This module provides Gemini streaming functionality using the
//! eventsource-stream infrastructure for JSON streaming.

use crate::error::LlmError;
use crate::providers::gemini::types::GeminiConfig;
use crate::stream::{ChatStream, ChatStreamEvent};
use crate::types::{ChatResponse, FinishReason, MessageContent, ResponseMetadata};
use crate::utils::streaming::{SseEventConverter, StreamFactory};
use serde::Deserialize;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Gemini stream response structure
#[derive(Debug, Clone, Deserialize)]
struct GeminiStreamResponse {
    candidates: Option<Vec<GeminiCandidate>>,
    #[serde(rename = "usageMetadata")]
    #[allow(dead_code)]
    usage_metadata: Option<GeminiUsageMetadata>,
}

/// Gemini candidate structure
#[derive(Debug, Clone, Deserialize)]
struct GeminiCandidate {
    content: Option<GeminiContent>,
    #[serde(rename = "finishReason")]
    finish_reason: Option<String>,
}

/// Gemini content structure
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct GeminiContent {
    parts: Option<Vec<GeminiPart>>,
    role: Option<String>,
}

/// Gemini part structure
#[derive(Debug, Clone, Deserialize)]
struct GeminiPart {
    text: Option<String>,
    /// Optional. Whether this is a thought summary (for thinking models)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[allow(dead_code)]
    thought: Option<bool>,
}

/// Gemini usage metadata
#[derive(Debug, Clone, Deserialize)]
struct GeminiUsageMetadata {
    #[serde(rename = "promptTokenCount")]
    #[allow(dead_code)]
    prompt_token_count: Option<u32>,
    #[serde(rename = "candidatesTokenCount")]
    #[allow(dead_code)]
    candidates_token_count: Option<u32>,
    #[serde(rename = "totalTokenCount")]
    #[allow(dead_code)]
    total_token_count: Option<u32>,
    /// Number of tokens used for thinking (only for thinking models)
    #[serde(rename = "thoughtsTokenCount")]
    #[allow(dead_code)]
    thoughts_token_count: Option<u32>,
}

/// Gemini event converter
#[derive(Clone)]
pub struct GeminiEventConverter {
    config: GeminiConfig,
    /// Track if StreamStart has been emitted
    stream_started: Arc<Mutex<bool>>,
}

impl GeminiEventConverter {
    pub fn new(config: GeminiConfig) -> Self {
        Self {
            config,
            stream_started: Arc::new(Mutex::new(false)),
        }
    }

    /// Convert Gemini stream response to multiple ChatStreamEvents
    async fn convert_gemini_response_async(
        &self,
        response: GeminiStreamResponse,
    ) -> Vec<ChatStreamEvent> {
        use crate::utils::streaming::EventBuilder;

        let mut builder = EventBuilder::new();

        // Check if we need to emit StreamStart first
        if self.needs_stream_start().await {
            builder = builder.add_stream_start(self.create_stream_start_metadata());
        }

        // Process content - NO MORE CONTENT LOSS!
        if let Some(content) = self.extract_content(&response) {
            builder = builder.add_content_delta(content, None);
        }

        // Process thinking content (if supported)
        if let Some(thinking) = self.extract_thinking(&response) {
            builder = builder.add_thinking_delta(thinking);
        }

        // Handle completion/finish reason
        if let Some(end_response) = self.extract_completion(&response) {
            builder = builder.add_stream_end(end_response);
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

    /// Extract content from Gemini response
    fn extract_content(&self, response: &GeminiStreamResponse) -> Option<String> {
        response
            .candidates
            .as_ref()?
            .first()?
            .content
            .as_ref()?
            .parts
            .as_ref()?
            .first()?
            .text
            .as_ref()
            .filter(|text| !text.is_empty())
            .cloned()
    }

    /// Extract thinking content from Gemini response
    fn extract_thinking(&self, _response: &GeminiStreamResponse) -> Option<String> {
        // Gemini may have thinking in different fields - implement based on actual API
        // For now, return None as Gemini thinking support is still evolving
        None
    }

    /// Extract completion information
    fn extract_completion(&self, response: &GeminiStreamResponse) -> Option<ChatResponse> {
        let candidate = response.candidates.as_ref()?.first()?;

        if let Some(finish_reason) = &candidate.finish_reason {
            let finish_reason = match finish_reason.as_str() {
                "STOP" => FinishReason::Stop,
                "MAX_TOKENS" => FinishReason::Length,
                "SAFETY" => FinishReason::ContentFilter,
                "RECITATION" => FinishReason::ContentFilter,
                _ => FinishReason::Stop,
            };

            let response = ChatResponse {
                id: None,
                model: None,
                content: MessageContent::Text("".to_string()),
                usage: None,
                finish_reason: Some(finish_reason),
                tool_calls: None,
                thinking: None,
                metadata: std::collections::HashMap::new(),
            };

            Some(response)
        } else {
            None
        }
    }

    /// Create StreamStart metadata
    fn create_stream_start_metadata(&self) -> ResponseMetadata {
        ResponseMetadata {
            id: None,                               // Gemini doesn't provide ID in stream events
            model: Some(self.config.model.clone()), // Use model from config
            created: Some(chrono::Utc::now()),
            provider: "gemini".to_string(),
            request_id: None,
        }
    }
}

impl SseEventConverter for GeminiEventConverter {
    fn convert_event(
        &self,
        event: eventsource_stream::Event,
    ) -> Pin<Box<dyn Future<Output = Vec<Result<ChatStreamEvent, LlmError>>> + Send + Sync + '_>>
    {
        Box::pin(async move {
            // Skip empty events
            if event.data.trim().is_empty() {
                return vec![];
            }

            // Parse the JSON data from the SSE event
            match serde_json::from_str::<GeminiStreamResponse>(&event.data) {
                Ok(gemini_response) => self
                    .convert_gemini_response_async(gemini_response)
                    .await
                    .into_iter()
                    .map(Ok)
                    .collect(),
                Err(e) => {
                    vec![Err(LlmError::ParseError(format!(
                        "Failed to parse Gemini SSE JSON: {e}"
                    )))]
                }
            }
        })
    }
}

/// Gemini streaming client
#[derive(Debug, Clone)]
pub struct GeminiStreaming {
    config: GeminiConfig,
    http_client: reqwest::Client,
}

impl GeminiStreaming {
    /// Create a new Gemini streaming client
    pub fn new(http_client: reqwest::Client) -> Self {
        Self {
            config: GeminiConfig::default(),
            http_client,
        }
    }

    /// Create a chat stream from URL, API key, and request
    pub async fn create_chat_stream(
        self,
        url: String,
        api_key: String,
        request: crate::providers::gemini::types::GenerateContentRequest,
    ) -> Result<ChatStream, LlmError> {
        // Make the HTTP request
        let response = self
            .http_client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("x-goog-api-key", &api_key)
            .json(&request)
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
                message: format!("Gemini API error {status}: {error_text}"),
                details: None,
            });
        }

        // Create the stream using SSE infrastructure (Gemini uses SSE format)
        let mut config = self.config;
        config.api_key = api_key.clone();
        let converter = GeminiEventConverter::new(config);
        StreamFactory::create_eventsource_stream(
            self.http_client
                .post(&url)
                .header("Content-Type", "application/json")
                .header("x-goog-api-key", &api_key)
                .json(&request),
            converter,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::gemini::types::GeminiConfig;

    fn create_test_config() -> GeminiConfig {
        GeminiConfig {
            api_key: "test-key".to_string(),
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_gemini_streaming_conversion() {
        let config = create_test_config();
        let converter = GeminiEventConverter::new(config);

        // Test content delta conversion
        let json_data = r#"{"candidates":[{"content":{"parts":[{"text":"Hello"}]}}]}"#;
        let event = eventsource_stream::Event {
            event: "".to_string(),
            data: json_data.to_string(),
            id: "".to_string(),
            retry: None,
        };

        let result = converter.convert_event(event).await;
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
    async fn test_gemini_finish_reason() {
        let config = create_test_config();
        let converter = GeminiEventConverter::new(config);

        // Test finish reason conversion
        let json_data = r#"{"candidates":[{"finishReason":"STOP"}]}"#;
        let event = eventsource_stream::Event {
            event: "".to_string(),
            data: json_data.to_string(),
            id: "".to_string(),
            retry: None,
        };

        let result = converter.convert_event(event).await;
        assert!(!result.is_empty());

        // In the new architecture, first event might be StreamStart, look for StreamEnd
        let stream_end_event = result
            .iter()
            .find(|event| matches!(event, Ok(ChatStreamEvent::StreamEnd { .. })));

        if let Some(Ok(ChatStreamEvent::StreamEnd { response })) = stream_end_event {
            assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        } else {
            panic!("Expected StreamEnd event in results: {:?}", result);
        }
    }
}
