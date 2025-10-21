//! Gemini streaming implementation using eventsource-stream
//!
//! This module provides Gemini streaming functionality using the
//! eventsource-stream infrastructure for JSON streaming.

use crate::error::LlmError;
use crate::providers::gemini::types::GeminiConfig;
use crate::streaming::{ChatStream, ChatStreamEvent};
use crate::streaming::{SseEventConverter, StreamFactory};
use crate::types::Usage;
use crate::types::{ChatResponse, FinishReason, MessageContent, ResponseMetadata};
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
struct GeminiContent {
    parts: Option<Vec<GeminiPart>>,
    #[allow(dead_code)]
    // Role appears in some responses but is not required by our unified event model
    role: Option<String>,
}

/// Gemini part structure
#[derive(Debug, Clone, Deserialize)]
struct GeminiPart {
    text: Option<String>,
    /// Optional. Whether this is a thought summary (for thinking models)
    #[serde(skip_serializing_if = "Option::is_none")]
    thought: Option<bool>,
}

/// Gemini usage metadata
#[derive(Debug, Clone, Deserialize)]
struct GeminiUsageMetadata {
    #[serde(rename = "promptTokenCount")]
    prompt_token_count: Option<u32>,
    #[serde(rename = "candidatesTokenCount")]
    candidates_token_count: Option<u32>,
    #[serde(rename = "totalTokenCount")]
    total_token_count: Option<u32>,
    /// Number of tokens used for thinking (only for thinking models)
    #[serde(rename = "thoughtsTokenCount")]
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
        use crate::streaming::EventBuilder;

        let mut builder = EventBuilder::new();

        // Check if we need to emit StreamStart first
        if self.needs_stream_start().await {
            builder = builder.add_stream_start(self.create_stream_start_metadata());
        }

        // Process content - support multiple candidates/parts per chunk
        let texts = self.extract_all_texts(&response);
        if !texts.is_empty() {
            for t in texts {
                builder = builder.add_content_delta(t, None);
            }
        }

        // Process thinking content (if supported)
        if let Some(thinking) = self.extract_thinking(&response) {
            builder = builder.add_thinking_delta(thinking);
        }

        // Process usage update if available
        if let Some(usage) = self.extract_usage(&response) {
            builder = builder.add_usage_update(usage);
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
    #[allow(dead_code)]
    fn extract_content(&self, response: &GeminiStreamResponse) -> Option<String> {
        let candidates = response.candidates.as_ref()?;
        for cand in candidates {
            if let Some(content) = &cand.content
                && let Some(parts) = &content.parts
            {
                for part in parts {
                    if let Some(text) = &part.text
                        && !text.is_empty()
                    {
                        return Some(text.clone());
                    }
                }
            }
        }
        None
    }

    /// Extract all non-empty texts across candidates/parts (for multi-candidate streams)
    fn extract_all_texts(&self, response: &GeminiStreamResponse) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(candidates) = &response.candidates {
            for cand in candidates {
                if let Some(content) = &cand.content
                    && let Some(parts) = &content.parts
                {
                    for part in parts {
                        if let Some(text) = &part.text
                            && !text.is_empty()
                        {
                            out.push(text.clone());
                        }
                    }
                }
            }
        }
        out
    }

    /// Extract thinking content from Gemini response
    fn extract_thinking(&self, response: &GeminiStreamResponse) -> Option<String> {
        // Extract thinking content from parts marked with thought: true
        response
            .candidates
            .as_ref()?
            .first()?
            .content
            .as_ref()?
            .parts
            .as_ref()?
            .iter()
            .find_map(|part| {
                if let Some(text) = &part.text {
                    if part.thought.unwrap_or(false) {
                        Some(text.clone())
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
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

    /// Extract usage information
    fn extract_usage(&self, response: &GeminiStreamResponse) -> Option<Usage> {
        if let Some(meta) = &response.usage_metadata {
            return Some(Usage {
                prompt_tokens: meta.prompt_token_count.unwrap_or(0),
                completion_tokens: meta.candidates_token_count.unwrap_or(0),
                total_tokens: meta.total_token_count.unwrap_or(
                    meta.prompt_token_count.unwrap_or(0) + meta.candidates_token_count.unwrap_or(0),
                ),
                cached_tokens: None,
                reasoning_tokens: meta.thoughts_token_count,
            });
        }
        None
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
            // Skip done marker or empty events
            if event.data.trim() == "[DONE]" || event.data.trim().is_empty() {
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
        request: serde_json::Value,
    ) -> Result<ChatStream, LlmError> {
        // Make the HTTP request
        let mut rb = self
            .http_client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&request);
        if !api_key.is_empty() {
            rb = rb.header("x-goog-api-key", &api_key);
        }
        let response = rb
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

        // Determine compression behavior for streaming
        let disable_compression = self.config.http_config.stream_disable_compression;

        // Create the stream using SSE infrastructure (Gemini uses SSE format)
        use secrecy::SecretString;
        let mut config = self.config;
        config.api_key = SecretString::from(api_key.clone());
        let converter = GeminiEventConverter::new(config);
        // Build closure for one-shot 401 retry with header rebuild
        let http = self.http_client.clone();
        let url_for_retry = url.clone();
        let body_for_retry = request.clone();
        let key_for_retry = api_key.clone();
        let build_request = move || {
            let mut headers = reqwest::header::HeaderMap::new();
            headers.insert(
                reqwest::header::CONTENT_TYPE,
                reqwest::header::HeaderValue::from_static("application/json"),
            );
            if !key_for_retry.is_empty() {
                headers.insert(
                    reqwest::header::HeaderName::from_static("x-goog-api-key"),
                    reqwest::header::HeaderValue::from_str(&key_for_retry).map_err(|e| {
                        LlmError::ConfigurationError(format!("Invalid api key: {e}"))
                    })?,
                );
            }
            crate::utils::http_headers::inject_tracing_headers(&mut headers);
            let mut builder = http
                .post(&url_for_retry)
                .headers(headers)
                // SSE expectations: explicit Accept + disable compression
                .header(reqwest::header::ACCEPT, "text/event-stream")
                .header(reqwest::header::CACHE_CONTROL, "no-cache")
                .header(reqwest::header::CONNECTION, "keep-alive")
                .json(&body_for_retry);
            if disable_compression {
                builder = builder.header(reqwest::header::ACCEPT_ENCODING, "identity");
            }
            Ok(builder)
        };
        StreamFactory::create_eventsource_stream_with_retry("gemini", build_request, converter)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::gemini::types::GeminiConfig;

    fn create_test_config() -> GeminiConfig {
        use secrecy::SecretString;
        GeminiConfig {
            api_key: SecretString::from("test-key".to_string()),
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
