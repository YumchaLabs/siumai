//! Gemini streaming implementation using eventsource-stream
//!
//! Provides SSE event conversion for Gemini streaming responses.
//! The legacy GeminiStreaming client has been removed in favor of the unified HttpChatExecutor.

use crate::error::LlmError;
use crate::providers::gemini::types::GeminiConfig;
use crate::streaming::SseEventConverter;
use crate::streaming::{ChatStreamEvent, StreamStateTracker};
use crate::types::Usage;
use crate::types::{ChatResponse, FinishReason, MessageContent, ResponseMetadata};
use serde::Deserialize;
use std::future::Future;
use std::pin::Pin;

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
    state_tracker: StreamStateTracker,
}

impl GeminiEventConverter {
    pub fn new(config: GeminiConfig) -> Self {
        Self {
            config,
            state_tracker: StreamStateTracker::new(),
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
        if self.needs_stream_start() {
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
    fn needs_stream_start(&self) -> bool {
        self.state_tracker.needs_stream_start()
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

            // Mark that StreamEnd is being emitted
            self.state_tracker.mark_stream_ended();

            let response = ChatResponse {
                id: None,
                model: None,
                content: MessageContent::Text("".to_string()),
                usage: None,
                finish_reason: Some(finish_reason),
                audio: None,
                system_fingerprint: None,
                service_tier: None,
                warnings: None,
                provider_metadata: None,
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
                #[allow(deprecated)]
                cached_tokens: None,
                #[allow(deprecated)]
                reasoning_tokens: meta.thoughts_token_count,
                prompt_tokens_details: None,
                completion_tokens_details: meta.thoughts_token_count.map(|reasoning| {
                    crate::types::CompletionTokensDetails {
                        reasoning_tokens: Some(reasoning),
                        audio_tokens: None,
                        accepted_prediction_tokens: None,
                        rejected_prediction_tokens: None,
                    }
                }),
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
            match crate::streaming::parse_json_with_repair::<GeminiStreamResponse>(&event.data) {
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

    fn handle_stream_end(&self) -> Option<Result<ChatStreamEvent, LlmError>> {
        // Gemini normally emits finish_reason in the stream (handled in extract_completion).
        // If we reach here without seeing finish_reason, the model has not transmitted
        // a finish reason (e.g., connection lost, server error, client cancelled).
        // Always emit StreamEnd with Unknown reason so users can detect this.

        // Check if StreamEnd was already emitted
        if !self.state_tracker.needs_stream_end() {
            return None; // StreamEnd already emitted
        }

        let response = ChatResponse {
            id: None,
            model: None,
            content: MessageContent::Text("".to_string()),
            usage: None,
            finish_reason: Some(FinishReason::Unknown),
            audio: None,
            system_fingerprint: None,
            service_tier: None,
            warnings: None,
            provider_metadata: None,
        };
        Some(Ok(ChatStreamEvent::StreamEnd { response }))
    }
}

// Legacy GeminiStreaming client has been removed in favor of the unified HttpChatExecutor.
// The GeminiEventConverter is still used for SSE event conversion in tests.

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

    #[tokio::test]
    async fn test_empty_event_is_ignored() {
        let config = create_test_config();
        let converter = GeminiEventConverter::new(config);
        let event = eventsource_stream::Event {
            event: "".into(),
            data: "".into(),
            id: "".into(),
            retry: None,
        };
        let result = converter.convert_event(event).await;
        assert!(result.is_empty(), "Empty SSE event should be ignored");
    }

    #[tokio::test]
    async fn test_invalid_json_emits_error() {
        let config = create_test_config();
        let converter = GeminiEventConverter::new(config);
        let event = eventsource_stream::Event {
            event: "".into(),
            data: "{ not json".into(),
            id: "".into(),
            retry: None,
        };
        let result = converter.convert_event(event).await;
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0], Err(LlmError::ParseError(_))));
    }

    #[tokio::test]
    async fn test_stream_start_emitted_once_across_events() {
        let config = create_test_config();
        let converter = GeminiEventConverter::new(config);
        let mk_event = |text: &str| eventsource_stream::Event {
            event: "".into(),
            data: format!(
                "{{\"candidates\":[{{\"content\":{{\"parts\":[{{\"text\":\"{}\"}}]}}}}]}}",
                text
            ),
            id: "".into(),
            retry: None,
        };

        let r1 = converter.convert_event(mk_event("first")).await;
        let r2 = converter.convert_event(mk_event("second")).await;

        // First batch should contain a StreamStart
        assert!(
            r1.iter()
                .any(|e| matches!(e, Ok(ChatStreamEvent::StreamStart { .. })))
        );
        // Second batch should not contain StreamStart
        assert!(
            !r2.iter()
                .any(|e| matches!(e, Ok(ChatStreamEvent::StreamStart { .. })))
        );
    }

    #[tokio::test]
    async fn test_multi_parts_emit_multiple_deltas_in_order() {
        let config = create_test_config();
        let converter = GeminiEventConverter::new(config);
        let json = r#"{"candidates":[{"content":{"parts":[{"text":"A"},{"text":"B"}]}}]}"#;
        let event = eventsource_stream::Event {
            event: "".into(),
            data: json.into(),
            id: "".into(),
            retry: None,
        };
        let result = converter.convert_event(event).await;
        let deltas: Vec<_> = result
            .into_iter()
            .filter_map(|e| match e {
                Ok(ChatStreamEvent::ContentDelta { delta, .. }) => Some(delta),
                _ => None,
            })
            .collect();
        assert!(deltas.contains(&"A".to_string()));
        assert!(deltas.contains(&"B".to_string()));
        // Order is preserved within a single event
        let a_pos = deltas.iter().position(|d| d == "A").unwrap();
        let b_pos = deltas.iter().position(|d| d == "B").unwrap();
        assert!(a_pos < b_pos);
    }

    #[tokio::test]
    async fn test_thinking_delta_extraction() {
        let config = create_test_config();
        let converter = GeminiEventConverter::new(config);
        let json =
            r#"{"candidates":[{"content":{"parts":[{"text":"thinking..","thought":true}]}}]}"#;
        let event = eventsource_stream::Event {
            event: "".into(),
            data: json.into(),
            id: "".into(),
            retry: None,
        };
        let result = converter.convert_event(event).await;
        assert!(
            result
                .iter()
                .any(|e| matches!(e, Ok(ChatStreamEvent::ThinkingDelta { .. })))
        );
    }
}
