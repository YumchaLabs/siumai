//! Ollama streaming implementation using eventsource-stream
//!
//! Provides JSON event conversion for Ollama streaming responses.
//! The legacy OllamaStreaming client has been removed in favor of the unified HttpChatExecutor.

use crate::error::LlmError;
use crate::streaming::JsonEventConverter;
use crate::streaming::{ChatStreamEvent, StreamStateTracker};
use crate::types::{ChatResponse, FinishReason, MessageContent, ResponseMetadata, Usage};
use serde::Deserialize;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

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
    state_tracker: StreamStateTracker,
    /// Best-effort model id captured from `StreamStart`/`StreamEnd` for reverse serialization.
    stream_model: Arc<Mutex<Option<String>>>,
    /// Whether tool calls have been emitted (Ollama streams tool_calls as full objects).
    tool_calls_emitted: Arc<Mutex<bool>>,
}

impl Default for OllamaEventConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl OllamaEventConverter {
    pub fn new() -> Self {
        Self {
            state_tracker: StreamStateTracker::new(),
            stream_model: Arc::new(Mutex::new(None)),
            tool_calls_emitted: Arc::new(Mutex::new(false)),
        }
    }

    /// Convert Ollama stream response to multiple ChatStreamEvents
    async fn convert_ollama_response_async(
        &self,
        response: OllamaStreamResponse,
    ) -> Vec<ChatStreamEvent> {
        use crate::streaming::EventBuilder;
        use crate::types::{ChatResponse, FinishReason, MessageContent};

        let mut builder = EventBuilder::new();

        // Check if we need to emit StreamStart first
        if self.needs_stream_start() {
            let metadata = self.create_stream_start_metadata(&response);
            builder = builder.add_stream_start(metadata);
        }

        // Process thinking content (for models like deepseek-r1)
        if let Some(thinking) = self.extract_thinking(&response) {
            builder = builder.add_thinking_delta(thinking);
        }

        // Process tool calls (when models request function execution).
        if let Some(tool_calls) = self.extract_tool_calls(&response) {
            let mut emitted = self
                .tool_calls_emitted
                .lock()
                .expect("tool_calls_emitted lock");
            if !*emitted {
                for (idx, tc) in tool_calls.into_iter().enumerate() {
                    let args = serde_json::to_string(&tc.function.arguments).unwrap_or_default();
                    builder = builder.add_tool_call_delta(
                        format!("call_{idx}"),
                        Some(tc.function.name),
                        Some(args),
                        None,
                    );
                }
                *emitted = true;
            }
        }

        // Process content - NO MORE CONTENT LOSS!
        if let Some(content) = self.extract_content(&response) {
            builder = builder.add_content_delta(content, None);
        }

        // Process usage updates
        if let Some(usage) = self.extract_usage(&response) {
            builder = builder.add_usage_update(usage);
        }

        // Process stream end
        if response.done == Some(true) {
            // Mark that StreamEnd is being emitted
            self.state_tracker.mark_stream_ended();

            let chat_response = ChatResponse {
                id: None,
                model: response.model.clone(),
                content: MessageContent::Text(String::new()),
                usage: self.extract_usage(&response),
                finish_reason: Some(FinishReason::Stop),
                audio: None,
                system_fingerprint: None,
                service_tier: None,
                warnings: None,
                provider_metadata: None,
            };
            builder = builder.add_stream_end(chat_response);
        }

        builder.build()
    }

    /// Check if StreamStart event needs to be emitted
    fn needs_stream_start(&self) -> bool {
        self.state_tracker.needs_stream_start()
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

    /// Extract thinking content from Ollama response (for models like deepseek-r1)
    fn extract_thinking(&self, response: &OllamaStreamResponse) -> Option<String> {
        response
            .message
            .as_ref()?
            .thinking
            .as_ref()
            .filter(|thinking| !thinking.is_empty())
            .cloned()
    }

    fn extract_tool_calls(
        &self,
        response: &OllamaStreamResponse,
    ) -> Option<Vec<super::types::OllamaToolCall>> {
        response
            .message
            .as_ref()?
            .tool_calls
            .as_ref()
            .filter(|calls| !calls.is_empty())
            .cloned()
    }

    /// Extract usage information
    fn extract_usage(&self, response: &OllamaStreamResponse) -> Option<Usage> {
        if response.done == Some(true)
            && let (Some(prompt_tokens), Some(completion_tokens)) =
                (response.prompt_eval_count, response.eval_count)
        {
            return Some(
                Usage::builder()
                    .prompt_tokens(prompt_tokens)
                    .completion_tokens(completion_tokens)
                    .total_tokens(prompt_tokens + completion_tokens)
                    .build(),
            );
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
            match crate::streaming::parse_json_with_repair::<OllamaStreamResponse>(json_data) {
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

    fn handle_stream_end(&self) -> Option<Result<ChatStreamEvent, LlmError>> {
        // Ollama normally emits `done: true` in the stream (handled in convert_ollama_response_async).
        // If we reach here without seeing `done: true`, the model has not transmitted
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

    fn serialize_event(&self, event: &ChatStreamEvent) -> Result<Vec<u8>, LlmError> {
        match event {
            ChatStreamEvent::StreamStart { metadata } => {
                if let Some(model) = metadata.model.clone()
                    && let Ok(mut guard) = self.stream_model.lock()
                {
                    *guard = Some(model);
                }
                Ok(Vec::new())
            }
            ChatStreamEvent::ContentDelta { delta, .. } => {
                let model = self.stream_model.lock().ok().and_then(|v| v.clone());
                let body = serde_json::json!({
                    "model": model,
                    "message": { "role": "assistant", "content": delta },
                    "done": false,
                });
                let mut out = serde_json::to_vec(&body).map_err(|e| {
                    LlmError::ParseError(format!("Failed to serialize Ollama JSONL event: {e}"))
                })?;
                out.push(b'\n');
                Ok(out)
            }
            ChatStreamEvent::ThinkingDelta { delta } => {
                let model = self.stream_model.lock().ok().and_then(|v| v.clone());
                let body = serde_json::json!({
                    "model": model,
                    "message": { "role": "assistant", "thinking": delta },
                    "done": false,
                });
                let mut out = serde_json::to_vec(&body).map_err(|e| {
                    LlmError::ParseError(format!("Failed to serialize Ollama JSONL event: {e}"))
                })?;
                out.push(b'\n');
                Ok(out)
            }
            ChatStreamEvent::StreamEnd { response } => {
                if let Some(model) = response.model.clone()
                    && let Ok(mut guard) = self.stream_model.lock()
                {
                    *guard = Some(model);
                }

                let usage = response.usage.clone();
                let prompt_eval_count = usage.as_ref().map(|u| u.prompt_tokens);
                let eval_count = usage.as_ref().map(|u| u.completion_tokens);

                let model = self.stream_model.lock().ok().and_then(|v| v.clone());
                let body = serde_json::json!({
                    "model": model,
                    "done": true,
                    "prompt_eval_count": prompt_eval_count,
                    "eval_count": eval_count,
                });
                let mut out = serde_json::to_vec(&body).map_err(|e| {
                    LlmError::ParseError(format!("Failed to serialize Ollama JSONL event: {e}"))
                })?;
                out.push(b'\n');
                Ok(out)
            }
            ChatStreamEvent::Error { error } => {
                // Ollama's JSONL protocol does not define a stable error frame. Emit a best-effort
                // JSON line so downstream proxies can surface the error.
                let body = serde_json::json!({ "error": error });
                let mut out = serde_json::to_vec(&body).map_err(|e| {
                    LlmError::ParseError(format!("Failed to serialize Ollama JSONL event: {e}"))
                })?;
                out.push(b'\n');
                Ok(out)
            }
            ChatStreamEvent::UsageUpdate { .. }
            | ChatStreamEvent::ToolCallDelta { .. }
            | ChatStreamEvent::Custom { .. } => Ok(Vec::new()),
        }
    }
}

// Legacy OllamaStreaming client has been removed in favor of the unified HttpChatExecutor.
// The OllamaEventConverter is still used for JSON event conversion in tests.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Usage;

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

    #[tokio::test]
    async fn test_ollama_emits_tool_call_delta() {
        let converter = OllamaEventConverter::new();

        let json_data = r#"{"model":"llama3.2","message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"get_weather","arguments":{"city":"Toronto"}}}]},"done":false}"#;
        let result = converter.convert_json(json_data).await;

        let tool_event = result
            .iter()
            .find(|event| matches!(event, Ok(ChatStreamEvent::ToolCallDelta { .. })));

        match tool_event {
            Some(Ok(ChatStreamEvent::ToolCallDelta {
                id,
                function_name,
                arguments_delta,
                ..
            })) => {
                assert_eq!(id, "call_0");
                assert_eq!(function_name.as_deref(), Some("get_weather"));
                assert_eq!(arguments_delta.as_deref(), Some(r#"{"city":"Toronto"}"#));
            }
            _ => panic!("Expected ToolCallDelta event in results: {:?}", result),
        }
    }

    #[test]
    fn test_ollama_serializes_content_delta_to_jsonl() {
        let converter = OllamaEventConverter::new();

        let _ = converter.serialize_event(&ChatStreamEvent::StreamStart {
            metadata: ResponseMetadata {
                id: None,
                model: Some("llama3.2".to_string()),
                created: None,
                provider: "ollama".to_string(),
                request_id: None,
            },
        });

        let bytes = converter
            .serialize_event(&ChatStreamEvent::ContentDelta {
                delta: "hi".to_string(),
                index: None,
            })
            .expect("serialize ok");

        let line = String::from_utf8(bytes).expect("utf8");
        let v: serde_json::Value = serde_json::from_str(line.trim()).expect("json");
        assert_eq!(v.get("done").and_then(|x| x.as_bool()), Some(false));
        assert_eq!(
            v.get("message")
                .and_then(|m| m.get("content"))
                .and_then(|x| x.as_str()),
            Some("hi")
        );
        assert_eq!(v.get("model").and_then(|x| x.as_str()), Some("llama3.2"));
    }

    #[test]
    fn test_ollama_serializes_stream_end_with_usage_counts() {
        let converter = OllamaEventConverter::new();

        let _ = converter.serialize_event(&ChatStreamEvent::StreamStart {
            metadata: ResponseMetadata {
                id: None,
                model: Some("llama3.2".to_string()),
                created: None,
                provider: "ollama".to_string(),
                request_id: None,
            },
        });

        let bytes = converter
            .serialize_event(&ChatStreamEvent::StreamEnd {
                response: ChatResponse {
                    id: None,
                    model: Some("llama3.2".to_string()),
                    content: MessageContent::Text(String::new()),
                    usage: Some(
                        Usage::builder()
                            .prompt_tokens(10)
                            .completion_tokens(20)
                            .total_tokens(30)
                            .build(),
                    ),
                    finish_reason: Some(FinishReason::Stop),
                    audio: None,
                    system_fingerprint: None,
                    service_tier: None,
                    warnings: None,
                    provider_metadata: None,
                },
            })
            .expect("serialize ok");

        let line = String::from_utf8(bytes).expect("utf8");
        let v: serde_json::Value = serde_json::from_str(line.trim()).expect("json");
        assert_eq!(v.get("done").and_then(|x| x.as_bool()), Some(true));
        assert_eq!(
            v.get("prompt_eval_count").and_then(|x| x.as_u64()),
            Some(10)
        );
        assert_eq!(v.get("eval_count").and_then(|x| x.as_u64()), Some(20));
    }
}
