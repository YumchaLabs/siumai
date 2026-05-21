use crate::error::LlmError;
use crate::standards::openai::compat::usage::OpenAiCompatibleUsagePolicy;
use crate::standards::openai::completion_metadata::{
    completion_created_at, completion_response_metadata, completion_stream_response_metadata,
    extract_completion_provider_metadata, flatten_completion_stream_provider_metadata,
    merge_completion_provider_metadata,
};
use crate::standards::openai::utils::parse_provider_openai_finish_reason;
use crate::streaming::{ChatStreamEvent, ChatStreamPart};
use crate::types::{
    ChatResponse, ChatStreamFinishInfo, CompletionResponse, FinishReason, MessageContent,
    ProviderMetadataMap, ResponseMetadata, Usage, Warning,
};
use std::sync::Arc;

/// Runtime configuration for OpenAI-compatible `/completions` response conversion.
#[derive(Debug, Clone)]
pub struct CompletionResponseConversion {
    provider_id: String,
    provider_metadata_key: String,
}

impl CompletionResponseConversion {
    pub fn new(provider_id: impl Into<String>, provider_metadata_key: impl Into<String>) -> Self {
        Self {
            provider_id: provider_id.into(),
            provider_metadata_key: provider_metadata_key.into(),
        }
    }

    pub fn build_response(
        &self,
        raw: serde_json::Value,
        headers: &reqwest::header::HeaderMap,
        warnings: Vec<Warning>,
    ) -> CompletionResponse {
        build_completion_response(
            &self.provider_id,
            &self.provider_metadata_key,
            raw,
            headers,
            warnings,
        )
    }
}

#[derive(Debug, Clone)]
struct CompletionStreamState {
    text: String,
    id: Option<String>,
    model: Option<String>,
    created: Option<chrono::DateTime<chrono::Utc>>,
    usage: Option<Usage>,
    finish_reason: Option<FinishReason>,
    finish_reason_raw: Option<String>,
    warnings: Vec<Warning>,
    provider_metadata: Option<ProviderMetadataMap>,
    stream_start_emitted: bool,
    response_metadata_emitted: bool,
    text_started: bool,
}

impl CompletionStreamState {
    fn response_metadata(&self, provider: &str) -> ResponseMetadata {
        completion_stream_response_metadata(
            provider,
            self.id.as_deref(),
            self.model.as_deref(),
            self.created,
        )
    }

    fn finish_usage(&self) -> Usage {
        self.usage.clone().unwrap_or_default()
    }

    fn finish_part_provider_metadata(&self) -> Option<ProviderMetadataMap> {
        flatten_completion_stream_provider_metadata(&self.provider_metadata)
    }

    fn final_response(&self) -> ChatResponse {
        let mut response = ChatResponse::new(MessageContent::Text(self.text.clone()));
        response.id = self.id.clone();
        response.model = self.model.clone();
        response.usage = self.usage.clone();
        response.finish_reason = Some(self.finish_reason.clone().unwrap_or(FinishReason::Unknown));
        response.raw_finish_reason = self.finish_reason_raw.clone();
        if !self.warnings.is_empty() {
            response.warnings = Some(self.warnings.clone());
        }
        response.provider_metadata = self.provider_metadata.clone();
        response
    }
}

#[derive(Clone)]
pub struct CompletionSseConverter {
    provider_id: String,
    provider_metadata_key: String,
    include_raw_chunks: bool,
    state: Arc<std::sync::Mutex<CompletionStreamState>>,
}

impl CompletionSseConverter {
    pub fn new(
        provider_id: impl Into<String>,
        provider_metadata_key: impl Into<String>,
        warnings: Vec<Warning>,
        include_raw_chunks: bool,
    ) -> Self {
        Self {
            provider_id: provider_id.into(),
            provider_metadata_key: provider_metadata_key.into(),
            include_raw_chunks,
            state: Arc::new(std::sync::Mutex::new(CompletionStreamState {
                text: String::new(),
                id: None,
                model: None,
                created: None,
                usage: None,
                finish_reason: None,
                finish_reason_raw: None,
                warnings,
                provider_metadata: None,
                stream_start_emitted: false,
                response_metadata_emitted: false,
                text_started: false,
            })),
        }
    }
}

/// Build a Siumai completion response from an OpenAI-compatible `/completions` JSON payload.
///
/// The provider runtime owns HTTP execution. Protocol-owned conversion keeps the OpenAI-compatible
/// finish-reason, usage, response-metadata, and provider-metadata interpretation in one place.
pub fn build_completion_response(
    provider_id: &str,
    provider_metadata_key: &str,
    raw: serde_json::Value,
    headers: &reqwest::header::HeaderMap,
    warnings: Vec<Warning>,
) -> CompletionResponse {
    let text = raw
        .get("choices")
        .and_then(|value| value.as_array())
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("text"))
        .and_then(|value| value.as_str())
        .map(ToString::to_string)
        .unwrap_or_default();
    let raw_finish_reason = raw
        .get("choices")
        .and_then(|value| value.as_array())
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("finish_reason"))
        .and_then(|value| value.as_str())
        .map(ToString::to_string);
    let finish_reason = raw_finish_reason
        .as_deref()
        .and_then(|value| parse_provider_openai_finish_reason(provider_id, Some(value)));

    CompletionResponse {
        text,
        finish_reason,
        raw_finish_reason,
        usage: raw.get("usage").and_then(|usage| {
            OpenAiCompatibleUsagePolicy::for_provider(provider_id).convert_usage_value(usage)
        }),
        response_metadata: Some(completion_response_metadata(
            provider_id.to_string(),
            &raw,
            headers,
            true,
        )),
        warnings: (!warnings.is_empty()).then_some(warnings),
        provider_metadata: extract_completion_provider_metadata(provider_metadata_key, &raw),
    }
}

impl crate::streaming::SseEventConverter for CompletionSseConverter {
    fn convert_event(
        &self,
        event: eventsource_stream::Event,
    ) -> crate::streaming::SseEventFuture<'_> {
        let provider_id = self.provider_id.clone();
        let provider_metadata_key = self.provider_metadata_key.clone();
        let include_raw_chunks = self.include_raw_chunks;
        let state = self.state.clone();
        Box::pin(async move {
            let raw: serde_json::Value = match serde_json::from_str(&event.data) {
                Ok(raw) => raw,
                Err(err) => {
                    let mut events = Vec::new();
                    {
                        let mut state = state.lock().expect("completion stream state");
                        if !state.stream_start_emitted {
                            let metadata = state.response_metadata(&provider_id);
                            events.push(Ok(ChatStreamEvent::StreamStart {
                                metadata: metadata.clone(),
                            }));
                            events.push(Ok(ChatStreamEvent::Part {
                                part: ChatStreamPart::StreamStart {
                                    warnings: state.warnings.clone(),
                                },
                            }));
                            state.stream_start_emitted = true;
                        }
                    }
                    if include_raw_chunks {
                        events.push(Ok(ChatStreamEvent::Part {
                            part: ChatStreamPart::Raw {
                                raw_value: serde_json::Value::String(event.data.clone()),
                            },
                        }));
                    }
                    events.push(Err(LlmError::ParseError(format!(
                        "Failed to parse completion stream event: {err}"
                    ))));
                    return events;
                }
            };

            let delta = raw
                .get("choices")
                .and_then(|value| value.as_array())
                .and_then(|choices| choices.first())
                .and_then(|choice| choice.get("text"))
                .and_then(|value| value.as_str())
                .map(ToString::to_string);
            let finish_reason_raw = raw
                .get("choices")
                .and_then(|value| value.as_array())
                .and_then(|choices| choices.first())
                .and_then(|choice| choice.get("finish_reason"))
                .and_then(|value| value.as_str())
                .map(ToString::to_string);
            let finish_reason = finish_reason_raw.as_deref().and_then(|value| {
                parse_provider_openai_finish_reason(provider_id.as_str(), Some(value))
            });
            let usage = raw.get("usage").and_then(|usage| {
                OpenAiCompatibleUsagePolicy::for_provider(provider_id.as_str())
                    .convert_usage_value(usage)
            });
            let provider_metadata =
                extract_completion_provider_metadata(&provider_metadata_key, &raw);
            let created = completion_created_at(&raw);

            let mut events = Vec::new();
            let (metadata, warnings, emit_stream_start, emit_response_metadata, emit_text_start) = {
                let mut state = state.lock().expect("completion stream state");
                if let Some(id) = raw.get("id").and_then(|value| value.as_str()) {
                    state.id = Some(id.to_string());
                }
                if let Some(model) = raw.get("model").and_then(|value| value.as_str()) {
                    state.model = Some(model.to_string());
                }
                if let Some(created) = created {
                    state.created = Some(created);
                }
                if let Some(usage) = usage {
                    state.usage = Some(usage);
                }
                if let Some(finish_reason) = finish_reason {
                    state.finish_reason = Some(finish_reason);
                }
                if let Some(raw_reason) = finish_reason_raw.clone() {
                    state.finish_reason_raw = Some(raw_reason);
                }
                merge_completion_provider_metadata(&mut state.provider_metadata, provider_metadata);
                if let Some(delta) = delta.as_deref() {
                    state.text.push_str(delta);
                }

                let metadata = state.response_metadata(&provider_id);
                let warnings = state.warnings.clone();
                let emit_stream_start = !state.stream_start_emitted;
                if emit_stream_start {
                    state.stream_start_emitted = true;
                }
                let emit_response_metadata = !state.response_metadata_emitted;
                if emit_response_metadata {
                    state.response_metadata_emitted = true;
                }
                let emit_text_start = delta.is_some() && !state.text_started;
                if emit_text_start {
                    state.text_started = true;
                }

                (
                    metadata,
                    warnings,
                    emit_stream_start,
                    emit_response_metadata,
                    emit_text_start,
                )
            };

            if emit_stream_start {
                events.push(Ok(ChatStreamEvent::StreamStart {
                    metadata: metadata.clone(),
                }));
                events.push(Ok(ChatStreamEvent::Part {
                    part: ChatStreamPart::StreamStart { warnings },
                }));
            }

            if include_raw_chunks {
                events.push(Ok(ChatStreamEvent::Part {
                    part: ChatStreamPart::Raw { raw_value: raw },
                }));
            }

            if emit_response_metadata {
                events.push(Ok(ChatStreamEvent::Part {
                    part: ChatStreamPart::ResponseMetadata(metadata),
                }));
            }

            if emit_text_start {
                events.push(Ok(ChatStreamEvent::Part {
                    part: ChatStreamPart::TextStart {
                        id: "0".to_string(),
                        provider_metadata: None,
                    },
                }));
            }

            if let Some(delta) = delta {
                events.push(Ok(ChatStreamEvent::text_delta_part("0", delta)));
            }

            events
        })
    }

    fn is_stream_end_event(&self, event: &eventsource_stream::Event) -> bool {
        event.data.trim() == "[DONE]"
    }

    fn handle_stream_end_events(&self) -> Vec<Result<ChatStreamEvent, LlmError>> {
        let state = self.state.lock().expect("completion stream state");
        let mut events = Vec::new();

        if state.text_started {
            events.push(Ok(ChatStreamEvent::Part {
                part: ChatStreamPart::TextEnd {
                    id: "0".to_string(),
                    provider_metadata: None,
                },
            }));
        }

        events.push(Ok(ChatStreamEvent::Part {
            part: ChatStreamPart::Finish {
                usage: state.finish_usage(),
                finish_reason: ChatStreamFinishInfo {
                    unified: state.finish_reason.clone().unwrap_or(FinishReason::Unknown),
                    raw: state.finish_reason_raw.clone(),
                },
                provider_metadata: state.finish_part_provider_metadata(),
            },
        }));
        events.push(Ok(ChatStreamEvent::StreamEnd {
            response: state.final_response(),
        }));

        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::SseEventConverter;

    fn convert_ok(
        converter: &CompletionSseConverter,
        raw: serde_json::Value,
    ) -> Vec<ChatStreamEvent> {
        let event = eventsource_stream::Event {
            event: String::new(),
            data: raw.to_string(),
            id: String::new(),
            retry: None,
        };

        futures::executor::block_on(converter.convert_event(event))
            .into_iter()
            .map(|event| event.expect("event ok"))
            .collect()
    }

    #[test]
    fn openai_compatible_completion_sse_converter_preserves_empty_and_whitespace_text_deltas() {
        let converter = CompletionSseConverter::new("openrouter", "openrouter", Vec::new(), false);

        let events = convert_ok(
            &converter,
            serde_json::json!({
                "id": "cmpl_1",
                "model": "text-model",
                "choices": [{ "text": "", "finish_reason": null }]
            }),
        );
        assert!(events.iter().any(|event| matches!(
            event,
            ChatStreamEvent::Part {
                part: ChatStreamPart::TextStart { .. }
            }
        )));
        assert!(
            events
                .iter()
                .any(|event| event.text_delta().is_some_and(|delta| delta.is_empty()))
        );

        let events = convert_ok(
            &converter,
            serde_json::json!({
                "choices": [{ "text": "   ", "finish_reason": null }]
            }),
        );
        assert!(
            events
                .iter()
                .any(|event| event.text_delta().is_some_and(|delta| delta == "   "))
        );
    }

    #[test]
    fn openai_compatible_completion_sse_converter_accumulates_metadata_and_emits_finish() {
        let converter = CompletionSseConverter::new("openrouter", "openrouter", Vec::new(), false);

        let first_events = convert_ok(
            &converter,
            serde_json::json!({
                "id": "cmpl_1",
                "model": "text-model",
                "created": 1_718_345_013,
                "choices": [{
                    "text": "hello",
                    "finish_reason": null,
                    "logprobs": {
                        "tokens": ["hello"],
                        "token_logprobs": [-0.2]
                    }
                }]
            }),
        );
        assert!(first_events.iter().any(|event| matches!(
            event,
            ChatStreamEvent::Part {
                part: ChatStreamPart::ResponseMetadata(metadata)
            } if metadata.id.as_deref() == Some("cmpl_1")
                && metadata.model.as_deref() == Some("text-model")
        )));

        let second_events = convert_ok(
            &converter,
            serde_json::json!({
                "choices": [{
                    "text": " world",
                    "finish_reason": "stop"
                }],
                "sources": [{ "url": "https://example.com/source" }],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5
                }
            }),
        );
        assert!(
            second_events
                .iter()
                .any(|event| event.text_delta().is_some_and(|delta| delta == " world"))
        );

        let done = eventsource_stream::Event {
            event: String::new(),
            data: "[DONE]".to_string(),
            id: String::new(),
            retry: None,
        };
        assert!(converter.is_stream_end_event(&done));

        let end_events = converter
            .handle_stream_end_events()
            .into_iter()
            .map(|event| event.expect("end event ok"))
            .collect::<Vec<_>>();

        assert!(end_events.iter().any(|event| matches!(
            event,
            ChatStreamEvent::Part {
                part: ChatStreamPart::TextEnd { .. }
            }
        )));
        assert!(end_events.iter().any(|event| matches!(
            event,
            ChatStreamEvent::Part {
                part: ChatStreamPart::Finish {
                    finish_reason,
                    provider_metadata,
                    ..
                }
            } if matches!(finish_reason.unified, FinishReason::Stop)
                && provider_metadata
                    .as_ref()
                    .and_then(|metadata| metadata.get("openrouter"))
                    .and_then(|metadata| metadata.get("sources"))
                    == Some(&serde_json::json!([{ "url": "https://example.com/source" }]))
        )));
        assert!(end_events.iter().any(|event| matches!(
            event,
            ChatStreamEvent::StreamEnd { response }
                if response.content_text() == Some("hello world")
                    && matches!(response.finish_reason, Some(FinishReason::Stop))
        )));
    }

    #[test]
    fn openai_compatible_completion_sse_converter_emits_stream_start_before_parse_error() {
        let converter = CompletionSseConverter::new(
            "openrouter",
            "openrouter",
            vec![Warning::other("warn")],
            false,
        );

        let events =
            futures::executor::block_on(converter.convert_event(eventsource_stream::Event {
                event: String::new(),
                data: "not-json".to_string(),
                id: String::new(),
                retry: None,
            }));

        assert_eq!(events.len(), 3);
        assert!(matches!(
            events.first(),
            Some(Ok(ChatStreamEvent::StreamStart { metadata }))
                if metadata.provider == "openrouter"
        ));
        assert!(matches!(
            events.get(1),
            Some(Ok(ChatStreamEvent::Part {
                part: ChatStreamPart::StreamStart { warnings }
            })) if warnings.len() == 1
        ));
        assert!(matches!(
            events.get(2),
            Some(Err(LlmError::ParseError(message)))
                if message.contains("Failed to parse completion stream event")
        ));
    }
}
