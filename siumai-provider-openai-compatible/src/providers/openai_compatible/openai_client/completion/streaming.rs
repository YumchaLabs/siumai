use crate::error::LlmError;
use crate::standards::openai::completion_metadata::{
    completion_created_at, completion_stream_response_metadata,
    extract_completion_provider_metadata, flatten_completion_stream_provider_metadata,
    merge_completion_provider_metadata,
};
use crate::standards::openai::utils::{
    parse_provider_openai_finish_reason, parse_provider_openai_usage_value,
};
use crate::streaming::{ChatStreamEvent, ChatStreamPart};
use crate::types::{
    ChatResponse, ChatStreamFinishInfo, FinishReason, MessageContent, ProviderMetadataMap,
    ResponseMetadata, Usage, Warning,
};
use std::sync::Arc;

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
pub(super) struct CompletionSseConverter {
    provider_id: String,
    provider_metadata_key: String,
    include_raw_chunks: bool,
    state: Arc<std::sync::Mutex<CompletionStreamState>>,
}

impl CompletionSseConverter {
    pub(super) fn new(
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
                    if include_raw_chunks {
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
            let usage = raw
                .get("usage")
                .and_then(|usage| parse_provider_openai_usage_value(provider_id.as_str(), usage));
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
