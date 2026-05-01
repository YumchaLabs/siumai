//! Cohere native chat standard aligned with AI SDK provider behavior.

use super::shared;
use crate::core::ChatTransformers;
use crate::error::LlmError;
use crate::execution::transformers::request::RequestTransformer;
use crate::execution::transformers::response::ResponseTransformer;
use crate::execution::transformers::stream::{StreamChunkTransformer, StreamEventFuture};
use crate::provider_options::{CohereChatOptions, CohereThinkingType};
use crate::streaming::EventBuilder;
use crate::types::{
    ChatRequest, ChatResponse, ChatStreamEvent, ChatStreamFinishInfo, ChatStreamPart,
    ChatStreamToolCall, ContentPart, FinishReason, HttpResponseInfo, MessageContent,
    ResponseFormat, SourcePart, Tool, ToolChoice, Usage, Warning,
};
use eventsource_stream::Event;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
pub struct CohereChatStandard;

impl CohereChatStandard {
    pub fn new() -> Self {
        Self
    }

    pub fn create_transformers(
        &self,
        provider_id: &str,
        request: &ChatRequest,
    ) -> ChatTransformers {
        let default_model = request.common_params.model.clone();
        let warnings = cohere_tool_warnings(request.tools.as_deref());
        ChatTransformers {
            request: Arc::new(CohereChatRequestTransformer {
                provider_id: provider_id.to_string(),
            }),
            response: Arc::new(CohereChatResponseTransformer {
                provider_id: provider_id.to_string(),
                default_model,
                warnings: warnings.clone(),
            }),
            stream: Some(Arc::new(CohereChatStreamTransformer::new(
                provider_id.to_string(),
                request.common_params.model.clone(),
                warnings,
                request.stream_options.include_raw_chunks,
            ))),
            json: None,
        }
    }
}

#[derive(Clone)]
struct CohereChatRequestTransformer {
    provider_id: String,
}

impl RequestTransformer for CohereChatRequestTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_chat(&self, req: &ChatRequest) -> Result<Value, LlmError> {
        let model = effective_chat_model(req)?;
        let options =
            shared::cohere_provider_options::<CohereChatOptions>(&req.provider_options_map)?;
        let (messages, documents) = convert_prompt_messages(req)?;
        let (tools, tool_choice) = prepare_tools(req.tools.as_deref(), req.tool_choice.as_ref());

        let mut body = serde_json::Map::new();
        body.insert("model".to_string(), Value::String(model.to_string()));
        body.insert("messages".to_string(), Value::Array(messages));

        if let Some(value) = req.common_params.frequency_penalty {
            body.insert("frequency_penalty".to_string(), json!(value));
        }
        if let Some(value) = req.common_params.presence_penalty {
            body.insert("presence_penalty".to_string(), json!(value));
        }
        if let Some(value) = req.common_params.max_tokens {
            body.insert("max_tokens".to_string(), json!(value));
        }
        if let Some(value) = req.common_params.temperature {
            body.insert("temperature".to_string(), json!(value));
        }
        if let Some(value) = req.common_params.top_p {
            body.insert("p".to_string(), json!(value));
        }
        if let Some(value) = req.common_params.top_k {
            body.insert("k".to_string(), json!(value));
        }
        if let Some(value) = req.common_params.seed {
            body.insert("seed".to_string(), json!(value));
        }
        if let Some(value) = req.common_params.stop_sequences.as_ref()
            && !value.is_empty()
        {
            body.insert("stop_sequences".to_string(), json!(value));
        }
        if let Some(format) = req.response_format.as_ref()
            && let Some(mapped) = map_response_format(format)
        {
            body.insert("response_format".to_string(), mapped);
        }
        if !tools.is_empty() {
            body.insert("tools".to_string(), Value::Array(tools));
        }
        if let Some(tool_choice) = tool_choice {
            body.insert("tool_choice".to_string(), Value::String(tool_choice));
        }
        if !documents.is_empty() {
            body.insert("documents".to_string(), Value::Array(documents));
        }
        if let Some(thinking) = options.and_then(|options| options.thinking)
            && let Some(mapped) = map_thinking_config(&thinking)
        {
            body.insert("thinking".to_string(), mapped);
        }

        Ok(Value::Object(body))
    }
}

#[derive(Clone)]
struct CohereChatResponseTransformer {
    provider_id: String,
    default_model: String,
    warnings: Vec<Warning>,
}

fn cohere_citation_provider_metadata(citation: &Value) -> Value {
    let mut metadata = serde_json::Map::new();

    for key in ["start", "end", "text", "sources"] {
        if let Some(value) = citation.get(key) {
            metadata.insert(key.to_string(), value.clone());
        }
    }

    if let Some(value) = citation.get("type") {
        metadata.insert("citationType".to_string(), value.clone());
    }

    Value::Object(metadata)
}

impl ResponseTransformer for CohereChatResponseTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_chat_response(&self, raw: &Value) -> Result<ChatResponse, LlmError> {
        let mut parts = Vec::new();

        if let Some(content_items) = raw
            .get("message")
            .and_then(|message| message.get("content"))
            .and_then(Value::as_array)
        {
            for item in content_items {
                let item_type = item.get("type").and_then(Value::as_str).unwrap_or_default();
                match item_type {
                    "text" => {
                        let text = item.get("text").and_then(Value::as_str).unwrap_or_default();
                        if !text.is_empty() {
                            parts.push(ContentPart::text(text));
                        }
                    }
                    "thinking" => {
                        let text = item
                            .get("thinking")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        if !text.is_empty() {
                            parts.push(ContentPart::reasoning(text));
                        }
                    }
                    _ => {}
                }
            }
        }

        if let Some(citations) = raw
            .get("message")
            .and_then(|message| message.get("citations"))
            .and_then(Value::as_array)
        {
            for (index, citation) in citations.iter().enumerate() {
                let title = citation
                    .get("sources")
                    .and_then(Value::as_array)
                    .and_then(|sources| sources.first())
                    .and_then(|source| source.get("document"))
                    .and_then(|document| document.get("title"))
                    .and_then(Value::as_str)
                    .unwrap_or("Document")
                    .to_string();

                parts.push(ContentPart::Source {
                    id: format!("cohere-citation-{index}"),
                    source: SourcePart::Document {
                        media_type: "text/plain".to_string(),
                        title,
                        filename: None,
                    },
                    provider_metadata: shared::provider_metadata_entry(
                        "cohere",
                        cohere_citation_provider_metadata(citation),
                    ),
                });
            }
        }

        if let Some(tool_calls) = raw
            .get("message")
            .and_then(|message| message.get("tool_calls"))
            .and_then(Value::as_array)
        {
            for tool_call in tool_calls {
                let tool_call_id =
                    tool_call.get("id").and_then(Value::as_str).ok_or_else(|| {
                        LlmError::ParseError("Missing Cohere tool call id".to_string())
                    })?;
                let tool_name = tool_call
                    .get("function")
                    .and_then(|function| function.get("name"))
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        LlmError::ParseError("Missing Cohere tool call function.name".to_string())
                    })?;
                let arguments = tool_call
                    .get("function")
                    .and_then(|function| function.get("arguments"))
                    .and_then(Value::as_str)
                    .unwrap_or("{}");

                parts.push(ContentPart::tool_call(
                    tool_call_id,
                    tool_name,
                    parse_tool_call_input(arguments)?,
                    None,
                ));
            }
        }

        let usage = raw
            .get("usage")
            .and_then(|usage| usage.get("tokens"))
            .map(|tokens| {
                let input_tokens = tokens
                    .get("input_tokens")
                    .and_then(Value::as_u64)
                    .map(|value| value as u32);
                let output_tokens = tokens
                    .get("output_tokens")
                    .and_then(Value::as_u64)
                    .map(|value| value as u32);
                shared::build_usage(input_tokens, output_tokens, Some(tokens.clone()))
            });

        let model = (!self.default_model.trim().is_empty())
            .then_some(self.default_model.clone())
            .or_else(|| {
                raw.get("response")
                    .and_then(|value| value.get("model"))
                    .and_then(Value::as_str)
                    .map(|value| value.to_string())
            });

        Ok(ChatResponse {
            id: raw
                .get("generation_id")
                .and_then(Value::as_str)
                .map(|value| value.to_string()),
            content: shared::message_content_from_parts(parts),
            model: model.clone(),
            usage,
            finish_reason: Some(shared::map_finish_reason(
                raw.get("finish_reason").and_then(Value::as_str),
            )),
            raw_finish_reason: raw
                .get("finish_reason")
                .and_then(Value::as_str)
                .map(ToString::to_string),
            audio: None,
            system_fingerprint: None,
            service_tier: None,
            warnings: (!self.warnings.is_empty()).then_some(self.warnings.clone()),
            request: None,
            provider_metadata: None,
            response: Some(HttpResponseInfo {
                timestamp: chrono::Utc::now(),
                model_id: model,
                headers: HashMap::new(),
                body: Some(raw.clone()),
            }),
        })
    }
}

#[derive(Debug, Clone)]
struct PendingToolCall {
    id: String,
    name: String,
    arguments: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamContentKind {
    Text,
    Reasoning,
}

#[derive(Debug, Default)]
struct CohereChatStreamState {
    response_id: Option<String>,
    content_kinds: HashMap<usize, StreamContentKind>,
    pending_tool_call: Option<PendingToolCall>,
    finish_reason_raw: Option<String>,
    finish_reason: Option<FinishReason>,
    usage: Option<Usage>,
    stream_start_emitted: bool,
}

#[derive(Clone)]
struct CohereChatStreamTransformer {
    provider_id: String,
    default_model: String,
    warnings: Vec<Warning>,
    include_raw_chunks: bool,
    state: Arc<Mutex<CohereChatStreamState>>,
}

impl CohereChatStreamTransformer {
    fn new(
        provider_id: String,
        default_model: String,
        warnings: Vec<Warning>,
        include_raw_chunks: bool,
    ) -> Self {
        Self {
            provider_id,
            default_model,
            warnings,
            include_raw_chunks,
            state: Arc::new(Mutex::new(CohereChatStreamState::default())),
        }
    }

    fn response_model(&self) -> Option<String> {
        (!self.default_model.trim().is_empty()).then_some(self.default_model.clone())
    }
}

impl StreamChunkTransformer for CohereChatStreamTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn convert_event(&self, event: Event) -> StreamEventFuture<'_> {
        Box::pin(async move {
            if event.data.trim().is_empty() {
                return vec![];
            }

            let value: Value = match serde_json::from_str(&event.data) {
                Ok(value) => value,
                Err(error) => {
                    let mut out = fallback_stream_start_events(self);
                    if self.include_raw_chunks {
                        out.push(Ok(ChatStreamEvent::Part {
                            part: ChatStreamPart::Raw {
                                raw_value: Value::String(event.data.clone()),
                            },
                        }));
                    }
                    out.push(Err(LlmError::ParseError(format!(
                        "Failed to parse Cohere stream event JSON: {error}"
                    ))));
                    return out;
                }
            };

            let event_type = value
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default();

            match event_type {
                "message-start" => handle_message_start(self, &value),
                "content-start" => with_raw_chunk(handle_content_start(self, &value), &value, self),
                "content-delta" => with_raw_chunk(handle_content_delta(self, &value), &value, self),
                "content-end" => with_raw_chunk(handle_content_end(self, &value), &value, self),
                "tool-call-start" => {
                    with_raw_chunk(handle_tool_call_start(self, &value), &value, self)
                }
                "tool-call-delta" => {
                    with_raw_chunk(handle_tool_call_delta(self, &value), &value, self)
                }
                "tool-call-end" => with_raw_chunk(handle_tool_call_end(self), &value, self),
                "message-end" => {
                    handle_message_end(self, &value);
                    with_raw_chunk(vec![], &value, self)
                }
                "citation-start" | "citation-end" | "tool-plan-delta" => {
                    with_raw_chunk(vec![], &value, self)
                }
                _ => with_raw_chunk(vec![], &value, self),
            }
        })
    }

    fn handle_stream_end_events(&self) -> Vec<Result<ChatStreamEvent, LlmError>> {
        let (response_id, usage, finish_reason, raw_finish_reason) = match self.state.lock() {
            Ok(state) => (
                state.response_id.clone(),
                state.usage.clone(),
                state.finish_reason.clone().unwrap_or(FinishReason::Unknown),
                state.finish_reason_raw.clone(),
            ),
            Err(_) => (None, None, FinishReason::Unknown, None),
        };

        let finish_usage = usage.clone().unwrap_or_else(Usage::unknown);
        let mut response = shared::build_stream_end_response(
            response_id,
            self.response_model(),
            usage.clone(),
            finish_reason.clone(),
        );
        response.raw_finish_reason = raw_finish_reason.clone();
        if !self.warnings.is_empty() {
            response.warnings = Some(self.warnings.clone());
        }

        let mut builder = EventBuilder::with_capacity(3);
        if let Some(usage) = usage {
            builder = builder.add_usage_update(usage);
        }
        builder
            .add_part(ChatStreamPart::Finish {
                usage: finish_usage,
                finish_reason: ChatStreamFinishInfo {
                    unified: finish_reason,
                    raw: raw_finish_reason,
                },
                provider_metadata: None,
            })
            .add_stream_end(response)
            .build_results()
    }

    fn finalize_on_disconnect(&self) -> bool {
        true
    }
}

fn handle_message_start(
    transformer: &CohereChatStreamTransformer,
    value: &Value,
) -> Vec<Result<ChatStreamEvent, LlmError>> {
    let id = value
        .get("id")
        .and_then(Value::as_str)
        .map(|value| value.to_string());
    let emit_stream_start_part = if let Ok(mut state) = transformer.state.lock() {
        state.response_id = id.clone();
        let emit = !state.stream_start_emitted;
        if emit {
            state.stream_start_emitted = true;
        }
        emit
    } else {
        false
    };

    let metadata =
        shared::response_metadata(&transformer.provider_id, id, transformer.response_model());
    let mut builder = EventBuilder::with_capacity(4);
    if emit_stream_start_part {
        builder =
            builder
                .add_stream_start(metadata.clone())
                .add_part(ChatStreamPart::StreamStart {
                    warnings: transformer.warnings.clone(),
                });
    }
    if transformer.include_raw_chunks {
        builder = builder.add_part(ChatStreamPart::Raw {
            raw_value: value.clone(),
        });
    }
    builder
        .add_part(ChatStreamPart::ResponseMetadata(metadata))
        .build_results()
}

fn handle_content_start(
    transformer: &CohereChatStreamTransformer,
    value: &Value,
) -> Vec<Result<ChatStreamEvent, LlmError>> {
    let index = value.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
    let kind = match value
        .get("delta")
        .and_then(|delta| delta.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(|content| content.get("type"))
        .and_then(Value::as_str)
    {
        Some("thinking") => StreamContentKind::Reasoning,
        _ => StreamContentKind::Text,
    };

    if let Ok(mut state) = transformer.state.lock() {
        state.content_kinds.insert(index, kind);
    }

    let part = match kind {
        StreamContentKind::Text => ChatStreamPart::TextStart {
            id: index.to_string(),
            provider_metadata: None,
        },
        StreamContentKind::Reasoning => ChatStreamPart::ReasoningStart {
            id: index.to_string(),
            provider_metadata: None,
        },
    };

    EventBuilder::new().add_part(part).build_results()
}

fn handle_content_delta(
    _transformer: &CohereChatStreamTransformer,
    value: &Value,
) -> Vec<Result<ChatStreamEvent, LlmError>> {
    let index = value.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
    let content = value
        .get("delta")
        .and_then(|delta| delta.get("message"))
        .and_then(|message| message.get("content"))
        .cloned()
        .unwrap_or(Value::Null);

    if let Some(delta) = content.get("thinking").and_then(Value::as_str) {
        return EventBuilder::with_capacity(2)
            .add_thinking_delta(delta.to_string())
            .add_part(ChatStreamPart::ReasoningDelta {
                id: index.to_string(),
                delta: delta.to_string(),
                provider_metadata: None,
            })
            .build_results();
    }

    let delta = content
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();

    EventBuilder::with_capacity(2)
        .add_content_delta(delta.clone(), None)
        .add_part(ChatStreamPart::TextDelta {
            id: index.to_string(),
            delta,
            provider_metadata: None,
        })
        .build_results()
}

fn handle_content_end(
    transformer: &CohereChatStreamTransformer,
    value: &Value,
) -> Vec<Result<ChatStreamEvent, LlmError>> {
    let index = value.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
    let kind = transformer
        .state
        .lock()
        .ok()
        .and_then(|mut state| state.content_kinds.remove(&index))
        .unwrap_or(StreamContentKind::Text);

    let part = match kind {
        StreamContentKind::Text => ChatStreamPart::TextEnd {
            id: index.to_string(),
            provider_metadata: None,
        },
        StreamContentKind::Reasoning => ChatStreamPart::ReasoningEnd {
            id: index.to_string(),
            provider_metadata: None,
        },
    };

    EventBuilder::new().add_part(part).build_results()
}

fn handle_tool_call_start(
    transformer: &CohereChatStreamTransformer,
    value: &Value,
) -> Vec<Result<ChatStreamEvent, LlmError>> {
    let tool_call = value
        .get("delta")
        .and_then(|delta| delta.get("message"))
        .and_then(|message| message.get("tool_calls"))
        .cloned()
        .unwrap_or(Value::Null);
    let id = tool_call
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let name = tool_call
        .get("function")
        .and_then(|function| function.get("name"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let arguments = tool_call
        .get("function")
        .and_then(|function| function.get("arguments"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();

    if let Ok(mut state) = transformer.state.lock() {
        state.pending_tool_call = Some(PendingToolCall {
            id: id.clone(),
            name: name.clone(),
            arguments: arguments.clone(),
        });
    }

    let mut builder = EventBuilder::with_capacity(3)
        .add_tool_call_delta(
            id.clone(),
            Some(name.clone()),
            (!arguments.is_empty()).then_some(arguments.clone()),
            None,
        )
        .add_part(ChatStreamPart::ToolInputStart {
            id: id.clone(),
            tool_name: name,
            provider_metadata: None,
            provider_executed: None,
            dynamic: None,
            title: None,
        });

    if !arguments.is_empty() {
        builder = builder.add_part(ChatStreamPart::ToolInputDelta {
            id,
            delta: arguments,
            provider_metadata: None,
        });
    }

    builder.build_results()
}

fn handle_tool_call_delta(
    transformer: &CohereChatStreamTransformer,
    value: &Value,
) -> Vec<Result<ChatStreamEvent, LlmError>> {
    let delta = value
        .get("delta")
        .and_then(|delta| delta.get("message"))
        .and_then(|message| message.get("tool_calls"))
        .and_then(|tool_calls| tool_calls.get("function"))
        .and_then(|function| function.get("arguments"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();

    let id = transformer
        .state
        .lock()
        .ok()
        .and_then(|mut state| {
            let pending = state.pending_tool_call.as_mut()?;
            pending.arguments.push_str(&delta);
            Some(pending.id.clone())
        })
        .unwrap_or_default();

    EventBuilder::with_capacity(2)
        .add_tool_call_delta(id.clone(), None, Some(delta.clone()), None)
        .add_part(ChatStreamPart::ToolInputDelta {
            id,
            delta,
            provider_metadata: None,
        })
        .build_results()
}

fn handle_tool_call_end(
    transformer: &CohereChatStreamTransformer,
) -> Vec<Result<ChatStreamEvent, LlmError>> {
    let pending = transformer
        .state
        .lock()
        .ok()
        .and_then(|mut state| state.pending_tool_call.take());

    let Some(pending) = pending else {
        return vec![];
    };

    let input = match tool_call_input_string(&pending.arguments) {
        Ok(input) => input,
        Err(error) => return vec![Err(error)],
    };

    EventBuilder::with_capacity(2)
        .add_part(ChatStreamPart::ToolInputEnd {
            id: pending.id.clone(),
            provider_metadata: None,
        })
        .add_part(ChatStreamPart::ToolCall(ChatStreamToolCall {
            tool_call_id: pending.id,
            tool_name: pending.name,
            input,
            provider_executed: None,
            dynamic: None,
            provider_metadata: None,
        }))
        .build_results()
}

fn handle_message_end(transformer: &CohereChatStreamTransformer, value: &Value) {
    let raw_finish_reason = value
        .get("delta")
        .and_then(|delta| delta.get("finish_reason"))
        .and_then(Value::as_str)
        .map(|value| value.to_string());
    let usage_tokens = value
        .get("delta")
        .and_then(|delta| delta.get("usage"))
        .and_then(|usage| usage.get("tokens"))
        .cloned();

    if let Ok(mut state) = transformer.state.lock() {
        state.finish_reason = Some(shared::map_finish_reason(raw_finish_reason.as_deref()));
        state.finish_reason_raw = raw_finish_reason;
        state.usage = usage_tokens.as_ref().map(|tokens| {
            let input_tokens = tokens
                .get("input_tokens")
                .and_then(Value::as_u64)
                .map(|value| value as u32);
            let output_tokens = tokens
                .get("output_tokens")
                .and_then(Value::as_u64)
                .map(|value| value as u32);
            shared::build_usage(input_tokens, output_tokens, Some(tokens.clone()))
        });
    }
}

fn with_raw_chunk(
    events: Vec<Result<ChatStreamEvent, LlmError>>,
    raw_value: &Value,
    transformer: &CohereChatStreamTransformer,
) -> Vec<Result<ChatStreamEvent, LlmError>> {
    if !transformer.include_raw_chunks {
        return events;
    }

    let mut out = Vec::with_capacity(events.len() + 1);
    out.push(Ok(ChatStreamEvent::Part {
        part: ChatStreamPart::Raw {
            raw_value: raw_value.clone(),
        },
    }));
    out.extend(events);
    out
}

fn fallback_stream_start_events(
    transformer: &CohereChatStreamTransformer,
) -> Vec<Result<ChatStreamEvent, LlmError>> {
    let emit = if let Ok(mut state) = transformer.state.lock() {
        let emit = !state.stream_start_emitted;
        if emit {
            state.stream_start_emitted = true;
        }
        emit
    } else {
        false
    };

    if !emit {
        return Vec::new();
    }

    EventBuilder::with_capacity(2)
        .add_stream_start(shared::response_metadata(
            &transformer.provider_id,
            None,
            transformer.response_model(),
        ))
        .add_part(ChatStreamPart::StreamStart {
            warnings: transformer.warnings.clone(),
        })
        .build_results()
}

fn cohere_tool_warnings(tools: Option<&[Tool]>) -> Vec<Warning> {
    let Some(tools) = tools else {
        return Vec::new();
    };

    tools
        .iter()
        .filter_map(|tool| match tool {
            Tool::ProviderDefined(tool) => Some(Warning::unsupported(
                format!("provider-defined tool {}", tool.id),
                None::<String>,
            )),
            _ => None,
        })
        .collect()
}

fn effective_chat_model(req: &ChatRequest) -> Result<&str, LlmError> {
    let model = req.common_params.model.trim();
    if model.is_empty() {
        Err(LlmError::ConfigurationError(
            "Cohere chat request requires a non-empty model id".to_string(),
        ))
    } else {
        Ok(model)
    }
}

fn convert_prompt_messages(req: &ChatRequest) -> Result<(Vec<Value>, Vec<Value>), LlmError> {
    let mut messages = Vec::new();
    let mut documents = Vec::new();

    for message in &req.messages {
        match &message.role {
            crate::types::MessageRole::System | crate::types::MessageRole::Developer => {
                messages.push(json!({
                    "role": "system",
                    "content": text_only_message_content(&message.content, "system/developer")?,
                }));
            }
            crate::types::MessageRole::User => {
                messages.push(json!({
                    "role": "user",
                    "content": convert_user_content(&message.content, &mut documents)?,
                }));
            }
            crate::types::MessageRole::Assistant => {
                let (content, tool_calls) = convert_assistant_content(&message.content)?;
                let mut value = serde_json::Map::new();
                value.insert("role".to_string(), Value::String("assistant".to_string()));
                if tool_calls.is_empty() {
                    value.insert("content".to_string(), Value::String(content));
                } else {
                    value.insert("content".to_string(), Value::Null);
                    value.insert("tool_calls".to_string(), Value::Array(tool_calls));
                }
                value.insert("tool_plan".to_string(), Value::Null);
                messages.push(Value::Object(value));
            }
            crate::types::MessageRole::Tool => {
                messages.extend(convert_tool_messages(&message.content)?);
            }
        }
    }

    Ok((messages, documents))
}

#[allow(unreachable_patterns)]
fn text_only_message_content(
    content: &MessageContent,
    role_label: &str,
) -> Result<String, LlmError> {
    match content {
        MessageContent::Text(text) => Ok(text.clone()),
        MessageContent::MultiModal(parts) => {
            let mut text = String::new();
            for part in parts {
                match part {
                    ContentPart::Text { text: value, .. } => text.push_str(value),
                    _ => {
                        return Err(LlmError::UnsupportedOperation(format!(
                            "Cohere {role_label} messages only support text content"
                        )));
                    }
                }
            }
            Ok(text)
        }
        _ => Ok(content.all_text()),
    }
}

#[allow(unreachable_patterns)]
fn convert_user_content(
    content: &MessageContent,
    documents: &mut Vec<Value>,
) -> Result<String, LlmError> {
    match content {
        MessageContent::Text(text) => Ok(text.clone()),
        MessageContent::MultiModal(parts) => {
            let mut text = String::new();
            for part in parts {
                match part {
                    ContentPart::Text { text: value, .. } => text.push_str(value),
                    ContentPart::File {
                        source,
                        media_type,
                        filename,
                        ..
                    } => {
                        let document_text = shared::decode_text_media(source, media_type)?;
                        let mut data = serde_json::Map::new();
                        data.insert("text".to_string(), Value::String(document_text));
                        if let Some(filename) = filename.as_ref()
                            && !filename.is_empty()
                        {
                            data.insert("title".to_string(), Value::String(filename.clone()));
                        }
                        documents.push(json!({ "data": data }));
                    }
                    ContentPart::Reasoning { .. } => {}
                    ContentPart::Image { .. } | ContentPart::Audio { .. } => {
                        return Err(LlmError::UnsupportedOperation(
                            "Cohere user messages do not support image or audio parts".to_string(),
                        ));
                    }
                    other => {
                        return Err(LlmError::UnsupportedOperation(format!(
                            "Cohere user messages do not support {:?} content parts",
                            other
                        )));
                    }
                }
            }
            Ok(text)
        }
        _ => Ok(content.all_text()),
    }
}

#[allow(unreachable_patterns)]
fn convert_assistant_content(content: &MessageContent) -> Result<(String, Vec<Value>), LlmError> {
    match content {
        MessageContent::Text(text) => Ok((text.clone(), Vec::new())),
        MessageContent::MultiModal(parts) => {
            let mut text = String::new();
            let mut tool_calls = Vec::new();

            for part in parts {
                match part {
                    ContentPart::Text { text: value, .. } => text.push_str(value),
                    ContentPart::ToolCall {
                        tool_call_id,
                        tool_name,
                        arguments,
                        ..
                    } => {
                        tool_calls.push(json!({
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": serde_json::to_string(arguments).map_err(|error| {
                                    LlmError::ParseError(format!(
                                        "Failed to serialize Cohere assistant tool call arguments: {error}"
                                    ))
                                })?,
                            }
                        }));
                    }
                    ContentPart::Reasoning { .. }
                    | ContentPart::Source { .. }
                    | ContentPart::Custom { .. }
                    | ContentPart::ToolApprovalResponse { .. }
                    | ContentPart::ToolApprovalRequest { .. } => {}
                    other => {
                        return Err(LlmError::UnsupportedOperation(format!(
                            "Cohere assistant messages do not support {:?} content parts",
                            other
                        )));
                    }
                }
            }

            Ok((text, tool_calls))
        }
        _ => Ok((content.all_text(), Vec::new())),
    }
}

#[allow(unreachable_patterns)]
fn convert_tool_messages(content: &MessageContent) -> Result<Vec<Value>, LlmError> {
    let mut messages = Vec::new();
    let parts = match content {
        MessageContent::Text(text) => {
            return Err(LlmError::UnsupportedOperation(format!(
                "Cohere tool messages require tool-result parts, got plain text: {text}"
            )));
        }
        MessageContent::MultiModal(parts) => parts,
        _ => {
            return Err(LlmError::UnsupportedOperation(
                "Cohere tool messages require tool-result parts".to_string(),
            ));
        }
    };

    for part in parts {
        match part {
            ContentPart::ToolResult {
                tool_call_id,
                output,
                ..
            } => {
                messages.push(json!({
                    "role": "tool",
                    "content": output.to_string_lossy(),
                    "tool_call_id": tool_call_id,
                }));
            }
            ContentPart::ToolApprovalResponse { .. } => {}
            _ => {
                return Err(LlmError::UnsupportedOperation(
                    "Cohere tool messages only support tool-result parts".to_string(),
                ));
            }
        }
    }

    Ok(messages)
}

fn prepare_tools(
    tools: Option<&[Tool]>,
    tool_choice: Option<&ToolChoice>,
) -> (Vec<Value>, Option<String>) {
    let mut prepared_tools = Vec::new();

    if let Some(tools) = tools {
        for tool in tools {
            match tool {
                Tool::Function { function } => prepared_tools.push(json!({
                    "type": "function",
                    "function": {
                        "name": function.name,
                        "description": function.description,
                        "parameters": function.parameters,
                    }
                })),
                Tool::ProviderDefined(_) => {}
            }
        }
    }

    match tool_choice {
        Some(ToolChoice::None) => (prepared_tools, Some("NONE".to_string())),
        Some(ToolChoice::Required) => (prepared_tools, Some("REQUIRED".to_string())),
        Some(ToolChoice::Tool { name }) => (
            prepared_tools
                .into_iter()
                .filter(|tool| {
                    tool.get("function")
                        .and_then(|function| function.get("name"))
                        .and_then(Value::as_str)
                        == Some(name.as_str())
                })
                .collect(),
            Some("REQUIRED".to_string()),
        ),
        _ => (prepared_tools, None),
    }
}

fn map_response_format(format: &ResponseFormat) -> Option<Value> {
    match format {
        ResponseFormat::JsonObject { .. } => Some(json!({
            "type": "json_object",
        })),
        ResponseFormat::Json { schema, .. } => Some(json!({
            "type": "json_object",
            "json_schema": schema,
        })),
    }
}

fn map_thinking_config(config: &crate::provider_options::CohereThinkingConfig) -> Option<Value> {
    let mut value = serde_json::Map::new();
    if let Some(thinking_type) = config.thinking_type {
        let thinking_type = match thinking_type {
            CohereThinkingType::Enabled => "enabled",
            CohereThinkingType::Disabled => "disabled",
        };
        value.insert("type".to_string(), Value::String(thinking_type.to_string()));
    }
    if let Some(token_budget) = config.token_budget {
        value.insert("token_budget".to_string(), json!(token_budget));
    }

    (!value.is_empty()).then_some(Value::Object(value))
}

fn parse_tool_call_input(arguments: &str) -> Result<Value, LlmError> {
    let normalized = normalize_tool_call_arguments(arguments);
    serde_json::from_str(normalized).map_err(|error| {
        LlmError::ParseError(format!(
            "Failed to parse Cohere tool call arguments: {error}"
        ))
    })
}

fn tool_call_input_string(arguments: &str) -> Result<String, LlmError> {
    let parsed = parse_tool_call_input(arguments)?;
    serde_json::to_string(&parsed).map_err(|error| {
        LlmError::ParseError(format!(
            "Failed to serialize normalized Cohere tool call arguments: {error}"
        ))
    })
}

fn normalize_tool_call_arguments(arguments: &str) -> &str {
    let trimmed = arguments.trim();
    if trimmed.is_empty() || trimmed == "null" {
        "{}"
    } else {
        trimmed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, Tool};
    use eventsource_stream::Event;

    fn sse_event(value: serde_json::Value) -> Event {
        Event {
            event: String::new(),
            data: value.to_string(),
            id: String::new(),
            retry: None,
        }
    }

    #[test]
    fn cohere_maps_schema_less_json_response_format() {
        let mapped = map_response_format(&ResponseFormat::json_object())
            .expect("schema-less JSON response format");

        assert_eq!(
            mapped,
            json!({
                "type": "json_object",
            })
        );
    }

    #[test]
    fn cohere_maps_json_schema_response_format() {
        let schema = json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                }
            },
            "required": ["text"],
        });

        let mapped = map_response_format(&ResponseFormat::json_schema(schema.clone()))
            .expect("JSON schema response format");

        assert_eq!(
            mapped,
            json!({
                "type": "json_object",
                "json_schema": schema,
            })
        );
    }

    #[test]
    fn cohere_non_stream_response_keeps_provider_defined_tool_warning() {
        let request = ChatRequest::new(vec![ChatMessage::user("hello").build()]).with_tools(vec![
            Tool::provider_defined("openai.web_search", "web_search"),
        ]);
        let transformers = CohereChatStandard::new().create_transformers("cohere", &request);

        let response = transformers
            .response
            .transform_chat_response(&json!({
                "generation_id": "gen-1",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "hello from cohere"
                        }
                    ]
                },
                "finish_reason": "COMPLETE",
                "usage": {
                    "tokens": {
                        "input_tokens": 3,
                        "output_tokens": 5
                    }
                }
            }))
            .expect("transform response");

        assert_eq!(response.content_text(), Some("hello from cohere"));
        assert_eq!(
            response.warnings,
            Some(vec![Warning::unsupported(
                "provider-defined tool openai.web_search",
                None::<String>,
            )])
        );
    }

    #[test]
    fn cohere_non_stream_response_preserves_raw_response_body() {
        let mut request = ChatRequest::new(vec![ChatMessage::user("hello").build()]);
        request.common_params.model = "command-a-03-2025".to_string();
        let transformers = CohereChatStandard::new().create_transformers("cohere", &request);

        let response = transformers
            .response
            .transform_chat_response(&json!({
                "generation_id": "gen-body",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "hello from cohere"
                        }
                    ]
                },
                "finish_reason": "COMPLETE",
                "usage": {
                    "tokens": {
                        "input_tokens": 3,
                        "output_tokens": 5
                    }
                }
            }))
            .expect("transform response");
        let response_info = response.response.expect("response metadata");

        assert_eq!(response_info.model_id.as_deref(), Some("command-a-03-2025"));
        assert!(response_info.headers.is_empty());
        assert_eq!(
            response_info.body.as_ref().expect("raw body")["generation_id"],
            "gen-body"
        );
    }

    #[test]
    fn cohere_non_stream_response_maps_citation_metadata_like_ai_sdk() {
        let request = ChatRequest::new(vec![ChatMessage::user("hello").build()]);
        let transformers = CohereChatStandard::new().create_transformers("cohere", &request);

        let response = transformers
            .response
            .transform_chat_response(&json!({
                "generation_id": "gen-citation",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Automation of tasks"
                        }
                    ],
                    "citations": [
                        {
                            "start": 0,
                            "end": 19,
                            "text": "Automation of tasks",
                            "sources": [
                                {
                                    "type": "document",
                                    "id": "doc:0",
                                    "document": {
                                        "id": "doc:0",
                                        "text": "AI provides automation of tasks",
                                        "title": "benefits.txt"
                                    }
                                }
                            ],
                            "type": "TEXT_CONTENT"
                        }
                    ]
                },
                "finish_reason": "COMPLETE",
                "usage": {
                    "tokens": {
                        "input_tokens": 3,
                        "output_tokens": 5
                    }
                }
            }))
            .expect("transform response");

        let source_meta = match &response.content {
            MessageContent::MultiModal(parts) => parts.iter().find_map(|part| match part {
                ContentPart::Source {
                    provider_metadata,
                    source,
                    ..
                } => {
                    assert_eq!(source.title(), Some("benefits.txt"));
                    provider_metadata.as_ref()
                }
                _ => None,
            }),
            _ => None,
        }
        .and_then(|metadata| metadata.get("cohere"))
        .expect("cohere citation metadata");

        assert_eq!(source_meta["start"], json!(0));
        assert_eq!(source_meta["end"], json!(19));
        assert_eq!(source_meta["text"], json!("Automation of tasks"));
        assert_eq!(source_meta["citationType"], json!("TEXT_CONTENT"));
        assert!(source_meta.get("type").is_none());
        assert_eq!(
            source_meta["sources"][0]["document"]["title"],
            json!("benefits.txt")
        );
    }

    #[tokio::test]
    async fn cohere_stream_emits_stable_stream_start_raw_and_warning_carryover() {
        let request = ChatRequest::new(vec![ChatMessage::user("hello").build()])
            .with_tools(vec![Tool::provider_defined(
                "openai.web_search",
                "web_search",
            )])
            .with_include_raw_chunks(true);
        let transformers = CohereChatStandard::new().create_transformers("cohere", &request);
        let stream = transformers.stream.expect("stream transformer");

        let start_events = stream
            .convert_event(sse_event(json!({
                "type": "message-start",
                "id": "gen-stream-1"
            })))
            .await;

        assert!(matches!(
            start_events.first(),
            Some(Ok(ChatStreamEvent::StreamStart { metadata }))
                if metadata.id.as_deref() == Some("gen-stream-1")
        ));
        assert!(matches!(
            start_events.get(1),
            Some(Ok(ChatStreamEvent::Part {
                part: ChatStreamPart::StreamStart { warnings }
            })) if warnings == &vec![Warning::unsupported(
                "provider-defined tool openai.web_search",
                None::<String>,
            )]
        ));
        assert!(matches!(
            start_events.get(2),
            Some(Ok(ChatStreamEvent::Part {
                part: ChatStreamPart::Raw { raw_value }
            })) if raw_value["type"] == json!("message-start")
        ));
        assert!(matches!(
            start_events.get(3),
            Some(Ok(ChatStreamEvent::Part {
                part: ChatStreamPart::ResponseMetadata(metadata)
            })) if metadata.id.as_deref() == Some("gen-stream-1")
        ));

        let end_events = stream
            .convert_event(sse_event(json!({
                "type": "message-end",
                "delta": {
                    "finish_reason": "STOP_SEQUENCE",
                    "usage": {
                        "tokens": {
                            "input_tokens": 4,
                            "output_tokens": 6
                        }
                    }
                }
            })))
            .await;
        assert!(matches!(
            end_events.first(),
            Some(Ok(ChatStreamEvent::Part {
                part: ChatStreamPart::Raw { raw_value }
            })) if raw_value["type"] == json!("message-end")
        ));

        let flush_events = stream.handle_stream_end_events();
        assert!(flush_events.iter().any(|event| {
            matches!(
                event,
                Ok(ChatStreamEvent::Part {
                    part: ChatStreamPart::Finish { finish_reason, .. }
                }) if finish_reason.raw.as_deref() == Some("STOP_SEQUENCE")
            )
        }));
        assert!(flush_events.iter().any(|event| {
            matches!(
                event,
                Ok(ChatStreamEvent::StreamEnd { response })
                    if response.raw_finish_reason.as_deref() == Some("STOP_SEQUENCE")
                        && response.warnings == Some(vec![Warning::unsupported(
                            "provider-defined tool openai.web_search",
                            None::<String>,
                )])
            )
        }));
    }

    #[tokio::test]
    async fn cohere_parse_error_emits_stream_start_before_raw_and_does_not_duplicate_later() {
        let request = ChatRequest::new(vec![ChatMessage::user("hello").build()])
            .with_include_raw_chunks(true);
        let transformers = CohereChatStandard::new().create_transformers("cohere", &request);
        let stream = transformers.stream.expect("stream transformer");

        let invalid = stream
            .convert_event(Event {
                event: "".to_string(),
                data: "{ not json".to_string(),
                id: "".to_string(),
                retry: None,
            })
            .await;

        assert_eq!(invalid.len(), 4);
        assert!(matches!(
            invalid.first(),
            Some(Ok(ChatStreamEvent::StreamStart { metadata }))
                if metadata.provider == "cohere"
        ));
        assert!(matches!(
            invalid.get(1),
            Some(Ok(ChatStreamEvent::Part {
                part: ChatStreamPart::StreamStart { warnings }
            })) if warnings.is_empty()
        ));
        assert!(matches!(
            invalid.get(2),
            Some(Ok(ChatStreamEvent::Part {
                part: ChatStreamPart::Raw { raw_value }
            })) if raw_value == &Value::String("{ not json".to_string())
        ));
        assert!(matches!(
            invalid.get(3),
            Some(Err(LlmError::ParseError(message)))
                if message.contains("Failed to parse Cohere stream event JSON")
        ));

        let later_message_start = stream
            .convert_event(sse_event(json!({
                "type": "message-start",
                "id": "gen-stream-after-error"
            })))
            .await;

        assert!(!later_message_start.iter().any(|event| matches!(
            event,
            Ok(ChatStreamEvent::StreamStart { .. })
                | Ok(ChatStreamEvent::Part {
                    part: ChatStreamPart::StreamStart { .. }
                })
        )));
        assert!(matches!(
            later_message_start.first(),
            Some(Ok(ChatStreamEvent::Part {
                part: ChatStreamPart::Raw { raw_value }
            })) if raw_value["type"] == json!("message-start")
        ));
        assert!(matches!(
            later_message_start.get(1),
            Some(Ok(ChatStreamEvent::Part {
                part: ChatStreamPart::ResponseMetadata(metadata)
            })) if metadata.id.as_deref() == Some("gen-stream-after-error")
        ));
    }
}
