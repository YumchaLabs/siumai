use std::collections::BTreeMap;

use serde_json::{Map, Value};
use siumai_core::{
    ContentPart, DEFAULT_OPAQUE_ITEM_LIMIT, DEFAULT_TOOL_INPUT_BYTE_LIMIT, DecoderLifecycle, Error,
    ErrorKind, ExecutionOwner, LanguageStreamDecoder, LanguageStreamEvent, ModelId,
    PartialLanguageOutput, PartialLanguageOutputPart, ProviderScope, ResponseDiagnostics,
    StreamTerminal, Usage, UsageUpdate, UsageValue,
};

use super::MessagesCodecError;
use super::error::classify_stream_failure;
use super::response::{
    StreamResponseParts, build_stream_response, decode_content_block, decode_refusal_reason,
    decode_usage,
};
use super::wire::{StreamEventWire, StreamMessageDeltaWire, StreamMessageStartWire, UsageWire};

/// Incremental decoder for already-framed Anthropic Messages SSE data.
///
/// Transport owns SSE framing and passes each `data` payload to [`decode`](LanguageStreamDecoder::decode).
/// The explicit `message_stop` event is the only successful protocol terminal;
/// clean EOF before it is an unexpected-EOF failure.
#[derive(Debug)]
pub struct MessagesStreamDecoder {
    scope: ProviderScope,
    requested_model: ModelId,
    lifecycle: DecoderLifecycle,
    started: bool,
    message_id: Option<String>,
    response_model: Option<ModelId>,
    usage: UsageWire,
    stop_reason: Option<String>,
    stop_sequence: Option<String>,
    stop_details: Option<Value>,
    provider_metadata: BTreeMap<String, Value>,
    refusal_event_emitted: bool,
    active_blocks: BTreeMap<u64, ActiveBlock>,
    completed_blocks: BTreeMap<u64, Vec<ContentPart>>,
    response_diagnostics: ResponseDiagnostics,
}

impl MessagesStreamDecoder {
    pub fn new(scope: ProviderScope, requested_model: ModelId) -> Self {
        Self {
            scope,
            requested_model,
            lifecycle: DecoderLifecycle::default(),
            started: false,
            message_id: None,
            response_model: None,
            usage: UsageWire::default(),
            stop_reason: None,
            stop_sequence: None,
            stop_details: None,
            provider_metadata: BTreeMap::new(),
            refusal_event_emitted: false,
            active_blocks: BTreeMap::new(),
            completed_blocks: BTreeMap::new(),
            response_diagnostics: ResponseDiagnostics::default(),
        }
    }

    pub fn with_response_diagnostics(mut self, diagnostics: ResponseDiagnostics) -> Self {
        self.response_diagnostics = diagnostics;
        self
    }

    fn decode_frame(&mut self, frame: &str) -> Result<Vec<LanguageStreamEvent>, Error> {
        if frame.trim() == "[DONE]" {
            return Err(MessagesCodecError::UnexpectedEof.into());
        }
        let envelope = serde_json::from_str::<Value>(frame)
            .map_err(MessagesCodecError::JsonDecode)
            .map_err(Error::from)?;
        let event = serde_json::from_value::<StreamEventWire>(envelope.clone())
            .map_err(MessagesCodecError::JsonDecode)
            .map_err(Error::from)?;
        match event.kind.as_str() {
            "message_start" => self.message_start(&event).map_err(Error::from),
            "content_block_start" => self.content_block_start(&event).map_err(Error::from),
            "content_block_delta" => self.content_block_delta(&event).map_err(Error::from),
            "content_block_stop" => self.content_block_stop(&event).map_err(Error::from),
            "message_delta" => self.message_delta(&event).map_err(Error::from),
            "message_stop" => self.message_stop().map_err(Error::from),
            "ping" => Ok(Vec::new()),
            "error" => {
                let error = classify_stream_failure(&envelope, self.response_diagnostics.clone())
                    .map_err(Error::from)?;
                let partial = self.partial_output();
                let terminal = if error.kind() == ErrorKind::Cancelled {
                    StreamTerminal::Cancelled {
                        reason: error.message().to_string(),
                        partial,
                    }
                } else {
                    StreamTerminal::Failed { error, partial }
                };
                Ok(vec![LanguageStreamEvent::Terminal(terminal)])
            }
            _ => Err(MessagesCodecError::Unsupported {
                feature: "this Anthropic Messages stream event",
            }
            .into()),
        }
    }

    fn message_start(
        &mut self,
        event: &StreamEventWire,
    ) -> Result<Vec<LanguageStreamEvent>, MessagesCodecError> {
        if self.started {
            return Err(MessagesCodecError::ProtocolViolation {
                reason: "stream emitted message_start more than once",
            });
        }
        let message = required_field(event, "message")?;
        let message = serde_json::from_value::<StreamMessageStartWire>(message.clone())
            .map_err(MessagesCodecError::JsonDecode)?;
        if message.kind != "message" || message.role != "assistant" || message.id.trim().is_empty()
        {
            return Err(MessagesCodecError::ProtocolViolation {
                reason: "message_start did not contain an assistant message identity",
            });
        }
        let model = message.model.map_or_else(
            || Ok(self.requested_model.clone()),
            |model| ModelId::new(model).map_err(MessagesCodecError::InvalidModelId),
        )?;
        self.started = true;
        self.message_id = Some(message.id.clone());
        self.response_model = Some(model.clone());
        self.usage.merge(message.usage);
        self.provider_metadata.extend(message.extra);
        if let Some(stop_sequence) = message.stop_sequence {
            self.stop_sequence = Some(stop_sequence);
        }
        if let Some(stop_details) = message.stop_details {
            decode_refusal_reason(Some(&stop_details))?;
            self.stop_details = Some(stop_details);
        }
        if let Some(stop_reason) = message.stop_reason {
            self.stop_reason = Some(stop_reason);
        }
        Ok(vec![LanguageStreamEvent::Started {
            id: Some(message.id),
            model: Some(model),
        }])
    }

    fn content_block_start(
        &mut self,
        event: &StreamEventWire,
    ) -> Result<Vec<LanguageStreamEvent>, MessagesCodecError> {
        self.ensure_started()?;
        let index = required_u64(event, "index")?;
        if self.active_blocks.contains_key(&index) || self.completed_blocks.contains_key(&index) {
            return Err(MessagesCodecError::ProtocolViolation {
                reason: "content block index was reused",
            });
        }
        let value = required_field(event, "content_block")?.clone();
        if let Some(model) = fallback_target_model(&value)? {
            self.response_model = Some(model);
        }
        let block = ActiveBlock::new(index, value)?;
        let events = block.start_events();
        self.active_blocks.insert(index, block);
        Ok(events)
    }

    fn content_block_delta(
        &mut self,
        event: &StreamEventWire,
    ) -> Result<Vec<LanguageStreamEvent>, MessagesCodecError> {
        self.ensure_started()?;
        let index = required_u64(event, "index")?;
        let delta = required_field(event, "delta")?;
        let block =
            self.active_blocks
                .get_mut(&index)
                .ok_or(MessagesCodecError::ProtocolViolation {
                    reason: "content delta referenced an unknown block",
                })?;
        block.apply_delta(delta)
    }

    fn content_block_stop(
        &mut self,
        event: &StreamEventWire,
    ) -> Result<Vec<LanguageStreamEvent>, MessagesCodecError> {
        self.ensure_started()?;
        let index = required_u64(event, "index")?;
        let block =
            self.active_blocks
                .remove(&index)
                .ok_or(MessagesCodecError::ProtocolViolation {
                    reason: "content stop referenced an unknown block",
                })?;
        let ending = block.ending_event();
        let wire_block = block.into_value()?;
        let model = self
            .response_model
            .as_ref()
            .ok_or(MessagesCodecError::ProtocolViolation {
                reason: "content block completed before message identity",
            })?;
        let parts = decode_content_block(&wire_block, &self.scope, model)?;
        let mut events = Vec::new();
        for part in &parts {
            match part {
                ContentPart::ToolCall(call) => {
                    events.push(LanguageStreamEvent::ToolCall(call.clone()))
                }
                ContentPart::Citation(citation) => {
                    events.push(LanguageStreamEvent::Citation(citation.clone()))
                }
                ContentPart::Refusal { reason } => events.push(LanguageStreamEvent::Refusal {
                    reason: reason.clone(),
                }),
                ContentPart::ProviderOpaque(item) => {
                    events.push(LanguageStreamEvent::ProviderOpaque(item.clone()))
                }
                _ => {}
            }
        }
        if let Some(event) = ending {
            events.push(event);
        }
        self.completed_blocks.insert(index, parts);
        Ok(events)
    }

    fn message_delta(
        &mut self,
        event: &StreamEventWire,
    ) -> Result<Vec<LanguageStreamEvent>, MessagesCodecError> {
        self.ensure_started()?;
        let delta = serde_json::from_value::<StreamMessageDeltaWire>(
            required_field(event, "delta")?.clone(),
        )
        .map_err(MessagesCodecError::JsonDecode)?;
        for (key, value) in &event.fields {
            if key != "delta" && key != "usage" {
                self.provider_metadata.insert(key.clone(), value.clone());
            }
        }
        self.provider_metadata.extend(delta.extra);
        if let Some(stop_reason) = delta.stop_reason {
            if self
                .stop_reason
                .as_deref()
                .is_some_and(|existing| existing != stop_reason.as_str())
            {
                return Err(MessagesCodecError::ProtocolViolation {
                    reason: "stream changed its stop reason",
                });
            }
            let is_refusal = stop_reason == "refusal";
            self.stop_reason = Some(stop_reason);
            if is_refusal {
                // A refusal invalidates all content accumulated before the
                // terminal delta. The final canonical response is rebuilt as
                // a refusal rather than a successful answer.
                self.completed_blocks.clear();
            }
        }
        if let Some(stop_sequence) = delta.stop_sequence {
            self.stop_sequence = Some(stop_sequence);
        }
        let mut events = Vec::new();
        if let Some(stop_details) = delta.stop_details {
            let refusal_reason = decode_refusal_reason(Some(&stop_details))?;
            self.stop_details = Some(stop_details);
            if self.stop_reason.as_deref() == Some("refusal") && !self.refusal_event_emitted {
                self.refusal_event_emitted = true;
                events.push(LanguageStreamEvent::Refusal {
                    reason: refusal_reason,
                });
            }
        } else if self.stop_reason.as_deref() == Some("refusal") && !self.refusal_event_emitted {
            let refusal_reason = decode_refusal_reason(self.stop_details.as_ref())?;
            self.refusal_event_emitted = true;
            events.push(LanguageStreamEvent::Refusal {
                reason: refusal_reason,
            });
        }
        if let Some(usage) = event.field("usage") {
            let usage = serde_json::from_value::<UsageWire>(usage.clone())
                .map_err(MessagesCodecError::JsonDecode)?;
            self.usage.merge(usage);
        }
        Ok(events)
    }

    fn message_stop(&mut self) -> Result<Vec<LanguageStreamEvent>, MessagesCodecError> {
        self.ensure_started()?;
        if !self.active_blocks.is_empty() {
            return Err(MessagesCodecError::ProtocolViolation {
                reason: "message_stop arrived before all content blocks stopped",
            });
        }
        let id = self
            .message_id
            .take()
            .ok_or(MessagesCodecError::ProtocolViolation {
                reason: "message_stop omitted message identity",
            })?;
        let model = self
            .response_model
            .take()
            .ok_or(MessagesCodecError::ProtocolViolation {
                reason: "message_stop omitted model identity",
            })?;
        let content = std::mem::take(&mut self.completed_blocks)
            .into_values()
            .flatten()
            .collect();
        let refusal_reason = if self.stop_reason.as_deref() == Some("refusal") {
            decode_refusal_reason(self.stop_details.as_ref())?
        } else {
            None
        };
        let mut events = Vec::new();
        if self.stop_reason.as_deref() == Some("refusal") && !self.refusal_event_emitted {
            self.refusal_event_emitted = true;
            events.push(LanguageStreamEvent::Refusal {
                reason: refusal_reason,
            });
        }
        let response = build_stream_response(StreamResponseParts {
            id,
            model,
            stop_reason: self.stop_reason.as_deref(),
            stop_sequence: self.stop_sequence.take(),
            stop_details: self.stop_details.take(),
            content,
            usage_wire: &self.usage,
            provider: std::mem::take(&mut self.provider_metadata),
        })?;
        let usage = response.usage().clone();
        events.push(LanguageStreamEvent::Usage(UsageUpdate::snapshot(usage)));
        events.push(LanguageStreamEvent::Terminal(StreamTerminal::Completed {
            response: Box::new(response),
        }));
        Ok(events)
    }

    fn ensure_started(&self) -> Result<(), MessagesCodecError> {
        if self.started {
            Ok(())
        } else {
            Err(MessagesCodecError::ProtocolViolation {
                reason: "stream content arrived before message_start",
            })
        }
    }

    fn partial_output(&self) -> Option<PartialLanguageOutput> {
        let mut by_index = BTreeMap::<u64, Vec<PartialLanguageOutputPart>>::new();
        for (index, parts) in &self.completed_blocks {
            let parts = parts.iter().filter_map(partial_part).collect::<Vec<_>>();
            if !parts.is_empty() {
                by_index.insert(*index, parts);
            }
        }
        for (index, block) in &self.active_blocks {
            let parts = block.partial_parts();
            if !parts.is_empty() {
                by_index.insert(*index, parts);
            }
        }
        let content = by_index.into_values().flatten().collect::<Vec<_>>();
        let usage = decode_usage(&self.usage);
        if content.is_empty() && !usage_observed(&usage) {
            return None;
        }
        PartialLanguageOutput::new(content, usage).ok()
    }
}

fn partial_part(part: &ContentPart) -> Option<PartialLanguageOutputPart> {
    match part {
        ContentPart::Text { text } => Some(PartialLanguageOutputPart::Text { text: text.clone() }),
        ContentPart::Reasoning { text } => {
            Some(PartialLanguageOutputPart::Reasoning { text: text.clone() })
        }
        ContentPart::Refusal { reason } => Some(PartialLanguageOutputPart::Refusal {
            reason: reason.clone(),
        }),
        ContentPart::Media(_)
        | ContentPart::Citation(_)
        | ContentPart::ToolCall(_)
        | ContentPart::ToolResult(_)
        | ContentPart::ProviderOpaque(_) => None,
        _ => None,
    }
}

fn usage_observed(usage: &Usage) -> bool {
    [
        usage.input_tokens,
        usage.output_tokens,
        usage.total_tokens,
        usage.reasoning_tokens,
        usage.cache_read_tokens,
        usage.cache_write_tokens,
        usage.audio_input_tokens,
        usage.audio_output_tokens,
        usage.orchestration_tokens,
    ]
    .into_iter()
    .any(|value| matches!(value, UsageValue::Known(_)))
}

fn fallback_target_model(value: &Value) -> Result<Option<ModelId>, MessagesCodecError> {
    let Some(object) = value.as_object() else {
        return Ok(None);
    };
    if object.get("type").and_then(Value::as_str) != Some("fallback") {
        return Ok(None);
    }
    let target = object.get("to").and_then(Value::as_object).ok_or(
        MessagesCodecError::ProtocolViolation {
            reason: "fallback block omitted its target model",
        },
    )?;
    let Some(model) = target.get("model").and_then(Value::as_str) else {
        return Err(MessagesCodecError::ProtocolViolation {
            reason: "fallback block omitted its target model",
        });
    };
    ModelId::new(model)
        .map(Some)
        .map_err(MessagesCodecError::InvalidModelId)
}

impl LanguageStreamDecoder for MessagesStreamDecoder {
    type ProtocolFrame = str;

    fn set_response_diagnostics(&mut self, diagnostics: ResponseDiagnostics) {
        self.response_diagnostics = diagnostics;
    }

    fn decode(&mut self, frame: &Self::ProtocolFrame) -> Result<Vec<LanguageStreamEvent>, Error> {
        self.lifecycle
            .ensure_decode_allowed()
            .map_err(Error::from)?;
        let events = self.decode_frame(frame)?;
        self.lifecycle.record(&events).map_err(Error::from)?;
        Ok(events)
    }

    fn finish(&mut self) -> Result<Vec<LanguageStreamEvent>, Error> {
        if self.lifecycle.begin_finish().map_err(Error::from)? {
            Ok(Vec::new())
        } else {
            Err(MessagesCodecError::UnexpectedEof.into())
        }
    }

    fn terminal_seen(&self) -> bool {
        self.lifecycle.terminal_seen()
    }
}

#[derive(Debug)]
enum ActiveBlock {
    Text {
        index: u64,
        object: Map<String, Value>,
        text: String,
    },
    Thinking {
        index: u64,
        object: Map<String, Value>,
        thinking: String,
        signature: String,
    },
    Compaction {
        object: Map<String, Value>,
        content: Option<String>,
        delta_seen: bool,
    },
    ToolUse {
        object: Map<String, Value>,
        id: String,
        name: String,
        initial_input: Value,
        partial_input: String,
    },
    Static {
        object: Map<String, Value>,
    },
}

impl ActiveBlock {
    fn new(index: u64, value: Value) -> Result<Self, MessagesCodecError> {
        let object = value
            .as_object()
            .cloned()
            .ok_or(MessagesCodecError::ProtocolViolation {
                reason: "content_block_start contained a non-object block",
            })?;
        let kind = object.get("type").and_then(Value::as_str).ok_or(
            MessagesCodecError::ProtocolViolation {
                reason: "content_block_start omitted its block type",
            },
        )?;
        match kind {
            "text" => Ok(Self::Text {
                index,
                text: object
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                object,
            }),
            "thinking" => Ok(Self::Thinking {
                index,
                thinking: object
                    .get("thinking")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                signature: object
                    .get("signature")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                object,
            }),
            "compaction" => {
                let content = match object.get("content") {
                    Some(Value::Null) => None,
                    Some(Value::String(content)) => {
                        ensure_compaction_content_bound(content.len())?;
                        Some(content.clone())
                    }
                    Some(_) => {
                        return Err(MessagesCodecError::ProtocolViolation {
                            reason: "streamed compaction block content was not a string",
                        });
                    }
                    None => {
                        return Err(MessagesCodecError::ProtocolViolation {
                            reason: "streamed compaction block omitted its content",
                        });
                    }
                };
                Ok(Self::Compaction {
                    object,
                    content,
                    delta_seen: false,
                })
            }
            "tool_use" => {
                let id = required_object_string(&object, "id")?.to_string();
                let name = required_object_string(&object, "name")?.to_string();
                let initial_input = object.get("input").cloned().unwrap_or(Value::Null);
                let initial_bytes = serde_json::to_vec(&initial_input)
                    .map_err(MessagesCodecError::JsonEncode)?
                    .len();
                if initial_bytes > DEFAULT_TOOL_INPUT_BYTE_LIMIT {
                    return Err(MessagesCodecError::ToolInputTooLarge {
                        maximum: DEFAULT_TOOL_INPUT_BYTE_LIMIT,
                    });
                }
                Ok(Self::ToolUse {
                    object,
                    id,
                    name,
                    initial_input,
                    partial_input: String::new(),
                })
            }
            _ => Ok(Self::Static { object }),
        }
    }

    fn start_events(&self) -> Vec<LanguageStreamEvent> {
        match self {
            Self::Text { index, text, .. } => {
                let id = block_id("text", *index);
                let mut events = vec![LanguageStreamEvent::TextStart { id: id.clone() }];
                if !text.is_empty() {
                    events.push(LanguageStreamEvent::TextDelta {
                        id,
                        delta: text.clone(),
                    });
                }
                events
            }
            Self::Thinking {
                index, thinking, ..
            } => {
                let id = block_id("thinking", *index);
                let mut events = vec![LanguageStreamEvent::ReasoningStart { id: id.clone() }];
                if !thinking.is_empty() {
                    events.push(LanguageStreamEvent::ReasoningDelta {
                        id,
                        delta: thinking.clone(),
                    });
                }
                events
            }
            Self::Compaction { .. } => Vec::new(),
            Self::ToolUse { id, name, .. } => vec![LanguageStreamEvent::ToolInputStart {
                id: id.clone(),
                name: name.clone(),
                owner: ExecutionOwner::Local,
            }],
            Self::Static { .. } => Vec::new(),
        }
    }

    fn apply_delta(
        &mut self,
        delta: &Value,
    ) -> Result<Vec<LanguageStreamEvent>, MessagesCodecError> {
        let object = delta
            .as_object()
            .ok_or(MessagesCodecError::ProtocolViolation {
                reason: "content block delta was not an object",
            })?;
        let kind = required_object_string(object, "type")?;
        match (self, kind) {
            (Self::Text { index, text, .. }, "text_delta") => {
                let delta = required_object_string(object, "text")?;
                text.push_str(delta);
                Ok(vec![LanguageStreamEvent::TextDelta {
                    id: block_id("text", *index),
                    delta: delta.to_string(),
                }])
            }
            (Self::Text { object: block, .. }, "citations_delta") => {
                let citation = object.get("citation").cloned().ok_or(
                    MessagesCodecError::ProtocolViolation {
                        reason: "citations_delta omitted its citation",
                    },
                )?;
                let citations = block
                    .entry("citations".to_string())
                    .or_insert_with(|| Value::Array(Vec::new()))
                    .as_array_mut()
                    .ok_or(MessagesCodecError::ProtocolViolation {
                        reason: "streamed text citations were not an array",
                    })?;
                citations.push(citation);
                Ok(Vec::new())
            }
            (
                Self::Thinking {
                    index, thinking, ..
                },
                "thinking_delta",
            ) => {
                let delta = required_object_string(object, "thinking")?;
                thinking.push_str(delta);
                Ok(vec![LanguageStreamEvent::ReasoningDelta {
                    id: block_id("thinking", *index),
                    delta: delta.to_string(),
                }])
            }
            (Self::Thinking { signature, .. }, "signature_delta") => {
                signature.push_str(required_object_string(object, "signature")?);
                Ok(Vec::new())
            }
            (
                Self::Compaction {
                    content,
                    delta_seen,
                    ..
                },
                "compaction_delta",
            ) => {
                if *delta_seen {
                    return Err(MessagesCodecError::ProtocolViolation {
                        reason: "stream emitted compaction_delta more than once",
                    });
                }
                let delta = required_object_string(object, "content")?;
                ensure_compaction_content_bound(delta.len())?;
                if content
                    .as_deref()
                    .is_some_and(|initial| !initial.is_empty() && initial != delta)
                {
                    return Err(MessagesCodecError::ProtocolViolation {
                        reason: "compaction_delta disagreed with the started compaction content",
                    });
                }
                *content = Some(delta.to_string());
                *delta_seen = true;
                Ok(Vec::new())
            }
            (
                Self::ToolUse {
                    id, partial_input, ..
                },
                "input_json_delta",
            ) => {
                let delta = required_object_string(object, "partial_json")?;
                append_tool_input(partial_input, delta)?;
                Ok(vec![LanguageStreamEvent::ToolInputDelta {
                    id: id.clone(),
                    delta: delta.to_string(),
                }])
            }
            (Self::Static { .. }, _) => Err(MessagesCodecError::Unsupported {
                feature: "deltas for this native Anthropic content block",
            }),
            _ => Err(MessagesCodecError::ProtocolViolation {
                reason: "content delta type did not match its active block",
            }),
        }
    }

    fn ending_event(&self) -> Option<LanguageStreamEvent> {
        match self {
            Self::Text { index, .. } => Some(LanguageStreamEvent::TextEnd {
                id: block_id("text", *index),
            }),
            Self::Thinking { index, .. } => Some(LanguageStreamEvent::ReasoningEnd {
                id: block_id("thinking", *index),
            }),
            Self::Compaction { .. } | Self::ToolUse { .. } | Self::Static { .. } => None,
        }
    }

    fn partial_parts(&self) -> Vec<PartialLanguageOutputPart> {
        match self {
            Self::Text { text, .. } if !text.is_empty() => {
                vec![PartialLanguageOutputPart::Text { text: text.clone() }]
            }
            Self::Thinking { thinking, .. } if !thinking.is_empty() => {
                vec![PartialLanguageOutputPart::Reasoning {
                    text: thinking.clone(),
                }]
            }
            Self::Static { object }
                if object.get("type").and_then(Value::as_str) == Some("refusal") =>
            {
                vec![PartialLanguageOutputPart::Refusal {
                    reason: object
                        .get("refusal")
                        .or_else(|| object.get("reason"))
                        .and_then(Value::as_str)
                        .map(ToString::to_string),
                }]
            }
            Self::Text { .. }
            | Self::Thinking { .. }
            | Self::Compaction { .. }
            | Self::ToolUse { .. }
            | Self::Static { .. } => Vec::new(),
        }
    }

    fn into_value(mut self) -> Result<Value, MessagesCodecError> {
        let object = match &mut self {
            Self::Text { object, text, .. } => {
                object.insert("text".to_string(), Value::String(std::mem::take(text)));
                std::mem::take(object)
            }
            Self::Thinking {
                object,
                thinking,
                signature,
                ..
            } => {
                object.insert(
                    "thinking".to_string(),
                    Value::String(std::mem::take(thinking)),
                );
                object.insert(
                    "signature".to_string(),
                    Value::String(std::mem::take(signature)),
                );
                std::mem::take(object)
            }
            Self::Compaction {
                object, content, ..
            } => {
                object.insert(
                    "content".to_string(),
                    content.take().map_or(Value::Null, Value::String),
                );
                std::mem::take(object)
            }
            Self::ToolUse {
                object,
                initial_input,
                partial_input,
                ..
            } => {
                let input = if partial_input.is_empty() {
                    std::mem::take(initial_input)
                } else {
                    if !initial_input.is_null()
                        && !initial_input.as_object().is_some_and(Map::is_empty)
                    {
                        return Err(MessagesCodecError::ProtocolViolation {
                            reason: "tool input mixed a populated start value with JSON deltas",
                        });
                    }
                    serde_json::from_str(partial_input).map_err(MessagesCodecError::JsonDecode)?
                };
                object.insert("input".to_string(), input);
                std::mem::take(object)
            }
            Self::Static { object } => std::mem::take(object),
        };
        Ok(Value::Object(object))
    }
}

fn ensure_compaction_content_bound(actual: usize) -> Result<(), MessagesCodecError> {
    if actual > DEFAULT_OPAQUE_ITEM_LIMIT {
        return Err(MessagesCodecError::ProtocolViolation {
            reason: "streamed compaction content exceeded its retained-state bound",
        });
    }
    Ok(())
}

fn append_tool_input(buffer: &mut String, delta: &str) -> Result<(), MessagesCodecError> {
    if buffer
        .len()
        .checked_add(delta.len())
        .is_none_or(|total| total > DEFAULT_TOOL_INPUT_BYTE_LIMIT)
    {
        return Err(MessagesCodecError::ToolInputTooLarge {
            maximum: DEFAULT_TOOL_INPUT_BYTE_LIMIT,
        });
    }
    buffer.push_str(delta);
    Ok(())
}

fn required_field<'a>(
    event: &'a StreamEventWire,
    field: &'static str,
) -> Result<&'a Value, MessagesCodecError> {
    event
        .field(field)
        .ok_or(MessagesCodecError::ProtocolViolation {
            reason: "stream event omitted a required field",
        })
}

fn required_u64(event: &StreamEventWire, field: &'static str) -> Result<u64, MessagesCodecError> {
    required_field(event, field)?
        .as_u64()
        .ok_or(MessagesCodecError::ProtocolViolation {
            reason: "stream event field was not an unsigned integer",
        })
}

fn required_object_string<'a>(
    object: &'a Map<String, Value>,
    field: &'static str,
) -> Result<&'a str, MessagesCodecError> {
    object
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or(MessagesCodecError::ProtocolViolation {
            reason: "stream object omitted a required string field",
        })
}

fn block_id(kind: &str, index: u64) -> String {
    format!("{kind}-{index}")
}
