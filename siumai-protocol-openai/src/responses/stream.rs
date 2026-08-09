//! Transport-neutral decoder for OpenAI Responses server events.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value};
use siumai_core::{
    DEFAULT_TOOL_INPUT_BYTE_LIMIT, DecoderLifecycle, Error, ErrorKind, ExecutionOwner,
    LanguageStreamDecoder, LanguageStreamEvent, ModelId, ProviderScope, ResponseDiagnostics,
    StreamTerminal, ToolCall,
};

use crate::openai_error::classify_stream_error;

use super::OPENAI_RESPONSES_OPAQUE_KIND;
use super::response::{
    decode_response_wire, decode_usage, failed_response_error, opaque_item, project_citation,
    protocol_error,
};
use super::wire::{
    AnnotationWire, OutputContentPart, OutputItem, ResponseErrorWire, ResponseStatus, ResponseWire,
    StreamEventWire,
};

const DEFAULT_RESPONSES_TURN_EVENT_BYTES_LIMIT: usize = 64 * 1024 * 1024;
const DEFAULT_RESPONSES_TURN_OUTPUT_ITEM_LIMIT: usize = 16_384;

/// Typed classification for one native Responses streaming event.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ResponsesStreamEventKind {
    ResponseCreated,
    ResponseQueued,
    ResponseInProgress,
    OutputItemAdded,
    OutputItemDone,
    OutputTextDelta,
    OutputTextDone,
    RefusalDelta,
    RefusalDone,
    OutputTextAnnotationAdded,
    ReasoningSummaryTextDelta,
    ReasoningSummaryTextDone,
    ReasoningTextDelta,
    ReasoningTextDone,
    FunctionCallArgumentsDelta,
    FunctionCallArgumentsDone,
    CustomToolCallInputDelta,
    CustomToolCallInputDone,
    ResponseCompleted,
    ResponseIncomplete,
    ResponseCancelled,
    ResponseFailed,
    Error,
    Unknown(String),
}

/// Controls which abbreviated terminal response shapes may be reconstructed.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ResponsesTerminalPolicy {
    /// Enforce the verified OpenAI terminal contract.
    #[default]
    Strict,
    /// Permit bounded contractions used by explicitly compatible endpoints.
    Compatible,
}

impl ResponsesStreamEventKind {
    fn from_wire(kind: &str) -> Self {
        match kind {
            "response.created" => Self::ResponseCreated,
            "response.queued" => Self::ResponseQueued,
            "response.in_progress" => Self::ResponseInProgress,
            "response.output_item.added" => Self::OutputItemAdded,
            "response.output_item.done" => Self::OutputItemDone,
            "response.output_text.delta" => Self::OutputTextDelta,
            "response.output_text.done" => Self::OutputTextDone,
            "response.refusal.delta" => Self::RefusalDelta,
            "response.refusal.done" => Self::RefusalDone,
            "response.output_text.annotation.added" => Self::OutputTextAnnotationAdded,
            "response.reasoning_summary_text.delta" => Self::ReasoningSummaryTextDelta,
            "response.reasoning_summary_text.done" => Self::ReasoningSummaryTextDone,
            "response.reasoning_text.delta" => Self::ReasoningTextDelta,
            "response.reasoning_text.done" => Self::ReasoningTextDone,
            "response.function_call_arguments.delta" => Self::FunctionCallArgumentsDelta,
            "response.function_call_arguments.done" => Self::FunctionCallArgumentsDone,
            "response.custom_tool_call_input.delta" => Self::CustomToolCallInputDelta,
            "response.custom_tool_call_input.done" => Self::CustomToolCallInputDone,
            "response.completed" => Self::ResponseCompleted,
            "response.incomplete" => Self::ResponseIncomplete,
            "response.cancelled" => Self::ResponseCancelled,
            "response.failed" => Self::ResponseFailed,
            "error" => Self::Error,
            other => Self::Unknown(other.to_string()),
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::ResponseCompleted
                | Self::ResponseIncomplete
                | Self::ResponseCancelled
                | Self::ResponseFailed
                | Self::Error
        )
    }
}

/// One lossless native Responses event with a typed event kind.
#[derive(Clone, PartialEq)]
pub struct ResponsesStreamEvent {
    kind: ResponsesStreamEventKind,
    wire: StreamEventWire,
    payload: ResponsesStreamEventPayload,
}

#[derive(Clone, PartialEq)]
enum ResponsesStreamEventPayload {
    None,
    Response(Box<ResponseWire>),
    Item(Box<OutputItem>),
    Error(ResponseErrorWire),
    Annotation(AnnotationWire),
}

impl std::fmt::Debug for ResponsesStreamEvent {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResponsesStreamEvent")
            .field("kind", &self.kind)
            .field("sequence_number", &self.wire.sequence_number)
            .field("field_names", &self.wire.fields.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl ResponsesStreamEvent {
    fn decode(wire: StreamEventWire) -> Result<Self, Error> {
        let kind = ResponsesStreamEventKind::from_wire(&wire.kind);
        let payload = match &kind {
            ResponsesStreamEventKind::ResponseCreated
            | ResponsesStreamEventKind::ResponseQueued
            | ResponsesStreamEventKind::ResponseInProgress => wire
                .response()
                .map_err(|source| {
                    Error::new(
                        ErrorKind::Protocol,
                        "OpenAI Responses stream event contained a malformed response resource",
                    )
                    .with_source(source)
                })?
                .map(Box::new)
                .map(ResponsesStreamEventPayload::Response)
                .unwrap_or(ResponsesStreamEventPayload::None),
            ResponsesStreamEventKind::ResponseCompleted
            | ResponsesStreamEventKind::ResponseIncomplete
            | ResponsesStreamEventKind::ResponseCancelled
            | ResponsesStreamEventKind::ResponseFailed => ResponsesStreamEventPayload::None,
            ResponsesStreamEventKind::OutputItemAdded
            | ResponsesStreamEventKind::OutputItemDone => wire
                .item()
                .map_err(|source| {
                    Error::new(
                        ErrorKind::Protocol,
                        "OpenAI Responses stream event contained a malformed output item",
                    )
                    .with_source(source)
                })?
                .map(Box::new)
                .map(ResponsesStreamEventPayload::Item)
                .unwrap_or(ResponsesStreamEventPayload::None),
            ResponsesStreamEventKind::OutputTextAnnotationAdded => wire
                .field("annotation")
                .cloned()
                .map(serde_json::from_value::<AnnotationWire>)
                .transpose()
                .map_err(|source| {
                    Error::new(
                        ErrorKind::Protocol,
                        "OpenAI citation event contained a malformed annotation",
                    )
                    .with_source(source)
                })?
                .map(ResponsesStreamEventPayload::Annotation)
                .unwrap_or(ResponsesStreamEventPayload::None),
            ResponsesStreamEventKind::Error => {
                let envelope = wire.to_value().map_err(|source| {
                    Error::new(
                        ErrorKind::Protocol,
                        "failed to preserve an OpenAI Responses stream event",
                    )
                    .with_source(source)
                })?;
                let error = wire.field("error").cloned().unwrap_or(envelope);
                ResponsesStreamEventPayload::Error(serde_json::from_value(error).map_err(
                    |source| {
                        Error::new(
                            ErrorKind::Protocol,
                            "OpenAI Responses error event was malformed",
                        )
                        .with_source(source)
                    },
                )?)
            }
            _ => ResponsesStreamEventPayload::None,
        };
        Ok(Self {
            kind,
            wire,
            payload,
        })
    }

    pub fn kind(&self) -> &ResponsesStreamEventKind {
        &self.kind
    }

    pub fn kind_str(&self) -> &str {
        &self.wire.kind
    }

    pub fn sequence_number(&self) -> Option<u64> {
        self.wire.sequence_number
    }

    /// Return a fully decoded non-terminal response resource.
    pub fn response_resource(&self) -> Option<&ResponseWire> {
        match &self.payload {
            ResponsesStreamEventPayload::Response(response) => Some(response.as_ref()),
            _ => None,
        }
    }

    /// Return the exact response JSON carried by this provider event.
    pub fn raw_response(&self) -> Option<&Value> {
        self.wire.field("response")
    }

    pub fn item(&self) -> Option<&OutputItem> {
        match &self.payload {
            ResponsesStreamEventPayload::Item(item) => Some(item.as_ref()),
            _ => None,
        }
    }

    pub fn error(&self) -> Option<&ResponseErrorWire> {
        match &self.payload {
            ResponsesStreamEventPayload::Error(error) => Some(error),
            _ => None,
        }
    }

    pub fn annotation(&self) -> Option<&AnnotationWire> {
        match &self.payload {
            ResponsesStreamEventPayload::Annotation(annotation) => Some(annotation),
            _ => None,
        }
    }

    pub fn field(&self, name: &str) -> Option<&Value> {
        self.wire.field(name)
    }

    pub fn item_id(&self) -> Option<&str> {
        self.field("item_id").and_then(Value::as_str)
    }

    pub fn output_index(&self) -> Option<u64> {
        self.field("output_index").and_then(Value::as_u64)
    }

    pub fn content_index(&self) -> Option<u64> {
        self.field("content_index").and_then(Value::as_u64)
    }

    pub fn delta(&self) -> Option<&str> {
        self.field("delta").and_then(Value::as_str)
    }

    /// Explicit access to the complete provider wire event.
    pub fn wire(&self) -> &StreamEventWire {
        &self.wire
    }

    pub fn into_wire(self) -> StreamEventWire {
        self.wire
    }
}

/// A single decoded provider event and every portable event projected from it.
pub struct DecodedResponsesStreamFrame {
    native: ResponsesStreamEvent,
    portable_events: Vec<LanguageStreamEvent>,
}

impl std::fmt::Debug for DecodedResponsesStreamFrame {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DecodedResponsesStreamFrame")
            .field("native", &self.native)
            .field("portable_event_count", &self.portable_events.len())
            .finish()
    }
}

impl DecodedResponsesStreamFrame {
    pub fn native(&self) -> &ResponsesStreamEvent {
        &self.native
    }

    pub fn portable_events(&self) -> &[LanguageStreamEvent] {
        &self.portable_events
    }

    pub fn terminal(&self) -> Option<&StreamTerminal> {
        self.portable_events
            .iter()
            .find_map(LanguageStreamEvent::terminal)
    }

    pub fn is_terminal(&self) -> bool {
        self.terminal().is_some()
    }

    pub fn into_portable_events(self) -> Vec<LanguageStreamEvent> {
        self.portable_events
    }

    pub fn into_parts(self) -> (ResponsesStreamEvent, Vec<LanguageStreamEvent>) {
        (self.native, self.portable_events)
    }
}

/// Stateful, native Responses stream decoder.
///
/// SSE framing is transport-owned. This type consumes one `data` value at a
/// time, preserves native items, and emits exactly one canonical terminal.
pub struct ResponsesStreamDecoder {
    scope: ProviderScope,
    requested_model: ModelId,
    lifecycle: DecoderLifecycle,
    started: bool,
    response_id: Option<String>,
    response_model: Option<ModelId>,
    last_sequence: Option<u64>,
    items: BTreeMap<u64, OutputItem>,
    item_indices: BTreeMap<String, u64>,
    completed_items: BTreeSet<u64>,
    text_started: BTreeSet<String>,
    text_ended: BTreeSet<String>,
    reasoning_started: BTreeSet<String>,
    reasoning_ended: BTreeSet<String>,
    tool_inputs: BTreeMap<String, ToolInputAssembly>,
    refusals: BTreeMap<String, String>,
    emitted_refusals: BTreeSet<String>,
    emitted_citations: BTreeSet<String>,
    turn_budget: ResponsesTurnBudget,
    terminal_policy: ResponsesTerminalPolicy,
    terminal_response: Option<ResponseWire>,
    response_diagnostics: ResponseDiagnostics,
}

impl std::fmt::Debug for ResponsesStreamDecoder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResponsesStreamDecoder")
            .field("scope", &self.scope)
            .field("requested_model", &self.requested_model)
            .field("started", &self.started)
            .field("terminal", &self.lifecycle.terminal_seen())
            .field("finished", &self.lifecycle.finish_seen())
            .field("item_count", &self.items.len())
            .field("turn_event_bytes", &self.turn_budget.event_bytes)
            .field("terminal_policy", &self.terminal_policy)
            .finish()
    }
}

#[derive(Debug)]
struct ResponsesTurnBudget {
    event_bytes: usize,
    maximum_event_bytes: usize,
    maximum_output_items: usize,
}

impl Default for ResponsesTurnBudget {
    fn default() -> Self {
        Self {
            event_bytes: 0,
            maximum_event_bytes: DEFAULT_RESPONSES_TURN_EVENT_BYTES_LIMIT,
            maximum_output_items: DEFAULT_RESPONSES_TURN_OUTPUT_ITEM_LIMIT,
        }
    }
}

impl ResponsesTurnBudget {
    fn observe_event(&mut self, bytes: usize) -> Result<(), Error> {
        let total = self.event_bytes.checked_add(bytes).ok_or_else(|| {
            turn_response_limit("OpenAI Responses turn event bytes exceeded the protocol limit")
        })?;
        if total > self.maximum_event_bytes {
            return Err(turn_response_limit(
                "OpenAI Responses turn event bytes exceeded the protocol limit",
            ));
        }
        self.event_bytes = total;
        Ok(())
    }

    fn ensure_output_items(&self, count: usize) -> Result<(), Error> {
        if count > self.maximum_output_items {
            return Err(turn_response_limit(
                "OpenAI Responses turn output-item count exceeded the protocol limit",
            ));
        }
        Ok(())
    }
}

fn turn_response_limit(message: &'static str) -> Error {
    Error::new(ErrorKind::ResponseLimit, message)
}

#[derive(Debug, Default)]
struct ToolInputAssembly {
    call_id: String,
    name: String,
    input: String,
    custom: bool,
}

impl ResponsesStreamDecoder {
    pub fn new(scope: ProviderScope, requested_model: ModelId) -> Self {
        Self {
            scope,
            requested_model,
            lifecycle: DecoderLifecycle::default(),
            started: false,
            response_id: None,
            response_model: None,
            last_sequence: None,
            items: BTreeMap::new(),
            item_indices: BTreeMap::new(),
            completed_items: BTreeSet::new(),
            text_started: BTreeSet::new(),
            text_ended: BTreeSet::new(),
            reasoning_started: BTreeSet::new(),
            reasoning_ended: BTreeSet::new(),
            tool_inputs: BTreeMap::new(),
            refusals: BTreeMap::new(),
            emitted_refusals: BTreeSet::new(),
            emitted_citations: BTreeSet::new(),
            turn_budget: ResponsesTurnBudget::default(),
            terminal_policy: ResponsesTerminalPolicy::Strict,
            terminal_response: None,
            response_diagnostics: ResponseDiagnostics::default(),
        }
    }

    /// Select the endpoint-specific terminal reconciliation policy.
    pub fn with_terminal_policy(mut self, policy: ResponsesTerminalPolicy) -> Self {
        self.terminal_policy = policy;
        self
    }

    pub fn with_response_diagnostics(mut self, diagnostics: ResponseDiagnostics) -> Self {
        self.response_diagnostics = diagnostics;
        self
    }

    pub fn decode(&mut self, data: &str) -> Result<Vec<LanguageStreamEvent>, Error> {
        <Self as LanguageStreamDecoder>::decode(self, data)
    }

    /// Decode one SSE data value once and retain both its native and portable views.
    pub fn decode_native(&mut self, data: &str) -> Result<DecodedResponsesStreamFrame, Error> {
        self.lifecycle
            .ensure_decode_allowed()
            .map_err(Error::from)?;
        if data.trim() == "[DONE]" {
            return Err(Error::unexpected_eof());
        }
        self.turn_budget.observe_event(data.len())?;
        let wire = serde_json::from_str::<StreamEventWire>(data).map_err(|source| {
            Error::new(
                ErrorKind::Protocol,
                "provider returned malformed OpenAI Responses event JSON",
            )
            .with_source(source)
        })?;
        let event = ResponsesStreamEvent::decode(wire)?;
        self.observe_sequence(event.sequence_number())?;
        let portable_events = self.decode_event(&event)?;
        self.lifecycle
            .record(&portable_events)
            .map_err(Error::from)?;
        Ok(DecodedResponsesStreamFrame {
            native: event,
            portable_events,
        })
    }

    pub fn finish(&mut self) -> Result<Vec<LanguageStreamEvent>, Error> {
        <Self as LanguageStreamDecoder>::finish(self)
    }

    pub fn terminal_seen(&self) -> bool {
        self.lifecycle.terminal_seen()
    }

    /// Return the reconstructed canonical terminal response resource.
    pub fn terminal_response(&self) -> Option<&ResponseWire> {
        self.terminal_response.as_ref()
    }

    fn decode_event(
        &mut self,
        event: &ResponsesStreamEvent,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        match event.kind_str() {
            "response.created" | "response.queued" | "response.in_progress" => {
                self.observe_progress_response(event)
            }
            "response.output_item.added" => self.output_item_added(event),
            "response.output_item.done" => self.output_item_done(event),
            "response.output_text.delta" => self.text_delta(event.wire()),
            "response.output_text.done" => self.text_done(event.wire()),
            "response.refusal.delta" => self.refusal_delta(event.wire()),
            "response.refusal.done" => self.refusal_done(event.wire()),
            "response.output_text.annotation.added" => self.annotation_added(event),
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                self.reasoning_delta(event.wire())
            }
            "response.reasoning_summary_text.done" | "response.reasoning_text.done" => {
                self.reasoning_done(event.wire())
            }
            "response.function_call_arguments.delta" => self.tool_input_delta(event.wire(), false),
            "response.function_call_arguments.done" => self.tool_input_done(event.wire(), false),
            "response.custom_tool_call_input.delta" => self.tool_input_delta(event.wire(), true),
            "response.custom_tool_call_input.done" => self.tool_input_done(event.wire(), true),
            "response.completed"
            | "response.incomplete"
            | "response.cancelled"
            | "response.failed" => self.terminal_event(event),
            "error" => self.error_event(event),
            _ => self.opaque_stream_event(event.wire()),
        }
    }

    fn observe_sequence(&mut self, sequence: Option<u64>) -> Result<(), Error> {
        let Some(sequence) = sequence else {
            return Ok(());
        };
        if self
            .last_sequence
            .is_some_and(|previous| sequence <= previous)
        {
            return Err(protocol_error(
                "OpenAI Responses stream sequence numbers were not strictly increasing",
            ));
        }
        self.last_sequence = Some(sequence);
        Ok(())
    }

    fn observe_progress_response(
        &mut self,
        event: &ResponsesStreamEvent,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        let response = event.response_resource().ok_or_else(|| {
            protocol_error("OpenAI Responses lifecycle event omitted its response")
        })?;
        if matches!(
            &response.status,
            ResponseStatus::Completed
                | ResponseStatus::Incomplete
                | ResponseStatus::Cancelled
                | ResponseStatus::Failed
        ) {
            return Err(protocol_error(
                "OpenAI Responses progress event carried a terminal response status",
            ));
        }
        self.observe_identity(response)
    }

    fn observe_identity(
        &mut self,
        response: &ResponseWire,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        if response.id.trim().is_empty() || response.model.trim().is_empty() {
            return Err(protocol_error(
                "OpenAI Responses lifecycle event omitted response identity",
            ));
        }
        if self
            .response_id
            .as_deref()
            .is_some_and(|existing| existing != response.id)
        {
            return Err(protocol_error(
                "OpenAI Responses stream changed its response ID",
            ));
        }
        let model = ModelId::new(response.model.clone()).map_err(|source| {
            Error::new(
                ErrorKind::Protocol,
                "OpenAI Responses stream contained an invalid model ID",
            )
            .with_source(source)
        })?;
        if self
            .response_model
            .as_ref()
            .is_some_and(|existing| existing != &model)
        {
            return Err(protocol_error(
                "OpenAI Responses stream changed its model ID",
            ));
        }
        self.response_id.get_or_insert_with(|| response.id.clone());
        self.response_model.get_or_insert(model);
        if self.started {
            return Ok(Vec::new());
        }
        self.started = true;
        Ok(vec![LanguageStreamEvent::Started {
            id: self.response_id.clone(),
            model: self.response_model.clone(),
        }])
    }

    fn output_item_added(
        &mut self,
        event: &ResponsesStreamEvent,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        let output_index = required_u64(event.wire(), "output_index")?;
        if self.items.contains_key(&output_index) {
            return Err(protocol_error(
                "OpenAI Responses stream reused an output index",
            ));
        }
        self.turn_budget
            .ensure_output_items(self.items.len().saturating_add(1))?;
        let item = event
            .item()
            .cloned()
            .ok_or_else(|| protocol_error("OpenAI Responses output-item event omitted its item"))?;
        let item_id = item
            .id()
            .ok_or_else(|| protocol_error("OpenAI Responses output item omitted its item ID"))?;
        if item_id.is_empty() || self.item_indices.contains_key(item_id) {
            return Err(protocol_error(
                "OpenAI Responses stream reused or omitted an output item ID",
            ));
        }
        self.item_indices.insert(item_id.to_string(), output_index);

        let mut events = Vec::new();
        match &item {
            OutputItem::FunctionCall(call) => {
                self.start_tool_input(
                    item_id,
                    &call.call_id,
                    &call.name,
                    &call.arguments,
                    false,
                    &mut events,
                )?;
            }
            OutputItem::CustomToolCall(call) => {
                self.start_tool_input(
                    item_id,
                    &call.call_id,
                    &call.name,
                    &call.input,
                    true,
                    &mut events,
                )?;
            }
            _ => {}
        }
        self.items.insert(output_index, item);
        Ok(events)
    }

    fn output_item_done(
        &mut self,
        event: &ResponsesStreamEvent,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        let output_index = required_u64(event.wire(), "output_index")?;
        if self.completed_items.contains(&output_index) {
            return Err(protocol_error(
                "OpenAI Responses stream completed an output item more than once",
            ));
        }
        let item = event
            .item()
            .cloned()
            .ok_or_else(|| protocol_error("OpenAI Responses output-item event omitted its item"))?;
        let item_id = item
            .id()
            .ok_or_else(|| protocol_error("OpenAI Responses output item omitted its item ID"))?;
        let expected_index = self.item_indices.get(item_id).copied().ok_or_else(|| {
            protocol_error("OpenAI Responses completed an item that was never added")
        })?;
        if expected_index != output_index {
            return Err(protocol_error(
                "OpenAI Responses output item changed its output index",
            ));
        }
        if self
            .items
            .get(&output_index)
            .is_some_and(|added| added.kind() != item.kind())
        {
            return Err(protocol_error(
                "OpenAI Responses output item changed its type",
            ));
        }

        let mut events = Vec::new();
        match &item {
            OutputItem::Message(message) => self.finish_message(message, &mut events),
            OutputItem::Reasoning(reasoning) => {
                self.finish_reasoning_item(item_id, reasoning, &mut events);
            }
            OutputItem::FunctionCall(call) => {
                self.emit_function_call(item_id, call, &mut events)?;
            }
            OutputItem::CustomToolCall(call) => {
                self.validate_custom_tool_call(item_id, call)?;
            }
            OutputItem::Program(_) | OutputItem::ProgramOutput(_) => {}
            OutputItem::ProviderTool(_) | OutputItem::Unknown(_) => {}
        }
        events.push(LanguageStreamEvent::ProviderOpaque(opaque_item(
            &item,
            &self.scope,
            self.response_model
                .as_ref()
                .unwrap_or(&self.requested_model),
        )?));
        self.items.insert(output_index, item);
        self.completed_items.insert(output_index);
        Ok(events)
    }

    fn text_delta(&mut self, event: &StreamEventWire) -> Result<Vec<LanguageStreamEvent>, Error> {
        let item_id = required_str(event, "item_id")?;
        self.ensure_known_item(item_id)?;
        let content_index = optional_u64(event, "content_index").unwrap_or(0);
        let id = content_id(item_id, "text", content_index);
        let delta = required_str(event, "delta")?.to_string();
        let mut events = Vec::new();
        if self.text_started.insert(id.clone()) {
            events.push(LanguageStreamEvent::TextStart { id: id.clone() });
        }
        events.push(LanguageStreamEvent::TextDelta { id, delta });
        Ok(events)
    }

    fn text_done(&mut self, event: &StreamEventWire) -> Result<Vec<LanguageStreamEvent>, Error> {
        let item_id = required_str(event, "item_id")?;
        self.ensure_known_item(item_id)?;
        let content_index = optional_u64(event, "content_index").unwrap_or(0);
        let id = content_id(item_id, "text", content_index);
        let mut events = Vec::new();
        if self.text_started.insert(id.clone()) {
            let text = event
                .field("text")
                .and_then(Value::as_str)
                .ok_or_else(|| protocol_error("OpenAI text completion omitted its final text"))?;
            events.push(LanguageStreamEvent::TextStart { id: id.clone() });
            events.push(LanguageStreamEvent::TextDelta {
                id: id.clone(),
                delta: text.to_string(),
            });
        }
        if self.text_ended.insert(id.clone()) {
            events.push(LanguageStreamEvent::TextEnd { id });
        }
        Ok(events)
    }

    fn reasoning_delta(
        &mut self,
        event: &StreamEventWire,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        let item_id = required_str(event, "item_id")?;
        self.ensure_known_item(item_id)?;
        let index = optional_u64(event, "summary_index")
            .or_else(|| optional_u64(event, "content_index"))
            .unwrap_or(0);
        let lane = if event.kind.contains("summary") {
            "summary"
        } else {
            "content"
        };
        let id = content_id(item_id, lane, index);
        let delta = required_str(event, "delta")?.to_string();
        let mut events = Vec::new();
        self.start_reasoning(&id, &mut events);
        events.push(LanguageStreamEvent::ReasoningDelta { id, delta });
        Ok(events)
    }

    fn reasoning_done(
        &mut self,
        event: &StreamEventWire,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        let item_id = required_str(event, "item_id")?;
        self.ensure_known_item(item_id)?;
        let index = optional_u64(event, "summary_index")
            .or_else(|| optional_u64(event, "content_index"))
            .unwrap_or(0);
        let lane = if event.kind.contains("summary") {
            "summary"
        } else {
            "content"
        };
        let id = content_id(item_id, lane, index);
        let mut events = Vec::new();
        if !self.reasoning_started.contains(&id)
            && let Some(text) = event.field("text").and_then(Value::as_str)
        {
            self.start_reasoning(&id, &mut events);
            events.push(LanguageStreamEvent::ReasoningDelta {
                id: id.clone(),
                delta: text.to_string(),
            });
        }
        self.end_reasoning(&id, &mut events);
        Ok(events)
    }

    fn start_reasoning(&mut self, id: &str, events: &mut Vec<LanguageStreamEvent>) {
        if self.reasoning_started.insert(id.to_string()) {
            events.push(LanguageStreamEvent::ReasoningStart { id: id.to_string() });
        }
    }

    fn end_reasoning(&mut self, id: &str, events: &mut Vec<LanguageStreamEvent>) {
        if self.reasoning_started.contains(id) && self.reasoning_ended.insert(id.to_string()) {
            events.push(LanguageStreamEvent::ReasoningEnd { id: id.to_string() });
        }
    }

    fn finish_reasoning_item(
        &mut self,
        item_id: &str,
        reasoning: &super::wire::ReasoningItemWire,
        events: &mut Vec<LanguageStreamEvent>,
    ) {
        for (index, part) in reasoning.summary.iter().enumerate() {
            self.finish_reasoning_part(item_id, "summary", index as u64, &part.text, events);
        }
        for (index, part) in reasoning.content.iter().enumerate() {
            self.finish_reasoning_part(item_id, "content", index as u64, &part.text, events);
        }
    }

    fn finish_reasoning_part(
        &mut self,
        item_id: &str,
        lane: &str,
        index: u64,
        text: &str,
        events: &mut Vec<LanguageStreamEvent>,
    ) {
        let id = content_id(item_id, lane, index);
        if !self.reasoning_started.contains(&id) {
            self.start_reasoning(&id, events);
            events.push(LanguageStreamEvent::ReasoningDelta {
                id: id.clone(),
                delta: text.to_string(),
            });
        }
        self.end_reasoning(&id, events);
    }

    fn refusal_delta(
        &mut self,
        event: &StreamEventWire,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        let item_id = required_str(event, "item_id")?;
        self.ensure_known_item(item_id)?;
        let content_index = optional_u64(event, "content_index").unwrap_or(0);
        let id = content_id(item_id, "refusal", content_index);
        self.refusals
            .entry(id)
            .or_default()
            .push_str(required_str(event, "delta")?);
        Ok(Vec::new())
    }

    fn refusal_done(&mut self, event: &StreamEventWire) -> Result<Vec<LanguageStreamEvent>, Error> {
        let item_id = required_str(event, "item_id")?;
        self.ensure_known_item(item_id)?;
        let content_index = optional_u64(event, "content_index").unwrap_or(0);
        let id = content_id(item_id, "refusal", content_index);
        let reason = event
            .field("refusal")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| self.refusals.remove(&id));
        if self.emitted_refusals.insert(id) {
            Ok(vec![LanguageStreamEvent::Refusal { reason }])
        } else {
            Ok(Vec::new())
        }
    }

    fn annotation_added(
        &mut self,
        event: &ResponsesStreamEvent,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        let item_id = required_str(event.wire(), "item_id")?;
        self.ensure_known_item(item_id)?;
        let annotation_index = required_u64(event.wire(), "annotation_index")?;
        let annotation_position = usize::try_from(annotation_index).map_err(|_| {
            protocol_error("OpenAI Responses annotation index exceeded the platform limit")
        })?;
        let annotation = event
            .annotation()
            .ok_or_else(|| protocol_error("OpenAI citation event omitted its annotation"))?;
        let key = format!("{item_id}:{annotation_index}");
        if !self.emitted_citations.insert(key) {
            return Ok(Vec::new());
        }
        Ok(vec![LanguageStreamEvent::Citation(project_citation(
            item_id,
            annotation_position,
            annotation,
        ))])
    }

    fn tool_input_delta(
        &mut self,
        event: &StreamEventWire,
        custom: bool,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        let item_id = required_str(event, "item_id")?;
        let delta = required_str(event, "delta")?.to_string();
        let assembly = self
            .tool_inputs
            .get_mut(item_id)
            .ok_or_else(|| protocol_error("OpenAI tool-input delta preceded its output item"))?;
        if assembly.custom != custom {
            return Err(protocol_error(
                "OpenAI tool-input event changed its tool kind",
            ));
        }
        append_tool_input(&mut assembly.input, &delta)?;
        if custom {
            return Ok(Vec::new());
        }
        Ok(vec![LanguageStreamEvent::ToolInputDelta {
            id: assembly.call_id.clone(),
            delta,
        }])
    }

    fn tool_input_done(
        &mut self,
        event: &StreamEventWire,
        custom: bool,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        let item_id = required_str(event, "item_id")?;
        let field = if custom { "input" } else { "arguments" };
        let completed = required_str(event, field)?.to_string();
        ensure_tool_input_limit(&completed)?;
        let assembly = self.tool_inputs.get_mut(item_id).ok_or_else(|| {
            protocol_error("OpenAI tool-input completion preceded its output item")
        })?;
        if assembly.custom != custom || (!assembly.input.is_empty() && assembly.input != completed)
        {
            return Err(protocol_error(
                "OpenAI tool-input completion disagreed with its streamed deltas",
            ));
        }
        assembly.input = completed;
        Ok(Vec::new())
    }

    fn start_tool_input(
        &mut self,
        item_id: &str,
        call_id: &str,
        name: &str,
        initial_input: &str,
        custom: bool,
        events: &mut Vec<LanguageStreamEvent>,
    ) -> Result<(), Error> {
        if call_id.is_empty() || name.is_empty() || self.tool_inputs.contains_key(item_id) {
            return Err(protocol_error(
                "OpenAI tool output item omitted or reused its identity",
            ));
        }
        ensure_tool_input_limit(initial_input)?;
        self.tool_inputs.insert(
            item_id.to_string(),
            ToolInputAssembly {
                call_id: call_id.to_string(),
                name: name.to_string(),
                input: initial_input.to_string(),
                custom,
            },
        );
        if !custom {
            events.push(LanguageStreamEvent::ToolInputStart {
                id: call_id.to_string(),
                name: name.to_string(),
                owner: ExecutionOwner::Local,
            });
        }
        Ok(())
    }

    fn emit_function_call(
        &mut self,
        item_id: &str,
        call: &super::wire::FunctionCallItemWire,
        events: &mut Vec<LanguageStreamEvent>,
    ) -> Result<(), Error> {
        let assembly = self
            .tool_inputs
            .get(item_id)
            .ok_or_else(|| protocol_error("OpenAI function call completed before it was added"))?;
        if assembly.call_id != call.call_id || assembly.name != call.name {
            return Err(protocol_error("OpenAI function call changed its identity"));
        }
        ensure_tool_input_limit(&call.arguments)?;
        if !assembly.input.is_empty() && assembly.input != call.arguments {
            return Err(protocol_error(
                "OpenAI function call arguments disagreed with streamed deltas",
            ));
        }
        let arguments = serde_json::from_str(&call.arguments).map_err(|source| {
            Error::new(
                ErrorKind::Protocol,
                "OpenAI function call ended with incomplete JSON arguments",
            )
            .with_source(source)
        })?;
        let call = ToolCall::local(call.call_id.clone(), call.name.clone(), arguments).map_err(
            |source| {
                Error::new(
                    ErrorKind::Protocol,
                    "OpenAI function call violated the canonical tool contract",
                )
                .with_source(source)
            },
        )?;
        events.push(LanguageStreamEvent::ToolCall(call));
        Ok(())
    }

    fn validate_custom_tool_call(
        &self,
        item_id: &str,
        call: &super::wire::CustomToolCallItemWire,
    ) -> Result<(), Error> {
        let assembly = self.tool_inputs.get(item_id).ok_or_else(|| {
            protocol_error("OpenAI custom tool call completed before it was added")
        })?;
        if assembly.call_id != call.call_id
            || assembly.name != call.name
            || (!assembly.input.is_empty() && assembly.input != call.input)
        {
            return Err(protocol_error(
                "OpenAI custom tool call disagreed with its streamed state",
            ));
        }
        Ok(())
    }

    fn finish_message(
        &mut self,
        message: &super::wire::MessageItemWire,
        events: &mut Vec<LanguageStreamEvent>,
    ) {
        for (content_index, part) in message.content.iter().enumerate() {
            match part {
                OutputContentPart::Text(text) => {
                    let id = content_id(&message.id, "text", content_index as u64);
                    if self.text_started.insert(id.clone()) {
                        events.push(LanguageStreamEvent::TextStart { id: id.clone() });
                        events.push(LanguageStreamEvent::TextDelta {
                            id: id.clone(),
                            delta: text.text.clone(),
                        });
                    }
                    if self.text_ended.insert(id.clone()) {
                        events.push(LanguageStreamEvent::TextEnd { id });
                    }
                    for (annotation_index, annotation) in text.annotations.iter().enumerate() {
                        let key = format!("{}:{annotation_index}", message.id);
                        if self.emitted_citations.insert(key) {
                            events.push(LanguageStreamEvent::Citation(project_citation(
                                &message.id,
                                annotation_index,
                                annotation,
                            )));
                        }
                    }
                }
                OutputContentPart::Refusal(refusal) => {
                    let id = content_id(&message.id, "refusal", content_index as u64);
                    if !self.emitted_refusals.insert(id) {
                        continue;
                    }
                    events.push(LanguageStreamEvent::Refusal {
                        reason: Some(refusal.refusal.clone()),
                    });
                }
                OutputContentPart::Unknown(_) => {}
            }
        }
    }

    fn terminal_event(
        &mut self,
        event: &ResponsesStreamEvent,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        let mut projected_value = event.raw_response().cloned().ok_or_else(|| {
            protocol_error("OpenAI Responses terminal event omitted its response")
        })?;
        self.reconcile_terminal_value(&mut projected_value)?;
        let mut projected =
            serde_json::from_value::<ResponseWire>(projected_value).map_err(|source| {
                Error::new(
                    ErrorKind::Protocol,
                    "OpenAI Responses terminal event contained a malformed response resource",
                )
                .with_source(source)
            })?;
        let expected = match event.kind_str() {
            "response.completed" => "completed",
            "response.incomplete" => "incomplete",
            "response.cancelled" => "cancelled",
            "response.failed" => "failed",
            _ => unreachable!("terminal dispatch only passes terminal events"),
        };
        if projected.status.as_str() != expected {
            return Err(protocol_error(
                "OpenAI Responses terminal event disagreed with response status",
            ));
        }
        self.reconcile_terminal_items(&mut projected)?;
        let mut events = self.observe_identity(&projected)?;
        let decoded = decode_response_wire(projected.clone(), &self.scope, &self.requested_model)?;
        if let Some(usage) = &projected.usage {
            events.push(LanguageStreamEvent::Usage(decode_usage(usage)));
        }
        let (_, canonical) = decoded.into_parts();
        let terminal = match &projected.status {
            ResponseStatus::Completed | ResponseStatus::Incomplete => StreamTerminal::Completed {
                response: Box::new(canonical),
            },
            ResponseStatus::Cancelled => StreamTerminal::Cancelled {
                reason: "OpenAI cancelled the Responses generation".to_string(),
                response: Some(Box::new(canonical)),
            },
            ResponseStatus::Failed => StreamTerminal::Failed {
                error: failed_response_error(
                    &projected,
                    &self.scope,
                    &self.requested_model,
                    self.response_diagnostics.clone(),
                ),
                response: Some(Box::new(canonical)),
            },
            ResponseStatus::Queued | ResponseStatus::InProgress | ResponseStatus::Other(_) => {
                return Err(protocol_error(
                    "OpenAI Responses terminal event carried a non-terminal status",
                ));
            }
        };
        self.terminal_response = Some(projected);
        events.push(LanguageStreamEvent::Terminal(terminal));
        Ok(events)
    }

    fn reconcile_terminal_value(&self, response: &mut Value) -> Result<(), Error> {
        let object = response.as_object_mut().ok_or_else(|| {
            protocol_error("OpenAI Responses terminal response must be a JSON object")
        })?;
        let output = object
            .entry("output")
            .or_insert_with(|| Value::Array(Vec::new()))
            .as_array_mut()
            .ok_or_else(|| protocol_error("OpenAI Responses terminal output must be an array"))?;
        self.turn_budget.ensure_output_items(output.len())?;

        for (position, terminal) in output.iter_mut().enumerate() {
            let terminal_object = terminal.as_object_mut().ok_or_else(|| {
                protocol_error("OpenAI Responses terminal output item must be an object")
            })?;
            let Some((_, streamed)) =
                self.terminal_streamed_candidate(terminal_object, position)?
            else {
                continue;
            };
            reconcile_terminal_item_value(terminal_object, streamed, self.terminal_policy)?;
        }
        Ok(())
    }

    fn terminal_streamed_candidate<'a>(
        &'a self,
        terminal: &Map<String, Value>,
        position: usize,
    ) -> Result<Option<(u64, &'a OutputItem)>, Error> {
        let kind = terminal
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                protocol_error("OpenAI Responses terminal output item omitted its type")
            })?;
        let id = terminal
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty());
        let call_id = terminal
            .get("call_id")
            .and_then(Value::as_str)
            .filter(|call_id| !call_id.is_empty());

        let mut matches = self.items.iter().filter(|(output_index, streamed)| {
            if streamed.kind() != kind || !self.completed_items.contains(output_index) {
                return false;
            }
            id.is_some_and(|id| streamed.id() == Some(id))
                || (has_stable_call_identity(kind)
                    && call_id.is_some_and(|call_id| streamed.call_id() == Some(call_id)))
        });
        let first = matches.next();
        if matches.next().is_some() {
            return Err(protocol_error(
                "OpenAI terminal response matched multiple completed output items",
            ));
        }
        if let Some((index, item)) = first {
            return Ok(Some((*index, item)));
        }

        let position = u64::try_from(position).map_err(|_| {
            protocol_error("OpenAI terminal output position exceeded the platform limit")
        })?;
        if has_stable_call_identity(kind)
            && let Some(item) = self.items.get(&position)
            && item.kind() == kind
            && self.completed_items.contains(&position)
        {
            return Ok(Some((position, item)));
        }

        if self.terminal_policy == ResponsesTerminalPolicy::Compatible
            && kind == "message"
            && id.is_none()
            && let Some(item) = self.items.get(&position)
            && item.kind() == kind
            && self.completed_items.contains(&position)
        {
            return Ok(Some((position, item)));
        }
        Ok(None)
    }

    fn error_event(
        &mut self,
        event: &ResponsesStreamEvent,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        let envelope = encode_event_value(event.wire())?;
        event
            .error()
            .ok_or_else(|| protocol_error("OpenAI Responses error event omitted its error"))?;
        let error = classify_stream_error(
            &envelope,
            self.response_diagnostics.clone(),
            "OpenAI emitted an error after establishing the Responses stream",
        );
        Ok(vec![LanguageStreamEvent::Terminal(
            StreamTerminal::Failed {
                error,
                response: None,
            },
        )])
    }

    fn opaque_stream_event(
        &mut self,
        event: &StreamEventWire,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        let data = event.to_value().map_err(|source| {
            Error::new(
                ErrorKind::Protocol,
                "failed to preserve an OpenAI Responses stream event",
            )
            .with_source(source)
        })?;
        let model = self
            .response_model
            .clone()
            .unwrap_or_else(|| self.requested_model.clone());
        let provenance =
            siumai_core::ProviderProvenance::from_scope(&self.scope, model).map_err(|source| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "OpenAI Responses replay requires an explicit provider replay domain",
                )
                .with_source(source)
            })?;
        let item = siumai_core::OpaqueProviderItem::new(
            provenance,
            format!("{OPENAI_RESPONSES_OPAQUE_KIND}.stream_event"),
            data,
        )
        .map_err(|source| {
            Error::new(
                ErrorKind::ResponseLimit,
                "OpenAI Responses stream event exceeded the opaque replay limit",
            )
            .with_source(source)
        })?;
        Ok(vec![LanguageStreamEvent::ProviderOpaque(item)])
    }

    fn ensure_known_item(&self, item_id: &str) -> Result<(), Error> {
        if self
            .item_indices
            .get(item_id)
            .is_some_and(|index| !self.completed_items.contains(index))
        {
            Ok(())
        } else {
            Err(protocol_error(
                "OpenAI Responses delta referenced an unknown or completed output item",
            ))
        }
    }

    fn reconcile_terminal_items(&self, response: &mut ResponseWire) -> Result<(), Error> {
        let mut missing_functions = Vec::new();
        for (output_index, streamed) in &self.items {
            if let Some(index) = terminal_item_index(&response.output, streamed)? {
                let terminal = &response.output[index];
                compare_terminal_item(
                    streamed,
                    terminal,
                    self.completed_items.contains(output_index),
                )?;
            } else if self.completed_items.contains(output_index) {
                match streamed {
                    OutputItem::FunctionCall(_)
                        if self.terminal_policy == ResponsesTerminalPolicy::Compatible =>
                    {
                        missing_functions.push((*output_index, streamed.clone()));
                    }
                    OutputItem::FunctionCall(_)
                    | OutputItem::Message(_)
                    | OutputItem::Reasoning(_) => {
                        return Err(protocol_error(
                            "OpenAI terminal response omitted a completed portable output item",
                        ));
                    }
                    _ => {}
                }
            }
        }
        for (output_index, function) in missing_functions {
            let insert_at = response
                .output
                .iter()
                .position(|terminal| {
                    self.streamed_output_index(terminal)
                        .is_some_and(|terminal_index| terminal_index > output_index)
                })
                .unwrap_or(response.output.len());
            response.output.insert(insert_at, function);
        }
        self.turn_budget
            .ensure_output_items(response.output.len())?;
        if matches!(&response.status, ResponseStatus::Completed)
            && self.completed_items.len() != self.items.len()
        {
            return Err(protocol_error(
                "OpenAI terminal response arrived before all output items completed",
            ));
        }
        Ok(())
    }

    fn streamed_output_index(&self, terminal: &OutputItem) -> Option<u64> {
        terminal
            .id()
            .and_then(|item_id| self.item_indices.get(item_id).copied())
            .or_else(|| {
                let OutputItem::FunctionCall(terminal) = terminal else {
                    return None;
                };
                self.items.iter().find_map(|(output_index, streamed)| {
                    let OutputItem::FunctionCall(streamed) = streamed else {
                        return None;
                    };
                    (streamed.call_id == terminal.call_id).then_some(*output_index)
                })
            })
    }
}

fn reconcile_terminal_item_value(
    terminal: &mut Map<String, Value>,
    streamed: &OutputItem,
    policy: ResponsesTerminalPolicy,
) -> Result<(), Error> {
    let streamed_value = streamed.to_value().map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "failed to inspect a completed OpenAI Responses output item",
        )
        .with_source(source)
    })?;
    let streamed = streamed_value.as_object().ok_or_else(|| {
        protocol_error("completed OpenAI Responses output item was not an object")
    })?;
    compare_required_terminal_field(terminal, streamed, "type")?;

    match terminal.get("type").and_then(Value::as_str) {
        Some("message") => {
            reconcile_missing_terminal_field(
                terminal,
                streamed,
                "id",
                policy == ResponsesTerminalPolicy::Compatible,
            )?;
            reconcile_missing_terminal_field(terminal, streamed, "status", true)?;
            compare_required_terminal_field(terminal, streamed, "role")?;
            compare_required_terminal_field(terminal, streamed, "content")?;
            compare_optional_terminal_field(terminal, streamed, "phase")?;
        }
        Some("function_call") => {
            reconcile_missing_terminal_field(terminal, streamed, "id", true)?;
            reconcile_missing_terminal_field(terminal, streamed, "status", true)?;
            for field in ["call_id", "name"] {
                compare_required_terminal_field(terminal, streamed, field)?;
            }
            compare_function_arguments_field(terminal, streamed)?;
            compare_optional_terminal_field(terminal, streamed, "namespace")?;
            compare_tool_caller_field(terminal, streamed)?;
        }
        Some("custom_tool_call") => {
            for field in ["id", "call_id", "name", "input"] {
                compare_required_terminal_field(terminal, streamed, field)?;
            }
            compare_optional_terminal_field(terminal, streamed, "status")?;
            compare_optional_terminal_field(terminal, streamed, "namespace")?;
            compare_tool_caller_field(terminal, streamed)?;
        }
        Some("program") => {
            for field in ["id", "call_id", "code", "fingerprint"] {
                compare_required_terminal_field(terminal, streamed, field)?;
            }
        }
        Some("program_output") => {
            for field in ["id", "status", "call_id", "result"] {
                compare_required_terminal_field(terminal, streamed, field)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn reconcile_missing_terminal_field(
    terminal: &mut Map<String, Value>,
    streamed: &Map<String, Value>,
    field: &'static str,
    allow_missing: bool,
) -> Result<(), Error> {
    match (terminal.get(field), streamed.get(field)) {
        (Some(terminal), Some(streamed)) if terminal != streamed => Err(protocol_error(
            "OpenAI terminal response changed a completed output item field",
        )),
        (None, Some(streamed)) if allow_missing => {
            terminal.insert(field.to_string(), streamed.clone());
            Ok(())
        }
        (None, Some(_)) => Err(protocol_error(
            "OpenAI terminal response omitted a required completed output item field",
        )),
        _ => Ok(()),
    }
}

fn compare_required_terminal_field(
    terminal: &Map<String, Value>,
    streamed: &Map<String, Value>,
    field: &'static str,
) -> Result<(), Error> {
    match (terminal.get(field), streamed.get(field)) {
        (Some(terminal), Some(streamed)) if terminal == streamed => Ok(()),
        _ => Err(protocol_error(
            "OpenAI terminal response changed or omitted completed output semantics",
        )),
    }
}

fn compare_optional_terminal_field(
    terminal: &Map<String, Value>,
    streamed: &Map<String, Value>,
    field: &'static str,
) -> Result<(), Error> {
    if terminal.get(field) == streamed.get(field) {
        Ok(())
    } else {
        Err(protocol_error(
            "OpenAI terminal response changed completed output semantics",
        ))
    }
}

fn compare_function_arguments_field(
    terminal: &Map<String, Value>,
    streamed: &Map<String, Value>,
) -> Result<(), Error> {
    let terminal = terminal
        .get("arguments")
        .and_then(Value::as_str)
        .ok_or_else(|| protocol_error("OpenAI terminal function call omitted arguments"))?;
    let streamed = streamed
        .get("arguments")
        .and_then(Value::as_str)
        .ok_or_else(|| protocol_error("OpenAI completed function call omitted arguments"))?;
    if function_arguments_equal(streamed, terminal)? {
        Ok(())
    } else {
        Err(protocol_error(
            "OpenAI terminal response changed completed function arguments",
        ))
    }
}

fn compare_tool_caller_field(
    terminal: &Map<String, Value>,
    streamed: &Map<String, Value>,
) -> Result<(), Error> {
    match (terminal.get("caller"), streamed.get("caller")) {
        (None, None) => Ok(()),
        (Some(terminal), Some(streamed)) => {
            let terminal = terminal.as_object().ok_or_else(|| {
                protocol_error("OpenAI terminal function caller must be an object")
            })?;
            let streamed = streamed.as_object().ok_or_else(|| {
                protocol_error("OpenAI completed function caller must be an object")
            })?;
            compare_required_terminal_field(terminal, streamed, "type")?;
            compare_optional_terminal_field(terminal, streamed, "caller_id")
        }
        _ => Err(protocol_error(
            "OpenAI terminal response changed completed function caller semantics",
        )),
    }
}

fn terminal_item_index(
    terminal: &[OutputItem],
    streamed: &OutputItem,
) -> Result<Option<usize>, Error> {
    let streamed_id = streamed.id();
    let streamed_call_id = matches!(streamed, OutputItem::FunctionCall(_))
        .then(|| streamed.call_id())
        .flatten();
    let mut matches = terminal.iter().enumerate().filter_map(|(index, terminal)| {
        let same_item_id = streamed_id.is_some() && streamed_id == terminal.id();
        let same_call_id = streamed.kind() == terminal.kind()
            && has_stable_call_identity(terminal.kind())
            && streamed_call_id.is_some()
            && streamed_call_id == terminal.call_id();
        (same_item_id || same_call_id).then_some(index)
    });
    let first = matches.next();
    if matches.next().is_some() {
        return Err(protocol_error(
            "OpenAI terminal response duplicated a streamed output identity",
        ));
    }
    Ok(first)
}

fn compare_terminal_item(
    streamed: &OutputItem,
    terminal: &OutputItem,
    compare_semantics: bool,
) -> Result<(), Error> {
    if streamed.id() != terminal.id() || streamed.kind() != terminal.kind() {
        return Err(protocol_error(
            "OpenAI terminal response changed a streamed output item identity",
        ));
    }
    if compare_semantics {
        let matches = match (streamed, terminal) {
            (OutputItem::FunctionCall(_), OutputItem::FunctionCall(_)) => {
                function_calls_semantically_equal(streamed, terminal)?
            }
            (OutputItem::Message(streamed), OutputItem::Message(terminal)) => {
                streamed.id == terminal.id
                    && streamed.status == terminal.status
                    && streamed.role == terminal.role
                    && streamed.content == terminal.content
                    && streamed.phase == terminal.phase
            }
            (OutputItem::Reasoning(streamed), OutputItem::Reasoning(terminal)) => {
                streamed.id == terminal.id
                    && streamed.status == terminal.status
                    && streamed.summary == terminal.summary
                    && streamed.content == terminal.content
                    && streamed.encrypted_content == terminal.encrypted_content
            }
            (OutputItem::CustomToolCall(streamed), OutputItem::CustomToolCall(terminal)) => {
                streamed.id == terminal.id
                    && streamed.status == terminal.status
                    && streamed.call_id == terminal.call_id
                    && streamed.name == terminal.name
                    && streamed.input == terminal.input
                    && streamed.namespace == terminal.namespace
                    && tool_callers_semantically_equal(
                        streamed.caller.as_ref(),
                        terminal.caller.as_ref(),
                    )
            }
            (OutputItem::Program(streamed), OutputItem::Program(terminal)) => {
                streamed.id == terminal.id
                    && streamed.call_id == terminal.call_id
                    && streamed.code == terminal.code
                    && streamed.fingerprint == terminal.fingerprint
            }
            (OutputItem::ProgramOutput(streamed), OutputItem::ProgramOutput(terminal)) => {
                streamed.id == terminal.id
                    && streamed.status == terminal.status
                    && streamed.call_id == terminal.call_id
                    && streamed.result == terminal.result
            }
            _ => true,
        };
        if !matches {
            return Err(protocol_error(
                "OpenAI terminal response changed completed streamed output semantics",
            ));
        }
    }
    Ok(())
}

fn has_stable_call_identity(kind: &str) -> bool {
    matches!(
        kind,
        "function_call" | "custom_tool_call" | "program" | "program_output"
    )
}

fn function_calls_semantically_equal(
    streamed: &OutputItem,
    terminal: &OutputItem,
) -> Result<bool, Error> {
    match (streamed, terminal) {
        (OutputItem::FunctionCall(streamed), OutputItem::FunctionCall(terminal)) => Ok(streamed
            .call_id
            == terminal.call_id
            && streamed.name == terminal.name
            && streamed.namespace == terminal.namespace
            && tool_callers_semantically_equal(streamed.caller.as_ref(), terminal.caller.as_ref())
            && function_arguments_equal(&streamed.arguments, &terminal.arguments)?),
        _ => Ok(false),
    }
}

fn tool_callers_semantically_equal(
    streamed: Option<&super::wire::ToolCallerWire>,
    terminal: Option<&super::wire::ToolCallerWire>,
) -> bool {
    match (streamed, terminal) {
        (None, None) => true,
        (Some(streamed), Some(terminal)) => {
            streamed.kind == terminal.kind && streamed.caller_id == terminal.caller_id
        }
        _ => false,
    }
}

fn function_arguments_equal(streamed: &str, terminal: &str) -> Result<bool, Error> {
    ensure_tool_input_limit(streamed)?;
    ensure_tool_input_limit(terminal)?;
    let streamed = serde_json::from_str::<Value>(streamed).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "OpenAI streamed function call ended with malformed JSON arguments",
        )
        .with_source(source)
    })?;
    let terminal = serde_json::from_str::<Value>(terminal).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "OpenAI terminal function call contained malformed JSON arguments",
        )
        .with_source(source)
    })?;
    Ok(streamed == terminal)
}

fn ensure_tool_input_limit(input: &str) -> Result<(), Error> {
    if input.len() > DEFAULT_TOOL_INPUT_BYTE_LIMIT {
        return Err(Error::new(
            ErrorKind::ResponseLimit,
            "OpenAI Responses streamed tool input exceeded the byte limit",
        ));
    }
    Ok(())
}

fn append_tool_input(buffer: &mut String, delta: &str) -> Result<(), Error> {
    if buffer
        .len()
        .checked_add(delta.len())
        .is_none_or(|total| total > DEFAULT_TOOL_INPUT_BYTE_LIMIT)
    {
        return Err(Error::new(
            ErrorKind::ResponseLimit,
            "OpenAI Responses streamed tool input exceeded the byte limit",
        ));
    }
    buffer.push_str(delta);
    Ok(())
}

impl LanguageStreamDecoder for ResponsesStreamDecoder {
    type ProtocolFrame = str;

    fn set_response_diagnostics(&mut self, diagnostics: ResponseDiagnostics) {
        self.response_diagnostics = diagnostics;
    }

    fn decode(&mut self, frame: &Self::ProtocolFrame) -> Result<Vec<LanguageStreamEvent>, Error> {
        Ok(self.decode_native(frame)?.into_portable_events())
    }

    fn finish(&mut self) -> Result<Vec<LanguageStreamEvent>, Error> {
        if self.lifecycle.begin_finish().map_err(Error::from)? {
            Ok(Vec::new())
        } else {
            Err(Error::unexpected_eof())
        }
    }

    fn terminal_seen(&self) -> bool {
        self.lifecycle.terminal_seen()
    }
}

fn required_str<'a>(event: &'a StreamEventWire, field: &str) -> Result<&'a str, Error> {
    event
        .field(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| protocol_error("OpenAI Responses stream event omitted a string field"))
}

fn required_u64(event: &StreamEventWire, field: &str) -> Result<u64, Error> {
    event
        .field(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| protocol_error("OpenAI Responses stream event omitted an integer field"))
}

fn optional_u64(event: &StreamEventWire, field: &str) -> Option<u64> {
    event.field(field).and_then(Value::as_u64)
}

fn encode_event_value(event: &StreamEventWire) -> Result<Value, Error> {
    event.to_value().map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "failed to preserve an OpenAI Responses stream event",
        )
        .with_source(source)
    })
}

fn content_id(item_id: &str, lane: &str, index: u64) -> String {
    format!("{item_id}:{lane}:{index}")
}

#[cfg(test)]
mod turn_budget_tests {
    use super::*;

    #[test]
    fn aggregate_event_bytes_and_output_items_are_bounded() {
        let mut budget = ResponsesTurnBudget {
            event_bytes: 0,
            maximum_event_bytes: 8,
            maximum_output_items: 2,
        };

        budget.observe_event(3).unwrap();
        budget.observe_event(5).unwrap();
        assert_eq!(
            budget.observe_event(1).unwrap_err().kind(),
            ErrorKind::ResponseLimit
        );
        budget.ensure_output_items(2).unwrap();
        assert_eq!(
            budget.ensure_output_items(3).unwrap_err().kind(),
            ErrorKind::ResponseLimit
        );
    }
}
