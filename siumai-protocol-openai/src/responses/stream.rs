//! Transport-neutral decoder for OpenAI Responses server events.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Bound::{Excluded, Unbounded};

use serde_json::{Map, Value};
use siumai_core::{
    DEFAULT_TOOL_INPUT_BYTE_LIMIT, DecoderLifecycle, Error, ErrorKind, ExecutionOwner,
    LanguageStreamDecoder, LanguageStreamEvent, ModelId, ProviderScope, ResponseDiagnostics,
    StreamTerminal, ToolCall, UsageUpdate,
};

use crate::openai_error::classify_stream_error;

use super::response::{
    decode_response_wire_with_replay, decode_usage, failed_response_error, project_citation,
    protocol_error,
};
use super::wire::{
    AnnotationWire, OutputContentPart, OutputItem, ResponseErrorWire, ResponseStatus, ResponseWire,
    StreamEventWire,
};

const DEFAULT_RESPONSES_TURN_EVENT_BYTES_LIMIT: usize = 64 * 1024 * 1024;
const DEFAULT_RESPONSES_TURN_OUTPUT_ITEM_LIMIT: usize = 16_384;

/// Typed classification for one native Responses streaming event.
#[derive(Clone, PartialEq, Eq)]
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

/// Describes the documented wire contractions a configured Responses codec may normalize.
///
/// This descriptor is selected by the concrete protocol/dialect owner. It is deliberately
/// independent of support claims, model catalogs, endpoint labels, and lifecycle metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ResponsesWireDialect {
    allow_message_position_identity: bool,
    allow_omitted_message_annotations: bool,
}

impl ResponsesWireDialect {
    /// The official OpenAI Responses wire shape.
    pub const fn openai() -> Self {
        Self {
            allow_message_position_identity: false,
            allow_omitted_message_annotations: false,
        }
    }

    /// A maintained OpenAI-compatible dialect with the documented abbreviated message shape.
    pub const fn compatible() -> Self {
        Self {
            allow_message_position_identity: true,
            allow_omitted_message_annotations: true,
        }
    }

    /// Allow a missing optional message identity to be recovered from one unique output position.
    pub const fn with_message_position_identity(mut self, enabled: bool) -> Self {
        self.allow_message_position_identity = enabled;
        self
    }

    /// Allow omitted output-text annotations to be recovered from the completed stream item.
    pub const fn with_omitted_message_annotations(mut self, enabled: bool) -> Self {
        self.allow_omitted_message_annotations = enabled;
        self
    }

    pub const fn allows_message_position_identity(self) -> bool {
        self.allow_message_position_identity
    }

    pub const fn allows_omitted_message_annotations(self) -> bool {
        self.allow_omitted_message_annotations
    }
}

impl Default for ResponsesWireDialect {
    fn default() -> Self {
        Self::openai()
    }
}

/// Bounded public summary of native replay conflicts observed during reconciliation.
///
/// The summary intentionally contains counts and field classes only. Raw provider values,
/// encrypted reasoning material, fingerprints, and event payloads remain native or sensitive.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct ResponsesReplayStatus {
    settled: bool,
    terminal_resource_available: bool,
    item_identity_conflicts: usize,
    reasoning_state_conflicts: usize,
    provider_item_conflicts: usize,
}

impl ResponsesReplayStatus {
    /// Return whether the turn has not reached a terminal provider outcome yet.
    pub const fn is_pending(self) -> bool {
        !self.settled
    }

    /// Return whether a settled terminal resource remains safe for native replay.
    pub const fn is_available(self) -> bool {
        self.settled
            && self.terminal_resource_available
            && self.item_identity_conflicts == 0
            && self.reasoning_state_conflicts == 0
            && self.provider_item_conflicts == 0
    }

    pub const fn total_conflicts(self) -> usize {
        self.item_identity_conflicts + self.reasoning_state_conflicts + self.provider_item_conflicts
    }

    pub const fn item_identity_conflicts(self) -> usize {
        self.item_identity_conflicts
    }

    pub const fn reasoning_state_conflicts(self) -> usize {
        self.reasoning_state_conflicts
    }

    pub const fn provider_item_conflicts(self) -> usize {
        self.provider_item_conflicts
    }

    fn record_item_identity_conflict(&mut self) {
        self.item_identity_conflicts = self.item_identity_conflicts.saturating_add(1);
    }

    fn record_reasoning_state_conflict(&mut self) {
        self.reasoning_state_conflicts = self.reasoning_state_conflicts.saturating_add(1);
    }

    fn record_provider_item_conflict(&mut self) {
        self.provider_item_conflicts = self.provider_item_conflicts.saturating_add(1);
    }

    fn settle_with_terminal_resource(&mut self) {
        self.settled = true;
        self.terminal_resource_available = true;
    }

    fn settle_without_terminal_resource(&mut self) {
        self.settled = true;
        self.terminal_resource_available = false;
    }
}

impl std::fmt::Debug for ResponsesReplayStatus {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResponsesReplayStatus")
            .field("pending", &self.is_pending())
            .field("available", &self.is_available())
            .field("item_identity_conflicts", &self.item_identity_conflicts)
            .field("reasoning_state_conflicts", &self.reasoning_state_conflicts)
            .field("provider_item_conflicts", &self.provider_item_conflicts)
            .finish()
    }
}

#[derive(Debug)]
struct TerminalAlignment {
    terminal_to_streamed: Vec<Option<u64>>,
    streamed_to_terminal: BTreeMap<u64, usize>,
}

impl TerminalAlignment {
    fn terminal_index(&self, output_index: u64) -> Option<usize> {
        self.streamed_to_terminal.get(&output_index).copied()
    }

    fn streamed_index(&self, terminal_index: usize) -> Option<u64> {
        self.terminal_to_streamed
            .get(terminal_index)
            .copied()
            .flatten()
    }

    fn previous_terminal_index(&self, output_index: u64) -> Option<usize> {
        self.streamed_to_terminal
            .range(..output_index)
            .next_back()
            .map(|(_, terminal_index)| *terminal_index)
    }

    fn next_terminal_index(&self, output_index: u64) -> Option<usize> {
        self.streamed_to_terminal
            .range((Excluded(output_index), Unbounded))
            .next()
            .map(|(_, terminal_index)| *terminal_index)
    }
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

impl std::fmt::Debug for ResponsesStreamEventKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown(kind) => formatter
                .debug_struct("Unknown")
                .field("kind_bytes", &kind.len())
                .finish(),
            known => formatter.write_str(known.debug_name()),
        }
    }
}

impl ResponsesStreamEventKind {
    fn debug_name(&self) -> &'static str {
        match self {
            Self::ResponseCreated => "ResponseCreated",
            Self::ResponseQueued => "ResponseQueued",
            Self::ResponseInProgress => "ResponseInProgress",
            Self::OutputItemAdded => "OutputItemAdded",
            Self::OutputItemDone => "OutputItemDone",
            Self::OutputTextDelta => "OutputTextDelta",
            Self::OutputTextDone => "OutputTextDone",
            Self::RefusalDelta => "RefusalDelta",
            Self::RefusalDone => "RefusalDone",
            Self::OutputTextAnnotationAdded => "OutputTextAnnotationAdded",
            Self::ReasoningSummaryTextDelta => "ReasoningSummaryTextDelta",
            Self::ReasoningSummaryTextDone => "ReasoningSummaryTextDone",
            Self::ReasoningTextDelta => "ReasoningTextDelta",
            Self::ReasoningTextDone => "ReasoningTextDone",
            Self::FunctionCallArgumentsDelta => "FunctionCallArgumentsDelta",
            Self::FunctionCallArgumentsDone => "FunctionCallArgumentsDone",
            Self::CustomToolCallInputDelta => "CustomToolCallInputDelta",
            Self::CustomToolCallInputDone => "CustomToolCallInputDone",
            Self::ResponseCompleted => "ResponseCompleted",
            Self::ResponseIncomplete => "ResponseIncomplete",
            Self::ResponseCancelled => "ResponseCancelled",
            Self::ResponseFailed => "ResponseFailed",
            Self::Error => "Error",
            Self::Unknown(_) => "Unknown",
        }
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
            .field("field_count", &self.wire.fields.len())
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
    replay_status: ResponsesReplayStatus,
}

impl std::fmt::Debug for DecodedResponsesStreamFrame {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DecodedResponsesStreamFrame")
            .field("native", &self.native)
            .field("portable_event_count", &self.portable_events.len())
            .field("replay_status", &self.replay_status)
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

    pub fn replay_status(&self) -> ResponsesReplayStatus {
        self.replay_status
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

    pub fn into_parts(
        self,
    ) -> (
        ResponsesStreamEvent,
        Vec<LanguageStreamEvent>,
        ResponsesReplayStatus,
    ) {
        (self.native, self.portable_events, self.replay_status)
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
    text_started: BTreeSet<ContentLaneId>,
    text_ended: BTreeSet<ContentLaneId>,
    text_values: BTreeMap<ContentLaneId, String>,
    reasoning_started: BTreeSet<ContentLaneId>,
    reasoning_ended: BTreeSet<ContentLaneId>,
    reasoning_values: BTreeMap<ContentLaneId, String>,
    tool_inputs: BTreeMap<String, ToolInputAssembly>,
    refusals: BTreeMap<ContentLaneId, String>,
    refusal_ended: BTreeSet<ContentLaneId>,
    emitted_refusals: BTreeSet<ContentLaneId>,
    citations: BTreeMap<CitationId, AnnotationWire>,
    turn_budget: ResponsesTurnBudget,
    wire_dialect: ResponsesWireDialect,
    terminal_response: Option<ResponseWire>,
    response_diagnostics: ResponseDiagnostics,
    replay_status: ResponsesReplayStatus,
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
            .field("wire_dialect", &self.wire_dialect)
            .field("replay_status", &self.replay_status)
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
    done: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamItemKind {
    Message,
    Reasoning,
    FunctionCall,
    CustomToolCall,
}

impl StreamItemKind {
    const fn matches(self, item: &OutputItem) -> bool {
        matches!(
            (self, item),
            (Self::Message, OutputItem::Message(_))
                | (Self::Reasoning, OutputItem::Reasoning(_))
                | (Self::FunctionCall, OutputItem::FunctionCall(_))
                | (Self::CustomToolCall, OutputItem::CustomToolCall(_))
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ContentLaneKind {
    Text,
    Refusal,
    ReasoningSummary,
    ReasoningContent,
}

impl ContentLaneKind {
    const fn label(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Refusal => "refusal",
            Self::ReasoningSummary => "summary",
            Self::ReasoningContent => "content",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ContentLaneId {
    item_id: String,
    kind: ContentLaneKind,
    index: u64,
}

impl ContentLaneId {
    fn new(item_id: &str, kind: ContentLaneKind, index: u64) -> Self {
        Self {
            item_id: item_id.to_string(),
            kind,
            index,
        }
    }

    fn event_id(&self) -> String {
        format!("{}:{}:{}", self.item_id, self.kind.label(), self.index)
    }

    fn bounds(item_id: &str) -> (Self, Self) {
        (
            Self::new(item_id, ContentLaneKind::Text, 0),
            Self::new(item_id, ContentLaneKind::ReasoningContent, u64::MAX),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CitationId {
    item_id: String,
    content_index: u64,
    annotation_index: u64,
}

impl CitationId {
    fn bounds(item_id: &str) -> (Self, Self) {
        (
            Self {
                item_id: item_id.to_string(),
                content_index: 0,
                annotation_index: 0,
            },
            Self {
                item_id: item_id.to_string(),
                content_index: u64::MAX,
                annotation_index: u64::MAX,
            },
        )
    }
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
            text_values: BTreeMap::new(),
            reasoning_started: BTreeSet::new(),
            reasoning_ended: BTreeSet::new(),
            reasoning_values: BTreeMap::new(),
            tool_inputs: BTreeMap::new(),
            refusals: BTreeMap::new(),
            refusal_ended: BTreeSet::new(),
            emitted_refusals: BTreeSet::new(),
            citations: BTreeMap::new(),
            turn_budget: ResponsesTurnBudget::default(),
            wire_dialect: ResponsesWireDialect::default(),
            terminal_response: None,
            response_diagnostics: ResponseDiagnostics::default(),
            replay_status: ResponsesReplayStatus::default(),
        }
    }

    /// Select the provider-owned wire normalization descriptor.
    pub fn with_wire_dialect(mut self, dialect: ResponsesWireDialect) -> Self {
        self.wire_dialect = dialect;
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
        if !event.kind().is_terminal() {
            self.observe_sequence(event.sequence_number())?;
        }
        let portable_events = self.decode_event(&event)?;
        self.lifecycle
            .record(&portable_events)
            .map_err(Error::from)?;
        Ok(DecodedResponsesStreamFrame {
            native: event,
            portable_events,
            replay_status: self.replay_status,
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

    /// Return the bounded native replay status observed for this turn.
    pub fn replay_status(&self) -> ResponsesReplayStatus {
        self.replay_status
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
        let added = self.items.get(&output_index).cloned().ok_or_else(|| {
            protocol_error("OpenAI Responses completed an item that was never added")
        })?;
        compare_item_transition(&added, &item, false, &mut self.replay_status)?;

        let mut events = Vec::new();
        match &item {
            OutputItem::Message(message) => self.finish_message(message, &mut events)?,
            OutputItem::Reasoning(reasoning) => {
                self.finish_reasoning_item(item_id, reasoning, &mut events)?;
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
        self.items.insert(output_index, item);
        self.completed_items.insert(output_index);
        Ok(events)
    }

    fn text_delta(&mut self, event: &StreamEventWire) -> Result<Vec<LanguageStreamEvent>, Error> {
        let item_id = required_str(event, "item_id")?;
        self.ensure_known_item_kind(item_id, StreamItemKind::Message)?;
        let content_index = optional_u64(event, "content_index").unwrap_or(0);
        let lane = ContentLaneId::new(item_id, ContentLaneKind::Text, content_index);
        if self.text_ended.contains(&lane) {
            return Err(protocol_error(
                "OpenAI text delta arrived after the content lane completed",
            ));
        }
        let delta = required_str(event, "delta")?.to_string();
        self.text_values
            .entry(lane.clone())
            .or_default()
            .push_str(&delta);
        let mut events = Vec::new();
        let event_id = lane.event_id();
        if self.text_started.insert(lane) {
            events.push(LanguageStreamEvent::TextStart {
                id: event_id.clone(),
            });
        }
        events.push(LanguageStreamEvent::TextDelta {
            id: event_id,
            delta,
        });
        Ok(events)
    }

    fn text_done(&mut self, event: &StreamEventWire) -> Result<Vec<LanguageStreamEvent>, Error> {
        let item_id = required_str(event, "item_id")?;
        self.ensure_known_item_kind(item_id, StreamItemKind::Message)?;
        let content_index = optional_u64(event, "content_index").unwrap_or(0);
        let lane = ContentLaneId::new(item_id, ContentLaneKind::Text, content_index);
        if self.text_ended.contains(&lane) {
            return Err(protocol_error(
                "OpenAI text completion repeated a completed content lane",
            ));
        }
        let mut events = Vec::new();
        let completed = event.field("text").and_then(Value::as_str);
        if let (Some(streamed), Some(completed)) = (self.text_values.get(&lane), completed)
            && streamed != completed
        {
            return Err(protocol_error(
                "OpenAI text completion disagreed with its streamed deltas",
            ));
        }
        let event_id = lane.event_id();
        if self.text_started.insert(lane.clone()) {
            let text = completed
                .ok_or_else(|| protocol_error("OpenAI text completion omitted its final text"))?;
            self.text_values.insert(lane.clone(), text.to_string());
            events.push(LanguageStreamEvent::TextStart {
                id: event_id.clone(),
            });
            events.push(LanguageStreamEvent::TextDelta {
                id: event_id.clone(),
                delta: text.to_string(),
            });
        }
        self.text_ended.insert(lane);
        events.push(LanguageStreamEvent::TextEnd { id: event_id });
        Ok(events)
    }

    fn reasoning_delta(
        &mut self,
        event: &StreamEventWire,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        let item_id = required_str(event, "item_id")?;
        self.ensure_known_item_kind(item_id, StreamItemKind::Reasoning)?;
        let index = optional_u64(event, "summary_index")
            .or_else(|| optional_u64(event, "content_index"))
            .unwrap_or(0);
        let kind = if event.kind.contains("summary") {
            ContentLaneKind::ReasoningSummary
        } else {
            ContentLaneKind::ReasoningContent
        };
        let lane = ContentLaneId::new(item_id, kind, index);
        if self.reasoning_ended.contains(&lane) {
            return Err(protocol_error(
                "OpenAI reasoning delta arrived after the content lane completed",
            ));
        }
        let delta = required_str(event, "delta")?.to_string();
        self.reasoning_values
            .entry(lane.clone())
            .or_default()
            .push_str(&delta);
        let mut events = Vec::new();
        self.start_reasoning(&lane, &mut events);
        events.push(LanguageStreamEvent::ReasoningDelta {
            id: lane.event_id(),
            delta,
        });
        Ok(events)
    }

    fn reasoning_done(
        &mut self,
        event: &StreamEventWire,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        let item_id = required_str(event, "item_id")?;
        self.ensure_known_item_kind(item_id, StreamItemKind::Reasoning)?;
        let index = optional_u64(event, "summary_index")
            .or_else(|| optional_u64(event, "content_index"))
            .unwrap_or(0);
        let kind = if event.kind.contains("summary") {
            ContentLaneKind::ReasoningSummary
        } else {
            ContentLaneKind::ReasoningContent
        };
        let lane = ContentLaneId::new(item_id, kind, index);
        if self.reasoning_ended.contains(&lane) {
            return Err(protocol_error(
                "OpenAI reasoning completion repeated a completed content lane",
            ));
        }
        let mut events = Vec::new();
        let completed = event.field("text").and_then(Value::as_str);
        if let (Some(streamed), Some(completed)) = (self.reasoning_values.get(&lane), completed)
            && streamed != completed
        {
            return Err(protocol_error(
                "OpenAI reasoning completion disagreed with its streamed deltas",
            ));
        }
        if !self.reasoning_started.contains(&lane)
            && let Some(text) = completed
        {
            self.reasoning_values.insert(lane.clone(), text.to_string());
            self.start_reasoning(&lane, &mut events);
            events.push(LanguageStreamEvent::ReasoningDelta {
                id: lane.event_id(),
                delta: text.to_string(),
            });
        }
        self.end_reasoning(&lane, &mut events);
        Ok(events)
    }

    fn start_reasoning(&mut self, lane: &ContentLaneId, events: &mut Vec<LanguageStreamEvent>) {
        if self.reasoning_started.insert(lane.clone()) {
            events.push(LanguageStreamEvent::ReasoningStart {
                id: lane.event_id(),
            });
        }
    }

    fn end_reasoning(&mut self, lane: &ContentLaneId, events: &mut Vec<LanguageStreamEvent>) {
        if self.reasoning_started.contains(lane) && self.reasoning_ended.insert(lane.clone()) {
            events.push(LanguageStreamEvent::ReasoningEnd {
                id: lane.event_id(),
            });
        }
    }

    fn finish_reasoning_item(
        &mut self,
        item_id: &str,
        reasoning: &super::wire::ReasoningItemWire,
        events: &mut Vec<LanguageStreamEvent>,
    ) -> Result<(), Error> {
        self.validate_reasoning_lanes(item_id, reasoning)?;
        for (index, part) in reasoning.summary.iter().enumerate() {
            self.finish_reasoning_part(
                item_id,
                ContentLaneKind::ReasoningSummary,
                index as u64,
                &part.text,
                events,
            )?;
        }
        for (index, part) in reasoning.content.iter().enumerate() {
            self.finish_reasoning_part(
                item_id,
                ContentLaneKind::ReasoningContent,
                index as u64,
                &part.text,
                events,
            )?;
        }
        Ok(())
    }

    fn finish_reasoning_part(
        &mut self,
        item_id: &str,
        kind: ContentLaneKind,
        index: u64,
        text: &str,
        events: &mut Vec<LanguageStreamEvent>,
    ) -> Result<(), Error> {
        let lane = ContentLaneId::new(item_id, kind, index);
        let streamed = self.reasoning_values.get(&lane).map(String::as_str);
        if self.reasoning_ended.contains(&lane) && streamed != Some(text) {
            return Err(protocol_error(
                "OpenAI reasoning output item disagreed with a completed content lane",
            ));
        }
        if let Some(streamed) = streamed
            && !text.starts_with(streamed)
        {
            return Err(protocol_error(
                "OpenAI reasoning output item disagreed with its streamed deltas",
            ));
        }
        if !self.reasoning_started.contains(&lane) {
            self.start_reasoning(&lane, events);
            events.push(LanguageStreamEvent::ReasoningDelta {
                id: lane.event_id(),
                delta: text.to_string(),
            });
        } else if let Some(streamed) = streamed
            && streamed.len() < text.len()
        {
            events.push(LanguageStreamEvent::ReasoningDelta {
                id: lane.event_id(),
                delta: text[streamed.len()..].to_string(),
            });
        }
        self.reasoning_values.insert(lane.clone(), text.to_string());
        self.end_reasoning(&lane, events);
        Ok(())
    }

    fn refusal_delta(
        &mut self,
        event: &StreamEventWire,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        let item_id = required_str(event, "item_id")?;
        self.ensure_known_item_kind(item_id, StreamItemKind::Message)?;
        let content_index = optional_u64(event, "content_index").unwrap_or(0);
        let lane = ContentLaneId::new(item_id, ContentLaneKind::Refusal, content_index);
        if self.refusal_ended.contains(&lane) {
            return Err(protocol_error(
                "OpenAI refusal delta arrived after the content lane completed",
            ));
        }
        self.refusals
            .entry(lane)
            .or_default()
            .push_str(required_str(event, "delta")?);
        Ok(Vec::new())
    }

    fn refusal_done(&mut self, event: &StreamEventWire) -> Result<Vec<LanguageStreamEvent>, Error> {
        let item_id = required_str(event, "item_id")?;
        self.ensure_known_item_kind(item_id, StreamItemKind::Message)?;
        let content_index = optional_u64(event, "content_index").unwrap_or(0);
        let lane = ContentLaneId::new(item_id, ContentLaneKind::Refusal, content_index);
        if !self.refusal_ended.insert(lane.clone()) {
            return Err(protocol_error(
                "OpenAI refusal completion repeated a completed content lane",
            ));
        }
        let completed = event.field("refusal").and_then(Value::as_str);
        if let (Some(streamed), Some(completed)) = (self.refusals.get(&lane), completed)
            && streamed != completed
        {
            return Err(protocol_error(
                "OpenAI refusal completion disagreed with its streamed deltas",
            ));
        }
        let reason = completed
            .map(str::to_string)
            .or_else(|| self.refusals.get(&lane).cloned());
        self.emitted_refusals.insert(lane);
        Ok(vec![LanguageStreamEvent::Refusal { reason }])
    }

    fn annotation_added(
        &mut self,
        event: &ResponsesStreamEvent,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        let item_id = required_str(event.wire(), "item_id")?;
        self.ensure_known_item_kind(item_id, StreamItemKind::Message)?;
        let content_index = optional_u64(event.wire(), "content_index").unwrap_or(0);
        let annotation_index = required_u64(event.wire(), "annotation_index")?;
        let annotation_position = usize::try_from(annotation_index).map_err(|_| {
            protocol_error("OpenAI Responses annotation index exceeded the platform limit")
        })?;
        let annotation = event
            .annotation()
            .ok_or_else(|| protocol_error("OpenAI citation event omitted its annotation"))?;
        let id = CitationId {
            item_id: item_id.to_string(),
            content_index,
            annotation_index,
        };
        if self.citations.contains_key(&id) {
            return Err(protocol_error(
                "OpenAI citation event repeated an annotation identity",
            ));
        }
        self.citations.insert(id, annotation.clone());
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
        self.ensure_known_item_kind(
            item_id,
            if custom {
                StreamItemKind::CustomToolCall
            } else {
                StreamItemKind::FunctionCall
            },
        )?;
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
        if assembly.done {
            return Err(protocol_error(
                "OpenAI tool-input delta arrived after completion",
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
        self.ensure_known_item_kind(
            item_id,
            if custom {
                StreamItemKind::CustomToolCall
            } else {
                StreamItemKind::FunctionCall
            },
        )?;
        let field = if custom { "input" } else { "arguments" };
        let completed = required_str(event, field)?.to_string();
        ensure_tool_input_limit(&completed)?;
        let assembly = self.tool_inputs.get_mut(item_id).ok_or_else(|| {
            protocol_error("OpenAI tool-input completion preceded its output item")
        })?;
        if assembly.done {
            return Err(protocol_error(
                "OpenAI tool-input completion was emitted more than once",
            ));
        }
        if assembly.custom != custom {
            return Err(protocol_error(
                "OpenAI tool-input completion changed its tool kind",
            ));
        }
        let mut custom_conflict = false;
        let suffix = if assembly.input == completed {
            None
        } else if let Some(suffix) = completed.strip_prefix(&assembly.input) {
            (!suffix.is_empty()).then(|| suffix.to_string())
        } else if !custom && function_arguments_equal(&assembly.input, &completed)? {
            None
        } else if custom {
            custom_conflict = true;
            None
        } else {
            return Err(protocol_error(
                "OpenAI tool-input completion disagreed with its streamed deltas",
            ));
        };
        let mut events = Vec::new();
        if !custom && let Some(delta) = suffix {
            events.push(LanguageStreamEvent::ToolInputDelta {
                id: assembly.call_id.clone(),
                delta,
            });
        }
        assembly.input = completed;
        assembly.done = true;
        if custom_conflict {
            self.replay_status.record_provider_item_conflict();
        }
        Ok(events)
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
                done: false,
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
            .get_mut(item_id)
            .ok_or_else(|| protocol_error("OpenAI function call completed before it was added"))?;
        if assembly.call_id != call.call_id || assembly.name != call.name {
            return Err(protocol_error("OpenAI function call changed its identity"));
        }
        ensure_tool_input_limit(&call.arguments)?;
        let suffix = if assembly.input == call.arguments {
            None
        } else if !assembly.done {
            if let Some(suffix) = call.arguments.strip_prefix(&assembly.input) {
                (!suffix.is_empty()).then(|| suffix.to_string())
            } else if function_arguments_equal(&assembly.input, &call.arguments)? {
                None
            } else {
                return Err(protocol_error(
                    "OpenAI function call arguments disagreed with streamed deltas",
                ));
            }
        } else if function_arguments_equal(&assembly.input, &call.arguments)? {
            None
        } else {
            return Err(protocol_error(
                "OpenAI function call arguments disagreed with streamed deltas",
            ));
        };
        if !assembly.custom
            && let Some(delta) = suffix
        {
            events.push(LanguageStreamEvent::ToolInputDelta {
                id: assembly.call_id.clone(),
                delta,
            });
        }
        assembly.input.clone_from(&call.arguments);
        assembly.done = true;
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
        &mut self,
        item_id: &str,
        call: &super::wire::CustomToolCallItemWire,
    ) -> Result<(), Error> {
        ensure_tool_input_limit(&call.input)?;
        let assembly = self.tool_inputs.get_mut(item_id).ok_or_else(|| {
            protocol_error("OpenAI custom tool call completed before it was added")
        })?;
        if !assembly.custom {
            return Err(protocol_error(
                "OpenAI custom tool call changed its established tool kind",
            ));
        }
        let input_conflict = (assembly.done && assembly.input != call.input)
            || (!assembly.done && !call.input.starts_with(&assembly.input));
        assembly.input.clone_from(&call.input);
        assembly.done = true;
        if input_conflict {
            self.replay_status.record_provider_item_conflict();
        }
        Ok(())
    }

    fn validate_message_lanes(&self, message: &super::wire::MessageItemWire) -> Result<(), Error> {
        let mut expected = BTreeSet::new();
        for (content_index, part) in message.content.iter().enumerate() {
            let content_index = u64::try_from(content_index).map_err(|_| {
                protocol_error("OpenAI message content index exceeded the platform limit")
            })?;
            match part {
                OutputContentPart::Text(_) => {
                    expected.insert(ContentLaneId::new(
                        &message.id,
                        ContentLaneKind::Text,
                        content_index,
                    ));
                }
                OutputContentPart::Refusal(_) => {
                    expected.insert(ContentLaneId::new(
                        &message.id,
                        ContentLaneKind::Refusal,
                        content_index,
                    ));
                }
                OutputContentPart::Unknown(_) => {}
            }
        }

        let (lower, upper) = ContentLaneId::bounds(&message.id);
        ensure_observed_lanes_are_covered(
            self.text_started.range(lower.clone()..=upper.clone()),
            &expected,
        )?;
        ensure_observed_lanes_are_covered(
            self.text_ended.range(lower.clone()..=upper.clone()),
            &expected,
        )?;
        ensure_observed_lanes_are_covered(
            self.text_values
                .range(lower.clone()..=upper.clone())
                .map(|(lane, _)| lane),
            &expected,
        )?;
        ensure_observed_lanes_are_covered(
            self.refusals
                .range(lower.clone()..=upper.clone())
                .map(|(lane, _)| lane),
            &expected,
        )?;
        ensure_observed_lanes_are_covered(self.refusal_ended.range(lower..=upper), &expected)?;

        let (lower, upper) = CitationId::bounds(&message.id);
        for (identity, streamed) in self.citations.range(lower..=upper) {
            let content_index = usize::try_from(identity.content_index).map_err(|_| {
                protocol_error("OpenAI citation content index exceeded the platform limit")
            })?;
            let annotation_index = usize::try_from(identity.annotation_index).map_err(|_| {
                protocol_error("OpenAI citation annotation index exceeded the platform limit")
            })?;
            let Some(OutputContentPart::Text(text)) = message.content.get(content_index) else {
                return Err(protocol_error(
                    "OpenAI message item omitted a content lane with streamed citations",
                ));
            };
            if text.annotations.get(annotation_index) != Some(streamed) {
                return Err(protocol_error(
                    "OpenAI message item omitted or changed a streamed citation",
                ));
            }
        }
        Ok(())
    }

    fn validate_reasoning_lanes(
        &self,
        item_id: &str,
        reasoning: &super::wire::ReasoningItemWire,
    ) -> Result<(), Error> {
        let mut expected = BTreeSet::new();
        for (index, _) in reasoning.summary.iter().enumerate() {
            expected.insert(ContentLaneId::new(
                item_id,
                ContentLaneKind::ReasoningSummary,
                u64::try_from(index).map_err(|_| {
                    protocol_error("OpenAI reasoning index exceeded the platform limit")
                })?,
            ));
        }
        for (index, _) in reasoning.content.iter().enumerate() {
            expected.insert(ContentLaneId::new(
                item_id,
                ContentLaneKind::ReasoningContent,
                u64::try_from(index).map_err(|_| {
                    protocol_error("OpenAI reasoning index exceeded the platform limit")
                })?,
            ));
        }
        let (lower, upper) = ContentLaneId::bounds(item_id);
        ensure_observed_lanes_are_covered(
            self.reasoning_started.range(lower.clone()..=upper.clone()),
            &expected,
        )?;
        ensure_observed_lanes_are_covered(
            self.reasoning_ended.range(lower.clone()..=upper.clone()),
            &expected,
        )?;
        ensure_observed_lanes_are_covered(
            self.reasoning_values
                .range(lower..=upper)
                .map(|(lane, _)| lane),
            &expected,
        )
    }

    fn finish_message(
        &mut self,
        message: &super::wire::MessageItemWire,
        events: &mut Vec<LanguageStreamEvent>,
    ) -> Result<(), Error> {
        self.validate_message_lanes(message)?;
        for (content_index, part) in message.content.iter().enumerate() {
            let content_index = u64::try_from(content_index).map_err(|_| {
                protocol_error("OpenAI message content index exceeded the platform limit")
            })?;
            match part {
                OutputContentPart::Text(text) => {
                    let lane =
                        ContentLaneId::new(&message.id, ContentLaneKind::Text, content_index);
                    let streamed = self.text_values.get(&lane).map(String::as_str);
                    if self.text_ended.contains(&lane) && streamed != Some(text.text.as_str()) {
                        return Err(protocol_error(
                            "OpenAI message item disagreed with a completed text lane",
                        ));
                    }
                    let suffix = match streamed {
                        Some(streamed) => {
                            Some(text.text.strip_prefix(streamed).ok_or_else(|| {
                                protocol_error(
                                    "OpenAI message item disagreed with its streamed text deltas",
                                )
                            })?)
                        }
                        None => None,
                    };
                    let event_id = lane.event_id();
                    if self.text_started.insert(lane.clone()) {
                        events.push(LanguageStreamEvent::TextStart {
                            id: event_id.clone(),
                        });
                        events.push(LanguageStreamEvent::TextDelta {
                            id: event_id.clone(),
                            delta: text.text.clone(),
                        });
                    } else if let Some(suffix) = suffix
                        && !suffix.is_empty()
                    {
                        events.push(LanguageStreamEvent::TextDelta {
                            id: event_id.clone(),
                            delta: suffix.to_string(),
                        });
                    }
                    self.text_values.insert(lane.clone(), text.text.clone());
                    if self.text_ended.insert(lane) {
                        events.push(LanguageStreamEvent::TextEnd { id: event_id });
                    }
                    for (annotation_index, annotation) in text.annotations.iter().enumerate() {
                        let annotation_index = u64::try_from(annotation_index).map_err(|_| {
                            protocol_error(
                                "OpenAI message annotation index exceeded the platform limit",
                            )
                        })?;
                        let identity = CitationId {
                            item_id: message.id.clone(),
                            content_index,
                            annotation_index,
                        };
                        if let Some(streamed) = self.citations.get(&identity) {
                            if streamed != annotation {
                                return Err(protocol_error(
                                    "OpenAI message item changed a streamed citation",
                                ));
                            }
                        } else {
                            self.citations.insert(identity, annotation.clone());
                            events.push(LanguageStreamEvent::Citation(project_citation(
                                &message.id,
                                usize::try_from(annotation_index).map_err(|_| {
                                    protocol_error(
                                        "OpenAI message annotation index exceeded the platform limit",
                                    )
                                })?,
                                annotation,
                            )));
                        }
                    }
                }
                OutputContentPart::Refusal(refusal) => {
                    let lane =
                        ContentLaneId::new(&message.id, ContentLaneKind::Refusal, content_index);
                    let streamed = self.refusals.get(&lane).map(String::as_str);
                    if self.refusal_ended.contains(&lane)
                        && streamed != Some(refusal.refusal.as_str())
                    {
                        return Err(protocol_error(
                            "OpenAI message item disagreed with a completed refusal lane",
                        ));
                    }
                    if let Some(streamed) = streamed
                        && refusal.refusal.strip_prefix(streamed).is_none()
                    {
                        return Err(protocol_error(
                            "OpenAI message item disagreed with its streamed refusal deltas",
                        ));
                    }
                    self.refusals.insert(lane.clone(), refusal.refusal.clone());
                    self.refusal_ended.insert(lane.clone());
                    if self.emitted_refusals.insert(lane) {
                        events.push(LanguageStreamEvent::Refusal {
                            reason: Some(refusal.refusal.clone()),
                        });
                    }
                }
                OutputContentPart::Unknown(_) => {}
            }
        }
        Ok(())
    }

    fn terminal_event(
        &mut self,
        event: &ResponsesStreamEvent,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        let mut projected_value = event.raw_response().cloned().ok_or_else(|| {
            protocol_error("OpenAI Responses terminal event omitted its response")
        })?;
        let alignment = self.reconcile_terminal_value(&mut projected_value)?;
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
        let terminal_status = projected.status.clone();
        let completion_events =
            self.reconcile_terminal_items(&mut projected, &alignment, &terminal_status)?;
        let mut events = self.observe_identity(&projected)?;
        events.extend(completion_events);
        self.replay_status.settle_with_terminal_resource();
        let decoded = decode_response_wire_with_replay(
            projected.clone(),
            &self.scope,
            &self.requested_model,
            self.replay_status.is_available(),
        )?;
        if let Some(usage) = &projected.usage {
            events.push(LanguageStreamEvent::Usage(UsageUpdate::snapshot(
                decode_usage(usage),
            )));
        }
        let (_, portable) = decoded.into_parts();
        let terminal = match &projected.status {
            ResponseStatus::Completed | ResponseStatus::Incomplete => {
                let response = portable.map_err(|error| error.into_error())?;
                StreamTerminal::Completed {
                    response: Box::new(response),
                }
            }
            ResponseStatus::Cancelled => {
                let (_, partial) = match portable {
                    Err(error) => error.into_parts(),
                    Ok(_) => {
                        return Err(protocol_error(
                            "cancelled Responses resource produced a successful portable response",
                        ));
                    }
                };
                StreamTerminal::Cancelled {
                    reason: "OpenAI cancelled the Responses generation".to_string(),
                    partial,
                }
            }
            ResponseStatus::Failed => {
                let (_, partial) = match portable {
                    Err(error) => error.into_parts(),
                    Ok(_) => {
                        return Err(protocol_error(
                            "failed Responses resource produced a successful portable response",
                        ));
                    }
                };
                StreamTerminal::Failed {
                    error: failed_response_error(
                        &projected,
                        &self.scope,
                        &self.requested_model,
                        self.response_diagnostics.clone(),
                    ),
                    partial,
                }
            }
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

    fn reconcile_terminal_value(
        &mut self,
        response: &mut Value,
    ) -> Result<TerminalAlignment, Error> {
        let object = response.as_object_mut().ok_or_else(|| {
            protocol_error("OpenAI Responses terminal response must be a JSON object")
        })?;
        let output = object
            .entry("output")
            .or_insert_with(|| Value::Array(Vec::new()))
            .as_array_mut()
            .ok_or_else(|| protocol_error("OpenAI Responses terminal output must be an array"))?;
        self.turn_budget.ensure_output_items(output.len())?;

        let alignment = self.align_terminal_output(output)?;
        for (position, terminal) in output.iter_mut().enumerate() {
            let Some(output_index) = alignment.streamed_index(position) else {
                continue;
            };
            let terminal_object = terminal.as_object_mut().ok_or_else(|| {
                protocol_error("OpenAI Responses terminal output item must be an object")
            })?;
            let streamed = self.items.get(&output_index).ok_or_else(|| {
                protocol_error("OpenAI terminal alignment referenced an unknown streamed item")
            })?;
            reconcile_terminal_item_value(
                terminal_object,
                streamed,
                self.completed_items.contains(&output_index),
                self.wire_dialect,
            )?;
        }
        Ok(alignment)
    }

    fn align_terminal_output(&mut self, output: &[Value]) -> Result<TerminalAlignment, Error> {
        let mut call_indices = BTreeMap::<(String, String), Option<u64>>::new();
        for (output_index, item) in &self.items {
            let Some(call_id) = item.call_id() else {
                continue;
            };
            if !has_stable_call_identity(item.kind()) {
                continue;
            }
            call_indices
                .entry((item.kind().to_string(), call_id.to_string()))
                .and_modify(|candidate| *candidate = None)
                .or_insert(Some(*output_index));
        }

        let mut alignment = TerminalAlignment {
            terminal_to_streamed: vec![None; output.len()],
            streamed_to_terminal: BTreeMap::new(),
        };

        // First bind only explicit item and call identities. Position fallback runs after every
        // explicit anchor is known, so omitted leading items cannot shift a later message onto the
        // wrong streamed output index.
        for (position, terminal) in output.iter().enumerate() {
            let terminal = terminal.as_object().ok_or_else(|| {
                protocol_error("OpenAI Responses terminal output item must be an object")
            })?;
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

            let by_id = id.and_then(|id| self.item_indices.get(id).copied());
            if let Some(output_index) = by_id
                && self
                    .items
                    .get(&output_index)
                    .is_none_or(|item| item.kind() != kind)
            {
                return Err(protocol_error(
                    "OpenAI terminal response changed a streamed output item type",
                ));
            }
            let by_call = if has_stable_call_identity(kind) {
                call_id
                    .and_then(|call_id| call_indices.get(&(kind.to_string(), call_id.to_string())))
                    .copied()
                    .flatten()
            } else {
                None
            };
            if by_id.is_some() && by_call.is_some() && by_id != by_call {
                return Err(protocol_error(
                    "OpenAI terminal response combined conflicting output identities",
                ));
            }

            let Some(output_index) = by_id.or(by_call) else {
                continue;
            };
            record_terminal_alignment(&mut alignment, output_index, position)?;
        }

        self.validate_terminal_alignment_order(&alignment)?;

        for (position, terminal) in output.iter().enumerate() {
            if alignment.streamed_index(position).is_some() {
                continue;
            }
            let terminal = terminal.as_object().ok_or_else(|| {
                protocol_error("OpenAI Responses terminal output item must be an object")
            })?;
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
            let lower = alignment.terminal_to_streamed[..position]
                .iter()
                .rev()
                .flatten()
                .next()
                .copied();
            let upper = alignment.terminal_to_streamed[position.saturating_add(1)..]
                .iter()
                .flatten()
                .next()
                .copied();
            let candidates = self
                .items
                .iter()
                .filter(|(output_index, item)| {
                    !alignment.streamed_to_terminal.contains_key(output_index)
                        && item.kind() == kind
                        && lower.is_none_or(|lower| **output_index > lower)
                        && upper.is_none_or(|upper| **output_index < upper)
                })
                .map(|(output_index, _)| *output_index)
                .collect::<Vec<_>>();
            let has_unaligned_same_kind = self.items.iter().any(|(output_index, item)| {
                !alignment.streamed_to_terminal.contains_key(output_index) && item.kind() == kind
            });

            let allows_position_identity = (kind == "message"
                && (id.is_some() || self.wire_dialect.allows_message_position_identity()))
                || kind == "reasoning";
            if !allows_position_identity {
                if is_portable_kind(kind) && (has_unaligned_same_kind || !candidates.is_empty()) {
                    return Err(protocol_error(
                        "OpenAI terminal response could not uniquely align a portable output item",
                    ));
                }
                continue;
            }

            match candidates.as_slice() {
                [] => {}
                [output_index] => {
                    record_terminal_alignment(&mut alignment, *output_index, position)?;
                }
                _ => {
                    return Err(protocol_error(
                        "OpenAI terminal response had an ambiguous positional output identity",
                    ));
                }
            }
        }

        self.validate_terminal_alignment_order(&alignment)?;
        Ok(alignment)
    }

    fn validate_terminal_alignment_order(
        &mut self,
        alignment: &TerminalAlignment,
    ) -> Result<(), Error> {
        let mut previous = None::<(u64, usize)>;
        for (output_index, terminal_index) in &alignment.streamed_to_terminal {
            if let Some((previous_output, previous_terminal)) = previous
                && *terminal_index <= previous_terminal
            {
                let portable = self
                    .items
                    .get(&previous_output)
                    .is_some_and(is_portable_item)
                    || self.items.get(output_index).is_some_and(is_portable_item);
                if portable {
                    return Err(protocol_error(
                        "OpenAI terminal response reordered portable output items",
                    ));
                }
                self.replay_status.record_provider_item_conflict();
            }
            previous = Some((*output_index, *terminal_index));
        }
        Ok(())
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
        self.replay_status.settle_without_terminal_resource();
        Ok(vec![LanguageStreamEvent::Terminal(
            StreamTerminal::Failed {
                error,
                partial: None,
            },
        )])
    }

    fn opaque_stream_event(
        &mut self,
        _event: &StreamEventWire,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        // Native stream consumers already receive the exact bounded event. Portable streams do not
        // publish replay fragments before the turn's replay eligibility is known.
        Ok(Vec::new())
    }

    fn ensure_known_item_kind(&self, item_id: &str, expected: StreamItemKind) -> Result<(), Error> {
        let output_index = self.item_indices.get(item_id).ok_or_else(|| {
            protocol_error("OpenAI Responses delta referenced an unknown output item")
        })?;
        if self.completed_items.contains(output_index) {
            return Err(protocol_error(
                "OpenAI Responses delta referenced a completed output item",
            ));
        }
        let item = self.items.get(output_index).ok_or_else(|| {
            protocol_error("OpenAI Responses delta referenced an untracked output item")
        })?;
        if !expected.matches(item) {
            return Err(protocol_error(
                "OpenAI Responses event targeted an incompatible output item type",
            ));
        }
        Ok(())
    }

    fn reconcile_terminal_items(
        &mut self,
        response: &mut ResponseWire,
        alignment: &TerminalAlignment,
        status: &ResponseStatus,
    ) -> Result<Vec<LanguageStreamEvent>, Error> {
        let mut missing_portable_items = Vec::new();
        let mut completion_events = Vec::new();
        let tracked = std::mem::take(&mut self.items);

        for (output_index, streamed) in tracked {
            let streamed_complete = self.completed_items.contains(&output_index);
            if let Some(index) = alignment.terminal_index(output_index) {
                let terminal = response.output.get(index).ok_or_else(|| {
                    protocol_error("OpenAI terminal alignment exceeded the output array")
                })?;
                if streamed_complete {
                    compare_terminal_item(&streamed, terminal, true, &mut self.replay_status)?;
                } else if is_portable_item(&streamed) {
                    if matches!(status, ResponseStatus::Failed | ResponseStatus::Cancelled)
                        && matches!(&streamed, OutputItem::FunctionCall(_))
                    {
                        compare_item_transition(
                            &streamed,
                            terminal,
                            true,
                            &mut self.replay_status,
                        )?;
                    } else {
                        self.complete_portable_item(&streamed, terminal, &mut completion_events)?;
                    }
                } else {
                    compare_item_transition(&streamed, terminal, true, &mut self.replay_status)?;
                }
            } else if streamed_complete {
                match streamed {
                    item @ (OutputItem::FunctionCall(_)
                    | OutputItem::Message(_)
                    | OutputItem::Reasoning(_)) => {
                        missing_portable_items.push((output_index, item));
                    }
                    OutputItem::CustomToolCall(_)
                    | OutputItem::Program(_)
                    | OutputItem::ProgramOutput(_)
                    | OutputItem::ProviderTool(_)
                    | OutputItem::Unknown(_) => {
                        self.replay_status.record_provider_item_conflict();
                    }
                }
            } else if is_portable_item(&streamed) {
                if matches!(status, ResponseStatus::Failed | ResponseStatus::Cancelled) {
                    self.replay_status.record_provider_item_conflict();
                } else {
                    return Err(protocol_error(
                        "OpenAI terminal response omitted an in-progress portable output item",
                    ));
                }
            } else {
                self.replay_status.record_provider_item_conflict();
            }
        }

        let mut insertions = BTreeMap::<usize, Vec<OutputItem>>::new();
        if !missing_portable_items.is_empty() {
            let mut terminal_only_prefix = Vec::with_capacity(response.output.len() + 1);
            terminal_only_prefix.push(0usize);
            let mut related_positions = BTreeMap::<String, Option<usize>>::new();
            for (position, item) in response.output.iter().enumerate() {
                let terminal_only = alignment.streamed_index(position).is_none();
                terminal_only_prefix.push(
                    terminal_only_prefix[position].saturating_add(usize::from(terminal_only)),
                );
                if terminal_only && let Some(call_id) = item.call_id() {
                    related_positions
                        .entry(call_id.to_string())
                        .and_modify(|candidate| *candidate = None)
                        .or_insert(Some(position));
                }
            }

            let mut gaps = BTreeMap::<(usize, usize), Vec<(u64, OutputItem)>>::new();
            for (output_index, item) in missing_portable_items {
                let start = alignment
                    .previous_terminal_index(output_index)
                    .map_or(0, |position| position.saturating_add(1));
                let end = alignment
                    .next_terminal_index(output_index)
                    .unwrap_or(response.output.len());
                let gap = if start <= end {
                    (start, end)
                } else {
                    self.replay_status.record_provider_item_conflict();
                    (0, response.output.len())
                };
                gaps.entry(gap).or_default().push((output_index, item));
            }

            for ((start, end), mut items) in gaps {
                items.sort_unstable_by_key(|(output_index, _)| *output_index);
                let terminal_only_count =
                    terminal_only_prefix[end].saturating_sub(terminal_only_prefix[start]);
                let related_position = match items.as_slice() {
                    [(_, item)] => item
                        .call_id()
                        .and_then(|call_id| related_positions.get(call_id))
                        .copied()
                        .flatten()
                        .filter(|position| (start..end).contains(position)),
                    _ => None,
                };
                let insert_at = related_position.unwrap_or(end);
                if terminal_only_count > usize::from(related_position.is_some()) {
                    self.replay_status.record_provider_item_conflict();
                }
                insertions
                    .entry(insert_at)
                    .or_default()
                    .extend(items.into_iter().map(|(_, item)| item));
            }
        }
        if !insertions.is_empty() {
            let terminal_output = std::mem::take(&mut response.output);
            let terminal_len = terminal_output.len();
            let inserted = insertions.values().map(Vec::len).sum::<usize>();
            let mut merged = Vec::with_capacity(terminal_output.len().saturating_add(inserted));
            for (position, terminal) in terminal_output.into_iter().enumerate() {
                if let Some(items) = insertions.remove(&position) {
                    merged.extend(items);
                }
                merged.push(terminal);
            }
            if let Some(items) = insertions.remove(&terminal_len) {
                merged.extend(items);
            }
            debug_assert!(insertions.is_empty());
            response.output = merged;
        }
        self.turn_budget
            .ensure_output_items(response.output.len())?;
        Ok(completion_events)
    }

    fn complete_portable_item(
        &mut self,
        streamed: &OutputItem,
        terminal: &OutputItem,
        events: &mut Vec<LanguageStreamEvent>,
    ) -> Result<(), Error> {
        let item_id = streamed
            .id()
            .ok_or_else(|| protocol_error("OpenAI portable output item omitted its identity"))?
            .to_string();
        if streamed.kind() != terminal.kind() {
            return Err(protocol_error(
                "OpenAI terminal response changed an in-progress output item type",
            ));
        }
        compare_item_transition(streamed, terminal, true, &mut self.replay_status)?;
        match terminal {
            OutputItem::Message(message) => self.finish_message(message, events)?,
            OutputItem::Reasoning(reasoning) => {
                self.finish_reasoning_item(&item_id, reasoning, events)?;
            }
            OutputItem::FunctionCall(call) => self.emit_function_call(&item_id, call, events)?,
            _ => unreachable!("complete_portable_item only accepts portable items"),
        }
        Ok(())
    }
}

fn reconcile_terminal_item_value(
    terminal: &mut Map<String, Value>,
    streamed: &OutputItem,
    streamed_complete: bool,
    dialect: ResponsesWireDialect,
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
            normalize_missing_terminal_field(
                terminal,
                streamed,
                "id",
                dialect.allows_message_position_identity(),
            )?;
            compare_required_terminal_field(terminal, streamed, "role")?;
            if streamed_complete {
                reconcile_message_content_field(terminal, streamed, dialect)?;
            }
        }
        Some("function_call") => {
            normalize_optional_terminal_identity(terminal, streamed, "id");
            for field in ["call_id", "name"] {
                compare_required_terminal_field(terminal, streamed, field)?;
            }
            if streamed_complete {
                compare_function_arguments_field(terminal, streamed)?;
                compare_optional_terminal_field(terminal, streamed, "namespace")?;
                compare_tool_caller_field(terminal, streamed)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn normalize_optional_terminal_identity(
    terminal: &mut Map<String, Value>,
    streamed: &Map<String, Value>,
    field: &'static str,
) {
    if !terminal.contains_key(field)
        && let Some(streamed) = streamed.get(field)
    {
        terminal.insert(field.to_string(), streamed.clone());
    }
}

fn reconcile_message_content_field(
    terminal: &mut Map<String, Value>,
    streamed: &Map<String, Value>,
    dialect: ResponsesWireDialect,
) -> Result<(), Error> {
    let streamed_parts = streamed
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| protocol_error("completed OpenAI message omitted content"))?;
    let terminal_parts = terminal
        .get_mut("content")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| protocol_error("terminal OpenAI message omitted content"))?;
    if terminal_parts.len() != streamed_parts.len() {
        return Err(protocol_error(
            "OpenAI terminal response changed completed message content length",
        ));
    }
    for (terminal_part, streamed_part) in terminal_parts.iter_mut().zip(streamed_parts) {
        reconcile_message_content_part(terminal_part, streamed_part, dialect)?;
    }
    Ok(())
}

fn reconcile_message_content_part(
    terminal: &mut Value,
    streamed: &Value,
    dialect: ResponsesWireDialect,
) -> Result<(), Error> {
    let terminal = terminal
        .as_object_mut()
        .ok_or_else(|| protocol_error("OpenAI terminal message content part must be an object"))?;
    let streamed = streamed
        .as_object()
        .ok_or_else(|| protocol_error("completed OpenAI message content part must be an object"))?;
    compare_required_terminal_field(terminal, streamed, "type")?;
    match terminal.get("type").and_then(Value::as_str) {
        Some("output_text") => {
            compare_required_terminal_field(terminal, streamed, "text")?;
            normalize_missing_terminal_field(
                terminal,
                streamed,
                "annotations",
                dialect.allows_omitted_message_annotations(),
            )
        }
        Some("refusal") => compare_required_terminal_field(terminal, streamed, "refusal"),
        _ => Ok(()),
    }
}

fn normalize_missing_terminal_field(
    terminal: &mut Map<String, Value>,
    streamed: &Map<String, Value>,
    field: &'static str,
    allow_missing: bool,
) -> Result<(), Error> {
    match (terminal.get(field), streamed.get(field)) {
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

fn compare_terminal_item(
    streamed: &OutputItem,
    terminal: &OutputItem,
    compare_semantics: bool,
    replay_status: &mut ResponsesReplayStatus,
) -> Result<(), Error> {
    if streamed.kind() != terminal.kind() {
        return Err(protocol_error(
            "OpenAI terminal response changed a streamed output item identity",
        ));
    }
    if streamed.id() != terminal.id() {
        match streamed {
            OutputItem::FunctionCall(_) | OutputItem::Message(_) | OutputItem::Reasoning(_) => {
                replay_status.record_item_identity_conflict()
            }
            _ => replay_status.record_provider_item_conflict(),
        }
    }
    if !compare_semantics {
        return Ok(());
    }

    match (streamed, terminal) {
        (OutputItem::FunctionCall(_), OutputItem::FunctionCall(_)) => {
            if !function_calls_semantically_equal(streamed, terminal)? {
                return Err(protocol_error(
                    "OpenAI terminal response changed completed streamed function semantics",
                ));
            }
        }
        (OutputItem::Message(streamed), OutputItem::Message(terminal)) => {
            let (portable_equal, native_conflict) = messages_semantically_equal(streamed, terminal);
            if !portable_equal {
                return Err(protocol_error(
                    "OpenAI terminal response changed completed streamed message semantics",
                ));
            }
            if native_conflict {
                replay_status.record_provider_item_conflict();
            }
        }
        (OutputItem::Reasoning(streamed), OutputItem::Reasoning(terminal)) => {
            if !reasoning_text_semantically_equal(&streamed.summary, &terminal.summary)
                || !reasoning_text_semantically_equal(&streamed.content, &terminal.content)
            {
                return Err(protocol_error(
                    "OpenAI terminal response changed portable reasoning text",
                ));
            }
            if streamed.encrypted_content != terminal.encrypted_content {
                replay_status.record_reasoning_state_conflict();
            }
        }
        (OutputItem::CustomToolCall(streamed), OutputItem::CustomToolCall(terminal)) => {
            if streamed != terminal {
                replay_status.record_provider_item_conflict();
            }
        }
        (OutputItem::Program(streamed), OutputItem::Program(terminal)) => {
            if streamed != terminal {
                replay_status.record_provider_item_conflict();
            }
        }
        (OutputItem::ProgramOutput(streamed), OutputItem::ProgramOutput(terminal)) => {
            if streamed != terminal {
                replay_status.record_provider_item_conflict();
            }
        }
        (streamed, terminal) if streamed != terminal => {
            replay_status.record_provider_item_conflict();
        }
        _ => {}
    }
    Ok(())
}

fn compare_item_transition(
    added: &OutputItem,
    completed: &OutputItem,
    allow_item_id_drift: bool,
    replay_status: &mut ResponsesReplayStatus,
) -> Result<(), Error> {
    if added.kind() != completed.kind() {
        return Err(protocol_error(
            "OpenAI completed output item changed its established identity",
        ));
    }
    if added.id() != completed.id() {
        if allow_item_id_drift {
            replay_status.record_item_identity_conflict();
        } else {
            return Err(protocol_error(
                "OpenAI completed output item changed its established identity",
            ));
        }
    }

    match (added, completed) {
        (OutputItem::Message(added), OutputItem::Message(completed)) => {
            if added.role != completed.role {
                return Err(protocol_error(
                    "OpenAI completed message changed its established role",
                ));
            }
        }
        (OutputItem::Reasoning(added), OutputItem::Reasoning(completed)) => {
            if stable_optional_changed(
                added.encrypted_content.as_deref(),
                completed.encrypted_content.as_deref(),
            ) {
                replay_status.record_reasoning_state_conflict();
            }
        }
        (OutputItem::FunctionCall(added), OutputItem::FunctionCall(completed)) => {
            if added.call_id != completed.call_id || added.name != completed.name {
                return Err(protocol_error(
                    "OpenAI completed function call changed its executable identity",
                ));
            }
            if stable_optional_changed(added.namespace.as_deref(), completed.namespace.as_deref())
                || (added.caller.is_some()
                    && !tool_callers_semantically_equal(
                        added.caller.as_ref(),
                        completed.caller.as_ref(),
                    ))
            {
                return Err(protocol_error(
                    "OpenAI completed function call changed its caller identity",
                ));
            }
        }
        (OutputItem::CustomToolCall(added), OutputItem::CustomToolCall(completed)) => {
            if added.call_id != completed.call_id
                || added.name != completed.name
                || stable_optional_changed(
                    added.namespace.as_deref(),
                    completed.namespace.as_deref(),
                )
                || stable_optional_changed(added.caller.as_ref(), completed.caller.as_ref())
                || (!added.input.is_empty() && !completed.input.starts_with(&added.input))
            {
                replay_status.record_provider_item_conflict();
            }
        }
        (OutputItem::Program(added), OutputItem::Program(completed)) => {
            if added.call_id != completed.call_id
                || added.code != completed.code
                || added.fingerprint != completed.fingerprint
            {
                replay_status.record_provider_item_conflict();
            }
        }
        (OutputItem::ProgramOutput(added), OutputItem::ProgramOutput(completed)) => {
            if added.call_id != completed.call_id {
                replay_status.record_provider_item_conflict();
            }
        }
        (OutputItem::ProviderTool(added), OutputItem::ProviderTool(completed)) => {
            if stable_optional_changed(added.call_id(), completed.call_id())
                || stable_optional_changed(added.caller(), completed.caller())
            {
                replay_status.record_provider_item_conflict();
            }
        }
        (OutputItem::Unknown(added), OutputItem::Unknown(completed)) => {
            if stable_optional_changed(added.call_id(), completed.call_id())
                || stable_optional_changed(added.caller(), completed.caller())
            {
                replay_status.record_provider_item_conflict();
            }
        }
        _ => {
            return Err(protocol_error(
                "OpenAI completed output item changed its established type",
            ));
        }
    }
    Ok(())
}

fn stable_optional_changed<T: PartialEq>(established: Option<T>, completed: Option<T>) -> bool {
    established.is_some() && established != completed
}

fn reasoning_text_semantically_equal(
    streamed: &[super::wire::ReasoningTextWire],
    terminal: &[super::wire::ReasoningTextWire],
) -> bool {
    streamed.len() == terminal.len()
        && streamed.iter().zip(terminal).all(|(streamed, terminal)| {
            streamed.kind == terminal.kind && streamed.text == terminal.text
        })
}

fn messages_semantically_equal(
    streamed: &super::wire::MessageItemWire,
    terminal: &super::wire::MessageItemWire,
) -> (bool, bool) {
    if streamed.role != terminal.role || streamed.content.len() != terminal.content.len() {
        return (false, false);
    }
    let mut native_conflict = false;
    for (streamed, terminal) in streamed.content.iter().zip(&terminal.content) {
        match (streamed, terminal) {
            (OutputContentPart::Text(streamed), OutputContentPart::Text(terminal)) => {
                if streamed.text != terminal.text {
                    return (false, false);
                }
                let (annotations_equal, annotation_native_conflict) =
                    annotations_semantically_equal(&streamed.annotations, &terminal.annotations);
                if !annotations_equal {
                    return (false, false);
                }
                native_conflict |= annotation_native_conflict;
            }
            (OutputContentPart::Refusal(streamed), OutputContentPart::Refusal(terminal)) => {
                if streamed.refusal != terminal.refusal {
                    return (false, false);
                }
            }
            (OutputContentPart::Unknown(streamed), OutputContentPart::Unknown(terminal)) => {
                if streamed.raw != terminal.raw {
                    native_conflict = true;
                }
            }
            _ => return (false, false),
        }
    }
    (true, native_conflict)
}

fn annotations_semantically_equal(
    streamed: &[AnnotationWire],
    terminal: &[AnnotationWire],
) -> (bool, bool) {
    if streamed.len() != terminal.len() {
        return (false, false);
    }
    let mut native_conflict = false;
    for (streamed, terminal) in streamed.iter().zip(terminal) {
        if streamed.kind != terminal.kind {
            return (false, false);
        }
        for field in [
            "file_id",
            "container_id",
            "url",
            "title",
            "filename",
            "start_index",
            "end_index",
        ] {
            if streamed.fields.get(field) != terminal.fields.get(field) {
                return (false, false);
            }
        }
        native_conflict |= streamed != terminal;
    }
    (true, native_conflict)
}

fn is_portable_item(item: &OutputItem) -> bool {
    matches!(
        item,
        OutputItem::Message(_) | OutputItem::Reasoning(_) | OutputItem::FunctionCall(_)
    )
}

fn is_portable_kind(kind: &str) -> bool {
    matches!(kind, "message" | "reasoning" | "function_call")
}

fn record_terminal_alignment(
    alignment: &mut TerminalAlignment,
    output_index: u64,
    terminal_index: usize,
) -> Result<(), Error> {
    if alignment.streamed_to_terminal.contains_key(&output_index)
        || alignment.terminal_to_streamed[terminal_index].is_some()
    {
        return Err(protocol_error(
            "OpenAI terminal response duplicated a streamed output identity",
        ));
    }
    alignment
        .streamed_to_terminal
        .insert(output_index, terminal_index);
    alignment.terminal_to_streamed[terminal_index] = Some(output_index);
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

fn ensure_observed_lanes_are_covered<'a>(
    observed: impl IntoIterator<Item = &'a ContentLaneId>,
    expected: &BTreeSet<ContentLaneId>,
) -> Result<(), Error> {
    if observed.into_iter().any(|lane| !expected.contains(lane)) {
        Err(protocol_error(
            "OpenAI terminal output item omitted or changed an observed content lane",
        ))
    } else {
        Ok(())
    }
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
