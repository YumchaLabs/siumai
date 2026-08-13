//! Canonical established-stream lifecycle and event vocabulary.

use std::collections::BTreeMap;
use std::fmt;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::{FutureExt, Stream, StreamExt, pin_mut, select_biased};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::error::{Error, ErrorKind, ResponseDiagnostics};
use crate::language::{
    Citation, DEFAULT_PARTIAL_LANGUAGE_OUTPUT_ITEM_BYTE_LIMIT,
    DEFAULT_PARTIAL_LANGUAGE_OUTPUT_ITEM_COUNT_LIMIT,
    DEFAULT_PARTIAL_LANGUAGE_OUTPUT_TOTAL_BYTE_LIMIT, LanguageResponse, OpaqueProviderItem,
    PartialLanguageOutput, PartialLanguageOutputPart,
};
use crate::options::Cancellation;
use crate::provider::ModelId;
use crate::tool::{ExecutionOwner, ToolCall, ToolResult};
use crate::usage::Usage;

/// An established stream that enforces the canonical terminal lifecycle.
///
/// Values can only be created through [`established_stream`]. Dropping the
/// carrier cancels the child operation owned by the stream without cancelling
/// the caller's parent token.
pub struct LanguageStream {
    inner: Pin<Box<dyn Stream<Item = LanguageStreamEvent> + Send + 'static>>,
    cancellation: Cancellation,
}

impl fmt::Debug for LanguageStream {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LanguageStream")
            .field("is_cancelled", &self.cancellation.is_cancelled())
            .finish_non_exhaustive()
    }
}

impl Stream for LanguageStream {
    type Item = LanguageStreamEvent;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(context)
    }
}

impl LanguageStream {
    /// Preserve the established stream lifecycle while adding outer route
    /// context to provider failures.
    pub fn with_route_context(self, route: crate::RouteId) -> Self {
        established_stream(Cancellation::new(), move |_| {
            self.map(move |event| Ok(with_route_context(event, &route)))
        })
    }
}

fn with_route_context(event: LanguageStreamEvent, route: &crate::RouteId) -> LanguageStreamEvent {
    match event {
        LanguageStreamEvent::Terminal(StreamTerminal::Failed { error, partial }) => {
            LanguageStreamEvent::Terminal(StreamTerminal::Failed {
                error: error.with_route(route.clone()),
                partial,
            })
        }
        event => event,
    }
}

impl Drop for LanguageStream {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

/// Events emitted after a language stream has been established.
#[derive(Debug)]
#[non_exhaustive]
pub enum LanguageStreamEvent {
    Started {
        id: Option<String>,
        model: Option<ModelId>,
    },
    TextStart {
        id: String,
    },
    TextDelta {
        id: String,
        delta: String,
    },
    TextEnd {
        id: String,
    },
    ReasoningStart {
        id: String,
    },
    ReasoningDelta {
        id: String,
        delta: String,
    },
    ReasoningEnd {
        id: String,
    },
    ToolInputStart {
        id: String,
        name: String,
        owner: ExecutionOwner,
    },
    ToolInputDelta {
        id: String,
        delta: String,
    },
    ToolCall(ToolCall),
    ToolResult(ToolResult),
    Citation(Citation),
    Refusal {
        reason: Option<String>,
    },
    ProviderDeferred {
        id: String,
        state: OpaqueProviderItem,
    },
    ProviderOpaque(OpaqueProviderItem),
    Usage(UsageUpdate),
    Terminal(StreamTerminal),
}

impl LanguageStreamEvent {
    pub fn terminal(&self) -> Option<&StreamTerminal> {
        match self {
            Self::Terminal(terminal) => Some(terminal),
            _ => None,
        }
    }
}

/// Whether a usage event replaces the latest per-call observation or adds to it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum UsageUpdateKind {
    #[default]
    Snapshot,
    Delta,
}

/// A provider usage observation with explicit accumulation semantics.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct UsageUpdate {
    pub kind: UsageUpdateKind,
    pub usage: Usage,
}

impl UsageUpdate {
    pub fn new(kind: UsageUpdateKind, usage: Usage) -> Self {
        Self { kind, usage }
    }

    pub fn snapshot(usage: Usage) -> Self {
        Self::new(UsageUpdateKind::Snapshot, usage)
    }

    pub fn delta(usage: Usage) -> Self {
        Self::new(UsageUpdateKind::Delta, usage)
    }

    pub fn kind(&self) -> UsageUpdateKind {
        self.kind
    }

    pub fn usage(&self) -> &Usage {
        &self.usage
    }

    pub fn into_usage(self) -> Usage {
        self.usage
    }
}

impl From<Usage> for UsageUpdate {
    fn from(usage: Usage) -> Self {
        Self::snapshot(usage)
    }
}

/// Exactly one terminal outcome for an observed established stream.
#[derive(Debug)]
#[non_exhaustive]
pub enum StreamTerminal {
    /// The stream reached a successful protocol terminal. The response
    /// termination may be completed or incomplete.
    Completed { response: Box<LanguageResponse> },
    /// Generation failed after establishment. Only bounded observational
    /// content and usage are retained; complete failed provider resources stay
    /// behind provider-native APIs.
    Failed {
        error: Error,
        partial: Option<PartialLanguageOutput>,
    },
    /// Generation was cancelled after establishment. Only bounded observational
    /// content and usage are retained.
    Cancelled {
        reason: String,
        partial: Option<PartialLanguageOutput>,
    },
}

impl StreamTerminal {
    fn validate(&self) -> Result<(), StreamContractError> {
        match self {
            Self::Completed { response } => response
                .validate()
                .map_err(|_| StreamContractError::InvalidTerminalResponse),
            Self::Failed { partial, .. } | Self::Cancelled { partial, .. } => {
                partial.as_ref().map_or(Ok(()), |partial| {
                    partial
                        .validate()
                        .map_err(|_| StreamContractError::InvalidPartialOutput)
                })
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum StreamContractError {
    #[error("stream emitted more than one terminal event")]
    DuplicateTerminal,
    #[error("stream emitted an event after its terminal event")]
    EventAfterTerminal,
    #[error("protocol decoder received a frame after its terminal event")]
    FrameAfterTerminal,
    #[error("protocol decoder received a frame after it was finished")]
    FrameAfterFinish,
    #[error("protocol decoder finish was called more than once")]
    DuplicateFinish,
    #[error("stream terminal contains an invalid language response")]
    InvalidTerminalResponse,
    #[error("stream terminal contains invalid partial language output")]
    InvalidPartialOutput,
}

impl From<StreamContractError> for Error {
    fn from(error: StreamContractError) -> Self {
        let message = match error {
            StreamContractError::DuplicateTerminal => "stream emitted more than one terminal event",
            StreamContractError::EventAfterTerminal => {
                "stream emitted an event after its terminal event"
            }
            StreamContractError::FrameAfterTerminal => {
                "protocol decoder received a frame after its terminal event"
            }
            StreamContractError::FrameAfterFinish => {
                "protocol decoder received a frame after it was finished"
            }
            StreamContractError::DuplicateFinish => {
                "protocol decoder finish was called more than once"
            }
            StreamContractError::InvalidTerminalResponse => {
                "stream terminal contains an invalid language response"
            }
            StreamContractError::InvalidPartialOutput => {
                "stream terminal contains invalid partial language output"
            }
        };
        Self::new(ErrorKind::Protocol, message).with_source(error)
    }
}

/// Decoder-side lifecycle guard used before events reach the public stream.
#[derive(Debug, Default)]
pub struct StreamLifecycle {
    terminal_seen: bool,
}

impl StreamLifecycle {
    pub fn record(&mut self, event: &LanguageStreamEvent) -> Result<(), StreamContractError> {
        self.record_all(std::iter::once(event))
    }

    /// Validate and record one decoder output batch atomically.
    ///
    /// A terminal event must be the final item in its batch. Invalid batches do
    /// not partially advance the lifecycle, so a decoder error cannot leave the
    /// shared guard in a misleading terminal state.
    pub fn record_all<'a>(
        &mut self,
        events: impl IntoIterator<Item = &'a LanguageStreamEvent>,
    ) -> Result<(), StreamContractError> {
        let mut terminal_seen = self.terminal_seen;
        for event in events {
            if let Some(terminal) = event.terminal() {
                terminal.validate()?;
            }
            if terminal_seen {
                return if event.terminal().is_some() {
                    Err(StreamContractError::DuplicateTerminal)
                } else {
                    Err(StreamContractError::EventAfterTerminal)
                };
            }
            terminal_seen = event.terminal().is_some();
        }
        self.terminal_seen = terminal_seen;
        Ok(())
    }

    pub fn terminal_seen(&self) -> bool {
        self.terminal_seen
    }
}

/// Shared lifecycle state for a protocol-specific language stream decoder.
///
/// Protocol decoders keep their own wire state machines and embed this guard to
/// enforce frame, finish, and canonical terminal ordering without depending on
/// a provider or transport implementation.
#[derive(Debug, Default)]
pub struct DecoderLifecycle {
    events: StreamLifecycle,
    finish_seen: bool,
}

impl DecoderLifecycle {
    /// Reject a wire frame after a terminal event or EOF finalization.
    pub fn ensure_decode_allowed(&self) -> Result<(), StreamContractError> {
        if self.finish_seen {
            return Err(StreamContractError::FrameAfterFinish);
        }
        if self.events.terminal_seen() {
            return Err(StreamContractError::FrameAfterTerminal);
        }
        Ok(())
    }

    /// Record one decoded canonical event batch.
    pub fn record(&mut self, events: &[LanguageStreamEvent]) -> Result<(), StreamContractError> {
        self.events.record_all(events)
    }

    /// Begin the decoder's single EOF/finalization path.
    ///
    /// The return value reports whether a protocol terminal was already emitted
    /// by `decode`. When it is `false`, the decoder's finalizer must emit one
    /// terminal event or return [`Error::unexpected_eof`].
    pub fn begin_finish(&mut self) -> Result<bool, StreamContractError> {
        if self.finish_seen {
            return Err(StreamContractError::DuplicateFinish);
        }
        self.finish_seen = true;
        Ok(self.events.terminal_seen())
    }

    pub fn terminal_seen(&self) -> bool {
        self.events.terminal_seen()
    }

    pub fn finish_seen(&self) -> bool {
        self.finish_seen
    }
}

/// Lifecycle contract implemented by each language wire protocol decoder.
///
/// Transport framing is deliberately outside this trait. An implementation
/// owns one protocol state machine, consumes its associated framed value, and
/// emits canonical events. `finish` is called exactly once at clean framing
/// EOF. A protocol may emit its terminal event from `decode` (for an explicit
/// wire terminal) or from `finish` (for EOF-terminated protocols), but a
/// successful lifecycle emits exactly one terminal event in total.
pub trait LanguageStreamDecoder {
    /// One transport-framed protocol value, such as SSE data, JSONL JSON, or a
    /// decoded WebSocket message.
    type ProtocolFrame: ?Sized;

    /// Attach bounded transport diagnostics before the first protocol frame is decoded.
    ///
    /// Decoders use this context only when an established stream reports an in-band failure.
    /// The default implementation ignores it for protocols that do not expose such failures.
    fn set_response_diagnostics(&mut self, _diagnostics: ResponseDiagnostics) {}

    /// Consume one protocol frame and emit zero or more canonical events.
    fn decode(&mut self, frame: &Self::ProtocolFrame) -> Result<Vec<LanguageStreamEvent>, Error>;

    /// Signal clean framing EOF and finalize pending protocol state.
    fn finish(&mut self) -> Result<Vec<LanguageStreamEvent>, Error>;

    /// Whether this decoder has successfully emitted its unique terminal event.
    fn terminal_seen(&self) -> bool;
}

/// Convert a protocol/runtime source into the public terminal-event contract.
///
/// Source errors occur after establishment and therefore become `Failed` events.
/// EOF without a terminal becomes `Failed(UnexpectedEof)`. Once a terminal event
/// is emitted the source is dropped immediately and cannot emit another event.
pub fn established_stream<S, F>(cancellation: Cancellation, source: F) -> LanguageStream
where
    F: FnOnce(Cancellation) -> S,
    S: Stream<Item = Result<LanguageStreamEvent, Error>> + Send + 'static,
{
    let stream_cancellation = cancellation.child();
    let lifecycle_cancellation = stream_cancellation.clone();
    let source = source(stream_cancellation.clone());
    let inner = Box::pin(async_stream::stream! {
        let source = source.fuse();
        let cancelled = lifecycle_cancellation.token().cancelled_owned().fuse();
        pin_mut!(source, cancelled);
        let mut lifecycle = StreamLifecycle::default();
        let mut observation = PartialObservation::default();

        loop {
            select_biased! {
                _ = cancelled => {
                    let terminal = LanguageStreamEvent::Terminal(StreamTerminal::Cancelled {
                        reason: "call cancelled".to_string(),
                        partial: observation.finish(),
                    });
                    let _ = lifecycle.record(&terminal);
                    yield terminal;
                    break;
                },
                item = source.next() => {
                    match item {
                        Some(Ok(mut event)) => {
                            observation.complete_terminal(&mut event);
                            if let Err(contract_error) = lifecycle.record(&event) {
                                yield LanguageStreamEvent::Terminal(StreamTerminal::Failed {
                                    error: Error::from(contract_error),
                                    partial: observation.finish(),
                                });
                                break;
                            }
                            let is_terminal = event.terminal().is_some();
                            if !is_terminal {
                                observation.observe(&event);
                            }
                            yield event;
                            if is_terminal {
                                break;
                            }
                        }
                        Some(Err(error)) => {
                            let partial = observation.finish();
                            let terminal = if error.kind() == ErrorKind::Cancelled {
                                LanguageStreamEvent::Terminal(StreamTerminal::Cancelled {
                                    reason: error.message().to_string(),
                                    partial,
                                })
                            } else {
                                LanguageStreamEvent::Terminal(StreamTerminal::Failed {
                                    error,
                                    partial,
                                })
                            };
                            let _ = lifecycle.record(&terminal);
                            yield terminal;
                            break;
                        }
                        None => {
                            let terminal = LanguageStreamEvent::Terminal(StreamTerminal::Failed {
                                error: Error::unexpected_eof(),
                                partial: observation.finish(),
                            });
                            let _ = lifecycle.record(&terminal);
                            yield terminal;
                            break;
                        }
                    }
                }
            }
        }
    });
    LanguageStream {
        inner,
        cancellation: stream_cancellation,
    }
}

#[derive(Default)]
struct PartialObservation {
    parts: Vec<PartialLanguageOutputPart>,
    text_parts: BTreeMap<String, usize>,
    reasoning_parts: BTreeMap<String, usize>,
    usage: Usage,
    has_usage: bool,
    observed_text_bytes: usize,
    disabled: bool,
}

impl PartialObservation {
    fn observe(&mut self, event: &LanguageStreamEvent) {
        if self.disabled {
            return;
        }
        match event {
            LanguageStreamEvent::TextStart { id } => {
                self.ensure_text_part(id, false);
            }
            LanguageStreamEvent::TextDelta { id, delta } => {
                self.push_delta(id, delta, false);
            }
            LanguageStreamEvent::ReasoningStart { id } => {
                self.ensure_text_part(id, true);
            }
            LanguageStreamEvent::ReasoningDelta { id, delta } => {
                self.push_delta(id, delta, true);
            }
            LanguageStreamEvent::Refusal { reason } => {
                if self.reserve_part(reason.as_ref().map_or(0, String::len)) {
                    self.parts.push(PartialLanguageOutputPart::Refusal {
                        reason: reason.clone(),
                    });
                }
            }
            LanguageStreamEvent::Usage(update) => self.observe_usage(update),
            _ => {}
        }
    }

    fn complete_terminal(&mut self, event: &mut LanguageStreamEvent) {
        let LanguageStreamEvent::Terminal(terminal) = event else {
            return;
        };
        match terminal {
            StreamTerminal::Failed { partial, .. } | StreamTerminal::Cancelled { partial, .. } => {
                if partial.is_none() {
                    *partial = self.finish();
                }
            }
            StreamTerminal::Completed { .. } => {}
        }
    }

    fn ensure_text_part(&mut self, id: &str, reasoning: bool) -> Option<usize> {
        let existing = if reasoning {
            self.reasoning_parts.get(id)
        } else {
            self.text_parts.get(id)
        };
        if let Some(index) = existing {
            return Some(*index);
        }
        if !self.reserve_part(0) {
            return None;
        }
        let index = self.parts.len();
        let part = if reasoning {
            PartialLanguageOutputPart::Reasoning {
                text: String::new(),
            }
        } else {
            PartialLanguageOutputPart::Text {
                text: String::new(),
            }
        };
        self.parts.push(part);
        if reasoning {
            self.reasoning_parts.insert(id.to_string(), index);
        } else {
            self.text_parts.insert(id.to_string(), index);
        }
        Some(index)
    }

    fn push_delta(&mut self, id: &str, delta: &str, reasoning: bool) {
        let Some(index) = self.ensure_text_part(id, reasoning) else {
            return;
        };
        let current_bytes = match &self.parts[index] {
            PartialLanguageOutputPart::Text { text }
            | PartialLanguageOutputPart::Reasoning { text } => text.len(),
            PartialLanguageOutputPart::Refusal { .. } => return,
        };
        let Some(item_bytes) = current_bytes.checked_add(delta.len()) else {
            self.disable();
            return;
        };
        if item_bytes > DEFAULT_PARTIAL_LANGUAGE_OUTPUT_ITEM_BYTE_LIMIT {
            self.disable();
            return;
        }
        let Some(total_bytes) = self.observed_text_bytes.checked_add(delta.len()) else {
            self.disable();
            return;
        };
        if total_bytes > DEFAULT_PARTIAL_LANGUAGE_OUTPUT_TOTAL_BYTE_LIMIT {
            self.disable();
            return;
        }
        self.observed_text_bytes = total_bytes;
        match &mut self.parts[index] {
            PartialLanguageOutputPart::Text { text }
            | PartialLanguageOutputPart::Reasoning { text } => text.push_str(delta),
            PartialLanguageOutputPart::Refusal { .. } => {}
        }
    }

    fn reserve_part(&mut self, bytes: usize) -> bool {
        if self.parts.len() >= DEFAULT_PARTIAL_LANGUAGE_OUTPUT_ITEM_COUNT_LIMIT
            || bytes > DEFAULT_PARTIAL_LANGUAGE_OUTPUT_ITEM_BYTE_LIMIT
        {
            self.disable();
            return false;
        }
        let Some(total_bytes) = self.observed_text_bytes.checked_add(bytes) else {
            self.disable();
            return false;
        };
        if total_bytes > DEFAULT_PARTIAL_LANGUAGE_OUTPUT_TOTAL_BYTE_LIMIT {
            self.disable();
            return false;
        }
        self.observed_text_bytes = total_bytes;
        true
    }

    fn observe_usage(&mut self, update: &UsageUpdate) {
        match update.kind() {
            UsageUpdateKind::Snapshot => merge_usage_snapshot(&mut self.usage, update.usage()),
            UsageUpdateKind::Delta => merge_usage_delta(&mut self.usage, update.usage()),
        }
        self.has_usage = true;
    }

    fn finish(&self) -> Option<PartialLanguageOutput> {
        if self.disabled || (self.parts.is_empty() && !self.has_usage) {
            return None;
        }
        PartialLanguageOutput::new(self.parts.clone(), self.usage.clone()).ok()
    }

    fn disable(&mut self) {
        self.disabled = true;
        self.parts.clear();
        self.text_parts.clear();
        self.reasoning_parts.clear();
        self.usage = Usage::default();
        self.has_usage = false;
        self.observed_text_bytes = 0;
    }
}

fn merge_usage_snapshot(current: &mut Usage, snapshot: &Usage) {
    merge_usage_values(current, snapshot, |existing, incoming| {
        match (existing, incoming) {
            (crate::UsageValue::Known(left), crate::UsageValue::Known(right)) => {
                crate::UsageValue::Known(left.max(right))
            }
            (crate::UsageValue::Unknown, crate::UsageValue::Known(value)) => {
                crate::UsageValue::Known(value)
            }
            (existing, crate::UsageValue::Unknown) => existing,
        }
    });
    if !snapshot.provider.is_empty() {
        current.provider.clone_from(&snapshot.provider);
    }
}

fn merge_usage_delta(current: &mut Usage, delta: &Usage) {
    merge_usage_values(current, delta, |existing, incoming| match incoming {
        crate::UsageValue::Unknown => existing,
        crate::UsageValue::Known(value) => match existing {
            crate::UsageValue::Unknown => crate::UsageValue::Known(value),
            crate::UsageValue::Known(existing) => existing
                .checked_add(value)
                .map_or(crate::UsageValue::Unknown, crate::UsageValue::Known),
        },
    });
    for (key, value) in &delta.provider {
        current.provider.insert(key.clone(), value.clone());
    }
}

fn merge_usage_values(
    current: &mut Usage,
    incoming: &Usage,
    merge: impl Fn(crate::UsageValue, crate::UsageValue) -> crate::UsageValue,
) {
    current.input_tokens = merge(current.input_tokens, incoming.input_tokens);
    current.output_tokens = merge(current.output_tokens, incoming.output_tokens);
    current.total_tokens = merge(current.total_tokens, incoming.total_tokens);
    current.reasoning_tokens = merge(current.reasoning_tokens, incoming.reasoning_tokens);
    current.cache_read_tokens = merge(current.cache_read_tokens, incoming.cache_read_tokens);
    current.cache_write_tokens = merge(current.cache_write_tokens, incoming.cache_write_tokens);
    current.audio_input_tokens = merge(current.audio_input_tokens, incoming.audio_input_tokens);
    current.audio_output_tokens = merge(current.audio_output_tokens, incoming.audio_output_tokens);
    current.orchestration_tokens =
        merge(current.orchestration_tokens, incoming.orchestration_tokens);
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use futures::stream;
    use serde_json::json;

    use super::*;
    use crate::language::{
        LanguageIncompleteReason, PartialLanguageOutputPart, ProviderProvenance,
    };
    use crate::provider::{ProtocolId, ProviderId, ProviderScope, ReplayDomain, ReplayDomainId};
    use crate::tool::ToolOutcome;

    #[tokio::test]
    async fn eof_without_terminal_becomes_failed_unexpected_eof() {
        let source = stream::iter(vec![
            Ok(LanguageStreamEvent::TextDelta {
                id: "text".to_string(),
                delta: "hello".to_string(),
            }),
            Ok(LanguageStreamEvent::Usage(UsageUpdate::snapshot(
                Usage::default().with_output_tokens(1_u64),
            ))),
        ]);
        let events = established_stream(Cancellation::new(), |_| source)
            .collect::<Vec<_>>()
            .await;

        assert!(matches!(
            events.last(),
            Some(LanguageStreamEvent::Terminal(StreamTerminal::Failed {
                error,
                partial: Some(partial),
            })) if error.kind() == ErrorKind::UnexpectedEof
                && matches!(
                    partial.content(),
                    [PartialLanguageOutputPart::Text { text }] if text == "hello"
                )
                && partial.usage().output_tokens == crate::UsageValue::Known(1)
        ));
    }

    #[tokio::test]
    async fn route_context_is_added_to_stream_failures_without_changing_events() {
        let source = stream::iter(vec![Err(Error::new(ErrorKind::Provider, "request failed"))]);
        let events = established_stream(Cancellation::new(), |_| source)
            .with_route_context(crate::RouteId::new("production").unwrap())
            .collect::<Vec<_>>()
            .await;

        assert!(matches!(
            events.as_slice(),
            [LanguageStreamEvent::Terminal(StreamTerminal::Failed { error, .. })]
                if error.context().route.as_ref().map(crate::RouteId::as_str)
                    == Some("production")
        ));
    }

    #[test]
    fn lifecycle_rejects_duplicate_terminal_events() {
        let mut lifecycle = StreamLifecycle::default();
        let terminal = LanguageStreamEvent::Terminal(StreamTerminal::Cancelled {
            reason: "cancelled".to_string(),
            partial: None,
        });
        lifecycle.record(&terminal).unwrap();
        assert_eq!(
            lifecycle.record(&terminal),
            Err(StreamContractError::DuplicateTerminal)
        );
    }

    #[test]
    fn lifecycle_rejects_invalid_batches_atomically() {
        let mut lifecycle = StreamLifecycle::default();
        let events = [
            LanguageStreamEvent::Terminal(StreamTerminal::Cancelled {
                reason: "cancelled".to_string(),
                partial: None,
            }),
            LanguageStreamEvent::TextDelta {
                id: "text".to_string(),
                delta: "late".to_string(),
            },
        ];

        assert_eq!(
            lifecycle.record_all(&events),
            Err(StreamContractError::EventAfterTerminal)
        );
        assert!(!lifecycle.terminal_seen());
    }

    #[test]
    fn failed_terminal_accepts_bounded_observational_partial_output() {
        let partial = PartialLanguageOutput::new(
            vec![PartialLanguageOutputPart::Text {
                text: "partial".to_string(),
            }],
            Usage::default().with_output_tokens(2_u64),
        )
        .unwrap();
        let terminal = LanguageStreamEvent::Terminal(StreamTerminal::Failed {
            error: Error::new(ErrorKind::Provider, "generation failed"),
            partial: Some(partial),
        });

        StreamLifecycle::default().record(&terminal).unwrap();
        assert!(matches!(
            terminal,
            LanguageStreamEvent::Terminal(StreamTerminal::Failed {
                partial: Some(partial),
                ..
            }) if matches!(
                partial.content(),
                [PartialLanguageOutputPart::Text { text }] if text == "partial"
            )
        ));
    }

    #[test]
    fn completed_terminal_accepts_an_incomplete_response() {
        let response = LanguageResponse::incomplete(
            Vec::new(),
            LanguageIncompleteReason::MaxOutputTokens,
            Usage::default(),
        )
        .unwrap();
        let terminal = LanguageStreamEvent::Terminal(StreamTerminal::Completed {
            response: Box::new(response),
        });

        StreamLifecycle::default().record(&terminal).unwrap();
    }

    #[test]
    fn usage_events_are_explicit_snapshots_or_deltas() {
        assert_eq!(UsageUpdate::default().kind(), UsageUpdateKind::Snapshot);
        let snapshot = UsageUpdate::from(Usage::default().with_input_tokens(4_u64));
        assert_eq!(snapshot.kind(), UsageUpdateKind::Snapshot);
        assert_eq!(snapshot.usage().input_tokens, crate::UsageValue::Known(4));

        let delta = UsageUpdate::delta(Usage::default().with_output_tokens(2_u64));
        assert_eq!(delta.kind(), UsageUpdateKind::Delta);
        assert_eq!(delta.usage().output_tokens, crate::UsageValue::Known(2));

        let event = LanguageStreamEvent::Usage(snapshot.clone());
        assert!(matches!(event, LanguageStreamEvent::Usage(update) if update == snapshot));
        let decoded: UsageUpdate =
            serde_json::from_value(serde_json::to_value(delta).unwrap()).unwrap();
        assert_eq!(decoded.kind(), UsageUpdateKind::Delta);
    }

    #[tokio::test]
    async fn explicit_cancellation_is_a_terminal_event() {
        let cancellation = Cancellation::new();
        cancellation.cancel();
        let source = stream::pending::<Result<LanguageStreamEvent, Error>>();
        let events = established_stream(cancellation, |_| source)
            .collect::<Vec<_>>()
            .await;

        assert!(matches!(
            events.as_slice(),
            [LanguageStreamEvent::Terminal(
                StreamTerminal::Cancelled { .. }
            )]
        ));
    }

    #[tokio::test]
    async fn post_establishment_error_is_a_failed_terminal_event() {
        let source = stream::iter(vec![
            Ok(LanguageStreamEvent::ReasoningDelta {
                id: "reasoning".to_string(),
                delta: "partial reasoning".to_string(),
            }),
            Err(Error::new(ErrorKind::Transport, "connection reset")),
        ]);
        let events = established_stream(Cancellation::new(), |_| source)
            .collect::<Vec<_>>()
            .await;

        assert!(matches!(
            events.last(),
            Some(LanguageStreamEvent::Terminal(StreamTerminal::Failed {
                error,
                partial: Some(partial),
            })) if error.kind() == ErrorKind::Transport
                && matches!(
                    partial.content(),
                    [PartialLanguageOutputPart::Reasoning { text }]
                        if text == "partial reasoning"
                )
        ));
    }

    #[tokio::test]
    async fn cancelled_source_error_preserves_the_cancelled_terminal_kind() {
        let source = stream::iter(vec![Err(Error::cancelled("provider cancelled"))]);
        let events = established_stream(Cancellation::new(), |_| source)
            .collect::<Vec<_>>()
            .await;

        assert!(matches!(
            events.as_slice(),
            [LanguageStreamEvent::Terminal(StreamTerminal::Cancelled { reason, .. })]
                if reason == "provider cancelled"
        ));
    }

    #[tokio::test]
    async fn provider_tool_results_are_first_class_stream_events() {
        let result = ToolResult {
            call_id: "call_1".to_string(),
            name: "web_search".to_string(),
            outcome: ToolOutcome::Success {
                value: json!({"answer": 42}),
            },
        };
        let source = stream::iter(vec![
            Ok(LanguageStreamEvent::ToolResult(result)),
            Ok(LanguageStreamEvent::Terminal(StreamTerminal::Cancelled {
                reason: "test complete".to_string(),
                partial: None,
            })),
        ]);
        let events = established_stream(Cancellation::new(), |_| source)
            .collect::<Vec<_>>()
            .await;

        assert!(matches!(
            events.first(),
            Some(LanguageStreamEvent::ToolResult(result))
                if result.call_id == "call_1" && result.name == "web_search"
        ));
    }

    #[tokio::test]
    async fn opaque_events_preserve_provider_provenance() {
        let scope = ProviderScope::new(ProviderId::new("openai").unwrap())
            .with_protocol(ProtocolId::new("responses").unwrap())
            .with_replay_domain(ReplayDomain::official(
                ReplayDomainId::new("openai-public-api").unwrap(),
            ));
        let item = OpaqueProviderItem::new(
            ProviderProvenance::from_scope(&scope, ModelId::new("future:model").unwrap()).unwrap(),
            "reasoning.encrypted",
            json!({"encrypted_content": "opaque"}),
        )
        .unwrap();
        let source = stream::iter(vec![
            Ok(LanguageStreamEvent::ProviderOpaque(item)),
            Ok(LanguageStreamEvent::Terminal(StreamTerminal::Cancelled {
                reason: "test complete".to_string(),
                partial: None,
            })),
        ]);
        let events = established_stream(Cancellation::new(), |_| source)
            .collect::<Vec<_>>()
            .await;

        assert!(matches!(
            events.first(),
            Some(LanguageStreamEvent::ProviderOpaque(item))
                if item.provenance().protocol().as_str() == "responses"
                    && item.provenance().model().as_str() == "future:model"
        ));
    }

    #[tokio::test]
    async fn consumer_drop_releases_the_source_without_fabricating_an_event() {
        struct DropMarker(Arc<AtomicBool>);
        impl Drop for DropMarker {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        let dropped = Arc::new(AtomicBool::new(false));
        let marker = DropMarker(dropped.clone());
        let source = async_stream::stream! {
            let _marker = marker;
            yield Ok(LanguageStreamEvent::TextDelta {
                id: "text".to_string(),
                delta: "a".to_string(),
            });
            std::future::pending::<()>().await;
        };
        let mut stream = established_stream(Cancellation::new(), |_| source);
        assert!(stream.next().await.is_some());
        drop(stream);
        assert!(dropped.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn consumer_drop_cancels_only_the_stream_child() {
        let parent = Cancellation::new();
        let cancelled = Arc::new(AtomicBool::new(false));
        let observed = cancelled.clone();
        let stream = established_stream(parent.clone(), move |stream_cancellation| {
            tokio::spawn(async move {
                stream_cancellation.cancelled().await;
                observed.store(true, Ordering::SeqCst);
            });
            stream::pending::<Result<LanguageStreamEvent, Error>>()
        });

        drop(stream);
        tokio::task::yield_now().await;

        assert!(!parent.is_cancelled());
        assert!(cancelled.load(Ordering::SeqCst));
    }
}
