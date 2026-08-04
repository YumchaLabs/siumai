//! Canonical established-stream lifecycle and event vocabulary.

use std::fmt;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::{FutureExt, Stream, StreamExt, pin_mut, select_biased};
use thiserror::Error;

use crate::error::{Error, ErrorKind};
use crate::language::{Citation, LanguageResponse, LanguageResponseStatus, OpaqueProviderItem};
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
        LanguageStreamEvent::Terminal(StreamTerminal::Failed { error, response }) => {
            LanguageStreamEvent::Terminal(StreamTerminal::Failed {
                error: error.with_route(route.clone()),
                response,
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
    Usage(Usage),
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

/// Exactly one terminal outcome for an observed established stream.
#[derive(Debug)]
#[non_exhaustive]
pub enum StreamTerminal {
    /// The stream reached a non-error protocol terminal. The response status
    /// may be `Completed` or `Incomplete`.
    Completed { response: Box<LanguageResponse> },
    /// Generation failed after establishment. A provider-returned failed
    /// response is retained when one exists.
    Failed {
        error: Error,
        response: Option<Box<LanguageResponse>>,
    },
    /// Generation was cancelled after establishment. A provider-returned
    /// cancelled response is retained when one exists.
    Cancelled {
        reason: String,
        response: Option<Box<LanguageResponse>>,
    },
}

impl StreamTerminal {
    fn validate(&self) -> Result<(), StreamContractError> {
        let response = match self {
            Self::Completed { response } => {
                if !matches!(
                    response.status(),
                    LanguageResponseStatus::Completed | LanguageResponseStatus::Incomplete { .. }
                ) {
                    return Err(StreamContractError::TerminalResponseStatusMismatch);
                }
                Some(response.as_ref())
            }
            Self::Failed { response, .. } => {
                if response.as_deref().is_some_and(|response| {
                    !matches!(response.status(), LanguageResponseStatus::Failed)
                }) {
                    return Err(StreamContractError::TerminalResponseStatusMismatch);
                }
                response.as_deref()
            }
            Self::Cancelled { response, .. } => {
                if response.as_deref().is_some_and(|response| {
                    !matches!(response.status(), LanguageResponseStatus::Cancelled)
                }) {
                    return Err(StreamContractError::TerminalResponseStatusMismatch);
                }
                response.as_deref()
            }
        };
        if response.is_some_and(|response| response.validate().is_err()) {
            return Err(StreamContractError::InvalidTerminalResponse);
        }
        Ok(())
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
    #[error("stream terminal response status does not match its terminal kind")]
    TerminalResponseStatusMismatch,
    #[error("stream terminal contains an invalid language response")]
    InvalidTerminalResponse,
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
            StreamContractError::TerminalResponseStatusMismatch => {
                "stream terminal response status does not match its terminal kind"
            }
            StreamContractError::InvalidTerminalResponse => {
                "stream terminal contains an invalid language response"
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

        loop {
            select_biased! {
                _ = cancelled => {
                    let terminal = LanguageStreamEvent::Terminal(StreamTerminal::Cancelled {
                        reason: "call cancelled".to_string(),
                        response: None,
                    });
                    let _ = lifecycle.record(&terminal);
                    yield terminal;
                    break;
                },
                item = source.next() => {
                    match item {
                        Some(Ok(event)) => {
                            if let Err(contract_error) = lifecycle.record(&event) {
                                yield LanguageStreamEvent::Terminal(StreamTerminal::Failed {
                                    error: Error::from(contract_error),
                                    response: None,
                                });
                                break;
                            }
                            let is_terminal = event.terminal().is_some();
                            yield event;
                            if is_terminal {
                                break;
                            }
                        }
                        Some(Err(error)) => {
                            let terminal = LanguageStreamEvent::Terminal(StreamTerminal::Failed {
                                error,
                                response: None,
                            });
                            let _ = lifecycle.record(&terminal);
                            yield terminal;
                            break;
                        }
                        None => {
                            let terminal = LanguageStreamEvent::Terminal(StreamTerminal::Failed {
                                error: Error::unexpected_eof(),
                                response: None,
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use futures::stream;
    use serde_json::json;

    use super::*;
    use crate::language::{FinishReason, LanguageIncompleteReason, ProviderProvenance};
    use crate::provider::ProviderId;
    use crate::tool::ToolOutcome;

    #[tokio::test]
    async fn eof_without_terminal_becomes_failed_unexpected_eof() {
        let source = stream::iter(vec![Ok(LanguageStreamEvent::TextDelta {
            id: "text".to_string(),
            delta: "hello".to_string(),
        })]);
        let events = established_stream(Cancellation::new(), |_| source)
            .collect::<Vec<_>>()
            .await;

        assert!(matches!(
            events.last(),
            Some(LanguageStreamEvent::Terminal(StreamTerminal::Failed { error, .. }))
                if error.kind() == ErrorKind::UnexpectedEof
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
            response: None,
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
                response: None,
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
    fn lifecycle_rejects_terminal_status_mismatch() {
        let response =
            LanguageResponse::completed(Vec::new(), FinishReason::Stop, Usage::default()).unwrap();
        let terminal = LanguageStreamEvent::Terminal(StreamTerminal::Failed {
            error: Error::new(ErrorKind::Provider, "generation failed"),
            response: Some(Box::new(response)),
        });

        assert_eq!(
            StreamLifecycle::default().record(&terminal),
            Err(StreamContractError::TerminalResponseStatusMismatch)
        );
    }

    #[test]
    fn completed_terminal_accepts_an_incomplete_response() {
        let response = LanguageResponse::new(
            LanguageResponseStatus::Incomplete {
                reason: Some(LanguageIncompleteReason::MaxOutputTokens),
            },
            Vec::new(),
            FinishReason::Length,
            Usage::default(),
        )
        .unwrap();
        let terminal = LanguageStreamEvent::Terminal(StreamTerminal::Completed {
            response: Box::new(response),
        });

        StreamLifecycle::default().record(&terminal).unwrap();
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
        let source = stream::iter(vec![Err(Error::new(
            ErrorKind::Transport,
            "connection reset",
        ))]);
        let events = established_stream(Cancellation::new(), |_| source)
            .collect::<Vec<_>>()
            .await;

        assert!(matches!(
            events.as_slice(),
            [LanguageStreamEvent::Terminal(StreamTerminal::Failed { error, .. })]
                if error.kind() == ErrorKind::Transport
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
                response: None,
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
        let item = OpaqueProviderItem::new(
            ProviderProvenance {
                provider: ProviderId::new("openai").unwrap(),
                platform: None,
                protocol: "responses".to_string(),
                model: ModelId::new("future:model").unwrap(),
            },
            "reasoning.encrypted",
            json!({"encrypted_content": "opaque"}),
        )
        .unwrap();
        let source = stream::iter(vec![
            Ok(LanguageStreamEvent::ProviderOpaque(item)),
            Ok(LanguageStreamEvent::Terminal(StreamTerminal::Cancelled {
                reason: "test complete".to_string(),
                response: None,
            })),
        ]);
        let events = established_stream(Cancellation::new(), |_| source)
            .collect::<Vec<_>>()
            .await;

        assert!(matches!(
            events.first(),
            Some(LanguageStreamEvent::ProviderOpaque(item))
                if item.provenance().protocol == "responses"
                    && item.provenance().model.as_str() == "future:model"
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
