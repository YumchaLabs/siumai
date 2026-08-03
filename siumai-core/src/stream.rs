//! Canonical established-stream lifecycle and event vocabulary.

use std::fmt;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::{FutureExt, Stream, StreamExt, pin_mut, select_biased};
use thiserror::Error;

use crate::error::{Error, ErrorKind};
use crate::language::{Citation, LanguageResponse, OpaqueProviderItem};
use crate::options::Cancellation;
use crate::provider::ModelId;
use crate::tool::{ExecutionOwner, ToolCall};
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
    Completed { response: Box<LanguageResponse> },
    Failed { error: Error },
    Cancelled { reason: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum StreamContractError {
    #[error("stream emitted more than one terminal event")]
    DuplicateTerminal,
    #[error("stream emitted an event after its terminal event")]
    EventAfterTerminal,
}

/// Decoder-side lifecycle guard used before events reach the public stream.
#[derive(Debug, Default)]
pub struct StreamLifecycle {
    terminal_seen: bool,
}

impl StreamLifecycle {
    pub fn record(&mut self, event: &LanguageStreamEvent) -> Result<(), StreamContractError> {
        if self.terminal_seen {
            return if event.terminal().is_some() {
                Err(StreamContractError::DuplicateTerminal)
            } else {
                Err(StreamContractError::EventAfterTerminal)
            };
        }
        if event.terminal().is_some() {
            self.terminal_seen = true;
        }
        Ok(())
    }

    pub fn terminal_seen(&self) -> bool {
        self.terminal_seen
    }
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
                                    error: Error::new(
                                        ErrorKind::Protocol,
                                        match contract_error {
                                            StreamContractError::DuplicateTerminal => {
                                                "stream emitted more than one terminal event"
                                            }
                                            StreamContractError::EventAfterTerminal => {
                                                "stream emitted an event after its terminal event"
                                            }
                                        },
                                    ),
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
                            let terminal = LanguageStreamEvent::Terminal(StreamTerminal::Failed { error });
                            let _ = lifecycle.record(&terminal);
                            yield terminal;
                            break;
                        }
                        None => {
                            let terminal = LanguageStreamEvent::Terminal(StreamTerminal::Failed {
                                error: Error::unexpected_eof(),
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
    use crate::language::ProviderProvenance;
    use crate::provider::ProviderId;

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
            Some(LanguageStreamEvent::Terminal(StreamTerminal::Failed { error }))
                if error.kind() == ErrorKind::UnexpectedEof
        ));
    }

    #[test]
    fn lifecycle_rejects_duplicate_terminal_events() {
        let mut lifecycle = StreamLifecycle::default();
        let terminal = LanguageStreamEvent::Terminal(StreamTerminal::Cancelled {
            reason: "cancelled".to_string(),
        });
        lifecycle.record(&terminal).unwrap();
        assert_eq!(
            lifecycle.record(&terminal),
            Err(StreamContractError::DuplicateTerminal)
        );
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
            [LanguageStreamEvent::Terminal(StreamTerminal::Failed { error })]
                if error.kind() == ErrorKind::Transport
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
