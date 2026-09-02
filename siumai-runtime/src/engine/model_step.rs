//! Incremental model-stream polling and settlement for one engine step.
//!
//! This module owns the call-local lifecycle only. The root engine remains the
//! owner of report and budget commits, provider-deferred ledger replacement,
//! phase transitions, and event queueing.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

use futures::StreamExt;
use siumai_core::stream::established_stream;
use siumai_core::{
    CallOptions, Cancellation, ContentPart, Error, ErrorKind, LanguageModel, LanguageRequest,
    LanguageResponse, LanguageStream, LanguageStreamEvent, PartialLanguageOutput, StreamTerminal,
    Usage,
};

use crate::provider_deferred::ProviderDeferredStep;
use crate::single_step::SingleStep;
use crate::usage::CallUsageReconciler;
use crate::{RunTimeoutKind, Runtime, StepOptions};

use super::{WaitResult, classify_deadline, wait_for};

/// Inputs needed to establish one model stream without moving engine policy
/// into the model-step module.
pub(super) struct ModelStepContext<'a> {
    pub(super) runtime: &'a Runtime,
    pub(super) model: &'a dyn LanguageModel,
    pub(super) step_options: &'a StepOptions,
    pub(super) request: LanguageRequest,
    pub(super) call_options: CallOptions,
    pub(super) cancellation: Cancellation,
    pub(super) total_deadline: Instant,
    pub(super) model_step_timeout: std::time::Duration,
    pub(super) first_chunk_timeout: std::time::Duration,
    pub(super) inter_chunk_timeout: std::time::Duration,
    pub(super) provider_deferred: ProviderDeferredStep,
    pub(super) first: bool,
}

/// Call-local state retained between engine polls.
pub(super) struct ModelStepState {
    stream: LanguageStream,
    provider_deferred: ProviderDeferredStep,
    usage: CallUsageReconciler,
    timeout: Arc<StreamTimeoutState>,
}

/// One result of polling an established model stream.
pub(super) enum ModelStepPoll {
    Event {
        event: LanguageStreamEvent,
        state: ModelStepState,
    },
    Completed {
        response: LanguageResponse,
        usage: Usage,
        provider_deferred: ProviderDeferredStep,
    },
    Failed {
        error: Error,
        partial: Option<PartialLanguageOutput>,
        usage: Usage,
        timeout: Option<RunTimeoutKind>,
    },
    Cancelled {
        reason: String,
        partial: Option<PartialLanguageOutput>,
        usage: Usage,
    },
}

impl ModelStepState {
    /// Establish the provider stream and install runtime-owned stream timing.
    pub(super) async fn establish(context: ModelStepContext<'_>) -> Result<Self, Error> {
        let ModelStepContext {
            runtime,
            model,
            step_options,
            request,
            call_options,
            cancellation,
            total_deadline,
            model_step_timeout,
            first_chunk_timeout,
            inter_chunk_timeout,
            provider_deferred,
            first,
        } = context;
        let now = Instant::now();
        let step_deadline = total_deadline.min(checked_deadline(now, model_step_timeout));
        let (deadline, timeout_kind) =
            classify_deadline(total_deadline, step_deadline, RunTimeoutKind::ModelStep);
        let single_step = SingleStep::new(runtime, model, step_options);
        let future = single_step.stream(request, call_options);

        match wait_for(future, &cancellation, deadline, timeout_kind).await {
            WaitResult::Ready(result) => result.map(|stream| {
                let timeout = Arc::new(StreamTimeoutState::default());
                let stream = stream_with_runtime_deadlines(
                    stream,
                    cancellation,
                    total_deadline,
                    step_deadline,
                    first_chunk_timeout,
                    inter_chunk_timeout,
                    Arc::clone(&timeout),
                );
                Self {
                    stream,
                    provider_deferred,
                    usage: CallUsageReconciler::default(),
                    timeout,
                }
            }),
            WaitResult::Cancelled => Err(Error::cancelled(if first {
                "tool loop cancelled during first model handshake"
            } else {
                "tool loop cancelled during model handshake"
            })),
            WaitResult::TimedOut(_) => Err(Error::new(
                ErrorKind::Timeout,
                if first {
                    "tool loop first model handshake timed out"
                } else {
                    "tool loop model handshake timed out"
                },
            )),
        }
    }

    /// Poll at most one established-stream item and settle terminal state once.
    pub(super) async fn poll(mut self) -> ModelStepPoll {
        match self.stream.next().await {
            None => ModelStepPoll::Failed {
                usage: self.usage.settle(None),
                error: Error::unexpected_eof(),
                partial: None,
                timeout: None,
            },
            Some(LanguageStreamEvent::Terminal(terminal)) => match terminal {
                StreamTerminal::Completed { response } => {
                    let response = *response;
                    for result in response.content().iter().filter_map(|part| match part {
                        ContentPart::ToolResult(result) => Some(result),
                        _ => None,
                    }) {
                        self.provider_deferred
                            .resolve_provider_result(&result.call_id);
                    }
                    ModelStepPoll::Completed {
                        usage: self.usage.settle(Some(response.usage())),
                        response,
                        provider_deferred: self.provider_deferred,
                    }
                }
                StreamTerminal::Failed { error, partial } => ModelStepPoll::Failed {
                    usage: self
                        .usage
                        .settle(partial.as_ref().map(PartialLanguageOutput::usage)),
                    timeout: (error.kind() == ErrorKind::Timeout)
                        .then(|| self.timeout.take())
                        .flatten(),
                    error,
                    partial,
                },
                StreamTerminal::Cancelled { reason, partial } => ModelStepPoll::Cancelled {
                    usage: self
                        .usage
                        .settle(partial.as_ref().map(PartialLanguageOutput::usage)),
                    reason,
                    partial,
                },
                _ => ModelStepPoll::Failed {
                    usage: self.usage.settle(None),
                    error: Error::protocol_violation("unsupported model stream terminal"),
                    partial: None,
                    timeout: None,
                },
            },
            Some(event) => {
                match &event {
                    LanguageStreamEvent::ProviderDeferred { id, state } => {
                        self.provider_deferred.observe(id, state);
                    }
                    LanguageStreamEvent::ToolResult(result) => {
                        self.provider_deferred
                            .resolve_provider_result(&result.call_id);
                    }
                    LanguageStreamEvent::Usage(update) => self.usage.observe(update),
                    _ => {}
                }
                ModelStepPoll::Event { event, state: self }
            }
        }
    }
}

struct StreamTimeoutState(AtomicU8);

impl Default for StreamTimeoutState {
    fn default() -> Self {
        Self(AtomicU8::new(0))
    }
}

impl StreamTimeoutState {
    fn record(&self, kind: RunTimeoutKind) {
        self.0.store(timeout_kind_code(kind), Ordering::Release);
    }

    fn take(&self) -> Option<RunTimeoutKind> {
        decode_timeout_kind(self.0.swap(0, Ordering::AcqRel))
    }
}

fn stream_with_runtime_deadlines(
    mut stream: LanguageStream,
    cancellation: Cancellation,
    total_deadline: Instant,
    step_deadline: Instant,
    first_chunk_timeout: std::time::Duration,
    inter_chunk_timeout: std::time::Duration,
    timeout: Arc<StreamTimeoutState>,
) -> LanguageStream {
    established_stream(cancellation, move |_| {
        async_stream::stream! {
            let mut first_event = true;
            loop {
                let chunk_kind = if first_event {
                    RunTimeoutKind::FirstChunk
                } else {
                    RunTimeoutKind::InterChunk
                };
                let chunk_timeout = if first_event {
                    first_chunk_timeout
                } else {
                    inter_chunk_timeout
                };
                let (deadline, kind) = earliest_timeout([
                    (total_deadline, RunTimeoutKind::Total),
                    (step_deadline, RunTimeoutKind::ModelStep),
                    (checked_deadline(Instant::now(), chunk_timeout), chunk_kind),
                ]);
                let item = tokio::select! {
                    biased;
                    _ = tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)) => None,
                    item = stream.next() => Some(item),
                };
                match item {
                    None => {
                        timeout.record(kind);
                        yield Err(runtime_stream_timeout(kind));
                        break;
                    }
                    Some(Some(event)) => {
                        let terminal = event.terminal().is_some();
                        if !terminal {
                            first_event = false;
                        }
                        yield Ok(event);
                        if terminal {
                            break;
                        }
                    }
                    Some(None) => break,
                }
            }
        }
    })
}

fn checked_deadline(now: Instant, duration: std::time::Duration) -> Instant {
    now.checked_add(duration).unwrap_or(now)
}

fn earliest_timeout<const N: usize>(
    deadlines: [(Instant, RunTimeoutKind); N],
) -> (Instant, RunTimeoutKind) {
    deadlines
        .into_iter()
        .min_by_key(|(deadline, _)| *deadline)
        .unwrap_or((Instant::now(), RunTimeoutKind::Total))
}

fn runtime_stream_timeout(kind: RunTimeoutKind) -> Error {
    Error::new(
        ErrorKind::Timeout,
        match kind {
            RunTimeoutKind::Total => "tool loop total deadline expired while reading model stream",
            RunTimeoutKind::ModelStep => {
                "tool loop model-step deadline expired while reading model stream"
            }
            RunTimeoutKind::FirstChunk => "tool loop timed out waiting for the first model event",
            RunTimeoutKind::InterChunk => "tool loop timed out waiting for the next model event",
            RunTimeoutKind::Tool => "tool loop tool deadline expired",
        },
    )
}

const fn timeout_kind_code(kind: RunTimeoutKind) -> u8 {
    match kind {
        RunTimeoutKind::Total => 1,
        RunTimeoutKind::ModelStep => 2,
        RunTimeoutKind::FirstChunk => 3,
        RunTimeoutKind::InterChunk => 4,
        RunTimeoutKind::Tool => 5,
    }
}

const fn decode_timeout_kind(code: u8) -> Option<RunTimeoutKind> {
    match code {
        1 => Some(RunTimeoutKind::Total),
        2 => Some(RunTimeoutKind::ModelStep),
        3 => Some(RunTimeoutKind::FirstChunk),
        4 => Some(RunTimeoutKind::InterChunk),
        5 => Some(RunTimeoutKind::Tool),
        _ => None,
    }
}
