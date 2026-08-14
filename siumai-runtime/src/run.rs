//! Observable lifecycle and terminal algebra for one high-level run.

use std::fmt;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use serde::{Deserialize, Deserializer, Serialize};
use siumai_core::{
    AssistantHistoryOmission, Cancellation, Error, LanguageResponse, LanguageStreamEvent, Message,
    OpaqueProviderItem, PartialLanguageOutput, ToolCall, ToolOutcome, ToolResult, Usage,
};

use crate::snapshot::ToolExecutionLog;
use crate::{
    BudgetError, BudgetLedger, ModelTarget, ProjectionLoss, ProjectionPolicy, ProjectionScope,
};

/// Whether a requested model transition was committed to the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ModelTransitionOutcome {
    Applied,
    Rejected,
}

/// Auditable history projection performed before one destination model step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelTransitionRecord {
    step: u32,
    source: ModelTarget,
    target: ModelTarget,
    policy: ProjectionPolicy,
    scope: ProjectionScope,
    outcome: ModelTransitionOutcome,
    losses: Vec<ProjectionLoss>,
}

impl ModelTransitionRecord {
    pub(crate) fn new(
        step: u32,
        source: ModelTarget,
        target: ModelTarget,
        policy: ProjectionPolicy,
        scope: ProjectionScope,
        outcome: ModelTransitionOutcome,
        losses: Vec<ProjectionLoss>,
    ) -> Self {
        Self {
            step,
            source,
            target,
            policy,
            scope,
            outcome,
            losses,
        }
    }

    pub const fn step(&self) -> u32 {
        self.step
    }

    pub const fn source(&self) -> &ModelTarget {
        &self.source
    }

    pub const fn target(&self) -> &ModelTarget {
        &self.target
    }

    pub const fn policy(&self) -> ProjectionPolicy {
        self.policy
    }

    pub const fn scope(&self) -> ProjectionScope {
        self.scope
    }

    pub const fn outcome(&self) -> ModelTransitionOutcome {
        self.outcome
    }

    pub fn losses(&self) -> &[ProjectionLoss] {
        &self.losses
    }

    pub const fn is_applied(&self) -> bool {
        matches!(self.outcome, ModelTransitionOutcome::Applied)
    }

    pub const fn is_rejected(&self) -> bool {
        matches!(self.outcome, ModelTransitionOutcome::Rejected)
    }

    pub fn is_lossy(&self) -> bool {
        !self.losses.is_empty()
    }
}

/// A completed model step retained in deterministic execution order.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StepRecord {
    index: u32,
    target: ModelTarget,
    response: LanguageResponse,
    tool_results: Vec<ToolResult>,
    assistant_history_omissions: Vec<AssistantHistoryOmission>,
}

#[derive(Deserialize)]
struct StepRecordWire {
    index: u32,
    target: ModelTarget,
    response: LanguageResponse,
    tool_results: Vec<ToolResult>,
    assistant_history_omissions: Vec<AssistantHistoryOmission>,
}

impl<'de> Deserialize<'de> for StepRecord {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = StepRecordWire::deserialize(deserializer)?;
        let record = Self::new(wire.index, wire.target, wire.response, wire.tool_results);
        if wire.assistant_history_omissions != record.assistant_history_omissions {
            return Err(serde::de::Error::custom(
                "invalid assistant-history omission records",
            ));
        }
        Ok(record)
    }
}

impl StepRecord {
    pub fn new(
        index: u32,
        target: ModelTarget,
        response: LanguageResponse,
        tool_results: Vec<ToolResult>,
    ) -> Self {
        let assistant_history_omissions = response.project_assistant_history().omissions().to_vec();
        Self {
            index,
            target,
            response,
            tool_results,
            assistant_history_omissions,
        }
    }

    pub fn index(&self) -> u32 {
        self.index
    }

    pub fn target(&self) -> &ModelTarget {
        &self.target
    }

    pub fn response(&self) -> &LanguageResponse {
        &self.response
    }

    pub fn tool_results(&self) -> &[ToolResult] {
        &self.tool_results
    }

    pub fn assistant_history_omissions(&self) -> &[AssistantHistoryOmission] {
        &self.assistant_history_omissions
    }
}

/// Durable, provider-neutral trace accumulated before a run terminal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunReport {
    initial_target: ModelTarget,
    messages: Vec<Message>,
    steps: Vec<StepRecord>,
    model_transitions: Vec<ModelTransitionRecord>,
    usage: Usage,
    usage_settled: bool,
    budget: BudgetLedger,
    execution_log: ToolExecutionLog,
    provider_deferred: Vec<OpaqueProviderItem>,
}

impl RunReport {
    pub fn new(initial_target: ModelTarget, messages: Vec<Message>) -> Self {
        Self {
            initial_target,
            messages,
            steps: Vec::new(),
            model_transitions: Vec::new(),
            usage: Usage::default(),
            usage_settled: false,
            budget: BudgetLedger::default(),
            execution_log: ToolExecutionLog::new(),
            provider_deferred: Vec::new(),
        }
    }

    pub fn initial_target(&self) -> &ModelTarget {
        &self.initial_target
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn steps(&self) -> &[StepRecord] {
        &self.steps
    }

    pub fn current_target(&self) -> &ModelTarget {
        self.model_transitions
            .iter()
            .rev()
            .find(|transition| transition.is_applied())
            .map_or(&self.initial_target, ModelTransitionRecord::target)
    }

    pub fn model_transitions(&self) -> &[ModelTransitionRecord] {
        &self.model_transitions
    }

    pub fn usage(&self) -> &Usage {
        &self.usage
    }

    pub(crate) fn usage_is_settled(&self) -> bool {
        self.usage_settled
    }

    pub fn budget(&self) -> &BudgetLedger {
        &self.budget
    }

    pub fn execution_log(&self) -> &ToolExecutionLog {
        &self.execution_log
    }

    pub fn provider_deferred(&self) -> &[OpaqueProviderItem] {
        &self.provider_deferred
    }

    pub fn final_response(&self) -> Option<&LanguageResponse> {
        self.steps.last().map(StepRecord::response)
    }

    pub(crate) fn messages_mut(&mut self) -> &mut Vec<Message> {
        &mut self.messages
    }

    pub(crate) fn replace_messages(&mut self, messages: Vec<Message>) {
        self.messages = messages;
    }

    pub(crate) fn steps_mut(&mut self) -> &mut Vec<StepRecord> {
        &mut self.steps
    }

    pub(crate) fn model_transitions_mut(&mut self) -> &mut Vec<ModelTransitionRecord> {
        &mut self.model_transitions
    }

    pub(crate) fn accumulate_usage(&mut self, usage: &Usage) {
        self.usage = if !self.usage_settled {
            usage.clone()
        } else {
            self.usage.checked_add(usage)
        };
        self.usage_settled = true;
    }

    pub(crate) fn budget_mut(&mut self) -> &mut BudgetLedger {
        &mut self.budget
    }

    pub(crate) fn execution_log_mut(&mut self) -> &mut ToolExecutionLog {
        &mut self.execution_log
    }

    pub(crate) fn provider_deferred_mut(&mut self) -> &mut Vec<OpaqueProviderItem> {
        &mut self.provider_deferred
    }
}

/// Why a run is quiescent but not complete.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SuspensionReason {
    AwaitingApproval { call_ids: Vec<String> },
    AwaitingProvider { state_ids: Vec<String> },
}

/// Runtime-owned timeout classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum RunTimeoutKind {
    Total,
    ModelStep,
    FirstChunk,
    InterChunk,
    Tool,
}

/// Explicit policy stop after a known tool outcome or host condition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum RunStopReason {
    ToolOutcome {
        call_id: String,
        outcome: ToolOutcome,
    },
    HostPolicy {
        reason: String,
    },
}

/// An effect that crossed dispatch but has no reliable completed checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndeterminateEffect {
    pub call_id: String,
    pub binding_fingerprint: String,
    pub reason: String,
}

/// Exactly one terminal outcome for an observed high-level run.
#[derive(Debug)]
#[non_exhaustive]
pub enum RunTerminal {
    Completed {
        report: Box<RunReport>,
    },
    Stopped {
        reason: RunStopReason,
        report: Box<RunReport>,
    },
    Suspended {
        reason: SuspensionReason,
        report: Box<RunReport>,
    },
    BudgetExceeded {
        error: BudgetError,
        report: Box<RunReport>,
    },
    TimedOut {
        kind: RunTimeoutKind,
        partial: Option<PartialLanguageOutput>,
        report: Box<RunReport>,
    },
    Indeterminate {
        effect: IndeterminateEffect,
        report: Box<RunReport>,
    },
    HistoryProjectionRejected {
        transition: Box<ModelTransitionRecord>,
        report: Box<RunReport>,
    },
    ResumeConflict {
        reason: String,
    },
    Failed {
        error: Error,
        partial: Option<PartialLanguageOutput>,
        report: Box<RunReport>,
    },
    Cancelled {
        reason: String,
        partial: Option<PartialLanguageOutput>,
        report: Box<RunReport>,
    },
}

impl RunTerminal {
    pub fn report(&self) -> Option<&RunReport> {
        match self {
            Self::Completed { report }
            | Self::Stopped { report, .. }
            | Self::Suspended { report, .. }
            | Self::BudgetExceeded { report, .. }
            | Self::TimedOut { report, .. }
            | Self::Indeterminate { report, .. }
            | Self::HistoryProjectionRejected { report, .. }
            | Self::Failed { report, .. }
            | Self::Cancelled { report, .. } => Some(report),
            Self::ResumeConflict { .. } => None,
        }
    }

    pub fn is_completed(&self) -> bool {
        matches!(self, Self::Completed { .. })
    }

    pub fn partial(&self) -> Option<&PartialLanguageOutput> {
        match self {
            Self::TimedOut { partial, .. }
            | Self::Failed { partial, .. }
            | Self::Cancelled { partial, .. } => partial.as_ref(),
            _ => None,
        }
    }
}

/// Events emitted by the shared step engine.
#[derive(Debug)]
#[non_exhaustive]
pub enum RunEvent {
    Started {
        target: ModelTarget,
    },
    StepStarted {
        index: u32,
        target: ModelTarget,
    },
    /// A non-terminal event from one established provider stream.
    Model {
        step: u32,
        event: LanguageStreamEvent,
    },
    ToolPrepared {
        step: u32,
        ordinal: usize,
        call: ToolCall,
    },
    ToolCompleted {
        step: u32,
        ordinal: usize,
        result: ToolResult,
    },
    StepFinished {
        record: Box<StepRecord>,
    },
    ModelTransition {
        transition: Box<ModelTransitionRecord>,
    },
    Terminal(RunTerminal),
}

impl RunEvent {
    pub fn terminal(&self) -> Option<&RunTerminal> {
        match self {
            Self::Terminal(terminal) => Some(terminal),
            _ => None,
        }
    }
}

/// Established run stream with structured cancellation and one terminal.
pub struct RunStream {
    inner: Pin<Box<dyn Stream<Item = RunEvent> + Send + 'static>>,
    cancellation: Cancellation,
}

impl fmt::Debug for RunStream {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RunStream")
            .field("is_cancelled", &self.cancellation.is_cancelled())
            .finish_non_exhaustive()
    }
}

impl Stream for RunStream {
    type Item = RunEvent;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(context)
    }
}

impl Drop for RunStream {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

pub(crate) fn established_run_stream<S>(
    cancellation: Cancellation,
    source: S,
    initial_report: RunReport,
) -> RunStream
where
    S: Stream<Item = Result<RunEvent, Error>> + Send + 'static,
{
    let checked = CheckedRunStream {
        source: Box::pin(source),
        terminal_seen: false,
        done: false,
        initial_report: Some(initial_report),
    };
    RunStream {
        inner: Box::pin(checked),
        cancellation,
    }
}

struct CheckedRunStream<S> {
    source: Pin<Box<S>>,
    terminal_seen: bool,
    done: bool,
    initial_report: Option<RunReport>,
}

impl<S> Stream for CheckedRunStream<S>
where
    S: Stream<Item = Result<RunEvent, Error>>,
{
    type Item = RunEvent;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.done {
            return Poll::Ready(None);
        }
        if self.terminal_seen {
            self.done = true;
            return Poll::Ready(None);
        }

        match self.source.as_mut().poll_next(context) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Some(Ok(event))) => {
                if event.terminal().is_some() {
                    self.terminal_seen = true;
                }
                Poll::Ready(Some(event))
            }
            Poll::Ready(Some(Err(error))) => {
                self.terminal_seen = true;
                Poll::Ready(Some(RunEvent::Terminal(RunTerminal::Failed {
                    error,
                    partial: None,
                    report: Box::new(self.take_initial_report()),
                })))
            }
            Poll::Ready(None) => {
                self.terminal_seen = true;
                Poll::Ready(Some(RunEvent::Terminal(RunTerminal::Failed {
                    error: Error::unexpected_eof(),
                    partial: None,
                    report: Box::new(self.take_initial_report()),
                })))
            }
        }
    }
}

impl<S> CheckedRunStream<S> {
    fn take_initial_report(&mut self) -> RunReport {
        self.initial_report
            .take()
            .expect("run lifecycle consumes its fallback report at most once")
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;
    use siumai_core::{ModelId, ProviderId};

    use super::*;

    fn report() -> RunReport {
        RunReport::new(
            ModelTarget::new(
                ProviderId::new("test").unwrap(),
                ModelId::new("model").unwrap(),
            ),
            Vec::new(),
        )
    }

    #[tokio::test]
    async fn eof_without_terminal_becomes_one_failed_terminal() {
        let events =
            established_run_stream(Cancellation::new(), futures::stream::empty(), report())
                .collect::<Vec<_>>()
                .await;

        assert!(matches!(
            events.as_slice(),
            [RunEvent::Terminal(RunTerminal::Failed { error, .. })]
                if error.kind() == siumai_core::ErrorKind::UnexpectedEof
        ));
    }

    #[tokio::test]
    async fn source_error_becomes_terminal_and_later_events_are_not_observed() {
        let source = futures::stream::iter(vec![
            Err(Error::protocol_violation("broken run source")),
            Ok(RunEvent::Terminal(RunTerminal::Completed {
                report: Box::new(report()),
            })),
        ]);
        let events = established_run_stream(Cancellation::new(), source, report())
            .collect::<Vec<_>>()
            .await;

        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            RunEvent::Terminal(RunTerminal::Failed { .. })
        ));
    }

    #[tokio::test]
    async fn dropping_run_stream_cancels_owned_work() {
        let cancellation = Cancellation::new();
        let stream =
            established_run_stream(cancellation.clone(), futures::stream::pending(), report());
        drop(stream);
        assert!(cancellation.is_cancelled());
    }
}
