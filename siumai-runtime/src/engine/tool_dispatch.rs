//! Tool scheduling, checkpointing, and attempt settlement for one engine step.
//!
//! The root engine owns phase transitions, event queueing, provider suspension,
//! and terminal selection. This module owns only the mutable tool-dispatch
//! lifecycle behind the concrete checkpoint port supplied by the caller.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::time::Instant;

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use siumai_core::{
    Cancellation, Error, ErrorKind, LanguageRequest, LanguageResponse, ToolOutcome, ToolResult,
};

use crate::snapshot::{CompletedToolSnapshot, IndeterminateReason, PreparedToolSnapshot};
use crate::tool::{
    AuthorizedToolCall, EffectCertainty, ToolConcurrency, ToolEffect, ToolExecutionError,
    ToolExecutionRequest,
};
use crate::tool_loop::{ToolOutcomeAction, ToolOutcomePolicy};
use crate::{IndeterminateEffect, ModelTarget, RunBudget, RunReport, RunTimeoutKind};

use super::checkpoint::{
    CheckpointBoundary, CheckpointControl, EngineCheckpoint, EngineCheckpointPort,
    EngineCheckpointState, PendingApprovalCheckpoint, PendingToolsCheckpoint,
};
use super::{WaitResult, checked_deadline, classify_deadline, wait_for};

type BoxToolFuture = Pin<Box<dyn Future<Output = ToolAttempt> + Send + 'static>>;

pub(super) struct IndexedAuthorizedCall {
    pub(super) ordinal: usize,
    pub(super) call: AuthorizedToolCall,
}

#[derive(Clone)]
pub(super) struct IndexedResult {
    pub(super) ordinal: usize,
    pub(super) result: ToolResult,
}

pub(super) struct PendingApprovalCall {
    pub(super) request: ToolExecutionRequest,
}

pub(super) struct ToolDispatchContext<'a, P>
where
    P: EngineCheckpointPort,
{
    pub(super) step: u32,
    pub(super) target: &'a ModelTarget,
    pub(super) continuation: &'a LanguageRequest,
    pub(super) response: &'a LanguageResponse,
    pub(super) prepared: &'a [PreparedToolSnapshot],
    pub(super) report: &'a mut RunReport,
    pub(super) budget_limits: &'a RunBudget,
    pub(super) cancellation: &'a Cancellation,
    pub(super) total_deadline: Instant,
    pub(super) outcome_policy: ToolOutcomePolicy,
    pub(super) checkpoint: &'a mut P,
}

pub(super) struct ToolDispatchOutcome {
    pub(super) results: Vec<IndexedResult>,
    pub(super) completed: Vec<IndexedResult>,
    pub(super) pending_approvals: Vec<PendingApprovalCall>,
    pub(super) stop: Option<ToolDispatchStop>,
}

pub(super) enum ToolDispatchStop {
    Policy {
        call_id: String,
        outcome: ToolOutcome,
    },
    Budget(crate::BudgetError),
    TimedOut(RunTimeoutKind),
    Cancelled,
    Indeterminate(IndeterminateEffect),
    Failed(Error),
}

pub(super) struct PendingToolsCheckpointContext<'a> {
    pub(super) step: u32,
    pub(super) target: &'a ModelTarget,
    pub(super) continuation: &'a LanguageRequest,
    pub(super) report: &'a RunReport,
    pub(super) response: &'a LanguageResponse,
    pub(super) prepared: &'a [PreparedToolSnapshot],
    pub(super) results: &'a [IndexedResult],
    pub(super) pending_approvals: &'a [PendingApprovalCall],
    pub(super) can_progress: bool,
}

pub(super) async fn dispatch_tools<P>(
    mut context: ToolDispatchContext<'_, P>,
    requests: Vec<IndexedAuthorizedCall>,
    mut results: Vec<IndexedResult>,
    mut pending_approvals: Vec<PendingApprovalCall>,
) -> Result<ToolDispatchOutcome, P::Error>
where
    P: EngineCheckpointPort,
{
    let previously_completed = results
        .iter()
        .map(|result| result.ordinal)
        .collect::<BTreeSet<_>>();
    let stop = context
        .execute_requests(requests, &mut results, &mut pending_approvals)
        .await?;
    results.sort_unstable_by_key(|result| result.ordinal);
    let completed = results
        .iter()
        .filter(|result| !previously_completed.contains(&result.ordinal))
        .cloned()
        .collect();
    if stop.is_some() {
        context.release_pending_approvals(&mut pending_approvals);
    }
    Ok(ToolDispatchOutcome {
        results,
        completed,
        pending_approvals,
        stop,
    })
}

pub(super) async fn checkpoint_pending_tools<P>(
    checkpoint: &mut P,
    boundary: CheckpointBoundary,
    context: PendingToolsCheckpointContext<'_>,
) -> Result<CheckpointControl, P::Error>
where
    P: EngineCheckpointPort,
{
    if !P::ENABLED {
        return Ok(CheckpointControl::Continue);
    }
    let mut completed = context
        .results
        .iter()
        .map(|result| {
            CompletedToolSnapshot::new(
                u32::try_from(result.ordinal)
                    .expect("response tool call ordinal is representable in a snapshot"),
                result.result.clone(),
            )
        })
        .collect::<Vec<_>>();
    completed.sort_unstable_by_key(CompletedToolSnapshot::ordinal);
    let pending = PendingToolsCheckpoint::new(
        context.step,
        context.target.clone(),
        context.response.clone(),
        context.prepared.to_vec(),
        completed,
        context
            .pending_approvals
            .iter()
            .map(|pending| {
                PendingApprovalCheckpoint::new(
                    pending.request.call().clone(),
                    pending.request.binding_identity().clone(),
                    pending.request.canonical_arguments_digest(),
                )
            })
            .collect(),
        context.can_progress,
    );
    checkpoint
        .commit(EngineCheckpoint::new(
            boundary,
            context.continuation.clone(),
            context.report.clone(),
            EngineCheckpointState::PendingTools(pending),
        ))
        .await
}

impl<P> ToolDispatchContext<'_, P>
where
    P: EngineCheckpointPort,
{
    async fn execute_requests(
        &mut self,
        requests: Vec<IndexedAuthorizedCall>,
        results: &mut Vec<IndexedResult>,
        pending_approvals: &mut Vec<PendingApprovalCall>,
    ) -> Result<Option<ToolDispatchStop>, P::Error> {
        let mut queue = VecDeque::from(requests);

        while let Some(front) = queue.front() {
            if is_parallel_safe(front.call.request()) {
                let mut batch = Vec::new();
                while queue
                    .front()
                    .is_some_and(|request| is_parallel_safe(request.call.request()))
                {
                    if let Some(request) = queue.pop_front() {
                        batch.push(request);
                    }
                }
                if let Some(stop) = self
                    .execute_parallel_batch(batch, results, pending_approvals)
                    .await?
                {
                    return Ok(Some(stop));
                }
            } else if let Some(request) = queue.pop_front() {
                if let Some(stop) = self.pre_dispatch_stop() {
                    return Ok(Some(stop));
                }
                self.release_dispatch_approval(request.call.request(), pending_approvals);
                if let Err(error) = self
                    .report
                    .tool_journal_mut()
                    .dispatch(request.call.request())
                    .map_err(tool_journal_error)
                {
                    return Ok(Some(ToolDispatchStop::Failed(error)));
                }
                self.checkpoint(
                    CheckpointBoundary::BeforeDispatch,
                    results,
                    pending_approvals,
                    true,
                )
                .await?;
                let attempt = self.await_tool(request).await;
                let progress = self.process_attempt(attempt);
                let completed = !progress.results.is_empty();
                results.extend(progress.results);
                results.sort_unstable_by_key(|result| result.ordinal);
                if completed {
                    self.checkpoint(
                        CheckpointBoundary::AfterDispatch,
                        results,
                        pending_approvals,
                        true,
                    )
                    .await?;
                }
                if let Some(stop) = progress.stop {
                    return Ok(Some(stop));
                }
            }
        }

        Ok(None)
    }

    async fn execute_parallel_batch(
        &mut self,
        requests: Vec<IndexedAuthorizedCall>,
        results: &mut Vec<IndexedResult>,
        pending_approvals: &mut Vec<PendingApprovalCall>,
    ) -> Result<Option<ToolDispatchStop>, P::Error> {
        let mut pending = VecDeque::from(requests);
        let mut active = FuturesUnordered::<BoxToolFuture>::new();
        let mut active_by_binding = BTreeMap::<String, usize>::new();
        let mut active_calls = BTreeMap::new();
        let mut policy_stop: Option<(usize, String, ToolOutcome)> = None;
        let max_concurrent = self
            .checkpoint
            .max_concurrent_tools(self.budget_limits.max_concurrent_tools());

        loop {
            while policy_stop.is_none() && active.len() < max_concurrent {
                if let Some(stop) = self.pre_dispatch_stop() {
                    if let Err(error) = self.mark_active_indeterminate(&active_calls) {
                        return Ok(Some(ToolDispatchStop::Failed(error)));
                    }
                    return Ok(Some(stop));
                }
                let eligible = pending.iter().position(|candidate| {
                    let key = binding_key(candidate.call.request());
                    let active_count = active_by_binding.get(&key).copied().unwrap_or(0);
                    active_count < binding_parallel_limit(candidate.call.request())
                });
                let Some(position) = eligible else {
                    break;
                };
                let Some(request) = pending.remove(position) else {
                    break;
                };
                self.release_dispatch_approval(request.call.request(), pending_approvals);
                if let Err(error) = self
                    .report
                    .tool_journal_mut()
                    .dispatch(request.call.request())
                    .map_err(tool_journal_error)
                {
                    if let Err(logging_error) = self.mark_active_indeterminate(&active_calls) {
                        return Ok(Some(ToolDispatchStop::Failed(logging_error)));
                    }
                    return Ok(Some(ToolDispatchStop::Failed(error)));
                }
                self.checkpoint(
                    CheckpointBoundary::BeforeDispatch,
                    results,
                    pending_approvals,
                    true,
                )
                .await?;
                let key = binding_key(request.call.request());
                *active_by_binding.entry(key).or_default() += 1;
                active_calls.insert(
                    request.ordinal,
                    (
                        request.call.request().call_id().to_string(),
                        request.call.request().attempt(),
                    ),
                );
                let cancellation = self.cancellation.clone();
                let total_deadline = self.total_deadline;
                let tool_timeout = self.budget_limits.timeouts().tool();
                active.push(Box::pin(async move {
                    await_tool_attempt(request, cancellation, total_deadline, tool_timeout).await
                }));
            }

            if active.is_empty() {
                break;
            }
            let Some(attempt) = active.next().await else {
                break;
            };
            let key = binding_key(&attempt.request);
            if let Some(count) = active_by_binding.get_mut(&key) {
                *count = count.saturating_sub(1);
            }
            active_calls.remove(&attempt.ordinal);
            let progress = self.process_attempt(attempt);
            let completed = !progress.results.is_empty();
            results.extend(progress.results.iter().cloned());
            results.sort_unstable_by_key(|result| result.ordinal);
            if completed {
                self.checkpoint(
                    CheckpointBoundary::AfterDispatch,
                    results,
                    pending_approvals,
                    true,
                )
                .await?;
            }
            if let Some(stop) = progress.stop {
                match stop {
                    ToolDispatchStop::Policy { call_id, outcome } => {
                        let ordinal = progress
                            .results
                            .first()
                            .map_or(usize::MAX, |result| result.ordinal);
                        let replace = policy_stop
                            .as_ref()
                            .is_none_or(|(current, _, _)| ordinal < *current);
                        if replace {
                            policy_stop = Some((ordinal, call_id, outcome));
                        }
                    }
                    terminal => {
                        if let Err(error) = self.mark_active_indeterminate(&active_calls) {
                            return Ok(Some(ToolDispatchStop::Failed(error)));
                        }
                        return Ok(Some(terminal));
                    }
                }
            }
        }

        Ok(policy_stop.map(|(_, call_id, outcome)| ToolDispatchStop::Policy { call_id, outcome }))
    }

    async fn checkpoint(
        &mut self,
        boundary: CheckpointBoundary,
        results: &[IndexedResult],
        pending_approvals: &[PendingApprovalCall],
        can_progress: bool,
    ) -> Result<CheckpointControl, P::Error> {
        checkpoint_pending_tools(
            self.checkpoint,
            boundary,
            PendingToolsCheckpointContext {
                step: self.step,
                target: self.target,
                continuation: self.continuation,
                report: self.report,
                response: self.response,
                prepared: self.prepared,
                results,
                pending_approvals,
                can_progress,
            },
        )
        .await
    }

    async fn await_tool(&self, request: IndexedAuthorizedCall) -> ToolAttempt {
        await_tool_attempt(
            request,
            self.cancellation.clone(),
            self.total_deadline,
            self.budget_limits.timeouts().tool(),
        )
        .await
    }

    fn process_attempt(&mut self, attempt: ToolAttempt) -> ExecutionProgress {
        let effect = attempt.request.effect();
        let call_id = attempt.request.call_id().to_string();
        let execution_attempt = attempt.request.attempt();
        let binding_fingerprint = attempt.request.binding_identity().fingerprint.clone();
        let ordinal = attempt.ordinal;

        let result = match attempt.outcome {
            ToolAttemptOutcome::Completed(Ok(result)) => result,
            ToolAttemptOutcome::Completed(Err(error)) => {
                if error.effect_certainty() == EffectCertainty::Indeterminate
                    && effect == ToolEffect::SideEffecting
                {
                    if let Err(error) = self
                        .report
                        .tool_journal_mut()
                        .mark_indeterminate(
                            &call_id,
                            execution_attempt,
                            IndeterminateReason::DispatchOutcomeUnknown,
                        )
                        .map_err(tool_journal_error)
                    {
                        return ExecutionProgress::stopped(ToolDispatchStop::Failed(error));
                    }
                    return ExecutionProgress::stopped(ToolDispatchStop::Indeterminate(
                        IndeterminateEffect {
                            call_id,
                            binding_fingerprint,
                            reason: "tool effect is indeterminate after executor failure"
                                .to_string(),
                        },
                    ));
                }
                ToolResult {
                    call_id: call_id.clone(),
                    name: attempt.request.name().to_string(),
                    outcome: execution_error_outcome(error),
                }
            }
            ToolAttemptOutcome::TimedOut(kind) => {
                if let Err(error) = self
                    .report
                    .tool_journal_mut()
                    .mark_indeterminate(
                        &call_id,
                        execution_attempt,
                        IndeterminateReason::DispatchOutcomeUnknown,
                    )
                    .map_err(tool_journal_error)
                {
                    return ExecutionProgress::stopped(ToolDispatchStop::Failed(error));
                }
                if effect == ToolEffect::SideEffecting {
                    return ExecutionProgress::stopped(ToolDispatchStop::Indeterminate(
                        IndeterminateEffect {
                            call_id,
                            binding_fingerprint,
                            reason: "tool execution timed out after dispatch".to_string(),
                        },
                    ));
                }
                return ExecutionProgress::stopped(ToolDispatchStop::TimedOut(kind));
            }
            ToolAttemptOutcome::Cancelled => {
                if let Err(error) = self
                    .report
                    .tool_journal_mut()
                    .mark_indeterminate(
                        &call_id,
                        execution_attempt,
                        IndeterminateReason::CancellationAfterDispatch,
                    )
                    .map_err(tool_journal_error)
                {
                    return ExecutionProgress::stopped(ToolDispatchStop::Failed(error));
                }
                if effect == ToolEffect::SideEffecting {
                    return ExecutionProgress::stopped(ToolDispatchStop::Indeterminate(
                        IndeterminateEffect {
                            call_id,
                            binding_fingerprint,
                            reason: "tool execution was cancelled after dispatch".to_string(),
                        },
                    ));
                }
                return ExecutionProgress::stopped(ToolDispatchStop::Cancelled);
            }
        };

        let result_bytes = match serde_json::to_vec(&result) {
            Ok(value) => value.len(),
            Err(error) => {
                if let Err(logging_error) = self
                    .report
                    .tool_journal_mut()
                    .mark_indeterminate(
                        &call_id,
                        execution_attempt,
                        IndeterminateReason::CheckpointFailure,
                    )
                    .map_err(tool_journal_error)
                {
                    return ExecutionProgress::stopped(ToolDispatchStop::Failed(logging_error));
                }
                if effect == ToolEffect::SideEffecting {
                    return ExecutionProgress::stopped(ToolDispatchStop::Indeterminate(
                        IndeterminateEffect {
                            call_id,
                            binding_fingerprint,
                            reason: "tool result could not be checkpointed".to_string(),
                        },
                    ));
                }
                return ExecutionProgress::stopped(ToolDispatchStop::Failed(
                    Error::new(
                        ErrorKind::Internal,
                        "tool result could not be serialized for budget accounting",
                    )
                    .with_source(error),
                ));
            }
        };
        if let Err(error) = self
            .report
            .budget_mut()
            .charge_tool_result(result_bytes, self.budget_limits)
        {
            if let Err(logging_error) = self
                .report
                .tool_journal_mut()
                .mark_indeterminate(
                    &call_id,
                    execution_attempt,
                    IndeterminateReason::CheckpointFailure,
                )
                .map_err(tool_journal_error)
            {
                return ExecutionProgress::stopped(ToolDispatchStop::Failed(logging_error));
            }
            if effect == ToolEffect::SideEffecting {
                return ExecutionProgress::stopped(ToolDispatchStop::Indeterminate(
                    IndeterminateEffect {
                        call_id,
                        binding_fingerprint,
                        reason: "tool result exceeded checkpoint budget".to_string(),
                    },
                ));
            }
            return ExecutionProgress::stopped(ToolDispatchStop::Budget(error));
        }
        if let Err(error) = self
            .report
            .tool_journal_mut()
            .complete(&attempt.request, &result)
            .map_err(tool_journal_error)
        {
            if let Err(logging_error) = self
                .report
                .tool_journal_mut()
                .mark_indeterminate(
                    &call_id,
                    execution_attempt,
                    IndeterminateReason::CheckpointFailure,
                )
                .map_err(tool_journal_error)
            {
                return ExecutionProgress::stopped(ToolDispatchStop::Failed(logging_error));
            }
            if effect == ToolEffect::SideEffecting {
                return ExecutionProgress::stopped(ToolDispatchStop::Indeterminate(
                    IndeterminateEffect {
                        call_id,
                        binding_fingerprint,
                        reason: "tool completion could not be checkpointed".to_string(),
                    },
                ));
            }
            return ExecutionProgress::stopped(ToolDispatchStop::Failed(error));
        }

        let stop =
            (self.outcome_policy.action(&result.outcome) == ToolOutcomeAction::Stop).then(|| {
                ToolDispatchStop::Policy {
                    call_id: result.call_id.clone(),
                    outcome: result.outcome.clone(),
                }
            });
        ExecutionProgress {
            results: vec![IndexedResult { ordinal, result }],
            stop,
        }
    }

    fn mark_active_indeterminate(
        &mut self,
        active: &BTreeMap<usize, (String, crate::tool::ToolExecutionAttempt)>,
    ) -> Result<(), Error> {
        for (call_id, attempt) in active.values() {
            self.report
                .tool_journal_mut()
                .mark_indeterminate(
                    call_id,
                    *attempt,
                    IndeterminateReason::DispatchOutcomeUnknown,
                )
                .map_err(tool_journal_error)?;
        }
        Ok(())
    }

    fn pre_dispatch_stop(&self) -> Option<ToolDispatchStop> {
        if self.cancellation.is_cancelled() {
            Some(ToolDispatchStop::Cancelled)
        } else if Instant::now() >= self.total_deadline {
            Some(ToolDispatchStop::TimedOut(RunTimeoutKind::Total))
        } else {
            None
        }
    }

    fn release_dispatch_approval(
        &mut self,
        request: &ToolExecutionRequest,
        pending: &mut Vec<PendingApprovalCall>,
    ) {
        if let Some(position) = pending
            .iter()
            .position(|approval| approval.request.call_id() == request.call_id())
        {
            pending.remove(position);
            self.report.budget_mut().release_pending_approval();
        }
    }

    fn release_pending_approvals(&mut self, pending: &mut Vec<PendingApprovalCall>) {
        for _ in pending.drain(..) {
            self.report.budget_mut().release_pending_approval();
        }
    }
}

struct ToolAttempt {
    ordinal: usize,
    request: ToolExecutionRequest,
    outcome: ToolAttemptOutcome,
}

enum ToolAttemptOutcome {
    Completed(Result<ToolResult, ToolExecutionError>),
    TimedOut(RunTimeoutKind),
    Cancelled,
}

struct ExecutionProgress {
    results: Vec<IndexedResult>,
    stop: Option<ToolDispatchStop>,
}

impl ExecutionProgress {
    fn stopped(stop: ToolDispatchStop) -> Self {
        Self {
            results: Vec::new(),
            stop: Some(stop),
        }
    }
}

async fn await_tool_attempt(
    request: IndexedAuthorizedCall,
    cancellation: Cancellation,
    total_deadline: Instant,
    tool_timeout: std::time::Duration,
) -> ToolAttempt {
    let tool_deadline = checked_deadline(Instant::now(), tool_timeout);
    let (deadline, kind) = classify_deadline(total_deadline, tool_deadline, RunTimeoutKind::Tool);
    let IndexedAuthorizedCall { ordinal, call } = request;
    let frozen_request = call.request().clone();
    let outcome = match wait_for(call.dispatch(), &cancellation, deadline, kind).await {
        WaitResult::Ready(result) => ToolAttemptOutcome::Completed(result),
        WaitResult::TimedOut(kind) => ToolAttemptOutcome::TimedOut(kind),
        WaitResult::Cancelled => ToolAttemptOutcome::Cancelled,
    };
    ToolAttempt {
        ordinal,
        request: frozen_request,
        outcome,
    }
}

fn is_parallel_safe(request: &ToolExecutionRequest) -> bool {
    request.effect() == ToolEffect::ReadOnly
        && matches!(request.concurrency(), ToolConcurrency::SafeParallel { .. })
}

fn binding_key(request: &ToolExecutionRequest) -> String {
    request.binding_identity().fingerprint.clone()
}

fn binding_parallel_limit(request: &ToolExecutionRequest) -> usize {
    match request.concurrency() {
        ToolConcurrency::Sequential => 1,
        ToolConcurrency::SafeParallel { max_in_flight } => max_in_flight.get(),
    }
}

pub(super) fn execution_error_outcome(error: ToolExecutionError) -> ToolOutcome {
    match error {
        ToolExecutionError::ExecutorFailed {
            message, retryable, ..
        } => ToolOutcome::ExecutionFailed {
            message,
            retryable,
            details: None,
        },
        ToolExecutionError::InvalidArguments { message, .. } => ToolOutcome::ExecutionFailed {
            message,
            retryable: false,
            details: None,
        },
        other => ToolOutcome::ExecutionFailed {
            message: other.to_string(),
            retryable: false,
            details: None,
        },
    }
}

pub(super) fn tool_journal_error(error: crate::tool::ToolJournalError) -> Error {
    Error::new(ErrorKind::Internal, "tool journal transition failed").with_source(error)
}
