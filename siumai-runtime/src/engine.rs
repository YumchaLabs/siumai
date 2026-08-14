pub(crate) mod checkpoint;

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use futures::stream::FuturesUnordered;
use futures::{Stream, StreamExt, stream};
use siumai_core::stream::established_stream;
use siumai_core::{
    CallOptions, Cancellation, ContentPart, Error, ErrorKind, LanguageModel, LanguageRequest,
    LanguageResponse, LanguageStream, LanguageStreamEvent, Message, MessagePart, MessageRole,
    PartialLanguageOutput, StreamTerminal, ToolCall, ToolOutcome, ToolResult, Usage,
};

use crate::approval::VerifiedApproval;
use crate::selection::{
    PreparedStepModel, SelectedStepModel, prepare_selected_step_model, select_step_model,
};
use crate::single_step::SingleStep;
use crate::snapshot::{
    CompletedToolSnapshot, IndeterminateReason, PendingProviderStepSnapshot, PendingStepSnapshot,
    PreparedToolSnapshot, ProviderStateSnapshot, ResumePoint, SnapshotReason, SnapshotTerminal,
    ToolExecutionEvent, ToolExecutionStatus,
};
use crate::tool::{
    ApprovalDecider, ApprovalDecision, ApprovalPolicy, ApprovalRequest, AuthorizedToolCall,
    EffectCertainty, PreparedVisibleToolCatalog, ToolConcurrency, ToolEffect, ToolExecutionError,
    ToolExecutionRequest, ToolSet, VisibleToolCatalogError, VisibleToolCatalogSource,
    prepare_visible_tool_catalog,
};
use crate::tool_loop::{ToolOutcomeAction, ToolOutcomePolicy};
use crate::usage::CallUsageReconciler;
use crate::{
    IndeterminateEffect, ModelTarget, ProjectionPolicy, RunBudget, RunEvent, RunReport,
    RunStopReason, RunTerminal, RunTimeoutKind, Runtime, StepModelSelector, StepOptions,
    StepRecord, SuspensionReason,
};

use self::checkpoint::{
    CheckpointBoundary, CheckpointControl, EngineCheckpoint, EngineCheckpointPort,
    EngineCheckpointState, EphemeralCheckpointPort, PendingApprovalCheckpoint,
    PendingToolsCheckpoint,
};

type BoxToolFuture = Pin<Box<dyn Future<Output = ToolAttempt> + Send + 'static>>;

pub(crate) struct StepEngine {
    runtime: Runtime,
    model: Arc<dyn LanguageModel>,
    tools: ToolSet,
    visible_tools: PreparedVisibleToolCatalog,
    request: LanguageRequest,
    step_options: StepOptions,
    call_options: CallOptions,
    outcome_policy: ToolOutcomePolicy,
    approval_decider: Arc<dyn ApprovalDecider>,
    model_selector: Option<Arc<dyn StepModelSelector>>,
    projection_policy: ProjectionPolicy,
    tool_handling: ToolHandling,
    budget: RunBudget,
    cancellation: Cancellation,
    target: ModelTarget,
    total_deadline: Instant,
    report: RunReport,
    step: u32,
    phase: EnginePhase,
    pending: VecDeque<RunEvent>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolHandling {
    Execute,
    ObserveOnly,
}

enum EnginePhase {
    Transitioning,
    Streaming(Box<StepStream>),
    Tools(Box<PendingToolPhase>),
    ReadyForModel,
    ReadyForSelectedModel(Box<SelectedStepModel>),
    Paused,
    Terminal,
}

pub(crate) struct EngineResumeSeed {
    pub(crate) continuation: LanguageRequest,
    pub(crate) visible_tools: PreparedVisibleToolCatalog,
    pub(crate) report: RunReport,
    pub(crate) resume_point: ResumePoint,
    pub(crate) verified_approvals: BTreeMap<String, VerifiedApproval>,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum EngineResumeError {
    #[error("selected model changed while resuming a frozen durable step")]
    SelectedModelTargetChanged {
        expected: Box<ModelTarget>,
        actual: Box<ModelTarget>,
    },
    #[error("recovery is not permitted for tool call `{call_id}`")]
    RecoveryNotPermitted { call_id: String },
    #[error("restored frozen request does not match tool call `{call_id}`")]
    FrozenRequestMismatch { call_id: String },
    #[error(transparent)]
    Runtime(#[from] Error),
}

impl StepEngine {
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn establish(
        runtime: Runtime,
        model: Arc<dyn LanguageModel>,
        tools: ToolSet,
        request: LanguageRequest,
        step_options: StepOptions,
        options: CallOptions,
        outcome_policy: ToolOutcomePolicy,
        approval_decider: Arc<dyn ApprovalDecider>,
        model_selector: Option<Arc<dyn StepModelSelector>>,
        projection_policy: ProjectionPolicy,
        tool_handling: ToolHandling,
    ) -> Result<Self, Error> {
        let target = ModelTarget::from_model(model.as_ref());
        let report = RunReport::new(target, request.messages.clone());
        Self::establish_seeded_inner(
            runtime,
            model,
            tools,
            request,
            step_options,
            options,
            outcome_policy,
            approval_decider,
            model_selector,
            projection_policy,
            tool_handling,
            report,
            0,
            false,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn establish_seeded(
        runtime: Runtime,
        model: Arc<dyn LanguageModel>,
        tools: ToolSet,
        request: LanguageRequest,
        step_options: StepOptions,
        options: CallOptions,
        outcome_policy: ToolOutcomePolicy,
        approval_decider: Arc<dyn ApprovalDecider>,
        model_selector: Option<Arc<dyn StepModelSelector>>,
        projection_policy: ProjectionPolicy,
        tool_handling: ToolHandling,
        report: RunReport,
        step: u32,
    ) -> Result<Self, Error> {
        Self::establish_seeded_inner(
            runtime,
            model,
            tools,
            request,
            step_options,
            options,
            outcome_policy,
            approval_decider,
            model_selector,
            projection_policy,
            tool_handling,
            report,
            step,
            true,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn establish_seeded_inner(
        runtime: Runtime,
        model: Arc<dyn LanguageModel>,
        tools: ToolSet,
        mut request: LanguageRequest,
        step_options: StepOptions,
        options: CallOptions,
        outcome_policy: ToolOutcomePolicy,
        approval_decider: Arc<dyn ApprovalDecider>,
        model_selector: Option<Arc<dyn StepModelSelector>>,
        projection_policy: ProjectionPolicy,
        tool_handling: ToolHandling,
        report: RunReport,
        step: u32,
        establishment_failure_as_terminal: bool,
    ) -> Result<Self, Error> {
        let visible_tools = prepare_visible_tool_catalog(
            VisibleToolCatalogSource::Caller(std::mem::take(&mut request.tools)),
            &tools,
        )
        .map_err(visible_catalog_error)?;
        request.tools = visible_tools.specs().to_vec();

        let budget = runtime.run_budget().clone();
        let cancellation = options.cancellation().child();
        let total_deadline = budget.deadline_from(Instant::now(), options.deadline());
        let call_options = options
            .with_cancellation(cancellation.clone())
            .with_deadline(total_deadline);
        let target = ModelTarget::from_model(model.as_ref());
        if report.current_target() != &target {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "seeded run report targets a different active model",
            ));
        }
        request.messages = report.messages().to_vec();
        let mut pending = VecDeque::new();
        if step == 0 {
            pending.push_back(RunEvent::Started {
                target: target.clone(),
            });
        }
        pending.push_back(RunEvent::StepStarted {
            index: step,
            target: target.clone(),
        });
        let mut engine = Self {
            runtime,
            model,
            tools,
            visible_tools,
            request,
            step_options,
            call_options,
            outcome_policy,
            approval_decider,
            model_selector,
            projection_policy,
            tool_handling,
            budget,
            cancellation,
            target: target.clone(),
            total_deadline,
            report,
            step,
            phase: EnginePhase::Transitioning,
            pending,
        };

        if let Err(error) = engine.report.budget_mut().charge_model_step(&engine.budget) {
            if establishment_failure_as_terminal {
                engine.queue_terminal(RunTerminal::BudgetExceeded {
                    error,
                    report: Box::new(engine.report.clone()),
                });
                return Ok(engine);
            }
            return Err(budget_start_error(error));
        }
        match engine.establish_model_stream(step == 0).await {
            Ok(stream) => {
                engine.phase = EnginePhase::Streaming(Box::new(stream));
                Ok(engine)
            }
            Err(error) if establishment_failure_as_terminal => {
                engine.queue_handshake_failure(error);
                Ok(engine)
            }
            Err(error) => Err(error),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn resume(
        runtime: Runtime,
        model: Arc<dyn LanguageModel>,
        tools: ToolSet,
        step_options: StepOptions,
        options: CallOptions,
        outcome_policy: ToolOutcomePolicy,
        approval_decider: Arc<dyn ApprovalDecider>,
        model_selector: Option<Arc<dyn StepModelSelector>>,
        projection_policy: ProjectionPolicy,
        seed: EngineResumeSeed,
    ) -> Result<Self, EngineResumeError> {
        let EngineResumeSeed {
            mut continuation,
            visible_tools,
            report,
            resume_point,
            verified_approvals,
        } = seed;
        continuation.messages = report.messages().to_vec();
        continuation.tools = visible_tools.specs().to_vec();

        let budget = runtime.run_budget().clone();
        let cancellation = options.cancellation().child();
        let total_deadline = budget.deadline_from(Instant::now(), options.deadline());
        let call_options = options
            .with_cancellation(cancellation.clone())
            .with_deadline(total_deadline);
        let target = report.current_target().clone();
        let mut engine = Self {
            runtime,
            model,
            tools,
            visible_tools,
            request: continuation,
            step_options,
            call_options,
            outcome_policy,
            approval_decider,
            model_selector,
            projection_policy,
            tool_handling: ToolHandling::Execute,
            budget,
            cancellation,
            target,
            total_deadline,
            report,
            step: 0,
            phase: EnginePhase::Transitioning,
            pending: VecDeque::new(),
        };

        match resume_point {
            ResumePoint::AwaitingApprovals(step) | ResumePoint::ReadyToDispatch(step) => {
                engine.step = step.index();
                if step.target() != &engine.target {
                    return Err(Error::new(
                        ErrorKind::InvalidInput,
                        "pending durable step targets a different active model",
                    )
                    .into());
                }
                engine.phase = EnginePhase::Tools(Box::new(
                    engine.restore_pending_phase(step, verified_approvals)?,
                ));
            }
            ResumePoint::ReadyForModel { next_step, target } => {
                engine.step = next_step;
                let selected = engine.select_next_model()?;
                if selected.target != target {
                    return Err(EngineResumeError::SelectedModelTargetChanged {
                        expected: Box::new(target),
                        actual: Box::new(selected.target),
                    });
                }
                engine.phase = EnginePhase::ReadyForSelectedModel(Box::new(selected));
            }
            ResumePoint::AwaitingProvider(_) | ResumePoint::Terminal(_) => {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    "durable resume point does not contain locally runnable work",
                )
                .into());
            }
        }
        Ok(engine)
    }

    pub(crate) fn visible_tool_catalog(&self) -> &PreparedVisibleToolCatalog {
        &self.visible_tools
    }

    pub(crate) fn cancellation(&self) -> &Cancellation {
        &self.cancellation
    }

    pub(crate) fn report(&self) -> &RunReport {
        &self.report
    }

    pub(crate) fn into_stream(
        self,
    ) -> impl Stream<Item = Result<RunEvent, Error>> + Send + 'static {
        stream::unfold(
            (self, EphemeralCheckpointPort),
            |(mut engine, mut checkpoint)| async move {
                let event = match engine.next_event(&mut checkpoint).await {
                    Ok(event) => event,
                    Err(error) => match error {},
                }?;
                Some((Ok(event), (engine, checkpoint)))
            },
        )
    }

    pub(crate) async fn drive<P>(&mut self, checkpoint: &mut P) -> Result<(), P::Error>
    where
        P: EngineCheckpointPort,
    {
        while self.next_event(checkpoint).await?.is_some() {}
        Ok(())
    }

    async fn next_event<P>(&mut self, checkpoint: &mut P) -> Result<Option<RunEvent>, P::Error>
    where
        P: EngineCheckpointPort,
    {
        loop {
            if let Some(event) = self.pending.pop_front() {
                if let RunEvent::Terminal(terminal) = &event
                    && P::ENABLED
                {
                    let control = self
                        .commit_terminal(checkpoint, terminal_checkpoint(terminal))
                        .await?;
                    if control == CheckpointControl::Pause {
                        self.phase = EnginePhase::Paused;
                        return Ok(None);
                    }
                }
                return Ok(Some(event));
            }
            if matches!(self.phase, EnginePhase::Paused | EnginePhase::Terminal) {
                return Ok(None);
            }
            let model_stream_active = matches!(self.phase, EnginePhase::Streaming(_));
            if self.cancellation.is_cancelled() && !model_stream_active {
                self.queue_terminal(RunTerminal::Cancelled {
                    reason: "tool loop cancelled".to_string(),
                    partial: None,
                    report: Box::new(self.report.clone()),
                });
                continue;
            }
            if Instant::now() >= self.total_deadline && !model_stream_active {
                self.queue_terminal(RunTerminal::TimedOut {
                    kind: RunTimeoutKind::Total,
                    partial: None,
                    report: Box::new(self.report.clone()),
                });
                continue;
            }
            match std::mem::replace(&mut self.phase, EnginePhase::Transitioning) {
                EnginePhase::Streaming(stream) => {
                    self.poll_model_stream(*stream, checkpoint).await?
                }
                EnginePhase::Tools(prepared) => {
                    self.execute_prepared_step(*prepared, checkpoint).await?
                }
                EnginePhase::ReadyForModel => {
                    self.step = self.step.saturating_add(1);
                    self.establish_later_step(checkpoint).await?;
                }
                EnginePhase::ReadyForSelectedModel(selected) => {
                    self.establish_selected_step(*selected, checkpoint).await?;
                }
                EnginePhase::Paused | EnginePhase::Terminal => return Ok(None),
                EnginePhase::Transitioning => self.queue_terminal(RunTerminal::Failed {
                    error: Error::new(
                        ErrorKind::Internal,
                        "tool loop reached an invalid engine phase",
                    ),
                    partial: None,
                    report: Box::new(self.report.clone()),
                }),
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn checkpoint_pending_tools<P>(
        &mut self,
        checkpoint: &mut P,
        boundary: CheckpointBoundary,
        response: &LanguageResponse,
        prepared: &[PreparedToolSnapshot],
        results: &[IndexedResult],
        pending_approvals: &[PendingApprovalCall],
        can_progress: bool,
    ) -> Result<CheckpointControl, P::Error>
    where
        P: EngineCheckpointPort,
    {
        if !P::ENABLED {
            return Ok(CheckpointControl::Continue);
        }
        let mut completed = results
            .iter()
            .filter_map(|result| {
                prepared
                    .iter()
                    .find(|prepared| {
                        usize::try_from(prepared.ordinal()).ok() == Some(result.ordinal)
                    })
                    .map(|prepared| {
                        CompletedToolSnapshot::new(prepared.ordinal(), result.result.clone())
                    })
            })
            .collect::<Vec<_>>();
        completed.sort_unstable_by_key(CompletedToolSnapshot::ordinal);
        let pending = PendingToolsCheckpoint::new(
            self.step,
            self.target.clone(),
            response.clone(),
            prepared.to_vec(),
            completed,
            pending_approvals
                .iter()
                .map(|pending| {
                    PendingApprovalCheckpoint::new(
                        pending.request.call().clone(),
                        pending.request.binding_identity().clone(),
                        pending.request.canonical_arguments_digest(),
                    )
                })
                .collect(),
            can_progress,
        );
        checkpoint
            .commit(EngineCheckpoint::new(
                boundary,
                self.continuation(),
                self.report.clone(),
                EngineCheckpointState::PendingTools(pending),
            ))
            .await
    }

    async fn checkpoint_state<P>(
        &mut self,
        checkpoint: &mut P,
        boundary: CheckpointBoundary,
        state: EngineCheckpointState,
    ) -> Result<CheckpointControl, P::Error>
    where
        P: EngineCheckpointPort,
    {
        if !P::ENABLED {
            return Ok(CheckpointControl::Continue);
        }
        checkpoint
            .commit(EngineCheckpoint::new(
                boundary,
                self.continuation(),
                self.report.clone(),
                state,
            ))
            .await
    }

    async fn commit_terminal<P>(
        &mut self,
        checkpoint: &mut P,
        terminal: SnapshotTerminal,
    ) -> Result<CheckpointControl, P::Error>
    where
        P: EngineCheckpointPort,
    {
        self.checkpoint_state(
            checkpoint,
            CheckpointBoundary::Quiescent,
            EngineCheckpointState::Terminal(terminal),
        )
        .await
    }

    fn continuation(&self) -> LanguageRequest {
        let mut continuation = self.request.clone();
        continuation.messages = self.report.messages().to_vec();
        continuation.tools = self.visible_tools.specs().to_vec();
        continuation
    }

    async fn establish_model_stream(&mut self, first: bool) -> Result<StepStream, Error> {
        let now = Instant::now();
        let step_deadline = self
            .total_deadline
            .min(checked_deadline(now, self.budget.timeouts().model_step()));
        let (deadline, timeout_kind) = classify_deadline(
            self.total_deadline,
            step_deadline,
            RunTimeoutKind::ModelStep,
        );
        let mut request = self.request.clone();
        request.messages = self.report.messages().to_vec();
        request.tools = self.visible_tools.specs().to_vec();
        let single_step = SingleStep::new(&self.runtime, self.model.as_ref(), &self.step_options);
        let future = single_step.stream(request, self.call_options.clone());

        match wait_for(future, &self.cancellation, deadline, timeout_kind).await {
            WaitResult::Ready(result) => result.map(|stream| {
                let timeout = Arc::new(StreamTimeoutState::default());
                let stream = stream_with_runtime_deadlines(
                    stream,
                    self.cancellation.clone(),
                    self.total_deadline,
                    step_deadline,
                    self.budget.timeouts().first_chunk(),
                    self.budget.timeouts().inter_chunk(),
                    Arc::clone(&timeout),
                );
                StepStream {
                    stream,
                    provider_states: Vec::new(),
                    provider_result_ids: BTreeSet::new(),
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

    async fn establish_later_step<P>(&mut self, checkpoint: &mut P) -> Result<(), P::Error>
    where
        P: EngineCheckpointPort,
    {
        let selected = match self.select_next_model() {
            Ok(selected) => selected,
            Err(error) => {
                self.queue_terminal(RunTerminal::Failed {
                    error,
                    partial: None,
                    report: Box::new(self.report.clone()),
                });
                return Ok(());
            }
        };
        self.establish_selected_step(selected, checkpoint).await
    }

    async fn establish_selected_step<P>(
        &mut self,
        selected: SelectedStepModel,
        checkpoint: &mut P,
    ) -> Result<(), P::Error>
    where
        P: EngineCheckpointPort,
    {
        let frozen_target = selected.target.clone();
        let control = self
            .checkpoint_state(
                checkpoint,
                CheckpointBoundary::Quiescent,
                EngineCheckpointState::ReadyForModel {
                    next_step: self.step,
                    target: frozen_target.clone(),
                },
            )
            .await?;
        if control == CheckpointControl::Pause {
            self.phase = EnginePhase::Paused;
            return Ok(());
        }

        match self.prepare_next_model(selected) {
            Ok(true) => {}
            Ok(false) => return Ok(()),
            Err(error) => {
                self.queue_terminal(RunTerminal::Failed {
                    error,
                    partial: None,
                    report: Box::new(self.report.clone()),
                });
                return Ok(());
            }
        }

        let control = self
            .checkpoint_state(
                checkpoint,
                CheckpointBoundary::Quiescent,
                EngineCheckpointState::ReadyForModel {
                    next_step: self.step,
                    target: frozen_target,
                },
            )
            .await?;
        if control == CheckpointControl::Pause {
            self.phase = EnginePhase::Paused;
            return Ok(());
        }

        if let Err(error) = self.report.budget_mut().charge_model_step(&self.budget) {
            self.queue_terminal(RunTerminal::BudgetExceeded {
                error,
                report: Box::new(self.report.clone()),
            });
            return Ok(());
        }

        match self.establish_model_stream(false).await {
            Ok(stream) => {
                self.pending.push_back(RunEvent::StepStarted {
                    index: self.step,
                    target: self.target.clone(),
                });
                self.phase = EnginePhase::Streaming(Box::new(stream));
            }
            Err(error) => self.queue_handshake_failure(error),
        }
        Ok(())
    }

    fn select_next_model(&self) -> Result<SelectedStepModel, Error> {
        let previous_step = self.report.steps().last().ok_or_else(|| {
            Error::new(
                ErrorKind::Internal,
                "model selection requires a completed previous step",
            )
        })?;
        select_step_model(
            self.model_selector.as_deref(),
            Arc::clone(&self.model),
            self.step,
            &self.target,
            previous_step,
            &self.report,
        )
    }

    fn prepare_next_model(&mut self, selected: SelectedStepModel) -> Result<bool, Error> {
        let mut request = self.request.clone();
        request.messages = self.report.messages().to_vec();
        request.tools = self.visible_tools.specs().to_vec();
        match prepare_selected_step_model(
            selected,
            request,
            self.step,
            &self.target,
            self.projection_policy,
        ) {
            PreparedStepModel::Ready(ready) => {
                let crate::selection::PreparedStepModelReady {
                    model,
                    target,
                    request,
                    transition,
                } = *ready;
                self.model = model;
                self.target = target;
                self.report.replace_messages(request.messages.clone());
                self.request = request;
                if let Some(transition) = transition {
                    self.report.model_transitions_mut().push(transition.clone());
                    self.pending.push_back(RunEvent::ModelTransition {
                        transition: Box::new(transition),
                    });
                }
                Ok(true)
            }
            PreparedStepModel::Rejected { transition } => {
                let transition = *transition;
                self.report.model_transitions_mut().push(transition.clone());
                self.pending.push_back(RunEvent::ModelTransition {
                    transition: Box::new(transition.clone()),
                });
                self.queue_terminal(RunTerminal::HistoryProjectionRejected {
                    transition: Box::new(transition),
                    report: Box::new(self.report.clone()),
                });
                Ok(false)
            }
        }
    }

    fn restore_pending_phase(
        &self,
        step: PendingStepSnapshot,
        mut verified_approvals: BTreeMap<String, VerifiedApproval>,
    ) -> Result<PendingToolPhase, EngineResumeError> {
        let mut requests = Vec::new();
        let mut results = step
            .completed()
            .iter()
            .map(|completed| {
                Ok(IndexedResult {
                    ordinal: usize::try_from(completed.ordinal()).map_err(|error| {
                        Error::new(
                            ErrorKind::LimitExceeded,
                            "completed tool ordinal cannot be restored on this platform",
                        )
                        .with_source(error)
                    })?,
                    result: completed.result().clone(),
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        results.sort_unstable_by_key(|result| result.ordinal);
        let completed = results
            .iter()
            .map(|result| result.ordinal)
            .collect::<BTreeSet<_>>();
        let approval_calls = step
            .pending_approvals()
            .iter()
            .map(|approval| approval.call.id())
            .collect::<BTreeSet<_>>();
        let mut pending_approvals = Vec::new();

        for prepared in step.prepared() {
            let ordinal = usize::try_from(prepared.ordinal()).map_err(|error| {
                Error::new(
                    ErrorKind::LimitExceeded,
                    "prepared tool ordinal cannot be restored on this platform",
                )
                .with_source(error)
            })?;
            if completed.contains(&ordinal) {
                continue;
            }
            if self.report.execution_log().status(prepared.call().id())
                != Some(ToolExecutionStatus::Prepared)
            {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    "durable pending tool is not in a dispatchable Prepared state",
                )
                .into());
            }
            let request = self.restore_request(prepared)?;
            match request.approval_policy() {
                ApprovalPolicy::NotRequired => {
                    let call = request
                        .authorize_not_required()
                        .map_err(execution_authorization_error)?;
                    requests.push(IndexedAuthorizedCall { ordinal, call });
                }
                ApprovalPolicy::Required => {
                    if !approval_calls.contains(request.call_id()) {
                        return Err(EngineResumeError::FrozenRequestMismatch {
                            call_id: request.call_id().to_string(),
                        });
                    }
                    if let Some(approval) = verified_approvals.remove(request.call_id()) {
                        pending_approvals.push(PendingApprovalCall {
                            request: request.clone(),
                        });
                        let call = request
                            .authorize_verified(approval)
                            .map_err(execution_authorization_error)?;
                        requests.push(IndexedAuthorizedCall { ordinal, call });
                    } else {
                        pending_approvals.push(PendingApprovalCall { request });
                    }
                }
            }
        }
        if let Some((call_id, _)) = verified_approvals.into_iter().next() {
            let _ = call_id;
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "approval was supplied for a non-pending tool call",
            )
            .into());
        }

        Ok(PendingToolPhase {
            response: step.response().clone(),
            prepared: step.prepared().to_vec(),
            requests,
            results,
            pending_approvals,
        })
    }

    fn restore_request(
        &self,
        prepared: &PreparedToolSnapshot,
    ) -> Result<ToolExecutionRequest, EngineResumeError> {
        let call_id = prepared.call().id().to_owned();
        let mut request = self
            .tools
            .resolve_frozen(prepared.call().clone(), prepared.binding())
            .map_err(|_| EngineResumeError::FrozenRequestMismatch {
                call_id: call_id.clone(),
            })?;
        while request.attempt() < prepared.attempt() {
            request = request
                .next_attempt(EffectCertainty::Indeterminate)
                .map_err(|_| EngineResumeError::RecoveryNotPermitted {
                    call_id: call_id.clone(),
                })?;
        }
        if request.attempt() != prepared.attempt()
            || request.recovery_policy() != prepared.recovery_policy()
            || request.idempotency_key() != prepared.stable_idempotency_key()
        {
            return Err(EngineResumeError::FrozenRequestMismatch { call_id });
        }
        request.validate().map_err(|error| {
            EngineResumeError::Runtime(
                Error::new(
                    ErrorKind::InvalidInput,
                    "restored frozen tool request is no longer valid",
                )
                .with_source(error),
            )
        })?;
        Ok(request)
    }

    fn queue_handshake_failure(&mut self, error: Error) {
        match error.kind() {
            ErrorKind::Cancelled => self.queue_terminal(RunTerminal::Cancelled {
                reason: "tool loop cancelled during model handshake".to_string(),
                partial: None,
                report: Box::new(self.report.clone()),
            }),
            ErrorKind::Timeout => {
                let kind = if Instant::now() >= self.total_deadline {
                    RunTimeoutKind::Total
                } else {
                    RunTimeoutKind::ModelStep
                };
                self.queue_terminal(RunTerminal::TimedOut {
                    kind,
                    partial: None,
                    report: Box::new(self.report.clone()),
                });
            }
            _ => self.queue_terminal(RunTerminal::Failed {
                error,
                partial: None,
                report: Box::new(self.report.clone()),
            }),
        }
    }

    async fn poll_model_stream<P>(
        &mut self,
        mut state: StepStream,
        checkpoint: &mut P,
    ) -> Result<(), P::Error>
    where
        P: EngineCheckpointPort,
    {
        match state.stream.next().await {
            None => {
                if let Err(error) = self.settle_model_usage(state.usage, None) {
                    self.queue_terminal(RunTerminal::BudgetExceeded {
                        error,
                        report: Box::new(self.report.clone()),
                    });
                } else {
                    self.queue_terminal(RunTerminal::Failed {
                        error: Error::unexpected_eof(),
                        partial: None,
                        report: Box::new(self.report.clone()),
                    });
                }
            }
            Some(LanguageStreamEvent::Terminal(terminal)) => {
                self.consume_model_terminal(terminal, state, checkpoint)
                    .await?;
            }
            Some(event) => {
                match &event {
                    LanguageStreamEvent::ProviderDeferred { id, state: item } => {
                        state.provider_states.push((id.clone(), item.clone()));
                        self.report.provider_deferred_mut().push(item.clone());
                    }
                    LanguageStreamEvent::ToolResult(result) => {
                        state.provider_result_ids.insert(result.call_id.clone());
                    }
                    LanguageStreamEvent::Usage(update) => state.usage.observe(update),
                    _ => {}
                }
                self.phase = EnginePhase::Streaming(Box::new(state));
                self.pending.push_back(RunEvent::Model {
                    step: self.step,
                    event,
                });
            }
        }
        Ok(())
    }

    async fn consume_model_terminal<P>(
        &mut self,
        terminal: StreamTerminal,
        state: StepStream,
        checkpoint: &mut P,
    ) -> Result<(), P::Error>
    where
        P: EngineCheckpointPort,
    {
        match terminal {
            StreamTerminal::Completed { response } => {
                self.prepare_completed_response(*response, state, checkpoint)
                    .await?;
            }
            StreamTerminal::Failed { error, partial } => {
                if let Err(error) = self.settle_model_usage(
                    state.usage,
                    partial.as_ref().map(PartialLanguageOutput::usage),
                ) {
                    self.queue_terminal(RunTerminal::BudgetExceeded {
                        error,
                        report: Box::new(self.report.clone()),
                    });
                    return Ok(());
                }
                if error.kind() == ErrorKind::Timeout
                    && let Some(kind) = state.timeout.take()
                {
                    self.queue_terminal(RunTerminal::TimedOut {
                        kind,
                        partial,
                        report: Box::new(self.report.clone()),
                    });
                } else {
                    self.queue_terminal(RunTerminal::Failed {
                        error,
                        partial,
                        report: Box::new(self.report.clone()),
                    });
                }
            }
            StreamTerminal::Cancelled { reason, partial } => {
                if let Err(error) = self.settle_model_usage(
                    state.usage,
                    partial.as_ref().map(PartialLanguageOutput::usage),
                ) {
                    self.queue_terminal(RunTerminal::BudgetExceeded {
                        error,
                        report: Box::new(self.report.clone()),
                    });
                    return Ok(());
                }
                self.queue_terminal(RunTerminal::Cancelled {
                    reason,
                    partial,
                    report: Box::new(self.report.clone()),
                });
            }
            _ => {
                if let Err(error) = self.settle_model_usage(state.usage, None) {
                    self.queue_terminal(RunTerminal::BudgetExceeded {
                        error,
                        report: Box::new(self.report.clone()),
                    });
                } else {
                    self.queue_terminal(RunTerminal::Failed {
                        error: Error::protocol_violation("unsupported model stream terminal"),
                        partial: None,
                        report: Box::new(self.report.clone()),
                    });
                }
            }
        }
        Ok(())
    }

    async fn prepare_completed_response<P>(
        &mut self,
        response: LanguageResponse,
        state: StepStream,
        checkpoint: &mut P,
    ) -> Result<(), P::Error>
    where
        P: EngineCheckpointPort,
    {
        let mut state = state;
        let reconciler = std::mem::take(&mut state.usage);
        if let Err(error) = self.settle_model_usage(reconciler, Some(response.usage())) {
            self.append_assistant_message(&response);
            self.finish_step(response, Vec::new());
            self.queue_terminal(RunTerminal::BudgetExceeded {
                error,
                report: Box::new(self.report.clone()),
            });
            return Ok(());
        }
        self.append_assistant_message(&response);

        if self.tool_handling == ToolHandling::ObserveOnly {
            self.finish_step(response, Vec::new());
            self.queue_terminal(RunTerminal::Completed {
                report: Box::new(self.report.clone()),
            });
            return Ok(());
        }

        let mut provider_result_ids = state.provider_result_ids;
        provider_result_ids.extend(response.content().iter().filter_map(|part| match part {
            ContentPart::ToolResult(result) => Some(result.call_id.clone()),
            _ => None,
        }));
        let calls = response
            .content()
            .iter()
            .filter_map(|part| match part {
                ContentPart::ToolCall(call) => Some(call.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();

        // Provider-owned work is an orchestration boundary. Inspect the whole
        // step before resolving any local name so a provider/local collision or
        // an unrelated invalid local call cannot cross that boundary.
        let provider_states =
            normalize_provider_states(&state.provider_states, &provider_result_ids).unwrap_or_else(
                |error| {
                    self.queue_terminal(RunTerminal::Failed {
                        error,
                        partial: None,
                        report: Box::new(self.report.clone()),
                    });
                    Vec::new()
                },
            );
        if matches!(self.phase, EnginePhase::Terminal) {
            return Ok(());
        }
        if !provider_states.is_empty() {
            let state_ids = provider_states
                .iter()
                .filter_map(|state| state.correlation_id.clone())
                .collect::<Vec<_>>();
            let pending = PendingProviderStepSnapshot::new(
                self.step,
                self.target.clone(),
                response.clone(),
                provider_states,
            );
            let control = self
                .checkpoint_state(
                    checkpoint,
                    CheckpointBoundary::Quiescent,
                    EngineCheckpointState::AwaitingProvider(pending),
                )
                .await?;
            if control == CheckpointControl::Pause {
                self.phase = EnginePhase::Paused;
                return Ok(());
            }
            self.finish_step(response, Vec::new());
            self.queue_terminal(RunTerminal::Suspended {
                reason: SuspensionReason::AwaitingProvider { state_ids },
                report: Box::new(self.report.clone()),
            });
            return Ok(());
        }

        let mut prepared_tools = Vec::new();
        let mut requests = Vec::new();
        let mut pending_approvals = Vec::new();
        let mut immediate_results = Vec::new();

        for (ordinal, call) in calls.into_iter().enumerate() {
            let request = match self.tools.resolve(call.clone()) {
                Ok(request) => request,
                Err(error) => {
                    self.finish_preparation_failure(response, ordinal, call, error);
                    return Ok(());
                }
            };
            if let Err(error) = request.validate() {
                self.finish_preparation_failure(response, ordinal, call, error);
                return Ok(());
            }
            let argument_bytes = match serde_json::to_vec(request.arguments()) {
                Ok(arguments) => arguments.len(),
                Err(error) => {
                    self.finish_step(response, Vec::new());
                    self.queue_terminal(RunTerminal::Failed {
                        error: Error::new(
                            ErrorKind::Internal,
                            "tool arguments could not be serialized for budget accounting",
                        )
                        .with_source(error),
                        partial: None,
                        report: Box::new(self.report.clone()),
                    });
                    return Ok(());
                }
            };
            if let Err(error) = self
                .report
                .budget_mut()
                .charge_tool_call(argument_bytes, &self.budget)
            {
                self.finish_step(response, Vec::new());
                self.queue_terminal(RunTerminal::BudgetExceeded {
                    error,
                    report: Box::new(self.report.clone()),
                });
                return Ok(());
            }
            let prepared = match self.log_prepared(ordinal, &request) {
                Ok(prepared) => prepared,
                Err(error) => {
                    self.finish_step(response, Vec::new());
                    self.queue_terminal(RunTerminal::Failed {
                        error,
                        partial: None,
                        report: Box::new(self.report.clone()),
                    });
                    return Ok(());
                }
            };
            prepared_tools.push(prepared);
            self.pending.push_back(RunEvent::ToolPrepared {
                step: self.step,
                ordinal,
                call: call.clone(),
            });
            let authorized = match request.approval_policy() {
                ApprovalPolicy::NotRequired => match request.authorize_not_required() {
                    Ok(authorized) => Some(authorized),
                    Err(error) => {
                        self.finish_step(
                            response,
                            immediate_results
                                .into_iter()
                                .map(|result: IndexedResult| result.result)
                                .collect(),
                        );
                        self.queue_terminal(RunTerminal::Failed {
                            error: execution_authorization_error(error),
                            partial: None,
                            report: Box::new(self.report.clone()),
                        });
                        return Ok(());
                    }
                },
                ApprovalPolicy::Required => {
                    let approval_request = ApprovalRequest::from_frozen(
                        self.step,
                        ordinal,
                        self.target.clone(),
                        &request,
                    );
                    let decision = wait_for(
                        self.approval_decider.decide(&approval_request),
                        &self.cancellation,
                        self.total_deadline,
                        RunTimeoutKind::Total,
                    )
                    .await;
                    match decision {
                        WaitResult::Ready(Ok(ApprovalDecision::Approve)) => {
                            Some(request.authorize_host_auto_approved())
                        }
                        WaitResult::Ready(Ok(ApprovalDecision::Deny(denial))) => {
                            let result = ToolResult {
                                call_id: request.call_id().to_string(),
                                name: request.name().to_string(),
                                outcome: ToolOutcome::Denied {
                                    reason: denial.reason().to_owned(),
                                },
                            };
                            if let Err(error) = self.record_non_dispatch_result(&request, &result) {
                                self.finish_step(
                                    response,
                                    immediate_results
                                        .into_iter()
                                        .map(|result: IndexedResult| result.result)
                                        .collect(),
                                );
                                match error {
                                    NonDispatchResultError::Budget(error) => {
                                        self.queue_terminal(RunTerminal::BudgetExceeded {
                                            error,
                                            report: Box::new(self.report.clone()),
                                        });
                                    }
                                    NonDispatchResultError::Failed(error) => {
                                        self.queue_terminal(RunTerminal::Failed {
                                            error,
                                            partial: None,
                                            report: Box::new(self.report.clone()),
                                        });
                                    }
                                }
                                return Ok(());
                            }
                            self.pending.push_back(RunEvent::ToolCompleted {
                                step: self.step,
                                ordinal,
                                result: result.clone(),
                            });
                            let stop = self.outcome_policy.action(&result.outcome)
                                == ToolOutcomeAction::Stop;
                            immediate_results.push(IndexedResult {
                                ordinal,
                                result: result.clone(),
                            });
                            if stop {
                                let tool_results = immediate_results
                                    .into_iter()
                                    .map(|result| result.result)
                                    .collect();
                                self.finish_step(response, tool_results);
                                self.queue_terminal(RunTerminal::Stopped {
                                    reason: RunStopReason::ToolOutcome {
                                        call_id: result.call_id,
                                        outcome: result.outcome,
                                    },
                                    report: Box::new(self.report.clone()),
                                });
                                return Ok(());
                            }
                            None
                        }
                        WaitResult::Ready(Ok(ApprovalDecision::AwaitExternal)) => {
                            if let Err(error) = self
                                .report
                                .budget_mut()
                                .reserve_pending_approval(&self.budget)
                            {
                                self.finish_step(
                                    response,
                                    immediate_results
                                        .into_iter()
                                        .map(|result: IndexedResult| result.result)
                                        .collect(),
                                );
                                self.queue_terminal(RunTerminal::BudgetExceeded {
                                    error,
                                    report: Box::new(self.report.clone()),
                                });
                                return Ok(());
                            }
                            pending_approvals.push(PendingApprovalCall { request });
                            None
                        }
                        WaitResult::Ready(Err(error)) => {
                            self.finish_step(
                                response,
                                immediate_results
                                    .into_iter()
                                    .map(|result: IndexedResult| result.result)
                                    .collect(),
                            );
                            self.queue_terminal(RunTerminal::Failed {
                                error: execution_approval_decision_error(error),
                                partial: None,
                                report: Box::new(self.report.clone()),
                            });
                            return Ok(());
                        }
                        WaitResult::TimedOut(kind) => {
                            self.finish_step(
                                response,
                                immediate_results
                                    .into_iter()
                                    .map(|result: IndexedResult| result.result)
                                    .collect(),
                            );
                            self.queue_terminal(RunTerminal::TimedOut {
                                kind,
                                partial: None,
                                report: Box::new(self.report.clone()),
                            });
                            return Ok(());
                        }
                        WaitResult::Cancelled => {
                            self.finish_step(
                                response,
                                immediate_results
                                    .into_iter()
                                    .map(|result: IndexedResult| result.result)
                                    .collect(),
                            );
                            self.queue_terminal(RunTerminal::Cancelled {
                                reason: "tool loop cancelled during approval decision".to_string(),
                                partial: None,
                                report: Box::new(self.report.clone()),
                            });
                            return Ok(());
                        }
                    }
                }
            };
            if let Some(call) = authorized {
                requests.push(IndexedAuthorizedCall { ordinal, call });
            }
        }

        if requests.is_empty() && pending_approvals.is_empty() {
            let continued = !immediate_results.is_empty();
            self.finish_step(
                response,
                immediate_results
                    .into_iter()
                    .map(|result| result.result)
                    .collect(),
            );
            if continued {
                self.phase = EnginePhase::ReadyForModel;
            } else {
                self.queue_terminal(RunTerminal::Completed {
                    report: Box::new(self.report.clone()),
                });
            }
        } else {
            let control = self
                .checkpoint_pending_tools(
                    checkpoint,
                    CheckpointBoundary::Quiescent,
                    &response,
                    &prepared_tools,
                    &immediate_results,
                    &pending_approvals,
                    !requests.is_empty(),
                )
                .await?;
            if control == CheckpointControl::Pause {
                self.phase = EnginePhase::Paused;
                return Ok(());
            }
            if requests.is_empty() {
                let call_ids = pending_approvals
                    .iter()
                    .map(|approval| approval.request.call_id().to_string())
                    .collect();
                self.finish_step(
                    response,
                    immediate_results
                        .into_iter()
                        .map(|result| result.result)
                        .collect(),
                );
                self.queue_terminal(RunTerminal::Suspended {
                    reason: SuspensionReason::AwaitingApproval { call_ids },
                    report: Box::new(self.report.clone()),
                });
                return Ok(());
            }
            self.phase = EnginePhase::Tools(Box::new(PendingToolPhase {
                response,
                prepared: prepared_tools,
                requests,
                results: immediate_results,
                pending_approvals,
            }));
        }
        Ok(())
    }

    fn finish_preparation_failure(
        &mut self,
        response: LanguageResponse,
        ordinal: usize,
        call: ToolCall,
        error: ToolExecutionError,
    ) {
        let outcome = execution_error_outcome(error);
        let result = ToolResult {
            call_id: call.id().to_owned(),
            name: call.name().to_owned(),
            outcome: outcome.clone(),
        };
        self.pending.push_back(RunEvent::ToolCompleted {
            step: self.step,
            ordinal,
            result: result.clone(),
        });
        self.finish_step(response, vec![result]);
        if self.outcome_policy.action(&outcome) == ToolOutcomeAction::Continue {
            self.phase = EnginePhase::ReadyForModel;
        } else {
            self.queue_terminal(RunTerminal::Stopped {
                reason: RunStopReason::ToolOutcome {
                    call_id: call.id().to_owned(),
                    outcome,
                },
                report: Box::new(self.report.clone()),
            });
        }
    }

    async fn execute_prepared_step<P>(
        &mut self,
        prepared: PendingToolPhase,
        checkpoint: &mut P,
    ) -> Result<(), P::Error>
    where
        P: EngineCheckpointPort,
    {
        let PendingToolPhase {
            response,
            prepared,
            requests,
            mut results,
            mut pending_approvals,
        } = prepared;
        let stop = self
            .execute_requests(
                &response,
                &prepared,
                requests,
                &mut results,
                &mut pending_approvals,
                checkpoint,
            )
            .await?;
        results.sort_unstable_by_key(|result| result.ordinal);
        for result in &results {
            self.pending.push_back(RunEvent::ToolCompleted {
                step: self.step,
                ordinal: result.ordinal,
                result: result.result.clone(),
            });
        }
        if stop.is_none() && !pending_approvals.is_empty() {
            let control = self
                .checkpoint_pending_tools(
                    checkpoint,
                    CheckpointBoundary::Quiescent,
                    &response,
                    &prepared,
                    &results,
                    &pending_approvals,
                    false,
                )
                .await?;
            if control == CheckpointControl::Pause {
                self.phase = EnginePhase::Paused;
                return Ok(());
            }
            let call_ids = pending_approvals
                .iter()
                .map(|approval| approval.request.call_id().to_string())
                .collect();
            self.finish_step(
                response,
                results.into_iter().map(|result| result.result).collect(),
            );
            self.queue_terminal(RunTerminal::Suspended {
                reason: SuspensionReason::AwaitingApproval { call_ids },
                report: Box::new(self.report.clone()),
            });
            return Ok(());
        }

        if stop.is_some() {
            self.release_pending_approvals(&mut pending_approvals);
        }
        let tool_results = results
            .iter()
            .map(|result| result.result.clone())
            .collect::<Vec<_>>();
        self.finish_step(response, tool_results);

        match stop {
            None => self.phase = EnginePhase::ReadyForModel,
            Some(ExecutionStop::Policy { call_id, outcome }) => {
                self.queue_terminal(RunTerminal::Stopped {
                    reason: RunStopReason::ToolOutcome { call_id, outcome },
                    report: Box::new(self.report.clone()),
                });
            }
            Some(ExecutionStop::Budget(error)) => {
                self.queue_terminal(RunTerminal::BudgetExceeded {
                    error,
                    report: Box::new(self.report.clone()),
                });
            }
            Some(ExecutionStop::TimedOut(kind)) => {
                self.queue_terminal(RunTerminal::TimedOut {
                    kind,
                    partial: None,
                    report: Box::new(self.report.clone()),
                });
            }
            Some(ExecutionStop::Cancelled) => {
                self.queue_terminal(RunTerminal::Cancelled {
                    reason: "tool loop cancelled during tool execution".to_string(),
                    partial: None,
                    report: Box::new(self.report.clone()),
                });
            }
            Some(ExecutionStop::Indeterminate(effect)) => {
                self.queue_terminal(RunTerminal::Indeterminate {
                    effect,
                    report: Box::new(self.report.clone()),
                });
            }
            Some(ExecutionStop::Failed(error)) => {
                self.queue_terminal(RunTerminal::Failed {
                    error,
                    partial: None,
                    report: Box::new(self.report.clone()),
                });
            }
        }
        Ok(())
    }

    async fn execute_requests<P>(
        &mut self,
        response: &LanguageResponse,
        prepared: &[PreparedToolSnapshot],
        requests: Vec<IndexedAuthorizedCall>,
        results: &mut Vec<IndexedResult>,
        pending_approvals: &mut Vec<PendingApprovalCall>,
        checkpoint: &mut P,
    ) -> Result<Option<ExecutionStop>, P::Error>
    where
        P: EngineCheckpointPort,
    {
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
                    .execute_parallel_batch(
                        response,
                        prepared,
                        batch,
                        results,
                        pending_approvals,
                        checkpoint,
                    )
                    .await?
                {
                    return Ok(Some(stop));
                }
            } else if let Some(request) = queue.pop_front() {
                if let Some(stop) = self.pre_dispatch_stop() {
                    return Ok(Some(stop));
                }
                self.release_dispatch_approval(request.call.request(), pending_approvals);
                if let Err(error) = self.log_dispatched(request.call.request()) {
                    return Ok(Some(ExecutionStop::Failed(error)));
                }
                self.checkpoint_pending_tools(
                    checkpoint,
                    CheckpointBoundary::BeforeDispatch,
                    response,
                    prepared,
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
                    self.checkpoint_pending_tools(
                        checkpoint,
                        CheckpointBoundary::AfterDispatch,
                        response,
                        prepared,
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

    async fn execute_parallel_batch<P>(
        &mut self,
        response: &LanguageResponse,
        prepared: &[PreparedToolSnapshot],
        requests: Vec<IndexedAuthorizedCall>,
        results: &mut Vec<IndexedResult>,
        pending_approvals: &mut Vec<PendingApprovalCall>,
        checkpoint: &mut P,
    ) -> Result<Option<ExecutionStop>, P::Error>
    where
        P: EngineCheckpointPort,
    {
        let mut pending = VecDeque::from(requests);
        let mut active = FuturesUnordered::<BoxToolFuture>::new();
        let mut active_by_binding = BTreeMap::<String, usize>::new();
        let mut active_calls = BTreeMap::new();
        let mut policy_stop: Option<(usize, String, ToolOutcome)> = None;
        let max_concurrent = checkpoint.max_concurrent_tools(self.budget.max_concurrent_tools());

        loop {
            while policy_stop.is_none() && active.len() < max_concurrent {
                if let Some(stop) = self.pre_dispatch_stop() {
                    if let Err(error) = self.mark_active_indeterminate(&active_calls) {
                        return Ok(Some(ExecutionStop::Failed(error)));
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
                if let Err(error) = self.log_dispatched(request.call.request()) {
                    if let Err(logging_error) = self.mark_active_indeterminate(&active_calls) {
                        return Ok(Some(ExecutionStop::Failed(logging_error)));
                    }
                    return Ok(Some(ExecutionStop::Failed(error)));
                }
                self.checkpoint_pending_tools(
                    checkpoint,
                    CheckpointBoundary::BeforeDispatch,
                    response,
                    prepared,
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
                let tool_timeout = self.budget.timeouts().tool();
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
                self.checkpoint_pending_tools(
                    checkpoint,
                    CheckpointBoundary::AfterDispatch,
                    response,
                    prepared,
                    results,
                    pending_approvals,
                    true,
                )
                .await?;
            }
            if let Some(stop) = progress.stop {
                match stop {
                    ExecutionStop::Policy { call_id, outcome } => {
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
                            return Ok(Some(ExecutionStop::Failed(error)));
                        }
                        return Ok(Some(terminal));
                    }
                }
            }
        }

        let stop =
            policy_stop.map(|(_, call_id, outcome)| ExecutionStop::Policy { call_id, outcome });
        Ok(stop)
    }

    async fn await_tool(&mut self, request: IndexedAuthorizedCall) -> ToolAttempt {
        await_tool_attempt(
            request,
            self.cancellation.clone(),
            self.total_deadline,
            self.budget.timeouts().tool(),
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
                    if let Err(error) = self.log_indeterminate(
                        &call_id,
                        execution_attempt,
                        IndeterminateReason::DispatchOutcomeUnknown,
                    ) {
                        return ExecutionProgress::stopped(ExecutionStop::Failed(error));
                    }
                    return ExecutionProgress::stopped(ExecutionStop::Indeterminate(
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
                if let Err(error) = self.log_indeterminate(
                    &call_id,
                    execution_attempt,
                    IndeterminateReason::DispatchOutcomeUnknown,
                ) {
                    return ExecutionProgress::stopped(ExecutionStop::Failed(error));
                }
                if effect == ToolEffect::SideEffecting {
                    return ExecutionProgress::stopped(ExecutionStop::Indeterminate(
                        IndeterminateEffect {
                            call_id,
                            binding_fingerprint,
                            reason: "tool execution timed out after dispatch".to_string(),
                        },
                    ));
                }
                return ExecutionProgress::stopped(ExecutionStop::TimedOut(kind));
            }
            ToolAttemptOutcome::Cancelled => {
                if let Err(error) = self.log_indeterminate(
                    &call_id,
                    execution_attempt,
                    IndeterminateReason::CancellationAfterDispatch,
                ) {
                    return ExecutionProgress::stopped(ExecutionStop::Failed(error));
                }
                if effect == ToolEffect::SideEffecting {
                    return ExecutionProgress::stopped(ExecutionStop::Indeterminate(
                        IndeterminateEffect {
                            call_id,
                            binding_fingerprint,
                            reason: "tool execution was cancelled after dispatch".to_string(),
                        },
                    ));
                }
                return ExecutionProgress::stopped(ExecutionStop::Cancelled);
            }
        };

        let result_bytes = match serde_json::to_vec(&result) {
            Ok(value) => value.len(),
            Err(error) => {
                if let Err(logging_error) = self.log_indeterminate(
                    &call_id,
                    execution_attempt,
                    IndeterminateReason::CheckpointFailure,
                ) {
                    return ExecutionProgress::stopped(ExecutionStop::Failed(logging_error));
                }
                if effect == ToolEffect::SideEffecting {
                    return ExecutionProgress::stopped(ExecutionStop::Indeterminate(
                        IndeterminateEffect {
                            call_id,
                            binding_fingerprint,
                            reason: "tool result could not be checkpointed".to_string(),
                        },
                    ));
                }
                return ExecutionProgress::stopped(ExecutionStop::Failed(
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
            .charge_tool_result(result_bytes, &self.budget)
        {
            if let Err(logging_error) = self.log_indeterminate(
                &call_id,
                execution_attempt,
                IndeterminateReason::CheckpointFailure,
            ) {
                return ExecutionProgress::stopped(ExecutionStop::Failed(logging_error));
            }
            if effect == ToolEffect::SideEffecting {
                return ExecutionProgress::stopped(ExecutionStop::Indeterminate(
                    IndeterminateEffect {
                        call_id,
                        binding_fingerprint,
                        reason: "tool result exceeded checkpoint budget".to_string(),
                    },
                ));
            }
            return ExecutionProgress::stopped(ExecutionStop::Budget(error));
        }
        if let Err(error) = self.log_completed(&attempt.request, &result) {
            if let Err(logging_error) = self.log_indeterminate(
                &call_id,
                execution_attempt,
                IndeterminateReason::CheckpointFailure,
            ) {
                return ExecutionProgress::stopped(ExecutionStop::Failed(logging_error));
            }
            if effect == ToolEffect::SideEffecting {
                return ExecutionProgress::stopped(ExecutionStop::Indeterminate(
                    IndeterminateEffect {
                        call_id,
                        binding_fingerprint,
                        reason: "tool completion could not be checkpointed".to_string(),
                    },
                ));
            }
            return ExecutionProgress::stopped(ExecutionStop::Failed(error));
        }

        let stop =
            (self.outcome_policy.action(&result.outcome) == ToolOutcomeAction::Stop).then(|| {
                ExecutionStop::Policy {
                    call_id: result.call_id.clone(),
                    outcome: result.outcome.clone(),
                }
            });
        ExecutionProgress {
            results: vec![IndexedResult { ordinal, result }],
            stop,
        }
    }

    fn append_assistant_message(&mut self, response: &LanguageResponse) {
        if let Some(message) = response.project_assistant_history().into_message() {
            self.report.messages_mut().push(message);
        }
    }

    fn settle_model_usage(
        &mut self,
        reconciler: CallUsageReconciler,
        terminal: Option<&Usage>,
    ) -> Result<(), crate::BudgetError> {
        let usage = reconciler.settle(terminal);
        self.report.accumulate_usage(&usage);
        self.report.budget_mut().charge_usage(&usage, &self.budget)
    }

    fn finish_step(&mut self, response: LanguageResponse, results: Vec<ToolResult>) {
        if !results.is_empty() {
            self.report.messages_mut().push(Message::new(
                MessageRole::Tool,
                results
                    .iter()
                    .cloned()
                    .map(ContentPart::ToolResult)
                    .map(MessagePart::from),
            ));
        }
        let record = StepRecord::new(self.step, self.target.clone(), response, results);
        self.report.steps_mut().push(record.clone());
        self.pending.push_back(RunEvent::StepFinished {
            record: Box::new(record),
        });
    }

    fn log_prepared(
        &mut self,
        ordinal: usize,
        request: &ToolExecutionRequest,
    ) -> Result<PreparedToolSnapshot, Error> {
        let ordinal = u32::try_from(ordinal).map_err(|error| {
            Error::new(
                ErrorKind::LimitExceeded,
                "tool ordinal cannot be represented in a durable execution log",
            )
            .with_source(error)
        })?;
        let prepared = PreparedToolSnapshot::new(
            ordinal,
            request.call().clone(),
            request.binding_identity().clone(),
            request.recovery_policy(),
            request.idempotency_key().cloned(),
            request.attempt(),
        );
        let sequence = self.report.execution_log().next_sequence();
        self.report
            .execution_log_mut()
            .append(ToolExecutionEvent::prepared(
                sequence,
                unix_millis(),
                self.step,
                prepared.clone(),
            ))
            .map_err(execution_log_error)?;
        Ok(prepared)
    }

    fn log_dispatched(&mut self, request: &ToolExecutionRequest) -> Result<(), Error> {
        let sequence = self.report.execution_log().next_sequence();
        self.report
            .execution_log_mut()
            .append(ToolExecutionEvent::dispatched(
                sequence,
                unix_millis(),
                request.call_id(),
                request.attempt(),
                None,
            ))
            .map_err(execution_log_error)
    }

    fn log_completed(
        &mut self,
        request: &ToolExecutionRequest,
        result: &ToolResult,
    ) -> Result<(), Error> {
        let sequence = self.report.execution_log().next_sequence();
        self.report
            .execution_log_mut()
            .append(ToolExecutionEvent::completed(
                sequence,
                unix_millis(),
                &result.call_id,
                request.attempt(),
                result.outcome.clone(),
            ))
            .map_err(execution_log_error)
    }

    fn record_non_dispatch_result(
        &mut self,
        request: &ToolExecutionRequest,
        result: &ToolResult,
    ) -> Result<(), NonDispatchResultError> {
        let result_bytes = serde_json::to_vec(result)
            .map_err(|error| {
                NonDispatchResultError::Failed(
                    Error::new(
                        ErrorKind::Internal,
                        "tool result could not be serialized for budget accounting",
                    )
                    .with_source(error),
                )
            })?
            .len();
        self.report
            .budget_mut()
            .charge_tool_result(result_bytes, &self.budget)
            .map_err(NonDispatchResultError::Budget)?;
        self.log_completed(request, result)
            .map_err(NonDispatchResultError::Failed)
    }

    fn log_indeterminate(
        &mut self,
        call_id: &str,
        attempt: crate::tool::ToolExecutionAttempt,
        reason: IndeterminateReason,
    ) -> Result<(), Error> {
        let sequence = self.report.execution_log().next_sequence();
        self.report
            .execution_log_mut()
            .append(ToolExecutionEvent::indeterminate(
                sequence,
                unix_millis(),
                call_id,
                attempt,
                reason,
            ))
            .map_err(execution_log_error)
    }

    fn mark_active_indeterminate(
        &mut self,
        active: &BTreeMap<usize, (String, crate::tool::ToolExecutionAttempt)>,
    ) -> Result<(), Error> {
        for (call_id, attempt) in active.values() {
            self.log_indeterminate(
                call_id,
                *attempt,
                IndeterminateReason::DispatchOutcomeUnknown,
            )?;
        }
        Ok(())
    }

    fn pre_dispatch_stop(&self) -> Option<ExecutionStop> {
        if self.cancellation.is_cancelled() {
            Some(ExecutionStop::Cancelled)
        } else if Instant::now() >= self.total_deadline {
            Some(ExecutionStop::TimedOut(RunTimeoutKind::Total))
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

    fn queue_terminal(&mut self, mut terminal: RunTerminal) {
        if matches!(self.phase, EnginePhase::Terminal) {
            return;
        }
        while self.report.budget().pending_approvals() > 0 {
            self.report.budget_mut().release_pending_approval();
        }
        let current_report = Box::new(self.report.clone());
        match &mut terminal {
            RunTerminal::Completed { report }
            | RunTerminal::Stopped { report, .. }
            | RunTerminal::Suspended { report, .. }
            | RunTerminal::BudgetExceeded { report, .. }
            | RunTerminal::TimedOut { report, .. }
            | RunTerminal::Indeterminate { report, .. }
            | RunTerminal::HistoryProjectionRejected { report, .. }
            | RunTerminal::Failed { report, .. }
            | RunTerminal::Cancelled { report, .. } => *report = current_report,
            RunTerminal::ResumeConflict { .. } => {}
        }
        self.phase = EnginePhase::Terminal;
        self.pending.push_back(RunEvent::Terminal(terminal));
    }
}

struct StepStream {
    stream: LanguageStream,
    provider_states: Vec<(String, siumai_core::OpaqueProviderItem)>,
    provider_result_ids: BTreeSet<String>,
    usage: CallUsageReconciler,
    timeout: Arc<StreamTimeoutState>,
}

#[derive(Default)]
struct StreamTimeoutState(AtomicU8);

impl StreamTimeoutState {
    fn record(&self, kind: RunTimeoutKind) {
        self.0.store(timeout_kind_code(kind), Ordering::Release);
    }

    fn take(&self) -> Option<RunTimeoutKind> {
        decode_timeout_kind(self.0.swap(0, Ordering::AcqRel))
    }
}

struct PendingToolPhase {
    response: LanguageResponse,
    prepared: Vec<PreparedToolSnapshot>,
    requests: Vec<IndexedAuthorizedCall>,
    results: Vec<IndexedResult>,
    pending_approvals: Vec<PendingApprovalCall>,
}

struct PendingApprovalCall {
    request: ToolExecutionRequest,
}

struct IndexedAuthorizedCall {
    ordinal: usize,
    call: AuthorizedToolCall,
}

#[derive(Clone)]
struct IndexedResult {
    ordinal: usize,
    result: ToolResult,
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
    stop: Option<ExecutionStop>,
}

impl ExecutionProgress {
    fn stopped(stop: ExecutionStop) -> Self {
        Self {
            results: Vec::new(),
            stop: Some(stop),
        }
    }
}

enum ExecutionStop {
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

enum NonDispatchResultError {
    Budget(crate::BudgetError),
    Failed(Error),
}

enum WaitResult<T> {
    Ready(T),
    TimedOut(RunTimeoutKind),
    Cancelled,
}

async fn wait_for<F, T>(
    future: F,
    cancellation: &Cancellation,
    deadline: Instant,
    timeout_kind: RunTimeoutKind,
) -> WaitResult<T>
where
    F: Future<Output = T>,
{
    tokio::select! {
        biased;
        _ = cancellation.cancelled() => WaitResult::Cancelled,
        _ = tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)) => {
            WaitResult::TimedOut(timeout_kind)
        }
        output = future => WaitResult::Ready(output),
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

fn visible_catalog_error(error: VisibleToolCatalogError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "model-visible tool catalog is incompatible with trusted local bindings",
    )
    .with_source(error)
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

fn execution_error_outcome(error: ToolExecutionError) -> ToolOutcome {
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

fn terminal_checkpoint(terminal: &RunTerminal) -> SnapshotTerminal {
    let reason = |code| SnapshotReason::runtime_code(code);
    match terminal {
        RunTerminal::Completed { .. } => SnapshotTerminal::Completed { reason: None },
        RunTerminal::Stopped { .. } => SnapshotTerminal::Completed {
            reason: Some(reason("runtime_stopped")),
        },
        RunTerminal::Suspended { .. } => SnapshotTerminal::Failed {
            reason: reason("unexpected_runtime_suspension"),
            partial: None,
        },
        RunTerminal::BudgetExceeded { .. } => SnapshotTerminal::Exhausted {
            reason: reason("runtime_budget_exhausted"),
            partial: None,
        },
        RunTerminal::TimedOut { partial, .. } => SnapshotTerminal::Exhausted {
            reason: reason("runtime_timed_out"),
            partial: partial.clone(),
        },
        RunTerminal::Indeterminate { .. } => SnapshotTerminal::Indeterminate {
            reason: reason("runtime_indeterminate"),
        },
        RunTerminal::HistoryProjectionRejected { .. } => SnapshotTerminal::Failed {
            reason: reason("history_projection_rejected"),
            partial: None,
        },
        RunTerminal::ResumeConflict { .. } => SnapshotTerminal::Failed {
            reason: reason("runtime_resume_conflict"),
            partial: None,
        },
        RunTerminal::Failed { partial, .. } => SnapshotTerminal::Failed {
            reason: reason("runtime_failed"),
            partial: partial.clone(),
        },
        RunTerminal::Cancelled { partial, .. } => SnapshotTerminal::Cancelled {
            reason: reason("runtime_cancelled"),
            partial: partial.clone(),
        },
    }
}

fn provider_state_from_deferred(
    id: &str,
    item: &siumai_core::OpaqueProviderItem,
) -> Result<ProviderStateSnapshot, Error> {
    let payload = serde_json::to_vec(item).map_err(|error| {
        Error::new(
            ErrorKind::Internal,
            "provider-deferred state could not be serialized for checkpointing",
        )
        .with_source(error)
    })?;
    Ok(ProviderStateSnapshot {
        namespace: format!(
            "provider-deferred:{}:{}:{id}",
            item.provenance().provider(),
            item.provenance().protocol()
        ),
        correlation_id: Some(id.to_string()),
        encoding: "siumai.opaque-provider-item+json".to_string(),
        payload,
    })
}

fn normalize_provider_states(
    observations: &[(String, siumai_core::OpaqueProviderItem)],
    completed_ids: &BTreeSet<String>,
) -> Result<Vec<ProviderStateSnapshot>, Error> {
    let mut positions = BTreeMap::<String, usize>::new();
    let mut states = Vec::new();
    for (id, item) in observations {
        if completed_ids.contains(id) {
            continue;
        }
        let state = provider_state_from_deferred(id, item)?;
        if let Some(position) = positions.get(&state.namespace).copied() {
            states[position] = state;
        } else {
            positions.insert(state.namespace.clone(), states.len());
            states.push(state);
        }
    }
    Ok(states)
}

fn execution_log_error(error: impl std::error::Error + Send + Sync + 'static) -> Error {
    Error::new(ErrorKind::Internal, "tool execution log transition failed").with_source(error)
}

fn execution_authorization_error(error: impl std::error::Error + Send + Sync + 'static) -> Error {
    Error::new(
        ErrorKind::Internal,
        "tool execution authorization invariant failed",
    )
    .with_source(error)
}

fn execution_approval_decision_error(
    error: impl std::error::Error + Send + Sync + 'static,
) -> Error {
    Error::new(ErrorKind::Internal, "host approval decision failed").with_source(error)
}

fn budget_start_error(error: crate::BudgetError) -> Error {
    Error::new(
        ErrorKind::LimitExceeded,
        "tool loop could not reserve its first model step",
    )
    .with_source(error)
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

fn checked_deadline(now: Instant, duration: std::time::Duration) -> Instant {
    now.checked_add(duration).unwrap_or(now)
}

fn classify_deadline(
    total_deadline: Instant,
    local_deadline: Instant,
    local_kind: RunTimeoutKind,
) -> (Instant, RunTimeoutKind) {
    if total_deadline <= local_deadline {
        (total_deadline, RunTimeoutKind::Total)
    } else {
        (local_deadline, local_kind)
    }
}

fn earliest_timeout<const N: usize>(
    deadlines: [(Instant, RunTimeoutKind); N],
) -> (Instant, RunTimeoutKind) {
    deadlines
        .into_iter()
        .min_by_key(|(deadline, _)| *deadline)
        .unwrap_or((Instant::now(), RunTimeoutKind::Total))
}

fn unix_millis() -> u64 {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    u64::try_from(millis).unwrap_or(u64::MAX)
}
