//! Durable orchestration over the shared step engine and snapshot store.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use futures::StreamExt;
use sha2::{Digest, Sha256};
use siumai_core::{
    CallOptions, ContentPart, Error, ExecutionOwner, LanguageModel, LanguageRequest, Message,
    MessageRole, ToolOutcome, ToolResult,
};
use thiserror::Error;

use crate::approval::{
    ApprovalConsumeStore, ApprovalEnvelope, ApprovalVerificationError, ApprovalVerifier,
    TrustContext, TrustContextBuildError, TrustIdentity, verify_and_consume_at_unix_ms,
};
use crate::engine::{StepEngine, ToolHandling};
use crate::snapshot::{
    CheckpointId, CompletedToolSnapshot, IndeterminateReason, LineageId, PendingApprovalSnapshot,
    PendingProviderStepSnapshot, PendingStepSnapshot, PreparedToolSnapshot, ProviderStateSnapshot,
    ResumePoint, RunId, RunLease, RunSnapshot, RunSnapshotError, RunStore, RunStoreError,
    SnapshotCheckpoint, SnapshotEngineVersion, SnapshotFingerprint, SnapshotFingerprints,
    SnapshotReason, SnapshotRevision, SnapshotTerminal, ToolExecutionEvent, ToolExecutionStatus,
};
use crate::tool::{
    ApprovalPolicy, AuthorizedToolCall, EffectCertainty, ExternalApprovalDecider,
    ToolExecutionError, ToolExecutionRequest, ToolSet,
};
use crate::tool_loop::{ToolOutcomeAction, ToolOutcomePolicy};
use crate::{ModelTarget, RunEvent, RunReport, RunTerminal, Runtime, StepOptions, StepRecord};

const DURABLE_ENGINE_VERSION: &str = "siumai-runtime-durable-v2";
const DEFAULT_LEASE_TTL: Duration = Duration::from_secs(300);
static NEXT_CHECKPOINT_ID: AtomicU64 = AtomicU64::new(1);

/// A host-authenticated approval envelope for one pending call.
pub struct DurableApproval {
    call_id: String,
    envelope: ApprovalEnvelope,
    identity: TrustIdentity,
}

impl DurableApproval {
    pub fn new(
        call_id: impl Into<String>,
        envelope: ApprovalEnvelope,
        identity: TrustIdentity,
    ) -> Self {
        Self {
            call_id: call_id.into(),
            envelope,
            identity,
        }
    }

    pub fn call_id(&self) -> &str {
        &self.call_id
    }
}

impl std::fmt::Debug for DurableApproval {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DurableApproval")
            .field("call_id", &self.call_id)
            .field("contents", &"<redacted>")
            .finish()
    }
}

/// Explicit policy for a crash-recovered `Dispatched` tool call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum IndeterminateRecoveryPolicy {
    /// Persist the recovered run as indeterminate and stop.
    #[default]
    Halt,
    /// Retry only bindings that own both a stable key and an explicit replay policy.
    RetryStable,
}

/// Inputs supplied when resuming one durable run.
#[derive(Debug, Default)]
pub struct DurableResume {
    approvals: Vec<DurableApproval>,
    indeterminate_recovery: IndeterminateRecoveryPolicy,
}

impl DurableResume {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_approval(mut self, approval: DurableApproval) -> Self {
        self.approvals.push(approval);
        self
    }

    pub fn with_indeterminate_recovery(mut self, policy: IndeterminateRecoveryPolicy) -> Self {
        self.indeterminate_recovery = policy;
        self
    }
}

/// The latest committed durable checkpoint.
#[derive(Debug)]
pub struct DurableRun {
    revision: SnapshotRevision,
    snapshot: RunSnapshot,
}

impl DurableRun {
    pub fn revision(&self) -> SnapshotRevision {
        self.revision
    }

    pub fn snapshot(&self) -> &RunSnapshot {
        &self.snapshot
    }

    pub fn into_snapshot(self) -> RunSnapshot {
        self.snapshot
    }

    pub fn is_terminal(&self) -> bool {
        self.snapshot.resume_point().is_terminal()
    }
}

struct ApprovalRuntime {
    verifier: Arc<dyn ApprovalVerifier>,
    consume_store: Arc<dyn ApprovalConsumeStore>,
}

/// Clone-cheap durable tool loop backed by a lease/CAS [`RunStore`].
#[derive(Clone)]
pub struct DurableToolLoop {
    runtime: Runtime,
    model: Arc<dyn LanguageModel>,
    tools: ToolSet,
    store: Arc<dyn RunStore>,
    step_options: StepOptions,
    outcome_policy: ToolOutcomePolicy,
    lease_ttl: Duration,
    engine_version: SnapshotEngineVersion,
    fingerprints: SnapshotFingerprints,
    approval_runtime: Option<Arc<ApprovalRuntime>>,
}

impl std::fmt::Debug for DurableToolLoop {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DurableToolLoop")
            .field("target", &ModelTarget::from_model(self.model.as_ref()))
            .field("tools", &self.tools.len())
            .field("lease_ttl", &self.lease_ttl)
            .field("engine_version", &self.engine_version)
            .field("fingerprints", &"<redacted>")
            .field("approval_verification", &self.approval_runtime.is_some())
            .finish_non_exhaustive()
    }
}

impl DurableToolLoop {
    /// Construct a durable loop with host-owned option and approval-policy fingerprints.
    pub fn new(
        model: Arc<dyn LanguageModel>,
        tools: ToolSet,
        store: Arc<dyn RunStore>,
        options_fingerprint: SnapshotFingerprint,
        approval_policy_fingerprint: SnapshotFingerprint,
    ) -> Result<Self, DurableRunError> {
        let tool_catalog = SnapshotFingerprint::new(tools.fingerprint().as_str())?;
        Ok(Self {
            runtime: Runtime::default(),
            model,
            tools,
            store,
            step_options: StepOptions::default(),
            outcome_policy: ToolOutcomePolicy::default(),
            lease_ttl: DEFAULT_LEASE_TTL,
            engine_version: SnapshotEngineVersion::new(DURABLE_ENGINE_VERSION)?,
            fingerprints: SnapshotFingerprints {
                options: options_fingerprint,
                tool_catalog,
                approval_policy: approval_policy_fingerprint,
            },
            approval_runtime: None,
        })
    }

    pub fn with_runtime(mut self, runtime: Runtime) -> Self {
        self.runtime = runtime;
        self
    }

    pub fn with_step_options(mut self, options: StepOptions) -> Self {
        self.step_options = options;
        self
    }

    pub fn with_outcome_policy(mut self, policy: ToolOutcomePolicy) -> Self {
        self.outcome_policy = policy;
        self
    }

    pub fn with_lease_ttl(mut self, ttl: Duration) -> Result<Self, DurableRunError> {
        if ttl.is_zero() {
            return Err(DurableRunError::InvalidLeaseTtl);
        }
        self.lease_ttl = ttl;
        Ok(self)
    }

    /// Install authenticity verification and atomic replay consumption.
    pub fn with_approval_verification(
        mut self,
        verifier: Arc<dyn ApprovalVerifier>,
        consume_store: Arc<dyn ApprovalConsumeStore>,
    ) -> Self {
        self.approval_runtime = Some(Arc::new(ApprovalRuntime {
            verifier,
            consume_store,
        }));
        self
    }

    /// Start a new run and advance until its next durable quiescent boundary.
    pub async fn start(
        &self,
        run_id: RunId,
        lineage_id: LineageId,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<DurableRun, DurableRunError> {
        let mut lease = self.acquire(&run_id).await?;
        let result = self
            .start_with_lease(&mut lease, run_id, lineage_id, request, options)
            .await;
        self.finish_lease(lease, result).await
    }

    async fn start_with_lease(
        &self,
        lease: &mut RunLease,
        run_id: RunId,
        lineage_id: LineageId,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<DurableRun, DurableRunError> {
        if self
            .store
            .load(lease)
            .await
            .map_err(|error| map_store_error_for_run(error, &run_id))?
            .is_some()
        {
            return Err(DurableRunError::RunAlreadyExists { run_id });
        }

        let deadline_unix_ms = deadline_to_unix_ms(
            options.deadline(),
            self.runtime.run_budget().timeouts().total(),
        )?;
        let terminal = self
            .run_model_step(None, 0, request, options.clone())
            .await?;
        let snapshot = self.snapshot_from_model_terminal(
            run_id,
            lineage_id,
            None,
            deadline_unix_ms,
            terminal,
        )?;
        let revision = self
            .checkpoint(lease, SnapshotRevision::EMPTY, snapshot.clone())
            .await?;
        let state = DurableRun { revision, snapshot };
        self.advance(lease, state, DurableResume::default(), options)
            .await
    }

    /// Resume an existing run under one exclusive store lease.
    pub async fn resume(
        &self,
        run_id: &RunId,
        resume: DurableResume,
        options: CallOptions,
    ) -> Result<DurableRun, DurableRunError> {
        let mut lease = self.acquire(run_id).await?;
        let result = self
            .resume_with_lease(&mut lease, run_id, resume, options)
            .await;
        self.finish_lease(lease, result).await
    }

    async fn resume_with_lease(
        &self,
        lease: &mut RunLease,
        run_id: &RunId,
        resume: DurableResume,
        options: CallOptions,
    ) -> Result<DurableRun, DurableRunError> {
        let stored = self
            .store
            .load(lease)
            .await
            .map_err(|error| map_store_error_for_run(error, run_id))?
            .ok_or_else(|| DurableRunError::RunNotFound {
                run_id: run_id.clone(),
            })?;
        self.ensure_compatible(stored.snapshot())?;

        let state = DurableRun {
            revision: stored.revision(),
            snapshot: stored.into_snapshot(),
        };
        let state = self
            .recover_if_needed(lease, state, resume.indeterminate_recovery)
            .await?;
        self.advance(lease, state, resume, options).await
    }

    async fn advance(
        &self,
        lease: &mut RunLease,
        mut state: DurableRun,
        resume: DurableResume,
        options: CallOptions,
    ) -> Result<DurableRun, DurableRunError> {
        let approvals = collect_approvals(resume.approvals)?;
        loop {
            match state.snapshot.resume_point().clone() {
                ResumePoint::Terminal(_) | ResumePoint::AwaitingProvider(_) => return Ok(state),
                ResumePoint::AwaitingApprovals(step) | ResumePoint::ReadyToDispatch(step) => {
                    let progressed = self
                        .execute_pending_step(lease, state, step, &approvals, &options)
                        .await?;
                    state = progressed.state;
                    if progressed.awaiting_approval || state.is_terminal() {
                        return Ok(state);
                    }
                }
                ResumePoint::ReadyForModel { next_step, .. } => {
                    self.renew(lease).await?;
                    let request = LanguageRequest::new(state.snapshot.history().to_vec());
                    let call_options = options_for_snapshot(&state.snapshot, options.clone())?;
                    let terminal = self
                        .run_model_step(
                            Some(state.snapshot.report().clone()),
                            next_step,
                            request,
                            call_options,
                        )
                        .await?;
                    let snapshot = self.snapshot_from_model_terminal(
                        state.snapshot.run_id().clone(),
                        state.snapshot.lineage_id().clone(),
                        Some(state.snapshot.checkpoint_id().clone()),
                        state.snapshot.deadline_unix_ms(),
                        terminal,
                    )?;
                    let revision = self
                        .checkpoint(lease, state.revision, snapshot.clone())
                        .await?;
                    state = DurableRun { revision, snapshot };
                }
            }
        }
    }

    async fn execute_pending_step(
        &self,
        lease: &mut RunLease,
        mut state: DurableRun,
        mut step: PendingStepSnapshot,
        approvals: &BTreeMap<String, DurableApproval>,
        options: &CallOptions,
    ) -> Result<PendingProgress, DurableRunError> {
        loop {
            let completed_ordinals = step
                .completed()
                .iter()
                .map(CompletedToolSnapshot::ordinal)
                .collect::<BTreeSet<_>>();
            let Some(prepared) = step
                .prepared()
                .iter()
                .find(|prepared| !completed_ordinals.contains(&prepared.ordinal()))
                .cloned()
            else {
                let snapshot = self.finish_pending_step(&state.snapshot, &step, None)?;
                let revision = self
                    .checkpoint(lease, state.revision, snapshot.clone())
                    .await?;
                return Ok(PendingProgress {
                    state: DurableRun { revision, snapshot },
                    awaiting_approval: false,
                });
            };

            match state.snapshot.execution_log().status(&prepared.call().id) {
                Some(ToolExecutionStatus::Prepared) => {}
                Some(ToolExecutionStatus::Completed) => {
                    return Err(DurableRunError::Invariant {
                        message: "completed execution is missing its pending-step receipt",
                    });
                }
                Some(ToolExecutionStatus::Dispatched | ToolExecutionStatus::Indeterminate) => {
                    return Err(DurableRunError::Invariant {
                        message: "resume recovery must settle dispatched work before execution",
                    });
                }
                None => {
                    return Err(DurableRunError::Invariant {
                        message: "pending tool is missing its prepared execution event",
                    });
                }
            }

            let request = self.restore_request(&prepared)?;
            let authorized = match request.approval_policy() {
                ApprovalPolicy::NotRequired => {
                    request
                        .authorize_not_required()
                        .map_err(|_| DurableRunError::Invariant {
                            message: "not-required tool authorization was rejected",
                        })?
                }
                ApprovalPolicy::Required => {
                    let Some(approval) = approvals.get(request.call_id()) else {
                        return Ok(PendingProgress {
                            state,
                            awaiting_approval: true,
                        });
                    };
                    self.verify_approval(&state.snapshot, &request, approval)
                        .await?
                }
            };

            self.renew(lease).await?;
            // Persist dispatch intent before crossing the executor boundary. A
            // crash in the narrow gap can conservatively over-report an
            // indeterminate effect, but can never silently replay an effect
            // that may already have happened.
            let dispatched = self.dispatched_snapshot(&state.snapshot, &step, &prepared)?;
            let revision = self
                .checkpoint(lease, state.revision, dispatched.clone())
                .await?;
            state = DurableRun {
                revision,
                snapshot: dispatched,
            };
            step = state
                .snapshot
                .resume_point()
                .pending_step()
                .cloned()
                .ok_or(DurableRunError::Invariant {
                    message: "dispatched checkpoint lost its pending step",
                })?;

            let call_options = options_for_snapshot(&state.snapshot, options.clone())?;
            let dispatch = dispatch_authorized(
                authorized,
                call_options,
                self.runtime.run_budget().timeouts().tool(),
            )
            .await;
            let result = match dispatch {
                DispatchResult::Completed(Ok(result)) => result,
                DispatchResult::Completed(Err(error))
                    if error.effect_certainty() == EffectCertainty::Indeterminate =>
                {
                    let snapshot = self.indeterminate_snapshot(
                        &state.snapshot,
                        &prepared,
                        IndeterminateReason::DispatchOutcomeUnknown,
                    )?;
                    let revision = self
                        .checkpoint(lease, state.revision, snapshot.clone())
                        .await?;
                    return Ok(PendingProgress {
                        state: DurableRun { revision, snapshot },
                        awaiting_approval: false,
                    });
                }
                DispatchResult::Completed(Err(error)) => ToolResult {
                    call_id: prepared.call().id.clone(),
                    name: prepared.call().name.clone(),
                    outcome: execution_error_outcome(error),
                },
                DispatchResult::Cancelled => {
                    let snapshot = self.indeterminate_snapshot(
                        &state.snapshot,
                        &prepared,
                        IndeterminateReason::CancellationAfterDispatch,
                    )?;
                    let revision = self
                        .checkpoint(lease, state.revision, snapshot.clone())
                        .await?;
                    return Ok(PendingProgress {
                        state: DurableRun { revision, snapshot },
                        awaiting_approval: false,
                    });
                }
                DispatchResult::TimedOut => {
                    let snapshot = self.indeterminate_snapshot(
                        &state.snapshot,
                        &prepared,
                        IndeterminateReason::DispatchOutcomeUnknown,
                    )?;
                    let revision = self
                        .checkpoint(lease, state.revision, snapshot.clone())
                        .await?;
                    return Ok(PendingProgress {
                        state: DurableRun { revision, snapshot },
                        awaiting_approval: false,
                    });
                }
            };

            let completed = match self.completed_snapshot(&state.snapshot, &step, &prepared, result)
            {
                Ok(snapshot) => snapshot,
                Err(DurableRunError::Budget(_)) => self.indeterminate_snapshot(
                    &state.snapshot,
                    &prepared,
                    IndeterminateReason::CheckpointFailure,
                )?,
                Err(error) => return Err(error),
            };
            let result_outcome = completed
                .resume_point()
                .pending_step()
                .and_then(|pending| {
                    pending
                        .completed()
                        .iter()
                        .find(|completed| completed.ordinal() == prepared.ordinal())
                })
                .map(|completed| completed.result().outcome.clone());
            let revision = self
                .checkpoint(lease, state.revision, completed.clone())
                .await?;
            state = DurableRun {
                revision,
                snapshot: completed,
            };
            if state.is_terminal() {
                return Ok(PendingProgress {
                    state,
                    awaiting_approval: false,
                });
            }
            step = state
                .snapshot
                .resume_point()
                .pending_step()
                .cloned()
                .ok_or(DurableRunError::Invariant {
                    message: "completed checkpoint lost its pending step",
                })?;

            if result_outcome.as_ref().is_some_and(|outcome| {
                self.outcome_policy.action(outcome) == ToolOutcomeAction::Stop
            }) {
                let snapshot = self.finish_pending_step(
                    &state.snapshot,
                    &step,
                    Some(SnapshotTerminal::Completed {
                        reason: Some(reason("tool_outcome_stopped")?),
                    }),
                )?;
                let revision = self
                    .checkpoint(lease, state.revision, snapshot.clone())
                    .await?;
                return Ok(PendingProgress {
                    state: DurableRun { revision, snapshot },
                    awaiting_approval: false,
                });
            }
        }
    }

    async fn verify_approval(
        &self,
        snapshot: &RunSnapshot,
        request: &ToolExecutionRequest,
        approval: &DurableApproval,
    ) -> Result<AuthorizedToolCall, DurableRunError> {
        let runtime = self
            .approval_runtime
            .as_ref()
            .ok_or(DurableRunError::ApprovalVerificationNotConfigured)?;
        let context = TrustContext::builder(approval.identity.clone())
            .model_target(snapshot.target().clone())
            .run_id(snapshot.run_id().clone())
            .lineage_id(snapshot.lineage_id().clone())
            .checkpoint_id(snapshot.checkpoint_id().clone())
            .execution_owner(request.owner().clone())
            .binding_identity(request.binding_identity().clone())
            .tool_call_id(request.call_id())
            .canonical_arguments_digest(request.canonical_arguments_digest())
            .catalog_fingerprint(snapshot.fingerprints().tool_catalog.as_str())
            .policy_fingerprint(snapshot.fingerprints().approval_policy.as_str())
            .build()?;
        let verified = verify_and_consume_at_unix_ms(
            runtime.verifier.as_ref(),
            runtime.consume_store.as_ref(),
            &approval.envelope,
            &context,
            unix_millis()?,
        )
        .await?;
        request
            .clone()
            .authorize_verified(verified)
            .map_err(|_| DurableRunError::Invariant {
                message: "verified approval did not authorize its frozen request",
            })
    }

    async fn recover_if_needed(
        &self,
        lease: &mut RunLease,
        state: DurableRun,
        policy: IndeterminateRecoveryPolicy,
    ) -> Result<DurableRun, DurableRunError> {
        let execution_log = state.snapshot.execution_log();
        let dispatched = execution_log
            .events()
            .iter()
            .filter(|event| {
                execution_log.status(event.call_id()) == Some(ToolExecutionStatus::Dispatched)
            })
            .map(|event| event.call_id().to_string())
            .collect::<BTreeSet<_>>();
        if dispatched.is_empty() {
            return Ok(state);
        }

        let recovered = state
            .snapshot
            .clone()
            .recovered_for_resume(unix_millis()?)?;
        let next = match policy {
            IndeterminateRecoveryPolicy::Halt => self.successor_snapshot(
                &state.snapshot,
                recovered.report().clone(),
                recovered.resume_point().clone(),
            )?,
            IndeterminateRecoveryPolicy::RetryStable => {
                self.retry_recovered_snapshot(&state.snapshot, recovered, &dispatched)?
            }
        };
        self.renew(lease).await?;
        let revision = self.checkpoint(lease, state.revision, next.clone()).await?;
        Ok(DurableRun {
            revision,
            snapshot: next,
        })
    }

    fn retry_recovered_snapshot(
        &self,
        previous: &RunSnapshot,
        recovered: RunSnapshot,
        dispatched: &BTreeSet<String>,
    ) -> Result<RunSnapshot, DurableRunError> {
        let Some(previous_step) = previous.resume_point().pending_step() else {
            return self.successor_snapshot(
                previous,
                recovered.report().clone(),
                recovered.resume_point().clone(),
            );
        };
        let mut report = recovered.report().clone();
        let mut prepared = previous_step.prepared().to_vec();
        for _ in previous_step.pending_approvals() {
            report
                .budget_mut()
                .reserve_pending_approval(self.runtime.run_budget())?;
        }

        for tool in &mut prepared {
            if !dispatched.contains(&tool.call().id) {
                continue;
            }
            let request = self.restore_request(tool)?;
            if request.approval_policy() == ApprovalPolicy::Required
                || !request.permits_retry(EffectCertainty::Indeterminate)
            {
                return self.successor_snapshot(
                    previous,
                    recovered.report().clone(),
                    recovered.resume_point().clone(),
                );
            }
            let next = request
                .next_attempt(EffectCertainty::Indeterminate)
                .map_err(|_| DurableRunError::RecoveryNotPermitted {
                    call_id: request.call_id().to_string(),
                })?;
            *tool = PreparedToolSnapshot::new(
                tool.ordinal(),
                next.call().clone(),
                next.binding_identity().clone(),
                next.recovery_policy(),
                next.idempotency_key().cloned(),
                next.attempt(),
            );
            let sequence = report.execution_log().next_sequence();
            report
                .execution_log_mut()
                .append(ToolExecutionEvent::prepared(
                    sequence,
                    unix_millis()?,
                    previous_step.index(),
                    tool.clone(),
                ))?;
        }

        let step = PendingStepSnapshot::new(
            previous_step.index(),
            previous_step.target().clone(),
            previous_step.response().clone(),
            prepared,
            previous_step.completed().to_vec(),
            previous_step.pending_approvals().to_vec(),
        );
        let resume_point = if step.pending_approvals().is_empty() {
            ResumePoint::ReadyToDispatch(step)
        } else {
            ResumePoint::AwaitingApprovals(step)
        };
        self.successor_snapshot(previous, report, resume_point)
    }

    fn restore_request(
        &self,
        prepared: &PreparedToolSnapshot,
    ) -> Result<ToolExecutionRequest, DurableRunError> {
        let mut request = self
            .tools
            .resolve_frozen(prepared.call().clone(), prepared.binding())?;
        while request.attempt() < prepared.attempt() {
            request = request
                .next_attempt(EffectCertainty::Indeterminate)
                .map_err(|_| DurableRunError::RecoveryNotPermitted {
                    call_id: request.call_id().to_string(),
                })?;
        }
        if request.attempt() != prepared.attempt()
            || request.recovery_policy() != prepared.recovery_policy()
            || request.idempotency_key() != prepared.stable_idempotency_key()
        {
            return Err(DurableRunError::FrozenRequestMismatch {
                call_id: prepared.call().id.clone(),
            });
        }
        request.validate()?;
        Ok(request)
    }

    async fn run_model_step(
        &self,
        report: Option<RunReport>,
        step: u32,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<RunTerminal, DurableRunError> {
        let engine = if let Some(report) = report {
            StepEngine::establish_seeded(
                self.runtime.clone(),
                Arc::clone(&self.model),
                self.tools.clone(),
                request,
                self.step_options.clone(),
                options,
                self.outcome_policy,
                Arc::new(ExternalApprovalDecider::default()),
                ToolHandling::ObserveOnly,
                report,
                step,
            )
            .await?
        } else {
            StepEngine::establish(
                self.runtime.clone(),
                Arc::clone(&self.model),
                self.tools.clone(),
                request,
                self.step_options.clone(),
                options,
                self.outcome_policy,
                Arc::new(ExternalApprovalDecider::default()),
                ToolHandling::ObserveOnly,
            )
            .await?
        };
        let mut stream = Box::pin(engine.into_stream());
        while let Some(event) = stream.next().await {
            if let RunEvent::Terminal(terminal) = event? {
                return Ok(terminal);
            }
        }
        Err(DurableRunError::Invariant {
            message: "step engine ended without a terminal",
        })
    }

    fn snapshot_from_model_terminal(
        &self,
        run_id: RunId,
        lineage_id: LineageId,
        parent: Option<CheckpointId>,
        deadline_unix_ms: Option<u64>,
        terminal: RunTerminal,
    ) -> Result<RunSnapshot, DurableRunError> {
        let (mut report, terminal) = match terminal {
            RunTerminal::Completed { report } => (*report, None),
            RunTerminal::Stopped { report, .. } => (
                *report,
                Some(SnapshotTerminal::Completed {
                    reason: Some(reason("runtime_stopped")?),
                }),
            ),
            RunTerminal::Suspended { report, .. } => (
                *report,
                Some(SnapshotTerminal::Failed {
                    reason: reason("unexpected_runtime_suspension")?,
                }),
            ),
            RunTerminal::BudgetExceeded { report, .. } | RunTerminal::TimedOut { report, .. } => (
                *report,
                Some(SnapshotTerminal::Exhausted {
                    reason: reason("runtime_budget_exhausted")?,
                }),
            ),
            RunTerminal::Indeterminate { report, .. } => (
                *report,
                Some(SnapshotTerminal::Indeterminate {
                    reason: reason("runtime_indeterminate")?,
                }),
            ),
            RunTerminal::Failed { report, .. } => (
                *report,
                Some(SnapshotTerminal::Failed {
                    reason: reason("runtime_failed")?,
                }),
            ),
            RunTerminal::Cancelled { report, .. } => (
                *report,
                Some(SnapshotTerminal::Cancelled {
                    reason: reason("runtime_cancelled")?,
                }),
            ),
            RunTerminal::ResumeConflict { .. } => {
                return Err(DurableRunError::Invariant {
                    message: "step engine returned an internal resume conflict",
                });
            }
        };

        let resume_point = if let Some(terminal) = terminal {
            ResumePoint::Terminal(terminal)
        } else {
            let record = report.steps_mut().pop().ok_or(DurableRunError::Invariant {
                message: "completed model step is missing its step record",
            })?;
            let calls = record
                .response()
                .content()
                .iter()
                .filter_map(|part| match part {
                    ContentPart::ToolCall(call) => Some(call.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            if calls.is_empty() {
                report.steps_mut().push(record);
                ResumePoint::Terminal(SnapshotTerminal::Completed { reason: None })
            } else if calls
                .iter()
                .any(|call| !matches!(call.owner, ExecutionOwner::Local))
            {
                let provider_state = calls
                    .iter()
                    .map(provider_state_from_call)
                    .collect::<Result<Vec<_>, _>>()?;
                ResumePoint::AwaitingProvider(PendingProviderStepSnapshot::new(
                    record.index(),
                    record.target().clone(),
                    record.response().clone(),
                    provider_state,
                ))
            } else {
                let pending = self.prepare_pending_step(&mut report, &record, calls)?;
                if pending.pending_approvals().is_empty() {
                    ResumePoint::ReadyToDispatch(pending)
                } else {
                    ResumePoint::AwaitingApprovals(pending)
                }
            }
        };

        let checkpoint = SnapshotCheckpoint::new(
            self.engine_version.clone(),
            run_id,
            lineage_id,
            next_checkpoint_id()?,
            parent,
        )?;
        Ok(RunSnapshot::new(
            checkpoint,
            self.fingerprints.clone(),
            report,
            deadline_unix_ms,
            resume_point,
        )?)
    }

    fn prepare_pending_step(
        &self,
        report: &mut RunReport,
        record: &StepRecord,
        calls: Vec<siumai_core::ToolCall>,
    ) -> Result<PendingStepSnapshot, DurableRunError> {
        let mut prepared = Vec::with_capacity(calls.len());
        let mut approvals = Vec::new();
        for (ordinal, call) in calls.into_iter().enumerate() {
            let request = self.tools.resolve(call)?;
            request.validate()?;
            let argument_bytes = serde_json::to_vec(request.arguments())?.len();
            report
                .budget_mut()
                .charge_tool_call(argument_bytes, self.runtime.run_budget())?;
            let ordinal = u32::try_from(ordinal).map_err(|_| DurableRunError::Invariant {
                message: "tool ordinal cannot be represented in a snapshot",
            })?;
            let tool = PreparedToolSnapshot::new(
                ordinal,
                request.call().clone(),
                request.binding_identity().clone(),
                request.recovery_policy(),
                request.idempotency_key().cloned(),
                request.attempt(),
            );
            let sequence = report.execution_log().next_sequence();
            report
                .execution_log_mut()
                .append(ToolExecutionEvent::prepared(
                    sequence,
                    unix_millis()?,
                    record.index(),
                    tool.clone(),
                ))?;
            if request.approval_policy() == ApprovalPolicy::Required {
                report
                    .budget_mut()
                    .reserve_pending_approval(self.runtime.run_budget())?;
                approvals.push(PendingApprovalSnapshot {
                    approval_id: format!("approval:{}", request.call_id()),
                    call: request.call().clone(),
                    binding: request.binding_identity().clone(),
                    claim_fingerprint: pending_approval_fingerprint(&request, &self.fingerprints)?,
                    expires_at_unix_ms: None,
                });
            }
            prepared.push(tool);
        }
        Ok(PendingStepSnapshot::new(
            record.index(),
            record.target().clone(),
            record.response().clone(),
            prepared,
            Vec::new(),
            approvals,
        ))
    }

    fn dispatched_snapshot(
        &self,
        previous: &RunSnapshot,
        step: &PendingStepSnapshot,
        prepared: &PreparedToolSnapshot,
    ) -> Result<RunSnapshot, DurableRunError> {
        let mut report = previous.report().clone();
        let sequence = report.execution_log().next_sequence();
        report
            .execution_log_mut()
            .append(ToolExecutionEvent::dispatched(
                sequence,
                unix_millis()?,
                &prepared.call().id,
                prepared.attempt(),
                None,
            ))?;
        let pending_approvals = step
            .pending_approvals()
            .iter()
            .filter(|approval| approval.call.id != prepared.call().id)
            .cloned()
            .collect::<Vec<_>>();
        if pending_approvals.len() != step.pending_approvals().len() {
            report.budget_mut().release_pending_approval();
        }
        let next_step = PendingStepSnapshot::new(
            step.index(),
            step.target().clone(),
            step.response().clone(),
            step.prepared().to_vec(),
            step.completed().to_vec(),
            pending_approvals,
        );
        let resume_point = if next_step.pending_approvals().is_empty() {
            ResumePoint::ReadyToDispatch(next_step)
        } else {
            ResumePoint::AwaitingApprovals(next_step)
        };
        self.successor_snapshot(previous, report, resume_point)
    }

    fn completed_snapshot(
        &self,
        previous: &RunSnapshot,
        step: &PendingStepSnapshot,
        prepared: &PreparedToolSnapshot,
        result: ToolResult,
    ) -> Result<RunSnapshot, DurableRunError> {
        let mut report = previous.report().clone();
        let result_bytes = serde_json::to_vec(&result)?.len();
        report
            .budget_mut()
            .charge_tool_result(result_bytes, self.runtime.run_budget())?;
        let sequence = report.execution_log().next_sequence();
        report
            .execution_log_mut()
            .append(ToolExecutionEvent::completed(
                sequence,
                unix_millis()?,
                &result.call_id,
                prepared.attempt(),
                result.outcome.clone(),
            ))?;
        let mut completed = step.completed().to_vec();
        completed.push(CompletedToolSnapshot::new(prepared.ordinal(), result));
        completed.sort_unstable_by_key(CompletedToolSnapshot::ordinal);
        let next_step = PendingStepSnapshot::new(
            step.index(),
            step.target().clone(),
            step.response().clone(),
            step.prepared().to_vec(),
            completed,
            step.pending_approvals().to_vec(),
        );
        let resume_point = if next_step.pending_approvals().is_empty() {
            ResumePoint::ReadyToDispatch(next_step)
        } else {
            ResumePoint::AwaitingApprovals(next_step)
        };
        self.successor_snapshot(previous, report, resume_point)
    }

    fn indeterminate_snapshot(
        &self,
        previous: &RunSnapshot,
        prepared: &PreparedToolSnapshot,
        reason_kind: IndeterminateReason,
    ) -> Result<RunSnapshot, DurableRunError> {
        let mut report = previous.report().clone();
        for _ in previous.pending_approvals() {
            report.budget_mut().release_pending_approval();
        }
        let sequence = report.execution_log().next_sequence();
        report
            .execution_log_mut()
            .append(ToolExecutionEvent::indeterminate(
                sequence,
                unix_millis()?,
                &prepared.call().id,
                prepared.attempt(),
                reason_kind,
            ))?;
        self.successor_snapshot(
            previous,
            report,
            ResumePoint::Terminal(SnapshotTerminal::Indeterminate {
                reason: reason("tool_effect_indeterminate")?,
            }),
        )
    }

    fn finish_pending_step(
        &self,
        previous: &RunSnapshot,
        step: &PendingStepSnapshot,
        terminal: Option<SnapshotTerminal>,
    ) -> Result<RunSnapshot, DurableRunError> {
        let mut report = previous.report().clone();
        let results = step
            .completed()
            .iter()
            .map(|completed| completed.result().clone())
            .collect::<Vec<_>>();
        if !results.is_empty() {
            report.messages_mut().push(Message {
                role: MessageRole::Tool,
                content: results
                    .iter()
                    .cloned()
                    .map(ContentPart::ToolResult)
                    .collect(),
            });
        }
        report.steps_mut().push(StepRecord::new(
            step.index(),
            step.target().clone(),
            step.response().clone(),
            results,
        ));
        if terminal.is_some() {
            for _ in step.pending_approvals() {
                report.budget_mut().release_pending_approval();
            }
        }
        let resume_point = terminal.map_or_else(
            || ResumePoint::ReadyForModel {
                next_step: step.index().saturating_add(1),
                target: step.target().clone(),
            },
            ResumePoint::Terminal,
        );
        self.successor_snapshot(previous, report, resume_point)
    }

    fn successor_snapshot(
        &self,
        previous: &RunSnapshot,
        report: RunReport,
        resume_point: ResumePoint,
    ) -> Result<RunSnapshot, DurableRunError> {
        let checkpoint = SnapshotCheckpoint::new(
            previous.engine_version().clone(),
            previous.run_id().clone(),
            previous.lineage_id().clone(),
            next_checkpoint_id()?,
            Some(previous.checkpoint_id().clone()),
        )?;
        Ok(RunSnapshot::new(
            checkpoint,
            previous.fingerprints().clone(),
            report,
            previous.deadline_unix_ms(),
            resume_point,
        )?)
    }

    async fn checkpoint(
        &self,
        lease: &RunLease,
        expected: SnapshotRevision,
        snapshot: RunSnapshot,
    ) -> Result<SnapshotRevision, DurableRunError> {
        let snapshot_bytes = serde_json::to_vec(&snapshot)?.len();
        snapshot
            .budget()
            .check_snapshot_bytes(snapshot_bytes, self.runtime.run_budget())?;
        self.store
            .compare_and_swap(lease, expected, snapshot)
            .await
            .map_err(|error| map_store_error_for_run(error, lease.run_id()))
    }

    async fn acquire(&self, run_id: &RunId) -> Result<RunLease, DurableRunError> {
        self.store
            .acquire(run_id, self.lease_ttl)
            .await
            .map_err(map_store_error)
    }

    async fn renew(&self, lease: &mut RunLease) -> Result<(), DurableRunError> {
        self.store
            .renew(lease, self.lease_ttl)
            .await
            .map_err(|error| map_store_error_for_run(error, lease.run_id()))
    }

    async fn finish_lease<T>(
        &self,
        lease: RunLease,
        result: Result<T, DurableRunError>,
    ) -> Result<T, DurableRunError> {
        let run_id = lease.run_id().clone();
        match (result, self.store.release(lease).await) {
            (result, Ok(())) => result,
            (Ok(_), Err(source)) => Err(DurableRunError::LeaseRelease {
                run_id,
                operation: None,
                source,
            }),
            (Err(operation), Err(source)) => Err(DurableRunError::LeaseRelease {
                run_id,
                operation: Some(Box::new(operation)),
                source,
            }),
        }
    }

    fn ensure_compatible(&self, snapshot: &RunSnapshot) -> Result<(), DurableRunError> {
        if snapshot.engine_version() != &self.engine_version {
            return Err(DurableRunError::IncompatibleSnapshot {
                field: "engine_version",
            });
        }
        if snapshot.fingerprints() != &self.fingerprints {
            return Err(DurableRunError::IncompatibleSnapshot {
                field: "fingerprints",
            });
        }
        if snapshot.report().initial_target() != &ModelTarget::from_model(self.model.as_ref()) {
            return Err(DurableRunError::IncompatibleSnapshot {
                field: "model_target",
            });
        }
        Ok(())
    }
}

struct PendingProgress {
    state: DurableRun,
    awaiting_approval: bool,
}

enum DispatchResult {
    Completed(Result<ToolResult, ToolExecutionError>),
    TimedOut,
    Cancelled,
}

async fn dispatch_authorized(
    authorized: AuthorizedToolCall,
    options: CallOptions,
    tool_timeout: Duration,
) -> DispatchResult {
    let cancellation = options.cancellation().clone();
    let timeout =
        Instant::now()
            .checked_add(tool_timeout)
            .map_or_else(Instant::now, |tool_deadline| {
                options
                    .deadline()
                    .map_or(tool_deadline, |deadline| deadline.min(tool_deadline))
            });
    tokio::select! {
        biased;
        _ = cancellation.cancelled() => DispatchResult::Cancelled,
        _ = tokio::time::sleep_until(tokio::time::Instant::from_std(timeout)) => {
            DispatchResult::TimedOut
        }
        result = authorized.dispatch() => DispatchResult::Completed(result),
    }
}

fn collect_approvals(
    approvals: Vec<DurableApproval>,
) -> Result<BTreeMap<String, DurableApproval>, DurableRunError> {
    let mut by_call = BTreeMap::new();
    for approval in approvals {
        let call_id = approval.call_id.clone();
        if by_call.insert(call_id.clone(), approval).is_some() {
            return Err(DurableRunError::DuplicateApproval { call_id });
        }
    }
    Ok(by_call)
}

fn provider_state_from_call(
    call: &siumai_core::ToolCall,
) -> Result<ProviderStateSnapshot, DurableRunError> {
    let namespace = match &call.owner {
        ExecutionOwner::Provider { provider } => format!("provider-tool:{}:{}", provider, call.id),
        _ => format!("external-tool:{}", call.id),
    };
    Ok(ProviderStateSnapshot {
        namespace,
        correlation_id: Some(call.id.clone()),
        encoding: "tool-call-id".to_string(),
        payload: Vec::new(),
    })
}

fn pending_approval_fingerprint(
    request: &ToolExecutionRequest,
    fingerprints: &SnapshotFingerprints,
) -> Result<SnapshotFingerprint, DurableRunError> {
    let mut digest = Sha256::new();
    digest.update(b"siumai.pending-approval.v1\0");
    digest.update(request.call_id().as_bytes());
    digest.update([0]);
    digest.update(request.binding_identity().fingerprint.as_bytes());
    digest.update([0]);
    digest.update(request.canonical_arguments_digest().as_bytes());
    digest.update([0]);
    digest.update(fingerprints.tool_catalog.as_str().as_bytes());
    digest.update([0]);
    digest.update(fingerprints.approval_policy.as_str().as_bytes());
    SnapshotFingerprint::new(format!("sha256:{:x}", digest.finalize())).map_err(Into::into)
}

fn execution_error_outcome(error: ToolExecutionError) -> ToolOutcome {
    match error {
        ToolExecutionError::ExecutorFailed {
            message, retryable, ..
        } => ToolOutcome::ExecutionFailed { message, retryable },
        ToolExecutionError::InvalidArguments { message, .. } => ToolOutcome::ExecutionFailed {
            message,
            retryable: false,
        },
        other => ToolOutcome::ExecutionFailed {
            message: other.to_string(),
            retryable: false,
        },
    }
}

fn options_for_snapshot(
    snapshot: &RunSnapshot,
    options: CallOptions,
) -> Result<CallOptions, DurableRunError> {
    let Some(deadline) = snapshot.deadline_unix_ms() else {
        return Ok(options);
    };
    let now_unix_ms = unix_millis()?;
    let remaining = deadline.saturating_sub(now_unix_ms);
    let stored_deadline = Instant::now()
        .checked_add(Duration::from_millis(remaining))
        .unwrap_or_else(Instant::now);
    let deadline = options
        .deadline()
        .map_or(stored_deadline, |caller| caller.min(stored_deadline));
    Ok(options.with_deadline(deadline))
}

fn deadline_to_unix_ms(
    caller_deadline: Option<Instant>,
    total_timeout: Duration,
) -> Result<Option<u64>, DurableRunError> {
    let now_instant = Instant::now();
    let runtime_deadline = now_instant
        .checked_add(total_timeout)
        .unwrap_or(now_instant);
    let deadline = caller_deadline.map_or(runtime_deadline, |caller| caller.min(runtime_deadline));
    let remaining = deadline.saturating_duration_since(now_instant);
    let now = unix_millis()?;
    let additional = u64::try_from(remaining.as_millis()).map_err(|_| DurableRunError::Clock)?;
    now.checked_add(additional)
        .map(Some)
        .ok_or(DurableRunError::Clock)
}

fn unix_millis() -> Result<u64, DurableRunError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| DurableRunError::Clock)?
        .as_millis()
        .try_into()
        .map_err(|_| DurableRunError::Clock)
}

fn next_checkpoint_id() -> Result<CheckpointId, DurableRunError> {
    let sequence = NEXT_CHECKPOINT_ID.fetch_add(1, Ordering::Relaxed);
    let now = unix_millis()?;
    Ok(CheckpointId::new(format!("checkpoint-{now}-{sequence}"))?)
}

fn reason(code: &'static str) -> Result<SnapshotReason, DurableRunError> {
    SnapshotReason::new(code, None).map_err(Into::into)
}

fn map_store_error(error: RunStoreError) -> DurableRunError {
    match error {
        RunStoreError::LeaseConflict { run_id } => DurableRunError::ResumeConflict { run_id },
        other => DurableRunError::Store(other),
    }
}

fn map_store_error_for_run(error: RunStoreError, run_id: &RunId) -> DurableRunError {
    match error {
        RunStoreError::LeaseConflict { run_id } => DurableRunError::ResumeConflict { run_id },
        RunStoreError::CasConflict { .. }
        | RunStoreError::LeaseLost
        | RunStoreError::LeaseExpired => DurableRunError::ResumeConflict {
            run_id: run_id.clone(),
        },
        other => DurableRunError::Store(other),
    }
}

/// Typed durable orchestration failure.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DurableRunError {
    #[error("run `{run_id}` is already stored")]
    RunAlreadyExists { run_id: RunId },
    #[error("run `{run_id}` was not found")]
    RunNotFound { run_id: RunId },
    #[error("run `{run_id}` is already being resumed")]
    ResumeConflict { run_id: RunId },
    #[error("durable run snapshot is incompatible in field `{field}`")]
    IncompatibleSnapshot { field: &'static str },
    #[error("approval verification is not configured")]
    ApprovalVerificationNotConfigured,
    #[error("approval for call `{call_id}` was supplied more than once")]
    DuplicateApproval { call_id: String },
    #[error("recovery is not permitted for call `{call_id}`")]
    RecoveryNotPermitted { call_id: String },
    #[error("restored frozen request does not match snapshot call `{call_id}`")]
    FrozenRequestMismatch { call_id: String },
    #[error("lease duration must be greater than zero")]
    InvalidLeaseTtl,
    #[error(
        "failed to release durable run lease for `{run_id}`; the operation may already have committed"
    )]
    LeaseRelease {
        run_id: RunId,
        operation: Option<Box<DurableRunError>>,
        #[source]
        source: RunStoreError,
    },
    #[error("system clock cannot represent the durable timestamp")]
    Clock,
    #[error("durable runtime invariant failed: {message}")]
    Invariant { message: &'static str },
    #[error(transparent)]
    Runtime(#[from] Error),
    #[error("run store failed: {0}")]
    Store(#[source] RunStoreError),
    #[error(transparent)]
    Snapshot(#[from] RunSnapshotError),
    #[error(transparent)]
    SnapshotId(#[from] crate::snapshot::InvalidSnapshotId),
    #[error(transparent)]
    ExecutionTransition(#[from] crate::snapshot::ToolExecutionTransitionError),
    #[error(transparent)]
    Tool(#[from] ToolExecutionError),
    #[error(transparent)]
    Budget(#[from] crate::BudgetError),
    #[error(transparent)]
    Approval(#[from] ApprovalVerificationError),
    #[error(transparent)]
    TrustContext(#[from] TrustContextBuildError),
    #[error(transparent)]
    Serialization(#[from] serde_json::Error),
}
