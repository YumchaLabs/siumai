//! Durable orchestration over the shared step engine and snapshot store.

use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};
use siumai_core::{CallOptions, Error, LanguageModel, LanguageRequest};
use thiserror::Error;

use crate::approval::{
    ApprovalConsumeStore, ApprovalEnvelope, ApprovalVerificationError, ApprovalVerificationInput,
    ApprovalVerifier, TrustContext, TrustContextBuildError, TrustIdentity, VerifiedApproval,
    verify_and_consume_many_at_unix_ms,
};
use crate::engine::checkpoint::{
    CheckpointBoundary, CheckpointControl, EngineCheckpoint, EngineCheckpointPort,
    EngineCheckpointState, PendingApprovalCheckpoint,
};
use crate::engine::{EngineResumeError, EngineResumeSeed, StepEngine, ToolHandling};
use crate::snapshot::{
    CheckpointId, LineageId, PendingApprovalSnapshot, PendingStepSnapshot, PreparedToolSnapshot,
    ResumePoint, RunId, RunLease, RunSnapshot, RunSnapshotError, RunStore, RunStoreError,
    SnapshotCheckpoint, SnapshotEngineVersion, SnapshotFingerprint, SnapshotFingerprints,
    SnapshotRevision, ToolExecutionEvent, ToolExecutionStatus,
};
use crate::tool::{
    ApprovalPolicy, EffectCertainty, ExternalApprovalDecider, ToolExecutionError,
    ToolExecutionRequest, ToolSet,
};
use crate::tool_loop::ToolOutcomePolicy;
use crate::{ModelTarget, ProjectionPolicy, RunReport, Runtime, StepModelSelector, StepOptions};

const DURABLE_ENGINE_VERSION: &str = "siumai-runtime-durable-v5";
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

struct DurableCheckpointPort<'a> {
    owner: &'a DurableToolLoop,
    lease: &'a mut RunLease,
    run_id: RunId,
    lineage_id: LineageId,
    deadline_unix_ms: Option<u64>,
    state: Option<DurableRun>,
}

impl<'a> DurableCheckpointPort<'a> {
    fn new(
        owner: &'a DurableToolLoop,
        lease: &'a mut RunLease,
        run_id: RunId,
        lineage_id: LineageId,
        deadline_unix_ms: Option<u64>,
        state: Option<DurableRun>,
    ) -> Self {
        Self {
            owner,
            lease,
            run_id,
            lineage_id,
            deadline_unix_ms,
            state,
        }
    }

    fn into_state(self) -> Result<DurableRun, DurableRunError> {
        self.state.ok_or(DurableRunError::Invariant {
            message: "step engine reached a durable boundary without committing a snapshot",
        })
    }

    async fn commit_candidate(
        &mut self,
        checkpoint: EngineCheckpoint,
    ) -> Result<CheckpointControl, DurableRunError> {
        let (boundary, continuation, report, state) = checkpoint.into_parts();
        let (resume_point, can_progress) = self.resume_point(state)?;
        let (expected, parent, engine_version, fingerprints, deadline_unix_ms) =
            match self.state.as_ref() {
                Some(current) => (
                    current.revision,
                    Some(current.snapshot.checkpoint_id().clone()),
                    current.snapshot.engine_version().clone(),
                    current.snapshot.fingerprints().clone(),
                    current.snapshot.deadline_unix_ms(),
                ),
                None => (
                    SnapshotRevision::EMPTY,
                    None,
                    self.owner.engine_version.clone(),
                    self.owner.fingerprints.clone(),
                    self.deadline_unix_ms,
                ),
            };
        let snapshot = RunSnapshot::new(
            SnapshotCheckpoint::new(
                engine_version,
                self.run_id.clone(),
                self.lineage_id.clone(),
                next_checkpoint_id()?,
                parent,
            )?,
            fingerprints,
            continuation,
            report,
            deadline_unix_ms,
            resume_point,
        )?;
        let snapshot_bytes = serde_json::to_vec(&snapshot)?.len();
        snapshot
            .budget()
            .check_snapshot_bytes(snapshot_bytes, self.owner.runtime.run_budget())?;
        self.owner.renew(self.lease).await?;
        let revision = self
            .owner
            .store
            .compare_and_swap(self.lease, expected, snapshot.clone())
            .await
            .map_err(|error| map_store_error_for_run(error, &self.run_id))?;
        self.state = Some(DurableRun { revision, snapshot });

        let pause = matches!(boundary, CheckpointBoundary::Quiescent) && !can_progress;
        Ok(if pause {
            CheckpointControl::Pause
        } else {
            CheckpointControl::Continue
        })
    }

    fn resume_point(
        &self,
        state: EngineCheckpointState,
    ) -> Result<(ResumePoint, bool), DurableRunError> {
        match state {
            EngineCheckpointState::PendingTools(pending) => {
                let (index, target, response, prepared, completed, approvals, can_progress) =
                    pending.into_parts();
                let approvals = approvals
                    .into_iter()
                    .map(|approval| self.pending_approval(approval))
                    .collect::<Result<Vec<_>, _>>()?;
                let step = PendingStepSnapshot::new(
                    index, target, response, prepared, completed, approvals,
                );
                if step.pending_approvals().is_empty() {
                    Ok((ResumePoint::ReadyToDispatch(step), can_progress))
                } else {
                    Ok((ResumePoint::AwaitingApprovals(step), can_progress))
                }
            }
            EngineCheckpointState::AwaitingProvider(step) => {
                Ok((ResumePoint::AwaitingProvider(step), false))
            }
            EngineCheckpointState::ReadyForModel { next_step, target } => {
                Ok((ResumePoint::ReadyForModel { next_step, target }, true))
            }
            EngineCheckpointState::Terminal(terminal) => {
                Ok((ResumePoint::Terminal(terminal), false))
            }
        }
    }

    fn pending_approval(
        &self,
        approval: PendingApprovalCheckpoint,
    ) -> Result<PendingApprovalSnapshot, DurableRunError> {
        Ok(PendingApprovalSnapshot {
            approval_id: format!("approval:{}", approval.call().id()),
            call: approval.call().clone(),
            binding: approval.binding().clone(),
            claim_fingerprint: pending_approval_fingerprint_parts(
                approval.call().id(),
                approval.binding().fingerprint.as_str(),
                approval.canonical_arguments_digest(),
                &self.owner.fingerprints,
            )?,
            expires_at_unix_ms: None,
        })
    }
}

impl EngineCheckpointPort for DurableCheckpointPort<'_> {
    type Error = DurableRunError;

    const ENABLED: bool = true;

    fn max_concurrent_tools(&self, configured: usize) -> usize {
        configured
    }

    fn commit<'a>(
        &'a mut self,
        checkpoint: EngineCheckpoint,
    ) -> Pin<Box<dyn Future<Output = Result<CheckpointControl, Self::Error>> + Send + 'a>> {
        Box::pin(self.commit_candidate(checkpoint))
    }
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
    model_selector: Option<Arc<dyn StepModelSelector>>,
    projection_policy: ProjectionPolicy,
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
            .field("model_selector", &self.model_selector.is_some())
            .field("projection_policy", &self.projection_policy)
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
            model_selector: None,
            projection_policy: ProjectionPolicy::Strict,
            lease_ttl: DEFAULT_LEASE_TTL,
            engine_version: SnapshotEngineVersion::new(DURABLE_ENGINE_VERSION)?,
            fingerprints: SnapshotFingerprints {
                options: options_fingerprint,
                tool_catalog,
                approval_policy: approval_policy_fingerprint,
                model_selector: None,
                projection_policy: ProjectionPolicy::Strict,
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

    /// Install a versioned model-selection policy for later durable steps.
    pub fn with_model_selector<S>(mut self, selector: S) -> Self
    where
        S: StepModelSelector,
    {
        self.fingerprints.model_selector = Some(selector.identity().clone());
        self.model_selector = Some(Arc::new(selector));
        self
    }

    /// Install a shared versioned model-selection policy.
    pub fn with_shared_model_selector(mut self, selector: Arc<dyn StepModelSelector>) -> Self {
        self.fingerprints.model_selector = Some(selector.identity().clone());
        self.model_selector = Some(selector);
        self
    }

    /// Set the history-loss policy used for durable target transitions.
    pub fn with_projection_policy(mut self, policy: ProjectionPolicy) -> Self {
        self.projection_policy = policy;
        self.fingerprints.projection_policy = policy;
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
        let mut engine = StepEngine::establish(
            self.runtime.clone(),
            Arc::clone(&self.model),
            self.tools.clone(),
            request,
            self.step_options.clone(),
            options,
            self.outcome_policy,
            Arc::new(ExternalApprovalDecider::default()),
            self.model_selector.clone(),
            self.projection_policy,
            ToolHandling::Execute,
        )
        .await?;
        let mut checkpoint =
            DurableCheckpointPort::new(self, lease, run_id, lineage_id, deadline_unix_ms, None);
        engine.drive(&mut checkpoint).await?;
        checkpoint.into_state()
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
        if matches!(
            state.snapshot.resume_point(),
            ResumePoint::Terminal(_) | ResumePoint::AwaitingProvider(_)
        ) {
            return Ok(state);
        }

        let approvals = collect_approvals(resume.approvals)?;
        let verified_approvals = self.verify_approvals(&state.snapshot, &approvals).await?;
        let call_options = options_for_snapshot(&state.snapshot, options)?;
        let mut engine = StepEngine::resume(
            self.runtime.clone(),
            Arc::clone(&self.model),
            self.tools.clone(),
            self.step_options.clone(),
            call_options,
            self.outcome_policy,
            Arc::new(ExternalApprovalDecider::default()),
            self.model_selector.clone(),
            self.projection_policy,
            EngineResumeSeed {
                continuation: state.snapshot.continuation().clone(),
                report: state.snapshot.report().clone(),
                resume_point: state.snapshot.resume_point().clone(),
                verified_approvals,
            },
        )
        .map_err(map_engine_resume_error)?;
        let mut checkpoint = DurableCheckpointPort::new(
            self,
            lease,
            state.snapshot.run_id().clone(),
            state.snapshot.lineage_id().clone(),
            state.snapshot.deadline_unix_ms(),
            Some(state),
        );
        engine.drive(&mut checkpoint).await?;
        checkpoint.into_state()
    }

    async fn verify_approvals(
        &self,
        snapshot: &RunSnapshot,
        approvals: &BTreeMap<String, DurableApproval>,
    ) -> Result<BTreeMap<String, VerifiedApproval>, DurableRunError> {
        if approvals.is_empty() {
            return Ok(BTreeMap::new());
        }
        let runtime = self
            .approval_runtime
            .as_ref()
            .ok_or(DurableRunError::ApprovalVerificationNotConfigured)?;
        let pending = snapshot
            .resume_point()
            .pending_step()
            .ok_or(DurableRunError::Invariant {
                message: "approvals were supplied outside a pending tool step",
            })?;
        let mut batch = Vec::with_capacity(approvals.len());
        for (call_id, approval) in approvals {
            let prepared = pending
                .prepared()
                .iter()
                .find(|prepared| prepared.call().id() == call_id)
                .ok_or_else(|| DurableRunError::UnexpectedApproval {
                    call_id: call_id.clone(),
                })?;
            if !pending
                .pending_approvals()
                .iter()
                .any(|pending| pending.call.id() == call_id)
            {
                return Err(DurableRunError::UnexpectedApproval {
                    call_id: call_id.clone(),
                });
            }
            let request = self.restore_request(prepared)?;
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
            batch.push((call_id.clone(), &approval.envelope, context));
        }
        let inputs = batch
            .iter()
            .map(|(_, envelope, context)| ApprovalVerificationInput::new(envelope, context))
            .collect::<Vec<_>>();
        let verified = verify_and_consume_many_at_unix_ms(
            runtime.verifier.as_ref(),
            runtime.consume_store.as_ref(),
            &inputs,
            unix_millis()?,
        )
        .await?;
        Ok(batch
            .into_iter()
            .map(|(call_id, _, _)| call_id)
            .zip(verified)
            .collect())
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
            if !dispatched.contains(tool.call().id()) {
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
                call_id: prepared.call().id().to_owned(),
            });
        }
        request.validate()?;
        Ok(request)
    }

    fn successor_snapshot(
        &self,
        previous: &RunSnapshot,
        report: RunReport,
        resume_point: ResumePoint,
    ) -> Result<RunSnapshot, DurableRunError> {
        let mut continuation = previous.continuation().clone();
        continuation.messages = report.messages().to_vec();
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
            continuation,
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
        if snapshot.continuation().tools != self.tools.specs() {
            return Err(DurableRunError::IncompatibleSnapshot {
                field: "continuation_tools",
            });
        }
        Ok(())
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

fn pending_approval_fingerprint_parts(
    call_id: &str,
    binding_fingerprint: &str,
    canonical_arguments_digest: &str,
    fingerprints: &SnapshotFingerprints,
) -> Result<SnapshotFingerprint, DurableRunError> {
    let mut digest = Sha256::new();
    digest.update(b"siumai.pending-approval.v1\0");
    digest.update(call_id.as_bytes());
    digest.update([0]);
    digest.update(binding_fingerprint.as_bytes());
    digest.update([0]);
    digest.update(canonical_arguments_digest.as_bytes());
    digest.update([0]);
    digest.update(fingerprints.tool_catalog.as_str().as_bytes());
    digest.update([0]);
    digest.update(fingerprints.approval_policy.as_str().as_bytes());
    SnapshotFingerprint::new(format!("sha256:{:x}", digest.finalize())).map_err(Into::into)
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

fn map_engine_resume_error(error: EngineResumeError) -> DurableRunError {
    match error {
        EngineResumeError::SelectedModelTargetChanged { expected, actual } => {
            DurableRunError::SelectedModelTargetChanged { expected, actual }
        }
        EngineResumeError::RecoveryNotPermitted { call_id } => {
            DurableRunError::RecoveryNotPermitted { call_id }
        }
        EngineResumeError::FrozenRequestMismatch { call_id } => {
            DurableRunError::FrozenRequestMismatch { call_id }
        }
        EngineResumeError::Runtime(error) => DurableRunError::Runtime(error),
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
    #[error("durable selector changed the frozen model target from {expected:?} to {actual:?}")]
    SelectedModelTargetChanged {
        expected: Box<ModelTarget>,
        actual: Box<ModelTarget>,
    },
    #[error("approval verification is not configured")]
    ApprovalVerificationNotConfigured,
    #[error("approval for call `{call_id}` was supplied more than once")]
    DuplicateApproval { call_id: String },
    #[error("approval was supplied for non-pending call `{call_id}`")]
    UnexpectedApproval { call_id: String },
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
