//! Durable orchestration over the shared step engine and snapshot store.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error as StdError;
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
    CheckpointId, InitialSnapshotParts, LineageId, PendingApprovalSnapshot, PendingStepSnapshot,
    PreparedToolSnapshot, ResumePoint, RunId, RunLease, RunSnapshot, RunSnapshotError,
    RunSnapshotSuccessorError, RunStore, RunStoreError, SnapshotEngineVersion, SnapshotFingerprint,
    SnapshotFingerprints, SnapshotRevision, ToolExecutionEvent, ToolExecutionStatus,
    assemble_initial_snapshot, assemble_successor_snapshot,
};
use crate::tool::{
    ApprovalPolicy, EffectCertainty, ExternalApprovalDecider, PreparedVisibleToolCatalog,
    ToolExecutionError, ToolExecutionRequest, ToolSet, VisibleToolCatalogError,
    VisibleToolCatalogSource, prepare_visible_tool_catalog,
};
use crate::tool_loop::ToolOutcomePolicy;
use crate::{ModelTarget, ProjectionPolicy, RunReport, Runtime, StepModelSelector, StepOptions};

const DURABLE_EXECUTION_ABI: &str = "siumai-runtime-durable-v6";
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
    fingerprints: SnapshotFingerprints,
    lease: &'a mut RunLease,
    run_id: RunId,
    lineage_id: LineageId,
    deadline_unix_ms: Option<u64>,
    state: Option<DurableRun>,
}

impl<'a> DurableCheckpointPort<'a> {
    fn new(
        owner: &'a DurableToolLoop,
        fingerprints: SnapshotFingerprints,
        lease: &'a mut RunLease,
        run_id: RunId,
        lineage_id: LineageId,
        deadline_unix_ms: Option<u64>,
        state: Option<DurableRun>,
    ) -> Self {
        Self {
            owner,
            fingerprints,
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
        let checkpoint_id = next_checkpoint_id()?;
        let snapshot = match self.state.as_ref() {
            Some(current) => assemble_successor_snapshot(
                &current.snapshot,
                checkpoint_id,
                continuation,
                report,
                resume_point,
            )?,
            None => assemble_initial_snapshot(InitialSnapshotParts {
                engine_version: self.owner.engine_version.clone(),
                run_id: self.run_id.clone(),
                lineage_id: self.lineage_id.clone(),
                checkpoint_id,
                fingerprints: self.fingerprints.clone(),
                continuation,
                report,
                deadline_unix_ms: self.deadline_unix_ms,
                resume_point,
            })?,
        };
        let state = self
            .owner
            .commit_snapshot(self.lease, self.state.as_ref(), snapshot)
            .await?;
        self.state = Some(state);

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
                &self.fingerprints,
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
            engine_version: SnapshotEngineVersion::new(DURABLE_EXECUTION_ABI)?,
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
        let fingerprints = self.fingerprints_for_catalog(engine.visible_tool_catalog())?;
        let mut checkpoint = DurableCheckpointPort::new(
            self,
            fingerprints,
            lease,
            run_id,
            lineage_id,
            deadline_unix_ms,
            None,
        );
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
        let visible_catalog = prepare_visible_tool_catalog(
            VisibleToolCatalogSource::Canonical(stored.snapshot().continuation().tools.clone()),
            &self.tools,
        )
        .map_err(map_visible_catalog_error)?;
        let fingerprints = self.fingerprints_for_catalog(&visible_catalog)?;
        self.ensure_compatible(stored.snapshot(), &fingerprints)?;

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
                visible_tools: visible_catalog,
                report: state.snapshot.report().clone(),
                resume_point: state.snapshot.resume_point().clone(),
                verified_approvals,
            },
        )
        .map_err(map_engine_resume_error)?;
        let mut checkpoint = DurableCheckpointPort::new(
            self,
            fingerprints,
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
        self.commit_snapshot(lease, Some(&state), next).await
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
        Ok(assemble_successor_snapshot(
            previous,
            next_checkpoint_id()?,
            continuation,
            report,
            resume_point,
        )?)
    }

    async fn commit_snapshot(
        &self,
        lease: &mut RunLease,
        previous: Option<&DurableRun>,
        snapshot: RunSnapshot,
    ) -> Result<DurableRun, DurableRunError> {
        let expected = previous.map_or(SnapshotRevision::EMPTY, DurableRun::revision);
        let snapshot_bytes = serde_json::to_vec(&snapshot)?.len();
        snapshot
            .budget()
            .check_snapshot_bytes(snapshot_bytes, self.runtime.run_budget())?;
        self.renew(lease).await?;
        if let Some(previous) = previous {
            previous.snapshot.validate_successor(&snapshot)?;
        }
        let revision = self
            .store
            .compare_and_swap(lease, expected, snapshot.clone())
            .await
            .map_err(|error| map_store_error_for_run(error, lease.run_id()))?;
        Ok(DurableRun { revision, snapshot })
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

    fn fingerprints_for_catalog(
        &self,
        catalog: &PreparedVisibleToolCatalog,
    ) -> Result<SnapshotFingerprints, DurableRunError> {
        let mut fingerprints = self.fingerprints.clone();
        fingerprints.tool_catalog = SnapshotFingerprint::new(catalog.fingerprint().as_str())?;
        Ok(fingerprints)
    }

    fn ensure_compatible(
        &self,
        snapshot: &RunSnapshot,
        fingerprints: &SnapshotFingerprints,
    ) -> Result<(), DurableRunError> {
        if snapshot.engine_version() != &self.engine_version {
            return Err(DurableRunError::IncompatibleSnapshot {
                field: "engine_version",
            });
        }
        if snapshot.fingerprints() != fingerprints {
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
    SnapshotFingerprint::new(sha256_fingerprint(&digest.finalize())).map_err(Into::into)
}

fn sha256_fingerprint(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity("sha256:".len() + bytes.len() * 2);
    output.push_str("sha256:");
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
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

fn map_visible_catalog_error(error: VisibleToolCatalogError) -> DurableRunError {
    match error {
        VisibleToolCatalogError::DuplicateName { name } => {
            let source = VisibleToolCatalogError::DuplicateName { name: name.clone() };
            DurableRunError::VisibleToolCatalogDuplicateName {
                name,
                source: Box::new(source),
            }
        }
        VisibleToolCatalogError::TrustedBindingCollision { name } => {
            let source = VisibleToolCatalogError::TrustedBindingCollision { name: name.clone() };
            DurableRunError::VisibleToolCatalogTrustedBindingCollision {
                name,
                source: Box::new(source),
            }
        }
        VisibleToolCatalogError::LocalCatalogMismatch => {
            DurableRunError::VisibleToolCatalogLocalSuffixMismatch {
                source: Box::new(VisibleToolCatalogError::LocalCatalogMismatch),
            }
        }
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
    #[error("invalid durable continuation tool catalog: {source}")]
    VisibleToolCatalogDuplicateName {
        name: String,
        #[source]
        source: Box<dyn StdError + Send + Sync + 'static>,
    },
    #[error("invalid durable continuation tool catalog: {source}")]
    VisibleToolCatalogTrustedBindingCollision {
        name: String,
        #[source]
        source: Box<dyn StdError + Send + Sync + 'static>,
    },
    #[error("invalid durable continuation tool catalog: {source}")]
    VisibleToolCatalogLocalSuffixMismatch {
        #[source]
        source: Box<dyn StdError + Send + Sync + 'static>,
    },
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
    #[error("invalid durable snapshot successor: {0}")]
    InvalidSuccessor(#[from] RunSnapshotSuccessorError),
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use siumai_core::{
        ContentPart, LanguageCallError, LanguageCompletionReason, LanguageResponse, LanguageStream,
        Message, Model, ModelDescriptor, ModelFamily, ModelId, OpaqueProviderItem, ProtocolId,
        ProviderId, ProviderProvenance, ReplayDomain, ReplayDomainId, ToolBindingIdentity,
        ToolCall, ToolOutcome, ToolSpec, Usage,
    };

    use super::*;
    use crate::StepRecord;
    use crate::snapshot::{
        InMemoryRunStore, SnapshotCheckpoint, SnapshotReason, SnapshotTerminal, StoredRun,
        ToolExecutionLog,
    };
    use crate::tool::{RecoveryPolicy, ToolBinding, ToolExecutionAttempt};

    struct TestModel {
        descriptor: ModelDescriptor,
    }

    impl TestModel {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                descriptor: ModelDescriptor::new(
                    ProviderId::new("durable-successor-test").unwrap(),
                    ModelId::new("test-model").unwrap(),
                    ModelFamily::Language,
                )
                .with_protocol(ProtocolId::new("durable.test").unwrap())
                .with_replay_domain(ReplayDomain::custom(
                    ReplayDomainId::new("durable-successor-test").unwrap(),
                )),
            })
        }
    }

    impl Model for TestModel {
        fn descriptor(&self) -> &ModelDescriptor {
            &self.descriptor
        }
    }

    #[async_trait]
    impl LanguageModel for TestModel {
        async fn generate(
            &self,
            _request: LanguageRequest,
            _options: CallOptions,
        ) -> Result<siumai_core::LanguageResponse, LanguageCallError> {
            Err(Error::protocol_violation("test model must not be called").into())
        }

        async fn stream(
            &self,
            _request: LanguageRequest,
            _options: CallOptions,
        ) -> Result<LanguageStream, Error> {
            Err(Error::protocol_violation("test model must not be called"))
        }
    }

    #[derive(Debug)]
    struct CountingStore {
        inner: InMemoryRunStore,
        cas_calls: AtomicUsize,
    }

    impl CountingStore {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                inner: InMemoryRunStore::new(),
                cas_calls: AtomicUsize::new(0),
            })
        }

        fn reset_cas_calls(&self) {
            self.cas_calls.store(0, Ordering::SeqCst);
        }

        fn cas_calls(&self) -> usize {
            self.cas_calls.load(Ordering::SeqCst)
        }
    }

    impl RunStore for CountingStore {
        fn acquire<'a>(
            &'a self,
            run_id: &'a RunId,
            ttl: Duration,
        ) -> crate::snapshot::RunStoreFuture<'a, RunLease> {
            self.inner.acquire(run_id, ttl)
        }

        fn load<'a>(
            &'a self,
            lease: &'a RunLease,
        ) -> crate::snapshot::RunStoreFuture<'a, Option<StoredRun>> {
            self.inner.load(lease)
        }

        fn compare_and_swap<'a>(
            &'a self,
            lease: &'a RunLease,
            expected: SnapshotRevision,
            snapshot: RunSnapshot,
        ) -> crate::snapshot::RunStoreFuture<'a, SnapshotRevision> {
            self.cas_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.compare_and_swap(lease, expected, snapshot)
        }

        fn renew<'a>(
            &'a self,
            lease: &'a mut RunLease,
            ttl: Duration,
        ) -> crate::snapshot::RunStoreFuture<'a, ()> {
            self.inner.renew(lease, ttl)
        }

        fn release<'a>(&'a self, lease: RunLease) -> crate::snapshot::RunStoreFuture<'a, ()> {
            self.inner.release(lease)
        }
    }

    fn test_loop(store: Arc<dyn RunStore>) -> DurableToolLoop {
        test_loop_with_tools(store, ToolSet::default())
    }

    fn test_loop_with_tools(store: Arc<dyn RunStore>, tools: ToolSet) -> DurableToolLoop {
        DurableToolLoop::new(
            TestModel::new(),
            tools,
            store,
            SnapshotFingerprint::new("sha256:test-options").unwrap(),
            SnapshotFingerprint::new("sha256:test-approval").unwrap(),
        )
        .unwrap()
    }

    fn test_binding(name: &str) -> ToolBinding {
        ToolBinding::from_fn(
            ToolSpec::new(
                name,
                Some(format!("{name} test tool")),
                serde_json::json!({"type": "object"}),
            )
            .unwrap(),
            "v1",
            |_| Ok(()),
            |_| async {
                Ok(ToolOutcome::Success {
                    value: serde_json::Value::Null,
                })
            },
        )
        .unwrap()
    }

    fn test_snapshot(
        loop_: &DurableToolLoop,
        checkpoint_id: &str,
        parent_checkpoint_id: Option<&str>,
    ) -> RunSnapshot {
        let target = ModelTarget::from_model(loop_.model.as_ref());
        test_snapshot_with_report(
            loop_,
            checkpoint_id,
            parent_checkpoint_id,
            RunReport::new(target, vec![Message::user("test")]),
            ResumePoint::ReadyForModel {
                next_step: 0,
                target: ModelTarget::from_model(loop_.model.as_ref()),
            },
        )
    }

    fn test_snapshot_with_report(
        loop_: &DurableToolLoop,
        checkpoint_id: &str,
        parent_checkpoint_id: Option<&str>,
        report: RunReport,
        resume_point: ResumePoint,
    ) -> RunSnapshot {
        test_snapshot_with_report_for_run(
            loop_,
            "run-successor-test",
            checkpoint_id,
            parent_checkpoint_id,
            report,
            resume_point,
        )
    }

    fn test_snapshot_with_report_for_run(
        loop_: &DurableToolLoop,
        run_id: &str,
        checkpoint_id: &str,
        parent_checkpoint_id: Option<&str>,
        report: RunReport,
        resume_point: ResumePoint,
    ) -> RunSnapshot {
        let messages = report.messages().to_vec();
        RunSnapshot::new(
            SnapshotCheckpoint::new(
                loop_.engine_version.clone(),
                RunId::new(run_id).unwrap(),
                LineageId::new("lineage-successor-test").unwrap(),
                CheckpointId::new(checkpoint_id).unwrap(),
                parent_checkpoint_id.map(|parent| CheckpointId::new(parent).unwrap()),
            )
            .unwrap(),
            loop_.fingerprints.clone(),
            LanguageRequest::new(messages.clone()),
            report,
            None,
            resume_point,
        )
        .unwrap()
    }

    fn completed_response(total_tokens: u64) -> LanguageResponse {
        LanguageResponse::completed(
            vec![ContentPart::Text {
                text: "done".to_string(),
            }],
            LanguageCompletionReason::Stop,
            Usage::default().with_total_tokens(total_tokens),
        )
        .unwrap()
    }

    fn prepared_tool() -> PreparedToolSnapshot {
        let call = ToolCall::local("call-1", "write_record", serde_json::json!({})).unwrap();
        PreparedToolSnapshot::new(
            0,
            call,
            ToolBindingIdentity {
                name: "write_record".to_string(),
                fingerprint: "binding-v1".to_string(),
            },
            RecoveryPolicy::NeverReplay,
            None,
            ToolExecutionAttempt::INITIAL,
        )
    }

    fn tool_response() -> LanguageResponse {
        LanguageResponse::completed(
            vec![ContentPart::ToolCall(prepared_tool().call().clone())],
            LanguageCompletionReason::ToolCalls,
            Usage::default(),
        )
        .unwrap()
    }

    fn prepared_log() -> ToolExecutionLog {
        let mut log = ToolExecutionLog::new();
        log.append(ToolExecutionEvent::prepared(0, 1, 0, prepared_tool()))
            .unwrap();
        log
    }

    fn dispatched_log() -> ToolExecutionLog {
        let mut log = prepared_log();
        log.append(ToolExecutionEvent::dispatched(
            1,
            2,
            "call-1",
            ToolExecutionAttempt::INITIAL,
            None,
        ))
        .unwrap();
        log
    }

    fn provider_item(loop_: &DurableToolLoop) -> OpaqueProviderItem {
        let target = ModelTarget::from_model(loop_.model.as_ref());
        OpaqueProviderItem::new(
            ProviderProvenance::from_scope(target.scope(), target.model().clone()).unwrap(),
            "test.deferred",
            serde_json::json!({"state": true}),
        )
        .unwrap()
    }

    #[test]
    fn visible_catalog_errors_remain_typed_and_source_preserving() {
        let duplicate = map_visible_catalog_error(VisibleToolCatalogError::DuplicateName {
            name: "duplicate".to_string(),
        });
        assert!(matches!(
            &duplicate,
            DurableRunError::VisibleToolCatalogDuplicateName { name, .. }
                if name == "duplicate"
        ));
        assert_eq!(
            std::error::Error::source(&duplicate).unwrap().to_string(),
            "model-visible tool `duplicate` is defined more than once"
        );

        let collision =
            map_visible_catalog_error(VisibleToolCatalogError::TrustedBindingCollision {
                name: "collision".to_string(),
            });
        assert!(matches!(
            &collision,
            DurableRunError::VisibleToolCatalogTrustedBindingCollision { name, .. }
                if name == "collision"
        ));
        assert_eq!(
            std::error::Error::source(&collision).unwrap().to_string(),
            "model-visible tool `collision` conflicts with a trusted local binding"
        );

        let mismatch = map_visible_catalog_error(VisibleToolCatalogError::LocalCatalogMismatch);
        assert!(matches!(
            &mismatch,
            DurableRunError::VisibleToolCatalogLocalSuffixMismatch { .. }
        ));
        assert_eq!(
            std::error::Error::source(&mismatch).unwrap().to_string(),
            "restored model-visible tool catalog does not contain the exact local binding suffix"
        );
    }

    #[tokio::test]
    async fn resume_preserves_the_typed_visible_catalog_mismatch() {
        let store = CountingStore::new();
        let tools = ToolSet::from_bindings([test_binding("write_record")]).unwrap();
        let loop_ = test_loop_with_tools(store.clone(), tools);
        let previous = test_snapshot(&loop_, "checkpoint-1", None);
        let lease = store
            .acquire(previous.run_id(), Duration::from_secs(30))
            .await
            .unwrap();
        store
            .compare_and_swap(&lease, SnapshotRevision::EMPTY, previous.clone())
            .await
            .unwrap();
        store.release(lease).await.unwrap();
        store.reset_cas_calls();

        let error = loop_
            .resume(
                previous.run_id(),
                DurableResume::new(),
                CallOptions::default(),
            )
            .await
            .unwrap_err();

        assert!(matches!(
            &error,
            DurableRunError::VisibleToolCatalogLocalSuffixMismatch { .. }
        ));
        assert_eq!(
            std::error::Error::source(&error).unwrap().to_string(),
            "restored model-visible tool catalog does not contain the exact local binding suffix"
        );
        assert_eq!(store.cas_calls(), 0);
    }

    #[tokio::test]
    async fn runtime_rejects_invalid_successor_before_store_cas() {
        let store = CountingStore::new();
        let loop_ = test_loop(store.clone());
        let previous = test_snapshot(&loop_, "checkpoint-1", None);
        let mut lease = store
            .acquire(previous.run_id(), Duration::from_secs(30))
            .await
            .unwrap();
        let revision = store
            .compare_and_swap(&lease, SnapshotRevision::EMPTY, previous.clone())
            .await
            .unwrap();
        store.reset_cas_calls();
        let invalid = test_snapshot(&loop_, "checkpoint-2", None);

        let previous = DurableRun {
            revision,
            snapshot: previous,
        };
        let error = loop_
            .commit_snapshot(&mut lease, Some(&previous), invalid)
            .await
            .unwrap_err();

        assert!(matches!(
            error,
            DurableRunError::InvalidSuccessor(
                RunSnapshotSuccessorError::ParentCheckpointMismatch { .. }
            )
        ));
        assert_eq!(store.cas_calls(), 0);
        store.release(lease).await.unwrap();
    }

    #[derive(Debug, Clone, Copy)]
    enum SuccessorRegression {
        Step,
        History,
        Budget,
        Usage,
        Provider,
        ExecutionLog,
    }

    impl SuccessorRegression {
        fn expected(self) -> RunSnapshotSuccessorError {
            match self {
                Self::Step => RunSnapshotSuccessorError::StepHistoryRegression,
                Self::History => RunSnapshotSuccessorError::MessageHistoryRegression,
                Self::Budget => RunSnapshotSuccessorError::BudgetRegression {
                    dimension: "model_steps",
                    previous: 1,
                    next: 0,
                },
                Self::Usage => RunSnapshotSuccessorError::UsageRegression {
                    dimension: "total_tokens",
                    previous: 10,
                    next: 5,
                },
                Self::Provider => RunSnapshotSuccessorError::ProviderHistoryRegression,
                Self::ExecutionLog => RunSnapshotSuccessorError::ExecutionLogRegression,
            }
        }
    }

    #[tokio::test]
    async fn runtime_rejects_representative_successor_regressions_before_store_cas() {
        let cases = [
            SuccessorRegression::Step,
            SuccessorRegression::History,
            SuccessorRegression::Budget,
            SuccessorRegression::Usage,
            SuccessorRegression::Provider,
            SuccessorRegression::ExecutionLog,
        ];

        for case in cases {
            let store = CountingStore::new();
            let loop_ = test_loop(store.clone());
            let target = ModelTarget::from_model(loop_.model.as_ref());
            let base_messages = vec![Message::user("test")];
            let mut previous_report = RunReport::new(target.clone(), base_messages.clone());
            let mut next_report = RunReport::new(target.clone(), base_messages.clone());

            match case {
                SuccessorRegression::Step => {
                    previous_report.accumulate_usage(&Usage::default());
                    previous_report.steps_mut().push(StepRecord::new(
                        0,
                        target.clone(),
                        completed_response(0),
                        Vec::new(),
                    ));
                }
                SuccessorRegression::History => {
                    previous_report
                        .messages_mut()
                        .push(Message::user("later message"));
                }
                SuccessorRegression::Budget => previous_report
                    .budget_mut()
                    .charge_model_step(loop_.runtime.run_budget())
                    .unwrap(),
                SuccessorRegression::Usage => {
                    previous_report.accumulate_usage(&Usage::default().with_total_tokens(10));
                    next_report.accumulate_usage(&Usage::default().with_total_tokens(5));
                }
                SuccessorRegression::Provider => {
                    let item = provider_item(&loop_);
                    previous_report.observe_provider_deferred("provider-state-1", &item);
                }
                SuccessorRegression::ExecutionLog => {
                    *previous_report.execution_log_mut() = prepared_log();
                }
            }

            let previous_next_step = u32::try_from(previous_report.steps().len()).unwrap();
            let next_step = u32::try_from(next_report.steps().len()).unwrap();
            let previous = test_snapshot_with_report(
                &loop_,
                "checkpoint-1",
                None,
                previous_report,
                ResumePoint::ReadyForModel {
                    next_step: previous_next_step,
                    target: target.clone(),
                },
            );
            let candidate = test_snapshot_with_report(
                &loop_,
                "checkpoint-2",
                Some("checkpoint-1"),
                next_report,
                ResumePoint::ReadyForModel { next_step, target },
            );
            let mut lease = store
                .acquire(previous.run_id(), Duration::from_secs(30))
                .await
                .unwrap();
            let revision = store
                .compare_and_swap(&lease, SnapshotRevision::EMPTY, previous.clone())
                .await
                .unwrap();
            store.reset_cas_calls();
            let state = DurableRun {
                revision,
                snapshot: previous,
            };

            let error = loop_
                .commit_snapshot(&mut lease, Some(&state), candidate)
                .await
                .unwrap_err();

            let DurableRunError::InvalidSuccessor(actual) = error else {
                panic!("{case:?}: expected invalid successor, got {error:?}");
            };
            assert_eq!(actual, case.expected(), "{case:?}");
            assert_eq!(store.cas_calls(), 0, "{case:?}");
            store.release(lease).await.unwrap();
        }
    }

    #[tokio::test]
    async fn recovery_commits_a_nonterminal_dispatch_through_the_shared_gate() {
        let store = CountingStore::new();
        let loop_ = test_loop(store.clone());
        let target = ModelTarget::from_model(loop_.model.as_ref());
        let messages = vec![Message::user("test")];
        let mut report = RunReport::new(target.clone(), messages);
        report.accumulate_usage(&Usage::default());
        *report.execution_log_mut() = dispatched_log();
        let pending = PendingStepSnapshot::new(
            0,
            target,
            tool_response(),
            vec![prepared_tool()],
            Vec::new(),
            Vec::new(),
        );
        let previous = test_snapshot_with_report(
            &loop_,
            "checkpoint-1",
            None,
            report,
            ResumePoint::ReadyToDispatch(pending),
        );
        let mut lease = store
            .acquire(previous.run_id(), Duration::from_secs(30))
            .await
            .unwrap();
        let revision = store
            .compare_and_swap(&lease, SnapshotRevision::EMPTY, previous.clone())
            .await
            .unwrap();
        store.reset_cas_calls();
        let state = DurableRun {
            revision,
            snapshot: previous,
        };

        let recovered = loop_
            .recover_if_needed(&mut lease, state, IndeterminateRecoveryPolicy::Halt)
            .await
            .unwrap();

        assert_eq!(store.cas_calls(), 1);
        assert!(matches!(
            recovered.snapshot.resume_point(),
            ResumePoint::Terminal(SnapshotTerminal::Indeterminate { .. })
        ));
        store.release(lease).await.unwrap();
    }

    #[tokio::test]
    async fn recovery_rejects_a_tampered_successor_before_store_cas() {
        let store = CountingStore::new();
        let loop_ = test_loop(store.clone());
        let target = ModelTarget::from_model(loop_.model.as_ref());
        let mut report = RunReport::new(target.clone(), vec![Message::user("test")]);
        report.accumulate_usage(&Usage::default());
        *report.execution_log_mut() = dispatched_log();
        let pending = PendingStepSnapshot::new(
            0,
            target,
            tool_response(),
            vec![prepared_tool()],
            Vec::new(),
            Vec::new(),
        );
        let previous = test_snapshot_with_report(
            &loop_,
            "checkpoint-1",
            None,
            report,
            ResumePoint::ReadyToDispatch(pending),
        );
        let mut lease = store
            .acquire(previous.run_id(), Duration::from_secs(30))
            .await
            .unwrap();
        let revision = store
            .compare_and_swap(&lease, SnapshotRevision::EMPTY, previous.clone())
            .await
            .unwrap();
        store.reset_cas_calls();

        let state = DurableRun {
            revision,
            snapshot: previous.clone(),
        };
        let recovered = previous.recovered_for_resume(3).unwrap();
        let candidate = loop_
            .successor_snapshot(
                &state.snapshot,
                recovered.report().clone(),
                recovered.resume_point().clone(),
            )
            .unwrap();
        let mut encoded = serde_json::to_value(candidate).unwrap();
        encoded["checkpoint"]["parent_checkpoint_id"] = serde_json::Value::Null;
        let invalid: RunSnapshot = serde_json::from_value(encoded).unwrap();

        let error = loop_
            .commit_snapshot(&mut lease, Some(&state), invalid)
            .await
            .unwrap_err();

        assert!(matches!(
            error,
            DurableRunError::InvalidSuccessor(
                RunSnapshotSuccessorError::ParentCheckpointMismatch { .. }
            )
        ));
        assert_eq!(store.cas_calls(), 0);
        store.release(lease).await.unwrap();
    }

    #[tokio::test]
    async fn runtime_defers_run_identity_and_terminal_write_rejection_to_store() {
        let store = CountingStore::new();
        let loop_ = test_loop(store.clone());
        let previous = test_snapshot(&loop_, "checkpoint-1", None);
        let mut lease = store
            .acquire(previous.run_id(), Duration::from_secs(30))
            .await
            .unwrap();
        let revision = store
            .compare_and_swap(&lease, SnapshotRevision::EMPTY, previous.clone())
            .await
            .unwrap();
        store.reset_cas_calls();
        let candidate = test_snapshot_with_report_for_run(
            &loop_,
            "another-run",
            "checkpoint-2",
            Some("checkpoint-1"),
            previous.report().clone(),
            previous.resume_point().clone(),
        );
        let state = DurableRun {
            revision,
            snapshot: previous,
        };

        let error = loop_
            .commit_snapshot(&mut lease, Some(&state), candidate)
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            DurableRunError::Store(RunStoreError::RunIdMismatch { .. })
        ));
        assert_eq!(store.cas_calls(), 1);
        store.release(lease).await.unwrap();

        let store = CountingStore::new();
        let loop_ = test_loop(store.clone());
        let target = ModelTarget::from_model(loop_.model.as_ref());
        let terminal = test_snapshot_with_report(
            &loop_,
            "checkpoint-terminal",
            None,
            RunReport::new(target, vec![Message::user("test")]),
            ResumePoint::Terminal(SnapshotTerminal::Failed {
                reason: SnapshotReason::new("worker_failed", None).unwrap(),
                partial: None,
            }),
        );
        let mut lease = store
            .acquire(terminal.run_id(), Duration::from_secs(30))
            .await
            .unwrap();
        let revision = store
            .compare_and_swap(&lease, SnapshotRevision::EMPTY, terminal.clone())
            .await
            .unwrap();
        store.reset_cas_calls();
        let candidate = test_snapshot(
            &loop_,
            "checkpoint-after-terminal",
            Some("checkpoint-terminal"),
        );
        let state = DurableRun {
            revision,
            snapshot: terminal,
        };

        let error = loop_
            .commit_snapshot(&mut lease, Some(&state), candidate)
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            DurableRunError::Store(RunStoreError::RunAlreadyTerminal)
        ));
        assert_eq!(store.cas_calls(), 1);
        store.release(lease).await.unwrap();
    }
}
