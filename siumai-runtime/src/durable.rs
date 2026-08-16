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
    CheckpointId, CheckpointIntent, CheckpointWriteError, CheckpointWriter,
    InitialCheckpointIntent, LineageId, PendingApprovalSnapshot, PendingStepSnapshot, ResumePoint,
    RunId, RunLease, RunSnapshot, RunSnapshotError, RunSnapshotSuccessorError, RunStore,
    RunStoreError, SnapshotEngineVersion, SnapshotFingerprint, SnapshotFingerprints,
    SnapshotRevision, ToolExecutionStatus,
};
use crate::tool::{
    EffectCertainty, ExternalApprovalDecider, PreparedVisibleToolCatalog, ToolExecutionError,
    ToolJournalError, ToolSet, VisibleToolCatalogError, VisibleToolCatalogSource,
    prepare_visible_tool_catalog,
};
use crate::tool_loop::ToolOutcomePolicy;
use crate::{ModelTarget, ProjectionPolicy, RunReport, Runtime, StepModelSelector, StepOptions};

const DURABLE_EXECUTION_ABI: &str = "siumai-runtime-durable-v7";
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
        let expected = self
            .state
            .as_ref()
            .map_or(SnapshotRevision::EMPTY, DurableRun::revision);
        let intent = match self.state.as_ref() {
            Some(current) => CheckpointIntent::successor(
                &current.snapshot,
                checkpoint_id,
                continuation,
                report,
                resume_point,
            ),
            None => CheckpointIntent::initial(InitialCheckpointIntent {
                engine_version: self.owner.engine_version.clone(),
                run_id: self.run_id.clone(),
                lineage_id: self.lineage_id.clone(),
                checkpoint_id,
                fingerprints: self.fingerprints.clone(),
                continuation,
                report,
                deadline_unix_ms: self.deadline_unix_ms,
                resume_point,
            }),
        };
        let state = self
            .owner
            .commit_checkpoint(self.lease, expected, intent)
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
                    Ok((ResumePoint::ready_to_dispatch(step), can_progress))
                } else {
                    Ok((ResumePoint::awaiting_approvals(step), can_progress))
                }
            }
            EngineCheckpointState::AwaitingProvider(step) => {
                Ok((ResumePoint::awaiting_provider(step), false))
            }
            EngineCheckpointState::ReadyForModel { next_step, target } => {
                Ok((ResumePoint::ready_for_model(next_step, target), true))
            }
            EngineCheckpointState::Terminal(terminal) => {
                Ok((ResumePoint::terminal_state(terminal), false))
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
            state.snapshot.resume_point().kind(),
            crate::snapshot::ResumePointKind::Terminal
                | crate::snapshot::ResumePointKind::AwaitingProvider
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
                .any(|pending| pending.call().id() == call_id)
            {
                return Err(DurableRunError::UnexpectedApproval {
                    call_id: call_id.clone(),
                });
            }
            let request = snapshot
                .report()
                .tool_journal()
                .restore(&self.tools, prepared)
                .map_err(|error| map_journal_error(prepared.call().id(), error))?;
            let context = TrustContext::builder(approval.identity.clone())
                .model_target(snapshot.target().clone())
                .run_id(snapshot.run_id().clone())
                .lineage_id(snapshot.lineage_id().clone())
                .checkpoint_id(snapshot.checkpoint_id().clone())
                .execution_owner(request.owner().clone())
                .binding_identity(request.binding_identity().clone())
                .tool_call_id(request.call_id())
                .canonical_arguments_digest(request.canonical_arguments_digest())
                .catalog_fingerprint(snapshot.fingerprints().tool_catalog().as_str())
                .policy_fingerprint(snapshot.fingerprints().approval_policy().as_str())
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

        let mut recovered_report = state.snapshot.report().clone();
        let recovered_count = recovered_report
            .tool_journal_mut()
            .recover_dispatched()
            .map_err(|error| map_journal_error("<recovered>", error))?;
        debug_assert!(recovered_count > 0);
        for _ in state.snapshot.pending_approvals() {
            recovered_report.budget_mut().release_pending_approval();
        }
        let recovered_resume =
            ResumePoint::terminal_state(crate::snapshot::SnapshotTerminal::indeterminate(
                crate::snapshot::SnapshotReason::runtime_code("dispatch_outcome_unknown"),
            ));
        let (report, resume_point) = match policy {
            IndeterminateRecoveryPolicy::Halt => (recovered_report, recovered_resume),
            IndeterminateRecoveryPolicy::RetryStable => self.retry_recovered_state(
                &state.snapshot,
                recovered_report,
                recovered_resume,
                &dispatched,
            )?,
        };
        let mut continuation = state.snapshot.continuation().clone();
        continuation.messages = report.messages().to_vec();
        let intent = CheckpointIntent::successor(
            &state.snapshot,
            next_checkpoint_id()?,
            continuation,
            report,
            resume_point,
        );
        self.commit_checkpoint(lease, state.revision, intent).await
    }

    fn retry_recovered_state(
        &self,
        previous: &RunSnapshot,
        recovered_report: RunReport,
        recovered_resume: ResumePoint,
        dispatched: &BTreeSet<String>,
    ) -> Result<(RunReport, ResumePoint), DurableRunError> {
        let Some(previous_step) = previous.resume_point().pending_step() else {
            return Ok((recovered_report, recovered_resume));
        };
        let mut prepared = previous_step.prepared().to_vec();
        let mut journal = recovered_report.tool_journal().clone();

        for tool in &mut prepared {
            if !dispatched.contains(tool.call().id()) {
                continue;
            }
            let Some((next_prepared, _request)) = journal
                .retry(
                    &self.tools,
                    previous_step.index(),
                    tool,
                    EffectCertainty::Indeterminate,
                )
                .map_err(|error| map_journal_error(tool.call().id(), error))?
            else {
                return Ok((recovered_report, recovered_resume));
            };
            *tool = next_prepared;
        }
        let mut report = recovered_report;
        report.replace_tool_journal(journal);
        for _ in previous_step.pending_approvals() {
            report
                .budget_mut()
                .reserve_pending_approval(self.runtime.run_budget())?;
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
            ResumePoint::ready_to_dispatch(step)
        } else {
            ResumePoint::awaiting_approvals(step)
        };
        Ok((report, resume_point))
    }

    async fn commit_checkpoint(
        &self,
        lease: &mut RunLease,
        expected: SnapshotRevision,
        intent: CheckpointIntent<'_>,
    ) -> Result<DurableRun, DurableRunError> {
        let run_id = lease.run_id().clone();
        let committed = CheckpointWriter::new(
            self.store.as_ref(),
            lease,
            self.lease_ttl,
            self.runtime.run_budget(),
        )
        .commit(expected, intent)
        .await
        .map_err(|error| map_checkpoint_write_error(error, &run_id))?;
        let revision = committed.revision();
        let snapshot = committed.into_snapshot();
        Ok(DurableRun { revision, snapshot })
    }

    async fn acquire(&self, run_id: &RunId) -> Result<RunLease, DurableRunError> {
        self.store
            .acquire(run_id, self.lease_ttl)
            .await
            .map_err(map_store_error)
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

fn map_checkpoint_write_error(error: CheckpointWriteError, run_id: &RunId) -> DurableRunError {
    match error {
        CheckpointWriteError::Snapshot(error) => DurableRunError::Snapshot(error),
        CheckpointWriteError::InvalidSuccessor(error) => DurableRunError::InvalidSuccessor(error),
        CheckpointWriteError::Budget(error) => DurableRunError::Budget(error),
        CheckpointWriteError::Serialization(error) => DurableRunError::Serialization(error),
        CheckpointWriteError::Store(error) => map_store_error_for_run(error, run_id),
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

fn map_journal_error(call_id: &str, error: ToolJournalError) -> DurableRunError {
    match error {
        ToolJournalError::Clock => DurableRunError::Clock,
        ToolJournalError::RecoveryNotPermitted => DurableRunError::RecoveryNotPermitted {
            call_id: call_id.to_owned(),
        },
        ToolJournalError::FrozenWorkMismatch | ToolJournalError::InvalidRestoredRequest => {
            DurableRunError::FrozenRequestMismatch {
                call_id: call_id.to_owned(),
            }
        }
        ToolJournalError::Transition(error) => DurableRunError::ExecutionTransition(error),
        ToolJournalError::Ordinal => DurableRunError::Invariant {
            message: "durable tool ordinal cannot be represented",
        },
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
    use crate::snapshot::{
        InMemoryRunStore, PendingApprovalSnapshot, PendingProviderStepSnapshot,
        PendingStepSnapshot, PreparedToolSnapshot, SnapshotCheckpoint, SnapshotReason,
        SnapshotTerminal, StoredRun, ToolExecutionEvent, ToolExecutionLog,
    };
    use crate::tool::{RecoveryPolicy, ToolBinding, ToolExecutionAttempt, ToolJournal};
    use crate::{RunBudget, StepRecord};

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
            ResumePoint::ready_for_model(0, ModelTarget::from_model(loop_.model.as_ref())),
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

    fn observe_provider_state(
        report: &mut RunReport,
        target: &ModelTarget,
        correlation_id: &str,
        item: &OpaqueProviderItem,
    ) {
        let mut step = report.provider_deferred_ledger().begin_step(0, target);
        step.observe(correlation_id, item);
        let (ledger, _, _, _) = step
            .finish_completed(report.provider_deferred_ledger())
            .unwrap()
            .into_parts();
        report.replace_provider_deferred_ledger(ledger);
    }

    fn successor_intent<'a>(
        previous: &'a RunSnapshot,
        candidate: RunSnapshot,
    ) -> CheckpointIntent<'a> {
        CheckpointIntent::successor(
            previous,
            candidate.checkpoint_id().clone(),
            candidate.continuation().clone(),
            candidate.report().clone(),
            candidate.resume_point().clone(),
        )
    }

    fn initial_intent(candidate: RunSnapshot) -> CheckpointIntent<'static> {
        CheckpointIntent::initial(InitialCheckpointIntent {
            engine_version: candidate.engine_version().clone(),
            run_id: candidate.run_id().clone(),
            lineage_id: candidate.lineage_id().clone(),
            checkpoint_id: candidate.checkpoint_id().clone(),
            fingerprints: candidate.fingerprints().clone(),
            continuation: candidate.continuation().clone(),
            report: candidate.report().clone(),
            deadline_unix_ms: candidate.deadline_unix_ms(),
            resume_point: candidate.resume_point().clone(),
        })
    }

    #[derive(Debug, Clone, Copy)]
    enum CheckpointCandidateShape {
        ReadyForModel,
        AwaitingApproval,
        AwaitingProvider,
        RecoveryIndeterminate,
        TerminalFailure,
    }

    impl CheckpointCandidateShape {
        fn checkpoint_id(self) -> &'static str {
            match self {
                Self::ReadyForModel => "checkpoint-ready",
                Self::AwaitingApproval => "checkpoint-approval",
                Self::AwaitingProvider => "checkpoint-provider",
                Self::RecoveryIndeterminate => "checkpoint-recovery",
                Self::TerminalFailure => "checkpoint-terminal",
            }
        }
    }

    fn checkpoint_candidate(
        loop_: &DurableToolLoop,
        shape: CheckpointCandidateShape,
    ) -> RunSnapshot {
        let target = ModelTarget::from_model(loop_.model.as_ref());
        let messages = vec![Message::user("test")];
        let mut report = RunReport::new(target.clone(), messages);
        let resume_point = match shape {
            CheckpointCandidateShape::ReadyForModel => ResumePoint::ready_for_model(0, target),
            CheckpointCandidateShape::AwaitingApproval => {
                report.accumulate_usage(&Usage::default());
                report.replace_tool_journal(ToolJournal::from_log(prepared_log()));
                report
                    .budget_mut()
                    .reserve_pending_approval(loop_.runtime.run_budget())
                    .unwrap();
                let tool = prepared_tool();
                let approval = PendingApprovalSnapshot {
                    approval_id: "approval:call-1".to_string(),
                    call: tool.call().clone(),
                    binding: tool.binding().clone(),
                    claim_fingerprint: SnapshotFingerprint::new("sha256:test-claim").unwrap(),
                    expires_at_unix_ms: None,
                };
                ResumePoint::awaiting_approvals(PendingStepSnapshot::new(
                    0,
                    target,
                    tool_response(),
                    vec![tool],
                    Vec::new(),
                    vec![approval],
                ))
            }
            CheckpointCandidateShape::AwaitingProvider => {
                report.accumulate_usage(&Usage::default());
                let item = provider_item(loop_);
                observe_provider_state(&mut report, &target, "provider-state-1", &item);
                let provider_state = report
                    .provider_deferred_ledger()
                    .pending_projection(target.scope())
                    .unwrap();
                ResumePoint::awaiting_provider(PendingProviderStepSnapshot::new(
                    0,
                    target,
                    completed_response(0),
                    provider_state,
                ))
            }
            CheckpointCandidateShape::RecoveryIndeterminate => {
                report.accumulate_usage(&Usage::default());
                let mut journal = ToolJournal::from_log(dispatched_log());
                journal.recover_dispatched().unwrap();
                report.replace_tool_journal(journal);
                ResumePoint::terminal_state(SnapshotTerminal::indeterminate(
                    SnapshotReason::runtime_code("dispatch_outcome_unknown"),
                ))
            }
            CheckpointCandidateShape::TerminalFailure => ResumePoint::terminal_state(
                SnapshotTerminal::failed(SnapshotReason::runtime_code("worker_failed"), None),
            ),
        };

        test_snapshot_with_report(loop_, shape.checkpoint_id(), None, report, resume_point)
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
    async fn checkpoint_writer_owns_parent_identity() {
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
        let intent = successor_intent(&previous.snapshot, invalid);
        let committed = loop_
            .commit_checkpoint(&mut lease, previous.revision, intent)
            .await
            .unwrap();

        assert_eq!(
            committed.snapshot.parent_checkpoint_id(),
            Some(previous.snapshot.checkpoint_id())
        );
        assert_eq!(store.cas_calls(), 1);
        store.release(lease).await.unwrap();
    }

    #[tokio::test]
    async fn checkpoint_writer_accepts_the_exact_limit_and_rejects_one_byte_less_before_cas() {
        let exact_store = CountingStore::new();
        let mut exact_loop = test_loop(exact_store.clone());
        let candidate = test_snapshot(&exact_loop, "checkpoint-exact", None);
        let encoded_bytes = serde_json::to_vec(&candidate).unwrap().len();
        exact_loop.runtime = Runtime::builder()
            .with_run_budget(
                RunBudget::builder()
                    .max_snapshot_bytes(encoded_bytes)
                    .build()
                    .unwrap(),
            )
            .build();
        let mut lease = exact_store
            .acquire(candidate.run_id(), Duration::from_secs(30))
            .await
            .unwrap();
        let committed = exact_loop
            .commit_checkpoint(
                &mut lease,
                SnapshotRevision::EMPTY,
                initial_intent(candidate),
            )
            .await
            .unwrap();
        assert_eq!(
            serde_json::to_vec(committed.snapshot()).unwrap().len(),
            encoded_bytes
        );
        assert_eq!(exact_store.cas_calls(), 1);
        exact_store.release(lease).await.unwrap();

        let bounded_store = CountingStore::new();
        let mut bounded_loop = test_loop(bounded_store.clone());
        let candidate = test_snapshot(&bounded_loop, "checkpoint-bound", None);
        let encoded_bytes = serde_json::to_vec(&candidate).unwrap().len();
        bounded_loop.runtime = Runtime::builder()
            .with_run_budget(
                RunBudget::builder()
                    .max_snapshot_bytes(encoded_bytes - 1)
                    .build()
                    .unwrap(),
            )
            .build();
        let mut lease = bounded_store
            .acquire(candidate.run_id(), Duration::from_secs(30))
            .await
            .unwrap();
        let error = bounded_loop
            .commit_checkpoint(
                &mut lease,
                SnapshotRevision::EMPTY,
                initial_intent(candidate),
            )
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            DurableRunError::Budget(crate::BudgetError::Exceeded {
                kind: crate::BudgetKind::SnapshotBytes,
                ..
            })
        ));
        assert_eq!(bounded_store.cas_calls(), 0);
        bounded_store.release(lease).await.unwrap();
    }

    #[tokio::test]
    async fn checkpoint_writer_rejects_every_candidate_shape_before_store_cas() {
        for shape in [
            CheckpointCandidateShape::ReadyForModel,
            CheckpointCandidateShape::AwaitingApproval,
            CheckpointCandidateShape::AwaitingProvider,
            CheckpointCandidateShape::RecoveryIndeterminate,
            CheckpointCandidateShape::TerminalFailure,
        ] {
            let store = CountingStore::new();
            let mut loop_ = test_loop(store.clone());
            loop_.runtime = Runtime::builder()
                .with_run_budget(RunBudget::builder().max_snapshot_bytes(1).build().unwrap())
                .build();
            let candidate = checkpoint_candidate(&loop_, shape);
            let mut lease = store
                .acquire(candidate.run_id(), Duration::from_secs(30))
                .await
                .unwrap();

            let error = loop_
                .commit_checkpoint(
                    &mut lease,
                    SnapshotRevision::EMPTY,
                    initial_intent(candidate),
                )
                .await
                .unwrap_err();

            assert!(
                matches!(
                    error,
                    DurableRunError::Budget(crate::BudgetError::Exceeded {
                        kind: crate::BudgetKind::SnapshotBytes,
                        ..
                    })
                ),
                "{shape:?}: expected the shared snapshot-size gate, got {error:?}"
            );
            assert_eq!(store.cas_calls(), 0, "{shape:?}");
            store.release(lease).await.unwrap();
        }
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
                    observe_provider_state(
                        &mut previous_report,
                        &target,
                        "provider-state-1",
                        &item,
                    );
                    previous_report.accumulate_usage(&Usage::default());
                }
                SuccessorRegression::ExecutionLog => {
                    previous_report.replace_tool_journal(ToolJournal::from_log(prepared_log()));
                }
            }

            let previous_next_step = u32::try_from(previous_report.steps().len()).unwrap();
            let next_step = u32::try_from(next_report.steps().len()).unwrap();
            let previous_resume = if matches!(case, SuccessorRegression::Provider) {
                let provider_state = previous_report
                    .provider_deferred_ledger()
                    .pending_projection(target.scope())
                    .unwrap();
                ResumePoint::awaiting_provider(PendingProviderStepSnapshot::new(
                    previous_next_step,
                    target.clone(),
                    completed_response(0),
                    provider_state,
                ))
            } else {
                ResumePoint::ready_for_model(previous_next_step, target.clone())
            };
            let previous = test_snapshot_with_report(
                &loop_,
                "checkpoint-1",
                None,
                previous_report,
                previous_resume,
            );
            let candidate = test_snapshot_with_report(
                &loop_,
                "checkpoint-2",
                Some("checkpoint-1"),
                next_report,
                ResumePoint::ready_for_model(next_step, target),
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

            let intent = successor_intent(&state.snapshot, candidate);
            let error = loop_
                .commit_checkpoint(&mut lease, state.revision, intent)
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
        report.replace_tool_journal(ToolJournal::from_log(dispatched_log()));
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
            ResumePoint::ready_to_dispatch(pending),
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
        assert_eq!(
            recovered.snapshot.resume_point().terminal().unwrap().kind(),
            crate::snapshot::SnapshotTerminalKind::Indeterminate
        );
        store.release(lease).await.unwrap();
    }

    #[tokio::test]
    async fn recovery_rejects_a_tampered_successor_before_store_cas() {
        let store = CountingStore::new();
        let loop_ = test_loop(store.clone());
        let target = ModelTarget::from_model(loop_.model.as_ref());
        let mut report = RunReport::new(target.clone(), vec![Message::user("test")]);
        report.accumulate_usage(&Usage::default());
        report.replace_tool_journal(ToolJournal::from_log(dispatched_log()));
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
            ResumePoint::ready_to_dispatch(pending),
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
        let mut recovered_report = previous.report().clone();
        recovered_report
            .tool_journal_mut()
            .recover_dispatched()
            .unwrap();
        recovered_report.messages_mut().clear();
        let resume_point = ResumePoint::terminal_state(SnapshotTerminal::indeterminate(
            SnapshotReason::runtime_code("dispatch_outcome_unknown"),
        ));
        let intent = CheckpointIntent::successor(
            &state.snapshot,
            next_checkpoint_id().unwrap(),
            LanguageRequest::new(Vec::new()),
            recovered_report,
            resume_point,
        );

        let error = loop_
            .commit_checkpoint(&mut lease, state.revision, intent)
            .await
            .unwrap_err();

        assert!(matches!(
            error,
            DurableRunError::InvalidSuccessor(RunSnapshotSuccessorError::MessageHistoryRegression)
        ));
        assert_eq!(store.cas_calls(), 0);
        store.release(lease).await.unwrap();
    }

    #[tokio::test]
    async fn checkpoint_writer_owns_identity_and_store_rejects_terminal_updates() {
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

        let intent = successor_intent(&state.snapshot, candidate);
        let committed = loop_
            .commit_checkpoint(&mut lease, state.revision, intent)
            .await
            .unwrap();
        assert_eq!(committed.snapshot.run_id(), state.snapshot.run_id());
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
            ResumePoint::terminal_state(SnapshotTerminal::failed(
                SnapshotReason::new("worker_failed", None).unwrap(),
                None,
            )),
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

        let intent = successor_intent(&state.snapshot, candidate);
        let error = loop_
            .commit_checkpoint(&mut lease, state.revision, intent)
            .await
            .unwrap_err();
        assert!(
            matches!(
                &error,
                DurableRunError::Store(RunStoreError::RunAlreadyTerminal)
            ),
            "unexpected terminal successor error: {error:?}"
        );
        assert_eq!(store.cas_calls(), 1);
        store.release(lease).await.unwrap();
    }
}
