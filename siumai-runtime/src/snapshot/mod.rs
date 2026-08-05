//! Versioned durable run snapshots and lease/CAS persistence hooks.

mod model;
mod store;

pub use model::{
    CheckpointId, CompletedToolSnapshot, IndeterminateReason, InvalidSnapshotId, LineageId,
    PendingApprovalSnapshot, PendingProviderStepSnapshot, PendingStepSnapshot,
    PreparedToolSnapshot, ProviderStateSnapshot, RUN_SNAPSHOT_SCHEMA_VERSION, ResumePoint,
    ResumePointKind, RunId, RunSnapshot, RunSnapshotError, RunSnapshotSuccessorError,
    SnapshotCheckpoint, SnapshotEngineVersion, SnapshotFingerprint, SnapshotFingerprints,
    SnapshotReason, SnapshotTerminal, ToolExecutionEvent, ToolExecutionLog, ToolExecutionStatus,
    ToolExecutionTransitionError, ToolReplayDisposition,
};
pub use store::{
    InMemoryRunStore, RunLease, RunStore, RunStoreError, RunStoreFuture, SnapshotRevision,
    StoredRun,
};

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use serde_json::json;
    use siumai_core::{
        ContentPart, ExecutionOwner, FinishReason, LanguageResponse, Message, MessageRole, Model,
        ModelDescriptor, ModelFamily, ModelId, ProviderId, RouteId, ToolBindingIdentity, ToolCall,
        ToolOutcome, Usage,
    };

    use super::*;
    use crate::tool::{RecoveryPolicy, ToolExecutionAttempt, ToolIdempotencyKey};
    use crate::{ModelTarget, RunBudget, RunReport};

    const DEADLINE: u64 = 4_102_444_800_000;

    struct TestModel {
        descriptor: ModelDescriptor,
        route: RouteId,
    }

    impl TestModel {
        fn new() -> Self {
            Self {
                descriptor: ModelDescriptor::new(
                    ProviderId::new("test-provider").unwrap(),
                    ModelId::new("test-model").unwrap(),
                    ModelFamily::Language,
                ),
                route: RouteId::new("production").unwrap(),
            }
        }
    }

    impl Model for TestModel {
        fn descriptor(&self) -> &ModelDescriptor {
            &self.descriptor
        }

        fn route_id(&self) -> Option<&RouteId> {
            Some(&self.route)
        }
    }

    fn target() -> ModelTarget {
        ModelTarget::from_model(&TestModel::new())
    }

    fn fingerprint(value: &str) -> SnapshotFingerprint {
        SnapshotFingerprint::new(value).unwrap()
    }

    fn fingerprints() -> SnapshotFingerprints {
        SnapshotFingerprints {
            options: fingerprint("sha256:options"),
            tool_catalog: fingerprint("sha256:catalog"),
            approval_policy: fingerprint("sha256:approval"),
        }
    }

    fn checkpoint(id: &str, parent: Option<&str>) -> SnapshotCheckpoint {
        SnapshotCheckpoint::new(
            SnapshotEngineVersion::new("runtime-test-v2").unwrap(),
            RunId::new("run-1").unwrap(),
            LineageId::new("lineage-1").unwrap(),
            CheckpointId::new(id).unwrap(),
            parent.map(|parent| CheckpointId::new(parent).unwrap()),
        )
        .unwrap()
    }

    fn tool_call() -> ToolCall {
        ToolCall {
            id: "call-1".to_string(),
            name: "write_record".to_string(),
            arguments: json!({"password": "tool-secret"}),
            owner: ExecutionOwner::Local,
        }
    }

    fn binding() -> ToolBindingIdentity {
        ToolBindingIdentity {
            name: "write_record".to_string(),
            fingerprint: "binding-secret".to_string(),
        }
    }

    fn prepared_tool_with(key: &str, attempt: ToolExecutionAttempt) -> PreparedToolSnapshot {
        PreparedToolSnapshot::new(
            0,
            tool_call(),
            binding(),
            RecoveryPolicy::ReplayWithStableIdempotencyKey,
            Some(ToolIdempotencyKey::new(key).unwrap()),
            attempt,
        )
    }

    fn prepared_tool() -> PreparedToolSnapshot {
        prepared_tool_with("idempotency-secret", ToolExecutionAttempt::INITIAL)
    }

    fn tool_response() -> LanguageResponse {
        LanguageResponse::completed(
            vec![ContentPart::ToolCall(tool_call())],
            FinishReason::ToolCalls,
            Usage::default(),
        )
        .unwrap()
    }

    fn report_with_log(log: ToolExecutionLog) -> RunReport {
        let mut report = RunReport::new(
            target(),
            vec![Message::text(MessageRole::User, "history-secret")],
        );
        *report.execution_log_mut() = log;
        report
    }

    fn prepared_log() -> ToolExecutionLog {
        let mut log = ToolExecutionLog::new();
        log.append(ToolExecutionEvent::prepared(0, 10, 0, prepared_tool()))
            .unwrap();
        log
    }

    fn completed_log() -> ToolExecutionLog {
        let mut log = prepared_log();
        log.append(ToolExecutionEvent::dispatched(
            1,
            20,
            "call-1",
            ToolExecutionAttempt::INITIAL,
            Some("dispatch-1".to_string()),
        ))
        .unwrap();
        log.append(ToolExecutionEvent::completed(
            2,
            30,
            "call-1",
            ToolExecutionAttempt::INITIAL,
            ToolOutcome::Success { value: json!(true) },
        ))
        .unwrap();
        log
    }

    fn approval() -> PendingApprovalSnapshot {
        PendingApprovalSnapshot {
            approval_id: "approval-1".to_string(),
            call: tool_call(),
            binding: binding(),
            claim_fingerprint: fingerprint("sha256:claim"),
            expires_at_unix_ms: Some(DEADLINE - 1),
        }
    }

    fn pending_step(pending_approvals: Vec<PendingApprovalSnapshot>) -> PendingStepSnapshot {
        PendingStepSnapshot::new(
            0,
            target(),
            tool_response(),
            vec![prepared_tool()],
            Vec::new(),
            pending_approvals,
        )
    }

    fn ready_report() -> RunReport {
        report_with_log(ToolExecutionLog::new())
    }

    fn snapshot(
        checkpoint_id: &str,
        parent_checkpoint_id: Option<&str>,
        report: RunReport,
        deadline_unix_ms: Option<u64>,
        resume_point: ResumePoint,
    ) -> RunSnapshot {
        RunSnapshot::new(
            checkpoint(checkpoint_id, parent_checkpoint_id),
            fingerprints(),
            report,
            deadline_unix_ms,
            resume_point,
        )
        .unwrap()
    }

    fn ready_snapshot(
        checkpoint_id: &str,
        parent_checkpoint_id: Option<&str>,
        report: RunReport,
        deadline_unix_ms: Option<u64>,
    ) -> RunSnapshot {
        snapshot(
            checkpoint_id,
            parent_checkpoint_id,
            report,
            deadline_unix_ms,
            ResumePoint::ReadyForModel {
                next_step: 0,
                target: target(),
            },
        )
    }

    fn awaiting_approval_snapshot(
        checkpoint_id: &str,
        parent_checkpoint_id: Option<&str>,
    ) -> RunSnapshot {
        let mut report = report_with_log(prepared_log());
        report
            .budget_mut()
            .reserve_pending_approval(&RunBudget::default())
            .unwrap();
        snapshot(
            checkpoint_id,
            parent_checkpoint_id,
            report,
            Some(DEADLINE),
            ResumePoint::AwaitingApprovals(pending_step(vec![approval()])),
        )
    }

    #[test]
    fn pending_approval_must_exactly_match_prepared_work() {
        let valid = awaiting_approval_snapshot("checkpoint-1", None);
        assert_eq!(valid.pending_approvals().len(), 1);

        let mut mismatched_approval = approval();
        mismatched_approval.binding.fingerprint = "different-binding".to_string();
        let mut report = report_with_log(prepared_log());
        report
            .budget_mut()
            .reserve_pending_approval(&RunBudget::default())
            .unwrap();
        let error = RunSnapshot::new(
            checkpoint("checkpoint-2", None),
            fingerprints(),
            report,
            Some(DEADLINE),
            ResumePoint::AwaitingApprovals(pending_step(vec![mismatched_approval])),
        )
        .unwrap_err();

        assert_eq!(
            error,
            RunSnapshotError::ApprovalPreparedMismatch {
                call_id: "call-1".to_string(),
            }
        );
    }

    #[test]
    fn pending_step_must_match_the_full_prepared_event() {
        let mismatched = prepared_tool_with("different-key", ToolExecutionAttempt::INITIAL);
        let pending = PendingStepSnapshot::new(
            0,
            target(),
            tool_response(),
            vec![mismatched],
            Vec::new(),
            Vec::new(),
        );
        let error = RunSnapshot::new(
            checkpoint("checkpoint-1", None),
            fingerprints(),
            report_with_log(prepared_log()),
            Some(DEADLINE),
            ResumePoint::ReadyToDispatch(pending),
        )
        .unwrap_err();

        assert_eq!(
            error,
            RunSnapshotError::PreparedEventMismatch {
                call_id: "call-1".to_string(),
            }
        );
    }

    #[test]
    fn local_denial_can_complete_without_dispatch() {
        let mut log = prepared_log();
        log.append(ToolExecutionEvent::completed(
            1,
            20,
            "call-1",
            ToolExecutionAttempt::INITIAL,
            ToolOutcome::Denied {
                reason: "host denied".to_string(),
            },
        ))
        .unwrap();

        assert_eq!(log.status("call-1"), Some(ToolExecutionStatus::Completed));
    }

    #[test]
    fn local_success_cannot_complete_without_dispatch() {
        let mut log = prepared_log();
        let error = log
            .append(ToolExecutionEvent::completed(
                1,
                20,
                "call-1",
                ToolExecutionAttempt::INITIAL,
                ToolOutcome::Success { value: json!(true) },
            ))
            .unwrap_err();

        assert_eq!(
            error,
            ToolExecutionTransitionError::DirectSuccessRequiresDispatch {
                call_id: "call-1".to_string(),
            }
        );
    }

    #[test]
    fn recovery_attempt_preserves_frozen_work_and_increments_attempt() {
        let mut log = prepared_log();
        log.append(ToolExecutionEvent::dispatched(
            1,
            20,
            "call-1",
            ToolExecutionAttempt::INITIAL,
            None,
        ))
        .unwrap();
        log.append(ToolExecutionEvent::indeterminate(
            2,
            30,
            "call-1",
            ToolExecutionAttempt::INITIAL,
            IndeterminateReason::DispatchOutcomeUnknown,
        ))
        .unwrap();
        let second_attempt = ToolExecutionAttempt::INITIAL.next().unwrap();
        let retry = prepared_tool_with("idempotency-secret", second_attempt);
        log.append(ToolExecutionEvent::prepared(3, 40, 0, retry))
            .unwrap();

        assert_eq!(log.attempt("call-1"), Some(second_attempt));
        assert_eq!(
            log.replay_disposition("call-1"),
            ToolReplayDisposition::ReadyToDispatch
        );
    }

    #[test]
    fn execution_log_rejects_wrong_attempt() {
        let mut log = prepared_log();
        let second_attempt = ToolExecutionAttempt::INITIAL.next().unwrap();
        let error = log
            .append(ToolExecutionEvent::dispatched(
                1,
                20,
                "call-1",
                second_attempt,
                None,
            ))
            .unwrap_err();

        assert_eq!(
            error,
            ToolExecutionTransitionError::AttemptMismatch {
                call_id: "call-1".to_string(),
                expected: 1,
                actual: 2,
            }
        );
    }

    #[tokio::test]
    async fn compare_and_swap_reports_stale_revision() {
        let store = InMemoryRunStore::new();
        let run_id = RunId::new("run-1").unwrap();
        let lease = store
            .acquire(&run_id, Duration::from_secs(30))
            .await
            .unwrap();
        let revision = store
            .compare_and_swap(
                &lease,
                SnapshotRevision::EMPTY,
                ready_snapshot("checkpoint-1", None, ready_report(), Some(DEADLINE)),
            )
            .await
            .unwrap();

        let error = store
            .compare_and_swap(
                &lease,
                SnapshotRevision::EMPTY,
                ready_snapshot(
                    "checkpoint-2",
                    Some("checkpoint-1"),
                    ready_report(),
                    Some(DEADLINE),
                ),
            )
            .await
            .unwrap_err();
        assert_eq!(
            error,
            RunStoreError::CasConflict {
                expected: SnapshotRevision::EMPTY,
                actual: revision,
            }
        );
    }

    #[tokio::test]
    async fn one_run_cannot_have_two_live_leases() {
        let store = InMemoryRunStore::new();
        let run_id = RunId::new("run-1").unwrap();
        let lease = store
            .acquire(&run_id, Duration::from_secs(30))
            .await
            .unwrap();

        assert_eq!(
            store
                .acquire(&run_id, Duration::from_secs(30))
                .await
                .unwrap_err(),
            RunStoreError::LeaseConflict {
                run_id: run_id.clone(),
            }
        );

        drop(lease);
        store
            .acquire(&run_id, Duration::from_secs(30))
            .await
            .unwrap();
    }

    #[test]
    fn deserialization_rejects_unknown_snapshot_version() {
        let snapshot = ready_snapshot("checkpoint-1", None, ready_report(), Some(DEADLINE));
        let mut value = serde_json::to_value(snapshot).unwrap();
        value["snapshot_version"] = json!(RUN_SNAPSHOT_SCHEMA_VERSION + 1);

        let error = serde_json::from_value::<RunSnapshot>(value).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("unsupported run snapshot version")
        );
    }

    #[tokio::test]
    async fn load_is_a_pure_read() {
        let mut log = prepared_log();
        log.append(ToolExecutionEvent::dispatched(
            1,
            20,
            "call-1",
            ToolExecutionAttempt::INITIAL,
            Some("dispatch-1".to_string()),
        ))
        .unwrap();
        let snapshot = snapshot(
            "checkpoint-1",
            None,
            report_with_log(log),
            Some(DEADLINE),
            ResumePoint::Terminal(SnapshotTerminal::Failed {
                reason: SnapshotReason::new("worker_crashed", None).unwrap(),
            }),
        );
        let expected = snapshot.clone();
        let store = InMemoryRunStore::new();
        let lease = store
            .acquire(snapshot.run_id(), Duration::from_secs(30))
            .await
            .unwrap();
        store
            .compare_and_swap(&lease, SnapshotRevision::EMPTY, snapshot)
            .await
            .unwrap();

        let loaded = store.load(&lease).await.unwrap().unwrap();
        assert_eq!(loaded.snapshot(), &expected);
        assert_eq!(
            loaded.snapshot().execution_log().status("call-1"),
            Some(ToolExecutionStatus::Dispatched)
        );
    }

    #[test]
    fn recovery_is_explicit_and_marks_dispatch_indeterminate() {
        let mut log = prepared_log();
        log.append(ToolExecutionEvent::dispatched(
            1,
            20,
            "call-1",
            ToolExecutionAttempt::INITIAL,
            None,
        ))
        .unwrap();
        let recovered = snapshot(
            "checkpoint-1",
            None,
            report_with_log(log),
            Some(DEADLINE),
            ResumePoint::Terminal(SnapshotTerminal::Failed {
                reason: SnapshotReason::new("worker_crashed", None).unwrap(),
            }),
        )
        .recovered_for_resume(30)
        .unwrap();

        assert_eq!(
            recovered.execution_log().status("call-1"),
            Some(ToolExecutionStatus::Indeterminate)
        );
        assert!(matches!(
            recovered.resume_point(),
            ResumePoint::Terminal(SnapshotTerminal::Indeterminate { .. })
        ));
    }

    async fn store_current(
        snapshot: RunSnapshot,
    ) -> (InMemoryRunStore, RunLease, SnapshotRevision) {
        let store = InMemoryRunStore::new();
        let lease = store
            .acquire(snapshot.run_id(), Duration::from_secs(30))
            .await
            .unwrap();
        let revision = store
            .compare_and_swap(&lease, SnapshotRevision::EMPTY, snapshot)
            .await
            .unwrap();
        (store, lease, revision)
    }

    #[tokio::test]
    async fn cas_rejects_wrong_parent_checkpoint() {
        let current = ready_snapshot("checkpoint-1", None, ready_report(), Some(DEADLINE));
        let (store, lease, revision) = store_current(current).await;
        let successor = ready_snapshot("checkpoint-2", None, ready_report(), Some(DEADLINE));

        assert_eq!(
            store
                .compare_and_swap(&lease, revision, successor)
                .await
                .unwrap_err(),
            RunStoreError::InvalidSuccessor(RunSnapshotSuccessorError::ParentCheckpointMismatch {
                expected: CheckpointId::new("checkpoint-1").unwrap(),
                actual: None,
            })
        );
    }

    #[tokio::test]
    async fn cas_rejects_message_history_regression() {
        let current = ready_snapshot("checkpoint-1", None, ready_report(), Some(DEADLINE));
        let (store, lease, revision) = store_current(current).await;
        let successor_report = RunReport::new(target(), Vec::new());
        let successor = ready_snapshot(
            "checkpoint-2",
            Some("checkpoint-1"),
            successor_report,
            Some(DEADLINE),
        );

        assert_eq!(
            store
                .compare_and_swap(&lease, revision, successor)
                .await
                .unwrap_err(),
            RunStoreError::InvalidSuccessor(RunSnapshotSuccessorError::MessageHistoryRegression)
        );
    }

    #[tokio::test]
    async fn cas_rejects_budget_regression() {
        let mut current_report = ready_report();
        current_report
            .budget_mut()
            .charge_model_step(&RunBudget::default())
            .unwrap();
        let current = ready_snapshot("checkpoint-1", None, current_report, Some(DEADLINE));
        let (store, lease, revision) = store_current(current).await;
        let successor = ready_snapshot(
            "checkpoint-2",
            Some("checkpoint-1"),
            ready_report(),
            Some(DEADLINE),
        );

        assert_eq!(
            store
                .compare_and_swap(&lease, revision, successor)
                .await
                .unwrap_err(),
            RunStoreError::InvalidSuccessor(RunSnapshotSuccessorError::BudgetRegression {
                dimension: "model_steps",
                previous: 1,
                next: 0,
            })
        );
    }

    #[tokio::test]
    async fn cas_rejects_deadline_extension() {
        let current = ready_snapshot("checkpoint-1", None, ready_report(), Some(DEADLINE));
        let (store, lease, revision) = store_current(current).await;
        let successor = ready_snapshot(
            "checkpoint-2",
            Some("checkpoint-1"),
            ready_report(),
            Some(DEADLINE + 1),
        );

        assert_eq!(
            store
                .compare_and_swap(&lease, revision, successor)
                .await
                .unwrap_err(),
            RunStoreError::InvalidSuccessor(RunSnapshotSuccessorError::DeadlineRegression {
                previous: Some(DEADLINE),
                next: Some(DEADLINE + 1),
            })
        );
    }

    #[tokio::test]
    async fn cas_rejects_execution_log_regression() {
        let current = ready_snapshot(
            "checkpoint-1",
            None,
            report_with_log(completed_log()),
            Some(DEADLINE),
        );
        let (store, lease, revision) = store_current(current).await;
        let successor = ready_snapshot(
            "checkpoint-2",
            Some("checkpoint-1"),
            ready_report(),
            Some(DEADLINE),
        );

        assert_eq!(
            store
                .compare_and_swap(&lease, revision, successor)
                .await
                .unwrap_err(),
            RunStoreError::InvalidSuccessor(RunSnapshotSuccessorError::ExecutionLogRegression)
        );
    }

    #[tokio::test]
    async fn cas_rejects_illegal_resume_transition() {
        let current = snapshot(
            "checkpoint-1",
            None,
            report_with_log(prepared_log()),
            Some(DEADLINE),
            ResumePoint::ReadyToDispatch(pending_step(Vec::new())),
        );
        let (store, lease, revision) = store_current(current).await;
        let provider_step = PendingProviderStepSnapshot::new(
            0,
            target(),
            tool_response(),
            vec![ProviderStateSnapshot {
                namespace: "test.provider".to_string(),
                correlation_id: Some("correlation-secret".to_string()),
                encoding: "application/json".to_string(),
                payload: b"provider-secret".to_vec(),
            }],
        );
        let successor = snapshot(
            "checkpoint-2",
            Some("checkpoint-1"),
            report_with_log(prepared_log()),
            Some(DEADLINE),
            ResumePoint::AwaitingProvider(provider_step),
        );

        assert_eq!(
            store
                .compare_and_swap(&lease, revision, successor)
                .await
                .unwrap_err(),
            RunStoreError::InvalidSuccessor(RunSnapshotSuccessorError::InvalidResumeTransition {
                from: ResumePointKind::ReadyToDispatch,
                to: ResumePointKind::AwaitingProvider,
            })
        );
    }

    #[tokio::test]
    async fn cas_accepts_approval_resolution_to_ready_to_dispatch() {
        let current = awaiting_approval_snapshot("checkpoint-1", None);
        let mut successor_report = current.report().clone();
        successor_report.budget_mut().release_pending_approval();
        let successor = snapshot(
            "checkpoint-2",
            Some("checkpoint-1"),
            successor_report,
            Some(DEADLINE),
            ResumePoint::ReadyToDispatch(pending_step(Vec::new())),
        );
        let (store, lease, revision) = store_current(current).await;

        let next_revision = store
            .compare_and_swap(&lease, revision, successor)
            .await
            .unwrap();
        assert_eq!(next_revision.value(), revision.value() + 1);
    }

    #[test]
    fn snapshot_debug_redacts_payloads() {
        let provider_step = PendingProviderStepSnapshot::new(
            0,
            target(),
            tool_response(),
            vec![ProviderStateSnapshot {
                namespace: "test.provider".to_string(),
                correlation_id: Some("correlation-secret".to_string()),
                encoding: "application/octet-stream".to_string(),
                payload: b"provider-secret".to_vec(),
            }],
        );
        let snapshot = snapshot(
            "checkpoint-1",
            None,
            report_with_log(prepared_log()),
            Some(DEADLINE),
            ResumePoint::AwaitingProvider(provider_step),
        );

        let debug = format!("{snapshot:?}");
        for secret in [
            "history-secret",
            "tool-secret",
            "binding-secret",
            "idempotency-secret",
            "correlation-secret",
            "provider-secret",
            "sha256:options",
        ] {
            assert!(!debug.contains(secret), "debug leaked {secret}");
        }
    }

    #[test]
    fn completed_execution_is_never_replayable() {
        let log = completed_log();
        assert_eq!(
            log.replay_disposition("call-1"),
            ToolReplayDisposition::DoNotReplayCompleted
        );
    }
}
