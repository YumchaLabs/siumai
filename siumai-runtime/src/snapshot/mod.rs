//! Versioned quiescent run snapshots and lease/CAS persistence hooks.

mod model;
mod store;

pub use model::{
    CheckpointId, IndeterminateReason, InvalidSnapshotId, LineageId, PendingApprovalSnapshot,
    ProviderStateSnapshot, RUN_SNAPSHOT_SCHEMA_VERSION, RunId, RunSnapshot, RunSnapshotError,
    RunSnapshotParts, SnapshotBudgetCounter, SnapshotBudgetLedger, SnapshotEngineVersion,
    SnapshotFingerprint, SnapshotFingerprints, SnapshotReason, SnapshotState, SnapshotSuspension,
    SnapshotTerminal, ToolExecutionEvent, ToolExecutionLog, ToolExecutionStatus,
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
        ExecutionOwner, Message, MessageRole, Model, ModelDescriptor, ModelFamily, ModelId,
        ProviderId, RouteId, ToolBindingIdentity, ToolCall, ToolOutcome, Usage,
    };

    use super::*;
    use crate::ModelTarget;

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

    fn fingerprint(value: &str) -> SnapshotFingerprint {
        SnapshotFingerprint::new(value).unwrap()
    }

    fn paused_state(message: Option<&str>) -> SnapshotState {
        SnapshotState::Suspended(SnapshotSuspension::Paused {
            reason: SnapshotReason::new(
                "external_pause",
                message.map(std::string::ToString::to_string),
            )
            .unwrap(),
        })
    }

    fn snapshot_with(
        checkpoint: &str,
        execution_log: ToolExecutionLog,
        provider_state: Vec<ProviderStateSnapshot>,
        state: SnapshotState,
    ) -> RunSnapshot {
        RunSnapshot::new(RunSnapshotParts {
            engine_version: SnapshotEngineVersion::new("runtime-test-v1").unwrap(),
            run_id: RunId::new("run-1").unwrap(),
            lineage_id: LineageId::new("lineage-1").unwrap(),
            checkpoint_id: CheckpointId::new(checkpoint).unwrap(),
            parent_checkpoint_id: None,
            target: ModelTarget::from_model(&TestModel::new()),
            fingerprints: SnapshotFingerprints {
                options: fingerprint("sha256:options"),
                tool_catalog: fingerprint("sha256:catalog"),
                approval_policy: fingerprint("sha256:approval"),
            },
            history: vec![Message::text(MessageRole::User, "history-secret")],
            pending_approvals: Vec::new(),
            provider_state,
            execution_log,
            budget: SnapshotBudgetLedger::default(),
            usage: Usage::default(),
            deadline_unix_ms: Some(4_102_444_800_000),
            state,
        })
        .unwrap()
    }

    fn prepared_log() -> ToolExecutionLog {
        let mut log = ToolExecutionLog::new();
        log.append(ToolExecutionEvent::prepared(
            0,
            10,
            ToolCall {
                id: "call-1".to_string(),
                name: "write_record".to_string(),
                arguments: json!({"password": "tool-secret"}),
                owner: ExecutionOwner::Local,
            },
            ToolBindingIdentity {
                name: "write_record".to_string(),
                fingerprint: "binding-secret".to_string(),
            },
            Some("idempotency-secret".to_string()),
        ))
        .unwrap();
        log
    }

    #[tokio::test]
    async fn compare_and_swap_reports_stale_revision() {
        let store = InMemoryRunStore::new();
        let run_id = RunId::new("run-1").unwrap();
        let lease = store
            .acquire(&run_id, Duration::from_secs(30))
            .await
            .unwrap();
        let first = snapshot_with(
            "checkpoint-1",
            ToolExecutionLog::new(),
            Vec::new(),
            paused_state(None),
        );
        let revision = store
            .compare_and_swap(&lease, SnapshotRevision::EMPTY, first)
            .await
            .unwrap();
        assert_eq!(revision.value(), 1);

        let stale = snapshot_with(
            "checkpoint-2",
            ToolExecutionLog::new(),
            Vec::new(),
            paused_state(None),
        );
        let error = store
            .compare_and_swap(&lease, SnapshotRevision::EMPTY, stale)
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

        let error = store
            .acquire(&run_id, Duration::from_secs(30))
            .await
            .unwrap_err();
        assert_eq!(
            error,
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
        let snapshot = snapshot_with(
            "checkpoint-1",
            ToolExecutionLog::new(),
            Vec::new(),
            paused_state(None),
        );
        let mut value = serde_json::to_value(snapshot).unwrap();
        value["snapshot_version"] = json!(RUN_SNAPSHOT_SCHEMA_VERSION + 1);

        let error = serde_json::from_value::<RunSnapshot>(value).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("unsupported run snapshot version")
        );
    }

    #[test]
    fn execution_log_rejects_invalid_transition() {
        let mut log = ToolExecutionLog::new();
        let error = log
            .append(ToolExecutionEvent::dispatched(
                0,
                10,
                "call-1",
                Some("dispatch-1".to_string()),
            ))
            .unwrap_err();

        assert_eq!(
            error,
            ToolExecutionTransitionError::InvalidTransition {
                call_id: "call-1".to_string(),
                from: None,
                to: ToolExecutionStatus::Dispatched,
            }
        );
    }

    #[tokio::test]
    async fn recovery_marks_lingering_dispatch_indeterminate() {
        let mut log = prepared_log();
        log.append(ToolExecutionEvent::dispatched(
            1,
            20,
            "call-1",
            Some("dispatch-1".to_string()),
        ))
        .unwrap();
        let snapshot = snapshot_with("checkpoint-1", log, Vec::new(), paused_state(None));

        let store = InMemoryRunStore::new();
        let lease = store
            .acquire(snapshot.run_id(), Duration::from_secs(30))
            .await
            .unwrap();
        store
            .compare_and_swap(&lease, SnapshotRevision::EMPTY, snapshot)
            .await
            .unwrap();
        let recovered = store.load(&lease).await.unwrap().unwrap();

        assert_eq!(
            recovered.snapshot().execution_log().status("call-1"),
            Some(ToolExecutionStatus::Indeterminate)
        );
        assert_eq!(
            recovered
                .snapshot()
                .execution_log()
                .replay_disposition("call-1"),
            ToolReplayDisposition::HaltIndeterminate
        );
        assert!(matches!(
            recovered.snapshot().state(),
            SnapshotState::Terminal(SnapshotTerminal::Indeterminate { .. })
        ));
    }

    #[tokio::test]
    async fn completed_execution_is_not_replayable_or_removable() {
        let mut log = prepared_log();
        log.append(ToolExecutionEvent::dispatched(1, 20, "call-1", None))
            .unwrap();
        log.append(ToolExecutionEvent::completed(
            2,
            30,
            "call-1",
            ToolOutcome::Success { value: json!(true) },
        ))
        .unwrap();
        assert_eq!(
            log.replay_disposition("call-1"),
            ToolReplayDisposition::DoNotReplayCompleted
        );

        let store = InMemoryRunStore::new();
        let run_id = RunId::new("run-1").unwrap();
        let lease = store
            .acquire(&run_id, Duration::from_secs(30))
            .await
            .unwrap();
        let first = snapshot_with("checkpoint-1", log, Vec::new(), paused_state(None));
        let revision = store
            .compare_and_swap(&lease, SnapshotRevision::EMPTY, first)
            .await
            .unwrap();
        let regressed = snapshot_with(
            "checkpoint-2",
            ToolExecutionLog::new(),
            Vec::new(),
            paused_state(None),
        );

        assert_eq!(
            store
                .compare_and_swap(&lease, revision, regressed)
                .await
                .unwrap_err(),
            RunStoreError::ExecutionLogRegression
        );
    }

    #[test]
    fn snapshot_debug_redacts_payloads() {
        let snapshot = snapshot_with(
            "checkpoint-1",
            prepared_log(),
            vec![ProviderStateSnapshot {
                namespace: "test.provider".to_string(),
                correlation_id: Some("correlation-secret".to_string()),
                encoding: "application/octet-stream".to_string(),
                payload: b"provider-secret".to_vec(),
            }],
            paused_state(Some("reason-secret")),
        );

        let debug = format!("{snapshot:?}");
        for secret in [
            "history-secret",
            "tool-secret",
            "binding-secret",
            "idempotency-secret",
            "correlation-secret",
            "provider-secret",
            "reason-secret",
            "sha256:options",
        ] {
            assert!(!debug.contains(secret), "debug leaked {secret}");
        }
    }
}
