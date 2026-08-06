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

    use serde::{Deserialize, Serialize};
    use serde_json::json;
    use siumai_core::{
        ContentAnnotationTarget, ContentPart, ExecutionOwner, FinishReason, LanguageRequest,
        LanguageResponse, Message, MessageAnnotationTarget, MessagePart, MessageRole, Model,
        ModelDescriptor, ModelFamily, ModelId, OpaqueProviderItem, ProtocolId, ProviderId,
        ProviderProvenance, RouteId, ToolAnnotationTarget, ToolBindingIdentity, ToolCall,
        ToolOutcome, ToolSpec, TypedProviderAnnotation, Usage,
    };

    use super::*;
    use crate::tool::{RecoveryPolicy, ToolExecutionAttempt, ToolIdempotencyKey};
    use crate::{
        ModelTarget, ModelTransitionOutcome, ModelTransitionRecord, ProjectionPolicy,
        ProjectionScope, RunBudget, RunReport, StepModelSelectorIdentity, StepRecord,
        project_history,
    };

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
            model_selector: None,
            projection_policy: crate::ProjectionPolicy::Strict,
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
        let continuation = LanguageRequest::new(report.messages().to_vec());
        RunSnapshot::new(
            checkpoint(checkpoint_id, parent_checkpoint_id),
            fingerprints(),
            continuation,
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

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct SnapshotMessageAnnotation {
        turn_id: String,
    }

    impl TypedProviderAnnotation for SnapshotMessageAnnotation {
        type Target = MessageAnnotationTarget;

        const NAMESPACE: &'static str = "anthropic";
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct SnapshotContentAnnotation {
        cache_boundary: bool,
    }

    impl TypedProviderAnnotation for SnapshotContentAnnotation {
        type Target = ContentAnnotationTarget;

        const NAMESPACE: &'static str = "anthropic";
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct SnapshotToolAnnotation {
        defer_loading: bool,
    }

    impl TypedProviderAnnotation for SnapshotToolAnnotation {
        type Target = ToolAnnotationTarget;

        const NAMESPACE: &'static str = "anthropic";
    }

    #[test]
    fn snapshot_round_trip_preserves_node_annotations() {
        let message_annotation = SnapshotMessageAnnotation {
            turn_id: "turn-1".to_string(),
        };
        let content_annotation = SnapshotContentAnnotation {
            cache_boundary: true,
        };
        let tool_annotation = SnapshotToolAnnotation {
            defer_loading: true,
        };
        let part = MessagePart::text("durable input")
            .with_provider_annotation(&content_annotation)
            .expect("valid content annotation");
        let message = Message::new(MessageRole::User, [part])
            .with_provider_annotation(&message_annotation)
            .expect("valid message annotation");
        let tool = ToolSpec::new(
            "lookup",
            Some("Look up a value".to_string()),
            json!({"type": "object"}),
        )
        .unwrap()
        .with_provider_annotation(&tool_annotation)
        .expect("valid tool annotation");
        let mut continuation = LanguageRequest::new(vec![message.clone()]);
        continuation.tools.push(tool);
        let report = RunReport::new(target(), vec![message]);
        let snapshot = RunSnapshot::new(
            checkpoint("checkpoint-annotations", None),
            fingerprints(),
            continuation,
            report,
            Some(DEADLINE),
            ResumePoint::ReadyForModel {
                next_step: 0,
                target: target(),
            },
        )
        .unwrap();

        let encoded = serde_json::to_value(&snapshot).unwrap();
        let restored: RunSnapshot = serde_json::from_value(encoded).unwrap();
        let restored_message = &restored.continuation().messages[0];
        assert_eq!(
            restored_message
                .annotations()
                .decode::<SnapshotMessageAnnotation>()
                .expect("message annotation decodes"),
            Some(message_annotation)
        );
        assert_eq!(
            restored_message.content()[0]
                .annotations()
                .decode::<SnapshotContentAnnotation>()
                .expect("content annotation decodes"),
            Some(content_annotation)
        );
        assert_eq!(
            restored.continuation().tools[0]
                .annotations()
                .decode::<SnapshotToolAnnotation>()
                .expect("tool annotation decodes"),
            Some(tool_annotation)
        );
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
            LanguageRequest::new(report.messages().to_vec()),
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
            LanguageRequest::new(report_with_log(prepared_log()).messages().to_vec()),
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
    async fn cas_accepts_only_an_exact_reprojection_for_history_regression() {
        let source = ModelTarget::new(
            ProviderId::new("source-provider").unwrap(),
            ModelId::new("source-model").unwrap(),
        )
        .with_protocol(ProtocolId::new("source.responses").unwrap());
        let destination = ModelTarget::new(
            ProviderId::new("destination-provider").unwrap(),
            ModelId::new("destination-model").unwrap(),
        )
        .with_protocol(ProtocolId::new("destination.messages").unwrap());
        let native = OpaqueProviderItem::new(
            ProviderProvenance {
                provider: source.provider().clone(),
                platform: None,
                protocol: source.protocol().unwrap().as_str().to_string(),
                model: source.model().clone(),
            },
            "response.output",
            json!({"id": "native-only"}),
        )
        .unwrap();
        let source_response = LanguageResponse::completed(
            vec![ContentPart::ProviderOpaque(native.clone())],
            FinishReason::Stop,
            Usage::default(),
        )
        .unwrap();
        let source_messages = vec![
            Message::text(MessageRole::User, "portable"),
            Message::new(
                MessageRole::Assistant,
                [ContentPart::ProviderOpaque(native)],
            ),
        ];
        let mut previous_report = RunReport::new(source.clone(), source_messages.clone());
        previous_report.steps_mut().push(StepRecord::new(
            0,
            source.clone(),
            source_response,
            Vec::new(),
        ));
        let mut policy_fingerprints = fingerprints();
        policy_fingerprints.model_selector = Some(StepModelSelectorIdentity::new(
            1,
            fingerprint("sha256:selector"),
        ));
        policy_fingerprints.projection_policy = ProjectionPolicy::BestEffort;
        let previous = RunSnapshot::new(
            checkpoint("checkpoint-1", None),
            policy_fingerprints.clone(),
            LanguageRequest::new(source_messages),
            previous_report,
            Some(DEADLINE),
            ResumePoint::ReadyForModel {
                next_step: 1,
                target: destination.clone(),
            },
        )
        .unwrap();
        let projected = project_history(
            previous.continuation().clone(),
            &source,
            &destination,
            ProjectionPolicy::BestEffort,
        )
        .unwrap();
        assert!(!projected.losses().is_empty());
        assert!(!projected.request().messages.starts_with(previous.history()));
        let final_response = LanguageResponse::completed(
            vec![ContentPart::Text {
                text: "done".to_string(),
            }],
            FinishReason::Stop,
            Usage::default(),
        )
        .unwrap();
        let mut continuation = projected.request().clone();
        continuation.messages.push(Message::new(
            MessageRole::Assistant,
            final_response
                .content()
                .iter()
                .cloned()
                .map(MessagePart::from),
        ));
        let mut successor_report = previous.report().clone();
        successor_report.replace_messages(continuation.messages.clone());
        successor_report
            .model_transitions_mut()
            .push(ModelTransitionRecord::new(
                1,
                source.clone(),
                destination.clone(),
                ProjectionPolicy::BestEffort,
                projected.scope(),
                ModelTransitionOutcome::Applied,
                projected.losses().to_vec(),
            ));
        successor_report.steps_mut().push(StepRecord::new(
            1,
            destination.clone(),
            final_response,
            Vec::new(),
        ));
        let mut tampered_continuation = continuation.clone();
        tampered_continuation.messages[0] =
            Message::text(MessageRole::User, "rewritten portable history");
        let mut tampered_report = successor_report.clone();
        tampered_report.replace_messages(tampered_continuation.messages.clone());
        let tampered = RunSnapshot::new(
            checkpoint("checkpoint-tampered", Some("checkpoint-1")),
            policy_fingerprints.clone(),
            tampered_continuation,
            tampered_report,
            Some(DEADLINE),
            ResumePoint::Terminal(SnapshotTerminal::Completed { reason: None }),
        )
        .unwrap();
        assert_eq!(
            previous.validate_successor(&tampered).unwrap_err(),
            RunSnapshotSuccessorError::ProjectionResultMismatch
        );
        let successor = RunSnapshot::new(
            checkpoint("checkpoint-2", Some("checkpoint-1")),
            policy_fingerprints,
            continuation,
            successor_report,
            Some(DEADLINE),
            ResumePoint::Terminal(SnapshotTerminal::Completed { reason: None }),
        )
        .unwrap();
        let (store, lease, revision) = store_current(previous).await;

        let next_revision = store
            .compare_and_swap(&lease, revision, successor)
            .await
            .unwrap();
        assert_eq!(next_revision.value(), revision.value() + 1);
    }

    #[test]
    fn snapshot_rejects_a_forged_model_transition_chain() {
        let source = target();
        let forged_source = ModelTarget::new(
            ProviderId::new("forged-provider").unwrap(),
            ModelId::new("forged-model").unwrap(),
        );
        let destination = ModelTarget::new(
            ProviderId::new("destination-provider").unwrap(),
            ModelId::new("destination-model").unwrap(),
        );
        let response = LanguageResponse::completed(
            vec![ContentPart::Text {
                text: "first step".to_string(),
            }],
            FinishReason::Stop,
            Usage::default(),
        )
        .unwrap();
        let messages = vec![Message::text(MessageRole::User, "continue")];
        let mut report = RunReport::new(source.clone(), messages.clone());
        report
            .steps_mut()
            .push(StepRecord::new(0, source, response, Vec::new()));
        report
            .model_transitions_mut()
            .push(ModelTransitionRecord::new(
                1,
                forged_source,
                destination,
                ProjectionPolicy::Strict,
                ProjectionScope::PortableOnly,
                ModelTransitionOutcome::Applied,
                Vec::new(),
            ));

        let error = RunSnapshot::new(
            checkpoint("checkpoint-forged", None),
            fingerprints(),
            LanguageRequest::new(messages),
            report,
            Some(DEADLINE),
            ResumePoint::Terminal(SnapshotTerminal::Completed { reason: None }),
        )
        .unwrap_err();

        assert_eq!(
            error,
            RunSnapshotError::ModelTransitionSourceMismatch { step: 1 }
        );
    }

    #[test]
    fn terminal_snapshot_exposes_the_current_model_target() {
        let source = target();
        let destination = ModelTarget::new(
            ProviderId::new("destination-provider").unwrap(),
            ModelId::new("destination-model").unwrap(),
        );
        let first = LanguageResponse::completed(
            vec![ContentPart::Text {
                text: "first step".to_string(),
            }],
            FinishReason::Stop,
            Usage::default(),
        )
        .unwrap();
        let second = LanguageResponse::completed(
            vec![ContentPart::Text {
                text: "second step".to_string(),
            }],
            FinishReason::Stop,
            Usage::default(),
        )
        .unwrap();
        let messages = vec![Message::text(MessageRole::User, "continue")];
        let mut report = RunReport::new(source.clone(), messages.clone());
        report
            .steps_mut()
            .push(StepRecord::new(0, source.clone(), first, Vec::new()));
        report
            .model_transitions_mut()
            .push(ModelTransitionRecord::new(
                1,
                source,
                destination.clone(),
                ProjectionPolicy::Strict,
                ProjectionScope::PortableOnly,
                ModelTransitionOutcome::Applied,
                Vec::new(),
            ));
        report
            .steps_mut()
            .push(StepRecord::new(1, destination.clone(), second, Vec::new()));
        let snapshot = RunSnapshot::new(
            checkpoint("checkpoint-terminal", None),
            fingerprints(),
            LanguageRequest::new(messages),
            report,
            Some(DEADLINE),
            ResumePoint::Terminal(SnapshotTerminal::Completed { reason: None }),
        )
        .unwrap();

        assert_eq!(snapshot.target(), &destination);
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
