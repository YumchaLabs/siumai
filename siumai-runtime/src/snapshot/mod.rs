//! Versioned durable run snapshots and lease/CAS persistence hooks.

mod checkpoint;
mod model;
mod store;

pub(crate) use checkpoint::{
    InitialSnapshotParts, assemble_initial_snapshot, assemble_successor_snapshot,
};

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
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Mutex, MutexGuard};
    use std::time::{Duration, Instant};

    use serde::{Deserialize, Serialize};
    use serde_json::json;
    use siumai_core::{
        ApiModeId, ContentAnnotationTarget, ContentPart, LanguageCompletionReason,
        LanguageIncompleteReason, LanguageRequest, LanguageResponse, LanguageTermination, Message,
        MessageAnnotationTarget, MessagePart, MessageRole, Model, ModelDescriptor, ModelFamily,
        ModelId, OpaqueProviderItem, PartialLanguageOutput, PartialLanguageOutputPart, PlatformId,
        ProtocolId, ProviderId, ProviderProvenance, ReplayDomain, ReplayDomainId, RouteId,
        ToolAnnotationTarget, ToolBindingIdentity, ToolCall, ToolOutcome, ToolSpec,
        TypedProviderAnnotation, Usage,
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
        checkpoint_for_run("run-1", id, parent)
    }

    fn checkpoint_for_run(run_id: &str, id: &str, parent: Option<&str>) -> SnapshotCheckpoint {
        SnapshotCheckpoint::new(
            SnapshotEngineVersion::new("runtime-test-v2").unwrap(),
            RunId::new(run_id).unwrap(),
            LineageId::new("lineage-1").unwrap(),
            CheckpointId::new(id).unwrap(),
            parent.map(|parent| CheckpointId::new(parent).unwrap()),
        )
        .unwrap()
    }

    fn tool_call() -> ToolCall {
        ToolCall::local("call-1", "write_record", json!({"password": "tool-secret"}))
            .expect("valid tool call")
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
            LanguageCompletionReason::ToolCalls,
            Usage::default(),
        )
        .unwrap()
    }

    fn report_with_log(log: ToolExecutionLog) -> RunReport {
        let mut report = RunReport::new(
            target(),
            vec![Message::text(MessageRole::User, "history-secret")],
        );
        if !log.events().is_empty() {
            report.accumulate_usage(&Usage::default());
        }
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

    fn deferred_item(status: &str) -> OpaqueProviderItem {
        deferred_item_in_scope(
            status,
            "deferred-platform",
            "deferred-mode",
            "snapshot-deferred",
        )
    }

    fn deferred_item_in_scope(
        status: &str,
        platform: &str,
        api_mode: &str,
        replay_domain: &str,
    ) -> OpaqueProviderItem {
        let target = ModelTarget::new(
            ProviderId::new("deferred-provider").unwrap(),
            ModelId::new("deferred-model").unwrap(),
        )
        .with_platform(PlatformId::new(platform).unwrap())
        .with_protocol(ProtocolId::new("deferred.protocol").unwrap())
        .with_api_mode(ApiModeId::new(api_mode).unwrap())
        .with_replay_domain(ReplayDomain::custom(
            ReplayDomainId::new(replay_domain).unwrap(),
        ));
        OpaqueProviderItem::new(
            ProviderProvenance::from_scope(target.scope(), target.model().clone()).unwrap(),
            "provider.deferred",
            json!({"status": status}),
        )
        .unwrap()
    }

    fn report_with_refusal_step() -> RunReport {
        let mut report = ready_report();
        let response = LanguageResponse::completed(
            vec![ContentPart::Refusal {
                reason: Some("private refusal".to_string()),
            }],
            LanguageCompletionReason::Refusal,
            Usage::default(),
        )
        .unwrap();
        report.accumulate_usage(response.usage());
        report
            .steps_mut()
            .push(StepRecord::new(0, target(), response, Vec::new()));
        report
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

    fn ready_snapshot_for_run(
        run_id: &str,
        checkpoint_id: &str,
        parent_checkpoint_id: Option<&str>,
    ) -> RunSnapshot {
        let report = ready_report();
        RunSnapshot::new(
            checkpoint_for_run(run_id, checkpoint_id, parent_checkpoint_id),
            fingerprints(),
            LanguageRequest::new(report.messages().to_vec()),
            report,
            Some(DEADLINE),
            ResumePoint::ReadyForModel {
                next_step: 0,
                target: target(),
            },
        )
        .unwrap()
    }

    static NEXT_JSON_STORE_ID: AtomicU64 = AtomicU64::new(1);

    #[derive(Debug)]
    struct JsonRoundTripStore {
        store_id: u64,
        state: Mutex<JsonRoundTripState>,
    }

    #[derive(Debug, Default)]
    struct JsonRoundTripState {
        next_lease_token: u64,
        leases: BTreeMap<RunId, JsonLeaseRecord>,
        runs: BTreeMap<RunId, JsonStoredRun>,
    }

    #[derive(Debug, Clone)]
    struct JsonLeaseToken {
        store_id: u64,
        run_id: RunId,
        token: u64,
    }

    #[derive(Debug, Clone, Copy)]
    struct JsonLeaseRecord {
        token: u64,
        expires_at: Instant,
    }

    #[derive(Debug, Clone)]
    struct JsonStoredRun {
        revision: u64,
        snapshot: Vec<u8>,
    }

    impl Default for JsonRoundTripStore {
        fn default() -> Self {
            Self {
                store_id: NEXT_JSON_STORE_ID.fetch_add(1, Ordering::Relaxed),
                state: Mutex::new(JsonRoundTripState {
                    next_lease_token: 1,
                    ..JsonRoundTripState::default()
                }),
            }
        }
    }

    impl JsonRoundTripStore {
        fn lock(&self) -> Result<MutexGuard<'_, JsonRoundTripState>, RunStoreError> {
            self.state.lock().map_err(|_| RunStoreError::Unavailable)
        }

        fn validate_lease(
            &self,
            state: &mut JsonRoundTripState,
            lease: &RunLease,
        ) -> Result<(), RunStoreError> {
            let token = lease
                .store_token::<JsonLeaseToken>()
                .ok_or(RunStoreError::ForeignLease)?;
            if token.store_id != self.store_id || &token.run_id != lease.run_id() {
                return Err(RunStoreError::ForeignLease);
            }
            let Some(record) = state.leases.get(lease.run_id()).copied() else {
                return Err(RunStoreError::LeaseLost);
            };
            if record.token != token.token {
                return Err(RunStoreError::LeaseLost);
            }
            if Instant::now() >= record.expires_at {
                state.leases.remove(lease.run_id());
                return Err(RunStoreError::LeaseExpired);
            }
            Ok(())
        }
    }

    impl RunStore for JsonRoundTripStore {
        fn acquire<'a>(&'a self, run_id: &'a RunId, ttl: Duration) -> RunStoreFuture<'a, RunLease> {
            Box::pin(async move {
                let expires_at = Instant::now()
                    .checked_add(ttl)
                    .filter(|_| !ttl.is_zero())
                    .ok_or(RunStoreError::InvalidLeaseDuration)?;
                let mut state = self.lock()?;
                if state
                    .leases
                    .get(run_id)
                    .is_some_and(|record| Instant::now() < record.expires_at)
                {
                    return Err(RunStoreError::LeaseConflict {
                        run_id: run_id.clone(),
                    });
                }
                state.leases.remove(run_id);
                let token = state.next_lease_token;
                state.next_lease_token = token
                    .checked_add(1)
                    .ok_or(RunStoreError::LeaseTokenExhausted)?;
                state
                    .leases
                    .insert(run_id.clone(), JsonLeaseRecord { token, expires_at });
                Ok(RunLease::from_store_token(
                    run_id.clone(),
                    expires_at,
                    JsonLeaseToken {
                        store_id: self.store_id,
                        run_id: run_id.clone(),
                        token,
                    },
                ))
            })
        }

        fn load<'a>(&'a self, lease: &'a RunLease) -> RunStoreFuture<'a, Option<StoredRun>> {
            Box::pin(async move {
                let mut state = self.lock()?;
                self.validate_lease(&mut state, lease)?;
                state
                    .runs
                    .get(lease.run_id())
                    .map(|stored| {
                        let snapshot = serde_json::from_slice::<RunSnapshot>(&stored.snapshot)
                            .map_err(|_| RunStoreError::Unavailable)?;
                        Ok(StoredRun::new(
                            SnapshotRevision::from_value(stored.revision),
                            snapshot,
                        ))
                    })
                    .transpose()
            })
        }

        fn compare_and_swap<'a>(
            &'a self,
            lease: &'a RunLease,
            expected: SnapshotRevision,
            snapshot: RunSnapshot,
        ) -> RunStoreFuture<'a, SnapshotRevision> {
            Box::pin(async move {
                if snapshot.run_id() != lease.run_id() {
                    return Err(RunStoreError::RunIdMismatch {
                        leased_run_id: lease.run_id().clone(),
                        snapshot_run_id: snapshot.run_id().clone(),
                    });
                }
                let mut state = self.lock()?;
                self.validate_lease(&mut state, lease)?;
                let actual = state.runs.get(lease.run_id()).map_or(0, |run| run.revision);
                if actual != expected.value() {
                    return Err(RunStoreError::CasConflict {
                        expected,
                        actual: SnapshotRevision::from_value(actual),
                    });
                }
                if let Some(current) = state.runs.get(lease.run_id()) {
                    let current = serde_json::from_slice::<RunSnapshot>(&current.snapshot)
                        .map_err(|_| RunStoreError::Unavailable)?;
                    if current.resume_point().is_terminal() {
                        return Err(RunStoreError::RunAlreadyTerminal);
                    }
                }
                let revision = actual
                    .checked_add(1)
                    .ok_or(RunStoreError::RevisionExhausted)?;
                let encoded =
                    serde_json::to_vec(&snapshot).map_err(|_| RunStoreError::Unavailable)?;
                state.runs.insert(
                    lease.run_id().clone(),
                    JsonStoredRun {
                        revision,
                        snapshot: encoded,
                    },
                );
                Ok(SnapshotRevision::from_value(revision))
            })
        }

        fn renew<'a>(&'a self, lease: &'a mut RunLease, ttl: Duration) -> RunStoreFuture<'a, ()> {
            Box::pin(async move {
                let expires_at = Instant::now()
                    .checked_add(ttl)
                    .filter(|_| !ttl.is_zero())
                    .ok_or(RunStoreError::InvalidLeaseDuration)?;
                let mut state = self.lock()?;
                self.validate_lease(&mut state, lease)?;
                state
                    .leases
                    .get_mut(lease.run_id())
                    .ok_or(RunStoreError::LeaseLost)?
                    .expires_at = expires_at;
                lease.set_expires_at(expires_at);
                Ok(())
            })
        }

        fn release<'a>(&'a self, lease: RunLease) -> RunStoreFuture<'a, ()> {
            Box::pin(async move {
                let mut state = self.lock()?;
                self.validate_lease(&mut state, &lease)?;
                state.leases.remove(lease.run_id());
                Ok(())
            })
        }
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

    #[test]
    fn version_seven_round_trip_preserves_language_termination() {
        let mut report = ready_report();
        let response = LanguageResponse::incomplete(
            vec![ContentPart::Text {
                text: "truncated output".to_string(),
            }],
            LanguageIncompleteReason::MaxOutputTokens,
            Usage::default().with_total_tokens(9_u64),
        )
        .unwrap();
        report.accumulate_usage(response.usage());
        report
            .steps_mut()
            .push(StepRecord::new(0, target(), response, Vec::new()));
        let snapshot = snapshot(
            "checkpoint-incomplete",
            None,
            report,
            Some(DEADLINE),
            ResumePoint::ReadyForModel {
                next_step: 1,
                target: target(),
            },
        );

        let encoded = serde_json::to_value(&snapshot).unwrap();
        assert_eq!(encoded["snapshot_version"], json!(7));
        let restored: RunSnapshot = serde_json::from_value(encoded).unwrap();

        assert!(matches!(
            restored.report().steps()[0].response().termination(),
            LanguageTermination::Incomplete(LanguageIncompleteReason::MaxOutputTokens)
        ));
    }

    #[test]
    fn version_seven_requires_explicit_usage_settlement_state() {
        let base_snapshot = ready_snapshot("checkpoint-1", None, ready_report(), Some(DEADLINE));
        let encoded = serde_json::to_value(base_snapshot).unwrap();
        let mut missing = encoded.clone();
        missing["report"]
            .as_object_mut()
            .unwrap()
            .remove("usage_settled");

        let error = serde_json::from_value::<RunSnapshot>(missing).unwrap_err();
        assert!(error.to_string().contains("missing field `usage_settled`"));

        let mut wrong_kind = encoded;
        wrong_kind["report"]["usage_settled"] = json!("not-a-boolean");
        let error = serde_json::from_value::<RunSnapshot>(wrong_kind).unwrap_err();
        assert!(error.to_string().contains("invalid type"));

        let usage = Usage::default().with_output_tokens(4_u64);
        let mut report = ready_report();
        report.accumulate_usage(&usage);
        let partial = PartialLanguageOutput::new(
            vec![PartialLanguageOutputPart::Text {
                text: "partial output".to_string(),
            }],
            usage,
        )
        .unwrap();
        let snapshot = snapshot(
            "checkpoint-partial",
            None,
            report,
            Some(DEADLINE),
            ResumePoint::Terminal(SnapshotTerminal::Failed {
                reason: SnapshotReason::new("provider_failed", None).unwrap(),
                partial: Some(partial),
            }),
        );
        let mut encoded = serde_json::to_value(snapshot).unwrap();
        encoded["report"]["usage_settled"] = json!(false);
        let error = serde_json::from_value::<RunSnapshot>(encoded).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("snapshot usage observations require settled report usage state")
        );
    }

    #[test]
    fn version_seven_rejects_unsettled_unknown_usage_after_a_completed_call() {
        let snapshot = snapshot(
            "checkpoint-settled-unknown",
            None,
            report_with_refusal_step(),
            Some(DEADLINE),
            ResumePoint::ReadyForModel {
                next_step: 1,
                target: target(),
            },
        );
        assert_eq!(snapshot.usage(), &Usage::default());

        let mut encoded = serde_json::to_value(&snapshot).unwrap();
        encoded["report"]["usage_settled"] = json!(false);

        let error = serde_json::from_value::<RunSnapshot>(encoded).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("snapshot usage observations require settled report usage state")
        );

        let mut resumed_report = snapshot.report().clone();
        resumed_report.accumulate_usage(&Usage::default().with_total_tokens(7_u64));
        assert_eq!(resumed_report.usage(), &Usage::default());
    }

    #[test]
    fn version_seven_round_trip_preserves_partial_terminals() {
        let usage = Usage::default().with_output_tokens(4_u64);
        let partial = PartialLanguageOutput::new(
            vec![PartialLanguageOutputPart::Text {
                text: "partial output".to_string(),
            }],
            usage.clone(),
        )
        .unwrap();
        let terminals = [
            ResumePoint::Terminal(SnapshotTerminal::Failed {
                reason: SnapshotReason::new("provider_failed", None).unwrap(),
                partial: Some(partial.clone()),
            }),
            ResumePoint::Terminal(SnapshotTerminal::Cancelled {
                reason: SnapshotReason::new("call_cancelled", None).unwrap(),
                partial: Some(partial.clone()),
            }),
            ResumePoint::Terminal(SnapshotTerminal::Exhausted {
                reason: SnapshotReason::new("runtime_timed_out", None).unwrap(),
                partial: Some(partial),
            }),
        ];

        for (index, resume_point) in terminals.into_iter().enumerate() {
            let mut report = ready_report();
            report.accumulate_usage(&usage);
            let snapshot = snapshot(
                &format!("checkpoint-terminal-{index}"),
                None,
                report,
                Some(DEADLINE),
                resume_point,
            );
            let restored: RunSnapshot =
                serde_json::from_value(serde_json::to_value(&snapshot).unwrap()).unwrap();
            let ResumePoint::Terminal(terminal) = restored.resume_point() else {
                panic!("expected terminal resume point");
            };
            match terminal {
                SnapshotTerminal::Failed { partial, .. }
                | SnapshotTerminal::Cancelled { partial, .. }
                | SnapshotTerminal::Exhausted { partial, .. } => {
                    assert_eq!(
                        partial
                            .as_ref()
                            .and_then(|value| value.usage().output_tokens.value()),
                        Some(4)
                    );
                }
                _ => panic!("expected a partial terminal"),
            }
        }
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

    async fn assert_atomic_store_contract(store: &dyn RunStore) {
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

        assert_eq!(
            store
                .compare_and_swap(
                    &lease,
                    revision,
                    ready_snapshot_for_run("another-run", "checkpoint-other", None),
                )
                .await
                .unwrap_err(),
            RunStoreError::RunIdMismatch {
                leased_run_id: run_id.clone(),
                snapshot_run_id: RunId::new("another-run").unwrap(),
            }
        );
        assert_eq!(
            store
                .compare_and_swap(
                    &lease,
                    SnapshotRevision::EMPTY,
                    ready_snapshot(
                        "checkpoint-stale",
                        Some("checkpoint-1"),
                        ready_report(),
                        Some(DEADLINE),
                    ),
                )
                .await
                .unwrap_err(),
            RunStoreError::CasConflict {
                expected: SnapshotRevision::EMPTY,
                actual: revision,
            }
        );

        let mut terminal_report = ready_report();
        terminal_report.accumulate_usage(&Usage::default());
        let terminal = snapshot(
            "checkpoint-2",
            Some("checkpoint-1"),
            terminal_report,
            Some(DEADLINE),
            ResumePoint::Terminal(SnapshotTerminal::Completed { reason: None }),
        );
        let terminal_revision = store
            .compare_and_swap(&lease, revision, terminal)
            .await
            .unwrap();
        assert_eq!(
            store
                .compare_and_swap(
                    &lease,
                    terminal_revision,
                    ready_snapshot(
                        "checkpoint-3",
                        Some("checkpoint-2"),
                        ready_report(),
                        Some(DEADLINE),
                    ),
                )
                .await
                .unwrap_err(),
            RunStoreError::RunAlreadyTerminal
        );
        store.release(lease).await.unwrap();
    }

    #[tokio::test]
    async fn built_in_and_external_store_share_atomic_cas_contract() {
        let built_in = InMemoryRunStore::new();
        assert_atomic_store_contract(&built_in).await;

        let external = JsonRoundTripStore::default();
        assert_atomic_store_contract(&external).await;
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

    #[tokio::test]
    async fn expired_lease_is_fenced_after_a_new_owner_acquires() {
        let store = InMemoryRunStore::new();
        let run_id = RunId::new("run-1").unwrap();
        let mut old_lease = store
            .acquire(&run_id, Duration::from_secs(30))
            .await
            .unwrap();
        let initial = ready_snapshot("checkpoint-1", None, ready_report(), Some(DEADLINE));
        let revision = store
            .compare_and_swap(&old_lease, SnapshotRevision::EMPTY, initial)
            .await
            .unwrap();

        store.expire_lease_for_test(&old_lease).unwrap();
        let new_lease = store
            .acquire(&run_id, Duration::from_secs(30))
            .await
            .unwrap();
        let successor = ready_snapshot(
            "checkpoint-2",
            Some("checkpoint-1"),
            ready_report(),
            Some(DEADLINE),
        );

        assert_eq!(
            store.load(&old_lease).await.unwrap_err(),
            RunStoreError::LeaseLost
        );
        assert_eq!(
            store
                .compare_and_swap(&old_lease, revision, successor.clone())
                .await
                .unwrap_err(),
            RunStoreError::LeaseLost
        );
        assert_eq!(
            store
                .renew(&mut old_lease, Duration::from_secs(30))
                .await
                .unwrap_err(),
            RunStoreError::LeaseLost
        );
        assert_eq!(
            store.release(old_lease).await.unwrap_err(),
            RunStoreError::LeaseLost
        );

        let next_revision = store
            .compare_and_swap(&new_lease, revision, successor)
            .await
            .unwrap();
        assert_eq!(next_revision.value(), revision.value() + 1);
        store.release(new_lease).await.unwrap();
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

    #[test]
    fn deserialization_rejects_real_version_six_before_payload_decode() {
        let snapshot = snapshot(
            "checkpoint-1",
            None,
            report_with_refusal_step(),
            Some(DEADLINE),
            ResumePoint::ReadyForModel {
                next_step: 1,
                target: target(),
            },
        );
        let mut value = serde_json::to_value(snapshot).unwrap();
        value["snapshot_version"] = json!(6);
        let response = value["report"]["steps"][0]["response"]
            .as_object_mut()
            .unwrap();
        response.remove("termination");
        response.insert("status".to_string(), json!("Completed"));
        response.insert("finish_reason".to_string(), json!("Refusal"));

        let error = serde_json::from_value::<RunSnapshot>(value).unwrap_err();
        let public = error.to_string();
        assert!(public.contains("unsupported run snapshot version 6"));
        assert!(!public.contains("missing field `termination`"));
        assert!(!public.contains("history-secret"));
        assert!(!public.contains("tool-secret"));
    }

    #[test]
    fn deserialization_rejects_forged_assistant_history_omissions() {
        let snapshot = snapshot(
            "checkpoint-1",
            None,
            report_with_refusal_step(),
            Some(DEADLINE),
            ResumePoint::ReadyForModel {
                next_step: 1,
                target: target(),
            },
        );
        let mut value = serde_json::to_value(snapshot).unwrap();
        value["report"]["steps"][0]["assistant_history_omissions"] = json!([]);

        let error = serde_json::from_value::<RunSnapshot>(value).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("invalid assistant-history omission records")
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
                partial: None,
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
                partial: None,
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
    async fn store_cas_does_not_own_successor_semantics() {
        let current = ready_snapshot("checkpoint-1", None, ready_report(), Some(DEADLINE));
        let (store, lease, revision) = store_current(current).await;
        let successor = ready_snapshot("checkpoint-2", None, ready_report(), Some(DEADLINE));

        let next_revision = store
            .compare_and_swap(&lease, revision, successor)
            .await
            .unwrap();
        assert_eq!(next_revision.value(), revision.value() + 1);
    }

    #[test]
    fn successor_rejects_message_history_regression() {
        let current = ready_snapshot("checkpoint-1", None, ready_report(), Some(DEADLINE));
        let successor_report = RunReport::new(target(), Vec::new());
        let successor = ready_snapshot(
            "checkpoint-2",
            Some("checkpoint-1"),
            successor_report,
            Some(DEADLINE),
        );

        assert_eq!(
            current.validate_successor(&successor).unwrap_err(),
            RunSnapshotSuccessorError::MessageHistoryRegression
        );
    }

    #[test]
    fn successor_accepts_only_an_exact_reprojection_for_history_regression() {
        let source = ModelTarget::new(
            ProviderId::new("source-provider").unwrap(),
            ModelId::new("source-model").unwrap(),
        )
        .with_protocol(ProtocolId::new("source.responses").unwrap())
        .with_replay_domain(ReplayDomain::custom(
            ReplayDomainId::new("snapshot-source").unwrap(),
        ));
        let destination = ModelTarget::new(
            ProviderId::new("destination-provider").unwrap(),
            ModelId::new("destination-model").unwrap(),
        )
        .with_protocol(ProtocolId::new("destination.messages").unwrap())
        .with_replay_domain(ReplayDomain::custom(
            ReplayDomainId::new("snapshot-destination").unwrap(),
        ));
        let native = OpaqueProviderItem::new(
            ProviderProvenance::from_scope(source.scope(), source.model().clone()).unwrap(),
            "response.output",
            json!({"id": "native-only"}),
        )
        .unwrap();
        let source_response = LanguageResponse::completed(
            vec![ContentPart::ProviderOpaque(native.clone())],
            LanguageCompletionReason::Stop,
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
        previous_report.accumulate_usage(source_response.usage());
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
            LanguageCompletionReason::Stop,
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
        successor_report.accumulate_usage(final_response.usage());
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
        previous.validate_successor(&successor).unwrap();
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
            LanguageCompletionReason::Stop,
            Usage::default(),
        )
        .unwrap();
        let messages = vec![Message::text(MessageRole::User, "continue")];
        let mut report = RunReport::new(source.clone(), messages.clone());
        report.accumulate_usage(response.usage());
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
            LanguageCompletionReason::Stop,
            Usage::default(),
        )
        .unwrap();
        let second = LanguageResponse::completed(
            vec![ContentPart::Text {
                text: "second step".to_string(),
            }],
            LanguageCompletionReason::Stop,
            Usage::default(),
        )
        .unwrap();
        let messages = vec![Message::text(MessageRole::User, "continue")];
        let mut report = RunReport::new(source.clone(), messages.clone());
        report.accumulate_usage(first.usage());
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
        report.accumulate_usage(second.usage());
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

    #[test]
    fn successor_rejects_budget_regression() {
        let mut current_report = ready_report();
        current_report
            .budget_mut()
            .charge_model_step(&RunBudget::default())
            .unwrap();
        let current = ready_snapshot("checkpoint-1", None, current_report, Some(DEADLINE));
        let successor = ready_snapshot(
            "checkpoint-2",
            Some("checkpoint-1"),
            ready_report(),
            Some(DEADLINE),
        );

        assert_eq!(
            current.validate_successor(&successor).unwrap_err(),
            RunSnapshotSuccessorError::BudgetRegression {
                dimension: "model_steps",
                previous: 1,
                next: 0,
            }
        );
    }

    #[test]
    fn successor_rejects_usage_settlement_regression_with_unknown_usage() {
        let mut current_report = ready_report();
        current_report.accumulate_usage(&Usage::default());
        let current = ready_snapshot("checkpoint-1", None, current_report, Some(DEADLINE));
        let successor = ready_snapshot(
            "checkpoint-2",
            Some("checkpoint-1"),
            ready_report(),
            Some(DEADLINE),
        );

        assert_eq!(
            current.validate_successor(&successor).unwrap_err(),
            RunSnapshotSuccessorError::UsageSettlementRegression
        );
    }

    #[test]
    fn provider_deferred_successor_accepts_same_key_payload_update() {
        let mut current_report = ready_report();
        let queued = deferred_item("queued");
        current_report.observe_provider_deferred("provider-state-1", &queued);
        let current = ready_snapshot("checkpoint-1", None, current_report, Some(DEADLINE));

        let mut successor_report = current.report().clone();
        let in_progress = deferred_item("in_progress");
        successor_report.observe_provider_deferred("provider-state-1", &in_progress);
        let successor = ready_snapshot(
            "checkpoint-2",
            Some("checkpoint-1"),
            successor_report,
            Some(DEADLINE),
        );

        current.validate_successor(&successor).unwrap();
        assert_eq!(
            successor.report().provider_deferred()[0].item().data()["status"],
            "in_progress"
        );
    }

    #[test]
    fn provider_deferred_identity_includes_complete_replay_scope() {
        let observations = [
            deferred_item_in_scope("platform", "platform-a", "mode-a", "domain-a"),
            deferred_item_in_scope("api-mode", "platform-b", "mode-b", "domain-a"),
            deferred_item_in_scope("replay-domain", "platform-b", "mode-a", "domain-b"),
            deferred_item_in_scope("baseline", "platform-b", "mode-a", "domain-a"),
        ];
        let mut report = ready_report();
        for item in &observations {
            report.observe_provider_deferred("shared-correlation", item);
        }

        assert_eq!(report.provider_deferred().len(), observations.len());
        let round_trip: RunReport = serde_json::from_value(serde_json::to_value(&report).unwrap())
            .expect("distinct replay scopes remain valid after serialization");
        assert_eq!(round_trip.provider_deferred().len(), observations.len());
    }

    #[test]
    fn provider_deferred_successor_rejects_scope_substitution() {
        let mut current_report = ready_report();
        let current_item = deferred_item_in_scope("queued", "platform-a", "mode-a", "domain-a");
        current_report.observe_provider_deferred("shared-correlation", &current_item);
        let current = ready_snapshot("checkpoint-1", None, current_report, Some(DEADLINE));

        let mut successor_report = ready_report();
        let substituted_item = deferred_item_in_scope("queued", "platform-a", "mode-a", "domain-b");
        successor_report.observe_provider_deferred("shared-correlation", &substituted_item);
        let successor = ready_snapshot(
            "checkpoint-2",
            Some("checkpoint-1"),
            successor_report,
            Some(DEADLINE),
        );

        assert_eq!(
            current.validate_successor(&successor).unwrap_err(),
            RunSnapshotSuccessorError::ProviderHistoryRegression
        );
    }

    #[test]
    fn provider_deferred_successor_rejects_key_deletion_reorder_and_substitution() {
        let mut current_report = ready_report();
        let first = deferred_item("first");
        let second = deferred_item("second");
        current_report.observe_provider_deferred("provider-state-1", &first);
        current_report.observe_provider_deferred("provider-state-2", &second);
        let current = ready_snapshot("checkpoint-1", None, current_report, Some(DEADLINE));

        let deletion = ready_snapshot(
            "checkpoint-2",
            Some("checkpoint-1"),
            ready_report(),
            Some(DEADLINE),
        );
        assert_eq!(
            current.validate_successor(&deletion).unwrap_err(),
            RunSnapshotSuccessorError::ProviderHistoryRegression
        );

        let mut reordered_report = ready_report();
        reordered_report.observe_provider_deferred("provider-state-2", &second);
        reordered_report.observe_provider_deferred("provider-state-1", &first);
        let reordered = ready_snapshot(
            "checkpoint-3",
            Some("checkpoint-1"),
            reordered_report,
            Some(DEADLINE),
        );
        assert_eq!(
            current.validate_successor(&reordered).unwrap_err(),
            RunSnapshotSuccessorError::ProviderHistoryRegression
        );

        let mut substituted_report = ready_report();
        substituted_report.observe_provider_deferred("provider-state-3", &first);
        substituted_report.observe_provider_deferred("provider-state-2", &second);
        let substituted = ready_snapshot(
            "checkpoint-4",
            Some("checkpoint-1"),
            substituted_report,
            Some(DEADLINE),
        );
        assert_eq!(
            current.validate_successor(&substituted).unwrap_err(),
            RunSnapshotSuccessorError::ProviderHistoryRegression
        );
    }

    #[test]
    fn deserialization_rejects_duplicate_provider_deferred_keys_and_unbounded_ids() {
        let mut report = ready_report();
        let item = deferred_item("queued");
        report.observe_provider_deferred("provider-state-1", &item);
        let snapshot = ready_snapshot("checkpoint-1", None, report, Some(DEADLINE));
        let encoded = serde_json::to_value(&snapshot).unwrap();
        assert_eq!(
            encoded["report"]["provider_deferred"][0]["correlation_id"],
            "provider-state-1"
        );
        assert!(encoded["report"]["provider_deferred"][0]["item"].is_object());

        let mut duplicate = encoded.clone();
        let observation = duplicate["report"]["provider_deferred"][0].clone();
        duplicate["report"]["provider_deferred"] = json!([observation.clone(), observation]);
        let report_error =
            serde_json::from_value::<RunReport>(duplicate["report"].clone()).unwrap_err();
        assert!(
            report_error
                .to_string()
                .contains("provider-deferred observation keys")
        );
        let error = serde_json::from_value::<RunSnapshot>(duplicate).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("provider-deferred observation keys")
        );

        let mut oversized = encoded;
        oversized["report"]["provider_deferred"][0]["correlation_id"] = json!("x".repeat(1_025));
        let error = serde_json::from_value::<RunSnapshot>(oversized).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("provider correlation identifier")
        );
    }

    #[test]
    fn provider_deferred_observation_debug_redacts_identity_and_payload() {
        let mut report = ready_report();
        let item = deferred_item("private-payload");
        report.observe_provider_deferred("correlation-secret", &item);

        let debug = format!("{:?}", report.provider_deferred()[0]);
        assert!(!debug.contains("correlation-secret"));
        assert!(!debug.contains("private-payload"));
    }

    #[test]
    fn successor_rejects_deadline_extension() {
        let current = ready_snapshot("checkpoint-1", None, ready_report(), Some(DEADLINE));
        let successor = ready_snapshot(
            "checkpoint-2",
            Some("checkpoint-1"),
            ready_report(),
            Some(DEADLINE + 1),
        );

        assert_eq!(
            current.validate_successor(&successor).unwrap_err(),
            RunSnapshotSuccessorError::DeadlineRegression {
                previous: Some(DEADLINE),
                next: Some(DEADLINE + 1),
            }
        );
    }

    #[test]
    fn successor_rejects_execution_log_regression() {
        let current = ready_snapshot(
            "checkpoint-1",
            None,
            report_with_log(completed_log()),
            Some(DEADLINE),
        );
        let successor = ready_snapshot(
            "checkpoint-2",
            Some("checkpoint-1"),
            ready_report(),
            Some(DEADLINE),
        );

        assert_eq!(
            current.validate_successor(&successor).unwrap_err(),
            RunSnapshotSuccessorError::ExecutionLogRegression
        );
    }

    #[test]
    fn successor_rejects_illegal_resume_transition() {
        let current = snapshot(
            "checkpoint-1",
            None,
            report_with_log(prepared_log()),
            Some(DEADLINE),
            ResumePoint::ReadyToDispatch(pending_step(Vec::new())),
        );
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
            current.validate_successor(&successor).unwrap_err(),
            RunSnapshotSuccessorError::InvalidResumeTransition {
                from: ResumePointKind::ReadyToDispatch,
                to: ResumePointKind::AwaitingProvider,
            }
        );
    }

    #[test]
    fn successor_accepts_approval_resolution_to_ready_to_dispatch() {
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
        current.validate_successor(&successor).unwrap();
    }

    #[test]
    fn snapshot_debug_redacts_payloads() {
        let provider_step = PendingProviderStepSnapshot::new(
            0,
            target(),
            tool_response(),
            vec![ProviderStateSnapshot {
                namespace: "provider-deferred:test-provider:test-protocol".to_string(),
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
