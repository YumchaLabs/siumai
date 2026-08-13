use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};
use siumai_core::stream::established_stream;
use siumai_core::{
    CallOptions, ContentPart, Error, ErrorKind, GenerationConfig, LanguageCallError,
    LanguageCompletionReason, LanguageModel, LanguageRequest, LanguageResponse, LanguageStream,
    LanguageStreamEvent, Message, MessageRole, Model, ModelDescriptor, ModelFamily, ModelId,
    OpaqueProviderItem, ProtocolId, ProviderId, ProviderProvenance, ReplayDomain, ReplayDomainId,
    StreamTerminal, StructuredOutputSpec, ToolCall, ToolChoice, ToolOutcome, ToolSpec, Usage,
};
use siumai_runtime::approval::{
    ApprovalClaims, ApprovalEnvelope, ApprovalVerifier, ApprovalVerifierError,
    InMemoryApprovalConsumeStore, TrustContext, TrustIdentity,
};
use siumai_runtime::snapshot::{
    InMemoryRunStore, LineageId, ResumePoint, RunId, RunLease, RunStore, RunStoreError,
    RunStoreFuture, SnapshotFingerprint, SnapshotRevision, StoredRun, ToolExecutionStatus,
};
use siumai_runtime::tool::{
    ApprovalPolicy, RecoveryPolicy, ToolBinding, ToolExecutionRequest, ToolIdempotencyKey, ToolSet,
};
use siumai_runtime::{
    DurableApproval, DurableResume, DurableRunError, DurableToolLoop, IndeterminateRecoveryPolicy,
    ModelTransitionOutcome, StepModelContext, StepModelSelectorIdentity,
    VersionedStepModelSelector,
};

type AttemptLog = Arc<Mutex<Vec<(u32, Option<String>)>>>;

struct ScriptedModel {
    descriptor: ModelDescriptor,
    responses: Mutex<VecDeque<LanguageResponse>>,
    requests: Mutex<Vec<LanguageRequest>>,
}

struct DeferredModel {
    descriptor: ModelDescriptor,
    item: OpaqueProviderItem,
}

impl DeferredModel {
    fn new(item: OpaqueProviderItem) -> Arc<Self> {
        Arc::new(Self {
            descriptor: ModelDescriptor::new(
                ProviderId::new("deferred-test").expect("valid provider"),
                ModelId::new("deferred-model").expect("valid model"),
                ModelFamily::Language,
            )
            .with_protocol(ProtocolId::new("native-orchestration").expect("valid protocol"))
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("durable-deferred-test").expect("valid replay domain"),
            )),
            item,
        })
    }
}

impl Model for DeferredModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl LanguageModel for DeferredModel {
    async fn generate(
        &self,
        _request: LanguageRequest,
        _options: CallOptions,
    ) -> Result<LanguageResponse, LanguageCallError> {
        Err(Error::new(
            ErrorKind::Internal,
            "deferred test model must use streaming",
        )
        .into())
    }

    async fn stream(
        &self,
        _request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        let item = self.item.clone();
        let cancellation = options.cancellation().clone();
        Ok(established_stream(cancellation, move |_| {
            futures::stream::iter([
                Ok(LanguageStreamEvent::ProviderDeferred {
                    id: "provider-state-1".to_string(),
                    state: item,
                }),
                Ok(LanguageStreamEvent::Terminal(StreamTerminal::Completed {
                    response: Box::new(final_response()),
                })),
            ])
        }))
    }
}

impl ScriptedModel {
    fn new(responses: impl IntoIterator<Item = LanguageResponse>) -> Arc<Self> {
        Self::named("durable-test", "durable-model", responses)
    }

    fn named(
        provider: &str,
        model: &str,
        responses: impl IntoIterator<Item = LanguageResponse>,
    ) -> Arc<Self> {
        Arc::new(Self {
            descriptor: ModelDescriptor::new(
                ProviderId::new(provider).expect("valid provider"),
                ModelId::new(model).expect("valid model"),
                ModelFamily::Language,
            ),
            responses: Mutex::new(responses.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
        })
    }

    fn requests(&self) -> Vec<LanguageRequest> {
        self.requests.lock().expect("request lock").clone()
    }
}

impl Model for ScriptedModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl LanguageModel for ScriptedModel {
    async fn generate(
        &self,
        _request: LanguageRequest,
        _options: CallOptions,
    ) -> Result<LanguageResponse, LanguageCallError> {
        Err(Error::new(
            ErrorKind::Internal,
            "durable tool loop must use streaming model steps",
        )
        .into())
    }

    async fn stream(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        self.requests.lock().expect("request lock").push(request);
        let response = self
            .responses
            .lock()
            .expect("response lock")
            .pop_front()
            .ok_or_else(|| Error::new(ErrorKind::Internal, "missing scripted response"))?;
        let cancellation = options.cancellation().clone();
        Ok(established_stream(cancellation, move |_| {
            futures::stream::iter([Ok(LanguageStreamEvent::Terminal(
                StreamTerminal::Completed {
                    response: Box::new(response),
                },
            ))])
        }))
    }
}

#[derive(Debug)]
struct FailOnceStore {
    inner: InMemoryRunStore,
    fail_on_cas: usize,
    cas_calls: AtomicUsize,
    release_calls: AtomicUsize,
}

impl FailOnceStore {
    fn new(fail_on_cas: usize) -> Arc<Self> {
        Arc::new(Self {
            inner: InMemoryRunStore::new(),
            fail_on_cas,
            cas_calls: AtomicUsize::new(0),
            release_calls: AtomicUsize::new(0),
        })
    }

    fn release_calls(&self) -> usize {
        self.release_calls.load(Ordering::SeqCst)
    }
}

impl RunStore for FailOnceStore {
    fn acquire<'a>(&'a self, run_id: &'a RunId, ttl: Duration) -> RunStoreFuture<'a, RunLease> {
        self.inner.acquire(run_id, ttl)
    }

    fn load<'a>(&'a self, lease: &'a RunLease) -> RunStoreFuture<'a, Option<StoredRun>> {
        self.inner.load(lease)
    }

    fn compare_and_swap<'a>(
        &'a self,
        lease: &'a RunLease,
        expected: SnapshotRevision,
        snapshot: siumai_runtime::snapshot::RunSnapshot,
    ) -> RunStoreFuture<'a, SnapshotRevision> {
        let call = self.cas_calls.fetch_add(1, Ordering::SeqCst) + 1;
        if call == self.fail_on_cas {
            return Box::pin(async { Err(RunStoreError::Unavailable) });
        }
        self.inner.compare_and_swap(lease, expected, snapshot)
    }

    fn renew<'a>(&'a self, lease: &'a mut RunLease, ttl: Duration) -> RunStoreFuture<'a, ()> {
        self.inner.renew(lease, ttl)
    }

    fn release<'a>(&'a self, lease: RunLease) -> RunStoreFuture<'a, ()> {
        self.release_calls.fetch_add(1, Ordering::SeqCst);
        self.inner.release(lease)
    }
}

#[derive(Default)]
struct StaticVerifier {
    claims: Mutex<Option<ApprovalClaims>>,
}

#[derive(Default)]
struct MapVerifier {
    claims: Mutex<BTreeMap<Vec<u8>, ApprovalClaims>>,
}

impl MapVerifier {
    fn insert(&self, envelope: &ApprovalEnvelope, claims: ApprovalClaims) {
        self.claims
            .lock()
            .expect("claims lock")
            .insert(envelope.as_bytes().to_vec(), claims);
    }
}

impl ApprovalVerifier for MapVerifier {
    fn verify(&self, envelope: &ApprovalEnvelope) -> Result<ApprovalClaims, ApprovalVerifierError> {
        self.claims
            .lock()
            .expect("claims lock")
            .get(envelope.as_bytes())
            .cloned()
            .ok_or(ApprovalVerifierError::Rejected)
    }
}

impl StaticVerifier {
    fn set_claims(&self, claims: ApprovalClaims) {
        *self.claims.lock().expect("claims lock") = Some(claims);
    }
}

impl ApprovalVerifier for StaticVerifier {
    fn verify(
        &self,
        _envelope: &ApprovalEnvelope,
    ) -> Result<ApprovalClaims, ApprovalVerifierError> {
        self.claims
            .lock()
            .expect("claims lock")
            .clone()
            .ok_or(ApprovalVerifierError::Unavailable)
    }
}

fn request() -> LanguageRequest {
    LanguageRequest::new(vec![Message::text(MessageRole::User, "run the tool")])
}

fn tool_spec() -> ToolSpec {
    ToolSpec::new(
        "write_record",
        Some("write one record".to_string()),
        json!({"type": "object"}),
    )
    .expect("valid tool spec")
}

fn tool_call() -> ToolCall {
    ToolCall::local("call-1", "write_record", json!({"value": 7})).expect("valid tool call")
}

fn second_tool_call() -> ToolCall {
    ToolCall::local("call-2", "write_record_2", json!({"value": 8})).expect("valid tool call")
}

fn tool_response() -> LanguageResponse {
    LanguageResponse::completed(
        vec![ContentPart::ToolCall(tool_call())],
        LanguageCompletionReason::ToolCalls,
        Usage::default(),
    )
    .expect("valid tool response")
}

fn two_tool_response() -> LanguageResponse {
    LanguageResponse::completed(
        vec![
            ContentPart::ToolCall(tool_call()),
            ContentPart::ToolCall(second_tool_call()),
        ],
        LanguageCompletionReason::ToolCalls,
        Usage::default(),
    )
    .expect("valid two-tool response")
}

fn final_response() -> LanguageResponse {
    LanguageResponse::completed(
        vec![ContentPart::Text {
            text: "done".to_string(),
        }],
        LanguageCompletionReason::Stop,
        Usage::default(),
    )
    .expect("valid final response")
}

fn fingerprints() -> (SnapshotFingerprint, SnapshotFingerprint) {
    (
        SnapshotFingerprint::new("sha256:test-options").expect("valid fingerprint"),
        SnapshotFingerprint::new("sha256:test-approval-policy").expect("valid fingerprint"),
    )
}

fn durable_loop(
    model: Arc<dyn LanguageModel>,
    tools: ToolSet,
    store: Arc<dyn RunStore>,
) -> DurableToolLoop {
    let (options, approval) = fingerprints();
    DurableToolLoop::new(model, tools, store, options, approval).expect("valid durable loop")
}

fn not_required_binding(executions: Arc<AtomicUsize>, attempts: Option<AttemptLog>) -> ToolBinding {
    ToolBinding::from_fn(
        tool_spec(),
        "v1",
        |_| Ok(()),
        move |request: ToolExecutionRequest| {
            let executions = Arc::clone(&executions);
            let attempts = attempts.clone();
            async move {
                executions.fetch_add(1, Ordering::SeqCst);
                if let Some(attempts) = attempts {
                    attempts.lock().expect("attempt lock").push((
                        request.attempt().get(),
                        request
                            .idempotency_key()
                            .map(|key| key.as_str().to_string()),
                    ));
                }
                Ok(ToolOutcome::Success {
                    value: Value::Bool(true),
                })
            }
        },
    )
    .expect("valid binding")
    .with_approval_policy(ApprovalPolicy::NotRequired)
}

fn required_binding(name: &str, executions: Arc<AtomicUsize>, revision: &str) -> ToolBinding {
    let spec = ToolSpec::new(
        name,
        Some("write one record".to_string()),
        json!({"type": "object"}),
    )
    .expect("valid tool spec");
    ToolBinding::from_fn(
        spec,
        revision,
        |_| Ok(()),
        move |_| {
            let executions = Arc::clone(&executions);
            async move {
                executions.fetch_add(1, Ordering::SeqCst);
                Ok(ToolOutcome::Success { value: json!(true) })
            }
        },
    )
    .expect("valid binding")
    .with_approval_policy(ApprovalPolicy::Required)
}

fn run_id(suffix: &str) -> RunId {
    RunId::new(format!("run-{suffix}")).expect("valid run id")
}

fn lineage_id(suffix: &str) -> LineageId {
    LineageId::new(format!("lineage-{suffix}")).expect("valid lineage id")
}

#[tokio::test]
async fn completed_checkpoint_is_never_replayed() {
    let executions = Arc::new(AtomicUsize::new(0));
    let binding = not_required_binding(Arc::clone(&executions), None);
    let tools = ToolSet::from_bindings([binding]).expect("unique tool");
    let store = Arc::new(InMemoryRunStore::new());
    let model = ScriptedModel::new([tool_response(), final_response()]);
    let loop_ = durable_loop(model, tools, store);
    let run = run_id("completed");

    let completed = loop_
        .start(
            run.clone(),
            lineage_id("completed"),
            request(),
            CallOptions::default(),
        )
        .await
        .expect("run completes");
    assert!(completed.is_terminal());
    assert_eq!(executions.load(Ordering::SeqCst), 1);

    let resumed = loop_
        .resume(&run, DurableResume::default(), CallOptions::default())
        .await
        .expect("terminal load is idempotent");
    assert!(resumed.is_terminal());
    assert_eq!(executions.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn durable_continuation_preserves_complete_request_state() {
    let executions = Arc::new(AtomicUsize::new(0));
    let binding = not_required_binding(Arc::clone(&executions), None);
    let tools = ToolSet::from_bindings([binding]).expect("unique tool");
    let store = Arc::new(InMemoryRunStore::new());
    let model = ScriptedModel::new([tool_response(), final_response()]);
    let loop_ = durable_loop(model.clone(), tools, store);
    let mut request = request();
    request.generation = GenerationConfig {
        max_output_tokens: Some(321),
        temperature: Some(0.25),
        top_p: Some(0.8),
        stop_sequences: vec!["STOP".to_string()],
        seed: Some(42),
    };
    request.tool_choice = Some(ToolChoice::Required);
    request.structured_output = Some(StructuredOutputSpec {
        name: "durable_result".to_string(),
        description: Some("A durable result".to_string()),
        schema: json!({"type": "object"}),
        strict: true,
    });

    let completed = loop_
        .start(
            run_id("continuation"),
            lineage_id("continuation"),
            request.clone(),
            CallOptions::default(),
        )
        .await
        .expect("run completes");

    let requests = model.requests();
    assert_eq!(requests.len(), 2);
    for observed in &requests {
        assert_eq!(observed.generation, request.generation);
        assert_eq!(observed.tool_choice, request.tool_choice);
        assert_eq!(observed.structured_output, request.structured_output);
        assert_eq!(observed.tools.len(), 1);
        assert_eq!(observed.tools[0].name(), "write_record");
    }
    let continuation = completed.snapshot().continuation();
    assert_eq!(continuation.generation, request.generation);
    assert_eq!(continuation.tool_choice, request.tool_choice);
    assert_eq!(continuation.structured_output, request.structured_output);
    assert_eq!(continuation.tools.len(), 1);
}

#[tokio::test]
async fn durable_selector_freezes_target_and_records_reproducible_transition() {
    let executions = Arc::new(AtomicUsize::new(0));
    let binding = not_required_binding(Arc::clone(&executions), None);
    let tools = ToolSet::from_bindings([binding]).expect("unique tool");
    let store = Arc::new(InMemoryRunStore::new());
    let source = ScriptedModel::named("durable-source", "source", [tool_response()]);
    let target = ScriptedModel::named("durable-target", "target", [final_response()]);
    let target_model: Arc<dyn LanguageModel> = target.clone();
    let selector_calls = Arc::new(AtomicUsize::new(0));
    let observed_selector_calls = Arc::clone(&selector_calls);
    let selector = VersionedStepModelSelector::new(
        StepModelSelectorIdentity::new(
            7,
            SnapshotFingerprint::new("sha256:durable-selector-v7")
                .expect("valid selector fingerprint"),
        ),
        move |_: StepModelContext<'_>| {
            observed_selector_calls.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::clone(&target_model))
        },
    );
    let source_model: Arc<dyn LanguageModel> = source.clone();
    let loop_ = durable_loop(source_model, tools, store).with_model_selector(selector);

    let completed = loop_
        .start(
            run_id("selector"),
            lineage_id("selector"),
            request(),
            CallOptions::default(),
        )
        .await
        .expect("run completes after switching models");

    assert_eq!(selector_calls.load(Ordering::SeqCst), 1);
    assert_eq!(source.requests().len(), 1);
    assert_eq!(target.requests().len(), 1);
    let transitions = completed.snapshot().report().model_transitions();
    assert_eq!(transitions.len(), 1);
    assert_eq!(transitions[0].outcome(), ModelTransitionOutcome::Applied);
    assert_eq!(
        transitions[0].target().provider().as_str(),
        "durable-target"
    );
    let identity = completed
        .snapshot()
        .fingerprints()
        .model_selector
        .as_ref()
        .expect("selector identity is persisted");
    assert_eq!(identity.version(), 7);
}

#[tokio::test]
async fn durable_resume_rejects_selector_target_drift() {
    let executions = Arc::new(AtomicUsize::new(0));
    let binding = not_required_binding(Arc::clone(&executions), None);
    let tools = ToolSet::from_bindings([binding]).expect("unique tool");
    // Fail after the selected target is frozen but before the destination
    // model is called. Resume must re-evaluate the selector against that
    // durable boundary and reject drift.
    let store = FailOnceStore::new(5);
    let source = ScriptedModel::named("durable-source", "source", [tool_response()]);
    let first = ScriptedModel::named("durable-target", "first", []);
    let second = ScriptedModel::named("durable-target", "second", []);
    let first_model: Arc<dyn LanguageModel> = first;
    let second_model: Arc<dyn LanguageModel> = second;
    let selector_calls = Arc::new(AtomicUsize::new(0));
    let observed_selector_calls = Arc::clone(&selector_calls);
    let selector = VersionedStepModelSelector::new(
        StepModelSelectorIdentity::new(
            1,
            SnapshotFingerprint::new("sha256:drifting-selector")
                .expect("valid selector fingerprint"),
        ),
        move |_: StepModelContext<'_>| {
            if observed_selector_calls.fetch_add(1, Ordering::SeqCst) == 0 {
                Ok(Arc::clone(&first_model))
            } else {
                Ok(Arc::clone(&second_model))
            }
        },
    );
    let source_model: Arc<dyn LanguageModel> = source;
    let loop_ = durable_loop(source_model, tools, store).with_model_selector(selector);

    let run = run_id("selector-drift");
    let initial = loop_
        .start(
            run.clone(),
            lineage_id("selector-drift"),
            request(),
            CallOptions::default(),
        )
        .await
        .expect_err("checkpoint failure leaves the frozen target resumable");
    assert!(matches!(
        initial,
        DurableRunError::Store(RunStoreError::Unavailable)
    ));

    let error = loop_
        .resume(&run, DurableResume::default(), CallOptions::default())
        .await
        .expect_err("selector target drift must fail closed on resume");

    assert!(matches!(
        error,
        DurableRunError::SelectedModelTargetChanged { .. }
    ));
    assert_eq!(selector_calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn dispatching_unapproved_work_preserves_other_pending_approval_budget() {
    let first_executions = Arc::new(AtomicUsize::new(0));
    let observed_first = Arc::clone(&first_executions);
    let first = ToolBinding::from_fn(
        ToolSpec::new(
            "read_record",
            Some("read one record".to_string()),
            json!({"type": "object"}),
        )
        .expect("valid first spec"),
        "v1",
        |_| Ok(()),
        move |_| {
            let observed_first = Arc::clone(&observed_first);
            async move {
                observed_first.fetch_add(1, Ordering::SeqCst);
                Ok(ToolOutcome::Success {
                    value: Value::Bool(true),
                })
            }
        },
    )
    .expect("valid first binding")
    .with_approval_policy(ApprovalPolicy::NotRequired);
    let second_executions = Arc::new(AtomicUsize::new(0));
    let observed_second = Arc::clone(&second_executions);
    let second = ToolBinding::from_fn(
        tool_spec(),
        "v1",
        |_| Ok(()),
        move |_| {
            let observed_second = Arc::clone(&observed_second);
            async move {
                observed_second.fetch_add(1, Ordering::SeqCst);
                Ok(ToolOutcome::Success {
                    value: Value::Bool(true),
                })
            }
        },
    )
    .expect("valid second binding");
    let tools = ToolSet::from_bindings([first, second]).expect("unique tools");
    let response = LanguageResponse::completed(
        vec![
            ContentPart::ToolCall(
                ToolCall::local("call-read", "read_record", json!({})).expect("valid tool call"),
            ),
            ContentPart::ToolCall(tool_call()),
        ],
        LanguageCompletionReason::ToolCalls,
        Usage::default(),
    )
    .expect("valid mixed tool response");
    let store = Arc::new(InMemoryRunStore::new());
    let model = ScriptedModel::new([response]);
    let loop_ = durable_loop(model, tools, store);

    let suspended = loop_
        .start(
            run_id("mixed-approval"),
            lineage_id("mixed-approval"),
            request(),
            CallOptions::default(),
        )
        .await
        .expect("unapproved second call remains resumable");

    assert!(matches!(
        suspended.snapshot().resume_point(),
        ResumePoint::AwaitingApprovals(_)
    ));
    assert_eq!(suspended.snapshot().pending_approvals().len(), 1);
    assert_eq!(suspended.snapshot().budget().pending_approvals(), 1);
    assert_eq!(first_executions.load(Ordering::SeqCst), 1);
    assert_eq!(second_executions.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn lease_conflict_is_a_typed_resume_outcome() {
    let store = Arc::new(InMemoryRunStore::new());
    let run = run_id("lease-conflict");
    let _lease = store
        .acquire(&run, Duration::from_secs(60))
        .await
        .expect("first lease");
    let tools = ToolSet::default();
    let model = ScriptedModel::new([final_response()]);
    let loop_ = durable_loop(model, tools, store);

    let error = loop_
        .resume(&run, DurableResume::default(), CallOptions::default())
        .await
        .expect_err("second lease must conflict");
    assert!(matches!(error, DurableRunError::ResumeConflict { .. }));
}

#[tokio::test]
async fn crash_matrix_preserves_dispatch_boundaries() {
    for (name, failed_cas, expected_before_resume, retry) in [
        ("before-dispatch", 2, 0, true),
        ("after-result", 3, 1, false),
        ("after-completed-checkpoint", 4, 1, true),
    ] {
        let executions = Arc::new(AtomicUsize::new(0));
        let binding = not_required_binding(Arc::clone(&executions), None);
        let tools = ToolSet::from_bindings([binding]).expect("unique tool");
        let store = FailOnceStore::new(failed_cas);
        let model = ScriptedModel::new([tool_response(), final_response()]);
        let loop_ = durable_loop(model, tools, store.clone());
        let run = run_id(name);

        let error = loop_
            .start(
                run.clone(),
                lineage_id(name),
                request(),
                CallOptions::default(),
            )
            .await
            .expect_err("injected checkpoint failure");
        assert!(matches!(
            error,
            DurableRunError::Store(RunStoreError::Unavailable)
        ));
        assert_eq!(store.release_calls(), 1);
        assert_eq!(executions.load(Ordering::SeqCst), expected_before_resume);

        let resumed = loop_
            .resume(&run, DurableResume::default(), CallOptions::default())
            .await
            .expect("resume after injected failure");
        assert_eq!(store.release_calls(), 2);
        if retry {
            assert!(resumed.is_terminal());
            assert_eq!(executions.load(Ordering::SeqCst), 1);
        } else {
            assert!(matches!(
                resumed.snapshot().resume_point(),
                ResumePoint::Terminal(_)
            ));
            assert_eq!(
                resumed.snapshot().execution_log().status("call-1"),
                Some(ToolExecutionStatus::Indeterminate)
            );
            assert_eq!(executions.load(Ordering::SeqCst), 1);
        }
    }
}

#[tokio::test]
async fn stable_recovery_reuses_key_and_increments_attempt() {
    let executions = Arc::new(AtomicUsize::new(0));
    let attempts = Arc::new(Mutex::new(Vec::new()));
    let binding = not_required_binding(Arc::clone(&executions), Some(Arc::clone(&attempts)))
        .with_recovery_policy(RecoveryPolicy::ReplayWithStableIdempotencyKey)
        .with_stable_idempotency_key(|_| ToolIdempotencyKey::new("stable-logical-call"));
    let tools = ToolSet::from_bindings([binding]).expect("unique tool");
    let store = FailOnceStore::new(3);
    let model = ScriptedModel::new([tool_response(), final_response()]);
    let loop_ = durable_loop(model, tools, store);
    let run = run_id("stable-retry");

    loop_
        .start(
            run.clone(),
            lineage_id("stable-retry"),
            request(),
            CallOptions::default(),
        )
        .await
        .expect_err("completed checkpoint is injected to fail");

    let resumed = loop_
        .resume(
            &run,
            DurableResume::new()
                .with_indeterminate_recovery(IndeterminateRecoveryPolicy::RetryStable),
            CallOptions::default(),
        )
        .await
        .expect("stable retry resumes");
    assert!(resumed.is_terminal());
    assert_eq!(executions.load(Ordering::SeqCst), 2);
    assert_eq!(
        *attempts.lock().expect("attempt lock"),
        vec![
            (1, Some("stable-logical-call".to_string())),
            (2, Some("stable-logical-call".to_string())),
        ]
    );
}

#[tokio::test]
async fn verified_approval_executes_only_the_exact_frozen_binding() {
    let executions = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&executions);
    let binding = ToolBinding::from_fn(
        tool_spec(),
        "v1",
        |_| Ok(()),
        move |_| {
            let observed = Arc::clone(&observed);
            async move {
                observed.fetch_add(1, Ordering::SeqCst);
                Ok(ToolOutcome::Success { value: json!(true) })
            }
        },
    )
    .expect("valid binding");
    let tools = ToolSet::from_bindings([binding]).expect("unique tool");
    let store = Arc::new(InMemoryRunStore::new());
    let verifier = Arc::new(StaticVerifier::default());
    let consume_store = Arc::new(InMemoryApprovalConsumeStore::default());
    let model = ScriptedModel::new([tool_response(), final_response()]);
    let loop_ = durable_loop(model.clone(), tools, store.clone())
        .with_approval_verification(verifier.clone(), consume_store);
    let run = run_id("approval");

    let suspended = loop_
        .start(
            run.clone(),
            lineage_id("approval"),
            request(),
            CallOptions::default(),
        )
        .await
        .expect("approval suspension");
    let pending = suspended
        .snapshot()
        .resume_point()
        .pending_step()
        .expect("pending step");
    let prepared = &pending.prepared()[0];
    let identity =
        TrustIdentity::new("issuer", "audience", "subject", "tenant").expect("valid identity");
    let context = TrustContext::builder(identity.clone())
        .model_target(pending.target().clone())
        .run_id(suspended.snapshot().run_id().clone())
        .lineage_id(suspended.snapshot().lineage_id().clone())
        .checkpoint_id(suspended.snapshot().checkpoint_id().clone())
        .execution_owner(prepared.call().owner().clone())
        .binding_identity(prepared.binding().clone())
        .tool_call_id(prepared.call().id())
        .canonical_arguments_digest(siumai_runtime::tool::canonical_arguments_digest(
            prepared.call().arguments(),
        ))
        .catalog_fingerprint(suspended.snapshot().fingerprints().tool_catalog.as_str())
        .policy_fingerprint(suspended.snapshot().fingerprints().approval_policy.as_str())
        .build()
        .expect("exact trust context");
    let claims =
        ApprovalClaims::issue(&context, u64::MAX, "nonce-1", "key-1").expect("valid claims");
    verifier.set_claims(claims);
    let envelope =
        ApprovalEnvelope::from_bytes(b"signed-approval".to_vec()).expect("valid envelope");

    let replacement_executions = Arc::new(AtomicUsize::new(0));
    let replacement = not_required_binding(Arc::clone(&replacement_executions), None);
    let replacement_tools = ToolSet::from_bindings([replacement]).expect("unique tool");
    let replacement_loop = durable_loop(model, replacement_tools, store.clone());
    let drift = replacement_loop
        .resume(&run, DurableResume::default(), CallOptions::default())
        .await
        .expect_err("catalog drift must fail before name-only execution");
    assert!(matches!(
        drift,
        DurableRunError::IncompatibleSnapshot {
            field: "fingerprints"
        }
    ));
    assert_eq!(replacement_executions.load(Ordering::SeqCst), 0);

    let completed = loop_
        .resume(
            &run,
            DurableResume::new().with_approval(DurableApproval::new("call-1", envelope, identity)),
            CallOptions::default(),
        )
        .await
        .expect("exact frozen binding executes");
    assert!(completed.is_terminal());
    assert_eq!(executions.load(Ordering::SeqCst), 1);
    assert_eq!(replacement_executions.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn provider_deferred_without_a_tool_call_is_a_durable_boundary() {
    let scope = ModelDescriptor::new(
        ProviderId::new("deferred-test").expect("valid provider"),
        ModelId::new("deferred-model").expect("valid model"),
        ModelFamily::Language,
    )
    .with_protocol(ProtocolId::new("native-orchestration").expect("valid protocol"))
    .with_replay_domain(ReplayDomain::custom(
        ReplayDomainId::new("durable-deferred-test").expect("valid replay domain"),
    ));
    let item = OpaqueProviderItem::new(
        ProviderProvenance::from_scope(scope.scope(), scope.model().clone())
            .expect("valid provenance"),
        "provider.deferred",
        json!({"opaque": true}),
    )
    .expect("valid opaque provider item");
    let store = Arc::new(InMemoryRunStore::new());
    let model: Arc<dyn LanguageModel> = DeferredModel::new(item);
    let loop_ = durable_loop(model, ToolSet::default(), store);

    let suspended = loop_
        .start(
            run_id("provider-deferred-only"),
            lineage_id("provider-deferred-only"),
            request(),
            CallOptions::default(),
        )
        .await
        .expect("provider-owned suspension is persisted");

    assert!(matches!(
        suspended.snapshot().resume_point(),
        ResumePoint::AwaitingProvider(_)
    ));
    assert_eq!(suspended.snapshot().report().steps().len(), 0);
    assert_eq!(suspended.snapshot().provider_state().len(), 1);
    assert!(!suspended.snapshot().provider_state()[0].payload.is_empty());
    assert_eq!(suspended.snapshot().report().provider_deferred().len(), 1);
}

#[tokio::test]
async fn multiple_approvals_are_verified_against_one_checkpoint_batch() {
    let first_executions = Arc::new(AtomicUsize::new(0));
    let second_executions = Arc::new(AtomicUsize::new(0));
    let tools = ToolSet::from_bindings([
        required_binding("write_record", Arc::clone(&first_executions), "v1"),
        required_binding("write_record_2", Arc::clone(&second_executions), "v1"),
    ])
    .expect("unique required bindings");
    let store = Arc::new(InMemoryRunStore::new());
    let verifier = Arc::new(MapVerifier::default());
    let consume_store = Arc::new(InMemoryApprovalConsumeStore::default());
    let model = ScriptedModel::new([two_tool_response(), final_response()]);
    let loop_ = durable_loop(model, tools, store)
        .with_approval_verification(verifier.clone(), consume_store);
    let run = run_id("approval-batch");

    let suspended = loop_
        .start(
            run.clone(),
            lineage_id("approval-batch"),
            request(),
            CallOptions::default(),
        )
        .await
        .expect("both calls wait at one checkpoint");
    let pending = suspended
        .snapshot()
        .resume_point()
        .pending_step()
        .expect("pending step");
    assert_eq!(pending.pending_approvals().len(), 2);

    let identity =
        TrustIdentity::new("issuer", "audience", "subject", "tenant").expect("valid identity");
    let mut approvals = Vec::new();
    for (index, prepared) in pending.prepared().iter().enumerate() {
        let context = TrustContext::builder(identity.clone())
            .model_target(pending.target().clone())
            .run_id(suspended.snapshot().run_id().clone())
            .lineage_id(suspended.snapshot().lineage_id().clone())
            .checkpoint_id(suspended.snapshot().checkpoint_id().clone())
            .execution_owner(prepared.call().owner().clone())
            .binding_identity(prepared.binding().clone())
            .tool_call_id(prepared.call().id())
            .canonical_arguments_digest(siumai_runtime::tool::canonical_arguments_digest(
                prepared.call().arguments(),
            ))
            .catalog_fingerprint(suspended.snapshot().fingerprints().tool_catalog.as_str())
            .policy_fingerprint(suspended.snapshot().fingerprints().approval_policy.as_str())
            .build()
            .expect("exact trust context");
        let claims = ApprovalClaims::issue(&context, u64::MAX, format!("nonce-{index}"), "key-1")
            .expect("valid claims");
        let envelope = ApprovalEnvelope::from_bytes(format!("signed-{index}").into_bytes())
            .expect("valid envelope");
        verifier.insert(&envelope, claims);
        approvals.push(DurableApproval::new(
            prepared.call().id().to_owned(),
            envelope,
            identity.clone(),
        ));
    }

    let completed = loop_
        .resume(
            &run,
            approvals
                .into_iter()
                .fold(DurableResume::new(), |resume, approval| {
                    resume.with_approval(approval)
                }),
            CallOptions::default(),
        )
        .await
        .expect("batch-approved calls execute");
    assert!(completed.is_terminal());
    assert_eq!(first_executions.load(Ordering::SeqCst), 1);
    assert_eq!(second_executions.load(Ordering::SeqCst), 1);
    assert_eq!(completed.snapshot().budget().pending_approvals(), 0);
}
