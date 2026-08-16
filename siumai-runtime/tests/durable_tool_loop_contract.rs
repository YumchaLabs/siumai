use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde::Serialize;
use serde_json::{Value, json};
use siumai_core::stream::established_stream;
use siumai_core::{
    CallOptions, ContentPart, Error, ErrorKind, GenerationConfig, LanguageCallError,
    LanguageCompletionReason, LanguageModel, LanguageRequest, LanguageResponse, LanguageStream,
    LanguageStreamEvent, Message, MessageRole, Model, ModelDescriptor, ModelFamily, ModelId,
    OpaqueProviderItem, PartialLanguageOutput, PartialLanguageOutputPart, ProtocolId, ProviderId,
    ProviderProvenance, ReplayDomain, ReplayDomainId, StreamTerminal, StructuredOutputSpec,
    ToolAnnotationTarget, ToolCall, ToolChoice, ToolOutcome, ToolResult, ToolSpec,
    TypedProviderAnnotation, Usage,
};
use siumai_runtime::approval::{
    ApprovalClaims, ApprovalEnvelope, ApprovalVerifier, ApprovalVerifierError,
    InMemoryApprovalConsumeStore, TrustContext, TrustIdentity,
};
use siumai_runtime::snapshot::{
    InMemoryRunStore, LineageId, ResumePointKind, RunId, RunLease, RunStore, RunStoreError,
    RunStoreFuture, SnapshotFingerprint, SnapshotRevision, StoredRun, ToolExecutionStatus,
};
use siumai_runtime::tool::{
    ApprovalPolicy, RecoveryPolicy, ToolBinding, ToolExecutionRequest, ToolIdempotencyKey, ToolSet,
};
use siumai_runtime::{
    Agent, DurableApproval, DurableResume, DurableRunError, DurableToolLoop,
    IndeterminateRecoveryPolicy, ModelTransitionOutcome, StepModelContext,
    StepModelSelectorIdentity, ToolLoop, ToolOutcomeAction, ToolOutcomePolicy,
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
    items: Vec<(String, OpaqueProviderItem)>,
}

struct CompletedDeferredModel {
    descriptor: ModelDescriptor,
    turns: Mutex<VecDeque<CompletedDeferredTurn>>,
}

struct CompletedDeferredTurn {
    state: OpaqueProviderItem,
    response: LanguageResponse,
    resolve: bool,
}

struct TerminalScriptedModel {
    descriptor: ModelDescriptor,
    terminals: Mutex<VecDeque<StreamTerminal>>,
}

impl TerminalScriptedModel {
    fn new(terminals: impl IntoIterator<Item = StreamTerminal>) -> Arc<Self> {
        Arc::new(Self {
            descriptor: ModelDescriptor::new(
                ProviderId::new("durable-terminal-test").expect("valid provider"),
                ModelId::new("durable-terminal-model").expect("valid model"),
                ModelFamily::Language,
            ),
            terminals: Mutex::new(terminals.into_iter().collect()),
        })
    }
}

impl Model for TerminalScriptedModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl LanguageModel for TerminalScriptedModel {
    async fn generate(
        &self,
        _request: LanguageRequest,
        _options: CallOptions,
    ) -> Result<LanguageResponse, LanguageCallError> {
        Err(Error::new(
            ErrorKind::Internal,
            "durable terminal test model must use streaming",
        )
        .into())
    }

    async fn stream(
        &self,
        _request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        let terminal = self
            .terminals
            .lock()
            .expect("terminal lock")
            .pop_front()
            .ok_or_else(|| Error::new(ErrorKind::Internal, "missing scripted terminal"))?;
        let cancellation = options.cancellation().clone();
        Ok(established_stream(cancellation, move |_| {
            futures::stream::iter([Ok(LanguageStreamEvent::Terminal(terminal))])
        }))
    }
}

impl DeferredModel {
    fn new(item: OpaqueProviderItem) -> Arc<Self> {
        Self::sequence([item])
    }

    fn sequence(items: impl IntoIterator<Item = OpaqueProviderItem>) -> Arc<Self> {
        Self::identified_sequence(
            items
                .into_iter()
                .map(|item| ("provider-state-1".to_string(), item)),
        )
    }

    fn identified_sequence(
        items: impl IntoIterator<Item = (String, OpaqueProviderItem)>,
    ) -> Arc<Self> {
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
            items: items.into_iter().collect(),
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
        let mut events = self
            .items
            .iter()
            .cloned()
            .map(|(id, state)| Ok(LanguageStreamEvent::ProviderDeferred { id, state }))
            .collect::<Vec<_>>();
        events.push(Ok(LanguageStreamEvent::Terminal(
            StreamTerminal::Completed {
                response: Box::new(final_response()),
            },
        )));
        let cancellation = options.cancellation().clone();
        Ok(established_stream(cancellation, move |_| {
            futures::stream::iter(events)
        }))
    }
}

impl CompletedDeferredModel {
    fn new(turns: impl IntoIterator<Item = (OpaqueProviderItem, LanguageResponse)>) -> Arc<Self> {
        Self::with_resolution(turns, true)
    }

    fn unresolved(
        turns: impl IntoIterator<Item = (OpaqueProviderItem, LanguageResponse)>,
    ) -> Arc<Self> {
        Self::with_resolution(turns, false)
    }

    fn with_resolution(
        turns: impl IntoIterator<Item = (OpaqueProviderItem, LanguageResponse)>,
        resolve: bool,
    ) -> Arc<Self> {
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
            turns: Mutex::new(
                turns
                    .into_iter()
                    .map(|(state, response)| CompletedDeferredTurn {
                        state,
                        response,
                        resolve,
                    })
                    .collect(),
            ),
        })
    }
}

impl Model for CompletedDeferredModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl LanguageModel for CompletedDeferredModel {
    async fn generate(
        &self,
        _request: LanguageRequest,
        _options: CallOptions,
    ) -> Result<LanguageResponse, LanguageCallError> {
        Err(Error::new(
            ErrorKind::Internal,
            "completed deferred test model must use streaming",
        )
        .into())
    }

    async fn stream(
        &self,
        _request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        let turn = self
            .turns
            .lock()
            .expect("deferred turn lock")
            .pop_front()
            .ok_or_else(|| Error::new(ErrorKind::Internal, "missing deferred test turn"))?;
        let cancellation = options.cancellation().clone();
        Ok(established_stream(cancellation, move |_| {
            let mut events = vec![Ok(LanguageStreamEvent::ProviderDeferred {
                id: "provider-state-1".to_string(),
                state: turn.state,
            })];
            if turn.resolve {
                events.push(Ok(LanguageStreamEvent::ToolResult(ToolResult {
                    call_id: "provider-state-1".to_string(),
                    name: "provider_task".to_string(),
                    outcome: ToolOutcome::Success { value: json!(true) },
                })));
            }
            events.push(Ok(LanguageStreamEvent::Terminal(
                StreamTerminal::Completed {
                    response: Box::new(turn.response),
                },
            )));
            futures::stream::iter(events)
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
        let snapshot = serde_json::from_value(
            serde_json::to_value(snapshot).expect("fake external store encodes snapshot"),
        )
        .expect("fake external store decodes snapshot");
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

#[derive(Serialize)]
#[serde(transparent)]
struct TestToolAnnotation(Value);

impl TypedProviderAnnotation for TestToolAnnotation {
    type Target = ToolAnnotationTarget;

    const NAMESPACE: &'static str = "test-provider";
    const API_MODE: Option<&'static str> = Some("messages");
}

fn caller_tool_spec(annotation: Value) -> ToolSpec {
    ToolSpec::new(
        "web_search",
        Some("provider-visible search".to_string()),
        json!({ "type": "object" }),
    )
    .expect("valid caller-visible tool")
    .with_provider_annotation(&TestToolAnnotation(annotation))
    .expect("valid caller-visible annotation")
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

fn usage(input: u64, output: u64) -> Usage {
    Usage::default()
        .with_input_tokens(input)
        .with_output_tokens(output)
        .with_total_tokens(input + output)
}

fn tool_response_with_usage(usage: Usage) -> LanguageResponse {
    LanguageResponse::completed(
        vec![ContentPart::ToolCall(tool_call())],
        LanguageCompletionReason::ToolCalls,
        usage,
    )
    .expect("valid tool response")
}

fn final_response_with_usage(usage: Usage) -> LanguageResponse {
    LanguageResponse::completed(
        vec![ContentPart::Text {
            text: "done".to_string(),
        }],
        LanguageCompletionReason::Stop,
        usage,
    )
    .expect("valid final response")
}

fn partial_with_usage(usage: Usage) -> PartialLanguageOutput {
    PartialLanguageOutput::new(
        vec![PartialLanguageOutputPart::Text {
            text: "partial".to_string(),
        }],
        usage,
    )
    .expect("valid partial output")
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

fn parity_model() -> Arc<ScriptedModel> {
    ScriptedModel::new([
        tool_response_with_usage(usage(1, 2)),
        final_response_with_usage(usage(2, 3)),
    ])
}

fn parity_tools(executions: Arc<AtomicUsize>) -> ToolSet {
    ToolSet::from_bindings([not_required_binding(executions, None)]).expect("unique tool")
}

fn assert_report_parity(expected: &siumai_runtime::RunReport, actual: &siumai_runtime::RunReport) {
    assert_eq!(actual.messages(), expected.messages());
    assert_eq!(actual.steps(), expected.steps());
    assert_eq!(actual.usage(), expected.usage());
    assert_eq!(actual.budget(), expected.budget());
    assert_eq!(actual.provider_deferred(), expected.provider_deferred());
    assert_eq!(
        actual.execution_log().status("call-1"),
        expected.execution_log().status("call-1")
    );
}

#[tokio::test]
async fn agent_tool_loop_and_durable_share_completed_step_semantics() {
    let agent_executions = Arc::new(AtomicUsize::new(0));
    let agent = Agent::from_tool_loop(ToolLoop::new(
        parity_model(),
        parity_tools(Arc::clone(&agent_executions)),
    ));
    let agent_terminal = agent.run(request()).await.expect("agent completes");

    let loop_executions = Arc::new(AtomicUsize::new(0));
    let tool_loop = ToolLoop::new(parity_model(), parity_tools(Arc::clone(&loop_executions)));
    let loop_terminal = tool_loop
        .run(request(), CallOptions::default())
        .await
        .expect("tool loop completes");

    let durable_executions = Arc::new(AtomicUsize::new(0));
    let durable = durable_loop(
        parity_model(),
        parity_tools(Arc::clone(&durable_executions)),
        Arc::new(InMemoryRunStore::new()),
    );
    let durable_run = durable
        .start(
            run_id("completed-step-parity"),
            lineage_id("completed-step-parity"),
            request(),
            CallOptions::default(),
        )
        .await
        .expect("durable loop completes");

    assert!(agent_terminal.is_completed());
    assert!(loop_terminal.is_completed());
    assert!(durable_run.is_terminal());
    let expected = agent_terminal.report().expect("agent report");
    assert_report_parity(expected, loop_terminal.report().expect("tool-loop report"));
    assert_report_parity(expected, durable_run.snapshot().report());
    assert_eq!(agent_executions.load(Ordering::SeqCst), 1);
    assert_eq!(loop_executions.load(Ordering::SeqCst), 1);
    assert_eq!(durable_executions.load(Ordering::SeqCst), 1);
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
    let caller_tool = caller_tool_spec(json!({
        "type": "web_search_20250305",
        "maxUses": 4
    }));
    request.tools.push(caller_tool.clone());

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
        assert_eq!(
            observed
                .tools
                .iter()
                .map(ToolSpec::name)
                .collect::<Vec<_>>(),
            vec!["web_search", "write_record"]
        );
        assert_eq!(observed.tools[0], caller_tool);
    }
    let continuation = completed.snapshot().continuation();
    assert_eq!(continuation.generation, request.generation);
    assert_eq!(continuation.tool_choice, request.tool_choice);
    assert_eq!(continuation.structured_output, request.structured_output);
    assert_eq!(
        continuation
            .tools
            .iter()
            .map(ToolSpec::name)
            .collect::<Vec<_>>(),
        vec!["web_search", "write_record"]
    );
    assert_eq!(continuation.tools[0], caller_tool);
}

#[tokio::test]
async fn caller_visible_annotations_are_canonical_parts_of_durable_catalog_identity() {
    let store = Arc::new(InMemoryRunStore::new());
    let model = ScriptedModel::new([final_response(), final_response(), final_response()]);
    let loop_ = durable_loop(model, ToolSet::default(), store);
    let annotations = [
        serde_json::from_str(
            r#"{"type":"web_search_20250305","config":{"region":"us","maxUses":4}}"#,
        )
        .expect("valid annotation"),
        serde_json::from_str(
            r#"{"config":{"maxUses":4,"region":"us"},"type":"web_search_20250305"}"#,
        )
        .expect("valid reordered annotation"),
        serde_json::from_str(
            r#"{"type":"web_search_20250305","config":{"region":"eu","maxUses":4}}"#,
        )
        .expect("valid changed annotation"),
    ];
    let mut catalog_fingerprints = Vec::new();

    for (index, annotation) in annotations.into_iter().enumerate() {
        let mut request = request();
        request.tools.push(caller_tool_spec(annotation));
        let completed = loop_
            .start(
                run_id(&format!("catalog-{index}")),
                lineage_id(&format!("catalog-{index}")),
                request,
                CallOptions::default(),
            )
            .await
            .expect("catalog run completes");
        catalog_fingerprints.push(completed.snapshot().fingerprints().tool_catalog().clone());
    }

    assert_eq!(catalog_fingerprints[0], catalog_fingerprints[1]);
    assert_ne!(catalog_fingerprints[0], catalog_fingerprints[2]);
}

#[tokio::test]
async fn old_catalog_identity_and_execution_abi_fail_explicitly() {
    let original_store = Arc::new(InMemoryRunStore::new());
    let original_model = ScriptedModel::new([final_response()]);
    let original_loop = durable_loop(original_model, ToolSet::default(), original_store);
    let run = run_id("legacy-identity");
    let completed = original_loop
        .start(
            run.clone(),
            lineage_id("legacy-identity"),
            request(),
            CallOptions::default(),
        )
        .await
        .expect("baseline snapshot completes");
    let baseline = serde_json::to_value(completed.snapshot()).expect("snapshot serializes");

    let mut old_abi_value = baseline.clone();
    old_abi_value["checkpoint"]["engine_version"] = json!("siumai-runtime-durable-v5");
    let old_abi = serde_json::from_value(old_abi_value).expect("old ABI snapshot remains valid v7");
    let old_abi_store = Arc::new(InMemoryRunStore::new());
    let lease = old_abi_store
        .acquire(&run, Duration::from_secs(30))
        .await
        .expect("lease acquired");
    old_abi_store
        .compare_and_swap(&lease, SnapshotRevision::EMPTY, old_abi)
        .await
        .expect("old ABI snapshot seeded");
    old_abi_store.release(lease).await.expect("lease released");
    let old_abi_model = ScriptedModel::new(Vec::<LanguageResponse>::new());
    let old_abi_loop = durable_loop(old_abi_model, ToolSet::default(), old_abi_store);
    assert!(matches!(
        old_abi_loop
            .resume(&run, DurableResume::default(), CallOptions::default())
            .await,
        Err(DurableRunError::IncompatibleSnapshot {
            field: "engine_version"
        })
    ));

    let mut old_catalog_value = baseline;
    old_catalog_value["fingerprints"]["tool_catalog"] = json!("sha256:legacy-catalog");
    let old_catalog = serde_json::from_value(old_catalog_value)
        .expect("old catalog snapshot remains structurally valid");
    let old_catalog_store = Arc::new(InMemoryRunStore::new());
    let lease = old_catalog_store
        .acquire(&run, Duration::from_secs(30))
        .await
        .expect("lease acquired");
    old_catalog_store
        .compare_and_swap(&lease, SnapshotRevision::EMPTY, old_catalog)
        .await
        .expect("old catalog snapshot seeded");
    old_catalog_store
        .release(lease)
        .await
        .expect("lease released");
    let old_catalog_model = ScriptedModel::new(Vec::<LanguageResponse>::new());
    let old_catalog_loop = durable_loop(old_catalog_model, ToolSet::default(), old_catalog_store);
    assert!(matches!(
        old_catalog_loop
            .resume(&run, DurableResume::default(), CallOptions::default())
            .await,
        Err(DurableRunError::IncompatibleSnapshot {
            field: "fingerprints"
        })
    ));
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
        .model_selector()
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

    assert_eq!(
        suspended.snapshot().resume_point().kind(),
        ResumePointKind::AwaitingApprovals
    );
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
            assert_eq!(
                resumed.snapshot().resume_point().kind(),
                ResumePointKind::Terminal
            );
            assert_eq!(
                resumed.snapshot().execution_log().status("call-1"),
                Some(ToolExecutionStatus::Indeterminate)
            );
            assert_eq!(executions.load(Ordering::SeqCst), 1);
        }
    }
}

#[tokio::test]
async fn durable_mixed_unknown_and_bound_call_preserves_unbound_result_across_resume() {
    let response = LanguageResponse::completed(
        vec![
            ContentPart::ToolCall(
                ToolCall::local("call-unknown", "unknown_tool", json!({}))
                    .expect("valid unknown tool call"),
            ),
            ContentPart::ToolCall(tool_call()),
        ],
        LanguageCompletionReason::ToolCalls,
        Usage::default(),
    )
    .expect("valid mixed tool response");
    let outcome_policy =
        ToolOutcomePolicy::default().with_execution_failed(ToolOutcomeAction::Continue);

    let expected_executions = Arc::new(AtomicUsize::new(0));
    let expected = ToolLoop::new(
        ScriptedModel::new([response.clone(), final_response()]),
        ToolSet::from_bindings([not_required_binding(Arc::clone(&expected_executions), None)])
            .expect("unique tool"),
    )
    .with_outcome_policy(outcome_policy)
    .run(request(), CallOptions::default())
    .await
    .expect("non-durable mixed run completes");
    let expected_report = match expected {
        siumai_runtime::RunTerminal::Completed { report } => report,
        other => panic!("expected completed tool loop, got {other:?}"),
    };

    let executions = Arc::new(AtomicUsize::new(0));
    let store = FailOnceStore::new(4);
    let durable = durable_loop(
        ScriptedModel::new([response, final_response()]),
        ToolSet::from_bindings([not_required_binding(Arc::clone(&executions), None)])
            .expect("unique tool"),
        store.clone(),
    )
    .with_outcome_policy(outcome_policy);
    let run = run_id("mixed-unknown-bound");

    let error = durable
        .start(
            run.clone(),
            lineage_id("mixed-unknown-bound"),
            request(),
            CallOptions::default(),
        )
        .await
        .expect_err("injected post-completion checkpoint failure");
    assert!(matches!(
        error,
        DurableRunError::Store(RunStoreError::Unavailable)
    ));
    assert_eq!(executions.load(Ordering::SeqCst), 1);

    let resumed = durable
        .resume(&run, DurableResume::default(), CallOptions::default())
        .await
        .expect("resume preserves completed unbound result");
    assert!(resumed.is_terminal());
    assert_eq!(executions.load(Ordering::SeqCst), 1);
    assert_report_parity(&expected_report, resumed.snapshot().report());

    let unbound = resumed.snapshot().report().steps()[0]
        .tool_results()
        .iter()
        .find(|result| result.call_id == "call-unknown")
        .expect("unknown local tool result is retained");
    assert!(matches!(
        &unbound.outcome,
        ToolOutcome::ExecutionFailed { .. }
    ));
    assert_eq!(
        resumed.snapshot().execution_log().status("call-unknown"),
        None
    );
}

#[tokio::test]
async fn serialized_resume_settles_usage_exactly_once_for_all_model_terminals() {
    for (name, terminal) in [
        (
            "completed",
            StreamTerminal::Completed {
                response: Box::new(final_response_with_usage(usage(3, 4))),
            },
        ),
        (
            "failed-partial",
            StreamTerminal::Failed {
                error: Error::protocol_violation("scripted failure"),
                partial: Some(partial_with_usage(usage(3, 4))),
            },
        ),
        (
            "cancelled-partial",
            StreamTerminal::Cancelled {
                reason: "scripted cancellation".to_string(),
                partial: Some(partial_with_usage(usage(3, 4))),
            },
        ),
    ] {
        let executions = Arc::new(AtomicUsize::new(0));
        let binding = not_required_binding(Arc::clone(&executions), None);
        let tools = ToolSet::from_bindings([binding]).expect("unique tool");
        let store = FailOnceStore::new(4);
        let model = TerminalScriptedModel::new([
            StreamTerminal::Completed {
                response: Box::new(tool_response_with_usage(usage(2, 3))),
            },
            terminal,
        ]);
        let loop_ = durable_loop(model, tools, store);
        let run = run_id(&format!("usage-{name}"));

        loop_
            .start(
                run.clone(),
                lineage_id(&format!("usage-{name}")),
                request(),
                CallOptions::default(),
            )
            .await
            .expect_err("injected checkpoint failure separates model calls");

        let resumed = loop_
            .resume(&run, DurableResume::default(), CallOptions::default())
            .await
            .expect("serialized snapshot resumes");

        assert_eq!(
            resumed.snapshot().usage().input_tokens.value(),
            Some(5),
            "{name} input usage"
        );
        assert_eq!(
            resumed.snapshot().usage().output_tokens.value(),
            Some(7),
            "{name} output usage"
        );
        assert_eq!(
            resumed.snapshot().usage().total_tokens.value(),
            Some(12),
            "{name} total usage"
        );
        assert_eq!(resumed.snapshot().budget().known_tokens(), 12, "{name}");
        assert_eq!(executions.load(Ordering::SeqCst), 1, "{name}");
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
    let caller_tool = caller_tool_spec(json!({ "type": "web_search_20250305" }));
    let mut initial_request = request();
    initial_request.tools.push(caller_tool.clone());

    let suspended = loop_
        .start(
            run.clone(),
            lineage_id("approval"),
            initial_request,
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
        .catalog_fingerprint(suspended.snapshot().fingerprints().tool_catalog().as_str())
        .policy_fingerprint(
            suspended
                .snapshot()
                .fingerprints()
                .approval_policy()
                .as_str(),
        )
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
    let replacement_loop = durable_loop(model.clone(), replacement_tools, store.clone());
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
    let requests = model.requests();
    assert_eq!(requests.len(), 2);
    for request in requests {
        assert_eq!(
            request.tools.iter().map(ToolSpec::name).collect::<Vec<_>>(),
            vec!["web_search", "write_record"]
        );
        assert_eq!(request.tools[0], caller_tool);
    }
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

    assert_eq!(
        suspended.snapshot().resume_point().kind(),
        ResumePointKind::AwaitingProvider
    );
    assert_eq!(suspended.snapshot().report().steps().len(), 0);
    assert_eq!(suspended.snapshot().provider_state().len(), 1);
    assert!(
        !suspended.snapshot().provider_state()[0]
            .payload()
            .is_empty()
    );
    assert_eq!(suspended.snapshot().report().provider_deferred().len(), 1);
}

#[tokio::test]
async fn unresolved_provider_deferred_waits_for_local_progress_before_suspending() {
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
        json!({"status": "queued"}),
    )
    .expect("valid opaque provider item");
    let executions = Arc::new(AtomicUsize::new(0));
    let tools = ToolSet::from_bindings([not_required_binding(Arc::clone(&executions), None)])
        .expect("unique tool");
    let store = Arc::new(InMemoryRunStore::new());
    let model: Arc<dyn LanguageModel> =
        CompletedDeferredModel::unresolved([(item, tool_response())]);
    let loop_ = durable_loop(model, tools, store);

    let suspended = loop_
        .start(
            run_id("provider-deferred-with-local"),
            lineage_id("provider-deferred-with-local"),
            request(),
            CallOptions::default(),
        )
        .await
        .expect("local progress completes before provider suspension");

    assert_eq!(executions.load(Ordering::SeqCst), 1);
    assert_eq!(
        suspended.snapshot().resume_point().kind(),
        ResumePointKind::AwaitingProvider
    );
    assert_eq!(suspended.snapshot().provider_state().len(), 1);
    let report = suspended.snapshot().report();
    assert!(report.steps().is_empty());
    assert_eq!(
        report.execution_log().status("call-1"),
        Some(ToolExecutionStatus::Completed)
    );
    assert_eq!(report.provider_deferred().len(), 1);
    assert!(!report.provider_deferred()[0].is_resolved());
    let tool_message = suspended
        .snapshot()
        .continuation()
        .messages
        .last()
        .expect("completed local result is durable continuation history");
    let [part] = tool_message.content() else {
        panic!("expected one completed local result in continuation history");
    };
    assert!(matches!(
        part.content(),
        ContentPart::ToolResult(result) if result.call_id == "call-1"
    ));
}

#[tokio::test]
async fn provider_deferred_same_namespace_persists_only_the_latest_observation() {
    let scope = ModelDescriptor::new(
        ProviderId::new("deferred-test").expect("valid provider"),
        ModelId::new("deferred-model").expect("valid model"),
        ModelFamily::Language,
    )
    .with_protocol(ProtocolId::new("native-orchestration").expect("valid protocol"))
    .with_replay_domain(ReplayDomain::custom(
        ReplayDomainId::new("durable-deferred-test").expect("valid replay domain"),
    ));
    let provenance =
        ProviderProvenance::from_scope(scope.scope(), scope.model().clone()).expect("provenance");
    let queued = OpaqueProviderItem::new(
        provenance.clone(),
        "provider.deferred",
        json!({"status": "queued"}),
    )
    .expect("queued state");
    let in_progress = OpaqueProviderItem::new(
        provenance,
        "provider.deferred",
        json!({"status": "in_progress"}),
    )
    .expect("in-progress state");
    let store = Arc::new(InMemoryRunStore::new());
    let model: Arc<dyn LanguageModel> = DeferredModel::sequence([queued, in_progress]);
    let loop_ = durable_loop(model, ToolSet::default(), store);

    let suspended = loop_
        .start(
            run_id("provider-deferred-latest"),
            lineage_id("provider-deferred-latest"),
            request(),
            CallOptions::default(),
        )
        .await
        .expect("latest provider-owned state is persisted");

    let states = suspended.snapshot().provider_state();
    assert_eq!(states.len(), 1);
    let retained: OpaqueProviderItem =
        serde_json::from_slice(states[0].payload()).expect("provider state decodes");
    assert_eq!(retained.data()["status"], "in_progress");

    let report_items = suspended.snapshot().report().provider_deferred();
    assert_eq!(report_items.len(), 1);
    assert_eq!(report_items[0].item().data()["status"], "in_progress");
}

#[tokio::test]
async fn provider_deferred_distinct_keys_keep_first_seen_order() {
    let scope = ModelDescriptor::new(
        ProviderId::new("deferred-test").expect("valid provider"),
        ModelId::new("deferred-model").expect("valid model"),
        ModelFamily::Language,
    )
    .with_protocol(ProtocolId::new("native-orchestration").expect("valid protocol"))
    .with_replay_domain(ReplayDomain::custom(
        ReplayDomainId::new("durable-deferred-test").expect("valid replay domain"),
    ));
    let provenance =
        ProviderProvenance::from_scope(scope.scope(), scope.model().clone()).expect("provenance");
    let first_queued = OpaqueProviderItem::new(
        provenance.clone(),
        "provider.deferred",
        json!({"status": "first_queued"}),
    )
    .expect("first queued state");
    let second_queued = OpaqueProviderItem::new(
        provenance.clone(),
        "provider.deferred",
        json!({"status": "second_queued"}),
    )
    .expect("second queued state");
    let first_in_progress = OpaqueProviderItem::new(
        provenance,
        "provider.deferred",
        json!({"status": "first_in_progress"}),
    )
    .expect("first in-progress state");
    let store = Arc::new(InMemoryRunStore::new());
    let model: Arc<dyn LanguageModel> = DeferredModel::identified_sequence([
        ("provider-state-1".to_string(), first_queued),
        ("provider-state-2".to_string(), second_queued),
        ("provider-state-1".to_string(), first_in_progress),
    ]);
    let loop_ = durable_loop(model, ToolSet::default(), store);

    let suspended = loop_
        .start(
            run_id("provider-deferred-distinct"),
            lineage_id("provider-deferred-distinct"),
            request(),
            CallOptions::default(),
        )
        .await
        .expect("distinct provider-owned states are persisted");

    let states = suspended.snapshot().provider_state();
    assert_eq!(states.len(), 2);
    assert_eq!(states[0].namespace(), states[1].namespace());
    assert_eq!(states[0].correlation_id(), "provider-state-1");
    assert_eq!(states[1].correlation_id(), "provider-state-2");
    let retained = states
        .iter()
        .map(|state| {
            serde_json::from_slice::<OpaqueProviderItem>(state.payload())
                .expect("provider state decodes")
        })
        .collect::<Vec<_>>();
    assert_eq!(retained[0].data()["status"], "first_in_progress");
    assert_eq!(retained[1].data()["status"], "second_queued");

    let report_items = suspended.snapshot().report().provider_deferred();
    assert_eq!(report_items.len(), 2);
    assert_eq!(report_items[0].item().data()["status"], "first_in_progress");
    assert_eq!(report_items[1].item().data()["status"], "second_queued");
}

#[tokio::test]
async fn resolved_provider_deferred_key_cannot_reopen_across_model_steps() {
    let scope = ModelDescriptor::new(
        ProviderId::new("deferred-test").expect("valid provider"),
        ModelId::new("deferred-model").expect("valid model"),
        ModelFamily::Language,
    )
    .with_protocol(ProtocolId::new("native-orchestration").expect("valid protocol"))
    .with_replay_domain(ReplayDomain::custom(
        ReplayDomainId::new("durable-deferred-test").expect("valid replay domain"),
    ));
    let provenance =
        ProviderProvenance::from_scope(scope.scope(), scope.model().clone()).expect("provenance");
    let queued = OpaqueProviderItem::new(
        provenance.clone(),
        "provider.deferred",
        json!({"status": "queued"}),
    )
    .expect("queued state");
    let in_progress = OpaqueProviderItem::new(
        provenance,
        "provider.deferred",
        json!({"status": "in_progress"}),
    )
    .expect("in-progress state");
    let executions = Arc::new(AtomicUsize::new(0));
    let tools = ToolSet::from_bindings([not_required_binding(Arc::clone(&executions), None)])
        .expect("unique tool");
    let store = Arc::new(InMemoryRunStore::new());
    let model: Arc<dyn LanguageModel> =
        CompletedDeferredModel::new([(queued, tool_response()), (in_progress, final_response())]);
    let loop_ = durable_loop(model, tools, store);

    let completed = loop_
        .start(
            run_id("provider-deferred-cross-step"),
            lineage_id("provider-deferred-cross-step"),
            request(),
            CallOptions::default(),
        )
        .await
        .expect("resolved provider state fails closed when reopened");

    assert!(completed.is_terminal());
    assert_eq!(executions.load(Ordering::SeqCst), 1);
    assert_eq!(completed.snapshot().report().steps().len(), 1);
    assert_eq!(
        completed
            .snapshot()
            .resume_point()
            .terminal()
            .unwrap()
            .kind(),
        siumai_runtime::snapshot::SnapshotTerminalKind::Failed
    );
    let observations = completed.snapshot().report().provider_deferred();
    assert_eq!(observations.len(), 1);
    assert!(observations[0].is_resolved());
    assert_eq!(observations[0].item().data()["status"], "queued");
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
            .catalog_fingerprint(suspended.snapshot().fingerprints().tool_catalog().as_str())
            .policy_fingerprint(
                suspended
                    .snapshot()
                    .fingerprints()
                    .approval_policy()
                    .as_str(),
            )
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
