use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use serde::Serialize;
use serde_json::json;
use siumai_core::stream::established_stream;
use siumai_core::{
    CallOptions, ContentAnnotationTarget, ContentPart, Error, ErrorKind, GenerationConfig,
    LanguageCallError, LanguageCompletionReason, LanguageInput, LanguageModel, LanguageRequest,
    LanguageRequestError, LanguageResponse, LanguageStream, LanguageStreamEvent, Message,
    MessagePart, MessageRole, Model, ModelDescriptor, ModelFamily, ModelId, OpaqueProviderItem,
    ProtocolId, ProviderId, ProviderProvenance, ReplayDomain, ReplayDomainId, StreamTerminal,
    StructuredOutputSpec, ToolCall, ToolChoice, ToolOutcome, ToolSpec, TypedProviderAnnotation,
    Usage,
};
use siumai_runtime::snapshot::SnapshotFingerprint;
use siumai_runtime::tool::{ApprovalPolicy, ToolBinding, ToolSet};
use siumai_runtime::{
    Agent, AgentConfigError, ModelTransitionOutcome, ProjectionLossReason, ProjectionPolicy,
    ProjectionScope, RunEvent, RunTerminal, StepModelContext, StepModelSelectorIdentity, ToolLoop,
    VersionedStepModelSelector,
};

struct ScriptedModel {
    descriptor: ModelDescriptor,
    responses: Mutex<VecDeque<LanguageResponse>>,
    requests: Mutex<Vec<LanguageRequest>>,
    generate_calls: AtomicUsize,
    stream_calls: AtomicUsize,
}

impl ScriptedModel {
    fn new(
        provider: &str,
        protocol: &str,
        model: &str,
        responses: impl IntoIterator<Item = LanguageResponse>,
    ) -> Arc<Self> {
        Arc::new(Self {
            descriptor: ModelDescriptor::new(
                ProviderId::new(provider).expect("valid provider"),
                ModelId::new(model).expect("valid model"),
                ModelFamily::Language,
            )
            .with_protocol(ProtocolId::new(protocol).expect("valid protocol"))
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("model-switching-test").expect("valid replay domain"),
            )),
            responses: Mutex::new(responses.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
            generate_calls: AtomicUsize::new(0),
            stream_calls: AtomicUsize::new(0),
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
        self.generate_calls.fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorKind::Internal,
            "step engine must use streaming model calls",
        )
        .into())
    }

    async fn stream(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        self.stream_calls.fetch_add(1, Ordering::SeqCst);
        self.requests.lock().expect("request lock").push(request);
        let response = self
            .responses
            .lock()
            .expect("response lock")
            .pop_front()
            .ok_or_else(|| Error::new(ErrorKind::Internal, "missing scripted response"))?;
        Ok(established_stream(
            options.cancellation().clone(),
            move |_| {
                futures::stream::iter([Ok(LanguageStreamEvent::Terminal(
                    StreamTerminal::Completed {
                        response: Box::new(response),
                    },
                ))])
            },
        ))
    }
}

fn request() -> LanguageRequest {
    LanguageRequest::new(vec![Message::text(MessageRole::User, "research rust")])
}

struct LocalAgentInput(LanguageRequest);

impl From<LocalAgentInput> for LanguageInput {
    fn from(input: LocalAgentInput) -> Self {
        input.0.into()
    }
}

#[derive(Serialize)]
struct AgentContentAnnotation {
    cache: bool,
}

impl TypedProviderAnnotation for AgentContentAnnotation {
    type Target = ContentAnnotationTarget;

    const NAMESPACE: &'static str = "agent-test";
}

fn request_with_caller_tool() -> LanguageRequest {
    let mut request = request();
    request.tools.push(
        ToolSpec::new(
            "client_search",
            Some("provider-visible search".to_string()),
            json!({"type": "object"}),
        )
        .expect("valid caller-visible tool"),
    );
    request
}

fn local_call() -> ToolCall {
    ToolCall::local("call-1", "lookup", json!({"query": "rust"})).expect("valid tool call")
}

fn opaque(provider: &str, protocol: &str, model: &str, kind: &str) -> OpaqueProviderItem {
    let scope = ModelDescriptor::new(
        ProviderId::new(provider).expect("valid provider"),
        ModelId::new(model).expect("valid model"),
        ModelFamily::Language,
    )
    .with_protocol(ProtocolId::new(protocol).expect("valid protocol"))
    .with_replay_domain(ReplayDomain::custom(
        ReplayDomainId::new("model-switching-test").expect("valid replay domain"),
    ));
    OpaqueProviderItem::new(
        ProviderProvenance::from_scope(scope.scope(), scope.model().clone())
            .expect("valid provenance"),
        kind,
        json!({"id": "native-state"}),
    )
    .expect("valid opaque item")
}

fn rich_agent_request() -> LanguageRequest {
    let annotated_user = Message::new(
        MessageRole::User,
        [MessagePart::text("research rust")
            .with_provider_annotation(&AgentContentAnnotation { cache: true })
            .expect("valid content annotation")],
    );
    let replay = Message::new(
        MessageRole::Assistant,
        [ContentPart::ProviderOpaque(opaque(
            "agent",
            "agent.messages",
            "model",
            "response.output",
        ))],
    );
    LanguageRequest {
        messages: vec![annotated_user, replay],
        generation: GenerationConfig {
            max_output_tokens: Some(321),
            temperature: Some(0.25),
            top_p: Some(0.8),
            stop_sequences: vec!["STOP".to_string()],
            seed: Some(42),
        },
        tools: vec![
            ToolSpec::new(
                "client_search",
                Some("provider-visible search".to_string()),
                json!({"type": "object"}),
            )
            .expect("valid caller-visible tool"),
        ],
        tool_choice: Some(ToolChoice::Named {
            name: "client_search".to_string(),
        }),
        structured_output: Some(StructuredOutputSpec {
            name: "agent_result".to_string(),
            description: Some("Structured agent result".to_string()),
            schema: json!({"type": "object"}),
            strict: true,
        }),
    }
}

fn tool_response(native: Option<OpaqueProviderItem>) -> LanguageResponse {
    let mut content = Vec::new();
    if let Some(native) = native {
        content.push(ContentPart::ProviderOpaque(native));
    }
    content.push(ContentPart::ToolCall(local_call()));
    LanguageResponse::completed(
        content,
        LanguageCompletionReason::ToolCalls,
        Usage::default(),
    )
    .expect("valid tool response")
}

fn final_response(text: &str) -> LanguageResponse {
    LanguageResponse::completed(
        vec![ContentPart::Text {
            text: text.to_string(),
        }],
        LanguageCompletionReason::Stop,
        Usage::default(),
    )
    .expect("valid final response")
}

fn tools() -> ToolSet {
    let binding = ToolBinding::from_fn(
        ToolSpec::new(
            "lookup",
            Some("lookup one value".to_string()),
            json!({"type": "object"}),
        )
        .expect("valid tool spec"),
        "v1",
        |_| Ok(()),
        |_| async {
            Ok(ToolOutcome::Success {
                value: json!({"value": "rust"}),
            })
        },
    )
    .expect("valid binding")
    .with_approval_policy(ApprovalPolicy::NotRequired);
    ToolSet::from_bindings([binding]).expect("unique tool")
}

fn switching_loop(
    source: Arc<ScriptedModel>,
    target: Arc<ScriptedModel>,
    policy: ProjectionPolicy,
) -> ToolLoop {
    let source: Arc<dyn LanguageModel> = source;
    let target: Arc<dyn LanguageModel> = target;
    ToolLoop::new(source, tools())
        .with_projection_policy(policy)
        .with_model_selector(VersionedStepModelSelector::new(
            StepModelSelectorIdentity::new(
                1,
                SnapshotFingerprint::new("sha256:model-switching-contract")
                    .expect("valid selector fingerprint"),
            ),
            move |_: StepModelContext<'_>| Ok(Arc::clone(&target)),
        ))
}

#[tokio::test]
async fn strict_portable_switch_uses_one_engine_and_records_the_transition() {
    let source = ScriptedModel::new("source", "source.messages", "fast", [tool_response(None)]);
    let target = ScriptedModel::new(
        "target",
        "target.messages",
        "strong",
        [final_response("done")],
    );
    let mut stream = switching_loop(source.clone(), target.clone(), ProjectionPolicy::Strict)
        .stream(request_with_caller_tool(), CallOptions::default())
        .await
        .expect("run establishes");
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }

    let transition_index = events
        .iter()
        .position(|event| matches!(event, RunEvent::ModelTransition { .. }))
        .expect("transition event");
    let second_step_index = events
        .iter()
        .position(|event| matches!(event, RunEvent::StepStarted { index: 1, .. }))
        .expect("second step");
    assert!(transition_index < second_step_index);
    let RunEvent::Terminal(RunTerminal::Completed { report }) = events.last().expect("terminal")
    else {
        panic!("run should complete");
    };
    assert_eq!(report.steps().len(), 2);
    assert_eq!(report.model_transitions().len(), 1);
    let transition = &report.model_transitions()[0];
    assert_eq!(transition.outcome(), ModelTransitionOutcome::Applied);
    assert_eq!(transition.scope(), ProjectionScope::PortableOnly);
    assert!(transition.losses().is_empty());
    assert_eq!(source.stream_calls.load(Ordering::SeqCst), 1);
    assert_eq!(target.stream_calls.load(Ordering::SeqCst), 1);
    assert_eq!(source.generate_calls.load(Ordering::SeqCst), 0);
    assert_eq!(target.generate_calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        source.requests()[0]
            .tools
            .iter()
            .map(ToolSpec::name)
            .collect::<Vec<_>>(),
        vec!["client_search", "lookup"]
    );
    assert_eq!(
        target.requests()[0]
            .tools
            .iter()
            .map(ToolSpec::name)
            .collect::<Vec<_>>(),
        vec!["client_search", "lookup"]
    );
}

#[tokio::test]
async fn strict_required_loss_rejects_before_charging_or_calling_the_target() {
    let source = ScriptedModel::new(
        "source",
        "source.responses",
        "fast",
        [tool_response(Some(opaque(
            "source",
            "source.responses",
            "fast",
            "response.output",
        )))],
    );
    let target = ScriptedModel::new(
        "target",
        "target.messages",
        "strong",
        [final_response("must not run")],
    );
    let terminal = switching_loop(source, target.clone(), ProjectionPolicy::Strict)
        .run(request(), CallOptions::default())
        .await
        .expect("run establishes");

    let RunTerminal::HistoryProjectionRejected { transition, report } = terminal else {
        panic!("strict projection must reject");
    };
    assert_eq!(transition.outcome(), ModelTransitionOutcome::Rejected);
    assert!(
        transition
            .losses()
            .iter()
            .any(|loss| { loss.reason == ProjectionLossReason::ForeignProviderOpaque })
    );
    assert_eq!(report.budget().model_steps(), 1);
    assert_eq!(target.stream_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn best_effort_switch_drops_native_state_with_event_report_and_request_parity() {
    let source = ScriptedModel::new(
        "source",
        "source.responses",
        "fast",
        [tool_response(Some(opaque(
            "source",
            "source.responses",
            "fast",
            "response.output",
        )))],
    );
    let target = ScriptedModel::new(
        "target",
        "target.messages",
        "strong",
        [final_response("done")],
    );
    let terminal = switching_loop(source, target.clone(), ProjectionPolicy::BestEffort)
        .run(request(), CallOptions::default())
        .await
        .expect("run establishes");

    let RunTerminal::Completed { report } = terminal else {
        panic!("best effort should complete");
    };
    let transition = &report.model_transitions()[0];
    assert!(transition.is_applied());
    assert!(
        transition
            .losses()
            .iter()
            .any(|loss| { loss.reason == ProjectionLossReason::ForeignProviderOpaque })
    );
    let target_request = target.requests().pop().expect("target request");
    assert!(!target_request.messages.iter().any(|message| {
        message
            .content()
            .iter()
            .any(|part| matches!(part.content(), ContentPart::ProviderOpaque(_)))
    }));
    assert!(report.messages().starts_with(&target_request.messages));
}

#[tokio::test]
async fn best_effort_still_rejects_blocking_provider_state() {
    let source = ScriptedModel::new(
        "source",
        "source.responses",
        "fast",
        [tool_response(Some(opaque(
            "source",
            "source.responses",
            "fast",
            "mcp_approval_request",
        )))],
    );
    let target = ScriptedModel::new(
        "target",
        "target.messages",
        "strong",
        [final_response("must not run")],
    );
    let terminal = switching_loop(source, target.clone(), ProjectionPolicy::BestEffort)
        .run(request(), CallOptions::default())
        .await
        .expect("run establishes");

    let RunTerminal::HistoryProjectionRejected { transition, .. } = terminal else {
        panic!("blocking state must reject");
    };
    assert!(
        transition
            .losses()
            .iter()
            .any(|loss| { loss.reason == ProjectionLossReason::PendingApproval })
    );
    assert_eq!(target.stream_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn same_protocol_model_switch_preserves_native_state() {
    let source = ScriptedModel::new(
        "source",
        "source.responses",
        "fast",
        [tool_response(Some(opaque(
            "source",
            "source.responses",
            "fast",
            "response.output",
        )))],
    );
    let target = ScriptedModel::new(
        "source",
        "source.responses",
        "strong",
        [final_response("done")],
    );
    let terminal = switching_loop(source, target.clone(), ProjectionPolicy::Strict)
        .run(request(), CallOptions::default())
        .await
        .expect("run establishes");

    let RunTerminal::Completed { report } = terminal else {
        panic!("same replay domain should complete");
    };
    assert_eq!(
        report.model_transitions()[0].scope(),
        ProjectionScope::ReplayDomain
    );
    assert!(report.model_transitions()[0].losses().is_empty());
    assert!(target.requests()[0].messages.iter().any(|message| {
        message
            .content()
            .iter()
            .any(|part| matches!(part.content(), ContentPart::ProviderOpaque(_)))
    }));
}

#[tokio::test]
async fn agent_reuses_instructions_without_sharing_run_history() {
    let model = ScriptedModel::new(
        "agent",
        "agent.messages",
        "model",
        [final_response("one"), final_response("two")],
    );
    let shared: Arc<dyn LanguageModel> = model.clone();
    let agent = Agent::from_shared_model(shared).with_instructions("be concise");

    assert!(matches!(
        agent.run("first").await.unwrap(),
        RunTerminal::Completed { .. }
    ));
    assert!(matches!(
        agent.run("second").await.unwrap(),
        RunTerminal::Completed { .. }
    ));

    let requests = model.requests();
    assert_eq!(requests.len(), 2);
    for request in requests {
        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.messages[0].role(), MessageRole::System);
        assert_eq!(request.messages[1].role(), MessageRole::User);
    }
}

#[tokio::test]
async fn agent_accepts_downstream_wrappers_converted_into_language_input() {
    let model = ScriptedModel::new("agent", "agent.messages", "model", [final_response("done")]);
    let shared: Arc<dyn LanguageModel> = model.clone();
    let request = rich_agent_request();
    let mut expected = request.clone();
    expected
        .messages
        .insert(0, Message::text(MessageRole::System, "be concise"));

    let terminal = Agent::from_shared_model(shared)
        .with_instructions("be concise")
        .run(LocalAgentInput(request))
        .await
        .expect("local wrapper should use the shared language input path");

    assert!(terminal.is_completed());
    assert_eq!(model.requests(), vec![expected]);
}

#[tokio::test]
async fn agent_accepts_every_shared_language_input_form() {
    let model = ScriptedModel::new(
        "agent",
        "agent.messages",
        "model",
        [
            final_response("plain"),
            final_response("message"),
            final_response("messages"),
            final_response("request"),
        ],
    );
    let shared: Arc<dyn LanguageModel> = model.clone();
    let agent = Agent::from_shared_model(shared);
    let message = Message::text(MessageRole::User, "one message");
    let messages = vec![
        Message::text(MessageRole::Developer, "policy"),
        Message::text(MessageRole::User, "message list"),
    ];
    let request = LanguageRequest::new(vec![Message::text(MessageRole::User, "full request")]);

    agent.run("plain string").await.expect("plain input runs");
    agent
        .run(message.clone())
        .await
        .expect("message input runs");
    agent
        .run(messages.clone())
        .await
        .expect("message-list input runs");
    agent
        .run(request.clone())
        .await
        .expect("complete request input runs");

    assert_eq!(
        model.requests(),
        vec![
            LanguageRequest::new(vec![Message::text(MessageRole::User, "plain string")]),
            LanguageRequest::new(vec![message]),
            LanguageRequest::new(messages),
            request,
        ]
    );
}

#[tokio::test]
async fn agent_rejects_invalid_language_input_before_model_invocation() {
    let model = ScriptedModel::new(
        "agent",
        "agent.messages",
        "model",
        [final_response("unused")],
    );
    let shared: Arc<dyn LanguageModel> = model.clone();
    let mut request = request();
    request.structured_output = Some(StructuredOutputSpec {
        name: String::new(),
        description: None,
        schema: json!({"type": "object"}),
        strict: true,
    });

    let error = Agent::from_shared_model(shared)
        .run(request)
        .await
        .expect_err("invalid portable request must fail before model invocation");

    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert!(matches!(
        error
            .sensitive_source()
            .and_then(|source| source.expose().downcast_ref::<LanguageRequestError>()),
        Some(LanguageRequestError::EmptyStructuredOutputName)
    ));
    assert_eq!(model.stream_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn agent_rejects_non_instruction_roles() {
    let model = ScriptedModel::new(
        "agent",
        "agent.messages",
        "model",
        [final_response("unused")],
    );
    let shared: Arc<dyn LanguageModel> = model;
    let error = Agent::from_shared_model(shared)
        .with_instruction_messages([Message::text(MessageRole::User, "not trusted")])
        .expect_err("user messages are per-run input, not reusable instructions");

    assert!(matches!(
        error,
        AgentConfigError::InvalidInstructionRole {
            index: 0,
            role: MessageRole::User
        }
    ));
}
