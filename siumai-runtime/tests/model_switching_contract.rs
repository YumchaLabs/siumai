use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::json;
use siumai_core::stream::established_stream;
use siumai_core::{
    CallOptions, ContentPart, Error, ErrorKind, ExecutionOwner, FinishReason, LanguageModel,
    LanguageRequest, LanguageResponse, LanguageStream, LanguageStreamEvent, Message, MessageRole,
    Model, ModelDescriptor, ModelFamily, ModelId, OpaqueProviderItem, ProtocolId, ProviderId,
    ProviderProvenance, StreamTerminal, ToolCall, ToolOutcome, ToolSpec, Usage,
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
            .with_protocol(ProtocolId::new(protocol).expect("valid protocol")),
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
    ) -> Result<LanguageResponse, Error> {
        self.generate_calls.fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorKind::Internal,
            "step engine must use streaming model calls",
        ))
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

fn local_call() -> ToolCall {
    ToolCall {
        id: "call-1".to_string(),
        name: "lookup".to_string(),
        arguments: json!({"query": "rust"}),
        owner: ExecutionOwner::Local,
    }
}

fn opaque(provider: &str, protocol: &str, model: &str, kind: &str) -> OpaqueProviderItem {
    OpaqueProviderItem::new(
        ProviderProvenance {
            provider: ProviderId::new(provider).expect("valid provider"),
            platform: None,
            protocol: protocol.to_string(),
            model: ModelId::new(model).expect("valid model"),
        },
        kind,
        json!({"id": "native-state"}),
    )
    .expect("valid opaque item")
}

fn tool_response(native: Option<OpaqueProviderItem>) -> LanguageResponse {
    let mut content = Vec::new();
    if let Some(native) = native {
        content.push(ContentPart::ProviderOpaque(native));
    }
    content.push(ContentPart::ToolCall(local_call()));
    LanguageResponse::completed(content, FinishReason::ToolCalls, Usage::default())
        .expect("valid tool response")
}

fn final_response(text: &str) -> LanguageResponse {
    LanguageResponse::completed(
        vec![ContentPart::Text {
            text: text.to_string(),
        }],
        FinishReason::Stop,
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
        .stream(request(), CallOptions::default())
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
            .content
            .iter()
            .any(|part| matches!(part, ContentPart::ProviderOpaque(_)))
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
            .content
            .iter()
            .any(|part| matches!(part, ContentPart::ProviderOpaque(_)))
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
        assert_eq!(request.messages[0].role, MessageRole::System);
        assert_eq!(request.messages[1].role, MessageRole::User);
    }
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
