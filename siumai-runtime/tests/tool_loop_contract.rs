use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Value, json};
use siumai_core::stream::established_stream;
use siumai_core::{
    CallOptions, Cancellation, ContentPart, Error, ErrorKind, ExecutionOwner, FinishReason,
    LanguageModel, LanguageRequest, LanguageResponse, LanguageResponseStatus, LanguageStream,
    LanguageStreamEvent, Message, MessageRole, Model, ModelDescriptor, ModelFamily, ModelId,
    ProviderId, StreamTerminal, ToolCall, ToolOutcome, ToolSpec, Usage, UsageValue,
};
use siumai_runtime::snapshot::ToolExecutionStatus;
use siumai_runtime::tool::{
    ApprovalDecider, ApprovalDecision, ApprovalDecisionError, ApprovalDecisionFuture,
    ApprovalPolicy, ApprovalPolicyFingerprint, ApprovalRequest, EffectCertainty, ToolBinding,
    ToolConcurrency, ToolEffect, ToolExecutionError, ToolSet,
};
use siumai_runtime::{
    OutputDescriptor, RepairPolicy, RunBudget, RunEvent, RunReport, RunTerminal, RunTimeoutKind,
    RunTimeouts, Runtime, StructuredOutputRunError, SuspensionReason, ToolLoop, ToolOutcomeAction,
    ToolOutcomePolicy,
};

struct ScriptStep {
    handshake_delay: Duration,
    handshake_error: bool,
    events: Vec<(Duration, LanguageStreamEvent)>,
}

impl ScriptStep {
    fn immediate(events: Vec<LanguageStreamEvent>) -> Self {
        Self {
            handshake_delay: Duration::ZERO,
            handshake_error: false,
            events: events
                .into_iter()
                .map(|event| (Duration::ZERO, event))
                .collect(),
        }
    }

    fn delayed(events: Vec<(Duration, LanguageStreamEvent)>) -> Self {
        Self {
            handshake_delay: Duration::ZERO,
            handshake_error: false,
            events,
        }
    }

    fn handshake_error() -> Self {
        Self {
            handshake_delay: Duration::ZERO,
            handshake_error: true,
            events: Vec::new(),
        }
    }
}

struct ScriptedModel {
    descriptor: ModelDescriptor,
    scripts: Mutex<VecDeque<ScriptStep>>,
    requests: Mutex<Vec<LanguageRequest>>,
    generate_calls: AtomicUsize,
    stream_calls: AtomicUsize,
}

impl ScriptedModel {
    fn new(scripts: impl IntoIterator<Item = ScriptStep>) -> Arc<Self> {
        Arc::new(Self {
            descriptor: ModelDescriptor::new(
                ProviderId::new("scripted").expect("valid provider"),
                ModelId::new("tool-loop-model").expect("valid model"),
                ModelFamily::Language,
            ),
            scripts: Mutex::new(scripts.into_iter().collect()),
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
            "tool loop must not call generate",
        ))
    }

    async fn stream(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        self.stream_calls.fetch_add(1, Ordering::SeqCst);
        self.requests.lock().expect("request lock").push(request);
        let script = self
            .scripts
            .lock()
            .expect("script lock")
            .pop_front()
            .ok_or_else(|| Error::new(ErrorKind::Internal, "missing scripted model step"))?;
        if !script.handshake_delay.is_zero() {
            tokio::time::sleep(script.handshake_delay).await;
        }
        if script.handshake_error {
            return Err(Error::new(
                ErrorKind::Transport,
                "scripted handshake failed",
            ));
        }

        let cancellation = options.cancellation().clone();
        Ok(established_stream(cancellation, move |_| {
            async_stream::stream! {
                for (delay, event) in script.events {
                    if !delay.is_zero() {
                        tokio::time::sleep(delay).await;
                    }
                    yield Ok(event);
                }
            }
        }))
    }
}

fn user_request() -> LanguageRequest {
    LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")])
}

fn tool_spec(name: &str) -> ToolSpec {
    ToolSpec::new(
        name,
        Some(format!("{name} tool")),
        json!({ "type": "object" }),
    )
    .expect("valid tool spec")
}

fn local_call(id: &str, name: &str, arguments: Value) -> ToolCall {
    ToolCall {
        id: id.to_string(),
        name: name.to_string(),
        arguments,
        owner: ExecutionOwner::Local,
    }
}

fn provider_call(id: &str, name: &str) -> ToolCall {
    ToolCall {
        id: id.to_string(),
        name: name.to_string(),
        arguments: json!({}),
        owner: ExecutionOwner::Provider {
            provider: ProviderId::new("scripted").expect("valid provider"),
        },
    }
}

fn tool_response(calls: Vec<ToolCall>) -> LanguageResponse {
    LanguageResponse::completed(
        calls.into_iter().map(ContentPart::ToolCall).collect(),
        FinishReason::ToolCalls,
        Usage::default(),
    )
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

fn failed_response(text: &str, tokens: u64) -> LanguageResponse {
    LanguageResponse::new(
        LanguageResponseStatus::Failed,
        vec![ContentPart::Text {
            text: text.to_string(),
        }],
        FinishReason::Error,
        Usage::default().with_total_tokens(tokens),
    )
    .expect("valid failed response")
}

fn cancelled_response(text: &str, tokens: u64) -> LanguageResponse {
    LanguageResponse::new(
        LanguageResponseStatus::Cancelled,
        vec![ContentPart::Text {
            text: text.to_string(),
        }],
        FinishReason::Cancelled,
        Usage::default().with_total_tokens(tokens),
    )
    .expect("valid cancelled response")
}

fn terminal_step(response: LanguageResponse) -> ScriptStep {
    ScriptStep::immediate(vec![LanguageStreamEvent::Terminal(
        StreamTerminal::Completed {
            response: Box::new(response),
        },
    )])
}

fn executable_binding(
    name: &str,
    execute: impl Fn(siumai_runtime::tool::ToolExecutionRequest) -> ToolFuture + Send + Sync + 'static,
) -> ToolBinding {
    ToolBinding::from_fn(tool_spec(name), "v1", |_| Ok(()), execute)
        .expect("valid binding")
        .with_approval_policy(ApprovalPolicy::NotRequired)
}

type ToolFuture = std::pin::Pin<
    Box<
        dyn std::future::Future<
                Output = Result<ToolOutcome, siumai_runtime::tool::ToolExecutionError>,
            > + Send,
    >,
>;

struct StaticApprovalDecider {
    fingerprint: ApprovalPolicyFingerprint,
    decision: Result<ApprovalDecision, ApprovalDecisionError>,
    calls: Arc<AtomicUsize>,
}

impl StaticApprovalDecider {
    fn new(decision: ApprovalDecision, calls: Arc<AtomicUsize>) -> Arc<Self> {
        Arc::new(Self {
            fingerprint: ApprovalPolicyFingerprint::new("test.approval-policy.v1")
                .expect("valid fingerprint"),
            decision: Ok(decision),
            calls,
        })
    }

    fn failing(error: ApprovalDecisionError, calls: Arc<AtomicUsize>) -> Arc<Self> {
        Arc::new(Self {
            fingerprint: ApprovalPolicyFingerprint::new("test.approval-policy.v1")
                .expect("valid fingerprint"),
            decision: Err(error),
            calls,
        })
    }
}

impl ApprovalDecider for StaticApprovalDecider {
    fn decide<'a>(&'a self, _request: &'a ApprovalRequest) -> ApprovalDecisionFuture<'a> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let decision = self.decision.clone();
        Box::pin(async move { decision })
    }

    fn fingerprint(&self) -> &ApprovalPolicyFingerprint {
        &self.fingerprint
    }
}

struct PendingApprovalDecider {
    fingerprint: ApprovalPolicyFingerprint,
    cancel_on_decide: Option<Cancellation>,
}

impl PendingApprovalDecider {
    fn new(cancel_on_decide: Option<Cancellation>) -> Arc<Self> {
        Arc::new(Self {
            fingerprint: ApprovalPolicyFingerprint::new("test.pending-approval-policy.v1")
                .expect("valid fingerprint"),
            cancel_on_decide,
        })
    }
}

impl ApprovalDecider for PendingApprovalDecider {
    fn decide<'a>(&'a self, _request: &'a ApprovalRequest) -> ApprovalDecisionFuture<'a> {
        if let Some(cancellation) = &self.cancel_on_decide {
            cancellation.cancel();
        }
        Box::pin(std::future::pending())
    }

    fn fingerprint(&self) -> &ApprovalPolicyFingerprint {
        &self.fingerprint
    }
}

fn boxed_tool_future<F>(future: F) -> ToolFuture
where
    F: std::future::Future<Output = Result<ToolOutcome, siumai_runtime::tool::ToolExecutionError>>
        + Send
        + 'static,
{
    Box::pin(future)
}

async fn collect_terminal(
    loop_: &ToolLoop,
    request: LanguageRequest,
) -> (Vec<&'static str>, RunTerminal) {
    collect_terminal_with_options(loop_, request, CallOptions::default()).await
}

async fn collect_terminal_with_options(
    loop_: &ToolLoop,
    request: LanguageRequest,
    options: CallOptions,
) -> (Vec<&'static str>, RunTerminal) {
    let mut stream = loop_
        .stream(request, options)
        .await
        .expect("first stream establishes");
    let mut trace = Vec::new();
    while let Some(event) = stream.next().await {
        match event {
            RunEvent::Started { .. } => trace.push("started"),
            RunEvent::StepStarted { .. } => trace.push("step_started"),
            RunEvent::Model { .. } => trace.push("model"),
            RunEvent::ToolPrepared { .. } => trace.push("tool_prepared"),
            RunEvent::ToolCompleted { .. } => trace.push("tool_completed"),
            RunEvent::StepFinished { .. } => trace.push("step_finished"),
            RunEvent::Terminal(terminal) => {
                trace.push("terminal");
                assert!(
                    stream.next().await.is_none(),
                    "run stream must close immediately after its terminal"
                );
                return (trace, terminal);
            }
            _ => trace.push("other"),
        }
    }
    panic!("run stream must emit one terminal")
}

fn completed_report(terminal: &RunTerminal) -> &RunReport {
    match terminal {
        RunTerminal::Completed { report } => report,
        other => panic!("expected completed terminal, got {other:?}"),
    }
}

#[tokio::test]
async fn run_and_stream_share_one_stream_only_execution_trace() {
    fn scripts() -> Vec<ScriptStep> {
        vec![
            terminal_step(tool_response(vec![local_call(
                "call_1",
                "lookup",
                json!({ "key": "rust" }),
            )])),
            terminal_step(final_response("done")),
        ]
    }
    fn request_with_untrusted_catalog() -> LanguageRequest {
        let mut request = user_request();
        request.tools.push(tool_spec("client-only"));
        request
    }
    let tools = ToolSet::from_bindings([executable_binding("lookup", |_| {
        boxed_tool_future(async {
            Ok(ToolOutcome::Success {
                value: json!({ "answer": 42 }),
            })
        })
    })])
    .expect("unique tool");

    let stream_model = ScriptedModel::new(scripts());
    let stream_loop = ToolLoop::new(stream_model.clone(), tools.clone());
    let (trace, stream_terminal) =
        collect_terminal(&stream_loop, request_with_untrusted_catalog()).await;

    let run_model = ScriptedModel::new(scripts());
    let run_loop = ToolLoop::new(run_model.clone(), tools);
    let run_terminal = run_loop
        .run(request_with_untrusted_catalog(), CallOptions::default())
        .await
        .expect("run succeeds");

    assert_eq!(
        trace,
        vec![
            "started",
            "step_started",
            "tool_prepared",
            "tool_completed",
            "step_finished",
            "step_started",
            "step_finished",
            "terminal",
        ]
    );
    let stream_report = completed_report(&stream_terminal);
    let run_report = completed_report(&run_terminal);
    assert_eq!(stream_report.initial_target(), run_report.initial_target());
    assert_eq!(stream_report.messages(), run_report.messages());
    assert_eq!(stream_report.steps(), run_report.steps());
    assert_eq!(stream_report.usage(), run_report.usage());
    assert_eq!(stream_report.budget(), run_report.budget());
    assert_eq!(
        stream_report
            .execution_log()
            .events()
            .iter()
            .map(|event| event.status())
            .collect::<Vec<_>>(),
        run_report
            .execution_log()
            .events()
            .iter()
            .map(|event| event.status())
            .collect::<Vec<_>>()
    );
    assert_eq!(stream_model.stream_calls.load(Ordering::SeqCst), 2);
    assert_eq!(run_model.stream_calls.load(Ordering::SeqCst), 2);
    assert_eq!(stream_model.generate_calls.load(Ordering::SeqCst), 0);
    assert_eq!(run_model.generate_calls.load(Ordering::SeqCst), 0);

    let requests = run_model.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].tools.len(), 1);
    assert_eq!(requests[0].tools[0].name(), "lookup");
    assert_eq!(
        requests[1]
            .messages
            .iter()
            .map(Message::role)
            .collect::<Vec<_>>(),
        vec![MessageRole::User, MessageRole::Assistant, MessageRole::Tool]
    );
    assert!(requests[1].messages[1].annotations().is_empty());
    assert!(
        requests[1].messages[1]
            .content()
            .iter()
            .all(|part| part.annotations().is_empty())
    );
}

#[tokio::test]
async fn known_usage_is_aggregated_without_losing_the_first_step() {
    let first = LanguageResponse::completed(
        vec![ContentPart::ToolCall(local_call(
            "call_1",
            "lookup",
            json!({}),
        ))],
        FinishReason::ToolCalls,
        Usage::default().with_total_tokens(3_u64),
    )
    .unwrap();
    let second = LanguageResponse::completed(
        vec![ContentPart::Text {
            text: "done".to_string(),
        }],
        FinishReason::Stop,
        Usage::default().with_total_tokens(5_u64),
    )
    .unwrap();
    let model = ScriptedModel::new(vec![terminal_step(first), terminal_step(second)]);
    let tools = ToolSet::from_bindings([executable_binding("lookup", |_| {
        boxed_tool_future(async { Ok(ToolOutcome::Success { value: Value::Null }) })
    })])
    .unwrap();

    let terminal = ToolLoop::new(model, tools)
        .run(user_request(), CallOptions::default())
        .await
        .unwrap();

    assert_eq!(
        completed_report(&terminal).usage().total_tokens,
        UsageValue::Known(8)
    );
}

#[tokio::test]
async fn budget_exceeded_terminal_retains_the_observed_usage() {
    let response = LanguageResponse::completed(
        vec![ContentPart::Text {
            text: "over budget".to_string(),
        }],
        FinishReason::Stop,
        Usage::default().with_total_tokens(8_u64),
    )
    .expect("valid response");
    let budget = RunBudget::builder()
        .max_known_tokens(Some(4))
        .build()
        .expect("valid budget");
    let runtime = Runtime::builder().with_run_budget(budget).build();
    let model = ScriptedModel::new([terminal_step(response)]);
    let loop_ = ToolLoop::new(model, ToolSet::default()).with_runtime(runtime);

    let (_, terminal) = collect_terminal(&loop_, user_request()).await;
    assert!(matches!(&terminal, RunTerminal::BudgetExceeded { .. }));
    assert_eq!(
        terminal
            .report()
            .expect("budget terminal has report")
            .usage()
            .total_tokens,
        UsageValue::Known(8)
    );
}

#[tokio::test]
async fn failed_and_cancelled_terminal_responses_are_recorded_without_continuation_history() {
    let failed_model =
        ScriptedModel::new([ScriptStep::immediate(vec![LanguageStreamEvent::Terminal(
            StreamTerminal::Failed {
                error: Error::new(ErrorKind::Provider, "provider failed"),
                response: Some(Box::new(failed_response("partial failure", 7))),
            },
        )])]);
    let failed_loop = ToolLoop::new(failed_model, ToolSet::default());
    let (_, failed_terminal) = collect_terminal(&failed_loop, user_request()).await;
    assert!(matches!(&failed_terminal, RunTerminal::Failed { .. }));
    let failed_report = failed_terminal
        .report()
        .expect("failed terminal retains report");
    assert_eq!(failed_report.steps().len(), 1);
    assert_eq!(failed_report.usage().total_tokens, UsageValue::Known(7));
    assert_eq!(failed_report.budget().known_tokens(), 7);
    assert_eq!(failed_report.messages().len(), 1);
    assert!(matches!(
        failed_report.steps()[0].response().status(),
        LanguageResponseStatus::Failed
    ));

    let cancelled_model =
        ScriptedModel::new([ScriptStep::immediate(vec![LanguageStreamEvent::Terminal(
            StreamTerminal::Cancelled {
                reason: "provider cancelled".to_string(),
                response: Some(Box::new(cancelled_response("partial cancellation", 5))),
            },
        )])]);
    let cancelled_loop = ToolLoop::new(cancelled_model, ToolSet::default());
    let (_, cancelled_terminal) = collect_terminal(&cancelled_loop, user_request()).await;
    assert!(matches!(&cancelled_terminal, RunTerminal::Cancelled { .. }));
    let cancelled_report = cancelled_terminal
        .report()
        .expect("cancelled terminal retains report");
    assert_eq!(cancelled_report.steps().len(), 1);
    assert_eq!(cancelled_report.usage().total_tokens, UsageValue::Known(5));
    assert_eq!(cancelled_report.budget().known_tokens(), 5);
    assert_eq!(cancelled_report.messages().len(), 1);
    assert!(matches!(
        cancelled_report.steps()[0].response().status(),
        LanguageResponseStatus::Cancelled
    ));
}

#[tokio::test]
async fn terminal_response_usage_budget_exhaustion_preempts_provider_failure() {
    let budget = RunBudget::builder()
        .max_known_tokens(Some(4))
        .build()
        .expect("valid budget");
    let runtime = Runtime::builder().with_run_budget(budget).build();
    let model = ScriptedModel::new([ScriptStep::immediate(vec![LanguageStreamEvent::Terminal(
        StreamTerminal::Failed {
            error: Error::new(ErrorKind::Provider, "provider failed"),
            response: Some(Box::new(failed_response("over budget", 8))),
        },
    )])]);
    let loop_ = ToolLoop::new(model, ToolSet::default()).with_runtime(runtime);

    let (_, terminal) = collect_terminal(&loop_, user_request()).await;
    assert!(matches!(&terminal, RunTerminal::BudgetExceeded { .. }));
    let report = terminal.report().expect("budget terminal retains report");
    assert_eq!(report.steps().len(), 1);
    assert_eq!(report.usage().total_tokens, UsageValue::Known(8));
    assert_eq!(report.messages().len(), 1);
}

#[tokio::test]
async fn seeded_repair_budget_exhaustion_is_a_runtime_terminal() {
    let descriptor = OutputDescriptor::<Value>::typed_json("payload")
        .expect("valid descriptor")
        .with_repair_policy(RepairPolicy::OneAttempt);
    let budget = RunBudget::builder()
        .max_model_steps(1)
        .build()
        .expect("valid budget");
    let runtime = Runtime::builder().with_run_budget(budget).build();
    let model = ScriptedModel::new([terminal_step(final_response("{"))]);
    let runner = runtime.structured_output(model.clone(), descriptor);

    let error = runner
        .generate(user_request(), CallOptions::default())
        .await
        .expect_err("repair must exhaust the shared model-step budget");
    assert!(matches!(
        &error,
        StructuredOutputRunError::Runtime(terminal)
            if matches!(terminal.as_ref(), RunTerminal::BudgetExceeded { .. })
    ));
    let report = error.report().expect("runtime terminal retains report");
    assert_eq!(report.steps().len(), 1);
    assert_eq!(report.budget().model_steps(), 1);
    assert_eq!(model.stream_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn seeded_repair_handshake_failures_keep_the_established_report() {
    fn descriptor() -> OutputDescriptor<Value> {
        OutputDescriptor::<Value>::typed_json("payload")
            .expect("valid descriptor")
            .with_repair_policy(RepairPolicy::OneAttempt)
    }

    let failed_model = ScriptedModel::new([
        terminal_step(final_response("{")),
        ScriptStep::handshake_error(),
    ]);
    let failed = Runtime::default()
        .structured_output(failed_model.clone(), descriptor())
        .generate(user_request(), CallOptions::default())
        .await
        .expect_err("repair handshake failure must be terminal");
    assert!(matches!(
        &failed,
        StructuredOutputRunError::Validation { .. }
    ));
    assert_eq!(
        failed
            .report()
            .expect("failed terminal retains report")
            .budget()
            .model_steps(),
        2
    );
    assert_eq!(failed_model.stream_calls.load(Ordering::SeqCst), 2);

    let timed_out_model = ScriptedModel::new([
        terminal_step(final_response("{")),
        ScriptStep {
            handshake_delay: Duration::from_millis(100),
            handshake_error: false,
            events: Vec::new(),
        },
    ]);
    let timeout_runtime = runtime_with_timeouts(
        Duration::from_secs(2),
        Duration::from_millis(25),
        Duration::from_secs(2),
        Duration::from_secs(2),
        Duration::from_secs(2),
    );
    let timed_out = timeout_runtime
        .structured_output(timed_out_model.clone(), descriptor())
        .generate(user_request(), CallOptions::default())
        .await
        .expect_err("repair handshake timeout must be terminal");
    assert!(matches!(
        &timed_out,
        StructuredOutputRunError::Runtime(terminal)
            if matches!(
                terminal.as_ref(),
                RunTerminal::TimedOut {
                    kind: RunTimeoutKind::ModelStep,
                    ..
                }
            )
    ));
    assert_eq!(
        timed_out
            .report()
            .expect("timeout terminal retains report")
            .budget()
            .model_steps(),
        2
    );

    let cancelled_model = ScriptedModel::new([
        terminal_step(final_response("{")),
        ScriptStep {
            handshake_delay: Duration::from_millis(100),
            handshake_error: false,
            events: Vec::new(),
        },
    ]);
    let cancellation = Cancellation::new();
    let cancellation_signal = cancellation.clone();
    let cancellation_observer = cancelled_model.clone();
    let cancel_task = tokio::spawn(async move {
        tokio::time::timeout(Duration::from_secs(1), async {
            while cancellation_observer.stream_calls.load(Ordering::SeqCst) < 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("repair handshake must start");
        cancellation_signal.cancel();
    });
    let cancelled = Runtime::default()
        .structured_output(cancelled_model, descriptor())
        .generate(
            user_request(),
            CallOptions::default().with_cancellation(cancellation),
        )
        .await
        .expect_err("repair handshake cancellation must be terminal");
    cancel_task.await.expect("cancellation task succeeds");
    assert!(matches!(
        &cancelled,
        StructuredOutputRunError::Runtime(terminal)
            if matches!(terminal.as_ref(), RunTerminal::Cancelled { .. })
    ));
    assert_eq!(
        cancelled
            .report()
            .expect("cancelled terminal retains report")
            .budget()
            .model_steps(),
        2
    );
}

#[tokio::test]
async fn parallel_read_only_results_are_ordered_and_respect_both_limits() {
    let active = Arc::new(AtomicUsize::new(0));
    let maximum = Arc::new(AtomicUsize::new(0));
    let observed_active = Arc::clone(&active);
    let observed_maximum = Arc::clone(&maximum);
    let binding = executable_binding("lookup", move |request| {
        let active = Arc::clone(&observed_active);
        let maximum = Arc::clone(&observed_maximum);
        boxed_tool_future(async move {
            let ordinal = request
                .arguments()
                .get("ordinal")
                .and_then(Value::as_u64)
                .expect("ordinal argument");
            let current = active.fetch_add(1, Ordering::SeqCst) + 1;
            maximum.fetch_max(current, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis((5 - ordinal) * 10)).await;
            active.fetch_sub(1, Ordering::SeqCst);
            Ok(ToolOutcome::Success {
                value: json!({ "ordinal": ordinal }),
            })
        })
    })
    .with_effect(ToolEffect::ReadOnly)
    .with_concurrency(ToolConcurrency::SafeParallel {
        max_in_flight: std::num::NonZeroUsize::new(2).expect("non-zero"),
    });
    let tools = ToolSet::from_bindings([binding]).expect("unique tool");
    let calls = (1..=4)
        .map(|ordinal| {
            local_call(
                &format!("call_{ordinal}"),
                "lookup",
                json!({ "ordinal": ordinal }),
            )
        })
        .collect();
    let model = ScriptedModel::new([
        terminal_step(tool_response(calls)),
        terminal_step(final_response("done")),
    ]);
    let budget = RunBudget::builder()
        .max_concurrent_tools(3)
        .build()
        .expect("valid budget");
    let runtime = Runtime::builder().with_run_budget(budget).build();
    let loop_ = ToolLoop::new(model, tools).with_runtime(runtime);

    let (_, terminal) = collect_terminal(&loop_, user_request()).await;
    let first_step = &completed_report(&terminal).steps()[0];
    assert_eq!(maximum.load(Ordering::SeqCst), 2);
    assert_eq!(
        first_step
            .tool_results()
            .iter()
            .map(|result| result.call_id.as_str())
            .collect::<Vec<_>>(),
        vec!["call_1", "call_2", "call_3", "call_4"]
    );
}

#[tokio::test]
async fn parallel_read_only_calls_share_the_global_limit_across_bindings() {
    let active = Arc::new(AtomicUsize::new(0));
    let maximum = Arc::new(AtomicUsize::new(0));
    let make_binding = |name: &str| {
        let observed_active = Arc::clone(&active);
        let observed_maximum = Arc::clone(&maximum);
        executable_binding(name, move |_| {
            let active = Arc::clone(&observed_active);
            let maximum = Arc::clone(&observed_maximum);
            boxed_tool_future(async move {
                let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                maximum.fetch_max(current, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(20)).await;
                active.fetch_sub(1, Ordering::SeqCst);
                Ok(ToolOutcome::Success { value: Value::Null })
            })
        })
        .with_effect(ToolEffect::ReadOnly)
        .with_concurrency(ToolConcurrency::SafeParallel {
            max_in_flight: std::num::NonZeroUsize::new(4).expect("non-zero"),
        })
    };
    let tools = ToolSet::from_bindings([make_binding("alpha"), make_binding("beta")])
        .expect("unique tools");
    let model = ScriptedModel::new([
        terminal_step(tool_response(vec![
            local_call("alpha_1", "alpha", json!({})),
            local_call("beta_1", "beta", json!({})),
            local_call("alpha_2", "alpha", json!({})),
            local_call("beta_2", "beta", json!({})),
        ])),
        terminal_step(final_response("done")),
    ]);
    let budget = RunBudget::builder()
        .max_concurrent_tools(2)
        .build()
        .expect("valid budget");
    let runtime = Runtime::builder().with_run_budget(budget).build();
    let loop_ = ToolLoop::new(model, tools).with_runtime(runtime);

    let (_, terminal) = collect_terminal(&loop_, user_request()).await;
    assert!(matches!(terminal, RunTerminal::Completed { .. }));
    assert_eq!(maximum.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn side_effecting_tools_remain_sequential_even_if_parallel_is_declared() {
    let active = Arc::new(AtomicUsize::new(0));
    let maximum = Arc::new(AtomicUsize::new(0));
    let observed_active = Arc::clone(&active);
    let observed_maximum = Arc::clone(&maximum);
    let binding = executable_binding("write", move |_| {
        let active = Arc::clone(&observed_active);
        let maximum = Arc::clone(&observed_maximum);
        boxed_tool_future(async move {
            let current = active.fetch_add(1, Ordering::SeqCst) + 1;
            maximum.fetch_max(current, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(15)).await;
            active.fetch_sub(1, Ordering::SeqCst);
            Ok(ToolOutcome::Success { value: Value::Null })
        })
    })
    .with_concurrency(ToolConcurrency::SafeParallel {
        max_in_flight: std::num::NonZeroUsize::new(4).expect("non-zero"),
    });
    let tools = ToolSet::from_bindings([binding]).expect("unique tool");
    let model = ScriptedModel::new([
        terminal_step(tool_response(vec![
            local_call("write_1", "write", json!({})),
            local_call("write_2", "write", json!({})),
        ])),
        terminal_step(final_response("done")),
    ]);
    let loop_ = ToolLoop::new(model, tools);

    let (_, terminal) = collect_terminal(&loop_, user_request()).await;
    assert!(matches!(terminal, RunTerminal::Completed { .. }));
    assert_eq!(maximum.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn required_approval_prepares_and_suspends_without_execution() {
    let executions = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&executions);
    let binding = ToolBinding::from_fn(
        tool_spec("charge"),
        "v1",
        |_| Ok(()),
        move |_| {
            let observed = Arc::clone(&observed);
            async move {
                observed.fetch_add(1, Ordering::SeqCst);
                Ok(ToolOutcome::Success { value: Value::Null })
            }
        },
    )
    .expect("valid binding");
    let tools = ToolSet::from_bindings([binding]).expect("unique tool");
    let model = ScriptedModel::new([terminal_step(tool_response(vec![local_call(
        "charge_1",
        "charge",
        json!({ "amount": 10 }),
    )]))]);
    let loop_ = ToolLoop::new(model, tools);

    let (_, terminal) = collect_terminal(&loop_, user_request()).await;
    let report = terminal.report().expect("suspension has report");
    assert!(matches!(
        &terminal,
        RunTerminal::Suspended {
            reason: SuspensionReason::AwaitingApproval { .. },
            ..
        }
    ));
    assert_eq!(executions.load(Ordering::SeqCst), 0);
    assert_eq!(
        report.execution_log().status("charge_1"),
        Some(ToolExecutionStatus::Prepared)
    );
}

#[tokio::test]
async fn host_auto_approval_authorizes_the_frozen_binding_once() {
    let executions = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&executions);
    let binding = ToolBinding::from_fn(
        tool_spec("charge"),
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
    let model = ScriptedModel::new([
        terminal_step(tool_response(vec![local_call(
            "charge_1",
            "charge",
            json!({ "amount": 10 }),
        )])),
        terminal_step(final_response("done")),
    ]);
    let decisions = Arc::new(AtomicUsize::new(0));
    let decider = StaticApprovalDecider::new(ApprovalDecision::Approve, Arc::clone(&decisions));
    let loop_ = ToolLoop::new(model, tools).with_approval_decider(decider);

    let (_, terminal) = collect_terminal(&loop_, user_request()).await;
    assert!(matches!(terminal, RunTerminal::Completed { .. }));
    assert_eq!(decisions.load(Ordering::SeqCst), 1);
    assert_eq!(executions.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn approval_decider_is_never_called_for_not_required_bindings() {
    let executions = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&executions);
    let binding = executable_binding("lookup", move |_| {
        let observed = Arc::clone(&observed);
        boxed_tool_future(async move {
            observed.fetch_add(1, Ordering::SeqCst);
            Ok(ToolOutcome::Success { value: json!(true) })
        })
    });
    let tools = ToolSet::from_bindings([binding]).expect("unique tool");
    let model = ScriptedModel::new([
        terminal_step(tool_response(vec![local_call(
            "lookup_1",
            "lookup",
            json!({ "query": "rust" }),
        )])),
        terminal_step(final_response("done")),
    ]);
    let decisions = Arc::new(AtomicUsize::new(0));
    let denial = ApprovalDecision::deny("must not be observed").expect("valid denial");
    let decider = StaticApprovalDecider::new(denial, Arc::clone(&decisions));
    let loop_ = ToolLoop::new(model, tools).with_approval_decider(decider);

    let (_, terminal) = collect_terminal(&loop_, user_request()).await;
    assert!(matches!(terminal, RunTerminal::Completed { .. }));
    assert_eq!(decisions.load(Ordering::SeqCst), 0);
    assert_eq!(executions.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn host_denial_is_a_typed_result_and_never_dispatches() {
    let executions = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&executions);
    let binding = ToolBinding::from_fn(
        tool_spec("charge"),
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
    let model = ScriptedModel::new([terminal_step(tool_response(vec![local_call(
        "charge_1",
        "charge",
        json!({ "amount": 10 }),
    )]))]);
    let decisions = Arc::new(AtomicUsize::new(0));
    let denial = ApprovalDecision::deny("host policy denied execution").expect("valid denial");
    let decider = StaticApprovalDecider::new(denial, Arc::clone(&decisions));
    let loop_ = ToolLoop::new(model, tools).with_approval_decider(decider);

    let (_, terminal) = collect_terminal(&loop_, user_request()).await;
    let report = terminal.report().expect("stopped run has report");
    assert!(matches!(terminal, RunTerminal::Stopped { .. }));
    assert_eq!(decisions.load(Ordering::SeqCst), 1);
    assert_eq!(executions.load(Ordering::SeqCst), 0);
    assert!(matches!(
        report.steps()[0].tool_results()[0].outcome,
        ToolOutcome::Denied { .. }
    ));
}

#[tokio::test]
async fn approval_policy_failure_is_terminal_and_never_dispatches() {
    let executions = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&executions);
    let binding = ToolBinding::from_fn(
        tool_spec("charge"),
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
    let model = ScriptedModel::new([terminal_step(tool_response(vec![local_call(
        "charge_1",
        "charge",
        json!({ "amount": 10 }),
    )]))]);
    let decisions = Arc::new(AtomicUsize::new(0));
    let decider = StaticApprovalDecider::failing(
        ApprovalDecisionError::Unavailable {
            message: "policy backend unavailable".to_string(),
        },
        Arc::clone(&decisions),
    );
    let loop_ = ToolLoop::new(model, tools).with_approval_decider(decider);

    let (_, terminal) = collect_terminal(&loop_, user_request()).await;
    assert!(matches!(
        terminal,
        RunTerminal::Failed { ref error, .. }
            if error.kind() == ErrorKind::Internal
                && error.message() == "host approval decision failed"
    ));
    assert_eq!(decisions.load(Ordering::SeqCst), 1);
    assert_eq!(executions.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn pending_approval_policy_obeys_total_timeout_without_dispatch() {
    let executions = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&executions);
    let binding = ToolBinding::from_fn(
        tool_spec("charge"),
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
    let model = ScriptedModel::new([terminal_step(tool_response(vec![local_call(
        "charge_1",
        "charge",
        json!({ "amount": 10 }),
    )]))]);
    let timeouts = RunTimeouts::new(
        Duration::from_millis(30),
        Duration::from_secs(1),
        Duration::from_secs(1),
        Duration::from_secs(1),
        Duration::from_secs(1),
    )
    .expect("valid timeouts");
    let budget = RunBudget::builder()
        .timeouts(timeouts)
        .build()
        .expect("valid budget");
    let runtime = Runtime::builder().with_run_budget(budget).build();
    let loop_ = ToolLoop::new(model, tools)
        .with_runtime(runtime)
        .with_approval_decider(PendingApprovalDecider::new(None));

    let (_, terminal) = collect_terminal(&loop_, user_request()).await;
    assert!(matches!(
        terminal,
        RunTerminal::TimedOut {
            kind: RunTimeoutKind::Total,
            ..
        }
    ));
    assert_eq!(executions.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn cancellation_during_approval_policy_is_typed_and_never_dispatches() {
    let executions = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&executions);
    let binding = ToolBinding::from_fn(
        tool_spec("charge"),
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
    let model = ScriptedModel::new([terminal_step(tool_response(vec![local_call(
        "charge_1",
        "charge",
        json!({ "amount": 10 }),
    )]))]);
    let cancellation = Cancellation::new();
    let loop_ = ToolLoop::new(model, tools)
        .with_approval_decider(PendingApprovalDecider::new(Some(cancellation.clone())));
    let options = CallOptions::default().with_cancellation(cancellation);

    let (_, terminal) = collect_terminal_with_options(&loop_, user_request(), options).await;
    assert!(matches!(
        terminal,
        RunTerminal::Cancelled { ref reason, .. }
            if reason == "tool loop cancelled during approval decision"
    ));
    assert_eq!(executions.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn provider_owned_name_collision_suspends_without_local_lookup_or_execution() {
    let executions = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&executions);
    let binding = executable_binding("search", move |_| {
        let observed = Arc::clone(&observed);
        boxed_tool_future(async move {
            observed.fetch_add(1, Ordering::SeqCst);
            Ok(ToolOutcome::Success { value: Value::Null })
        })
    });
    let tools = ToolSet::from_bindings([binding]).expect("unique tool");
    let model = ScriptedModel::new([terminal_step(tool_response(vec![provider_call(
        "provider_1",
        "search",
    )]))]);
    let loop_ = ToolLoop::new(model, tools);

    let (_, terminal) = collect_terminal(&loop_, user_request()).await;
    assert!(matches!(
        terminal,
        RunTerminal::Suspended {
            reason: SuspensionReason::AwaitingProvider { .. },
            ..
        }
    ));
    assert_eq!(executions.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn unresolved_provider_work_preempts_all_local_resolution() {
    let model = ScriptedModel::new([terminal_step(tool_response(vec![
        local_call("local_1", "missing-local-binding", json!({})),
        provider_call("provider_1", "remote-search"),
    ]))]);
    let loop_ = ToolLoop::new(model, ToolSet::default());

    let (_, terminal) = collect_terminal(&loop_, user_request()).await;
    assert!(matches!(
        terminal,
        RunTerminal::Suspended {
            reason: SuspensionReason::AwaitingProvider { ref state_ids },
            ..
        } if state_ids.len() == 1 && state_ids[0] == "provider_1"
    ));
}

#[tokio::test]
async fn dropping_after_preparation_prevents_all_later_dispatch() {
    let executions = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&executions);
    let binding = executable_binding("lookup", move |_| {
        let observed = Arc::clone(&observed);
        boxed_tool_future(async move {
            observed.fetch_add(1, Ordering::SeqCst);
            Ok(ToolOutcome::Success { value: Value::Null })
        })
    });
    let tools = ToolSet::from_bindings([binding]).expect("unique tool");
    let model = ScriptedModel::new([terminal_step(tool_response(vec![
        local_call("call_1", "lookup", json!({})),
        local_call("call_2", "lookup", json!({})),
        local_call("call_3", "lookup", json!({})),
    ]))]);
    let loop_ = ToolLoop::new(model, tools);
    let mut stream = loop_
        .stream(user_request(), CallOptions::default())
        .await
        .expect("first stream establishes");

    while let Some(event) = stream.next().await {
        if matches!(event, RunEvent::ToolPrepared { .. }) {
            break;
        }
    }
    drop(stream);
    tokio::task::yield_now().await;
    assert_eq!(executions.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn total_timeout_after_preparation_prevents_dispatch() {
    let executions = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&executions);
    let binding = executable_binding("lookup", move |_| {
        let observed = Arc::clone(&observed);
        boxed_tool_future(async move {
            observed.fetch_add(1, Ordering::SeqCst);
            Ok(ToolOutcome::Success { value: Value::Null })
        })
    });
    let tools = ToolSet::from_bindings([binding]).expect("unique tool");
    let model = ScriptedModel::new([terminal_step(tool_response(vec![local_call(
        "call_1",
        "lookup",
        json!({}),
    )]))]);
    let runtime = runtime_with_timeouts(
        Duration::from_millis(25),
        Duration::from_secs(2),
        Duration::from_secs(2),
        Duration::from_secs(2),
        Duration::from_secs(2),
    );
    let loop_ = ToolLoop::new(model, tools).with_runtime(runtime);
    let mut stream = loop_
        .stream(user_request(), CallOptions::default())
        .await
        .expect("first stream establishes");

    while let Some(event) = stream.next().await {
        if matches!(event, RunEvent::ToolPrepared { .. }) {
            break;
        }
    }
    tokio::time::sleep(Duration::from_millis(50)).await;

    let terminal = loop {
        match stream.next().await {
            Some(RunEvent::Terminal(terminal)) => break terminal,
            Some(_) => {}
            None => panic!("run stream must emit one terminal"),
        }
    };
    assert!(matches!(
        terminal,
        RunTerminal::TimedOut {
            kind: RunTimeoutKind::Total,
            ..
        }
    ));
    assert_eq!(executions.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn non_success_outcomes_stop_by_default_and_continue_only_by_explicit_policy() {
    fn scripts() -> Vec<ScriptStep> {
        vec![
            terminal_step(tool_response(vec![local_call(
                "deny_1",
                "guarded",
                json!({}),
            )])),
            terminal_step(final_response("continued")),
        ]
    }
    let tools = ToolSet::from_bindings([executable_binding("guarded", |_| {
        boxed_tool_future(async {
            Ok(ToolOutcome::Denied {
                reason: "host denied execution".to_string(),
            })
        })
    })])
    .expect("unique tool");

    let stopped_model = ScriptedModel::new(scripts());
    let stopped_loop = ToolLoop::new(stopped_model.clone(), tools.clone());
    let (_, stopped) = collect_terminal(&stopped_loop, user_request()).await;
    assert!(matches!(stopped, RunTerminal::Stopped { .. }));
    assert_eq!(stopped_model.stream_calls.load(Ordering::SeqCst), 1);

    let continued_model = ScriptedModel::new(scripts());
    let continued_loop = ToolLoop::new(continued_model.clone(), tools)
        .with_outcome_policy(ToolOutcomePolicy::default().with_denied(ToolOutcomeAction::Continue));
    let (_, continued) = collect_terminal(&continued_loop, user_request()).await;
    assert!(matches!(continued, RunTerminal::Completed { .. }));
    assert_eq!(continued_model.stream_calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn indeterminate_side_effect_becomes_an_indeterminate_run_terminal() {
    let binding = executable_binding("charge", |request| {
        boxed_tool_future(async move {
            Err(ToolExecutionError::executor_failed(
                request.name(),
                "connection lost after dispatch",
                true,
                EffectCertainty::Indeterminate,
            ))
        })
    });
    let tools = ToolSet::from_bindings([binding]).expect("unique tool");
    let model = ScriptedModel::new([terminal_step(tool_response(vec![local_call(
        "charge_1",
        "charge",
        json!({ "amount": 10 }),
    )]))]);
    let loop_ = ToolLoop::new(model, tools);

    let (_, terminal) = collect_terminal(&loop_, user_request()).await;
    let report = terminal
        .report()
        .expect("indeterminate terminal has report");
    assert!(matches!(&terminal, RunTerminal::Indeterminate { .. }));
    assert_eq!(
        report.execution_log().status("charge_1"),
        Some(ToolExecutionStatus::Indeterminate)
    );
}

#[tokio::test]
async fn client_tool_collision_fails_before_the_first_model_call() {
    let tools = ToolSet::from_bindings([executable_binding("admin", |_| {
        boxed_tool_future(async { Ok(ToolOutcome::Success { value: Value::Null }) })
    })])
    .expect("unique tool");
    let model = ScriptedModel::new([terminal_step(final_response("unused"))]);
    let loop_ = ToolLoop::new(model.clone(), tools);
    let mut request = user_request();
    request.tools.push(tool_spec("admin"));

    let error = loop_
        .stream(request, CallOptions::default())
        .await
        .expect_err("client collision must fail");
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert_eq!(model.stream_calls.load(Ordering::SeqCst), 0);
}

fn runtime_with_timeouts(
    total: Duration,
    model_step: Duration,
    first_chunk: Duration,
    inter_chunk: Duration,
    tool: Duration,
) -> Runtime {
    let timeouts = RunTimeouts::new(total, model_step, first_chunk, inter_chunk, tool)
        .expect("non-zero timeouts");
    let budget = RunBudget::builder()
        .timeouts(timeouts)
        .build()
        .expect("valid budget");
    Runtime::builder().with_run_budget(budget).build()
}

async fn assert_model_timeout(expected: RunTimeoutKind, runtime: Runtime, step: ScriptStep) {
    let model = ScriptedModel::new([step]);
    let loop_ = ToolLoop::new(model, ToolSet::default()).with_runtime(runtime);
    let (_, terminal) = collect_terminal(&loop_, user_request()).await;
    assert!(matches!(
        terminal,
        RunTerminal::TimedOut { kind, .. } if kind == expected
    ));
}

#[tokio::test]
async fn total_model_first_and_inter_chunk_timeouts_are_typed_terminals() {
    let long = Duration::from_secs(2);
    let short = Duration::from_millis(25);
    let delayed = Duration::from_millis(100);

    assert_model_timeout(
        RunTimeoutKind::Total,
        runtime_with_timeouts(short, long, long, long, long),
        ScriptStep::delayed(vec![
            (
                Duration::ZERO,
                LanguageStreamEvent::Started {
                    id: None,
                    model: None,
                },
            ),
            (
                delayed,
                LanguageStreamEvent::Terminal(StreamTerminal::Completed {
                    response: Box::new(final_response("late")),
                }),
            ),
        ]),
    )
    .await;

    assert_model_timeout(
        RunTimeoutKind::ModelStep,
        runtime_with_timeouts(long, short, long, long, long),
        ScriptStep::delayed(vec![
            (
                Duration::ZERO,
                LanguageStreamEvent::Started {
                    id: None,
                    model: None,
                },
            ),
            (
                delayed,
                LanguageStreamEvent::Terminal(StreamTerminal::Completed {
                    response: Box::new(final_response("late")),
                }),
            ),
        ]),
    )
    .await;

    assert_model_timeout(
        RunTimeoutKind::FirstChunk,
        runtime_with_timeouts(long, long, short, long, long),
        ScriptStep::delayed(vec![(
            delayed,
            LanguageStreamEvent::Terminal(StreamTerminal::Completed {
                response: Box::new(final_response("late")),
            }),
        )]),
    )
    .await;

    assert_model_timeout(
        RunTimeoutKind::InterChunk,
        runtime_with_timeouts(long, long, long, short, long),
        ScriptStep::delayed(vec![
            (
                Duration::ZERO,
                LanguageStreamEvent::Started {
                    id: None,
                    model: None,
                },
            ),
            (
                delayed,
                LanguageStreamEvent::Terminal(StreamTerminal::Completed {
                    response: Box::new(final_response("late")),
                }),
            ),
        ]),
    )
    .await;
}

#[tokio::test]
async fn tool_timeout_is_a_typed_terminal_for_read_only_execution() {
    let binding = executable_binding("slow", |_| {
        boxed_tool_future(async {
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok(ToolOutcome::Success { value: Value::Null })
        })
    })
    .with_effect(ToolEffect::ReadOnly);
    let tools = ToolSet::from_bindings([binding]).expect("unique tool");
    let model = ScriptedModel::new([terminal_step(tool_response(vec![local_call(
        "slow_1",
        "slow",
        json!({}),
    )]))]);
    let runtime = runtime_with_timeouts(
        Duration::from_secs(2),
        Duration::from_secs(2),
        Duration::from_secs(2),
        Duration::from_secs(2),
        Duration::from_millis(25),
    );
    let loop_ = ToolLoop::new(model, tools).with_runtime(runtime);

    let (_, terminal) = collect_terminal(&loop_, user_request()).await;
    assert!(matches!(
        terminal,
        RunTerminal::TimedOut {
            kind: RunTimeoutKind::Tool,
            ..
        }
    ));
}

#[tokio::test]
async fn first_handshake_failure_is_outer_but_later_handshake_failure_is_terminal() {
    let first_model = ScriptedModel::new([ScriptStep::handshake_error()]);
    let first_loop = ToolLoop::new(first_model, ToolSet::default());
    let error = first_loop
        .stream(user_request(), CallOptions::default())
        .await
        .expect_err("first handshake failure stays outer");
    assert_eq!(error.kind(), ErrorKind::Transport);

    let tools = ToolSet::from_bindings([executable_binding("lookup", |_| {
        boxed_tool_future(async { Ok(ToolOutcome::Success { value: Value::Null }) })
    })])
    .expect("unique tool");
    let later_model = ScriptedModel::new([
        terminal_step(tool_response(vec![local_call(
            "call_1",
            "lookup",
            json!({}),
        )])),
        ScriptStep::handshake_error(),
    ]);
    let later_loop = ToolLoop::new(later_model, tools);
    let (_, terminal) = collect_terminal(&later_loop, user_request()).await;
    assert!(matches!(terminal, RunTerminal::Failed { .. }));
}
