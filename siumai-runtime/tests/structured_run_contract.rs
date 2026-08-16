use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use siumai_core::stream::established_stream;
use siumai_core::{
    CallOptions, Cancellation, ContentPart, Error, ErrorKind, LanguageCallError,
    LanguageCompletionReason, LanguageIncompleteReason, LanguageModel, LanguageRequest,
    LanguageResponse, LanguageStream, LanguageStreamEvent, Message, MessageRole, Model,
    ModelDescriptor, ModelFamily, ModelId, PartialLanguageOutput, PartialLanguageOutputPart,
    ProviderId, StreamTerminal, ToolCall, Usage,
};
use siumai_runtime::{
    OutputDescriptor, OutputSchemaValidator, RepairPolicy, RunBudget, RunTerminal, Runtime,
    SchemaValidationError, StructuredOutputAttemptKind, StructuredOutputFailureKind,
    StructuredOutputRunError, StructuredOutputRunner,
};

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct Person {
    name: String,
    age: u8,
}

enum ScriptStep {
    Completed(Box<LanguageResponse>),
    Failed(PartialLanguageOutput),
    HandshakeError,
    Pending,
}

#[derive(Clone)]
struct ObservedCall {
    request: LanguageRequest,
    deadline: Option<Instant>,
    cancelled_on_entry: bool,
}

struct ScriptedModel {
    descriptor: ModelDescriptor,
    scripts: Mutex<VecDeque<ScriptStep>>,
    calls: Mutex<Vec<ObservedCall>>,
    generate_calls: AtomicUsize,
    stream_calls: AtomicUsize,
}

impl ScriptedModel {
    fn new(scripts: impl IntoIterator<Item = ScriptStep>) -> Arc<Self> {
        Arc::new(Self {
            descriptor: ModelDescriptor::new(
                ProviderId::new("scripted").expect("valid provider"),
                ModelId::new("structured-model").expect("valid model"),
                ModelFamily::Language,
            ),
            scripts: Mutex::new(scripts.into_iter().collect()),
            calls: Mutex::new(Vec::new()),
            generate_calls: AtomicUsize::new(0),
            stream_calls: AtomicUsize::new(0),
        })
    }

    fn calls(&self) -> Vec<ObservedCall> {
        self.calls.lock().expect("call lock").clone()
    }

    fn generate_call_count(&self) -> usize {
        self.generate_calls.load(Ordering::SeqCst)
    }

    fn stream_call_count(&self) -> usize {
        self.stream_calls.load(Ordering::SeqCst)
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
            "structured-output execution must not call generate",
        )
        .into())
    }

    async fn stream(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        self.stream_calls.fetch_add(1, Ordering::SeqCst);
        self.calls.lock().expect("call lock").push(ObservedCall {
            request,
            deadline: options.deadline(),
            cancelled_on_entry: options.cancellation().is_cancelled(),
        });

        let step = self
            .scripts
            .lock()
            .expect("script lock")
            .pop_front()
            .ok_or_else(|| Error::new(ErrorKind::Internal, "missing scripted model step"))?;
        match step {
            ScriptStep::Completed(response) => {
                let cancellation = options.cancellation().clone();
                Ok(established_stream(cancellation, move |_| {
                    async_stream::stream! {
                        yield Ok(LanguageStreamEvent::Terminal(StreamTerminal::Completed {
                            response,
                        }));
                    }
                }))
            }
            ScriptStep::Failed(partial) => {
                let cancellation = options.cancellation().clone();
                Ok(established_stream(cancellation, move |_| {
                    async_stream::stream! {
                        yield Ok(LanguageStreamEvent::Terminal(StreamTerminal::Failed {
                            error: Error::new(ErrorKind::Provider, "scripted provider failure"),
                            partial: Some(partial),
                        }));
                    }
                }))
            }
            ScriptStep::HandshakeError => Err(Error::new(
                ErrorKind::Transport,
                "scripted repair handshake failed",
            )),
            ScriptStep::Pending => {
                let cancellation = options.cancellation().clone();
                Ok(established_stream(cancellation, move |_| {
                    futures::stream::pending::<Result<LanguageStreamEvent, Error>>()
                }))
            }
        }
    }
}

fn person_schema() -> Value {
    json!({
        "type": "object",
        "required": ["name", "age"],
        "additionalProperties": false,
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0, "maximum": 255}
        }
    })
}

fn person_validator() -> impl OutputSchemaValidator {
    |_schema: &Value, instance: &Value| {
        let object = instance
            .as_object()
            .ok_or_else(|| SchemaValidationError::new("expected an object"))?;
        if !object.get("name").is_some_and(Value::is_string) {
            return Err(SchemaValidationError::new("name must be a string"));
        }
        if !object
            .get("age")
            .and_then(Value::as_u64)
            .is_some_and(|age| age <= u8::MAX as u64)
        {
            return Err(SchemaValidationError::new("age must be an unsigned byte"));
        }
        Ok(())
    }
}

fn descriptor() -> OutputDescriptor<Person> {
    OutputDescriptor::new("person", person_schema(), person_validator())
        .expect("valid output descriptor")
}

fn request() -> LanguageRequest {
    LanguageRequest::new(vec![Message::text(MessageRole::User, "Describe Ada")])
}

fn usage(tokens: u64) -> Usage {
    Usage::default().with_total_tokens(tokens)
}

fn text_response(text: impl Into<String>, tokens: u64) -> LanguageResponse {
    LanguageResponse::completed(
        vec![ContentPart::Text { text: text.into() }],
        LanguageCompletionReason::Stop,
        usage(tokens),
    )
    .expect("valid text response")
}

fn completed_step(response: LanguageResponse) -> ScriptStep {
    ScriptStep::Completed(Box::new(response))
}

fn failed_step(partial: PartialLanguageOutput) -> ScriptStep {
    ScriptStep::Failed(partial)
}

fn refusal_response(tokens: u64) -> LanguageResponse {
    LanguageResponse::completed(
        vec![
            ContentPart::Refusal {
                reason: Some("unsafe request".to_string()),
            },
            ContentPart::Text {
                text: r#"{"name":"Ada","age":36}"#.to_string(),
            },
        ],
        LanguageCompletionReason::Refusal,
        usage(tokens),
    )
    .expect("valid refusal response")
}

fn content_filter_response(tokens: u64) -> LanguageResponse {
    LanguageResponse::incomplete(
        Vec::new(),
        LanguageIncompleteReason::ContentFilter,
        usage(tokens),
    )
    .expect("valid content-filter response")
}

fn provider_failure_partial(tokens: u64) -> PartialLanguageOutput {
    PartialLanguageOutput::new(
        vec![PartialLanguageOutputPart::Text {
            text: "partial provider output".to_string(),
        }],
        usage(tokens),
    )
    .expect("valid provider-failure partial")
}

fn unexpected_tool_response(tokens: u64) -> LanguageResponse {
    LanguageResponse::completed(
        vec![ContentPart::ToolCall(
            ToolCall::local("call-1", "lookup", json!({"query": "Ada"})).expect("valid tool call"),
        )],
        LanguageCompletionReason::ToolCalls,
        usage(tokens),
    )
    .expect("valid unexpected-tool response")
}

#[tokio::test]
async fn valid_strict_output_uses_one_stream_step_and_never_generate() {
    let model = ScriptedModel::new([completed_step(text_response(
        r#"{"name":"Ada","age":36}"#,
        4,
    ))]);
    let runner = StructuredOutputRunner::new(model.clone(), descriptor());

    let result = runner
        .generate(request(), CallOptions::default())
        .await
        .expect("valid structured output");

    assert_eq!(
        result.output().value(),
        &Person {
            name: "Ada".to_string(),
            age: 36,
        }
    );
    assert_eq!(
        result.output().attempt(),
        StructuredOutputAttemptKind::Initial
    );
    assert!(!result.was_repaired());
    assert!(result.initial_failure().is_none());
    assert_eq!(result.report().steps().len(), 1);
    assert_eq!(result.report().steps()[0].index(), 0);
    assert_eq!(result.report().budget().model_steps(), 1);
    assert_eq!(result.report().usage().total_tokens.value(), Some(4));
    assert_eq!(model.stream_call_count(), 1);
    assert_eq!(model.generate_call_count(), 0);

    let calls = model.calls();
    assert_eq!(calls.len(), 1);
    assert!(calls[0].request.structured_output.is_some());
}

#[tokio::test]
async fn repair_is_disabled_by_default_at_the_execution_boundary() {
    let model = ScriptedModel::new([completed_step(text_response("not-json", 2))]);
    let runner = StructuredOutputRunner::new(model.clone(), descriptor());

    let error = runner
        .generate(request(), CallOptions::default())
        .await
        .expect_err("default policy must not repair invalid JSON");

    let output_error = error.output_error().expect("validation error");
    assert_eq!(
        output_error.kind(),
        StructuredOutputFailureKind::InvalidJson
    );
    assert_eq!(output_error.attempt(), StructuredOutputAttemptKind::Initial);
    let report = error.report().expect("initial report is retained");
    assert_eq!(report.steps().len(), 1);
    assert_eq!(report.budget().model_steps(), 1);
    assert_eq!(report.usage().total_tokens.value(), Some(2));
    assert_eq!(model.stream_call_count(), 1);
    assert_eq!(model.generate_call_count(), 0);
}

#[tokio::test]
async fn structured_repair_reuses_shared_completed_step_history_and_settlement_contract() {
    let initial_response = text_response("not-json", 3);
    let repaired_response = text_response(r#"{"name":"Ada","age":36}"#, 5);
    let model = ScriptedModel::new([
        completed_step(initial_response.clone()),
        completed_step(repaired_response.clone()),
    ]);
    let runner = StructuredOutputRunner::new(
        model.clone(),
        descriptor().with_repair_policy(RepairPolicy::OneAttempt),
    );

    let result = runner
        .generate(request(), CallOptions::default())
        .await
        .expect("one repair succeeds");

    assert!(result.was_repaired());
    assert_eq!(
        result.output().attempt(),
        StructuredOutputAttemptKind::Repair
    );
    let initial_failure = result.initial_failure().expect("initial failure retained");
    assert_eq!(
        initial_failure.kind(),
        StructuredOutputFailureKind::InvalidJson
    );
    assert_eq!(
        initial_failure.attempt(),
        StructuredOutputAttemptKind::Initial
    );
    assert_eq!(
        result
            .report()
            .steps()
            .iter()
            .map(|step| step.index())
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(result.report().budget().model_steps(), 2);
    assert_eq!(result.report().budget().known_tokens(), 8);
    assert_eq!(result.report().usage().total_tokens.value(), Some(8));
    assert_eq!(result.report().steps()[0].response(), &initial_response);
    assert_eq!(result.report().steps()[1].response(), &repaired_response);
    assert!(result.report().steps()[0].tool_results().is_empty());
    assert!(result.report().steps()[1].tool_results().is_empty());
    assert_eq!(result.report().final_response(), Some(&repaired_response));
    assert_eq!(model.stream_call_count(), 2);
    assert_eq!(model.generate_call_count(), 0);

    let calls = model.calls();
    assert_eq!(calls.len(), 2);
    assert!(calls[1].request.tools.is_empty());
    assert!(calls[1].request.tool_choice.is_none());
    assert!(calls[1].request.structured_output.is_some());
    let repaired_history = repaired_response
        .project_assistant_history()
        .into_message()
        .expect("repair response projects to assistant history");
    assert_eq!(result.report().messages().last(), Some(&repaired_history));
    assert_eq!(
        calls[1]
            .request
            .messages
            .iter()
            .map(Message::role)
            .collect::<Vec<_>>(),
        vec![
            MessageRole::User,
            MessageRole::Assistant,
            MessageRole::Developer,
        ]
    );
}

#[tokio::test]
async fn model_step_budget_prevents_repair_before_a_second_stream_call() {
    let budget = RunBudget::builder()
        .max_model_steps(1)
        .build()
        .expect("valid one-step budget");
    let runtime = Runtime::builder().with_run_budget(budget).build();
    let model = ScriptedModel::new([completed_step(text_response("not-json", 3))]);
    let runner = StructuredOutputRunner::new(
        model.clone(),
        descriptor().with_repair_policy(RepairPolicy::OneAttempt),
    )
    .with_runtime(runtime);

    let error = runner
        .generate(request(), CallOptions::default())
        .await
        .expect_err("repair must remain inside the original model-step budget");

    assert!(matches!(
        &error,
        StructuredOutputRunError::Runtime(terminal)
            if matches!(terminal.as_ref(), RunTerminal::BudgetExceeded { .. })
    ));
    let report = error.report().expect("budget terminal retains the report");
    assert_eq!(report.steps().len(), 1);
    assert_eq!(report.steps()[0].index(), 0);
    assert_eq!(report.budget().model_steps(), 1);
    assert_eq!(report.usage().total_tokens.value(), Some(3));
    assert_eq!(model.stream_call_count(), 1);
    assert_eq!(model.generate_call_count(), 0);
}

#[tokio::test]
async fn refusal_and_content_filter_never_repair() {
    let cases = [
        (
            completed_step(refusal_response(2)),
            StructuredOutputFailureKind::Refusal,
        ),
        (
            completed_step(content_filter_response(3)),
            StructuredOutputFailureKind::ContentFilter,
        ),
    ];

    for (step, expected_kind) in cases {
        let model = ScriptedModel::new([step]);
        let runner = StructuredOutputRunner::new(
            model.clone(),
            descriptor().with_repair_policy(RepairPolicy::OneAttempt),
        );

        let error = runner
            .generate(request(), CallOptions::default())
            .await
            .expect_err("non-repairable provider outcome must stop after one step");

        let output_error = error.output_error().expect("typed output failure");
        assert_eq!(output_error.kind(), expected_kind);
        assert_eq!(output_error.attempt(), StructuredOutputAttemptKind::Initial);
        assert!(!output_error.is_repair_eligible());
        assert_eq!(error.report().expect("report retained").steps().len(), 1);
        assert_eq!(model.stream_call_count(), 1);
        assert_eq!(model.generate_call_count(), 0);
    }
}

#[tokio::test]
async fn provider_failure_partial_is_a_runtime_terminal_and_never_repairs() {
    let model = ScriptedModel::new([failed_step(provider_failure_partial(4))]);
    let runner = StructuredOutputRunner::new(
        model.clone(),
        descriptor().with_repair_policy(RepairPolicy::OneAttempt),
    );

    let error = runner
        .generate(request(), CallOptions::default())
        .await
        .expect_err("provider failure must stop after one step");

    assert!(matches!(
        &error,
        StructuredOutputRunError::Runtime(terminal)
            if matches!(
                terminal.as_ref(),
                RunTerminal::Failed { partial: Some(partial), .. }
                    if matches!(
                        partial.content(),
                        [PartialLanguageOutputPart::Text { text }]
                            if text == "partial provider output"
                    )
            )
    ));
    let report = error.report().expect("runtime report retained");
    assert!(report.steps().is_empty());
    assert_eq!(report.usage().total_tokens.value(), Some(4));
    assert_eq!(model.stream_call_count(), 1);
    assert_eq!(model.generate_call_count(), 0);
}

#[tokio::test]
async fn unexpected_tool_call_is_observed_but_never_executed_or_repaired() {
    let model = ScriptedModel::new([completed_step(unexpected_tool_response(2))]);
    let runner = StructuredOutputRunner::new(
        model.clone(),
        descriptor().with_repair_policy(RepairPolicy::OneAttempt),
    );

    let error = runner
        .generate(request(), CallOptions::default())
        .await
        .expect_err("structured output must reject unexpected tool calls");

    let output_error = error.output_error().expect("typed output failure");
    assert_eq!(
        output_error.kind(),
        StructuredOutputFailureKind::UnexpectedToolCall
    );
    assert_eq!(output_error.attempt(), StructuredOutputAttemptKind::Initial);
    assert!(!output_error.is_repair_eligible());
    let report = error.report().expect("response report retained");
    assert_eq!(report.steps().len(), 1);
    assert_eq!(report.budget().model_steps(), 1);
    assert_eq!(model.stream_call_count(), 1);
    assert_eq!(model.generate_call_count(), 0);
}

#[tokio::test]
async fn repair_handshake_failure_is_a_repair_error_with_the_initial_report() {
    let model = ScriptedModel::new([
        completed_step(text_response("not-json", 3)),
        ScriptStep::HandshakeError,
    ]);
    let runner = StructuredOutputRunner::new(
        model.clone(),
        descriptor().with_repair_policy(RepairPolicy::OneAttempt),
    );

    let error = runner
        .generate(request(), CallOptions::default())
        .await
        .expect_err("repair handshake failure must be observable");

    let output_error = error.output_error().expect("repair transport failure");
    assert_eq!(
        output_error.kind(),
        StructuredOutputFailureKind::TransportFailure
    );
    assert_eq!(output_error.attempt(), StructuredOutputAttemptKind::Repair);
    assert!(!output_error.is_repair_eligible());

    let report = error.report().expect("initial report is retained");
    assert_eq!(report.steps().len(), 1);
    assert_eq!(report.steps()[0].index(), 0);
    assert_eq!(report.budget().model_steps(), 2);
    assert_eq!(report.budget().known_tokens(), 3);
    assert_eq!(report.usage().total_tokens.value(), Some(3));
    match report
        .final_response()
        .expect("initial response is retained")
        .content()
    {
        [ContentPart::Text { text }] => assert_eq!(text, "not-json"),
        other => panic!("expected retained invalid text response, got {other:?}"),
    }
    assert_eq!(model.stream_call_count(), 2);
    assert_eq!(model.generate_call_count(), 0);
}

#[tokio::test]
async fn repair_inherits_the_caller_deadline() {
    let model = ScriptedModel::new([
        completed_step(text_response("not-json", 1)),
        completed_step(text_response(r#"{"name":"Ada","age":36}"#, 1)),
    ]);
    let runner = StructuredOutputRunner::new(
        model.clone(),
        descriptor().with_repair_policy(RepairPolicy::OneAttempt),
    );
    let deadline = Instant::now() + Duration::from_secs(30);
    let cancellation = Cancellation::new();
    let options = CallOptions::default()
        .with_deadline(deadline)
        .with_cancellation(cancellation.clone());

    runner
        .generate(request(), options)
        .await
        .expect("repair succeeds");

    let calls = model.calls();
    assert_eq!(calls.len(), 2);
    assert!(calls.iter().all(|call| call.deadline == Some(deadline)));
    assert!(calls.iter().all(|call| !call.cancelled_on_entry));
}

#[tokio::test]
async fn caller_cancellation_stops_an_active_repair_step() {
    let model = ScriptedModel::new([
        completed_step(text_response("not-json", 1)),
        ScriptStep::Pending,
    ]);
    let runner = StructuredOutputRunner::new(
        model.clone(),
        descriptor().with_repair_policy(RepairPolicy::OneAttempt),
    );
    let cancellation = Cancellation::new();
    let cancel_signal = cancellation.clone();
    let observed_model = model.clone();
    let cancel_task = tokio::spawn(async move {
        while observed_model.stream_call_count() < 2 {
            tokio::task::yield_now().await;
        }
        cancel_signal.cancel();
    });

    let error = tokio::time::timeout(
        Duration::from_secs(1),
        runner.generate(
            request(),
            CallOptions::default().with_cancellation(cancellation),
        ),
    )
    .await
    .expect("caller cancellation must stop the active repair")
    .expect_err("repair must terminate as cancelled");
    cancel_task.await.expect("canceller task completes");

    assert!(matches!(
        error,
        StructuredOutputRunError::Runtime(terminal)
            if matches!(terminal.as_ref(), RunTerminal::Cancelled { .. })
    ));
    assert_eq!(model.stream_call_count(), 2);
}
