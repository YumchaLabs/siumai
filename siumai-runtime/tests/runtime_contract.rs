use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use futures::StreamExt;
use serde::Serialize;
use serde_json::json;
use siumai_core::stream::established_stream;
use siumai_core::{
    ApiModeId, CallOptions, ContentPart, Error, FinishReason, LanguageModel, LanguageRequest,
    LanguageResponse, LanguageStream, LanguageStreamEvent, Message, MessageRole, Model,
    ModelDescriptor, ModelFamily, ModelId, ProviderId, ProviderOptionError, ProviderOptions,
    RouteId, StreamTerminal, ToolCall, TypedProviderOptions, Usage,
};
use siumai_runtime::{Runtime, RuntimeConfigError, StepOptions, generate, stream};

#[derive(Debug, Serialize)]
struct TestOptions {
    value: &'static str,
}

impl TypedProviderOptions for TestOptions {
    const NAMESPACE: &'static str = "test";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
}

#[derive(Debug, Serialize)]
struct OtherOptions {
    value: &'static str,
}

impl TypedProviderOptions for OtherOptions {
    const NAMESPACE: &'static str = "other";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
}

#[derive(Debug, Serialize)]
struct ChatOptions {
    value: &'static str,
}

impl TypedProviderOptions for ChatOptions {
    const NAMESPACE: &'static str = "test";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
    const API_MODE: Option<&'static str> = Some("chat");
}

#[derive(Debug)]
struct ScriptedModel {
    descriptor: ModelDescriptor,
    route: Option<RouteId>,
    calls: Arc<AtomicUsize>,
    observed: Arc<std::sync::Mutex<Vec<String>>>,
}

impl Model for ScriptedModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn route_id(&self) -> Option<&RouteId> {
        self.route.as_ref()
    }
}

#[async_trait]
impl LanguageModel for ScriptedModel {
    async fn generate(
        &self,
        _request: LanguageRequest,
        call: CallOptions,
    ) -> Result<LanguageResponse, Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let selection = call
            .provider_options_for(self)
            .map_err(provider_options_error)?;
        let mut observed = vec!["provider".to_string()];
        observed.extend(selection.typed().map(option_value));
        if let Some(raw) = selection.raw_override() {
            observed.push(option_value(raw));
        }
        *self.observed.lock().unwrap() = observed;
        LanguageResponse::completed(
            vec![ContentPart::ToolCall(
                ToolCall::local("call-1", "dangerous", json!({"value": 1}))
                    .expect("valid tool call"),
            )],
            FinishReason::ToolCalls,
            Usage::default(),
        )
        .map_err(|source| {
            Error::new(
                siumai_core::ErrorKind::Protocol,
                "invalid scripted response",
            )
            .with_source(source)
        })
    }

    async fn stream(
        &self,
        _request: LanguageRequest,
        _options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let response =
            LanguageResponse::completed(Vec::new(), FinishReason::Stop, Usage::default()).unwrap();
        Ok(established_stream(
            CallOptions::default().cancellation().clone(),
            |_| {
                futures::stream::iter(vec![Ok(LanguageStreamEvent::Terminal(
                    StreamTerminal::Completed {
                        response: Box::new(response),
                    },
                ))])
            },
        ))
    }
}

fn provider_options_error(source: ProviderOptionError) -> Error {
    Error::new(
        siumai_core::ErrorKind::Configuration,
        "invalid scripted provider options",
    )
    .with_source(source)
}

fn option_value(options: &ProviderOptions) -> String {
    options.value()["value"].as_str().unwrap().to_string()
}

fn model() -> ScriptedModel {
    model_for("test")
}

fn model_for(provider: &str) -> ScriptedModel {
    ScriptedModel {
        descriptor: ModelDescriptor::new(
            ProviderId::new(provider).unwrap(),
            ModelId::new("model-v1").unwrap(),
            ModelFamily::Language,
        ),
        route: Some(RouteId::new("production").unwrap()),
        calls: Arc::new(AtomicUsize::new(0)),
        observed: Arc::new(std::sync::Mutex::new(Vec::new())),
    }
}

fn model_for_mode(api_mode: &str) -> ScriptedModel {
    let mut model = model();
    model.descriptor = model
        .descriptor
        .clone()
        .with_api_mode(ApiModeId::new(api_mode).unwrap());
    model
}

fn request() -> LanguageRequest {
    LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")])
}

#[tokio::test]
async fn plain_generate_performs_one_call_and_never_executes_returned_tools() {
    let model = model();
    let calls = model.calls.clone();

    let response = generate(&model, request(), CallOptions::default())
        .await
        .unwrap();

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(matches!(response.content(), [ContentPart::ToolCall(_)]));
}

#[tokio::test]
async fn plain_stream_performs_one_streaming_call() {
    let model = model();
    let calls = model.calls.clone();
    let events = stream(&model, request(), CallOptions::default())
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(matches!(
        events.as_slice(),
        [LanguageStreamEvent::Terminal(
            StreamTerminal::Completed { .. }
        )]
    ));
}

#[tokio::test]
async fn runtime_composes_defaults_without_owning_provider_merge_semantics() {
    let model = model();
    let observed = model.observed.clone();
    let runtime = Runtime::builder()
        .with_route_defaults(&model, &TestOptions { value: "route" })
        .unwrap()
        .with_model_defaults(&model, &TestOptions { value: "model" })
        .unwrap()
        .build();
    let step = StepOptions::default()
        .with_provider_options(&model, &TestOptions { value: "step" })
        .unwrap();
    let call = CallOptions::default()
        .with_provider_options_for(&model, &TestOptions { value: "call" })
        .unwrap()
        .with_raw_provider_options_for(&model, json!({"value": "raw"}))
        .unwrap();

    runtime
        .generate(&model, request(), step, call)
        .await
        .unwrap();

    assert_eq!(
        *observed.lock().unwrap(),
        vec![
            "provider".to_string(),
            "route".to_string(),
            "model".to_string(),
            "step".to_string(),
            "call".to_string(),
            "raw".to_string(),
        ]
    );
}

#[tokio::test]
async fn step_options_select_only_the_exact_configured_target() {
    let model = model();
    let other_model = model_for("other");
    let observed = model.observed.clone();
    let step = StepOptions::default()
        .with_provider_options(&other_model, &OtherOptions { value: "foreign" })
        .unwrap()
        .with_provider_options(&model, &TestOptions { value: "selected" })
        .unwrap();

    Runtime::default()
        .generate(&model, request(), step, CallOptions::default())
        .await
        .unwrap();

    assert_eq!(
        *observed.lock().unwrap(),
        vec!["provider".to_string(), "selected".to_string()]
    );
}

#[tokio::test]
async fn same_label_instances_do_not_share_runtime_defaults_or_step_options() {
    let configured = model();
    let selected = model();
    let observed = selected.observed.clone();
    let runtime = Runtime::builder()
        .with_route_defaults(&configured, &TestOptions { value: "route" })
        .unwrap()
        .with_model_defaults(&configured, &TestOptions { value: "model" })
        .unwrap()
        .build();
    let step = StepOptions::default()
        .with_provider_options(&configured, &TestOptions { value: "step" })
        .unwrap();

    runtime
        .generate(&selected, request(), step, CallOptions::default())
        .await
        .unwrap();

    assert_eq!(*observed.lock().unwrap(), vec!["provider".to_string()]);
}

#[tokio::test]
async fn runtime_prefix_respects_the_call_option_entry_bound() {
    let model = model();
    let calls = model.calls.clone();
    let runtime = Runtime::builder()
        .with_model_defaults(&model, &TestOptions { value: "model" })
        .unwrap()
        .build();
    let mut call = CallOptions::default();
    for _ in 0..64 {
        call = call
            .with_provider_options_for(&model, &TestOptions { value: "call" })
            .unwrap();
    }

    let error = runtime
        .generate(&model, request(), StepOptions::default(), call)
        .await
        .unwrap_err();

    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(matches!(
        error
            .sensitive_source()
            .and_then(|source| source.expose().downcast_ref::<ProviderOptionError>()),
        Some(ProviderOptionError::TooManyEntries { maximum: 64 })
    ));
}

#[test]
fn model_defaults_reject_a_foreign_provider_namespace_at_build_time() {
    let model = model();
    let error = Runtime::builder()
        .with_model_defaults(&model, &OtherOptions { value: "foreign" })
        .unwrap_err();

    assert!(matches!(
        error,
        RuntimeConfigError::ProviderOptions(ProviderOptionError::NamespaceMismatch { .. })
    ));
}

#[test]
fn model_defaults_reject_a_foreign_api_mode_at_build_time() {
    let model = model_for_mode("responses");
    let error = Runtime::builder()
        .with_model_defaults(&model, &ChatOptions { value: "foreign" })
        .unwrap_err();

    assert!(matches!(
        error,
        RuntimeConfigError::ProviderOptions(ProviderOptionError::TargetMismatch { .. })
    ));
}
