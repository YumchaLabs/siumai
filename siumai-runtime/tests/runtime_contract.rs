use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use futures::StreamExt;
use serde::Serialize;
use serde_json::json;
use siumai_core::stream::established_stream;
use siumai_core::{
    CallOptions, ContentPart, Error, ExecutionOwner, FinishReason, LanguageModel, LanguageRequest,
    LanguageResponse, LanguageStream, LanguageStreamEvent, Message, MessageRole, Model,
    ModelDescriptor, ModelFamily, ModelId, ProviderId, ProviderOptionError, ProviderOptionLayers,
    ProviderOptionMerger, ProviderOptionOrigin, ProviderOptions, RouteId, StreamTerminal, ToolCall,
    TypedProviderOptions, Usage,
};
use siumai_runtime::{ModelTarget, Runtime, StepOptions, generate, stream};

#[derive(Debug, Serialize)]
struct TestOptions {
    value: &'static str,
}

impl TypedProviderOptions for TestOptions {
    const NAMESPACE: &'static str = "test";
}

fn options(value: &'static str) -> ProviderOptions {
    ProviderOptions::typed(&TestOptions { value }).unwrap()
}

struct CaptureMerger;

impl ProviderOptionMerger for CaptureMerger {
    type Output = Vec<(ProviderOptionOrigin, String)>;

    fn validate_layer(
        &self,
        _origin: ProviderOptionOrigin,
        _options: &ProviderOptions,
    ) -> Result<(), ProviderOptionError> {
        Ok(())
    }

    fn merge(&self, layers: &ProviderOptionLayers) -> Result<Self::Output, ProviderOptionError> {
        Ok(layers
            .in_precedence_order()
            .map(|(origin, options)| {
                (
                    origin,
                    options.value()["value"].as_str().unwrap().to_string(),
                )
            })
            .collect())
    }
}

#[derive(Debug)]
struct ScriptedModel {
    descriptor: ModelDescriptor,
    route: Option<RouteId>,
    calls: Arc<AtomicUsize>,
    observed: Arc<std::sync::Mutex<Vec<(ProviderOptionOrigin, String)>>>,
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
        let layers = call
            .apply_provider_options(
                self.provider_id(),
                ProviderOptionLayers::default()
                    .with_provider_default(options("provider"))
                    .unwrap(),
            )
            .map_err(provider_options_error)?;
        *self.observed.lock().unwrap() = layers
            .merge_for(self.provider_id(), &CaptureMerger)
            .map_err(provider_options_error)?;
        LanguageResponse::completed(
            vec![ContentPart::ToolCall(ToolCall {
                id: "call-1".to_string(),
                name: "dangerous".to_string(),
                arguments: json!({"value": 1}),
                owner: ExecutionOwner::Local,
            })],
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

fn model() -> ScriptedModel {
    ScriptedModel {
        descriptor: ModelDescriptor::new(
            ProviderId::new("test").unwrap(),
            ModelId::new("model-v1").unwrap(),
            ModelFamily::Language,
        ),
        route: Some(RouteId::new("production").unwrap()),
        calls: Arc::new(AtomicUsize::new(0)),
        observed: Arc::new(std::sync::Mutex::new(Vec::new())),
    }
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
        .with_route_defaults(RouteId::new("production").unwrap(), options("route"))
        .unwrap()
        .with_model_defaults(ModelTarget::from_model(&model), options("model"))
        .unwrap()
        .build();
    let step = StepOptions::default()
        .with_provider_options(options("step"))
        .unwrap();
    let call = CallOptions::default().with_provider_options(options("call"));

    runtime
        .generate(&model, request(), step, call)
        .await
        .unwrap();

    assert_eq!(
        *observed.lock().unwrap(),
        vec![
            (
                ProviderOptionOrigin::ProviderDefault,
                "provider".to_string()
            ),
            (ProviderOptionOrigin::RouteDefault, "route".to_string()),
            (ProviderOptionOrigin::ModelDefault, "model".to_string()),
            (ProviderOptionOrigin::RuntimeStep, "step".to_string()),
            (ProviderOptionOrigin::Call, "call".to_string()),
        ]
    );
}
