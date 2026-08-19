use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use siumai::core::RouteId;
use siumai::prelude::*;
use siumai::{EmbeddingLimits, ModelId, ResponseMetadata};

use siumai::{ErrorKind, LanguageCallError, LanguageCompletionReason, LanguageResponse};

use siumai::{ImageArtifact, MediaData};

#[cfg(feature = "registry")]
use siumai::core::ProviderRegistration;

#[derive(Debug)]
struct FakeEmbedding {
    descriptor: ModelDescriptor,
    calls: Arc<AtomicUsize>,
}

#[derive(Debug)]
struct DeadlineEmbedding {
    descriptor: ModelDescriptor,
    calls: Arc<AtomicUsize>,
    observed: Arc<std::sync::Mutex<Vec<CallOptions>>>,
}

#[derive(Debug)]
struct FakeRerank {
    descriptor: ModelDescriptor,
    calls: Arc<AtomicUsize>,
    expect_provider_options: bool,
}

#[derive(Debug)]
struct FakeSpeech {
    descriptor: ModelDescriptor,
    calls: Arc<AtomicUsize>,
    expect_provider_options: bool,
}

#[derive(Debug)]
struct FakeTranscription {
    descriptor: ModelDescriptor,
    calls: Arc<AtomicUsize>,
    expect_provider_options: bool,
}

fn validate_expected_provider_options<M>(
    model: &M,
    options: &CallOptions,
    expected: bool,
) -> Result<(), Error>
where
    M: Model + ?Sized,
{
    if !expected {
        return Ok(());
    }
    let selection = options.provider_options_for(model).map_err(|source| {
        Error::new(
            ErrorKind::Configuration,
            "invalid non-language facade provider options",
        )
        .with_source(source)
    })?;
    if selection.typed().count() != 1 {
        return Err(Error::new(
            ErrorKind::Configuration,
            "expected one typed provider option at trait dispatch",
        ));
    }
    Ok(())
}

#[derive(Debug)]
struct FakeLanguage {
    descriptor: ModelDescriptor,
    calls: Arc<AtomicUsize>,
}

#[derive(Debug)]
struct RecordingLanguage {
    descriptor: ModelDescriptor,
    route: Option<RouteId>,
    calls: Arc<AtomicUsize>,
    stream_calls: Arc<AtomicUsize>,
    observed: Arc<std::sync::Mutex<Vec<RecordedLanguageCall>>>,
}

#[derive(Debug)]
struct RecordedLanguageCall {
    request: LanguageRequest,
    options: CallOptions,
    provider_values: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
enum StreamFixture {
    Completed,
    Failed,
    UnexpectedEof,
}

#[derive(Debug)]
struct LifecycleLanguage {
    descriptor: ModelDescriptor,
    stream_calls: Arc<AtomicUsize>,
    fixture: StreamFixture,
}

#[derive(Debug)]
struct FailingLanguage {
    descriptor: ModelDescriptor,
    route: Option<RouteId>,
    generate_calls: Arc<AtomicUsize>,
    stream_calls: Arc<AtomicUsize>,
}

#[cfg(all(feature = "registry", feature = "runtime"))]
#[derive(Debug)]
struct RegistryRuntimeLanguage {
    descriptor: ModelDescriptor,
    generate_calls: Arc<AtomicUsize>,
    stream_calls: Arc<AtomicUsize>,
    observed_options: Arc<std::sync::Mutex<Vec<String>>>,
}

impl Model for FakeLanguage {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

impl Model for RecordingLanguage {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn route_id(&self) -> Option<&RouteId> {
        self.route.as_ref()
    }
}

impl Model for LifecycleLanguage {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

impl Model for FailingLanguage {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn route_id(&self) -> Option<&RouteId> {
        self.route.as_ref()
    }
}

#[async_trait]
impl LanguageModel for FakeLanguage {
    async fn generate(
        &self,
        _request: LanguageRequest,
        _options: CallOptions,
    ) -> Result<LanguageResponse, LanguageCallError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        LanguageResponse::completed(
            vec![ContentPart::Text {
                text: "facade runtime".to_string(),
            }],
            LanguageCompletionReason::Stop,
            Usage::default(),
        )
        .map_err(|source| {
            Error::new(ErrorKind::Protocol, "invalid facade test response")
                .with_source(source)
                .into()
        })
    }

    async fn stream(
        &self,
        _request: LanguageRequest,
        _options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        Err(Error::new(
            ErrorKind::Unsupported,
            "streaming is not part of this facade contract fixture",
        ))
    }
}

#[async_trait]
impl LanguageModel for RecordingLanguage {
    async fn generate(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageResponse, LanguageCallError> {
        let selection = options.provider_options_for(self).map_err(|source| {
            Error::new(
                ErrorKind::Configuration,
                "invalid recording language provider options",
            )
            .with_source(source)
        })?;
        let mut provider_values = selection
            .typed()
            .map(|options| {
                options
                    .value()
                    .get("value")
                    .or_else(|| options.value().get("instructions"))
                    .and_then(serde_json::Value::as_str)
                    .map_or_else(|| options.namespace().to_string(), ToString::to_string)
            })
            .collect::<Vec<_>>();
        if let Some(value) = selection
            .raw_override()
            .and_then(|options| options.value().get("value"))
            .and_then(serde_json::Value::as_str)
        {
            provider_values.push(value.to_string());
        }
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.observed
            .lock()
            .expect("recording language observation lock")
            .push(RecordedLanguageCall {
                request,
                options,
                provider_values,
            });
        LanguageResponse::completed(
            vec![ContentPart::Text {
                text: "recorded".to_string(),
            }],
            LanguageCompletionReason::Stop,
            Usage::default(),
        )
        .map_err(|source| {
            Error::new(ErrorKind::Protocol, "invalid recording language response")
                .with_source(source)
                .into()
        })
    }

    async fn stream(
        &self,
        _request: LanguageRequest,
        _options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        self.stream_calls.fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorKind::Unsupported,
            "streaming is not part of this recording fixture",
        ))
    }
}

#[async_trait]
impl LanguageModel for LifecycleLanguage {
    async fn generate(
        &self,
        _request: LanguageRequest,
        _options: CallOptions,
    ) -> Result<LanguageResponse, LanguageCallError> {
        Err(Error::new(
            ErrorKind::Unsupported,
            "generation is not part of this lifecycle fixture",
        )
        .into())
    }

    async fn stream(
        &self,
        _request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        self.stream_calls.fetch_add(1, Ordering::SeqCst);
        let fixture = self.fixture;
        Ok(siumai::core::stream::established_stream(
            options.cancellation().clone(),
            move |_| {
                let mut events = vec![Ok(LanguageStreamEvent::TextDelta {
                    id: "text".to_string(),
                    delta: "partial".to_string(),
                })];
                match fixture {
                    StreamFixture::Completed => {
                        let response = LanguageResponse::completed(
                            vec![ContentPart::Text {
                                text: "complete".to_string(),
                            }],
                            LanguageCompletionReason::Stop,
                            Usage::default(),
                        )
                        .expect("valid lifecycle fixture response");
                        events.push(Ok(LanguageStreamEvent::Terminal(
                            StreamTerminal::Completed {
                                response: Box::new(response),
                            },
                        )));
                    }
                    StreamFixture::Failed => events.push(Err(Error::new(
                        ErrorKind::Provider,
                        "provider stream failure",
                    ))),
                    StreamFixture::UnexpectedEof => {}
                }
                futures::stream::iter(events)
            },
        ))
    }
}

#[async_trait]
impl LanguageModel for FailingLanguage {
    async fn generate(
        &self,
        _request: LanguageRequest,
        _options: CallOptions,
    ) -> Result<LanguageResponse, LanguageCallError> {
        self.generate_calls.fetch_add(1, Ordering::SeqCst);
        let partial = PartialLanguageOutput::new(
            vec![siumai::PartialLanguageOutputPart::Text {
                text: "provider partial".to_string(),
            }],
            Usage::default().with_output_tokens(1_u64),
        )
        .expect("valid provider partial output");
        Err(LanguageCallError::new(
            Error::new(ErrorKind::Provider, "provider generate failure"),
            Some(partial),
        ))
    }

    async fn stream(
        &self,
        _request: LanguageRequest,
        _options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        self.stream_calls.fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorKind::Provider,
            "provider stream setup failure",
        ))
    }
}

#[cfg(all(feature = "registry", feature = "runtime"))]
impl RegistryRuntimeLanguage {
    fn observe_options(&self, options: &CallOptions) -> Result<(), Error> {
        let selection = options.provider_options_for(self).map_err(|source| {
            Error::new(
                ErrorKind::Configuration,
                "invalid Registry runtime provider options",
            )
            .with_source(source)
        })?;
        let value = selection
            .raw_override()
            .and_then(|options| options.value().get("value"))
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::Configuration,
                    "missing Registry runtime provider option",
                )
            })?;
        self.observed_options
            .lock()
            .expect("Registry runtime observation lock")
            .push(value.to_string());
        Ok(())
    }
}

#[cfg(all(feature = "registry", feature = "runtime"))]
impl Model for RegistryRuntimeLanguage {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[cfg(all(feature = "registry", feature = "runtime"))]
#[async_trait]
impl LanguageModel for RegistryRuntimeLanguage {
    async fn generate(
        &self,
        _request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageResponse, LanguageCallError> {
        self.observe_options(&options)?;
        self.generate_calls.fetch_add(1, Ordering::SeqCst);
        LanguageResponse::completed(
            vec![ContentPart::Text {
                text: "Registry runtime".to_string(),
            }],
            LanguageCompletionReason::Stop,
            Usage::default(),
        )
        .map_err(|source| {
            Error::new(
                ErrorKind::Protocol,
                "invalid Registry runtime test response",
            )
            .with_source(source)
            .into()
        })
    }

    async fn stream(
        &self,
        _request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        self.observe_options(&options)?;
        self.stream_calls.fetch_add(1, Ordering::SeqCst);
        let cancellation = options.cancellation().clone();
        let response = LanguageResponse::completed(
            vec![ContentPart::Text {
                text: "Registry tool loop".to_string(),
            }],
            LanguageCompletionReason::Stop,
            Usage::default(),
        )
        .map_err(|source| {
            Error::new(
                ErrorKind::Protocol,
                "invalid Registry tool-loop test response",
            )
            .with_source(source)
        })?;
        Ok(siumai::core::stream::established_stream(
            cancellation,
            |_| {
                futures::stream::iter([Ok(LanguageStreamEvent::Terminal(
                    StreamTerminal::Completed {
                        response: Box::new(response),
                    },
                ))])
            },
        ))
    }
}

impl Model for FakeEmbedding {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

impl Model for DeadlineEmbedding {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

impl Model for FakeRerank {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

impl Model for FakeSpeech {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

impl Model for FakeTranscription {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl EmbeddingModel for DeadlineEmbedding {
    async fn embed(
        &self,
        request: EmbeddingRequest,
        options: CallOptions,
    ) -> Result<EmbeddingResponse, Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.observed.lock().unwrap().push(options);
        Ok(EmbeddingResponse {
            embeddings: request.inputs().iter().map(|_| vec![1.0]).collect(),
            metadata: ResponseMetadata::default(),
            usage: Usage::default(),
            warnings: Vec::new(),
            provider: BTreeMap::new(),
        })
    }
}

#[async_trait]
impl EmbeddingModel for FakeEmbedding {
    fn limits(&self) -> EmbeddingLimits {
        EmbeddingLimits {
            max_inputs: Some(4),
            max_input_tokens: None,
        }
    }

    async fn embed(
        &self,
        request: EmbeddingRequest,
        _options: CallOptions,
    ) -> Result<EmbeddingResponse, Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.limits().validate(&request)?;
        let response = EmbeddingResponse {
            embeddings: request.inputs().iter().map(|_| vec![1.0, 2.0]).collect(),
            metadata: ResponseMetadata::default(),
            usage: Usage::default(),
            warnings: Vec::new(),
            provider: BTreeMap::new(),
        };
        response.validate(&request)?;
        Ok(response)
    }
}

#[async_trait]
impl RerankModel for FakeRerank {
    async fn rerank(
        &self,
        request: RerankRequest,
        options: CallOptions,
    ) -> Result<RerankResponse, Error> {
        validate_expected_provider_options(self, &options, self.expect_provider_options)?;
        self.calls.fetch_add(1, Ordering::SeqCst);
        let count = request.top_n().unwrap_or(request.candidates().len());
        let response = RerankResponse {
            results: request
                .candidates()
                .iter()
                .take(count)
                .enumerate()
                .map(|(index, candidate)| {
                    siumai::RerankResult::new(
                        index,
                        1.0 - (index as f64 * 0.1),
                        candidate.id().map(ToString::to_string),
                    )
                    .expect("valid fake rerank result")
                })
                .collect(),
            metadata: ResponseMetadata::default(),
            usage: Usage::default(),
            warnings: Vec::new(),
            provider: BTreeMap::from([(
                "query".to_string(),
                serde_json::Value::String(request.query().to_string()),
            )]),
        };
        response.validate(&request)?;
        Ok(response)
    }
}

#[derive(Debug)]
struct FakeImage {
    descriptor: ModelDescriptor,
    calls: Arc<AtomicUsize>,
    expect_provider_options: bool,
}

impl Model for FakeImage {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl ImageModel for FakeImage {
    async fn generate_image(
        &self,
        request: ImageRequest,
        options: CallOptions,
    ) -> Result<ImageResponse, Error> {
        validate_expected_provider_options(self, &options, self.expect_provider_options)?;
        self.calls.fetch_add(1, Ordering::SeqCst);
        let response = ImageResponse {
            images: (0..request.count())
                .map(|_| ImageArtifact {
                    media_type: "image/png".to_string(),
                    data: MediaData::Url("https://example.test/image.png".to_string()),
                    revised_prompt: Some(request.prompt().to_string()),
                })
                .collect(),
            metadata: ResponseMetadata::default(),
            usage: Usage::default(),
            warnings: Vec::new(),
            provider: BTreeMap::new(),
        };
        response.validate(&request)?;
        Ok(response)
    }
}

#[async_trait]
impl SpeechModel for FakeSpeech {
    async fn synthesize(
        &self,
        request: SpeechRequest,
        options: CallOptions,
    ) -> Result<SpeechResponse, Error> {
        validate_expected_provider_options(self, &options, self.expect_provider_options)?;
        self.calls.fetch_add(1, Ordering::SeqCst);
        let response = SpeechResponse {
            media_type: "audio/wav".to_string(),
            audio: request.text().as_bytes().to_vec().into(),
            duration_seconds: Some(1.25),
            sample_rate_hz: Some(24_000),
            metadata: ResponseMetadata::default(),
            usage: Usage::default(),
            warnings: Vec::new(),
            provider: BTreeMap::new(),
        };
        response.validate()?;
        Ok(response)
    }
}

#[async_trait]
impl TranscriptionModel for FakeTranscription {
    async fn transcribe(
        &self,
        request: TranscriptionRequest,
        options: CallOptions,
    ) -> Result<TranscriptionResponse, Error> {
        validate_expected_provider_options(self, &options, self.expect_provider_options)?;
        self.calls.fetch_add(1, Ordering::SeqCst);
        let response = TranscriptionResponse {
            text: format!("{}:{}", request.media_type(), request.audio().len()),
            language: request.language().map(ToString::to_string),
            confidence: Some(0.99),
            duration_seconds: Some(2.5),
            segments: Vec::new(),
            metadata: ResponseMetadata::default(),
            usage: Usage::default(),
            warnings: Vec::new(),
            provider: BTreeMap::new(),
        };
        response.validate()?;
        Ok(response)
    }
}

fn fake(model: ModelId, calls: Arc<AtomicUsize>) -> FakeEmbedding {
    FakeEmbedding {
        descriptor: ModelDescriptor::new(
            ProviderId::new("fake").unwrap(),
            model,
            ModelFamily::Embedding,
        ),
        calls,
    }
}

fn fake_language(model: ModelId, calls: Arc<AtomicUsize>) -> FakeLanguage {
    FakeLanguage {
        descriptor: ModelDescriptor::new(
            ProviderId::new("fake").unwrap(),
            model,
            ModelFamily::Language,
        ),
        calls,
    }
}

fn fake_rerank(model: ModelId, calls: Arc<AtomicUsize>) -> FakeRerank {
    FakeRerank {
        descriptor: ModelDescriptor::new(
            ProviderId::new("fake").unwrap(),
            model,
            ModelFamily::Rerank,
        ),
        calls,
        expect_provider_options: false,
    }
}

fn fake_speech(model: ModelId, calls: Arc<AtomicUsize>) -> FakeSpeech {
    FakeSpeech {
        descriptor: ModelDescriptor::new(
            ProviderId::new("fake").unwrap(),
            model,
            ModelFamily::Speech,
        ),
        calls,
        expect_provider_options: false,
    }
}

fn fake_transcription(model: ModelId, calls: Arc<AtomicUsize>) -> FakeTranscription {
    FakeTranscription {
        descriptor: ModelDescriptor::new(
            ProviderId::new("fake").unwrap(),
            model,
            ModelFamily::Transcription,
        ),
        calls,
        expect_provider_options: false,
    }
}

fn recording_language(
    provider: &str,
    api_mode: &str,
    instance_id: siumai::core::ProviderInstanceId,
    route: Option<RouteId>,
    calls: Arc<AtomicUsize>,
    stream_calls: Arc<AtomicUsize>,
    observed: Arc<std::sync::Mutex<Vec<RecordedLanguageCall>>>,
) -> RecordingLanguage {
    let scope = siumai::core::ProviderScope::new(ProviderId::new(provider).unwrap())
        .with_api_mode(siumai::core::ApiModeId::new(api_mode).unwrap());
    RecordingLanguage {
        descriptor: ModelDescriptor::from_scope(
            Arc::new(scope),
            ModelId::new("language-v1").unwrap(),
            ModelFamily::Language,
            instance_id,
        ),
        route,
        calls,
        stream_calls,
        observed,
    }
}

fn fake_image(model: ModelId, calls: Arc<AtomicUsize>) -> FakeImage {
    FakeImage {
        descriptor: ModelDescriptor::new(
            ProviderId::new("fake").unwrap(),
            model,
            ModelFamily::Image,
        ),
        calls,
        expect_provider_options: false,
    }
}

#[tokio::test]
async fn root_language_generate_accepts_a_prompt_and_dispatches_once() {
    let calls = Arc::new(AtomicUsize::new(0));
    let model = fake_language(ModelId::new("language-v1").unwrap(), calls.clone());

    let response = siumai::language::generate(&model, "hello").await.unwrap();

    assert!(matches!(
        response.content(),
        [ContentPart::Text { text }] if text == "facade runtime"
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn language_facade_accepts_concrete_and_erased_models_through_one_generic_function() {
    async fn invoke<M, I>(model: &M, input: I) -> Result<LanguageResponse, LanguageCallError>
    where
        M: LanguageModel + ?Sized,
        I: Into<siumai::LanguageInput>,
    {
        siumai::language::generate(model, input).await
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let direct = fake_language(ModelId::new("direct-language-v1").unwrap(), calls.clone());
    let erased: Arc<dyn LanguageModel> = Arc::new(fake_language(
        ModelId::new("erased-language-v1").unwrap(),
        calls.clone(),
    ));

    let direct_response = invoke(&direct, Message::user("hello")).await.unwrap();
    let erased_response = invoke(&erased, vec![Message::user("hello")]).await.unwrap();

    assert_eq!(direct_response, erased_response);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn language_facade_does_not_trim_or_reject_an_empty_prompt() {
    let calls = Arc::new(AtomicUsize::new(0));
    let stream_calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
    let model = recording_language(
        "fixture",
        "language",
        siumai::core::ProviderInstanceId::new(),
        None,
        calls.clone(),
        stream_calls,
        observed.clone(),
    );

    siumai::language::generate(&model, "").await.unwrap();
    siumai::language::generate(&model, "  exact spacing  ")
        .await
        .unwrap();

    let observed = observed
        .lock()
        .expect("recording language observation lock");
    assert_eq!(
        observed[0].request,
        LanguageRequest::new(vec![Message::user("")])
    );
    assert_eq!(
        observed[1].request,
        LanguageRequest::new(vec![Message::user("  exact spacing  ")])
    );
    assert!(!observed[0].options.has_provider_options());
    assert!(observed[0].provider_values.is_empty());
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn language_facade_validates_requests_before_generate_or_stream_dispatch() {
    let calls = Arc::new(AtomicUsize::new(0));
    let stream_calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
    let route = RouteId::new("production").unwrap();
    let model = recording_language(
        "fixture",
        "language",
        siumai::core::ProviderInstanceId::new(),
        Some(route.clone()),
        calls.clone(),
        stream_calls.clone(),
        observed,
    );
    let invalid = LanguageRequest::new(vec![Message::text(MessageRole::Tool, "invalid")]);

    let generate_error = siumai::language::generate(&model, invalid.clone())
        .await
        .unwrap_err();
    let stream_error = siumai::language::stream(&model, invalid).await.unwrap_err();

    assert_eq!(generate_error.kind(), ErrorKind::InvalidInput);
    assert_eq!(generate_error.message(), "language request is invalid");
    assert!(generate_error.partial().is_none());
    assert_eq!(generate_error.context().route.as_ref(), Some(&route));
    assert_eq!(stream_error.kind(), ErrorKind::InvalidInput);
    assert_eq!(stream_error.message(), "language request is invalid");
    assert_eq!(stream_error.context().route.as_ref(), Some(&route));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(stream_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn language_facade_resolves_deadlines_before_request_validation() {
    let calls = Arc::new(AtomicUsize::new(0));
    let stream_calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
    let route = RouteId::new("deadline-route").unwrap();
    let model = recording_language(
        "fixture",
        "language",
        siumai::core::ProviderInstanceId::new(),
        Some(route.clone()),
        calls.clone(),
        stream_calls,
        observed,
    );
    let invalid = LanguageRequest::new(vec![Message::text(MessageRole::Tool, "invalid")]);
    let options = CallOptions::default().with_timeout(Duration::MAX).unwrap();

    let error = siumai::language::call(&model, invalid)
        .with_options(options)
        .expect("unresolved timeout remains valid during builder setup")
        .generate()
        .await
        .unwrap_err();

    assert_eq!(error.message(), "invalid call options");
    assert_eq!(error.context().route.as_ref(), Some(&route));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[test]
fn language_facade_rejects_foreign_instance_and_route_baselines_synchronously() {
    use serde_json::json;

    let selected_calls = Arc::new(AtomicUsize::new(0));
    let selected_stream_calls = Arc::new(AtomicUsize::new(0));
    let selected = recording_language(
        "fixture",
        "language",
        siumai::core::ProviderInstanceId::new(),
        None,
        selected_calls.clone(),
        selected_stream_calls,
        Arc::new(std::sync::Mutex::new(Vec::new())),
    );
    let foreign_instance = recording_language(
        "fixture",
        "language",
        siumai::core::ProviderInstanceId::new(),
        None,
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
        Arc::new(std::sync::Mutex::new(Vec::new())),
    );
    let foreign_instance_options = CallOptions::default()
        .with_raw_provider_options_for(&foreign_instance, json!({"value": "foreign"}))
        .unwrap();
    let instance_error =
        match siumai::language::call(&selected, "hello").with_options(foreign_instance_options) {
            Ok(_) => panic!("foreign configured instance must be rejected"),
            Err(error) => error,
        };
    assert!(matches!(
        instance_error,
        siumai::ProviderOptionError::ExactTargetMismatch { .. }
    ));

    let shared_instance = siumai::core::ProviderInstanceId::new();
    let selected_route = recording_language(
        "fixture",
        "language",
        shared_instance.clone(),
        Some(RouteId::new("production").unwrap()),
        selected_calls.clone(),
        Arc::new(AtomicUsize::new(0)),
        Arc::new(std::sync::Mutex::new(Vec::new())),
    );
    let foreign_route = recording_language(
        "fixture",
        "language",
        shared_instance,
        Some(RouteId::new("staging").unwrap()),
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
        Arc::new(std::sync::Mutex::new(Vec::new())),
    );
    let foreign_route_options = CallOptions::default()
        .with_raw_provider_options_for(&foreign_route, json!({"value": "foreign"}))
        .unwrap();
    let route_error = match siumai::language::call(&selected_route, "hello")
        .with_options(foreign_route_options)
    {
        Ok(_) => panic!("foreign Registry route must be rejected"),
        Err(error) => error,
    };
    assert!(matches!(
        route_error,
        siumai::ProviderOptionError::ExactTargetMismatch { .. }
    ));
    assert_eq!(selected_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn language_facade_preserves_provider_errors_and_partials() {
    let route = RouteId::new("provider-route").unwrap();
    let generate_calls = Arc::new(AtomicUsize::new(0));
    let stream_calls = Arc::new(AtomicUsize::new(0));
    let model = FailingLanguage {
        descriptor: ModelDescriptor::new(
            ProviderId::new("fixture").unwrap(),
            ModelId::new("failure-v1").unwrap(),
            ModelFamily::Language,
        ),
        route: Some(route),
        generate_calls: generate_calls.clone(),
        stream_calls: stream_calls.clone(),
    };

    let generate_error = siumai::language::generate(&model, "hello")
        .await
        .unwrap_err();
    let stream_error = siumai::language::stream(&model, "hello").await.unwrap_err();

    assert_eq!(generate_error.message(), "provider generate failure");
    assert!(generate_error.context().route.is_none());
    assert_eq!(
        generate_error
            .partial()
            .and_then(PartialLanguageOutput::output_text)
            .as_deref(),
        Some("provider partial")
    );
    assert_eq!(stream_error.message(), "provider stream setup failure");
    assert!(stream_error.context().route.is_none());
    assert_eq!(generate_calls.load(Ordering::SeqCst), 1);
    assert_eq!(stream_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn language_facade_preserves_established_stream_terminal_semantics() {
    use futures::StreamExt;

    async fn terminal(fixture: StreamFixture, options: CallOptions) -> StreamTerminal {
        let model = LifecycleLanguage {
            descriptor: ModelDescriptor::new(
                ProviderId::new("fixture").unwrap(),
                ModelId::new("stream-v1").unwrap(),
                ModelFamily::Language,
            ),
            stream_calls: Arc::new(AtomicUsize::new(0)),
            fixture,
        };
        let events = siumai::language::call(&model, "hello")
            .with_options(options)
            .unwrap()
            .stream()
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        match events.into_iter().last() {
            Some(LanguageStreamEvent::Terminal(terminal)) => terminal,
            _ => panic!("established stream must produce one terminal event"),
        }
    }

    assert!(matches!(
        terminal(StreamFixture::Completed, CallOptions::default()).await,
        StreamTerminal::Completed { response }
            if response.output_text().as_deref() == Some("complete")
    ));
    assert!(matches!(
        terminal(StreamFixture::Failed, CallOptions::default()).await,
        StreamTerminal::Failed {
            error,
            partial: Some(partial),
        } if error.kind() == ErrorKind::Provider
            && partial.output_text().as_deref() == Some("partial")
    ));
    assert!(matches!(
        terminal(StreamFixture::UnexpectedEof, CallOptions::default()).await,
        StreamTerminal::Failed {
            error,
            partial: Some(partial),
        } if error.kind() == ErrorKind::UnexpectedEof
            && partial.output_text().as_deref() == Some("partial")
    ));

    let cancellation = Cancellation::new();
    cancellation.cancel();
    assert!(matches!(
        terminal(
            StreamFixture::UnexpectedEof,
            CallOptions::default().with_cancellation(cancellation),
        )
        .await,
        StreamTerminal::Cancelled { partial: None, .. }
    ));
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn language_call_preserves_baseline_patch_order_and_execution_time_options() {
    use siumai::providers::openai::responses::OpenAiResponsesOptions;

    let calls = Arc::new(AtomicUsize::new(0));
    let stream_calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
    let model = recording_language(
        "openai",
        "responses",
        siumai::core::ProviderInstanceId::new(),
        None,
        calls.clone(),
        stream_calls,
        observed.clone(),
    );
    let cancellation = Cancellation::new();
    let baseline = CallOptions::default()
        .with_cancellation(cancellation.clone())
        .with_max_attempts(2)
        .unwrap()
        .with_timeout(Duration::from_secs(1))
        .unwrap()
        .with_provider_options_for(
            &model,
            &OpenAiResponsesOptions {
                instructions: Some("A".to_string()),
                ..OpenAiResponsesOptions::default()
            },
        )
        .unwrap()
        .with_provider_options_for(
            &model,
            &OpenAiResponsesOptions {
                instructions: Some("B".to_string()),
                ..OpenAiResponsesOptions::default()
            },
        )
        .unwrap();
    let replaced_baseline = CallOptions::default()
        .with_provider_options_for(
            &model,
            &OpenAiResponsesOptions {
                instructions: Some("replaced".to_string()),
                ..OpenAiResponsesOptions::default()
            },
        )
        .unwrap();
    let call = siumai::language::call(&model, Message::user("hello"))
        .with_options(replaced_baseline)
        .unwrap()
        .with_provider_options(&OpenAiResponsesOptions {
            instructions: Some("C".to_string()),
            ..OpenAiResponsesOptions::default()
        })
        .unwrap()
        .with_options(baseline)
        .unwrap();

    tokio::time::sleep(Duration::from_millis(100)).await;
    let invoked_at = Instant::now();
    call.generate().await.unwrap();
    cancellation.cancel();

    let observed = observed
        .lock()
        .expect("recording language observation lock");
    let call = observed.first().expect("one recorded language call");
    assert_eq!(call.provider_values, ["A", "B", "C"]);
    assert_eq!(call.options.timeout(), None);
    assert!(call.options.deadline().unwrap() >= invoked_at + Duration::from_millis(925));
    assert_eq!(call.options.retry().maximum_attempts(), Some(2));
    assert!(call.options.cancellation().is_cancelled());
    assert_eq!(
        call.request,
        LanguageRequest::new(vec![Message::user("hello")])
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[cfg(feature = "openai")]
#[test]
fn language_call_option_setters_validate_the_full_candidate_synchronously() {
    use siumai::core::MAX_PROVIDER_OPTION_ENTRIES;
    use siumai::providers::openai::responses::OpenAiResponsesOptions;

    let calls = Arc::new(AtomicUsize::new(0));
    let responses = recording_language(
        "openai",
        "responses",
        siumai::core::ProviderInstanceId::new(),
        None,
        calls.clone(),
        Arc::new(AtomicUsize::new(0)),
        Arc::new(std::sync::Mutex::new(Vec::new())),
    );
    let chat = recording_language(
        "openai",
        "chat-completions",
        siumai::core::ProviderInstanceId::new(),
        None,
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
        Arc::new(std::sync::Mutex::new(Vec::new())),
    );

    let mismatch = match siumai::language::call(&chat, "hello")
        .with_provider_options(&OpenAiResponsesOptions::default())
    {
        Ok(_) => panic!("Responses options must not bind to Chat Completions"),
        Err(error) => error,
    };
    assert!(matches!(
        mismatch,
        siumai::ProviderOptionError::TargetMismatch { .. }
    ));

    let mut full_baseline = CallOptions::default();
    for index in 0..MAX_PROVIDER_OPTION_ENTRIES {
        full_baseline = full_baseline
            .with_provider_options_for(
                &responses,
                &OpenAiResponsesOptions {
                    instructions: Some(format!("baseline-{index}")),
                    ..OpenAiResponsesOptions::default()
                },
            )
            .unwrap();
    }
    let overflow = match siumai::language::call(&responses, "hello")
        .with_options(full_baseline.clone())
        .unwrap()
        .with_provider_options(&OpenAiResponsesOptions::default())
    {
        Ok(_) => panic!("builder patch must be validated with the baseline"),
        Err(error) => error,
    };
    assert!(matches!(
        overflow,
        siumai::ProviderOptionError::TooManyEntries { .. }
    ));

    let overflow = match siumai::language::call(&responses, "hello")
        .with_provider_options(&OpenAiResponsesOptions::default())
        .unwrap()
        .with_options(full_baseline)
    {
        Ok(_) => panic!("replacement baseline must be validated with builder patches"),
        Err(error) => error,
    };
    assert!(matches!(
        overflow,
        siumai::ProviderOptionError::TooManyEntries { .. }
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn language_facade_preserves_a_complete_rich_request() {
    use serde_json::json;
    use siumai::core::{ProtocolId, ProviderProvenance, ProviderScope};
    use siumai::providers::openai::prompt_cache::OpenAiContentOptions;
    use siumai::providers::openai::responses::OpenAiResponsesOptions;
    use siumai::{
        GenerationConfig, OpaqueProviderItem, ReplayDomain, ReplayDomainId, StructuredOutputSpec,
    };

    let calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
    let model = recording_language(
        "openai",
        "responses",
        siumai::core::ProviderInstanceId::new(),
        None,
        calls.clone(),
        Arc::new(AtomicUsize::new(0)),
        observed.clone(),
    );
    let annotated = MessagePart::text("annotated")
        .with_provider_annotation(&OpenAiContentOptions::prompt_cache_breakpoint())
        .unwrap();
    let replay_scope = ProviderScope::new(ProviderId::new("openai").unwrap())
        .with_protocol(ProtocolId::new("responses").unwrap())
        .with_api_mode(siumai::core::ApiModeId::new("responses").unwrap())
        .with_replay_domain(ReplayDomain::custom(
            ReplayDomainId::new("facade-rich-request").unwrap(),
        ));
    let provenance =
        ProviderProvenance::from_scope(&replay_scope, ModelId::new("language-v1").unwrap())
            .unwrap();
    let opaque = OpaqueProviderItem::new(
        provenance,
        "response_state",
        json!({"opaque": "preserve exactly"}),
    )
    .unwrap();
    let tool = ToolSpec::new(
        "lookup",
        Some("Look up a record".to_string()),
        json!({
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"]
        }),
    )
    .unwrap();
    let request = LanguageRequest {
        messages: vec![
            Message::new(MessageRole::User, [annotated]),
            Message::new(
                MessageRole::Assistant,
                [ContentPart::ProviderOpaque(opaque)],
            ),
        ],
        generation: GenerationConfig {
            max_output_tokens: Some(321),
            temperature: Some(0.25),
            top_p: Some(0.8),
            stop_sequences: vec!["done".to_string()],
            seed: Some(7),
        },
        tools: vec![tool],
        tool_choice: Some(ToolChoice::Named {
            name: "lookup".to_string(),
        }),
        structured_output: Some(StructuredOutputSpec {
            name: "answer".to_string(),
            description: Some("Structured answer".to_string()),
            schema: json!({
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"]
            }),
            strict: true,
        }),
    };
    let expected = request.clone();
    let call = siumai::language::call(&model, request)
        .with_provider_options(&OpenAiResponsesOptions {
            instructions: Some("preserve rich input".to_string()),
            ..OpenAiResponsesOptions::default()
        })
        .unwrap();
    assert_eq!(call.request(), &expected);
    assert!(!call.base_options().has_provider_options());

    call.generate().await.unwrap();

    let observed = observed
        .lock()
        .expect("recording language observation lock");
    assert_eq!(observed[0].request, expected);
    assert_eq!(observed[0].provider_values, ["preserve rich input"]);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn anthropic_provider_executes_typed_options_through_the_language_facade() {
    use serde_json::json;
    use siumai::providers::anthropic::messages::OutputEffort;
    use siumai::providers::anthropic::options::AnthropicMessagesOptions;
    use siumai::providers::anthropic::{AnthropicCredential, AnthropicProvider};
    use siumai::transport::EndpointConfig;
    use siumai::{GenerationConfig, ReplayDomain, ReplayDomainId};
    use wiremock::matchers::{body_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_json(json!({
            "model": "future-claude-model",
            "max_tokens": 4096,
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "caller intent"}]
            }],
            "stream": false,
            "temperature": 0.4,
            "top_p": 0.5,
            "top_k": 32,
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "output_config": {"effort": "high"}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_facade",
            "type": "message",
            "role": "assistant",
            "model": "future-claude-model",
            "content": [{"type": "text", "text": "done"}],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {"input_tokens": 1, "output_tokens": 1}
        })))
        .expect(1)
        .mount(&server)
        .await;
    let provider = AnthropicProvider::builder(AnthropicCredential::unauthenticated())
        .with_endpoint(EndpointConfig::local_explicit(format!("{}/v1/", server.uri())).unwrap())
        .with_replay_domain(ReplayDomain::custom(
            ReplayDomainId::new("facade-anthropic-test").unwrap(),
        ))
        .build()
        .unwrap();
    let model = provider.language("future-claude-model").unwrap();
    let request = LanguageRequest::new(vec![Message::user("caller intent")]).with_generation(
        GenerationConfig {
            max_output_tokens: Some(4_096),
            temperature: Some(0.4),
            top_p: Some(0.5),
            stop_sequences: Vec::new(),
            seed: None,
        },
    );
    let options = AnthropicMessagesOptions::new()
        .with_enabled_thinking(2_048)
        .with_output_effort(OutputEffort::High)
        .with_top_k(32);

    let response = siumai::language::call(&model, request)
        .with_provider_options(&options)
        .unwrap()
        .generate()
        .await
        .unwrap();

    assert_eq!(response.output_text().as_deref(), Some("done"));
}

#[cfg(feature = "registry")]
#[tokio::test]
async fn language_facade_preserves_registry_route_identity_on_the_selected_handle() {
    use serde_json::json;
    use siumai::core::{ProviderInstanceId, ProviderScope};
    use siumai::registry::Registry;

    let calls = Arc::new(AtomicUsize::new(0));
    let stream_calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
    let scope = Arc::new(
        ProviderScope::new(ProviderId::new("fixture").unwrap())
            .with_api_mode(siumai::core::ApiModeId::new("language").unwrap()),
    );
    let instance_id = ProviderInstanceId::new();

    let production_scope = scope.clone();
    let production_instance = instance_id.clone();
    let production_calls = calls.clone();
    let production_stream_calls = stream_calls.clone();
    let production_observed = observed.clone();
    let production = ProviderRegistration::from_language(
        scope.clone(),
        Arc::new(move |_model| {
            Ok(Arc::new(recording_language(
                production_scope.provider_id().as_str(),
                production_scope
                    .api_mode()
                    .expect("fixture language API mode")
                    .as_str(),
                production_instance.clone(),
                None,
                production_calls.clone(),
                production_stream_calls.clone(),
                production_observed.clone(),
            )) as Arc<dyn LanguageModel>)
        }),
    );
    let staging_scope = scope.clone();
    let staging_instance = instance_id;
    let staging = ProviderRegistration::from_language(
        scope,
        Arc::new(move |_model| {
            Ok(Arc::new(recording_language(
                staging_scope.provider_id().as_str(),
                staging_scope
                    .api_mode()
                    .expect("fixture language API mode")
                    .as_str(),
                staging_instance.clone(),
                None,
                Arc::new(AtomicUsize::new(0)),
                Arc::new(AtomicUsize::new(0)),
                Arc::new(std::sync::Mutex::new(Vec::new())),
            )) as Arc<dyn LanguageModel>)
        }),
    );
    let mut builder = Registry::builder();
    builder
        .register_named("production", production)
        .unwrap()
        .register_named("staging", staging)
        .unwrap();
    let registry = builder.build().unwrap();
    let production = registry.language_model("production:language-v1").unwrap();
    let staging = registry.language_model("staging:language-v1").unwrap();
    assert_eq!(
        production.route_id().map(RouteId::as_str),
        Some("production")
    );

    let options = CallOptions::default()
        .with_raw_provider_options_for(production.as_ref(), json!({"value": "production"}))
        .unwrap();
    siumai::language::call(production.as_ref(), "hello")
        .with_options(options)
        .unwrap()
        .generate()
        .await
        .unwrap();

    let foreign = CallOptions::default()
        .with_raw_provider_options_for(staging.as_ref(), json!({"value": "staging"}))
        .unwrap();
    let error = match siumai::language::call(production.as_ref(), "hello").with_options(foreign) {
        Ok(_) => panic!("a different resolved route must fail before dispatch"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        siumai::ProviderOptionError::ExactTargetMismatch { .. }
    ));

    let invalid = LanguageRequest::new(vec![Message::text(MessageRole::Tool, "invalid")]);
    let generate_error = siumai::language::generate(production.as_ref(), invalid.clone())
        .await
        .unwrap_err();
    let stream_error = siumai::language::stream(production.as_ref(), invalid)
        .await
        .unwrap_err();
    assert_eq!(
        generate_error.context().route.as_ref().map(RouteId::as_str),
        Some("production")
    );
    assert_eq!(
        stream_error.context().route.as_ref().map(RouteId::as_str),
        Some("production")
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(stream_calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        observed
            .lock()
            .expect("recording language observation lock")[0]
            .provider_values,
        ["production"]
    );
}

#[tokio::test]
async fn non_language_root_facades_support_concrete_and_erased_models() {
    let embedding_calls = Arc::new(AtomicUsize::new(0));
    let direct_embedding = fake(
        ModelId::new("direct-embedding-v1").unwrap(),
        embedding_calls.clone(),
    );
    let erased_embedding: Arc<dyn EmbeddingModel> = Arc::new(fake(
        ModelId::new("erased-embedding-v1").unwrap(),
        embedding_calls.clone(),
    ));
    let embedding_request = EmbeddingRequest::new(["one", "two"]).unwrap();
    let direct_embedding_response = embedding::embed(&direct_embedding, embedding_request.clone())
        .await
        .unwrap();
    let embedding_call = embedding::call(erased_embedding.as_ref(), embedding_request.clone())
        .with_options(CallOptions::default().with_max_attempts(2).unwrap())
        .unwrap();
    assert_eq!(embedding_call.request(), &embedding_request);
    assert_eq!(
        embedding_call.base_options().retry().maximum_attempts(),
        Some(2)
    );
    let erased_embedding_response = embedding_call.embed().await.unwrap();
    assert_eq!(direct_embedding_response, erased_embedding_response);
    assert_eq!(direct_embedding_response.embeddings.len(), 2);
    assert_eq!(embedding_calls.load(Ordering::SeqCst), 2);

    let rerank_calls = Arc::new(AtomicUsize::new(0));
    let direct_rerank = fake_rerank(
        ModelId::new("direct-rerank-v1").unwrap(),
        rerank_calls.clone(),
    );
    let erased_rerank: Arc<dyn RerankModel> = Arc::new(fake_rerank(
        ModelId::new("erased-rerank-v1").unwrap(),
        rerank_calls.clone(),
    ));
    let rerank_request = RerankRequest::new(
        "query",
        vec![
            RerankCandidate::new("first")
                .unwrap()
                .with_id("first-id")
                .unwrap(),
            RerankCandidate::new("second")
                .unwrap()
                .with_id("second-id")
                .unwrap(),
        ],
    )
    .unwrap();
    let direct_rerank_response = rerank::rerank(&direct_rerank, rerank_request.clone())
        .await
        .unwrap();
    let rerank_call = rerank::call(erased_rerank.as_ref(), rerank_request.clone())
        .with_options(CallOptions::default().with_max_attempts(2).unwrap())
        .unwrap();
    assert_eq!(rerank_call.request(), &rerank_request);
    assert_eq!(
        rerank_call.base_options().retry().maximum_attempts(),
        Some(2)
    );
    let erased_rerank_response = rerank_call.rerank().await.unwrap();
    assert_eq!(direct_rerank_response, erased_rerank_response);
    assert_eq!(direct_rerank_response.results.len(), 2);
    assert_eq!(direct_rerank_response.provider["query"], "query");
    assert_eq!(rerank_calls.load(Ordering::SeqCst), 2);

    let image_calls = Arc::new(AtomicUsize::new(0));
    let direct_image = fake_image(
        ModelId::new("direct-image-v1").unwrap(),
        image_calls.clone(),
    );
    let erased_image: Arc<dyn ImageModel> = Arc::new(fake_image(
        ModelId::new("erased-image-v1").unwrap(),
        image_calls.clone(),
    ));
    let image_request = ImageRequest::new("draw an exact red square")
        .unwrap()
        .with_count(2)
        .unwrap();
    let direct_image_response = image::generate(&direct_image, image_request.clone())
        .await
        .unwrap();
    let image_call = image::call(erased_image.as_ref(), image_request.clone())
        .with_options(CallOptions::default().with_max_attempts(2).unwrap())
        .unwrap();
    assert_eq!(image_call.request(), &image_request);
    assert_eq!(
        image_call.base_options().retry().maximum_attempts(),
        Some(2)
    );
    let erased_image_response = image_call.generate().await.unwrap();
    assert_eq!(direct_image_response, erased_image_response);
    assert_eq!(direct_image_response.images.len(), 2);
    assert_eq!(
        direct_image_response.images[0].revised_prompt.as_deref(),
        Some("draw an exact red square")
    );
    assert_eq!(image_calls.load(Ordering::SeqCst), 2);

    let speech_calls = Arc::new(AtomicUsize::new(0));
    let direct_speech = fake_speech(
        ModelId::new("direct-speech-v1").unwrap(),
        speech_calls.clone(),
    );
    let erased_speech: Arc<dyn SpeechModel> = Arc::new(fake_speech(
        ModelId::new("erased-speech-v1").unwrap(),
        speech_calls.clone(),
    ));
    let speech_request = SpeechRequest::new("speak exactly")
        .unwrap()
        .with_voice("fixture-voice")
        .unwrap()
        .with_format("wav")
        .unwrap()
        .with_language("en")
        .unwrap()
        .with_speed(1.25)
        .unwrap();
    let direct_speech_response = speech::synthesize(&direct_speech, speech_request.clone())
        .await
        .unwrap();
    let speech_call = speech::call(erased_speech.as_ref(), speech_request.clone())
        .with_options(CallOptions::default().with_max_attempts(2).unwrap())
        .unwrap();
    assert_eq!(speech_call.request(), &speech_request);
    assert_eq!(
        speech_call.base_options().retry().maximum_attempts(),
        Some(2)
    );
    let erased_speech_response = speech_call.synthesize().await.unwrap();
    assert_eq!(direct_speech_response, erased_speech_response);
    assert_eq!(direct_speech_response.audio.as_ref(), b"speak exactly");
    assert_eq!(speech_calls.load(Ordering::SeqCst), 2);

    let transcription_calls = Arc::new(AtomicUsize::new(0));
    let direct_transcription = fake_transcription(
        ModelId::new("direct-transcription-v1").unwrap(),
        transcription_calls.clone(),
    );
    let erased_transcription: Arc<dyn TranscriptionModel> = Arc::new(fake_transcription(
        ModelId::new("erased-transcription-v1").unwrap(),
        transcription_calls.clone(),
    ));
    let transcription_request = TranscriptionRequest::new(vec![1_u8, 2, 3], "audio/wav")
        .unwrap()
        .with_language("en")
        .unwrap()
        .with_prompt("preserve this prompt")
        .unwrap();
    let direct_transcription_response =
        transcription::transcribe(&direct_transcription, transcription_request.clone())
            .await
            .unwrap();
    let transcription_call =
        transcription::call(erased_transcription.as_ref(), transcription_request.clone())
            .with_options(CallOptions::default().with_max_attempts(2).unwrap())
            .unwrap();
    assert_eq!(transcription_call.request(), &transcription_request);
    assert_eq!(transcription_call.request().audio().as_ref(), [1_u8, 2, 3]);
    assert_eq!(transcription_call.request().media_type(), "audio/wav");
    assert_eq!(
        transcription_call.base_options().retry().maximum_attempts(),
        Some(2)
    );
    let erased_transcription_response = transcription_call.transcribe().await.unwrap();
    assert_eq!(direct_transcription_response, erased_transcription_response);
    assert_eq!(direct_transcription_response.text, "audio/wav:3");
    assert_eq!(transcription_calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn non_language_facade_validates_model_limits_before_dispatch() {
    let calls = Arc::new(AtomicUsize::new(0));
    let model = fake(ModelId::new("limited-embedding-v1").unwrap(), calls.clone());
    let request = EmbeddingRequest::new(["one", "two", "three", "four", "five"]).unwrap();

    let error = embedding::embed(&model, request).await.unwrap_err();

    assert_eq!(error.kind(), ErrorKind::LimitExceeded);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn embedding_call_resolves_relative_timeout_at_invocation() {
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
    let model = DeadlineEmbedding {
        descriptor: ModelDescriptor::new(
            ProviderId::new("fake").unwrap(),
            ModelId::new("deadline-v1").unwrap(),
            ModelFamily::Embedding,
        ),
        calls: calls.clone(),
        observed: observed.clone(),
    };
    let options = CallOptions::default()
        .with_timeout(Duration::from_secs(1))
        .unwrap();
    tokio::time::sleep(Duration::from_millis(150)).await;
    let invoked_at = Instant::now();

    embedding::call(&model, EmbeddingRequest::single("hello").unwrap())
        .with_options(options)
        .unwrap()
        .embed()
        .await
        .unwrap();

    {
        let observed = observed.lock().unwrap();
        let received = observed.first().expect("one call option observation");
        assert_eq!(received.timeout(), None);
        assert!(received.deadline().unwrap() >= invoked_at + Duration::from_millis(925));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    let error = embedding::call(&model, EmbeddingRequest::single("hello").unwrap())
        .with_options(CallOptions::default().with_timeout(Duration::MAX).unwrap())
        .unwrap()
        .embed()
        .await
        .unwrap_err();
    assert_eq!(error.kind(), siumai::ErrorKind::InvalidInput);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn facade_convergence_removes_the_redundant_public_paths() {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    assert!(!manifest_dir.join("src/families.rs").exists());

    let lib = include_str!("../src/lib.rs");
    let prelude = include_str!("../src/prelude.rs");
    assert!(!lib.contains("pub mod families;"));
    for module in [
        "embedding",
        "image",
        "language",
        "rerank",
        "speech",
        "transcription",
    ] {
        assert!(lib.contains(&format!("pub mod {module};")));
    }

    let runtime_exports = lib
        .split_once("pub use runtime::{")
        .expect("runtime root exports remain curated")
        .1
        .split_once("};")
        .expect("runtime root export block")
        .0;
    assert!(!runtime_exports.contains("generate"));
    assert!(!runtime_exports.contains("stream"));
    let prelude_runtime_exports = prelude
        .split_once("pub use crate::{RunBudget")
        .expect("runtime prelude exports remain curated")
        .1
        .split_once("};")
        .expect("runtime prelude export block")
        .0;
    assert!(!prelude_runtime_exports.contains("generate"));
    assert!(!prelude_runtime_exports.contains("stream"));

    let facade_modules = [
        include_str!("../src/language.rs"),
        include_str!("../src/embedding.rs"),
        include_str!("../src/rerank.rs"),
        include_str!("../src/image.rs"),
        include_str!("../src/speech.rs"),
        include_str!("../src/transcription.rs"),
    ];
    for removed in [
        "generate_with_options",
        "stream_with_options",
        "embed_with_options",
        "rerank_with_options",
        "synthesize_with_options",
        "transcribe_with_options",
    ] {
        assert!(
            facade_modules
                .iter()
                .all(|source| !source.contains(removed))
        );
    }
}

#[cfg(all(feature = "openai", feature = "cohere"))]
#[tokio::test]
async fn non_language_calls_bind_typed_options_to_the_exact_executing_model() {
    use siumai::core::{ApiModeId, ProviderInstanceId, ProviderScope};
    use siumai::providers::cohere::options::CohereRerankOptions;
    use siumai::providers::openai::audio::speech::OpenAiSpeechOptions;
    use siumai::providers::openai::audio::transcription::OpenAiTranscriptionOptions;
    use siumai::providers::openai::embeddings::OpenAiEmbeddingOptions;
    use siumai::providers::openai::images::OpenAiImageOptions;

    fn descriptor(
        provider: &str,
        api_mode: &str,
        family: ModelFamily,
        model: &str,
    ) -> ModelDescriptor {
        let scope = ProviderScope::new(ProviderId::new(provider).unwrap())
            .with_api_mode(ApiModeId::new(api_mode).unwrap());
        ModelDescriptor::from_scope(
            Arc::new(scope),
            ModelId::new(model).unwrap(),
            family,
            ProviderInstanceId::new(),
        )
    }

    let embedding_calls = Arc::new(AtomicUsize::new(0));
    let embedding_observed = Arc::new(std::sync::Mutex::new(Vec::new()));
    let embedding = DeadlineEmbedding {
        descriptor: descriptor(
            "openai",
            "embeddings",
            ModelFamily::Embedding,
            "embedding-v1",
        ),
        calls: embedding_calls.clone(),
        observed: embedding_observed.clone(),
    };
    let embedding_options = OpenAiEmbeddingOptions::new()
        .with_user("facade-test")
        .unwrap();
    let embedding_call = embedding::call(
        &embedding,
        EmbeddingRequest::single("typed embedding").unwrap(),
    )
    .with_provider_options(&embedding_options)
    .unwrap();
    assert!(!embedding_call.base_options().has_provider_options());
    embedding_call.embed().await.unwrap();
    {
        let observed = embedding_observed
            .lock()
            .expect("embedding option observation lock");
        let selection = observed[0].provider_options_for(&embedding).unwrap();
        let mut selected = selection.typed();
        let options = selected.next().expect("one typed embedding option");
        assert!(selected.next().is_none());
        assert_eq!(options.namespace().as_str(), "openai");
        assert_eq!(options.value()["user"], "facade-test");
    }
    assert_eq!(embedding_calls.load(Ordering::SeqCst), 1);

    let foreign_embedding = DeadlineEmbedding {
        descriptor: descriptor(
            "openai",
            "embeddings",
            ModelFamily::Embedding,
            "embedding-v1",
        ),
        calls: Arc::new(AtomicUsize::new(0)),
        observed: Arc::new(std::sync::Mutex::new(Vec::new())),
    };
    let foreign_baseline = CallOptions::default()
        .with_provider_options_for(&foreign_embedding, &embedding_options)
        .unwrap();
    let mismatch = match embedding::call(
        &embedding,
        EmbeddingRequest::single("typed embedding").unwrap(),
    )
    .with_options(foreign_baseline)
    {
        Ok(_) => panic!("foreign configured instance must fail during builder setup"),
        Err(error) => error,
    };
    assert!(matches!(
        mismatch,
        siumai::ProviderOptionError::ExactTargetMismatch { .. }
    ));
    assert_eq!(embedding_calls.load(Ordering::SeqCst), 1);

    let rerank_calls = Arc::new(AtomicUsize::new(0));
    let rerank_model = FakeRerank {
        descriptor: descriptor("cohere", "v2", ModelFamily::Rerank, "rerank-v1"),
        calls: rerank_calls.clone(),
        expect_provider_options: true,
    };
    rerank::call(
        &rerank_model,
        RerankRequest::new(
            "typed query",
            vec![RerankCandidate::new("candidate").unwrap()],
        )
        .unwrap(),
    )
    .with_provider_options(&CohereRerankOptions::new().with_priority(1))
    .unwrap()
    .rerank()
    .await
    .unwrap();
    assert_eq!(rerank_calls.load(Ordering::SeqCst), 1);

    let image_calls = Arc::new(AtomicUsize::new(0));
    let image_model = FakeImage {
        descriptor: descriptor(
            "openai",
            "image-generations",
            ModelFamily::Image,
            "image-v1",
        ),
        calls: image_calls.clone(),
        expect_provider_options: true,
    };
    image::call(&image_model, ImageRequest::new("typed image").unwrap())
        .with_provider_options(&OpenAiImageOptions::default())
        .unwrap()
        .generate()
        .await
        .unwrap();
    assert_eq!(image_calls.load(Ordering::SeqCst), 1);

    let speech_calls = Arc::new(AtomicUsize::new(0));
    let speech_model = FakeSpeech {
        descriptor: descriptor("openai", "audio-speech", ModelFamily::Speech, "speech-v1"),
        calls: speech_calls.clone(),
        expect_provider_options: true,
    };
    speech::call(&speech_model, SpeechRequest::new("typed speech").unwrap())
        .with_provider_options(&OpenAiSpeechOptions::default())
        .unwrap()
        .synthesize()
        .await
        .unwrap();
    assert_eq!(speech_calls.load(Ordering::SeqCst), 1);

    let transcription_calls = Arc::new(AtomicUsize::new(0));
    let transcription_model = FakeTranscription {
        descriptor: descriptor(
            "openai",
            "audio-transcriptions",
            ModelFamily::Transcription,
            "transcription-v1",
        ),
        calls: transcription_calls.clone(),
        expect_provider_options: true,
    };
    transcription::call(
        &transcription_model,
        TranscriptionRequest::new(vec![1_u8, 2, 3], "audio/wav").unwrap(),
    )
    .with_provider_options(&OpenAiTranscriptionOptions::default())
    .unwrap()
    .transcribe()
    .await
    .unwrap();
    assert_eq!(transcription_calls.load(Ordering::SeqCst), 1);
}

#[cfg(feature = "transport")]
#[test]
fn facade_exposes_curated_transport_configuration() {
    use siumai::transport::{
        AttemptLoopOutcome, EndpointConfig, EndpointError, EndpointPolicy, HttpTransportRoute,
        LocalNetworkGrant, OfficialOrigin, ProviderHttpTransportSettings, ProxyBasicCredential,
        ProxyEndpoint, RetryLimit, RetryPolicy, RetryReason, TransportCallId, TransportConfigError,
        TransportEvent, TransportLimits, TransportObserver, TransportRetryPolicyError,
    };

    struct FacadeObserver;

    impl TransportObserver for FacadeObserver {
        fn observe(&self, event: &TransportEvent) {
            let _: TransportCallId = event.call_id();
        }
    }

    let endpoint = EndpointConfig::new(
        "http://127.0.0.1:43191/v1",
        EndpointPolicy::LocalExplicit(LocalNetworkGrant::Loopback),
    )
    .unwrap();
    assert!(matches!(
        endpoint.policy(),
        EndpointPolicy::LocalExplicit(LocalNetworkGrant::Loopback)
    ));

    let official_origin = OfficialOrigin::new("https://api.example.com").unwrap();
    EndpointConfig::official("https://api.example.com", official_origin).unwrap();
    let invalid_endpoint: Result<EndpointConfig, EndpointError> =
        EndpointConfig::public_custom("http://api.example.com");
    assert!(invalid_endpoint.is_err());

    let limits = TransportLimits {
        max_connections: 8,
        max_in_flight_requests: 4,
        ..TransportLimits::default()
    };
    let retry_policy = RetryPolicy::new(4)
        .unwrap()
        .with_max_server_delay(Duration::from_secs(5));
    let proxy = ProxyEndpoint::https("https://proxy.example.com").unwrap();
    let route = HttpTransportRoute::trusted_connect(proxy)
        .with_basic_auth(ProxyBasicCredential::new("proxy-user", "proxy-secret").unwrap())
        .unwrap();
    let settings = ProviderHttpTransportSettings::default()
        .with_limits(limits.clone())
        .unwrap()
        .with_retry_policy(retry_policy)
        .with_connect_timeout(Duration::from_secs(2))
        .unwrap()
        .with_call_timeout(Duration::from_secs(30))
        .unwrap()
        .with_read_timeout(Duration::from_secs(10))
        .unwrap()
        .with_observer(Arc::new(FacadeObserver))
        .with_route(route)
        .unwrap();
    assert_eq!(settings.limits(), &limits);
    assert_eq!(settings.retry_policy().max_attempts(), 4);
    assert!(settings.route().has_basic_auth());
    let debug = format!("{settings:?}");
    assert!(!debug.contains("proxy-user"));
    assert!(!debug.contains("proxy-secret"));

    let invalid_settings: Result<ProviderHttpTransportSettings, TransportConfigError> =
        ProviderHttpTransportSettings::default().with_call_timeout(Duration::ZERO);
    assert!(matches!(
        invalid_settings,
        Err(TransportConfigError::ZeroTimeout {
            name: "call_timeout"
        })
    ));
    let invalid_retry: Result<RetryPolicy, TransportRetryPolicyError> = RetryPolicy::new(0);
    assert_eq!(invalid_retry, Err(TransportRetryPolicyError::ZeroAttempts));

    let options = CallOptions::default()
        .with_timeout(Duration::from_secs(3))
        .unwrap()
        .with_max_attempts(2)
        .unwrap();
    let retry_intent: RetryIntent = options.retry();
    assert_eq!(retry_intent.maximum_attempts(), Some(2));
    assert_eq!(options.timeout(), Some(Duration::from_secs(3)));
    let invalid_options: Result<CallOptions, siumai::CallOptionsError> =
        CallOptions::default().with_max_attempts(0);
    assert!(matches!(
        invalid_options,
        Err(siumai::CallOptionsError::ZeroAttempts)
    ));

    let _ = RetryReason::Transport;
    let _ = RetryLimit::CallerCap;
    let _ = AttemptLoopOutcome::Cancelled;
}

#[cfg(all(feature = "openai", feature = "anthropic"))]
#[test]
fn facade_reuses_http_settings_across_flagship_providers() {
    use siumai::providers::anthropic::{AnthropicCredential, AnthropicProvider};
    use siumai::providers::openai::{OpenAiCredential, OpenAiProvider};
    use siumai::transport::{ProviderHttpTransportSettings, RetryPolicy};

    let settings =
        ProviderHttpTransportSettings::default().with_retry_policy(RetryPolicy::new(2).unwrap());
    let openai = OpenAiProvider::builder(OpenAiCredential::api_key("test-openai-key"))
        .with_http_transport_settings(settings.clone())
        .build()
        .unwrap();
    let anthropic = AnthropicProvider::builder(AnthropicCredential::api_key("test-anthropic-key"))
        .with_http_transport_settings(settings)
        .build()
        .unwrap();

    assert_eq!(openai.provider_id().as_str(), "openai");
    assert_eq!(anthropic.provider_id().as_str(), "anthropic");
}

#[cfg(all(feature = "runtime", feature = "json-schema"))]
#[test]
fn facade_keeps_json_schema_validation_in_the_runtime_namespace() {
    use serde_json::json;
    use siumai::runtime::json_schema::JsonSchemaValidator;

    let validator = JsonSchemaValidator::new(&json!({
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": false
    }))
    .unwrap();

    assert!(validator.is_valid(&json!({"answer": "ok"})));
    assert!(!validator.is_valid(&json!({"answer": 42})));
}

#[cfg(feature = "runtime")]
#[test]
fn facade_exposes_runtime_budget_configuration() {
    use siumai::runtime::{BudgetError, BudgetKind, RunBudget, RunBudgetBuilder, RunTimeouts};

    let timeouts = RunTimeouts::new(
        Duration::from_secs(120),
        Duration::from_secs(30),
        Duration::from_secs(10),
        Duration::from_secs(15),
        Duration::from_secs(20),
    )
    .unwrap();
    let budget_result: Result<RunBudget, BudgetError> = RunBudgetBuilder::default()
        .max_model_steps(4)
        .max_tool_calls(8)
        .timeouts(timeouts)
        .build();
    let budget = budget_result.unwrap();
    let runtime = Runtime::builder().with_run_budget(budget.clone()).build();

    let _: siumai::RunBudget = budget.clone();
    let _: siumai::RunBudgetBuilder = siumai::RunBudget::builder();
    let _: siumai::RunTimeouts = timeouts;
    let _: BudgetKind = BudgetKind::ModelSteps;
    assert_eq!(runtime.run_budget(), &budget);
}

#[cfg(feature = "runtime")]
#[tokio::test]
async fn runtime_one_call_language_execution_remains_namespaced() {
    let calls = Arc::new(AtomicUsize::new(0));
    let model = fake_language(ModelId::new("language-v1").unwrap(), calls.clone());
    let response = siumai::runtime::generate(
        &model,
        LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")]),
        CallOptions::default(),
    )
    .await
    .unwrap();

    assert!(matches!(
        &response.content()[0],
        ContentPart::Text { text } if text == "facade runtime"
    ));
    let stream_error = siumai::runtime::stream(
        &model,
        LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")]),
        CallOptions::default(),
    )
    .await
    .unwrap_err();
    assert_eq!(stream_error.kind(), ErrorKind::Unsupported);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[cfg(feature = "volcengine")]
#[test]
fn provider_native_with_options_methods_remain_nameable() {
    use siumai::providers::volcengine::ArkImages;

    let _generate_with_options = ArkImages::generate_with_options;
}

#[cfg(all(feature = "registry", feature = "runtime"))]
#[tokio::test]
async fn registry_language_models_preserve_route_options_through_runtime_and_tool_loop() {
    use serde_json::json;
    use siumai::core::{ProviderInstanceId, ProviderScope};
    use siumai::registry::Registry;

    let generate_calls = Arc::new(AtomicUsize::new(0));
    let stream_calls = Arc::new(AtomicUsize::new(0));
    let observed_options = Arc::new(std::sync::Mutex::new(Vec::new()));
    let scope = Arc::new(ProviderScope::new(
        ProviderId::new("registry-runtime").unwrap(),
    ));
    let factory_scope = scope.clone();
    let instance_id = ProviderInstanceId::new();
    let factory_generate_calls = generate_calls.clone();
    let factory_stream_calls = stream_calls.clone();
    let factory_observed_options = observed_options.clone();
    let registration = ProviderRegistration::from_language(
        scope,
        Arc::new(move |model| {
            Ok(Arc::new(RegistryRuntimeLanguage {
                descriptor: ModelDescriptor::from_scope(
                    factory_scope.clone(),
                    model,
                    ModelFamily::Language,
                    instance_id.clone(),
                ),
                generate_calls: factory_generate_calls.clone(),
                stream_calls: factory_stream_calls.clone(),
                observed_options: factory_observed_options.clone(),
            }) as Arc<dyn LanguageModel>)
        }),
    );
    let mut builder = Registry::builder();
    builder
        .register_named("production", registration)
        .unwrap()
        .alias_named("recommended", "production")
        .unwrap();
    let registry = builder.build().unwrap();
    let model = registry.language_model("recommended:language-v1").unwrap();
    assert_eq!(model.route_id().map(RouteId::as_str), Some("production"));

    let runtime_options = CallOptions::default()
        .with_raw_provider_options_for(model.as_ref(), json!({"value": "runtime"}))
        .unwrap();
    let response = Runtime::default()
        .generate(
            model.as_ref(),
            LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")]),
            StepOptions::default(),
            runtime_options,
        )
        .await
        .unwrap();
    assert!(matches!(
        response.content(),
        [ContentPart::Text { text }] if text == "Registry runtime"
    ));

    let tool_loop_options = CallOptions::default()
        .with_raw_provider_options_for(model.as_ref(), json!({"value": "tool-loop"}))
        .unwrap();
    let terminal = Runtime::default()
        .tool_loop(model, Default::default())
        .run(
            LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")]),
            tool_loop_options,
        )
        .await
        .unwrap();
    assert!(terminal.is_completed());
    assert_eq!(generate_calls.load(Ordering::SeqCst), 1);
    assert_eq!(stream_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        *observed_options
            .lock()
            .expect("Registry runtime observation lock"),
        vec!["runtime".to_string(), "tool-loop".to_string()]
    );
}

#[cfg(feature = "registry")]
#[tokio::test]
async fn direct_and_registry_image_paths_share_one_family_contract() {
    use siumai::registry::Registry;

    let calls = Arc::new(AtomicUsize::new(0));
    let direct = fake_image(ModelId::new("image-v1").unwrap(), calls.clone());
    let scope = Arc::new(direct.descriptor().scope().clone());
    let factory_calls = calls.clone();
    let registration = ProviderRegistration::from_image(
        scope,
        Arc::new(move |model| {
            Ok(Arc::new(fake_image(model, factory_calls.clone())) as Arc<dyn ImageModel>)
        }),
    );
    let mut builder = Registry::builder();
    builder.register_named("primary", registration).unwrap();
    let registry = builder.build().unwrap();
    let erased = registry.image_model("primary:image-v1").unwrap();

    let request = ImageRequest::new("draw a tiny red square").unwrap();
    let direct_response = image::generate(&direct, request.clone()).await.unwrap();
    let erased_response = image::generate(&erased, request).await.unwrap();

    assert_eq!(direct_response, erased_response);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[cfg(feature = "registry")]
#[tokio::test]
async fn direct_and_registry_paths_use_the_same_family_contract() {
    use siumai::registry::Registry;

    let calls = Arc::new(AtomicUsize::new(0));
    let direct = fake(ModelId::new("embed-v1").unwrap(), calls.clone());
    let scope = Arc::new(direct.descriptor().scope().clone());
    let factory_calls = calls.clone();
    let registration = ProviderRegistration::from_embedding(
        scope,
        Arc::new(move |model| {
            Ok(Arc::new(fake(model, factory_calls.clone())) as Arc<dyn EmbeddingModel>)
        }),
    );
    let mut builder = Registry::builder();
    builder.register_named("primary", registration).unwrap();
    let registry = builder.build().unwrap();
    let erased = registry.embedding_model("primary:embed-v1").unwrap();

    let direct_response = embedding::embed(&direct, EmbeddingRequest::single("one").unwrap())
        .await
        .unwrap();
    let erased_response = embedding::embed(&erased, EmbeddingRequest::single("one").unwrap())
        .await
        .unwrap();

    assert_eq!(direct_response, erased_response);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[cfg(feature = "registry")]
#[test]
fn facade_exposes_typed_registry_resolve_context() {
    let context: Option<siumai::registry::RegistryModelContext> = None;
    assert!(context.is_none());
}

#[cfg(all(
    feature = "registry",
    feature = "openai",
    feature = "openai-compatible",
    feature = "google",
    feature = "alibaba",
    feature = "cohere",
    feature = "deepgram",
    feature = "elevenlabs"
))]
#[test]
fn facade_registration_sources_cover_all_six_stable_families() {
    use siumai::providers::{
        alibaba, cohere, deepgram, elevenlabs, google, openai, openai_compatible,
    };
    use siumai::registry::ProviderRegistrationSource;

    fn assert_registration_source<T: ProviderRegistrationSource>() {}

    assert_registration_source::<openai::OpenAiProvider>();
    assert_registration_source::<alibaba::AlibabaProvider>();
    assert_registration_source::<openai_compatible::OpenAiCompatibleProvider>();
    assert_registration_source::<google::GeminiProvider>();
    assert_registration_source::<cohere::CohereProvider>();
    assert_registration_source::<deepgram::DeepgramProvider>();
    assert_registration_source::<elevenlabs::ElevenLabsProvider>();
}

#[cfg(feature = "cohere")]
#[test]
fn facade_exposes_cohere_model_less_transcription_resource() {
    use siumai::providers::cohere::{
        CohereProvider, CohereTranscriptionRequest, CohereTranscriptions,
    };

    let provider = CohereProvider::builder("test-key").build().unwrap();
    let _: CohereTranscriptions = provider.transcriptions();
    let request = CohereTranscriptionRequest::new(vec![1_u8], "audio/wav", "en").unwrap();
    assert_eq!(request.language(), "en");
    assert_eq!(provider.support_manifest().native_claims().len(), 1);
}

#[cfg(feature = "openai-compatible")]
#[test]
fn facade_exposes_explicit_responses_wire_dialects() {
    use siumai::providers::openai_compatible::ResponsesWireDialect;

    let _dialect = ResponsesWireDialect::compatible();
}

#[cfg(all(feature = "deepgram", feature = "elevenlabs"))]
#[test]
fn facade_exposes_deepgram_speech_and_elevenlabs_transcription() {
    use siumai::core::{Model, ModelFamily, ProviderOptions};
    use siumai::providers::{deepgram, elevenlabs};

    let deepgram =
        deepgram::DeepgramProvider::builder(deepgram::DeepgramCredential::api_key("test-key"))
            .build()
            .unwrap();
    assert!(deepgram.registration().supports_family(ModelFamily::Speech));
    assert_eq!(
        deepgram
            .default_speech_model()
            .unwrap()
            .descriptor()
            .api_mode(),
        Some("tts")
    );

    let elevenlabs = elevenlabs::ElevenLabsProvider::builder(
        elevenlabs::ElevenLabsProfile::official().unwrap(),
        elevenlabs::ElevenLabsCredential::api_key("test-key"),
    )
    .build()
    .unwrap();
    assert!(
        elevenlabs
            .registration()
            .supports_family(ModelFamily::Transcription)
    );
    assert_eq!(
        elevenlabs
            .default_transcription_model()
            .unwrap()
            .descriptor()
            .api_mode(),
        Some("batch-transcription")
    );

    let options = ProviderOptions::typed(
        &elevenlabs::options::ElevenLabsTranscriptionOptions::new()
            .with_diarize(true)
            .with_timestamps_granularity(elevenlabs::options::ElevenLabsTimestampGranularity::Word),
    )
    .unwrap();
    assert_eq!(options.namespace().as_str(), "elevenlabs");
    assert_eq!(
        options.api_mode().map(siumai::core::ApiModeId::as_str),
        Some("batch-transcription")
    );
}

#[cfg(feature = "openai")]
#[test]
fn facade_exposes_openai_portable_families_and_provider_owned_resources() {
    use siumai::core::{ModelFamily, ReplayDomain, ReplayDomainId};
    use siumai::providers::openai::audio::speech::{GPT_4O_MINI_TTS, OpenAiSpeechOptions};
    use siumai::providers::openai::audio::transcription::{
        GPT_TRANSCRIBE, OpenAiTranscriptionOptions,
    };
    use siumai::providers::openai::chat_completions::{
        OpenAiChatCompletionsOptions, OpenAiReasoningEffort, OpenAiServiceTier, OpenAiTextVerbosity,
    };
    use siumai::providers::openai::embeddings::{OpenAiEmbeddingOptions, TEXT_EMBEDDING_3_SMALL};
    use siumai::providers::openai::experimental::skills::{
        OpenAiSkillUpload, OpenAiSkillsProviderExt,
    };
    use siumai::providers::openai::images::{GPT_IMAGE_1, OpenAiImageOptions};
    use siumai::providers::openai::prompt_cache::OpenAiContentOptions;
    use siumai::providers::openai::resources::conversations::OpenAiConversationCreateRequest;
    use siumai::providers::openai::resources::files::{
        OpenAiBinaryContent, OpenAiFileUploadPurpose,
    };
    use siumai::providers::openai::resources::vector_stores::OpenAiVectorStoreCreateRequest;
    use siumai::providers::openai::responses::{OpenAiContextManagement, OpenAiResponsesOptions};
    use siumai::providers::openai::{OpenAiCredential, OpenAiProvider};
    use siumai::transport::EndpointConfig;

    let provider = OpenAiProvider::builder(OpenAiCredential::unauthenticated())
        .with_endpoint(EndpointConfig::local_explicit("http://127.0.0.1:43191/v1").unwrap())
        .with_replay_domain(ReplayDomain::custom(
            ReplayDomainId::new("facade-openai-surfaces").unwrap(),
        ))
        .build()
        .unwrap();
    let registration = provider.registration();

    assert_eq!(
        registration.families().collect::<Vec<_>>(),
        vec![
            ModelFamily::Language,
            ModelFamily::Embedding,
            ModelFamily::Image,
            ModelFamily::Speech,
            ModelFamily::Transcription,
        ]
    );
    assert!(provider.embedding(TEXT_EMBEDDING_3_SMALL).is_ok());
    assert!(provider.image(GPT_IMAGE_1).is_ok());
    assert!(provider.speech(GPT_4O_MINI_TTS).is_ok());
    assert!(provider.transcription(GPT_TRANSCRIBE).is_ok());

    let _ = OpenAiEmbeddingOptions::default();
    let _ = OpenAiImageOptions::default();
    let _ = OpenAiSpeechOptions::default();
    let _ = OpenAiTranscriptionOptions::default();
    let _ = OpenAiContentOptions::prompt_cache_breakpoint();
    let _: siumai::MessagePart = MessagePart::text("facade message part");
    let _ = OpenAiChatCompletionsOptions {
        reasoning_effort: Some(OpenAiReasoningEffort::High),
        service_tier: Some(OpenAiServiceTier::Priority),
        text_verbosity: Some(OpenAiTextVerbosity::Low),
        ..OpenAiChatCompletionsOptions::default()
    };
    let _ = OpenAiResponsesOptions {
        context_management: vec![OpenAiContextManagement::compaction(None)],
        ..OpenAiResponsesOptions::default()
    };
    let _ = OpenAiConversationCreateRequest::new();
    let _ = OpenAiVectorStoreCreateRequest::new();
    let _: Option<OpenAiSkillUpload> = None;
    let _: Option<OpenAiFileUploadPurpose> = Some(OpenAiFileUploadPurpose::UserData);
    let _: Option<OpenAiBinaryContent> = None;

    assert!(!format!("{:?}", provider.conversations()).contains("credential"));
    assert!(!format!("{:?}", provider.files()).contains("credential"));
    assert!(!format!("{:?}", provider.vector_stores()).contains("credential"));
    assert!(!format!("{:?}", provider.skills()).contains("credential"));
}

#[cfg(feature = "google")]
#[test]
fn facade_exposes_the_current_google_multi_family_and_native_slices() {
    use siumai::core::{ApiStability, Model, ModelFamily, ProviderOptions, VerifiedFidelity};
    use siumai::providers::google::models::{
        GEMINI_3_1_FLASH_IMAGE, GEMINI_3_1_FLASH_TTS_PREVIEW, GEMINI_3_6_FLASH,
        GEMINI_EMBEDDING_001, GEMINI_EMBEDDING_2, current_image_models,
        current_interactions_models,
    };
    use siumai::providers::google::options::{
        GeminiEmbeddingOptions, GeminiEmbeddingTaskType, GeminiGenerateContentOptions,
        GeminiGenerateContentServiceTier, GeminiGenerateContentThinking,
        GeminiGenerateContentThinkingLevel, GeminiImageAspectRatio, GeminiImageOptions,
        GeminiImageSize, GeminiInteractionsOptions, GeminiSpeechOptions, GeminiThinkingLevel,
    };
    use siumai::providers::google::{
        GeminiCredential, GeminiEmbeddingContentPart, GeminiMultimodalEmbeddingRequest,
        GeminiProvider,
    };

    let provider = GeminiProvider::builder(GeminiCredential::api_key("test-key"))
        .build()
        .unwrap();
    let registration = provider.registration();
    assert!(registration.supports_family(ModelFamily::Language));
    assert!(registration.supports_family(ModelFamily::Embedding));
    assert!(registration.supports_family(ModelFamily::Image));
    assert!(registration.supports_family(ModelFamily::Speech));
    assert_eq!(
        provider
            .language(GEMINI_3_6_FLASH)
            .unwrap()
            .model_id()
            .as_str(),
        GEMINI_3_6_FLASH
    );
    assert_eq!(
        provider
            .image(GEMINI_3_1_FLASH_IMAGE)
            .unwrap()
            .model_id()
            .as_str(),
        GEMINI_3_1_FLASH_IMAGE
    );
    assert_eq!(
        provider
            .embedding(GEMINI_EMBEDDING_001)
            .unwrap()
            .model_id()
            .as_str(),
        GEMINI_EMBEDDING_001
    );
    let multimodal = provider.multimodal_embedding(GEMINI_EMBEDDING_2).unwrap();
    assert_eq!(
        multimodal.descriptor().api_mode(),
        Some("embed-content-v1beta-multimodal")
    );
    let _ =
        GeminiMultimodalEmbeddingRequest::new([GeminiEmbeddingContentPart::text("hello").unwrap()])
            .unwrap();
    assert_eq!(
        provider
            .speech(GEMINI_3_1_FLASH_TTS_PREVIEW)
            .unwrap()
            .model_id()
            .as_str(),
        GEMINI_3_1_FLASH_TTS_PREVIEW
    );
    assert_eq!(
        provider
            .generate_content(GEMINI_3_6_FLASH)
            .unwrap()
            .descriptor()
            .api_mode(),
        Some("generate-content")
    );
    let _veo = provider.veo();
    let generate_content_registration = provider.generate_content_registration();
    assert!(generate_content_registration.supports_family(ModelFamily::Language));
    assert_eq!(current_interactions_models().len(), 3);
    assert_eq!(current_image_models().len(), 3);

    let image_options = ProviderOptions::typed(
        &GeminiImageOptions::new()
            .with_aspect_ratio(GeminiImageAspectRatio::LandscapeSixteenNine)
            .with_image_size(GeminiImageSize::TwoK),
    )
    .unwrap();
    assert_eq!(image_options.namespace().as_str(), "google");
    let language_options = ProviderOptions::typed(
        &GeminiInteractionsOptions::new().with_thinking_level(GeminiThinkingLevel::High),
    )
    .unwrap();
    assert_eq!(language_options.namespace().as_str(), "google");
    let speech_options =
        ProviderOptions::typed(&GeminiSpeechOptions::new().with_voice("Kore").unwrap()).unwrap();
    assert_eq!(speech_options.namespace().as_str(), "google");
    let embedding_options = ProviderOptions::typed(
        &GeminiEmbeddingOptions::new().with_task_type(GeminiEmbeddingTaskType::RetrievalDocument),
    )
    .unwrap();
    assert_eq!(embedding_options.namespace().as_str(), "google");
    let generate_content_options = ProviderOptions::typed(
        &GeminiGenerateContentOptions::new()
            .with_service_tier(GeminiGenerateContentServiceTier::Standard)
            .with_thinking(
                GeminiGenerateContentThinking::new()
                    .with_level(GeminiGenerateContentThinkingLevel::High),
            ),
    )
    .unwrap();
    assert_eq!(generate_content_options.namespace().as_str(), "google");

    let claims = provider
        .profile()
        .provider_profile()
        .verified_claims()
        .unwrap();
    assert_eq!(claims.len(), 5);
    assert!(claims.iter().any(|claim| {
        claim.scope().api_mode().as_str() == "generate-content"
            && claim.evidence().upstream().support_status()
                == Some(siumai::core::UpstreamSupportStatus::Legacy)
    }));
    assert!(
        claims
            .iter()
            .all(|claim| claim.fidelity() == VerifiedFidelity::Native)
    );
    assert!(claims.iter().any(|claim| {
        claim.scope().api_mode().as_str() == "interactions-speech"
            && claim.stability() == ApiStability::Experimental
    }));
    assert!(
        claims
            .iter()
            .filter(|claim| claim.scope().api_mode().as_str() != "interactions-speech")
            .all(|claim| claim.stability() == ApiStability::Stable)
    );
    assert_eq!(provider.support_manifest().native_claims().len(), 3);
    assert!(
        provider
            .support_manifest()
            .native_claims()
            .iter()
            .any(|claim| {
                claim.scope().binding().surface_id().map(|id| id.as_str()) == Some("files-metadata")
            })
    );
    assert!(
        provider
            .support_manifest()
            .native_claims()
            .iter()
            .any(|claim| {
                claim.scope().binding().surface_id().map(|id| id.as_str())
                    == Some("veo-predict-long-running")
            })
    );
    assert!(
        provider
            .support_manifest()
            .native_claims()
            .iter()
            .any(|claim| {
                claim.scope().binding().api_mode().map(|id| id.as_str())
                    == Some("embed-content-v1beta-multimodal")
            })
    );
}

#[cfg(all(feature = "registry", feature = "deepseek"))]
#[test]
fn facade_exposes_deepseek_as_a_curated_registration_source() {
    use siumai::core::{ModelFamily, ProviderOptions};
    use siumai::providers::deepseek::options::{
        DeepSeekChatOptions, DeepSeekReasoningEffort, DeepSeekResponsesOptions,
    };
    use siumai::providers::deepseek::{DeepSeekCredential, DeepSeekLanguageApi, DeepSeekProvider};
    use siumai::registry::ProviderRegistrationSource;

    fn assert_registration_source<T: ProviderRegistrationSource>() {}

    assert_registration_source::<DeepSeekProvider>();

    let provider = DeepSeekProvider::builder(DeepSeekCredential::unauthenticated())
        .build()
        .unwrap();
    assert_eq!(
        provider.language("future-deepseek-model").unwrap().api(),
        DeepSeekLanguageApi::ChatCompletions
    );
    assert_eq!(
        provider.responses("future-deepseek-model").unwrap().api(),
        DeepSeekLanguageApi::Responses
    );
    assert_eq!(
        provider
            .beta_chat_completions("future-deepseek-model")
            .unwrap()
            .api(),
        DeepSeekLanguageApi::BetaChatCompletions
    );
    assert_eq!(
        provider
            .registration()
            .api_mode(ModelFamily::Language)
            .map(siumai::core::ApiModeId::as_str),
        Some("chat-completions")
    );

    let chat_options = ProviderOptions::typed(
        &DeepSeekChatOptions::new().with_reasoning_effort(DeepSeekReasoningEffort::High),
    )
    .unwrap();
    assert_eq!(chat_options.namespace().as_str(), "deepseek");
    assert_eq!(chat_options.value()["reasoning_effort"], "high");

    let responses_options = ProviderOptions::typed(
        &DeepSeekResponsesOptions::new()
            .with_reasoning_effort(DeepSeekReasoningEffort::Max)
            .with_web_search(),
    )
    .unwrap();
    assert_eq!(
        responses_options
            .api_mode()
            .map(siumai::core::ApiModeId::as_str),
        Some("responses")
    );
    assert_eq!(responses_options.value()["native_tools"][0], "web_search");
}

#[cfg(all(feature = "registry", feature = "minimax"))]
#[test]
fn facade_exposes_minimax_as_a_curated_composite_provider() {
    use siumai::core::ProviderOptions;
    use siumai::providers::minimax::options::{
        MinimaxMessagesOptions, MinimaxReasoningEffort, MinimaxResponsesOptions,
        MinimaxServiceTier, MinimaxThinking,
    };
    use siumai::providers::minimax::resources::{
        MinimaxCustomVoiceId, MinimaxResponsesInputTokenRequest, MinimaxVideoResolution,
    };
    use siumai::providers::minimax::{
        MinimaxCredential, MinimaxLanguageApi, MinimaxProvider, models,
    };
    use siumai::registry::ProviderRegistrationSource;

    fn assert_registration_source<T: ProviderRegistrationSource>() {}

    assert_registration_source::<MinimaxProvider>();
    assert_eq!(MinimaxVideoResolution::K2.as_str(), "2K");

    let provider = MinimaxProvider::builder(MinimaxCredential::api_key("test-key"))
        .build()
        .unwrap();
    let _files = provider.files();
    let _images = provider.images();
    let _music = provider.music();
    let _responses_resource = provider.responses_resource();
    let _speech = provider.speech();
    let _video = provider.video();
    let _voices = provider.voices();
    let _image_model = provider.image(models::image::IMAGE_01).unwrap();
    let _speech_model = provider
        .speech_model(siumai::core::ModelId::new(models::speech::SPEECH_2_8_HD).unwrap())
        .unwrap();
    let _input_tokens = MinimaxResponsesInputTokenRequest::from_language_request(
        "MiniMax-M3",
        siumai::core::LanguageRequest::new(vec![siumai::core::Message::user("hello")]),
    )
    .unwrap();
    let _custom_voice = MinimaxCustomVoiceId::new("FacadeVoice01").unwrap();
    assert_eq!(
        provider.language("future-minimax-model").unwrap().api(),
        MinimaxLanguageApi::Messages
    );
    assert_eq!(
        provider
            .chat_completions("future-minimax-model")
            .unwrap()
            .api(),
        MinimaxLanguageApi::ChatCompletions
    );
    assert_eq!(
        provider.responses("future-minimax-model").unwrap().api(),
        MinimaxLanguageApi::Responses
    );
    assert_eq!(
        provider
            .provider_registration()
            .unwrap()
            .api_mode(ModelFamily::Language)
            .map(siumai::core::ApiModeId::as_str),
        Some("messages")
    );
    assert_eq!(
        provider
            .provider_registration()
            .unwrap()
            .api_mode(ModelFamily::Image)
            .map(siumai::core::ApiModeId::as_str),
        Some("image-generation")
    );
    assert_eq!(
        provider
            .provider_registration()
            .unwrap()
            .api_mode(ModelFamily::Speech)
            .map(siumai::core::ApiModeId::as_str),
        Some("speech-http")
    );

    let messages = ProviderOptions::typed(
        &MinimaxMessagesOptions::new()
            .with_thinking(MinimaxThinking::Adaptive)
            .with_service_tier(MinimaxServiceTier::Priority),
    )
    .unwrap();
    assert_eq!(messages.namespace().as_str(), "minimax");
    assert_eq!(messages.value()["thinking"]["type"], "adaptive");

    let responses = ProviderOptions::typed(
        &MinimaxResponsesOptions::new()
            .with_reasoning_effort(MinimaxReasoningEffort::High)
            .with_prompt_cache_key("conversation-1"),
    )
    .unwrap();
    assert_eq!(
        responses.api_mode().map(siumai::core::ApiModeId::as_str),
        Some("responses")
    );
    assert_eq!(responses.value()["reasoning"]["effort"], "high");
}

#[cfg(all(feature = "registry", feature = "anthropic"))]
#[test]
fn facade_exposes_anthropic_as_a_curated_provider() {
    use siumai::core::ProviderOptions;
    use siumai::providers::anthropic::annotations::{AnthropicFileReference, AnthropicMessageFile};
    use siumai::providers::anthropic::messages::{
        AnthropicHostedToolBlockRef, AnthropicHostedToolResultKind, AnthropicOpaqueContentExt,
        MessagesCodecError,
    };
    use siumai::providers::anthropic::options::{AnthropicMessagesOptions, AnthropicThinking};
    use siumai::providers::anthropic::resources::{
        AnthropicBatchResultsStreamError, AnthropicSkillFile, AnthropicSkillVersionListQuery,
        AnthropicSkillVersionUpload,
    };
    use siumai::providers::anthropic::{AnthropicCredential, AnthropicProvider};
    use siumai::registry::ProviderRegistrationSource;

    fn assert_registration_source<T: ProviderRegistrationSource>() {}
    fn assert_public_type<T>() {}
    fn assert_opaque_extension<T: AnthropicOpaqueContentExt + ?Sized>() {}
    fn inspect_native(
        item: &siumai::core::OpaqueProviderItem,
    ) -> Result<Option<AnthropicHostedToolBlockRef<'_>>, MessagesCodecError> {
        item.anthropic_hosted_tool()
    }

    assert_registration_source::<AnthropicProvider>();
    assert_public_type::<AnthropicFileReference>();
    assert_public_type::<AnthropicMessageFile>();
    assert_public_type::<AnthropicHostedToolBlockRef<'static>>();
    assert_public_type::<AnthropicHostedToolResultKind>();
    assert_public_type::<MessagesCodecError>();
    assert_public_type::<AnthropicBatchResultsStreamError>();
    assert_public_type::<AnthropicSkillVersionListQuery>();
    assert_opaque_extension::<siumai::core::OpaqueProviderItem>();
    let _inspect_native = inspect_native;
    let _version_upload = AnthropicSkillVersionUpload::new(vec![
        AnthropicSkillFile::new("fixture/SKILL.md", "text/plain", "instructions")
            .expect("Skill file"),
    ])
    .expect("version upload");

    let provider = AnthropicProvider::builder(AnthropicCredential::api_key("test-key"))
        .build()
        .unwrap();
    let _files = provider.files();
    let _batches = provider.message_batches();
    let _tokens = provider.tokens();
    let _skills = provider.skills();
    let _language = provider.language("future-claude-model").unwrap();
    assert_eq!(
        provider
            .provider_registration()
            .unwrap()
            .api_mode(ModelFamily::Language)
            .map(siumai::core::ApiModeId::as_str),
        Some("messages")
    );

    let options = ProviderOptions::typed(
        &AnthropicMessagesOptions::new().with_thinking(AnthropicThinking::adaptive()),
    )
    .unwrap();
    assert_eq!(options.namespace().as_str(), "anthropic");
    assert_eq!(options.value()["thinking"]["type"], "adaptive");
}

#[cfg(all(feature = "registry", feature = "google-vertex-anthropic"))]
#[test]
fn facade_exposes_anthropic_on_vertex_as_a_curated_provider() {
    use siumai::core::{ApiModeId, Model, ProviderOptions, ReplayDomain, ReplayDomainId};
    use siumai::providers::google_vertex_anthropic::models::CLAUDE_SONNET_5;
    use siumai::providers::google_vertex_anthropic::options::GoogleVertexAnthropicMessagesOptions;
    use siumai::providers::google_vertex_anthropic::{
        GOOGLE_VERTEX_ANTHROPIC_REPLAY_AUDIENCE, GoogleVertexAnthropicProvider,
        GoogleVertexCredential,
    };
    use siumai::registry::ProviderRegistrationSource;

    fn assert_registration_source<T: ProviderRegistrationSource>() {}

    assert_registration_source::<GoogleVertexAnthropicProvider>();

    let provider = GoogleVertexAnthropicProvider::builder(
        "test-project",
        "us-east5",
        GoogleVertexCredential::access_token("test-token"),
    )
    .with_replay_domain(
        ReplayDomain::official(
            ReplayDomainId::new(GOOGLE_VERTEX_ANTHROPIC_REPLAY_AUDIENCE).unwrap(),
        )
        .with_caller_scope(ReplayDomainId::new("facade-vertex-test").unwrap()),
    )
    .build()
    .unwrap();
    assert_eq!(
        provider
            .language(CLAUDE_SONNET_5)
            .unwrap()
            .descriptor()
            .api_mode(),
        Some("messages")
    );
    assert_eq!(
        provider
            .provider_registration()
            .unwrap()
            .api_mode(ModelFamily::Language)
            .map(ApiModeId::as_str),
        Some("messages")
    );

    let options = ProviderOptions::typed(
        &GoogleVertexAnthropicMessagesOptions::new().with_adaptive_thinking(),
    )
    .unwrap();
    assert_eq!(options.namespace().as_str(), "google");
    assert_eq!(options.api_mode().map(ApiModeId::as_str), Some("messages"));
    assert_eq!(options.value()["thinking"]["type"], "adaptive");
}

#[cfg(all(feature = "registry", feature = "groq"))]
#[test]
fn facade_exposes_groq_language_and_audio_surfaces() {
    use siumai::core::ProviderOptions;
    use siumai::providers::groq::options::{
        GroqLanguageOptions, GroqResponsesOptions, GroqResponsesServiceTier, GroqServiceTier,
    };
    use siumai::providers::groq::{GroqCredential, GroqProvider};
    use siumai::registry::ProviderRegistrationSource;

    fn assert_registration_source<T: ProviderRegistrationSource>() {}

    assert_registration_source::<GroqProvider>();

    let provider = GroqProvider::builder(GroqCredential::api_key("test-key"))
        .build()
        .unwrap();
    assert_eq!(
        provider
            .language("future-groq-chat-model")
            .unwrap()
            .descriptor()
            .api_mode(),
        Some("chat-completions")
    );
    assert_eq!(
        provider
            .responses("future-groq-responses-model")
            .unwrap()
            .descriptor()
            .api_mode(),
        Some("responses")
    );
    assert_eq!(
        provider
            .transcription("future-groq-transcription-model")
            .unwrap()
            .descriptor()
            .provider()
            .as_str(),
        "groq"
    );
    assert_eq!(
        provider
            .default_speech_model()
            .unwrap()
            .descriptor()
            .api_mode(),
        Some("audio-speech")
    );
    let _audio = provider.audio();
    assert_eq!(
        provider
            .registration()
            .api_mode(ModelFamily::Language)
            .map(siumai::core::ApiModeId::as_str),
        Some("chat-completions")
    );

    let chat = ProviderOptions::typed(
        &GroqLanguageOptions::new().with_service_tier(GroqServiceTier::Flex),
    )
    .unwrap();
    assert_eq!(
        chat.api_mode().map(siumai::core::ApiModeId::as_str),
        Some("chat-completions")
    );
    assert_eq!(chat.value()["service_tier"], "flex");

    let responses = ProviderOptions::typed(
        &GroqResponsesOptions::new()
            .with_service_tier(GroqResponsesServiceTier::Flex)
            .with_code_execution(true),
    )
    .unwrap();
    assert_eq!(
        responses.api_mode().map(siumai::core::ApiModeId::as_str),
        Some("responses")
    );
    assert_eq!(responses.value()["code_execution"], true);

    let browser_search =
        ProviderOptions::typed(&siumai::providers::groq::tools::browser_search()).unwrap();
    assert_eq!(browser_search.value()["browser_search"], true);
    let remote_mcp = ProviderOptions::typed(&siumai::providers::groq::tools::responses_remote_mcp(
        siumai::providers::groq::tools::GroqRemoteMcpTool::new(
            "docs",
            "https://mcp.example.com/sse",
        )
        .with_require_approval(siumai::providers::groq::tools::GroqMcpApproval::Always),
    ))
    .unwrap();
    assert_eq!(
        remote_mcp.value()["remote_mcp_tools"][0]["server_label"],
        "docs"
    );
}

#[cfg(all(feature = "registry", feature = "xai"))]
#[test]
fn facade_exposes_xai_language_media_resources_and_typed_tools() {
    use siumai::core::{ApiModeId, Model, ModelFamily, ProviderOptions};
    use siumai::providers::xai::options::{XaiResponsesOptions, XaiResponsesReasoningEffort};
    use siumai::providers::xai::tools;
    use siumai::providers::xai::{XaiCredential, XaiProvider};
    use siumai::registry::ProviderRegistrationSource;

    fn assert_registration_source<T: ProviderRegistrationSource>() {}

    assert_registration_source::<XaiProvider>();

    let provider = XaiProvider::builder(XaiCredential::api_key("test-key"))
        .build()
        .unwrap();
    assert_eq!(
        provider
            .language("future-grok-model")
            .unwrap()
            .descriptor()
            .api_mode(),
        Some("responses")
    );
    assert_eq!(
        provider
            .chat_completions("future-grok-model")
            .unwrap()
            .descriptor()
            .api_mode(),
        Some("chat-completions")
    );
    assert_eq!(
        provider
            .registration()
            .api_mode(ModelFamily::Language)
            .map(ApiModeId::as_str),
        Some("responses")
    );
    assert_eq!(
        provider
            .default_image_model()
            .unwrap()
            .descriptor()
            .api_mode(),
        Some("image-generations")
    );
    assert_eq!(provider.speech().descriptor().api_mode(), Some("tts"));
    assert_eq!(
        provider.transcription().descriptor().api_mode(),
        Some("stt")
    );
    assert_eq!(provider.support_manifest().native_claims().len(), 2);
    let _files = provider.files();
    let _video_jobs = provider.video_jobs();

    let options = ProviderOptions::typed(
        &XaiResponsesOptions::new()
            .with_reasoning_effort(XaiResponsesReasoningEffort::High)
            .with_prompt_cache_key("stable-prompt")
            .with_native_tool(tools::web_search()),
    )
    .unwrap();
    assert_eq!(options.namespace().as_str(), "xai");
    assert_eq!(options.api_mode().map(ApiModeId::as_str), Some("responses"));
    assert_eq!(options.value()["promptCacheKey"], "stable-prompt");
    assert_eq!(options.value()["native_tools"][0]["type"], "web_search");
}

#[cfg(feature = "moonshotai")]
#[test]
fn facade_exposes_moonshotai_provider_and_typed_kimi_options() {
    use siumai::core::ProviderOptions;
    use siumai::providers::moonshotai::models;
    use siumai::providers::moonshotai::options::{KimiLanguageOptions, KimiReasoningEffort};
    use siumai::providers::moonshotai::{
        KimiAssistantPartial, KimiFileUploadPurpose, MoonshotCredential, MoonshotProvider,
    };

    let provider = MoonshotProvider::builder(MoonshotCredential::unauthenticated())
        .build()
        .unwrap();
    assert_eq!(provider.provider_id().as_str(), "moonshotai");
    assert_eq!(models::CHAT, models::KIMI_K3);
    assert_eq!(
        provider.profile().verified_claims().unwrap()[0]
            .evidence()
            .source()
            .as_str(),
        models::OFFICIAL_SOURCE
    );
    assert_eq!(
        provider
            .profile()
            .catalog()
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .evidence()
            .source()
            .as_str(),
        models::MODEL_SOURCE
    );

    let options = ProviderOptions::typed(
        &KimiLanguageOptions::new().with_reasoning_effort(KimiReasoningEffort::High),
    )
    .unwrap();
    assert_eq!(options.namespace().as_str(), "moonshotai");
    assert_eq!(options.value()["reasoning_effort"], "high");
    let _partial = KimiAssistantPartial::new();
    assert_eq!(KimiFileUploadPurpose::FileExtract.as_str(), "file-extract");
    assert_eq!(provider.support_manifest().native_claims().len(), 1);
}

#[cfg(all(feature = "moonshotai", feature = "registry"))]
#[test]
fn facade_registers_moonshotai_without_compatible_engine_ownership() {
    use siumai::providers::moonshotai::{MoonshotCredential, MoonshotProvider};
    use siumai::registry::{Registry, RegistryBuilderExt};

    let provider = MoonshotProvider::builder(MoonshotCredential::unauthenticated())
        .build()
        .unwrap();
    let mut builder = Registry::builder();
    builder.register_provider("kimi", &provider).unwrap();
    let registry = builder.build().unwrap();

    assert_eq!(
        registry
            .language_model("kimi:kimi-k4-future")
            .unwrap()
            .provider_id()
            .as_str(),
        "moonshotai"
    );
}

#[cfg(feature = "alibaba")]
#[test]
fn facade_exposes_alibaba_without_a_dashscope_route_surface() {
    use siumai::core::ProviderOptions;
    use siumai::providers::alibaba::experimental::{
        AlibabaVideoProviderBuilderExt, AlibabaVideoProviderExt, WAN_2_7_T2V,
    };
    use siumai::providers::alibaba::options::{
        AlibabaMessagesOptions, AlibabaMessagesThinking, AlibabaReasoningEffort,
        AlibabaResponsesOptions,
    };
    use siumai::providers::alibaba::{AlibabaCredential, AlibabaProvider};

    let provider = AlibabaProvider::builder(AlibabaCredential::api_key("test-key"))
        .with_legacy_singapore_language()
        .with_legacy_singapore_messages()
        .with_legacy_singapore_embedding()
        .with_legacy_singapore_video()
        .build()
        .unwrap();
    assert_eq!(
        provider
            .registration()
            .unwrap()
            .api_mode(ModelFamily::Language)
            .map(siumai::core::ApiModeId::as_str),
        Some("responses")
    );
    assert_eq!(
        provider
            .responses("future-qwen-model")
            .unwrap()
            .provider_id()
            .as_str(),
        "alibaba"
    );
    assert_eq!(
        provider
            .messages("future-qwen-model")
            .unwrap()
            .provider_id()
            .as_str(),
        "alibaba"
    );
    assert_eq!(
        provider
            .chat_completions("future-qwen-model")
            .unwrap()
            .provider_id()
            .as_str(),
        "alibaba"
    );
    ProviderOptions::typed(
        &AlibabaResponsesOptions::new().with_reasoning_effort(AlibabaReasoningEffort::Minimal),
    )
    .unwrap();
    AlibabaMessagesOptions::new()
        .with_thinking(AlibabaMessagesThinking::enabled(1_024))
        .provider_options()
        .unwrap();
    assert_eq!(
        provider.video(WAN_2_7_T2V).unwrap().model_id().as_str(),
        WAN_2_7_T2V
    );
}

#[cfg(all(feature = "alibaba", feature = "registry"))]
#[test]
fn facade_registers_alibaba_with_its_recommended_language_mode() {
    use siumai::providers::alibaba::{AlibabaCredential, AlibabaProvider};
    use siumai::registry::{Registry, RegistryBuilderExt};

    let provider = AlibabaProvider::builder(AlibabaCredential::api_key("test-key"))
        .with_legacy_singapore_language()
        .build()
        .unwrap();
    let mut builder = Registry::builder();
    builder.register_provider("alibaba", &provider).unwrap();
    let registry = builder.build().unwrap();
    assert_eq!(
        registry
            .language_model("alibaba:future-qwen-model")
            .unwrap()
            .provider_id()
            .as_str(),
        "alibaba"
    );
}

#[cfg(all(feature = "alibaba", feature = "registry"))]
#[test]
fn facade_rejects_provider_registration_without_a_portable_family() {
    use siumai::providers::alibaba::experimental::AlibabaVideoProviderBuilderExt;
    use siumai::providers::alibaba::{AlibabaCredential, AlibabaProvider};
    use siumai::registry::{RegisterProviderError, Registry, RegistryBuilderExt};

    let provider = AlibabaProvider::builder(AlibabaCredential::api_key("test-key"))
        .with_legacy_singapore_video()
        .build()
        .unwrap();
    let mut builder = Registry::builder();
    let error = builder
        .register_provider("alibaba-video", &provider)
        .unwrap_err();

    assert!(matches!(
        error,
        RegisterProviderError::NoPortableFamilyRegistration { .. }
    ));
}

#[cfg(feature = "volcengine")]
#[test]
fn facade_exposes_volcengine_provider_and_typed_ark_options() {
    use siumai::core::ProviderOptions;
    use siumai::providers::volcengine::options::{
        ArkCaching, ArkImageOptions, ArkMcpApproval, ArkMcpTool, ArkResponsesOptions,
        ArkResponsesTool,
    };
    use siumai::providers::volcengine::{VolcengineCredential, VolcengineProvider};

    let provider = VolcengineProvider::builder(VolcengineCredential::unauthenticated())
        .build()
        .unwrap();
    assert_eq!(provider.provider_id().as_str(), "volcengine");
    assert!(provider.profile().verified_claims().is_some());
    assert_eq!(
        provider
            .language("future-ark-deployment")
            .unwrap()
            .provider_id()
            .as_str(),
        "volcengine"
    );
    ProviderOptions::typed(&ArkResponsesOptions::new().with_caching(ArkCaching::disabled()))
        .unwrap();
    ProviderOptions::typed(&ArkImageOptions::new().with_watermark(false)).unwrap();
    let _mcp = ArkResponsesTool::remote_mcp(
        ArkMcpTool::new("docs", "https://mcp.example.test").with_approval(ArkMcpApproval::Always),
    );
    assert!(provider.registration().supports_family(ModelFamily::Image));
    assert_eq!(provider.support_manifest().native_claims().len(), 2);
}

#[cfg(all(feature = "volcengine", feature = "registry"))]
#[test]
fn facade_registers_volcengine_with_recommended_responses_mode() {
    use siumai::providers::volcengine::{VolcengineCredential, VolcengineProvider};
    use siumai::registry::{Registry, RegistryBuilderExt};

    let provider = VolcengineProvider::builder(VolcengineCredential::unauthenticated())
        .build()
        .unwrap();
    let mut builder = Registry::builder();
    builder.register_provider("ark", &provider).unwrap();
    let registry = builder.build().unwrap();

    assert_eq!(
        registry
            .language_model("ark:future-ark-deployment")
            .unwrap()
            .provider_id()
            .as_str(),
        "volcengine"
    );
}

#[cfg(all(feature = "registry", feature = "openai"))]
#[tokio::test]
async fn openai_direct_registry_and_language_facade_paths_share_one_wire_pipeline() {
    use serde_json::{Value, json};
    use siumai::core::{ReplayDomain, ReplayDomainId};
    use siumai::providers::openai::responses::OpenAiResponsesOptions;
    use siumai::providers::openai::{OpenAiCredential, OpenAiProvider};
    use siumai::registry::{Registry, RegistryBuilderExt};
    use siumai::transport::{
        EndpointConfig, ProviderHttpTransportSettings, TransportEvent, TransportObserver,
    };
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[derive(Default)]
    struct CountingObserver {
        attempts: AtomicUsize,
        completed: AtomicUsize,
    }

    impl TransportObserver for CountingObserver {
        fn observe(&self, event: &TransportEvent) {
            match event {
                TransportEvent::AttemptStarted { .. } => {
                    self.attempts.fetch_add(1, Ordering::SeqCst);
                }
                TransportEvent::AttemptLoopFinished { .. } => {
                    self.completed.fetch_add(1, Ordering::SeqCst);
                }
                _ => {}
            }
        }
    }

    fn request() -> LanguageRequest {
        LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")])
    }

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "resp_123",
            "created_at": 1,
            "model": "gpt-5.6-sol",
            "status": "completed",
            "output": [{
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "text": "hello back",
                    "annotations": []
                }]
            }]
        })))
        .expect(5)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chat_123",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-5.6-sol",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello back"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3
            }
        })))
        .expect(1)
        .mount(&server)
        .await;

    let observer = Arc::new(CountingObserver::default());
    let provider = OpenAiProvider::builder(OpenAiCredential::unauthenticated())
        .with_endpoint(EndpointConfig::local_explicit(format!("{}/v1", server.uri())).unwrap())
        .with_replay_domain(ReplayDomain::custom(
            ReplayDomainId::new("facade-openai-test").unwrap(),
        ))
        .with_http_transport_settings(
            ProviderHttpTransportSettings::default().with_observer(observer.clone()),
        )
        .build()
        .unwrap();
    let direct = provider.responses("gpt-5.6-sol").unwrap();
    let mut builder = Registry::builder();
    builder
        .register_provider("openai-responses", &provider)
        .unwrap()
        .register_named("openai-chat", provider.chat_completions_registration())
        .unwrap();
    let registry = builder.build().unwrap();
    let erased = registry
        .language_model("openai-responses:gpt-5.6-sol")
        .unwrap();
    let chat = registry.language_model("openai-chat:gpt-5.6-sol").unwrap();
    assert_eq!(erased.provider_id(), chat.provider_id());
    assert_eq!(erased.descriptor().api_mode(), Some("responses"));
    assert_eq!(chat.descriptor().api_mode(), Some("chat-completions"));

    let direct_response = direct
        .generate(request(), CallOptions::default())
        .await
        .unwrap();
    let erased_response = erased
        .generate(request(), CallOptions::default())
        .await
        .unwrap();
    let helper_response = language::generate(&direct, request()).await.unwrap();
    let typed_direct_response = language::call(&direct, request())
        .with_provider_options(&OpenAiResponsesOptions {
            instructions: Some("typed direct".to_string()),
            ..OpenAiResponsesOptions::default()
        })
        .unwrap()
        .generate()
        .await
        .unwrap();
    let typed_erased_response = language::call(erased.as_ref(), request())
        .with_provider_options(&OpenAiResponsesOptions {
            instructions: Some("typed Registry".to_string()),
            ..OpenAiResponsesOptions::default()
        })
        .unwrap()
        .generate()
        .await
        .unwrap();
    let chat_response = chat
        .generate(request(), CallOptions::default())
        .await
        .unwrap();

    assert_eq!(direct_response, erased_response);
    assert_eq!(erased_response, helper_response);
    assert_eq!(helper_response, typed_direct_response);
    assert_eq!(typed_direct_response, typed_erased_response);
    assert!(matches!(
        &chat_response.content()[0],
        ContentPart::Text { text } if text == "hello back"
    ));
    assert_eq!(observer.attempts.load(Ordering::SeqCst), 6);
    assert_eq!(observer.completed.load(Ordering::SeqCst), 6);

    let requests = server.received_requests().await.unwrap();
    let bodies = requests
        .iter()
        .filter(|request| request.url.path() == "/v1/responses")
        .map(|request| serde_json::from_slice::<Value>(&request.body).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(bodies[0], bodies[1]);
    assert_eq!(bodies[1], bodies[2]);
    assert_eq!(bodies[3]["instructions"], "typed direct");
    assert_eq!(bodies[4]["instructions"], "typed Registry");
}

#[cfg(feature = "openai-realtime")]
#[test]
fn facade_exposes_realtime_as_typed_provider_sessions() {
    use siumai::providers::openai::experimental::realtime::{
        OPENAI_REALTIME_MODEL, OPENAI_REALTIME_TRANSLATION_MODEL, OpenAiRealtimeClientEvent,
        OpenAiTranslationClientEvent,
    };
    use siumai::providers::openai::{OpenAiCredential, OpenAiProvider};

    let provider = OpenAiProvider::builder(OpenAiCredential::api_key("test-key"))
        .build()
        .unwrap();
    let conversation = provider.realtime(OPENAI_REALTIME_MODEL).unwrap();
    let translation = provider
        .translation(OPENAI_REALTIME_TRANSLATION_MODEL)
        .unwrap();

    assert_eq!(conversation.model(), OPENAI_REALTIME_MODEL);
    assert_eq!(translation.model(), OPENAI_REALTIME_TRANSLATION_MODEL);
    assert!(conversation.endpoint().is_official());
    assert!(translation.endpoint().is_official());
    assert_eq!(
        OpenAiRealtimeClientEvent::ResponseCreate {
            event_id: None,
            response: None,
        }
        .event_type(),
        "response.create"
    );
    assert_eq!(
        OpenAiTranslationClientEvent::SessionClose { event_id: None }.event_type(),
        "session.close"
    );
}

#[cfg(feature = "openai-responses-websocket")]
#[test]
fn facade_exposes_responses_websocket_as_an_experimental_native_session() {
    use siumai::providers::openai::experimental::responses_websocket::{
        OPENAI_RESPONSES_WEBSOCKET_URL, OpenAiResponsesWarmUpFrame, OpenAiResponsesWarmUpOutcome,
        OpenAiResponsesWebSocketConfig, OpenAiResponsesWebSocketConfigError,
        OpenAiResponsesWebSocketEvent, OpenAiResponsesWebSocketSession,
        OpenAiResponsesWebSocketSubmissionState, OpenAiResponsesWebSocketTurn,
        OpenAiResponsesWebSocketTurnKind,
    };
    use siumai::providers::openai::{OpenAiCredential, OpenAiProvider};

    fn assert_public_type<T>() {}

    assert_public_type::<OpenAiResponsesWebSocketConfig>();
    assert_public_type::<OpenAiResponsesWebSocketConfigError>();
    assert_public_type::<OpenAiResponsesWebSocketSession>();
    assert_public_type::<OpenAiResponsesWebSocketTurn>();
    assert_public_type::<OpenAiResponsesWebSocketEvent>();
    assert_public_type::<OpenAiResponsesWarmUpFrame>();
    assert_public_type::<OpenAiResponsesWarmUpOutcome>();

    let _connect = OpenAiResponsesWebSocketConfig::connect;
    let _generate = OpenAiResponsesWebSocketSession::generate;
    let _warm_up = OpenAiResponsesWebSocketSession::warm_up;
    let _kind = OpenAiResponsesWebSocketTurn::kind;
    let _submission_state = OpenAiResponsesWebSocketTurn::submission_state;

    let provider = OpenAiProvider::builder(OpenAiCredential::api_key("test-key"))
        .build()
        .unwrap();
    let config = provider.responses("gpt-5.6").unwrap().websocket().unwrap();

    assert_eq!(
        config.endpoint().expose_url().as_str(),
        OPENAI_RESPONSES_WEBSOCKET_URL
    );
    assert_ne!(
        OpenAiResponsesWebSocketTurnKind::Generate,
        OpenAiResponsesWebSocketTurnKind::WarmUp
    );
    assert_ne!(
        OpenAiResponsesWebSocketSubmissionState::NotSubmitted,
        OpenAiResponsesWebSocketSubmissionState::Indeterminate
    );
}
