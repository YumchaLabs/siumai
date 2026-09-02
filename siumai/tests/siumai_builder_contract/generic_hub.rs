use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use futures::StreamExt;
use siumai::core::{ProviderInstanceId, ProviderScope, RouteId};
use siumai::{
    CallOptions, ContentPart, EmbeddingLimits, EmbeddingModel, EmbeddingModelProvider,
    EmbeddingRequest, EmbeddingResponse, Error, ImageArtifact, ImageLimits, ImageModel,
    ImageModelProvider, ImageRequest, ImageResponse, LanguageCallError, LanguageCompletionReason,
    LanguageModel, LanguageModelProvider, LanguageRequest, LanguageResponse, LanguageStream,
    LanguageStreamEvent, MediaData, Message, Model, ModelDescriptor, ModelFamily, ModelId,
    ModelLookupError, Provider, ProviderId, RerankCandidate, RerankLimits, RerankModel,
    RerankModelProvider, RerankRequest, RerankResponse, RerankResult, ResponseMetadata, Siumai,
    SpeechLimits, SpeechModel, SpeechModelProvider, SpeechRequest, SpeechResponse, StreamTerminal,
    TranscriptionLimits, TranscriptionModel, TranscriptionModelProvider, TranscriptionRequest,
    TranscriptionResponse, TypedProviderOptions, Usage,
};

const PROVIDER_CANARY: &str = "provider-secret-canary";
const MODEL_CANARY: &str = "model-secret-canary";

#[derive(Default)]
struct Calls {
    language_generate: AtomicUsize,
    language_stream: AtomicUsize,
    embedding: AtomicUsize,
    rerank: AtomicUsize,
    image: AtomicUsize,
    speech: AtomicUsize,
    transcription: AtomicUsize,
}

struct FakeProvider {
    provider_id: ProviderId,
    instance_id: ProviderInstanceId,
    calls: Arc<Calls>,
}

impl FakeProvider {
    fn new() -> Self {
        Self {
            provider_id: ProviderId::new("fake").expect("the static fake provider ID is valid"),
            instance_id: ProviderInstanceId::new(),
            calls: Arc::new(Calls::default()),
        }
    }

    fn bind(&self, model: ModelId, family: ModelFamily) -> FakeModel {
        FakeModel {
            descriptor: ModelDescriptor::from_scope(
                ProviderScope::new(self.provider_id.clone()),
                model,
                family,
                self.instance_id.clone(),
            ),
            route: RouteId::new("fake-route").expect("the static fake route ID is valid"),
            calls: self.calls.clone(),
        }
    }
}

impl fmt::Debug for FakeProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(PROVIDER_CANARY)
    }
}

impl Provider for FakeProvider {
    fn provider_id(&self) -> &ProviderId {
        &self.provider_id
    }
}

struct FakeModel {
    descriptor: ModelDescriptor,
    route: RouteId,
    calls: Arc<Calls>,
}

impl fmt::Debug for FakeModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(MODEL_CANARY)
    }
}

impl Model for FakeModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn route_id(&self) -> Option<&RouteId> {
        Some(&self.route)
    }
}

fn retry_marker(options: &CallOptions) -> u8 {
    options.retry().maximum_attempts().unwrap_or_default()
}

fn language_response(request: &LanguageRequest, options: &CallOptions) -> LanguageResponse {
    LanguageResponse::completed(
        vec![ContentPart::Text {
            text: format!(
                "messages={};retry={}",
                request.messages.len(),
                retry_marker(options)
            ),
        }],
        LanguageCompletionReason::Stop,
        Usage::default(),
    )
    .expect("the fake language response is valid")
}

#[async_trait]
impl LanguageModel for FakeModel {
    async fn generate(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageResponse, LanguageCallError> {
        self.calls.language_generate.fetch_add(1, Ordering::SeqCst);
        Ok(language_response(&request, &options))
    }

    async fn stream(
        &self,
        request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        self.calls.language_stream.fetch_add(1, Ordering::SeqCst);
        let cancellation = options.cancellation().clone();
        let response = language_response(&request, &options);
        Ok(siumai::core::stream::established_stream(
            cancellation,
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

#[async_trait]
impl EmbeddingModel for FakeModel {
    fn limits(&self) -> EmbeddingLimits {
        EmbeddingLimits {
            max_inputs: Some(3),
            max_input_tokens: Some(100),
        }
    }

    async fn embed(
        &self,
        request: EmbeddingRequest,
        options: CallOptions,
    ) -> Result<EmbeddingResponse, Error> {
        self.calls.embedding.fetch_add(1, Ordering::SeqCst);
        let response = EmbeddingResponse {
            embeddings: request
                .inputs()
                .iter()
                .map(|_| vec![1.0, f32::from(retry_marker(&options))])
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
impl RerankModel for FakeModel {
    fn limits(&self) -> RerankLimits {
        RerankLimits {
            max_candidates: Some(4),
        }
    }

    async fn rerank(
        &self,
        request: RerankRequest,
        options: CallOptions,
    ) -> Result<RerankResponse, Error> {
        self.calls.rerank.fetch_add(1, Ordering::SeqCst);
        let count = request.top_n().unwrap_or(request.candidates().len());
        let response = RerankResponse {
            results: request
                .candidates()
                .iter()
                .take(count)
                .enumerate()
                .map(|(index, candidate)| {
                    RerankResult::new(
                        index,
                        1.0 - (index as f64 * 0.1),
                        candidate.id().map(ToString::to_string),
                    )
                    .expect("the fake rerank result is valid")
                })
                .collect(),
            metadata: ResponseMetadata::default(),
            usage: Usage::default(),
            warnings: Vec::new(),
            provider: BTreeMap::from([(
                "retry".to_string(),
                serde_json::json!(retry_marker(&options)),
            )]),
        };
        response.validate(&request)?;
        Ok(response)
    }
}

#[async_trait]
impl ImageModel for FakeModel {
    fn limits(&self) -> ImageLimits {
        ImageLimits {
            max_outputs_per_call: Some(2),
        }
    }

    async fn generate_image(
        &self,
        request: ImageRequest,
        options: CallOptions,
    ) -> Result<ImageResponse, Error> {
        self.calls.image.fetch_add(1, Ordering::SeqCst);
        let response = ImageResponse {
            images: (0..request.count())
                .map(|_| ImageArtifact {
                    media_type: "image/png".to_string(),
                    data: MediaData::Bytes(vec![1_u8].into()),
                    revised_prompt: Some(format!(
                        "{}:retry={}",
                        request.prompt(),
                        retry_marker(&options)
                    )),
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
impl SpeechModel for FakeModel {
    fn limits(&self) -> SpeechLimits {
        SpeechLimits {
            max_text_bytes: Some(100),
            max_text_chars: Some(50),
        }
    }

    async fn synthesize(
        &self,
        request: SpeechRequest,
        options: CallOptions,
    ) -> Result<SpeechResponse, Error> {
        self.calls.speech.fetch_add(1, Ordering::SeqCst);
        let response = SpeechResponse {
            media_type: "audio/wav".to_string(),
            audio: format!("{}:retry={}", request.text(), retry_marker(&options))
                .into_bytes()
                .into(),
            duration_seconds: Some(1.0),
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
impl TranscriptionModel for FakeModel {
    fn limits(&self) -> TranscriptionLimits {
        TranscriptionLimits {
            max_audio_bytes: Some(128),
            max_duration_seconds: Some(12.5),
        }
    }

    async fn transcribe(
        &self,
        request: TranscriptionRequest,
        options: CallOptions,
    ) -> Result<TranscriptionResponse, Error> {
        self.calls.transcription.fetch_add(1, Ordering::SeqCst);
        let response = TranscriptionResponse {
            text: format!(
                "{}:{}:retry={}",
                request.media_type(),
                request.audio().len(),
                retry_marker(&options)
            ),
            language: request.language().map(ToString::to_string),
            confidence: Some(0.99),
            duration_seconds: Some(1.0),
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

macro_rules! impl_fake_provider_family {
    ($trait_name:ident, $method:ident, $family:expr) => {
        impl $trait_name for FakeProvider {
            type Model = FakeModel;

            fn $method(&self, model: ModelId) -> Result<Self::Model, ModelLookupError> {
                Ok(self.bind(model, $family))
            }
        }
    };
}

impl_fake_provider_family!(LanguageModelProvider, language_model, ModelFamily::Language);
impl_fake_provider_family!(
    EmbeddingModelProvider,
    embedding_model,
    ModelFamily::Embedding
);
impl_fake_provider_family!(RerankModelProvider, rerank_model, ModelFamily::Rerank);
impl_fake_provider_family!(ImageModelProvider, image_model, ModelFamily::Image);
impl_fake_provider_family!(SpeechModelProvider, speech_model, ModelFamily::Speech);
impl_fake_provider_family!(
    TranscriptionModelProvider,
    transcription_model,
    ModelFamily::Transcription
);

fn assert_clone<T: Clone>(value: &T) {
    drop(value.clone());
}

async fn completed_stream_response(mut stream: LanguageStream) -> LanguageResponse {
    while let Some(event) = stream.next().await {
        if let LanguageStreamEvent::Terminal(StreamTerminal::Completed { response }) = event {
            assert!(
                stream.next().await.is_none(),
                "the established stream must stop after its terminal event"
            );
            return *response;
        }
    }
    panic!("the fake stream must emit one completed terminal response")
}

#[tokio::test]
async fn family_clients_preserve_identity_limits_lifetimes_and_dispatch()
-> Result<(), Box<dyn std::error::Error>> {
    let hub = Siumai::from_provider(FakeProvider::new());
    let hub_clone = hub.clone();
    assert_clone(&hub);
    assert!(std::ptr::eq(hub.provider(), hub_clone.provider()));

    let language = hub.language("fake-language")?;
    let second_language = hub.language("fake-language-two")?;
    let embedding = hub.embedding("fake-embedding")?;
    let rerank = hub.rerank("fake-rerank")?;
    let image = hub.image("fake-image")?;
    let speech = hub.speech("fake-speech")?;
    let transcription = hub.transcription("fake-transcription")?;

    assert_clone(&language);
    let language_clone = language.clone();
    assert!(std::ptr::eq(language.model(), language_clone.model()));
    assert!(std::ptr::eq(hub.provider(), language.provider()));
    assert!(std::ptr::eq(hub.provider(), second_language.provider()));
    assert!(std::ptr::eq(hub.provider(), embedding.provider()));
    assert!(std::ptr::eq(hub.provider(), rerank.provider()));
    assert!(std::ptr::eq(hub.provider(), image.provider()));
    assert!(std::ptr::eq(hub.provider(), speech.provider()));
    assert!(std::ptr::eq(hub.provider(), transcription.provider()));

    assert!(std::ptr::eq(
        Model::descriptor(&language),
        language.model().descriptor()
    ));
    assert_eq!(Model::route_id(&language), language.model().route_id());
    assert_eq!(
        language.descriptor().instance_id(),
        embedding.descriptor().instance_id()
    );
    assert_eq!(language.family(), ModelFamily::Language);
    assert_eq!(embedding.family(), ModelFamily::Embedding);
    assert_eq!(rerank.family(), ModelFamily::Rerank);
    assert_eq!(image.family(), ModelFamily::Image);
    assert_eq!(speech.family(), ModelFamily::Speech);
    assert_eq!(transcription.family(), ModelFamily::Transcription);
    assert_ne!(language.model_id(), second_language.model_id());

    assert_eq!(
        EmbeddingModel::limits(&embedding),
        EmbeddingLimits {
            max_inputs: Some(3),
            max_input_tokens: Some(100),
        }
    );
    assert_eq!(
        RerankModel::limits(&rerank),
        RerankLimits {
            max_candidates: Some(4),
        }
    );
    assert_eq!(
        ImageModel::limits(&image),
        ImageLimits {
            max_outputs_per_call: Some(2),
        }
    );
    assert_eq!(
        SpeechModel::limits(&speech),
        SpeechLimits {
            max_text_bytes: Some(100),
            max_text_chars: Some(50),
        }
    );
    assert_eq!(
        TranscriptionModel::limits(&transcription),
        TranscriptionLimits {
            max_audio_bytes: Some(128),
            max_duration_seconds: Some(12.5),
        }
    );

    let hub_debug = format!("{hub:?}");
    let language_debug = format!("{language:?}");
    assert!(!hub_debug.contains(PROVIDER_CANARY));
    assert!(!language_debug.contains(PROVIDER_CANARY));
    assert!(!language_debug.contains(MODEL_CANARY));

    drop(hub_clone);
    drop(hub);
    assert_eq!(language.provider().provider_id().as_str(), "fake");

    let response = LanguageModel::generate(
        &language,
        LanguageRequest::new(Vec::new()),
        CallOptions::default(),
    )
    .await?;
    assert_eq!(
        response.output_text().as_deref(),
        Some("messages=0;retry=0")
    );
    let stream = LanguageModel::stream(
        &language,
        LanguageRequest::new(Vec::new()),
        CallOptions::default(),
    )
    .await?;
    let stream_response = completed_stream_response(stream).await;
    assert_eq!(
        stream_response.output_text().as_deref(),
        Some("messages=0;retry=0")
    );
    let _ = EmbeddingModel::embed(
        &embedding,
        EmbeddingRequest::single("embedding input")?,
        CallOptions::default(),
    )
    .await?;
    let _ = RerankModel::rerank(
        &rerank,
        RerankRequest::new("query", vec![RerankCandidate::new("candidate")?])?,
        CallOptions::default(),
    )
    .await?;
    let _ = ImageModel::generate_image(
        &image,
        ImageRequest::new("image prompt")?,
        CallOptions::default(),
    )
    .await?;
    let _ = SpeechModel::synthesize(
        &speech,
        SpeechRequest::new("speech input")?,
        CallOptions::default(),
    )
    .await?;
    let _ = TranscriptionModel::transcribe(
        &transcription,
        TranscriptionRequest::new(vec![1_u8], "audio/wav")?,
        CallOptions::default(),
    )
    .await?;

    let calls = &language.provider().calls;
    assert_eq!(calls.language_generate.load(Ordering::SeqCst), 1);
    assert_eq!(calls.language_stream.load(Ordering::SeqCst), 1);
    assert_eq!(calls.embedding.load(Ordering::SeqCst), 1);
    assert_eq!(calls.rerank.load(Ordering::SeqCst), 1);
    assert_eq!(calls.image.load(Ordering::SeqCst), 1);
    assert_eq!(calls.speech.load(Ordering::SeqCst), 1);
    assert_eq!(calls.transcription.load(Ordering::SeqCst), 1);

    Ok(())
}

#[derive(serde::Serialize)]
struct InstanceBoundLanguageOptions {
    value: &'static str,
}

impl TypedProviderOptions for InstanceBoundLanguageOptions {
    const NAMESPACE: &'static str = "fake";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
}

#[tokio::test]
async fn family_client_methods_reuse_root_calls_and_keep_low_level_ufcs_available()
-> Result<(), Box<dyn std::error::Error>> {
    let hub = Siumai::from_provider(FakeProvider::new());
    let language = hub.language("fake-language")?;
    let embedding = hub.embedding("fake-embedding")?;
    let rerank = hub.rerank("fake-rerank")?;
    let image = hub.image("fake-image")?;
    let speech = hub.speech("fake-speech")?;
    let transcription = hub.transcription("fake-transcription")?;

    let language_request = LanguageRequest::new(vec![Message::user("hello")]);
    assert_eq!(language.call("hello").request(), &language_request);
    assert_eq!(
        language.call(String::from("hello")).request(),
        &language_request
    );
    assert_eq!(
        language.call(Message::user("hello")).request(),
        &language_request
    );
    assert_eq!(
        language.call(vec![Message::user("hello")]).request(),
        &language_request
    );
    assert_eq!(
        language.call(language_request.clone()).request(),
        &language_request
    );

    let language_method_response = language.generate("hello").await?;
    let language_root_response = siumai::language::generate(&language, "hello").await?;
    assert_eq!(language_method_response, language_root_response);
    let language_advanced_call = language
        .call("hello")
        .with_options(CallOptions::default().with_max_attempts(2)?)?;
    assert_eq!(
        language_advanced_call
            .base_options()
            .retry()
            .maximum_attempts(),
        Some(2)
    );
    let language_advanced_response = language_advanced_call.generate().await?;
    assert_eq!(
        language_advanced_response.output_text().as_deref(),
        Some("messages=1;retry=2")
    );

    let stream_method_response = completed_stream_response(language.stream("hello").await?).await;
    let stream_root_response =
        completed_stream_response(siumai::language::stream(&language, "hello").await?).await;
    assert_eq!(stream_method_response, stream_root_response);

    let embedding_request = EmbeddingRequest::single("embedding input")?;
    let embedding_method_response = embedding.embed(embedding_request.clone()).await?;
    let embedding_root_response =
        siumai::embedding::embed(&embedding, embedding_request.clone()).await?;
    assert_eq!(embedding_method_response, embedding_root_response);
    assert_eq!(
        embedding.call(embedding_request.clone()).request(),
        &embedding_request
    );
    let embedding_advanced_response = embedding
        .call(embedding_request)
        .with_options(CallOptions::default().with_max_attempts(2)?)?
        .embed()
        .await?;
    assert_eq!(embedding_advanced_response.embeddings[0][1], 2.0);

    let rerank_request = RerankRequest::new("query", vec![RerankCandidate::new("candidate")?])?;
    let rerank_method_response = rerank.rerank(rerank_request.clone()).await?;
    let rerank_root_response = siumai::rerank::rerank(&rerank, rerank_request.clone()).await?;
    assert_eq!(rerank_method_response, rerank_root_response);
    assert_eq!(
        rerank.call(rerank_request.clone()).request(),
        &rerank_request
    );
    let rerank_advanced_response = rerank
        .call(rerank_request)
        .with_options(CallOptions::default().with_max_attempts(2)?)?
        .rerank()
        .await?;
    assert_eq!(rerank_advanced_response.provider["retry"], 2);

    let image_request = ImageRequest::new("image prompt")?;
    let image_method_response = image.generate(image_request.clone()).await?;
    let image_root_response = siumai::image::generate(&image, image_request.clone()).await?;
    assert_eq!(image_method_response, image_root_response);
    assert_eq!(image.call(image_request.clone()).request(), &image_request);
    let image_advanced_response = image
        .call(image_request)
        .with_options(CallOptions::default().with_max_attempts(2)?)?
        .generate()
        .await?;
    assert_eq!(
        image_advanced_response.images[0].revised_prompt.as_deref(),
        Some("image prompt:retry=2")
    );

    let speech_request = SpeechRequest::new("speech input")?;
    let speech_method_response = speech.synthesize(speech_request.clone()).await?;
    let speech_root_response = siumai::speech::synthesize(&speech, speech_request.clone()).await?;
    assert_eq!(speech_method_response, speech_root_response);
    assert_eq!(
        speech.call(speech_request.clone()).request(),
        &speech_request
    );
    let speech_advanced_response = speech
        .call(speech_request)
        .with_options(CallOptions::default().with_max_attempts(2)?)?
        .synthesize()
        .await?;
    assert_eq!(
        speech_advanced_response.audio.as_ref(),
        b"speech input:retry=2"
    );

    let transcription_request = TranscriptionRequest::new(vec![1_u8], "audio/wav")?;
    let transcription_method_response = transcription
        .transcribe(transcription_request.clone())
        .await?;
    let transcription_root_response =
        siumai::transcription::transcribe(&transcription, transcription_request.clone()).await?;
    assert_eq!(transcription_method_response, transcription_root_response);
    assert_eq!(
        transcription.call(transcription_request.clone()).request(),
        &transcription_request
    );
    let transcription_advanced_response = transcription
        .call(transcription_request)
        .with_options(CallOptions::default().with_max_attempts(2)?)?
        .transcribe()
        .await?;
    assert_eq!(transcription_advanced_response.text, "audio/wav:1:retry=2");

    let foreign_hub = Siumai::from_provider(FakeProvider::new());
    let foreign_language = foreign_hub.language("fake-language")?;
    let foreign_options = CallOptions::default().with_provider_options_for(
        foreign_language.model(),
        &InstanceBoundLanguageOptions { value: "foreign" },
    )?;
    let calls_before_mismatch = language
        .provider()
        .calls
        .language_generate
        .load(Ordering::SeqCst);
    let mismatch = match language.call("hello").with_options(foreign_options) {
        Ok(_) => panic!("foreign configured-instance options must fail synchronously"),
        Err(error) => error,
    };
    assert!(matches!(
        mismatch,
        siumai::ProviderOptionError::ExactTargetMismatch { .. }
    ));
    assert_eq!(
        language
            .provider()
            .calls
            .language_generate
            .load(Ordering::SeqCst),
        calls_before_mismatch
    );

    let ufcs_response =
        LanguageModel::generate(&language, language_request, CallOptions::default()).await?;
    assert_eq!(
        ufcs_response.output_text().as_deref(),
        Some("messages=1;retry=0")
    );

    let calls = &language.provider().calls;
    assert_eq!(calls.language_generate.load(Ordering::SeqCst), 4);
    assert_eq!(calls.language_stream.load(Ordering::SeqCst), 2);
    assert_eq!(calls.embedding.load(Ordering::SeqCst), 3);
    assert_eq!(calls.rerank.load(Ordering::SeqCst), 3);
    assert_eq!(calls.image.load(Ordering::SeqCst), 3);
    assert_eq!(calls.speech.load(Ordering::SeqCst), 3);
    assert_eq!(calls.transcription.load(Ordering::SeqCst), 3);

    Ok(())
}

#[cfg(feature = "registry")]
#[tokio::test]
async fn direct_client_and_registry_model_share_the_root_language_seam_and_exact_identity()
-> Result<(), Box<dyn std::error::Error>> {
    use siumai::core::ProviderRegistration;
    use siumai::registry::Registry;

    async fn invoke<M>(model: &M) -> Result<LanguageResponse, LanguageCallError>
    where
        M: LanguageModel + ?Sized,
    {
        siumai::language::generate(model, "hello").await
    }

    let hub = Siumai::from_provider(FakeProvider::new());
    let direct = hub.language("shared-language")?;
    let scope = Arc::new(direct.descriptor().scope().clone());
    let instance_id = direct.descriptor().instance_id().clone();
    let calls = direct.provider().calls.clone();
    let inner_route =
        RouteId::new("registry-inner").expect("the static Registry route ID is valid");
    let registration = ProviderRegistration::from_language(
        scope.clone(),
        Arc::new(move |model| {
            Ok(Arc::new(FakeModel {
                descriptor: ModelDescriptor::from_scope(
                    scope.clone(),
                    model,
                    ModelFamily::Language,
                    instance_id.clone(),
                ),
                route: inner_route.clone(),
                calls: calls.clone(),
            }) as Arc<dyn LanguageModel>)
        }),
    );
    let mut registry = Registry::builder();
    registry
        .register_named("primary", registration.clone())?
        .register_named("secondary", registration)?;
    let registry = registry.build()?;
    let primary = registry.language_model("primary:shared-language")?;
    let secondary = registry.language_model("secondary:shared-language")?;

    assert_eq!(direct.descriptor().scope(), primary.descriptor().scope());
    assert_eq!(direct.model_id(), primary.model_id());
    assert_eq!(direct.family(), primary.family());
    assert_eq!(direct.family(), ModelFamily::Language);
    assert_eq!(
        direct.descriptor().instance_id(),
        primary.descriptor().instance_id()
    );
    assert_eq!(primary.route_id().map(RouteId::as_str), Some("primary"));
    assert_eq!(secondary.route_id().map(RouteId::as_str), Some("secondary"));

    let direct_response = invoke(&direct).await?;
    let registry_response = invoke(primary.as_ref()).await?;
    assert_eq!(direct_response, registry_response);
    assert_eq!(
        direct
            .provider()
            .calls
            .language_generate
            .load(Ordering::SeqCst),
        2
    );

    let matching = CallOptions::default().with_provider_options_for(
        primary.as_ref(),
        &InstanceBoundLanguageOptions { value: "primary" },
    )?;
    let matching_call = siumai::language::call(primary.as_ref(), "hello").with_options(matching)?;
    assert_eq!(
        matching_call
            .base_options()
            .provider_options_for(primary.as_ref())?
            .typed()
            .count(),
        1
    );

    let foreign_route = CallOptions::default().with_provider_options_for(
        secondary.as_ref(),
        &InstanceBoundLanguageOptions { value: "secondary" },
    )?;
    let mismatch =
        match siumai::language::call(primary.as_ref(), "hello").with_options(foreign_route) {
            Ok(_) => panic!("options bound to another Registry route must fail synchronously"),
            Err(error) => error,
        };
    assert!(matches!(
        mismatch,
        siumai::ProviderOptionError::ExactTargetMismatch { .. }
    ));

    Ok(())
}
