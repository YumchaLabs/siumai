use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use futures::{StreamExt, stream};
use serde::Serialize;
use siumai_core::language::{FinishReason, MediaData, Warning};
use siumai_core::stream::{StreamTerminal, established_stream};
use siumai_core::{
    ApiModeId, CallOptions, EmbeddingInput, EmbeddingModel, EmbeddingRequest, EmbeddingResponse,
    Error, ImageArtifact, ImageModel, ImageRequest, ImageResponse, LanguageModel, LanguageRequest,
    LanguageResponse, LanguageStream, LanguageStreamEvent, Message, MessageRole, Model,
    ModelDescriptor, ModelFamily, ModelId, ModelPolicy, ModelPolicyContext, ModelPolicyDecision,
    ProtocolId, ProviderId, ProviderOptions, ProviderRegistration, RerankCandidate, RerankModel,
    RerankRequest, RerankResponse, RerankResult, SpeechModel, SpeechRequest, SpeechResponse,
    TranscriptionModel, TranscriptionRequest, TranscriptionResponse, TypedProviderOptions, Usage,
};

fn descriptor(family: ModelFamily, model: &str) -> ModelDescriptor {
    ModelDescriptor::new(
        ProviderId::new("custom").unwrap(),
        ModelId::new(model).unwrap(),
        family,
    )
    .with_protocol(ProtocolId::new("test").unwrap())
}

fn language_response(model: &str) -> LanguageResponse {
    LanguageResponse {
        id: Some("response-1".to_string()),
        model: Some(ModelId::new(model).unwrap()),
        content: vec![siumai_core::ContentPart::Text {
            text: "ok".to_string(),
        }],
        finish_reason: FinishReason::Stop,
        usage: Usage::default(),
        warnings: Vec::new(),
        provider: BTreeMap::new(),
    }
}

struct FakeLanguage {
    descriptor: ModelDescriptor,
}

impl FakeLanguage {
    fn new(model: &str) -> Self {
        Self {
            descriptor: descriptor(ModelFamily::Language, model),
        }
    }

    fn with_api_mode(mut self, api_mode: &str) -> Self {
        self.descriptor = self
            .descriptor
            .clone()
            .with_api_mode(ApiModeId::new(api_mode).unwrap());
        self
    }
}

impl Model for FakeLanguage {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl LanguageModel for FakeLanguage {
    async fn generate(
        &self,
        _request: LanguageRequest,
        _options: CallOptions,
    ) -> Result<LanguageResponse, Error> {
        Ok(language_response(self.model_id().as_str()))
    }

    async fn stream(
        &self,
        _request: LanguageRequest,
        options: CallOptions,
    ) -> Result<LanguageStream, Error> {
        Ok(established_stream(options.cancellation().clone(), |_| {
            stream::iter(vec![Ok(LanguageStreamEvent::Terminal(
                StreamTerminal::Completed {
                    response: Box::new(language_response(self.model_id().as_str())),
                },
            ))])
        }))
    }
}

macro_rules! model_metadata {
    ($name:ident, $family:expr, $model:literal) => {
        struct $name {
            descriptor: ModelDescriptor,
        }

        impl $name {
            fn new() -> Self {
                Self {
                    descriptor: descriptor($family, $model),
                }
            }
        }

        impl Model for $name {
            fn descriptor(&self) -> &ModelDescriptor {
                &self.descriptor
            }
        }
    };
}

model_metadata!(FakeEmbedding, ModelFamily::Embedding, "embed-test");
model_metadata!(FakeRerank, ModelFamily::Rerank, "rerank-test");
model_metadata!(FakeImage, ModelFamily::Image, "image-test");
model_metadata!(FakeSpeech, ModelFamily::Speech, "speech-test");
model_metadata!(FakeTranscription, ModelFamily::Transcription, "stt-test");

#[async_trait]
impl EmbeddingModel for FakeEmbedding {
    async fn embed(
        &self,
        request: EmbeddingRequest,
        _options: CallOptions,
    ) -> Result<EmbeddingResponse, Error> {
        Ok(EmbeddingResponse {
            embeddings: request.inputs().iter().map(|_| vec![1.0, 2.0]).collect(),
            usage: Usage::default(),
            warnings: Vec::new(),
            provider: BTreeMap::new(),
        })
    }
}

#[async_trait]
impl RerankModel for FakeRerank {
    async fn rerank(
        &self,
        request: RerankRequest,
        _options: CallOptions,
    ) -> Result<RerankResponse, Error> {
        Ok(RerankResponse {
            results: request
                .candidates
                .into_iter()
                .enumerate()
                .map(|(index, candidate)| RerankResult {
                    index,
                    score: 1.0,
                    candidate: Some(candidate),
                })
                .collect(),
            usage: Usage::default(),
            warnings: Vec::new(),
            provider: BTreeMap::new(),
        })
    }
}

#[async_trait]
impl ImageModel for FakeImage {
    async fn generate_image(
        &self,
        _request: ImageRequest,
        _options: CallOptions,
    ) -> Result<ImageResponse, Error> {
        Ok(ImageResponse {
            images: vec![ImageArtifact {
                media_type: "image/png".to_string(),
                data: MediaData::Bytes(vec![1, 2, 3]),
                revised_prompt: None,
            }],
            usage: Usage::default(),
            warnings: Vec::new(),
            provider: BTreeMap::new(),
        })
    }
}

#[async_trait]
impl SpeechModel for FakeSpeech {
    async fn synthesize(
        &self,
        request: SpeechRequest,
        _options: CallOptions,
    ) -> Result<SpeechResponse, Error> {
        Ok(SpeechResponse {
            media_type: "audio/pcm".to_string(),
            audio: request.text.into_bytes(),
            usage: Usage::default(),
            warnings: Vec::new(),
            provider: BTreeMap::new(),
        })
    }
}

#[async_trait]
impl TranscriptionModel for FakeTranscription {
    async fn transcribe(
        &self,
        _request: TranscriptionRequest,
        _options: CallOptions,
    ) -> Result<TranscriptionResponse, Error> {
        Ok(TranscriptionResponse {
            text: "transcript".to_string(),
            language: Some("en".to_string()),
            segments: Vec::new(),
            usage: Usage::default(),
            warnings: Vec::new(),
            provider: BTreeMap::new(),
        })
    }
}

#[derive(Serialize)]
struct CustomOptions {
    strict: bool,
}

impl TypedProviderOptions for CustomOptions {
    const NAMESPACE: &'static str = "custom";
}

struct CustomPolicy;

impl ModelPolicy for CustomPolicy {
    fn evaluate(&self, _context: &ModelPolicyContext) -> ModelPolicyDecision {
        ModelPolicyDecision::unknown_model()
    }
}

fn prompt() -> LanguageRequest {
    LanguageRequest::new(vec![Message::text(MessageRole::User, "hello")])
}

#[tokio::test]
async fn external_models_are_object_safe_callable_and_task_safe() {
    let options = ProviderOptions::typed(&CustomOptions { strict: true }).unwrap();
    let call_options = CallOptions::default().with_provider_options(options);

    let language: Arc<dyn LanguageModel> = Arc::new(FakeLanguage::new("language-test"));
    let generated = tokio::spawn({
        let language = language.clone();
        let options = call_options.clone();
        async move { language.generate(prompt(), options).await }
    })
    .await
    .unwrap()
    .unwrap();
    assert_eq!(generated.model.unwrap().as_str(), "language-test");

    let mut language_stream = language.stream(prompt(), call_options).await.unwrap();
    assert!(matches!(
        language_stream.next().await,
        Some(LanguageStreamEvent::Terminal(
            StreamTerminal::Completed { .. }
        ))
    ));
    assert!(language_stream.next().await.is_none());

    let embedding: Arc<dyn EmbeddingModel> = Arc::new(FakeEmbedding::new());
    let embedded = embedding
        .embed(
            EmbeddingRequest::new(vec![EmbeddingInput::Text("hello".to_string())]).unwrap(),
            CallOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(embedded.embeddings.len(), 1);

    let rerank: Arc<dyn RerankModel> = Arc::new(FakeRerank::new());
    let reranked = rerank
        .rerank(
            RerankRequest {
                query: "query".to_string(),
                candidates: vec![RerankCandidate {
                    id: None,
                    text: "candidate".to_string(),
                    metadata: BTreeMap::new(),
                }],
                top_n: None,
            },
            CallOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(reranked.results.len(), 1);

    let image: Arc<dyn ImageModel> = Arc::new(FakeImage::new());
    assert_eq!(
        image
            .generate_image(
                ImageRequest {
                    prompt: "image".to_string(),
                    count: 1,
                    size: None,
                    format: None,
                },
                CallOptions::default(),
            )
            .await
            .unwrap()
            .images
            .len(),
        1
    );

    let speech: Arc<dyn SpeechModel> = Arc::new(FakeSpeech::new());
    assert!(
        !speech
            .synthesize(
                SpeechRequest {
                    text: "speech".to_string(),
                    voice: None,
                    format: None,
                    speed: None,
                },
                CallOptions::default(),
            )
            .await
            .unwrap()
            .audio
            .is_empty()
    );

    let transcription: Arc<dyn TranscriptionModel> = Arc::new(FakeTranscription::new());
    assert_eq!(
        transcription
            .transcribe(
                TranscriptionRequest {
                    audio: vec![1, 2, 3],
                    media_type: "audio/wav".to_string(),
                    language: None,
                    prompt: None,
                },
                CallOptions::default(),
            )
            .await
            .unwrap()
            .text,
        "transcript"
    );
}

#[tokio::test]
async fn registration_captures_shared_runtime_and_returns_cheap_models() {
    let constructions = Arc::new(AtomicUsize::new(0));
    let factory_count = constructions.clone();
    let registration =
        ProviderRegistration::new(ProviderId::new("custom").unwrap(), Arc::new(CustomPolicy))
            .with_protocol(ProtocolId::new("test").unwrap())
            .with_api_mode(ApiModeId::new("native").unwrap())
            .with_language(Arc::new(move |model| {
                factory_count.fetch_add(1, Ordering::SeqCst);
                Ok(Arc::new(
                    FakeLanguage::new(model.as_str()).with_api_mode("native"),
                ))
            }));

    let first = registration
        .language_model(ModelId::new("future:model").unwrap())
        .unwrap();
    let second = registration
        .language_model(ModelId::new("future:model").unwrap())
        .unwrap();

    assert_eq!(constructions.load(Ordering::SeqCst), 2);
    assert_eq!(
        registration.api_mode().map(ApiModeId::as_str),
        Some("native")
    );
    assert_eq!(first.model_id(), second.model_id());
    assert_eq!(
        first
            .generate(prompt(), CallOptions::default())
            .await
            .unwrap()
            .warnings,
        Vec::<Warning>::new()
    );
}

#[test]
fn registration_rejects_factory_identity_drift() {
    let registration =
        ProviderRegistration::new(ProviderId::new("other").unwrap(), Arc::new(CustomPolicy))
            .with_language(Arc::new(|model| {
                Ok(Arc::new(FakeLanguage::new(model.as_str())))
            }));

    let error = registration
        .language_model(ModelId::new("future:model").unwrap())
        .err()
        .expect("identity drift must fail model acquisition");
    assert!(matches!(
        error,
        siumai_core::ModelLookupError::IdentityMismatch { .. }
    ));
}
