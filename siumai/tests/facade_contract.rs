use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use siumai::prelude::*;
use siumai::{EmbeddingLimits, ModelId, ResponseMetadata};

#[cfg(feature = "runtime")]
use siumai::{ErrorKind, LanguageCallError, LanguageCompletionReason, LanguageResponse};

#[cfg(feature = "registry")]
use siumai::{ImageArtifact, MediaData};

#[cfg(feature = "registry")]
use siumai::core::ProviderRegistration;

#[derive(Debug)]
struct FakeEmbedding {
    descriptor: ModelDescriptor,
    calls: Arc<AtomicUsize>,
}

#[cfg(feature = "runtime")]
#[derive(Debug)]
struct FakeLanguage {
    descriptor: ModelDescriptor,
    calls: Arc<AtomicUsize>,
}

#[cfg(feature = "runtime")]
impl Model for FakeLanguage {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[cfg(feature = "runtime")]
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

impl Model for FakeEmbedding {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
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
        self.limits().validate(&request)?;
        self.calls.fetch_add(1, Ordering::SeqCst);
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

#[cfg(feature = "registry")]
#[derive(Debug)]
struct FakeImage {
    descriptor: ModelDescriptor,
    calls: Arc<AtomicUsize>,
}

#[cfg(feature = "registry")]
impl Model for FakeImage {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[cfg(feature = "registry")]
#[async_trait]
impl ImageModel for FakeImage {
    async fn generate_image(
        &self,
        request: ImageRequest,
        _options: CallOptions,
    ) -> Result<ImageResponse, Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let response = ImageResponse {
            images: (0..request.count())
                .map(|_| ImageArtifact {
                    media_type: "image/png".to_string(),
                    data: MediaData::Url("https://example.test/image.png".to_string()),
                    revised_prompt: None,
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

#[cfg(feature = "runtime")]
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

#[cfg(feature = "registry")]
fn fake_image(model: ModelId, calls: Arc<AtomicUsize>) -> FakeImage {
    FakeImage {
        descriptor: ModelDescriptor::new(
            ProviderId::new("fake").unwrap(),
            model,
            ModelFamily::Image,
        ),
        calls,
    }
}

#[tokio::test]
async fn direct_family_helper_preserves_one_call_per_batch() {
    let calls = Arc::new(AtomicUsize::new(0));
    let model = fake(ModelId::new("embed-v1").unwrap(), calls.clone());
    let request = EmbeddingRequest::new(["one", "two"]).unwrap();

    let response = embedding::embed(&model, request).await.unwrap();

    assert_eq!(response.embeddings.len(), 2);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
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
#[tokio::test]
async fn runtime_facade_reexports_one_call_language_execution() {
    let calls = Arc::new(AtomicUsize::new(0));
    let model = fake_language(ModelId::new("language-v1").unwrap(), calls.clone());
    let response = siumai::generate(
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
    assert_eq!(calls.load(Ordering::SeqCst), 1);
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
        GPT_4O_TRANSCRIBE, OpenAiTranscriptionOptions,
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
    use siumai::providers::openai::{OpenAiCredential, OpenAiProvider};
    use siumai_transport::EndpointConfig;

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
    assert!(provider.transcription(GPT_4O_TRANSCRIBE).is_ok());

    let _ = OpenAiEmbeddingOptions::default();
    let _ = OpenAiImageOptions::default();
    let _ = OpenAiSpeechOptions::default();
    let _ = OpenAiTranscriptionOptions::default();
    let _ = OpenAiContentOptions::cache_write_candidate();
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
        GEMINI_EMBEDDING_001, current_image_models, current_interactions_models,
    };
    use siumai::providers::google::options::{
        GeminiEmbeddingOptions, GeminiEmbeddingTaskType, GeminiGenerateContentOptions,
        GeminiGenerateContentServiceTier, GeminiGenerateContentThinking,
        GeminiGenerateContentThinkingLevel, GeminiImageAspectRatio, GeminiImageOptions,
        GeminiImageSize, GeminiInteractionsOptions, GeminiSpeechOptions, GeminiThinkingLevel,
    };
    use siumai::providers::google::{GeminiCredential, GeminiProvider};

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
    assert_eq!(provider.support_manifest().native_claims().len(), 2);
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
    use siumai::providers::anthropic::options::{AnthropicMessagesOptions, AnthropicThinking};
    use siumai::providers::anthropic::{AnthropicCredential, AnthropicProvider};
    use siumai::registry::ProviderRegistrationSource;

    fn assert_registration_source<T: ProviderRegistrationSource>() {}

    assert_registration_source::<AnthropicProvider>();

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
async fn openai_direct_registry_and_helper_paths_share_one_wire_pipeline() {
    use serde_json::{Value, json};
    use siumai::core::{ReplayDomain, ReplayDomainId};
    use siumai::providers::openai::{OpenAiCredential, OpenAiProvider};
    use siumai::registry::{Registry, RegistryBuilderExt};
    use siumai_transport::{EndpointConfig, TransportEvent, TransportObserver};
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
                TransportEvent::Completed { .. } => {
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
        .expect(3)
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
        .with_transport_observer(observer.clone())
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
    let chat_response = chat
        .generate(request(), CallOptions::default())
        .await
        .unwrap();

    assert_eq!(direct_response, erased_response);
    assert_eq!(erased_response, helper_response);
    assert!(matches!(
        &chat_response.content()[0],
        ContentPart::Text { text } if text == "hello back"
    ));
    assert_eq!(observer.attempts.load(Ordering::SeqCst), 4);
    assert_eq!(observer.completed.load(Ordering::SeqCst), 4);

    let requests = server.received_requests().await.unwrap();
    let bodies = requests
        .iter()
        .filter(|request| request.url.path() == "/v1/responses")
        .map(|request| serde_json::from_slice::<Value>(&request.body).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(bodies[0], bodies[1]);
    assert_eq!(bodies[1], bodies[2]);
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
        OPENAI_RESPONSES_WEBSOCKET_URL, OpenAiResponsesWebSocketTurnKind,
    };
    use siumai::providers::openai::{OpenAiCredential, OpenAiProvider};

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
}
