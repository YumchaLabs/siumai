use siumai::Siumai;

#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "openai-compatible"
))]
fn accepts_language_model<M: siumai::LanguageModel + ?Sized>(model: &M) {
    let _ = model;
}

#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "openai-compatible",
    feature = "alibaba",
    feature = "google-vertex-anthropic",
    feature = "elevenlabs"
))]
fn error_diagnostics(error: &(dyn std::error::Error + 'static)) -> String {
    let mut output = String::new();
    let mut current = Some(error);
    while let Some(error) = current {
        output.push_str(&format!("{error:?}\n{error}\n"));
        current = error.source();
    }
    output
}

#[test]
fn zero_state_builder_is_available_without_default_features() {
    let _builder = Siumai::builder();
}

mod generic_hub {
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
        ImageModelProvider, ImageRequest, ImageResponse, LanguageCallError,
        LanguageCompletionReason, LanguageModel, LanguageModelProvider, LanguageRequest,
        LanguageResponse, LanguageStream, LanguageStreamEvent, MediaData, Message, Model,
        ModelDescriptor, ModelFamily, ModelId, ModelLookupError, Provider, ProviderId,
        RerankCandidate, RerankLimits, RerankModel, RerankModelProvider, RerankRequest,
        RerankResponse, RerankResult, ResponseMetadata, Siumai, SpeechLimits, SpeechModel,
        SpeechModelProvider, SpeechRequest, SpeechResponse, StreamTerminal, TranscriptionLimits,
        TranscriptionModel, TranscriptionModelProvider, TranscriptionRequest,
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

        let stream_method_response =
            completed_stream_response(language.stream("hello").await?).await;
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
        let speech_root_response =
            siumai::speech::synthesize(&speech, speech_request.clone()).await?;
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
            siumai::transcription::transcribe(&transcription, transcription_request.clone())
                .await?;
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
        let matching_call =
            siumai::language::call(primary.as_ref(), "hello").with_options(matching)?;
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
}

#[cfg(feature = "openai")]
mod openai {
    use super::{Siumai, accepts_language_model, error_diagnostics};
    use siumai::providers::openai::chat_completions::{
        OpenAiChatCompletionsModel, OpenAiChatCompletionsOptions,
    };
    use siumai::providers::openai::embeddings::{
        OpenAiEmbeddingModel, OpenAiEmbeddingOptions, TEXT_EMBEDDING_3_SMALL,
    };
    use siumai::providers::openai::models::GPT_5_6;
    use siumai::providers::openai::resources::files::OpenAiFiles;
    use siumai::providers::openai::responses::{
        OpenAiResponsesModel, OpenAiResponsesOptions, OpenAiResponsesResource,
    };
    use siumai::providers::openai::{
        OpenAiConfigError, OpenAiCredential, OpenAiProvider, OpenAiProviderBuilder,
    };
    use siumai::{
        CallOptions, EmbeddingLimits, EmbeddingModel, Model, ModelFamily, ProviderOptionError,
    };

    const CREDENTIAL_CANARY: &str = "openai-facade-secret-canary";

    fn accepts_embedding_model<M>(model: &M)
    where
        M: EmbeddingModel + ?Sized,
    {
        let _ = model;
    }

    fn accepts_model<M>(model: &M)
    where
        M: Model + ?Sized,
    {
        let _ = model;
    }

    fn accepts_openai_provider(provider: &OpenAiProvider) {
        let _ = provider;
    }

    fn accepts_openai_responses_model(model: &OpenAiResponsesModel) {
        let _ = model;
    }

    fn accepts_openai_embedding_model(model: &OpenAiEmbeddingModel) {
        let _ = model;
    }

    fn accepts_openai_chat_model(model: &OpenAiChatCompletionsModel) {
        let _ = model;
    }

    #[test]
    fn api_key_chain_builds_a_reusable_multi_family_hub() -> Result<(), Box<dyn std::error::Error>>
    {
        let hub = Siumai::builder().openai().api_key("test-api-key").build()?;

        let language = hub.language(GPT_5_6)?;
        let embedding = hub.embedding(TEXT_EMBEDDING_3_SMALL)?;

        accepts_language_model(&language);
        accepts_embedding_model(&embedding);
        accepts_openai_provider(hub.provider());
        accepts_openai_provider(language.provider());
        accepts_openai_responses_model(language.model());
        accepts_openai_embedding_model(embedding.model());
        accepts_model(language.model());
        accepts_model(embedding.model());
        assert!(std::ptr::eq(hub.provider(), language.provider()));
        assert!(std::ptr::eq(hub.provider(), embedding.provider()));

        Ok(())
    }

    #[test]
    fn provider_owned_credential_uses_the_same_public_shape()
    -> Result<(), Box<dyn std::error::Error>> {
        let credential = OpenAiCredential::api_key("test-api-key");
        let hub = Siumai::builder().openai().credential(credential).build()?;
        let language = hub.language(GPT_5_6)?;

        accepts_language_model(&language);
        accepts_openai_provider(language.provider());
        accepts_model(language.model());

        Ok(())
    }

    #[test]
    fn native_access_identity_limits_and_exact_options_survive_typed_wrapping()
    -> Result<(), Box<dyn std::error::Error>> {
        let hub = Siumai::builder()
            .openai()
            .api_key(CREDENTIAL_CANARY)
            .build()?;
        let responses = hub.language(GPT_5_6)?;
        let chat = hub.chat_completions(GPT_5_6)?;
        let embedding = hub.embedding(TEXT_EMBEDDING_3_SMALL)?;

        let files: OpenAiFiles = responses.provider().files();
        let responses_resource: OpenAiResponsesResource = responses.provider().responses_resource();
        assert!(!format!("{files:?}").contains(CREDENTIAL_CANARY));
        assert!(!format!("{responses_resource:?}").contains(CREDENTIAL_CANARY));

        assert!(std::ptr::eq(hub.provider(), responses.provider()));
        assert!(std::ptr::eq(hub.provider(), chat.provider()));
        assert!(std::ptr::eq(hub.provider(), embedding.provider()));
        assert!(std::ptr::eq(
            Model::descriptor(&responses),
            responses.model().descriptor()
        ));
        assert!(responses.route_id().is_none());
        assert_eq!(responses.family(), ModelFamily::Language);
        assert_eq!(embedding.family(), ModelFamily::Embedding);
        assert_eq!(responses.descriptor().api_mode(), Some("responses"));
        assert_eq!(chat.descriptor().api_mode(), Some("chat-completions"));
        assert_eq!(embedding.descriptor().api_mode(), Some("embeddings"));
        assert_eq!(
            responses.descriptor().instance_id(),
            chat.descriptor().instance_id()
        );
        assert_eq!(
            responses.descriptor().instance_id(),
            embedding.descriptor().instance_id()
        );
        assert_eq!(
            EmbeddingModel::limits(&embedding),
            EmbeddingLimits {
                max_inputs: Some(2_048),
                max_input_tokens: Some(8_192),
            }
        );
        assert_eq!(
            EmbeddingModel::limits(&embedding),
            EmbeddingModel::limits(embedding.model())
        );

        let matching = CallOptions::default().with_provider_options_for(
            responses.model(),
            &OpenAiResponsesOptions {
                instructions: Some("preserve exact target".to_string()),
                ..OpenAiResponsesOptions::default()
            },
        )?;
        let matching_call = responses.call("hello").with_options(matching)?;
        assert_eq!(
            matching_call
                .base_options()
                .provider_options_for(&responses)?
                .typed()
                .count(),
            1
        );

        let independent = Siumai::builder()
            .openai()
            .api_key("independent-openai-key")
            .build()?
            .language(GPT_5_6)?;
        assert_ne!(
            responses.descriptor().instance_id(),
            independent.descriptor().instance_id()
        );
        let foreign_instance = CallOptions::default()
            .with_provider_options_for(independent.model(), &OpenAiResponsesOptions::default())?;
        let instance_error = match responses.call("hello").with_options(foreign_instance) {
            Ok(_) => panic!("options from another configured instance must be rejected"),
            Err(error) => error,
        };
        assert!(matches!(
            instance_error,
            ProviderOptionError::ExactTargetMismatch { .. }
        ));

        let foreign_mode = CallOptions::default()
            .with_provider_options_for(chat.model(), &OpenAiChatCompletionsOptions::default())?;
        let mode_error = match responses.call("hello").with_options(foreign_mode) {
            Ok(_) => panic!("options for Chat Completions must not target Responses"),
            Err(error) => error,
        };
        assert!(matches!(
            mode_error,
            ProviderOptionError::TargetMismatch { .. }
        ));

        let foreign_family = CallOptions::default().with_provider_options_for(
            embedding.model(),
            &OpenAiEmbeddingOptions::new().with_user("facade-test")?,
        )?;
        let family_error = match responses.call("hello").with_options(foreign_family) {
            Ok(_) => panic!("embedding options must not target a language client"),
            Err(error) => error,
        };
        assert!(matches!(
            family_error,
            ProviderOptionError::TargetMismatch { .. }
        ));

        Ok(())
    }

    #[cfg(feature = "openai-responses-websocket")]
    #[test]
    fn responses_model_exposes_the_provider_owned_websocket_configuration()
    -> Result<(), Box<dyn std::error::Error>> {
        use siumai::providers::openai::experimental::responses_websocket::OpenAiResponsesWebSocketConfig;

        let client = Siumai::builder()
            .openai()
            .api_key(CREDENTIAL_CANARY)
            .build()?
            .language(GPT_5_6)?;
        let websocket: OpenAiResponsesWebSocketConfig = client.model().websocket()?;

        assert_eq!(websocket.model().descriptor(), client.descriptor());
        assert_eq!(
            websocket.model().descriptor().instance_id(),
            client.descriptor().instance_id()
        );
        assert!(!format!("{websocket:?}").contains(CREDENTIAL_CANARY));

        Ok(())
    }

    #[test]
    fn explicit_chat_mode_and_future_model_ids_remain_open()
    -> Result<(), Box<dyn std::error::Error>> {
        let hub = Siumai::builder().openai().api_key("test-api-key").build()?;
        let responses = hub.language("future-openai-model")?;
        let chat = hub.chat_completions("future-openai-model")?;

        accepts_openai_responses_model(responses.model());
        accepts_openai_chat_model(chat.model());
        assert_eq!(
            responses.descriptor().model().as_str(),
            "future-openai-model"
        );
        assert_eq!(chat.descriptor().model().as_str(), "future-openai-model");
        assert_ne!(
            responses.descriptor().api_mode(),
            chat.descriptor().api_mode()
        );

        Ok(())
    }

    #[test]
    fn real_builder_configuration_and_secret_diagnostics_are_preserved()
    -> Result<(), Box<dyn std::error::Error>> {
        let required = Siumai::builder().openai();
        assert!(!format!("{required:?}").contains(CREDENTIAL_CANARY));

        let stage = Siumai::builder().openai().api_key(CREDENTIAL_CANARY);
        assert!(!format!("{stage:?}").contains(CREDENTIAL_CANARY));
        let hub = stage.build()?;
        let client = hub.language("future-openai-model")?;
        assert!(!format!("{hub:?}").contains(CREDENTIAL_CANARY));
        assert!(!format!("{client:?}").contains(CREDENTIAL_CANARY));

        let error = Siumai::builder()
            .openai()
            .api_key(CREDENTIAL_CANARY)
            .configure_provider(|builder: OpenAiProviderBuilder| {
                builder.with_project("project-without-replay-scope")
            })
            .build()
            .expect_err("the real OpenAI builder must validate project replay scope");
        assert!(matches!(
            error,
            OpenAiConfigError::AccountScopeRequiresReplayCallerScope
        ));
        assert!(!error_diagnostics(&error).contains(CREDENTIAL_CANARY));

        Ok(())
    }
}

#[cfg(feature = "anthropic")]
mod anthropic {
    use super::{Siumai, accepts_language_model, error_diagnostics};
    use siumai::providers::anthropic::annotations::{AnthropicCacheTtl, AnthropicContentOptions};
    use siumai::providers::anthropic::options::AnthropicMessagesOptions;
    use siumai::providers::anthropic::resources::{AnthropicFiles, AnthropicMessageBatches};
    use siumai::providers::anthropic::{
        AnthropicCredential, AnthropicLanguageModel, AnthropicProvider, AnthropicProviderBuilder,
    };
    use siumai::{LanguageRequest, Message, MessagePart, MessageRole, Model, ModelFamily};

    const CREDENTIAL_CANARY: &str = "anthropic-facade-secret-canary";

    fn accepts_anthropic_provider(provider: &AnthropicProvider) {
        let _ = provider;
    }

    fn accepts_anthropic_model(model: &AnthropicLanguageModel) {
        let _ = model;
    }

    #[test]
    fn api_key_and_provider_owned_credential_build_the_same_typed_hub()
    -> Result<(), Box<dyn std::error::Error>> {
        let api_key_hub = Siumai::builder()
            .anthropic()
            .api_key("test-api-key")
            .build()?;
        let credential_hub = Siumai::builder()
            .anthropic()
            .credential(AnthropicCredential::api_key("test-api-key"))
            .build()?;

        for hub in [&api_key_hub, &credential_hub] {
            let client = hub.language("future-anthropic-model")?;
            accepts_language_model(&client);
            accepts_anthropic_provider(hub.provider());
            accepts_anthropic_model(client.model());
            assert_eq!(
                client.descriptor().model().as_str(),
                "future-anthropic-model"
            );
        }

        Ok(())
    }

    #[test]
    fn real_builder_configuration_and_secret_diagnostics_are_preserved()
    -> Result<(), Box<dyn std::error::Error>> {
        let stage = Siumai::builder().anthropic().api_key(CREDENTIAL_CANARY);
        assert!(!format!("{stage:?}").contains(CREDENTIAL_CANARY));
        let hub = stage.build()?;
        let client = hub.language("future-anthropic-model")?;
        assert!(!format!("{hub:?}").contains(CREDENTIAL_CANARY));
        assert!(!format!("{client:?}").contains(CREDENTIAL_CANARY));

        let error = Siumai::builder()
            .anthropic()
            .api_key(CREDENTIAL_CANARY)
            .configure_provider(|builder: AnthropicProviderBuilder| {
                builder.with_base_url("not a valid URL")
            })
            .build()
            .expect_err("the real Anthropic builder must validate its endpoint");
        assert!(!error_diagnostics(&error).contains(CREDENTIAL_CANARY));

        Ok(())
    }

    #[test]
    fn native_resources_annotations_and_model_methods_remain_typed_after_binding()
    -> Result<(), Box<dyn std::error::Error>> {
        let hub = Siumai::builder()
            .anthropic()
            .api_key(CREDENTIAL_CANARY)
            .build()?;
        let client = hub.language("future-anthropic-model")?;
        let files: AnthropicFiles = client.provider().files();
        let batches: AnthropicMessageBatches = client.provider().message_batches();

        assert!(std::ptr::eq(hub.provider(), client.provider()));
        assert!(std::ptr::eq(
            Model::descriptor(&client),
            client.model().descriptor()
        ));
        assert_eq!(client.family(), ModelFamily::Language);
        assert_eq!(client.descriptor().api_mode(), Some("messages"));
        assert!(client.route_id().is_none());
        assert!(!format!("{files:?}").contains(CREDENTIAL_CANARY));
        assert!(!format!("{batches:?}").contains(CREDENTIAL_CANARY));

        let annotated = MessagePart::text("cache this prefix")
            .with_provider_annotation(&AnthropicContentOptions::one_hour())?;
        assert_eq!(
            annotated
                .annotations()
                .decode::<AnthropicContentOptions>()?,
            Some(AnthropicContentOptions::one_hour())
        );
        let request = LanguageRequest::new(vec![Message::new(MessageRole::User, [annotated])]);
        let options =
            AnthropicMessagesOptions::new().with_automatic_cache(AnthropicCacheTtl::OneHour);
        drop(client.model().prewarm_cache(request, options));

        Ok(())
    }
}

#[cfg(feature = "google")]
mod gemini {
    use super::{Siumai, accepts_language_model, error_diagnostics};
    use siumai::providers::google::{
        GEMINI_GENERATE_CONTENT_API_MODE_ID, GeminiCredential, GeminiFiles,
        GeminiGenerateContentModel, GeminiLanguageModel, GeminiProvider, GeminiProviderBuilder,
        GeminiVeo,
    };
    use siumai::{CallOptions, LanguageRequest, Model, ModelFamily};

    const CREDENTIAL_CANARY: &str = "gemini-facade-secret-canary";

    fn accepts_gemini_provider(provider: &GeminiProvider) {
        let _ = provider;
    }

    fn accepts_interactions_model(model: &GeminiLanguageModel) {
        let _ = model;
    }

    fn accepts_generate_content_model(model: &GeminiGenerateContentModel) {
        let _ = model;
    }

    #[test]
    fn canonical_interactions_and_explicit_generate_content_are_distinct()
    -> Result<(), Box<dyn std::error::Error>> {
        let hub = Siumai::builder().gemini().api_key("test-api-key").build()?;
        let interactions = hub.language("future-gemini-model")?;
        let generate_content = hub.generate_content("future-gemini-model")?;

        accepts_language_model(&interactions);
        accepts_language_model(&generate_content);
        accepts_gemini_provider(hub.provider());
        accepts_interactions_model(interactions.model());
        accepts_generate_content_model(generate_content.model());
        assert_eq!(
            generate_content.descriptor().api_mode(),
            Some(GEMINI_GENERATE_CONTENT_API_MODE_ID)
        );
        assert_ne!(
            interactions.descriptor().api_mode(),
            generate_content.descriptor().api_mode()
        );

        Ok(())
    }

    #[test]
    fn credential_path_real_builder_configuration_and_redaction_are_preserved()
    -> Result<(), Box<dyn std::error::Error>> {
        let hub = Siumai::builder()
            .gemini()
            .credential(GeminiCredential::api_key(CREDENTIAL_CANARY))
            .build()?;
        let client = hub.language("future-gemini-model")?;
        assert!(!format!("{hub:?}").contains(CREDENTIAL_CANARY));
        assert!(!format!("{client:?}").contains(CREDENTIAL_CANARY));

        let stage = Siumai::builder().gemini().api_key(CREDENTIAL_CANARY);
        assert!(!format!("{stage:?}").contains(CREDENTIAL_CANARY));
        let error = stage
            .configure_provider(|builder: GeminiProviderBuilder| {
                builder.with_base_url("not a valid URL")
            })
            .build()
            .expect_err("the real Gemini builder must validate its endpoint");
        assert!(!error_diagnostics(&error).contains(CREDENTIAL_CANARY));

        Ok(())
    }

    #[test]
    fn native_files_veo_and_interactions_methods_remain_typed_after_binding()
    -> Result<(), Box<dyn std::error::Error>> {
        let hub = Siumai::builder()
            .gemini()
            .api_key(CREDENTIAL_CANARY)
            .build()?;
        let interactions = hub.language("future-gemini-model")?;
        let generate_content = hub.generate_content("future-gemini-model")?;
        let files: GeminiFiles = interactions.provider().files();
        let veo: GeminiVeo = interactions.provider().veo();

        assert!(std::ptr::eq(hub.provider(), interactions.provider()));
        assert!(std::ptr::eq(hub.provider(), generate_content.provider()));
        assert!(std::ptr::eq(
            Model::descriptor(&interactions),
            interactions.model().descriptor()
        ));
        assert_eq!(interactions.family(), ModelFamily::Language);
        assert_eq!(interactions.descriptor().api_mode(), Some("interactions"));
        assert_eq!(
            generate_content.descriptor().api_mode(),
            Some(GEMINI_GENERATE_CONTENT_API_MODE_ID)
        );
        assert_eq!(
            interactions.descriptor().instance_id(),
            generate_content.descriptor().instance_id()
        );
        assert!(interactions.route_id().is_none());
        assert!(!format!("{files:?}").contains(CREDENTIAL_CANARY));
        assert!(!format!("{veo:?}").contains(CREDENTIAL_CANARY));

        drop(
            interactions
                .model()
                .generate_native(LanguageRequest::new(Vec::new()), CallOptions::default()),
        );

        Ok(())
    }
}

#[cfg(feature = "openai-compatible")]
mod openai_compatible {
    use super::{Siumai, accepts_language_model, error_diagnostics};
    use siumai::Model;
    use siumai::core::{
        ApiModeId, ApiStability, ModelCatalog, ModelFamily, OfficialSource, PlatformId, ProfileId,
        ProtocolContractId, ProtocolId, ProviderId, ProviderProfile, ReplayDomainId, SupportScope,
        VerificationDate, VerificationEvidence, VerifiedFidelity, VerifiedSupportClaim,
    };
    use siumai::providers::openai_compatible::{
        OpenAiCompatibleApiMode, OpenAiCompatibleCredential, OpenAiCompatibleLanguageModel,
        OpenAiCompatibleProfile, OpenAiCompatibleProvider, OpenAiCompatibleProviderBuilder,
    };
    use siumai::transport::{EndpointConfig, OfficialOrigin};
    use siumai_protocol_openai::chat_completions::{
        API_MODE_ID as CHAT_API_MODE_ID, ChatCompletionsDialect, PROTOCOL_ID as CHAT_PROTOCOL_ID,
    };

    const CREDENTIAL_CANARY: &str = "compatible-facade-secret-canary";

    pub(super) fn custom_profile() -> Result<OpenAiCompatibleProfile, Box<dyn std::error::Error>> {
        Ok(OpenAiCompatibleProfile::public_custom(
            ProviderId::new("custom-compatible")?,
            "https://custom-compatible.example/v1",
            ReplayDomainId::new("custom-compatible-replay")?,
            OpenAiCompatibleApiMode::Responses,
        )?)
    }

    fn verified_profile() -> Result<OpenAiCompatibleProfile, Box<dyn std::error::Error>> {
        let provider = ProviderId::new("verified-compatible")?;
        let scope = SupportScope::new(
            provider,
            PlatformId::new("official-api")?,
            ModelFamily::Language,
            ProtocolId::new(CHAT_PROTOCOL_ID)?,
            ApiModeId::new(CHAT_API_MODE_ID)?,
        );
        let verified_at: VerificationDate = serde_json::from_str("\"2026-08-19\"")?;
        let evidence = VerificationEvidence::new(
            OfficialSource::new("https://verified-compatible.example/docs")?,
            verified_at,
            ProtocolContractId::new("verified-compatible-2026-08")?,
        );
        let profile = ProviderProfile::verified(
            ProfileId::new("verified-compatible")?,
            vec![VerifiedSupportClaim::new(
                scope,
                VerifiedFidelity::Compatible,
                ApiStability::Stable,
                evidence,
            )],
            ModelCatalog::default(),
        )?;
        let endpoint = EndpointConfig::official(
            "https://verified-compatible.example/v1",
            OfficialOrigin::new("https://verified-compatible.example")?,
        )?;
        Ok(OpenAiCompatibleProfile::verified_chat(
            profile,
            endpoint,
            ChatCompletionsDialect::generic(),
        )?)
    }

    fn accepts_compatible_provider(provider: &OpenAiCompatibleProvider) {
        let _ = provider;
    }

    fn accepts_compatible_model(model: &OpenAiCompatibleLanguageModel) {
        let _ = model;
    }

    #[test]
    fn custom_and_verified_profiles_pass_through_the_typed_stages()
    -> Result<(), Box<dyn std::error::Error>> {
        let custom = Siumai::builder()
            .openai_compatible()
            .profile(custom_profile()?)
            .api_key("test-api-key")
            .configure_provider(|builder: OpenAiCompatibleProviderBuilder| builder)
            .build()?;
        let verified = Siumai::builder()
            .openai_compatible()
            .profile(verified_profile()?)
            .credential(OpenAiCompatibleCredential::api_key("test-api-key"))
            .build()?;

        let custom_model = custom.language("future-compatible-model")?;
        let verified_model = verified.language("future-compatible-model")?;
        accepts_language_model(&custom_model);
        accepts_language_model(&verified_model);
        accepts_compatible_provider(custom.provider());
        accepts_compatible_provider(verified.provider());
        accepts_compatible_model(custom_model.model());
        accepts_compatible_model(verified_model.model());
        assert_eq!(
            custom.provider().profile().provider_profile().id().as_str(),
            "custom-compatible"
        );
        assert_eq!(
            verified
                .provider()
                .profile()
                .provider_profile()
                .id()
                .as_str(),
            "verified-compatible"
        );
        assert_eq!(
            custom_model.descriptor().api_mode(),
            Some(OpenAiCompatibleApiMode::Responses.as_str())
        );
        assert_eq!(
            verified_model.descriptor().api_mode(),
            Some(OpenAiCompatibleApiMode::ChatCompletions.as_str())
        );

        Ok(())
    }

    #[test]
    fn compatible_stage_hub_client_and_error_diagnostics_redact_credentials()
    -> Result<(), Box<dyn std::error::Error>> {
        let stage = Siumai::builder()
            .openai_compatible()
            .profile(custom_profile()?)
            .api_key(CREDENTIAL_CANARY);
        assert!(!format!("{stage:?}").contains(CREDENTIAL_CANARY));
        let hub = stage.build()?;
        let client = hub.language("future-compatible-model")?;
        assert!(!format!("{hub:?}").contains(CREDENTIAL_CANARY));
        assert!(!format!("{client:?}").contains(CREDENTIAL_CANARY));

        let error = Siumai::builder()
            .openai_compatible()
            .profile(custom_profile()?)
            .api_key(format!("{CREDENTIAL_CANARY}\n"))
            .build()
            .expect_err("the provider-owned credential validator must reject control characters");
        assert!(!error_diagnostics(&error).contains(CREDENTIAL_CANARY));

        Ok(())
    }
}

#[cfg(feature = "alibaba")]
mod alibaba {
    use siumai::providers::alibaba::{
        AlibabaCredential, AlibabaEmbeddingModel, AlibabaLanguageModel, AlibabaProvider,
        AlibabaProviderBuilder,
    };
    use siumai::{Model, ModelFamily, Siumai};

    use super::error_diagnostics;

    const CREDENTIAL_CANARY: &str = "alibaba-facade-credential-canary";

    fn accepts_provider(provider: &AlibabaProvider) {
        let _ = provider;
    }

    fn accepts_language_model(model: &AlibabaLanguageModel) {
        let _ = model;
    }

    fn accepts_embedding_model(model: &AlibabaEmbeddingModel) {
        let _ = model;
    }

    #[test]
    fn required_configuration_builds_one_multi_family_hub() -> Result<(), Box<dyn std::error::Error>>
    {
        let hub = Siumai::builder()
            .alibaba()
            .credential(AlibabaCredential::api_key("test-api-key"))
            .configure_provider(|builder: AlibabaProviderBuilder| {
                builder
                    .with_legacy_singapore_language()
                    .with_legacy_singapore_embedding()
            })
            .build()?;

        let language = hub.language("future-alibaba-language-model")?;
        let embedding = hub.embedding("future-alibaba-embedding-model")?;
        accepts_provider(hub.provider());
        accepts_language_model(language.model());
        accepts_embedding_model(embedding.model());
        assert!(std::ptr::eq(hub.provider(), language.provider()));
        assert!(std::ptr::eq(hub.provider(), embedding.provider()));
        assert_eq!(language.family(), ModelFamily::Language);
        assert_eq!(embedding.family(), ModelFamily::Embedding);
        assert_eq!(language.descriptor().api_mode(), Some("responses"));
        assert_eq!(embedding.descriptor().api_mode(), Some("text-embedding"));
        assert_eq!(
            language.descriptor().instance_id(),
            embedding.descriptor().instance_id()
        );
        assert_eq!(
            language.descriptor().model().as_str(),
            "future-alibaba-language-model"
        );
        assert_eq!(
            embedding.descriptor().model().as_str(),
            "future-alibaba-embedding-model"
        );

        Ok(())
    }

    #[test]
    fn stages_and_provider_owned_errors_redact_alibaba_credentials() {
        let configuration = Siumai::builder().alibaba().api_key(CREDENTIAL_CANARY);
        assert!(!format!("{configuration:?}").contains(CREDENTIAL_CANARY));

        let hub = configuration
            .configure_provider(|builder| builder.with_legacy_singapore_language())
            .build()
            .expect("the provider-owned legacy endpoint is valid");
        let client = hub
            .language("future-alibaba-canary-model")
            .expect("future model IDs use baseline construction");
        assert!(!format!("{hub:?}").contains(CREDENTIAL_CANARY));
        assert!(!format!("{client:?}").contains(CREDENTIAL_CANARY));

        let error = Siumai::builder()
            .alibaba()
            .api_key(format!("{CREDENTIAL_CANARY}\n"))
            .configure_provider(|builder| builder.with_legacy_singapore_language())
            .build()
            .expect_err("the provider-owned credential validator must reject control characters");
        assert!(!error_diagnostics(&error).contains(CREDENTIAL_CANARY));
    }
}

#[cfg(feature = "moonshotai")]
mod moonshot {
    use siumai::providers::moonshotai::{
        MoonshotCredential, MoonshotLanguageModel, MoonshotProvider, MoonshotProviderBuilder,
    };
    use siumai::{Model, ModelFamily, Siumai};

    #[test]
    fn api_key_and_credential_paths_bind_future_language_models()
    -> Result<(), Box<dyn std::error::Error>> {
        let api_key_hub = Siumai::builder()
            .moonshot()
            .api_key("test-api-key")
            .configure_provider(|builder: MoonshotProviderBuilder| builder)
            .build()?;
        let credential_hub = Siumai::builder()
            .moonshot()
            .credential(MoonshotCredential::api_key("test-api-key"))
            .build()?;

        let client = api_key_hub.language("future-moonshot-model")?;
        let _: &MoonshotProvider = api_key_hub.provider();
        let _: &MoonshotLanguageModel = client.model();
        assert!(std::ptr::eq(api_key_hub.provider(), client.provider()));
        assert_eq!(client.family(), ModelFamily::Language);
        assert_eq!(client.descriptor().api_mode(), Some("chat-completions"));
        assert_eq!(
            client.descriptor().model().as_str(),
            "future-moonshot-model"
        );
        assert_eq!(
            credential_hub
                .language("future-moonshot-credential-model")?
                .descriptor()
                .model()
                .as_str(),
            "future-moonshot-credential-model"
        );

        Ok(())
    }
}

#[cfg(feature = "volcengine")]
mod volcengine {
    use siumai::providers::volcengine::{
        ArkImageModel, VolcengineCredential, VolcengineLanguageModel, VolcengineProvider,
        VolcengineProviderBuilder,
    };
    use siumai::{Model, ModelFamily, Siumai};

    #[test]
    fn one_hub_binds_language_and_image_models() -> Result<(), Box<dyn std::error::Error>> {
        let hub = Siumai::builder()
            .volcengine()
            .api_key("test-api-key")
            .configure_provider(|builder: VolcengineProviderBuilder| builder)
            .build()?;
        let _credential_stage = Siumai::builder()
            .volcengine()
            .credential(VolcengineCredential::api_key("test-api-key"));

        let language = hub.language("future-volcengine-language-model")?;
        let image = hub.image("future-volcengine-image-model")?;
        let _: &VolcengineProvider = hub.provider();
        let _: &VolcengineLanguageModel = language.model();
        let _: &ArkImageModel = image.model();
        assert!(std::ptr::eq(hub.provider(), language.provider()));
        assert!(std::ptr::eq(hub.provider(), image.provider()));
        assert_eq!(language.family(), ModelFamily::Language);
        assert_eq!(image.family(), ModelFamily::Image);
        assert_eq!(language.descriptor().api_mode(), Some("responses"));
        assert_eq!(image.descriptor().api_mode(), Some("images-generations"));
        assert_eq!(
            language.descriptor().instance_id(),
            image.descriptor().instance_id()
        );
        assert_eq!(
            language.descriptor().model().as_str(),
            "future-volcengine-language-model"
        );
        assert_eq!(
            image.descriptor().model().as_str(),
            "future-volcengine-image-model"
        );

        Ok(())
    }
}

#[cfg(feature = "groq")]
mod groq {
    use siumai::providers::groq::{
        GroqCredential, GroqLanguageModel, GroqProvider, GroqProviderBuilder, GroqSpeechModel,
        GroqTranscriptionModel,
    };
    use siumai::{Model, ModelFamily, Siumai};

    #[test]
    fn one_hub_binds_language_speech_and_transcription_models()
    -> Result<(), Box<dyn std::error::Error>> {
        let hub = Siumai::builder()
            .groq()
            .api_key("test-api-key")
            .configure_provider(|builder: GroqProviderBuilder| builder)
            .build()?;
        let _credential_stage = Siumai::builder()
            .groq()
            .credential(GroqCredential::api_key("test-api-key"));

        let language = hub.language("future-groq-language-model")?;
        let speech = hub.speech("future-groq-speech-model")?;
        let transcription = hub.transcription("future-groq-transcription-model")?;
        let _: &GroqProvider = hub.provider();
        let _: &GroqLanguageModel = language.model();
        let _: &GroqSpeechModel = speech.model();
        let _: &GroqTranscriptionModel = transcription.model();
        assert!(std::ptr::eq(hub.provider(), language.provider()));
        assert!(std::ptr::eq(hub.provider(), speech.provider()));
        assert!(std::ptr::eq(hub.provider(), transcription.provider()));
        assert_eq!(language.family(), ModelFamily::Language);
        assert_eq!(speech.family(), ModelFamily::Speech);
        assert_eq!(transcription.family(), ModelFamily::Transcription);
        assert_eq!(language.descriptor().api_mode(), Some("chat-completions"));
        assert_eq!(speech.descriptor().api_mode(), Some("audio-speech"));
        assert_eq!(
            transcription.descriptor().api_mode(),
            Some("audio-transcriptions")
        );
        assert_eq!(
            language.descriptor().instance_id(),
            speech.descriptor().instance_id()
        );
        assert_eq!(
            language.descriptor().instance_id(),
            transcription.descriptor().instance_id()
        );
        assert_eq!(
            language.descriptor().model().as_str(),
            "future-groq-language-model"
        );
        assert_eq!(
            speech.descriptor().model().as_str(),
            "future-groq-speech-model"
        );
        assert_eq!(
            transcription.descriptor().model().as_str(),
            "future-groq-transcription-model"
        );

        Ok(())
    }
}

#[cfg(feature = "xai")]
mod xai {
    use siumai::providers::xai::models;
    use siumai::providers::xai::{
        XaiCredential, XaiImageModel, XaiLanguageModel, XaiProvider, XaiProviderBuilder,
        XaiSpeechModel, XaiTranscriptionModel,
    };
    use siumai::{Model, ModelFamily, Siumai};

    #[test]
    fn one_hub_binds_all_portable_xai_families() -> Result<(), Box<dyn std::error::Error>> {
        let hub = Siumai::builder()
            .xai()
            .api_key("test-api-key")
            .configure_provider(|builder: XaiProviderBuilder| builder)
            .build()?;
        let _credential_stage = Siumai::builder()
            .xai()
            .credential(XaiCredential::api_key("test-api-key"));

        let language = hub.language("future-xai-language-model")?;
        let image = hub.image("future-xai-image-model")?;
        let speech = hub.speech(models::speech::TTS)?;
        let transcription = hub.transcription(models::transcription::STT)?;
        let _: &XaiProvider = hub.provider();
        let _: &XaiLanguageModel = language.model();
        let _: &XaiImageModel = image.model();
        let _: &XaiSpeechModel = speech.model();
        let _: &XaiTranscriptionModel = transcription.model();
        assert!(std::ptr::eq(hub.provider(), language.provider()));
        assert!(std::ptr::eq(hub.provider(), image.provider()));
        assert!(std::ptr::eq(hub.provider(), speech.provider()));
        assert!(std::ptr::eq(hub.provider(), transcription.provider()));
        assert_eq!(language.family(), ModelFamily::Language);
        assert_eq!(image.family(), ModelFamily::Image);
        assert_eq!(speech.family(), ModelFamily::Speech);
        assert_eq!(transcription.family(), ModelFamily::Transcription);
        assert_eq!(language.descriptor().api_mode(), Some("responses"));
        assert_eq!(image.descriptor().api_mode(), Some("image-generations"));
        assert_eq!(speech.descriptor().api_mode(), Some("tts"));
        assert_eq!(transcription.descriptor().api_mode(), Some("stt"));
        assert_eq!(
            language.descriptor().instance_id(),
            image.descriptor().instance_id()
        );
        assert_eq!(
            language.descriptor().instance_id(),
            speech.descriptor().instance_id()
        );
        assert_eq!(
            language.descriptor().instance_id(),
            transcription.descriptor().instance_id()
        );
        assert_eq!(
            language.descriptor().model().as_str(),
            "future-xai-language-model"
        );
        assert_eq!(
            image.descriptor().model().as_str(),
            "future-xai-image-model"
        );

        Ok(())
    }
}

#[cfg(feature = "minimax")]
mod minimax {
    use siumai::providers::minimax::{
        MinimaxCredential, MinimaxImageModel, MinimaxLanguageModel, MinimaxProvider,
        MinimaxProviderBuilder, MinimaxSpeechModel,
    };
    use siumai::{Model, ModelFamily, Siumai};

    #[test]
    fn one_hub_binds_language_image_and_speech_models() -> Result<(), Box<dyn std::error::Error>> {
        let hub = Siumai::builder()
            .minimax()
            .api_key("test-api-key")
            .configure_provider(|builder: MinimaxProviderBuilder| builder)
            .build()?;
        let _credential_stage = Siumai::builder()
            .minimax()
            .credential(MinimaxCredential::api_key("test-api-key"));

        let language = hub.language("future-minimax-language-model")?;
        let image = hub.image("future-minimax-image-model")?;
        let speech = hub.speech("future-minimax-speech-model")?;
        let _: &MinimaxProvider = hub.provider();
        let _: &MinimaxLanguageModel = language.model();
        let _: &MinimaxImageModel = image.model();
        let _: &MinimaxSpeechModel = speech.model();
        assert!(std::ptr::eq(hub.provider(), language.provider()));
        assert!(std::ptr::eq(hub.provider(), image.provider()));
        assert!(std::ptr::eq(hub.provider(), speech.provider()));
        assert_eq!(language.family(), ModelFamily::Language);
        assert_eq!(image.family(), ModelFamily::Image);
        assert_eq!(speech.family(), ModelFamily::Speech);
        assert_eq!(language.descriptor().api_mode(), Some("messages"));
        assert_eq!(image.descriptor().api_mode(), Some("image-generation"));
        assert_eq!(speech.descriptor().api_mode(), Some("speech-http"));
        assert_eq!(
            language.descriptor().instance_id(),
            image.descriptor().instance_id()
        );
        assert_eq!(
            language.descriptor().instance_id(),
            speech.descriptor().instance_id()
        );
        assert_eq!(
            language.descriptor().model().as_str(),
            "future-minimax-language-model"
        );
        assert_eq!(
            image.descriptor().model().as_str(),
            "future-minimax-image-model"
        );
        assert_eq!(
            speech.descriptor().model().as_str(),
            "future-minimax-speech-model"
        );

        Ok(())
    }
}

#[cfg(feature = "deepseek")]
mod deepseek {
    use siumai::providers::deepseek::{
        DeepSeekCredential, DeepSeekLanguageModel, DeepSeekProvider, DeepSeekProviderBuilder,
    };
    use siumai::{Model, ModelFamily, Siumai};

    #[test]
    fn api_key_and_credential_paths_bind_future_language_models()
    -> Result<(), Box<dyn std::error::Error>> {
        let api_key_hub = Siumai::builder()
            .deepseek()
            .api_key("test-api-key")
            .configure_provider(|builder: DeepSeekProviderBuilder| builder)
            .build()?;
        let credential_hub = Siumai::builder()
            .deepseek()
            .credential(DeepSeekCredential::api_key("test-api-key"))
            .build()?;

        let client = api_key_hub.language("future-deepseek-model")?;
        let _: &DeepSeekProvider = api_key_hub.provider();
        let _: &DeepSeekLanguageModel = client.model();
        assert!(std::ptr::eq(api_key_hub.provider(), client.provider()));
        assert_eq!(client.family(), ModelFamily::Language);
        assert_eq!(client.descriptor().api_mode(), Some("chat-completions"));
        assert_eq!(
            client.descriptor().model().as_str(),
            "future-deepseek-model"
        );
        assert_eq!(
            credential_hub
                .language("future-deepseek-credential-model")?
                .descriptor()
                .model()
                .as_str(),
            "future-deepseek-credential-model"
        );

        Ok(())
    }
}

#[cfg(feature = "cohere")]
mod cohere {
    use siumai::providers::cohere::{
        CohereEmbeddingModel, CohereProvider, CohereProviderBuilder, CohereRerankModel,
    };
    use siumai::{Model, ModelFamily, Siumai};

    #[test]
    fn api_key_path_binds_embedding_and_rerank_models() -> Result<(), Box<dyn std::error::Error>> {
        let hub = Siumai::builder()
            .cohere()
            .api_key("test-api-key")
            .configure_provider(|builder: CohereProviderBuilder| builder)
            .build()?;

        let embedding = hub.embedding("future-cohere-embedding-model")?;
        let rerank = hub.rerank("future-cohere-rerank-model")?;
        let _: &CohereProvider = hub.provider();
        let _: &CohereEmbeddingModel = embedding.model();
        let _: &CohereRerankModel = rerank.model();
        assert!(std::ptr::eq(hub.provider(), embedding.provider()));
        assert!(std::ptr::eq(hub.provider(), rerank.provider()));
        assert_eq!(embedding.family(), ModelFamily::Embedding);
        assert_eq!(rerank.family(), ModelFamily::Rerank);
        assert_eq!(embedding.descriptor().api_mode(), Some("v2"));
        assert_eq!(rerank.descriptor().api_mode(), Some("v2"));
        assert_eq!(
            embedding.descriptor().instance_id(),
            rerank.descriptor().instance_id()
        );
        assert_eq!(
            embedding.descriptor().model().as_str(),
            "future-cohere-embedding-model"
        );
        assert_eq!(
            rerank.descriptor().model().as_str(),
            "future-cohere-rerank-model"
        );

        Ok(())
    }
}

#[cfg(feature = "deepgram")]
mod deepgram {
    use siumai::providers::deepgram::{
        DeepgramCredential, DeepgramProvider, DeepgramProviderBuilder, DeepgramSpeechModel,
        DeepgramTranscriptionModel,
    };
    use siumai::{Model, ModelFamily, Siumai};

    #[test]
    fn one_hub_binds_speech_and_transcription_models() -> Result<(), Box<dyn std::error::Error>> {
        let hub = Siumai::builder()
            .deepgram()
            .api_key("test-api-key")
            .configure_provider(|builder: DeepgramProviderBuilder| builder)
            .build()?;
        let _credential_stage = Siumai::builder()
            .deepgram()
            .credential(DeepgramCredential::api_key("test-api-key"));

        let speech = hub.speech("future-deepgram-speech-model")?;
        let transcription = hub.transcription("future-deepgram-transcription-model")?;
        let _: &DeepgramProvider = hub.provider();
        let _: &DeepgramSpeechModel = speech.model();
        let _: &DeepgramTranscriptionModel = transcription.model();
        assert!(std::ptr::eq(hub.provider(), speech.provider()));
        assert!(std::ptr::eq(hub.provider(), transcription.provider()));
        assert_eq!(speech.family(), ModelFamily::Speech);
        assert_eq!(transcription.family(), ModelFamily::Transcription);
        assert_eq!(speech.descriptor().api_mode(), Some("tts"));
        assert_eq!(transcription.descriptor().api_mode(), Some("prerecorded"));
        assert_eq!(
            speech.descriptor().instance_id(),
            transcription.descriptor().instance_id()
        );
        assert_eq!(
            speech.descriptor().model().as_str(),
            "future-deepgram-speech-model"
        );
        assert_eq!(
            transcription.descriptor().model().as_str(),
            "future-deepgram-transcription-model"
        );

        Ok(())
    }
}

#[cfg(feature = "google-vertex-anthropic")]
mod vertex_anthropic {
    use siumai::providers::google_vertex_anthropic::{
        GOOGLE_VERTEX_ANTHROPIC_REPLAY_AUDIENCE, GoogleVertexAnthropicLanguageModel,
        GoogleVertexAnthropicProvider, GoogleVertexAnthropicProviderBuilder,
        GoogleVertexCredential,
    };
    use siumai::{Model, ModelFamily, ReplayDomain, ReplayDomainId, Siumai};

    use super::error_diagnostics;

    const CREDENTIAL_CANARY: &str = "vertex-anthropic-facade-credential-canary";

    fn replay_domain(label: &str) -> ReplayDomain {
        ReplayDomain::official(
            ReplayDomainId::new(GOOGLE_VERTEX_ANTHROPIC_REPLAY_AUDIENCE)
                .expect("the official replay audience is valid"),
        )
        .with_caller_scope(
            ReplayDomainId::new(label).expect("the test caller scope is a valid identifier"),
        )
    }

    #[test]
    fn ordered_inputs_and_access_token_sugar_build_language_hubs()
    -> Result<(), Box<dyn std::error::Error>> {
        let access_token_hub = Siumai::builder()
            .vertex_anthropic()
            .project("test-project")
            .location("global")
            .access_token("test-access-token")
            .configure_provider(|builder: GoogleVertexAnthropicProviderBuilder| {
                builder.with_replay_domain(replay_domain("vertex-access-token-test"))
            })
            .build()?;
        let credential_hub = Siumai::builder()
            .vertex_anthropic()
            .project("test-project")
            .location("global")
            .credential(GoogleVertexCredential::access_token("test-access-token"))
            .configure_provider(|builder| {
                builder.with_replay_domain(replay_domain("vertex-credential-test"))
            })
            .build()?;

        let client = access_token_hub.language("future-vertex-anthropic-model")?;
        let _: &GoogleVertexAnthropicProvider = access_token_hub.provider();
        let _: &GoogleVertexAnthropicLanguageModel = client.model();
        assert!(std::ptr::eq(access_token_hub.provider(), client.provider()));
        assert_eq!(client.family(), ModelFamily::Language);
        assert_eq!(client.descriptor().api_mode(), Some("messages"));
        assert_eq!(
            client.descriptor().model().as_str(),
            "future-vertex-anthropic-model"
        );
        assert_eq!(
            credential_hub
                .language("future-vertex-credential-model")?
                .descriptor()
                .model()
                .as_str(),
            "future-vertex-credential-model"
        );

        Ok(())
    }

    #[test]
    fn stages_hubs_clients_and_error_chains_redact_vertex_credentials()
    -> Result<(), Box<dyn std::error::Error>> {
        let stage = Siumai::builder()
            .vertex_anthropic()
            .project("private-project-canary")
            .location("global")
            .access_token(CREDENTIAL_CANARY)
            .configure_provider(|builder| {
                builder.with_replay_domain(replay_domain("vertex-canary-test"))
            });
        assert!(!format!("{stage:?}").contains(CREDENTIAL_CANARY));
        let hub = stage.build()?;
        let client = hub.language("future-vertex-canary-model")?;
        assert!(!format!("{hub:?}").contains(CREDENTIAL_CANARY));
        assert!(!format!("{client:?}").contains(CREDENTIAL_CANARY));

        let error = Siumai::builder()
            .vertex_anthropic()
            .project("private-project-canary")
            .location("global")
            .access_token(format!("{CREDENTIAL_CANARY}\n"))
            .configure_provider(|builder| {
                builder.with_replay_domain(replay_domain("vertex-error-test"))
            })
            .build()
            .expect_err("the provider-owned credential validator must reject control characters");
        assert!(!error_diagnostics(&error).contains(CREDENTIAL_CANARY));

        Ok(())
    }
}

#[cfg(feature = "elevenlabs")]
mod elevenlabs {
    use siumai::providers::elevenlabs::{
        ElevenLabsCredential, ElevenLabsProfile, ElevenLabsProvider, ElevenLabsProviderBuilder,
        ElevenLabsSpeechModel, ElevenLabsTranscriptionModel,
    };
    use siumai::{Model, ModelFamily, Siumai};

    use super::error_diagnostics;

    const CREDENTIAL_CANARY: &str = "elevenlabs-facade-credential-canary";

    #[test]
    fn profile_then_credential_binds_speech_and_transcription_models()
    -> Result<(), Box<dyn std::error::Error>> {
        let hub = Siumai::builder()
            .elevenlabs()
            .profile(ElevenLabsProfile::official()?)
            .credential(ElevenLabsCredential::api_key("test-api-key"))
            .configure_provider(|builder: ElevenLabsProviderBuilder| builder)
            .build()?;

        let speech = hub.speech("future-elevenlabs-speech-model")?;
        let transcription = hub.transcription("future-elevenlabs-transcription-model")?;
        let _: &ElevenLabsProvider = hub.provider();
        let _: &ElevenLabsSpeechModel = speech.model();
        let _: &ElevenLabsTranscriptionModel = transcription.model();
        assert!(std::ptr::eq(hub.provider(), speech.provider()));
        assert!(std::ptr::eq(hub.provider(), transcription.provider()));
        assert_eq!(speech.family(), ModelFamily::Speech);
        assert_eq!(transcription.family(), ModelFamily::Transcription);
        assert_eq!(speech.descriptor().api_mode(), Some("text-to-speech"));
        assert_eq!(
            transcription.descriptor().api_mode(),
            Some("batch-transcription")
        );
        assert_eq!(
            speech.descriptor().instance_id(),
            transcription.descriptor().instance_id()
        );
        assert_eq!(
            speech.descriptor().model().as_str(),
            "future-elevenlabs-speech-model"
        );
        assert_eq!(
            transcription.descriptor().model().as_str(),
            "future-elevenlabs-transcription-model"
        );

        Ok(())
    }

    #[test]
    fn stages_hubs_clients_and_error_chains_redact_elevenlabs_credentials()
    -> Result<(), Box<dyn std::error::Error>> {
        let stage = Siumai::builder()
            .elevenlabs()
            .profile(ElevenLabsProfile::official()?)
            .api_key(CREDENTIAL_CANARY);
        assert!(!format!("{stage:?}").contains(CREDENTIAL_CANARY));
        let hub = stage.build()?;
        let client = hub.speech("future-elevenlabs-canary-model")?;
        assert!(!format!("{hub:?}").contains(CREDENTIAL_CANARY));
        assert!(!format!("{client:?}").contains(CREDENTIAL_CANARY));

        let error = Siumai::builder()
            .elevenlabs()
            .profile(ElevenLabsProfile::official()?)
            .api_key(format!("{CREDENTIAL_CANARY}\n"))
            .build()
            .expect_err("the provider-owned credential validator must reject control characters");
        assert!(!error_diagnostics(&error).contains(CREDENTIAL_CANARY));

        Ok(())
    }
}

#[cfg(all(feature = "groq", feature = "openai-compatible"))]
mod branded_compatible_options {
    use siumai::providers::groq::options::GroqLanguageOptions;
    use siumai::{CallOptions, ProviderOptionError, Siumai};

    use super::openai_compatible::custom_profile;

    #[test]
    fn branded_options_cannot_cross_into_a_compatible_profile()
    -> Result<(), Box<dyn std::error::Error>> {
        let groq = Siumai::builder()
            .groq()
            .api_key("test-api-key")
            .build()?
            .language("future-groq-model")?;
        let compatible = Siumai::builder()
            .openai_compatible()
            .profile(custom_profile()?)
            .api_key("test-api-key")
            .build()?
            .language("future-compatible-model")?;
        let foreign = CallOptions::default()
            .with_provider_options_for(groq.model(), &GroqLanguageOptions::default())?;

        let error = match compatible.call("hello").with_options(foreign) {
            Ok(_) => panic!("Groq options must not target an explicit compatible profile"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            ProviderOptionError::ExactTargetMismatch { .. }
        ));

        Ok(())
    }
}
