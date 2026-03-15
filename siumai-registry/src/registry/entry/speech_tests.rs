use super::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone)]
struct BridgeSpeechClient;

#[async_trait::async_trait]
impl crate::traits::SpeechCapability for BridgeSpeechClient {
    async fn tts(
        &self,
        request: crate::types::TtsRequest,
    ) -> Result<crate::types::TtsResponse, LlmError> {
        Ok(crate::types::TtsResponse {
            audio_data: request.text.into_bytes(),
            format: "pcm".to_string(),
            duration: None,
            sample_rate: None,
            metadata: HashMap::new(),
        })
    }
}

impl LlmClient for BridgeSpeechClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("testprov_speech")
    }

    fn supported_models(&self) -> Vec<String> {
        vec!["speech-model".into()]
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new().with_speech()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }

    fn as_speech_capability(&self) -> Option<&dyn crate::traits::SpeechCapability> {
        Some(self)
    }
}

struct BridgeSpeechFactory;

#[async_trait::async_trait]
impl ProviderFactory for BridgeSpeechFactory {
    async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        Ok(Arc::new(BridgeSpeechClient))
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("testprov_speech")
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_speech()
    }
}

#[test]
fn speech_model_handle_implements_model_metadata() {
    let mut providers = HashMap::new();
    providers.insert(
        "testprov_speech".to_string(),
        Arc::new(BridgeSpeechFactory) as Arc<dyn ProviderFactory>,
    );

    let reg = create_provider_registry(providers, None);
    let handle = reg.speech_model("testprov_speech:model").unwrap();

    fn assert_speech_model<M>(model: &M)
    where
        M: siumai_core::speech::SpeechModel + ?Sized,
    {
        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model),
            "testprov_speech"
        );
        assert_eq!(crate::traits::ModelMetadata::model_id(model), "model");
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model),
            crate::traits::ModelSpecVersion::V1
        );
    }

    assert_speech_model(&handle);
}

#[test]
fn speech_model_handle_rejects_provider_without_speech_capability() {
    let mut providers = HashMap::new();
    providers.insert(
        "testprov_chat".to_string(),
        Arc::new(TestProviderFactory::new("testprov_chat")) as Arc<dyn ProviderFactory>,
    );
    let reg = create_provider_registry(providers, None);

    let err = match reg.speech_model("testprov_chat:model") {
        Ok(_) => panic!("speech handle should be rejected without speech capability"),
        Err(err) => err,
    };

    assert!(
        matches!(err, LlmError::UnsupportedOperation(message) if message.contains("speech_model handle"))
    );
}

#[tokio::test]
async fn provider_factory_speech_family_bridge_works() {
    let factory = BridgeSpeechFactory;
    let model = factory
        .speech_model_family("bridged-speech-model")
        .await
        .unwrap();

    assert_eq!(
        crate::traits::ModelMetadata::provider_id(model.as_ref()),
        "testprov_speech"
    );
    assert_eq!(
        crate::traits::ModelMetadata::model_id(model.as_ref()),
        "bridged-speech-model"
    );

    let response = model
        .synthesize(crate::types::TtsRequest::new("hello".to_string()))
        .await
        .unwrap();
    assert_eq!(response.audio_data, b"hello");
    assert_eq!(response.format, "pcm");
}

#[tokio::test]
async fn provider_factory_native_speech_family_path_works() {
    #[derive(Clone)]
    struct NativeSpeechModel;

    impl crate::traits::ModelMetadata for NativeSpeechModel {
        fn provider_id(&self) -> &str {
            "native-speech"
        }

        fn model_id(&self) -> &str {
            "native-speech-model"
        }
    }

    #[async_trait::async_trait]
    impl siumai_core::speech::SpeechModelV3 for NativeSpeechModel {
        async fn synthesize(
            &self,
            request: crate::types::TtsRequest,
        ) -> Result<crate::types::TtsResponse, LlmError> {
            Ok(crate::types::TtsResponse {
                audio_data: request.text.into_bytes(),
                format: "wav".to_string(),
                duration: None,
                sample_rate: Some(16_000),
                metadata: HashMap::new(),
            })
        }
    }

    struct NativeOnlySpeechFactory;

    #[async_trait::async_trait]
    impl ProviderFactory for NativeOnlySpeechFactory {
        async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
            panic!("legacy generic-client path should not be used by native speech-family test")
        }

        async fn speech_model_family_with_ctx(
            &self,
            _model_id: &str,
            _ctx: &BuildContext,
        ) -> Result<Arc<dyn siumai_core::speech::SpeechModel>, LlmError> {
            Ok(Arc::new(NativeSpeechModel))
        }

        fn provider_id(&self) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("native-speech")
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_speech()
        }
    }

    let factory = NativeOnlySpeechFactory;
    let model = factory
        .speech_model_family("native-speech-model")
        .await
        .unwrap();

    assert_eq!(
        crate::traits::ModelMetadata::provider_id(model.as_ref()),
        "native-speech"
    );
    assert_eq!(
        crate::traits::ModelMetadata::model_id(model.as_ref()),
        "native-speech-model"
    );

    let response = model
        .synthesize(crate::types::TtsRequest::new("native".to_string()))
        .await
        .unwrap();
    assert_eq!(response.audio_data, b"native");
    assert_eq!(response.format, "wav");
}

#[tokio::test]
async fn speech_model_handle_uses_native_family_path_when_available() {
    #[derive(Clone)]
    struct NativeHandleSpeechModel;

    impl crate::traits::ModelMetadata for NativeHandleSpeechModel {
        fn provider_id(&self) -> &str {
            "native-speech-handle"
        }

        fn model_id(&self) -> &str {
            "native-speech-handle-model"
        }
    }

    #[async_trait::async_trait]
    impl siumai_core::speech::SpeechModelV3 for NativeHandleSpeechModel {
        async fn synthesize(
            &self,
            request: crate::types::TtsRequest,
        ) -> Result<crate::types::TtsResponse, LlmError> {
            assert_eq!(request.model.as_deref(), Some("model"));
            Ok(crate::types::TtsResponse {
                audio_data: request.text.into_bytes(),
                format: "ogg".to_string(),
                duration: None,
                sample_rate: Some(24_000),
                metadata: HashMap::new(),
            })
        }
    }

    struct NativeHandleSpeechFactory;

    #[async_trait::async_trait]
    impl ProviderFactory for NativeHandleSpeechFactory {
        async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
            panic!("legacy generic-client path should not be used by speech handle")
        }

        async fn speech_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
            panic!("legacy speech client path should not be used by speech handle")
        }

        async fn speech_model_family_with_ctx(
            &self,
            _model_id: &str,
            _ctx: &BuildContext,
        ) -> Result<Arc<dyn siumai_core::speech::SpeechModel>, LlmError> {
            Ok(Arc::new(NativeHandleSpeechModel))
        }

        fn provider_id(&self) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("native-speech-handle")
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_speech()
        }
    }

    let mut providers = HashMap::new();
    providers.insert(
        "native-speech-handle".to_string(),
        Arc::new(NativeHandleSpeechFactory) as Arc<dyn ProviderFactory>,
    );
    let reg = create_provider_registry(providers, None);
    let handle = reg.speech_model("native-speech-handle:model").unwrap();

    let response = siumai_core::speech::SpeechModelV3::synthesize(
        &handle,
        crate::types::TtsRequest::new("family-path".to_string()),
    )
    .await
    .unwrap();
    assert_eq!(response.audio_data, b"family-path");
    assert_eq!(response.format, "ogg");
}

#[tokio::test]
async fn speech_model_handle_reuses_cached_family_model() {
    #[derive(Clone)]
    struct CountingSpeechModel;

    impl crate::traits::ModelMetadata for CountingSpeechModel {
        fn provider_id(&self) -> &str {
            "cached-speech"
        }

        fn model_id(&self) -> &str {
            "cached-speech-model"
        }
    }

    #[async_trait::async_trait]
    impl siumai_core::speech::SpeechModelV3 for CountingSpeechModel {
        async fn synthesize(
            &self,
            request: crate::types::TtsRequest,
        ) -> Result<crate::types::TtsResponse, LlmError> {
            Ok(crate::types::TtsResponse {
                audio_data: request.text.into_bytes(),
                format: "pcm".to_string(),
                duration: None,
                sample_rate: None,
                metadata: HashMap::new(),
            })
        }
    }

    struct CountingSpeechFactory {
        builds: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl ProviderFactory for CountingSpeechFactory {
        async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
            panic!("legacy generic-client path should not be used by speech cache test")
        }

        async fn speech_model_family_with_ctx(
            &self,
            _model_id: &str,
            _ctx: &BuildContext,
        ) -> Result<Arc<dyn siumai_core::speech::SpeechModel>, LlmError> {
            self.builds.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::new(CountingSpeechModel))
        }

        fn provider_id(&self) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("cached-speech")
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_speech()
        }
    }

    let builds = Arc::new(AtomicUsize::new(0));
    let mut providers = HashMap::new();
    providers.insert(
        "cached-speech".to_string(),
        Arc::new(CountingSpeechFactory {
            builds: builds.clone(),
        }) as Arc<dyn ProviderFactory>,
    );
    let reg = create_provider_registry(providers, None);
    let handle = reg.speech_model("cached-speech:model").unwrap();

    let first = siumai_core::speech::SpeechModelV3::synthesize(
        &handle,
        crate::types::TtsRequest::new("once".to_string()),
    )
    .await
    .unwrap();
    let second = siumai_core::speech::SpeechModelV3::synthesize(
        &handle,
        crate::types::TtsRequest::new("twice".to_string()),
    )
    .await
    .unwrap();

    assert_eq!(first.audio_data, b"once");
    assert_eq!(second.audio_data, b"twice");
    assert_eq!(builds.load(Ordering::SeqCst), 1);
}
