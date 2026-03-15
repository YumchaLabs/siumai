use super::*;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
struct BridgeTranscriptionClient;

#[async_trait::async_trait]
impl crate::traits::TranscriptionCapability for BridgeTranscriptionClient {
    async fn stt(
        &self,
        _request: crate::types::SttRequest,
    ) -> Result<crate::types::SttResponse, LlmError> {
        Ok(crate::types::SttResponse {
            text: "bridge transcription".to_string(),
            language: Some("en".to_string()),
            confidence: Some(0.99),
            words: None,
            duration: None,
            metadata: HashMap::new(),
        })
    }
}

impl LlmClient for BridgeTranscriptionClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("testprov_transcription")
    }

    fn supported_models(&self) -> Vec<String> {
        vec!["transcription-model".into()]
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new().with_transcription()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }

    fn as_transcription_capability(&self) -> Option<&dyn crate::traits::TranscriptionCapability> {
        Some(self)
    }
}

struct BridgeTranscriptionFactory;

#[async_trait::async_trait]
impl ProviderFactory for BridgeTranscriptionFactory {
    async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        Ok(Arc::new(BridgeTranscriptionClient))
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("testprov_transcription")
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_transcription()
    }
}

#[test]
fn transcription_model_handle_implements_model_metadata() {
    let mut providers = HashMap::new();
    providers.insert(
        "testprov_transcription".to_string(),
        Arc::new(BridgeTranscriptionFactory) as Arc<dyn ProviderFactory>,
    );

    let reg = create_provider_registry(providers, None);
    let handle = reg
        .transcription_model("testprov_transcription:model")
        .unwrap();

    fn assert_transcription_model<M>(model: &M)
    where
        M: siumai_core::transcription::TranscriptionModel + ?Sized,
    {
        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model),
            "testprov_transcription"
        );
        assert_eq!(crate::traits::ModelMetadata::model_id(model), "model");
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model),
            crate::traits::ModelSpecVersion::V1
        );
    }

    assert_transcription_model(&handle);
}

#[test]
fn transcription_model_handle_rejects_provider_without_transcription_capability() {
    let mut providers = HashMap::new();
    providers.insert(
        "testprov_chat".to_string(),
        Arc::new(TestProviderFactory::new("testprov_chat")) as Arc<dyn ProviderFactory>,
    );
    let reg = create_provider_registry(providers, None);

    let err = match reg.transcription_model("testprov_chat:model") {
        Ok(_) => panic!("transcription handle should be rejected without transcription capability"),
        Err(err) => err,
    };

    assert!(
        matches!(err, LlmError::UnsupportedOperation(message) if message.contains("transcription_model handle"))
    );
}

#[tokio::test]
async fn provider_factory_transcription_family_bridge_works() {
    let factory = BridgeTranscriptionFactory;
    let model = factory
        .transcription_model_family("bridged-transcription-model")
        .await
        .unwrap();

    assert_eq!(
        crate::traits::ModelMetadata::provider_id(model.as_ref()),
        "testprov_transcription"
    );
    assert_eq!(
        crate::traits::ModelMetadata::model_id(model.as_ref()),
        "bridged-transcription-model"
    );

    let response = model
        .transcribe(crate::types::SttRequest::from_audio(Vec::new()))
        .await
        .unwrap();
    assert_eq!(response.text, "bridge transcription");
    assert_eq!(response.language.as_deref(), Some("en"));
}

#[tokio::test]
async fn provider_factory_native_transcription_family_path_works() {
    #[derive(Clone)]
    struct NativeTranscriptionModel;

    impl crate::traits::ModelMetadata for NativeTranscriptionModel {
        fn provider_id(&self) -> &str {
            "native-transcription"
        }

        fn model_id(&self) -> &str {
            "native-transcription-model"
        }
    }

    #[async_trait::async_trait]
    impl siumai_core::transcription::TranscriptionModelV3 for NativeTranscriptionModel {
        async fn transcribe(
            &self,
            _request: crate::types::SttRequest,
        ) -> Result<crate::types::SttResponse, LlmError> {
            Ok(crate::types::SttResponse {
                text: "native transcription".to_string(),
                language: Some("zh".to_string()),
                confidence: Some(0.9),
                words: None,
                duration: None,
                metadata: HashMap::new(),
            })
        }
    }

    struct NativeOnlyTranscriptionFactory;

    #[async_trait::async_trait]
    impl ProviderFactory for NativeOnlyTranscriptionFactory {
        async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
            panic!(
                "legacy generic-client path should not be used by native transcription-family test"
            )
        }

        async fn transcription_model_family_with_ctx(
            &self,
            _model_id: &str,
            _ctx: &BuildContext,
        ) -> Result<Arc<dyn siumai_core::transcription::TranscriptionModel>, LlmError> {
            Ok(Arc::new(NativeTranscriptionModel))
        }

        fn provider_id(&self) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("native-transcription")
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_transcription()
        }
    }

    let factory = NativeOnlyTranscriptionFactory;
    let model = factory
        .transcription_model_family("native-transcription-model")
        .await
        .unwrap();

    assert_eq!(
        crate::traits::ModelMetadata::provider_id(model.as_ref()),
        "native-transcription"
    );
    assert_eq!(
        crate::traits::ModelMetadata::model_id(model.as_ref()),
        "native-transcription-model"
    );

    let response = model
        .transcribe(crate::types::SttRequest::from_audio(Vec::new()))
        .await
        .unwrap();
    assert_eq!(response.text, "native transcription");
    assert_eq!(response.language.as_deref(), Some("zh"));
}

#[tokio::test]
async fn transcription_model_handle_uses_native_family_path_when_available() {
    #[derive(Clone)]
    struct NativeHandleTranscriptionModel;

    impl crate::traits::ModelMetadata for NativeHandleTranscriptionModel {
        fn provider_id(&self) -> &str {
            "native-transcription-handle"
        }

        fn model_id(&self) -> &str {
            "native-transcription-handle-model"
        }
    }

    #[async_trait::async_trait]
    impl siumai_core::transcription::TranscriptionModelV3 for NativeHandleTranscriptionModel {
        async fn transcribe(
            &self,
            request: crate::types::SttRequest,
        ) -> Result<crate::types::SttResponse, LlmError> {
            assert_eq!(request.model.as_deref(), Some("model"));
            Ok(crate::types::SttResponse {
                text: "native handle transcription".to_string(),
                language: Some("ja".to_string()),
                confidence: Some(0.88),
                words: None,
                duration: None,
                metadata: HashMap::new(),
            })
        }
    }

    struct NativeHandleTranscriptionFactory;

    #[async_trait::async_trait]
    impl ProviderFactory for NativeHandleTranscriptionFactory {
        async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
            panic!("legacy generic-client path should not be used by transcription handle")
        }

        async fn transcription_model(
            &self,
            _model_id: &str,
        ) -> Result<Arc<dyn LlmClient>, LlmError> {
            panic!("legacy transcription client path should not be used by transcription handle")
        }

        async fn transcription_model_family_with_ctx(
            &self,
            _model_id: &str,
            _ctx: &BuildContext,
        ) -> Result<Arc<dyn siumai_core::transcription::TranscriptionModel>, LlmError> {
            Ok(Arc::new(NativeHandleTranscriptionModel))
        }

        fn provider_id(&self) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("native-transcription-handle")
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_transcription()
        }
    }

    let mut providers = HashMap::new();
    providers.insert(
        "native-transcription-handle".to_string(),
        Arc::new(NativeHandleTranscriptionFactory) as Arc<dyn ProviderFactory>,
    );
    let reg = create_provider_registry(providers, None);
    let handle = reg
        .transcription_model("native-transcription-handle:model")
        .unwrap();

    let response = siumai_core::transcription::TranscriptionModelV3::transcribe(
        &handle,
        crate::types::SttRequest::from_audio(Vec::new()),
    )
    .await
    .unwrap();
    assert_eq!(response.text, "native handle transcription");
    assert_eq!(response.language.as_deref(), Some("ja"));
}

#[tokio::test]
async fn transcription_model_handle_reuses_cached_family_model() {
    #[derive(Clone)]
    struct CountingTranscriptionModel;

    impl crate::traits::ModelMetadata for CountingTranscriptionModel {
        fn provider_id(&self) -> &str {
            "cached-transcription"
        }

        fn model_id(&self) -> &str {
            "cached-transcription-model"
        }
    }

    #[async_trait::async_trait]
    impl siumai_core::transcription::TranscriptionModelV3 for CountingTranscriptionModel {
        async fn transcribe(
            &self,
            _request: crate::types::SttRequest,
        ) -> Result<crate::types::SttResponse, LlmError> {
            Ok(crate::types::SttResponse {
                text: "cached transcription".to_string(),
                language: Some("en".to_string()),
                confidence: Some(1.0),
                words: None,
                duration: None,
                metadata: HashMap::new(),
            })
        }
    }

    struct CountingTranscriptionFactory {
        builds: Arc<std::sync::Mutex<usize>>,
    }

    #[async_trait::async_trait]
    impl ProviderFactory for CountingTranscriptionFactory {
        async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
            panic!("legacy generic-client path should not be used by transcription cache test")
        }

        async fn transcription_model_family_with_ctx(
            &self,
            _model_id: &str,
            _ctx: &BuildContext,
        ) -> Result<Arc<dyn siumai_core::transcription::TranscriptionModel>, LlmError> {
            *self.builds.lock().unwrap() += 1;
            Ok(Arc::new(CountingTranscriptionModel))
        }

        fn provider_id(&self) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("cached-transcription")
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_transcription()
        }
    }

    let builds = Arc::new(std::sync::Mutex::new(0usize));
    let mut providers = HashMap::new();
    providers.insert(
        "cached-transcription".to_string(),
        Arc::new(CountingTranscriptionFactory {
            builds: builds.clone(),
        }) as Arc<dyn ProviderFactory>,
    );
    let reg = create_provider_registry(providers, None);
    let handle = reg
        .transcription_model("cached-transcription:model")
        .unwrap();

    let first = siumai_core::transcription::TranscriptionModelV3::transcribe(
        &handle,
        crate::types::SttRequest::from_audio(Vec::new()),
    )
    .await
    .unwrap();
    let second = siumai_core::transcription::TranscriptionModelV3::transcribe(
        &handle,
        crate::types::SttRequest::from_audio(Vec::new()),
    )
    .await
    .unwrap();

    assert_eq!(first.text, "cached transcription");
    assert_eq!(second.text, "cached transcription");
    assert_eq!(*builds.lock().unwrap(), 1);
}
