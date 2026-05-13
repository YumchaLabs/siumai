use super::*;
use std::collections::HashMap;
use std::sync::Arc;

#[tokio::test]
async fn language_model_handle_builds_client() {
    let _g = reg_test_guard();
    let mut providers = HashMap::new();
    providers.insert(
        "testprov".to_string(),
        Arc::new(TestProviderFactory::new("testprov")) as Arc<dyn ProviderFactory>,
    );
    let reg = create_provider_registry(providers, None);
    let handle = reg.language_model("testprov:model").unwrap();

    TEST_BUILD_COUNT.store(0, std::sync::atomic::Ordering::SeqCst);
    let resp = handle.chat(vec![]).await.unwrap();
    assert_eq!(resp.content_text().unwrap_or_default(), "ok");
    assert_eq!(
        TEST_BUILD_COUNT.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "First call should build a new client"
    );

    let resp = handle.chat(vec![]).await.unwrap();
    assert_eq!(resp.content_text().unwrap_or_default(), "ok");
    assert_eq!(
        TEST_BUILD_COUNT.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "Second call should use cached client"
    );
}

#[test]
fn language_model_handle_rejects_provider_without_chat_capability() {
    let _g = reg_test_guard();
    let mut providers = HashMap::new();
    providers.insert(
        "testprov_embed".to_string(),
        Arc::new(TestProviderFactory::new("testprov_embed")) as Arc<dyn ProviderFactory>,
    );

    let reg = create_provider_registry(providers, None);
    let err = match reg.language_model("testprov_embed:model") {
        Ok(_) => panic!("language model handle should be rejected without chat capability"),
        Err(err) => err,
    };

    assert!(
        matches!(err, LlmError::UnsupportedOperation(message) if message.contains("language_model/chat handles"))
    );
}

#[test]
fn language_model_handle_implements_model_metadata() {
    let _g = reg_test_guard();
    let mut providers = HashMap::new();
    providers.insert(
        "testprov".to_string(),
        Arc::new(TestProviderFactory::new("testprov")) as Arc<dyn ProviderFactory>,
    );

    let reg = create_provider_registry(providers, None);
    let handle = reg.language_model("testprov:model").unwrap();

    fn assert_language_model<M>(model: &M)
    where
        M: siumai_core::text::LanguageModel + ?Sized,
    {
        assert_eq!(crate::traits::ModelMetadata::provider_id(model), "testprov");
        assert_eq!(crate::traits::ModelMetadata::model_id(model), "model");
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model),
            crate::traits::ModelSpecVersion::V1
        );
    }

    assert_language_model(&handle);
}

#[tokio::test]
async fn provider_factory_text_family_bridge_works() {
    let _g = reg_test_guard();
    TEST_BUILD_COUNT.store(0, std::sync::atomic::Ordering::SeqCst);

    let factory = TestProviderFactory::new("testprov");
    let model = factory.language_model_text("bridged-model").await.unwrap();

    assert_eq!(
        crate::traits::ModelMetadata::provider_id(model.as_ref()),
        "testprov"
    );
    assert_eq!(
        crate::traits::ModelMetadata::model_id(model.as_ref()),
        "bridged-model"
    );

    let response = model
        .generate(ChatRequest::new(vec![ChatMessage::user("hi").build()]))
        .await
        .unwrap();
    assert_eq!(response.content_text(), Some("ok"));
    assert_eq!(
        TEST_BUILD_COUNT.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "Bridge should reuse the existing language_model construction path"
    );
}

#[tokio::test]
async fn provider_factory_native_text_family_path_works() {
    let _g = reg_test_guard();

    #[derive(Clone)]
    struct NativeLanguageModel;

    impl crate::traits::ModelMetadata for NativeLanguageModel {
        fn provider_id(&self) -> &str {
            "native"
        }

        fn model_id(&self) -> &str {
            "native-model"
        }
    }

    #[async_trait::async_trait]
    impl siumai_core::text::TextModel for NativeLanguageModel {
        async fn generate(&self, _request: ChatRequest) -> Result<ChatResponse, LlmError> {
            Ok(ChatResponse::new(crate::types::MessageContent::Text(
                "native-ok".to_string(),
            )))
        }

        async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
            Err(LlmError::UnsupportedOperation(
                "native stream not implemented in test".to_string(),
            ))
        }

        async fn stream_with_cancel(
            &self,
            _request: ChatRequest,
        ) -> Result<ChatStreamHandle, LlmError> {
            Err(LlmError::UnsupportedOperation(
                "native stream_with_cancel not implemented in test".to_string(),
            ))
        }
    }

    struct NativeOnlyFactory;

    #[async_trait::async_trait]
    impl ProviderFactory for NativeOnlyFactory {
        async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
            panic!("legacy generic-client path should not be used by native text-family test")
        }

        async fn language_model_text_with_ctx(
            &self,
            _model_id: &str,
            _ctx: &BuildContext,
        ) -> Result<Arc<dyn siumai_core::text::LanguageModel>, LlmError> {
            Ok(Arc::new(NativeLanguageModel))
        }

        fn provider_id(&self) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("native")
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_chat()
        }
    }

    let factory = NativeOnlyFactory;
    let model = factory.language_model_text("native-model").await.unwrap();

    assert_eq!(
        crate::traits::ModelMetadata::provider_id(model.as_ref()),
        "native"
    );
    assert_eq!(
        crate::traits::ModelMetadata::model_id(model.as_ref()),
        "native-model"
    );

    let response = model.generate(ChatRequest::new(vec![])).await.unwrap();
    assert_eq!(response.content_text(), Some("native-ok"));
}

#[tokio::test]
async fn language_model_handle_preserves_chat_request_fields() {
    let _g = reg_test_guard();

    #[derive(Clone)]
    struct CapturingClient(std::sync::Arc<std::sync::Mutex<Option<ChatRequest>>>);

    #[async_trait::async_trait]
    impl ChatCapability for CapturingClient {
        async fn chat_with_tools(
            &self,
            _messages: Vec<crate::types::ChatMessage>,
            _tools: Option<Vec<crate::types::Tool>>,
        ) -> Result<crate::types::ChatResponse, LlmError> {
            Err(LlmError::UnsupportedOperation("use chat_request".into()))
        }

        async fn chat_stream(
            &self,
            _messages: Vec<crate::types::ChatMessage>,
            _tools: Option<Vec<crate::types::Tool>>,
        ) -> Result<crate::streaming::ChatStream, LlmError> {
            Err(LlmError::UnsupportedOperation(
                "use chat_stream_request".into(),
            ))
        }

        async fn chat_request(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
            *self.0.lock().unwrap() = Some(request);
            Ok(ChatResponse::new(crate::types::MessageContent::Text(
                "ok".to_string(),
            )))
        }

        async fn chat_stream_request(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
            *self.0.lock().unwrap() = Some(request);
            let end = ChatResponse::new(crate::types::MessageContent::Text("ok".to_string()));
            let events = vec![Ok(crate::types::ChatStreamEvent::StreamEnd {
                response: end,
            })];
            Ok(Box::pin(futures::stream::iter(events)))
        }
    }

    impl LlmClient for CapturingClient {
        fn provider_id(&self) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("cap")
        }

        fn supported_models(&self) -> Vec<String> {
            vec!["bound-model".into()]
        }

        fn capabilities(&self) -> crate::traits::ProviderCapabilities {
            crate::traits::ProviderCapabilities::new().with_chat()
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn clone_box(&self) -> Box<dyn LlmClient> {
            Box::new(self.clone())
        }

        fn as_chat_capability(&self) -> Option<&dyn ChatCapability> {
            Some(self)
        }
    }

    struct CaptureFactory {
        seen: std::sync::Arc<std::sync::Mutex<Option<ChatRequest>>>,
    }

    #[async_trait::async_trait]
    impl ProviderFactory for CaptureFactory {
        async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
            Ok(Arc::new(CapturingClient(self.seen.clone())))
        }

        fn provider_id(&self) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("cap")
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_chat().with_streaming()
        }
    }

    let seen = std::sync::Arc::new(std::sync::Mutex::new(None));
    let mut providers = HashMap::new();
    providers.insert(
        "cap".to_string(),
        Arc::new(CaptureFactory { seen: seen.clone() }) as Arc<dyn ProviderFactory>,
    );
    let reg = create_provider_registry(providers, None);
    let handle = reg.language_model("cap:bound-model").unwrap();

    let request = ChatRequest::builder()
        .message(ChatMessage::user("hi").build())
        .temperature(0.7)
        .max_tokens(123)
        .build()
        .with_provider_option(
            "openai",
            serde_json::json!({ "responsesApi": { "enabled": true } }),
        );

    let _ = handle.chat_request(request.clone()).await.unwrap();

    let captured = seen.lock().unwrap().clone().unwrap();
    assert_eq!(
        captured.common_params.temperature,
        request.common_params.temperature
    );
    assert_eq!(
        captured.common_params.max_tokens,
        request.common_params.max_tokens
    );
    assert_eq!(
        captured.provider_options_map.get("openai"),
        request.provider_options_map.get("openai")
    );
    assert_eq!(captured.common_params.model, "bound-model");
    assert!(!captured.stream);

    let _ = handle
        .chat_stream_request(ChatRequest::new(vec![ChatMessage::user("hi").build()]))
        .await
        .unwrap();
    let captured = seen.lock().unwrap().clone().unwrap();
    assert!(captured.stream);
    assert_eq!(captured.common_params.model, "bound-model");
}

#[tokio::test]
async fn language_model_handle_uses_native_family_path_when_available() {
    let _g = reg_test_guard();

    #[derive(Clone)]
    struct NativeLanguageModel;

    impl crate::traits::ModelMetadata for NativeLanguageModel {
        fn provider_id(&self) -> &str {
            "native-handle"
        }

        fn model_id(&self) -> &str {
            "native-handle-model"
        }
    }

    #[async_trait::async_trait]
    impl siumai_core::text::TextModel for NativeLanguageModel {
        async fn generate(&self, _request: ChatRequest) -> Result<ChatResponse, LlmError> {
            Ok(ChatResponse::new(crate::types::MessageContent::Text(
                "native-handle-ok".to_string(),
            )))
        }

        async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
            Err(LlmError::UnsupportedOperation(
                "stream not needed".to_string(),
            ))
        }

        async fn stream_with_cancel(
            &self,
            _request: ChatRequest,
        ) -> Result<ChatStreamHandle, LlmError> {
            Err(LlmError::UnsupportedOperation(
                "stream_with_cancel not needed".to_string(),
            ))
        }
    }

    struct NativeHandleFactory;

    #[async_trait::async_trait]
    impl ProviderFactory for NativeHandleFactory {
        async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
            panic!("legacy generic-client path should not be used by language handle")
        }

        async fn language_model_text_with_ctx(
            &self,
            _model_id: &str,
            _ctx: &BuildContext,
        ) -> Result<Arc<dyn siumai_core::text::LanguageModel>, LlmError> {
            Ok(Arc::new(NativeLanguageModel))
        }

        fn provider_id(&self) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("native-handle")
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_chat()
        }
    }

    let mut providers = HashMap::new();
    providers.insert(
        "native-handle".to_string(),
        Arc::new(NativeHandleFactory) as Arc<dyn ProviderFactory>,
    );
    let reg = create_provider_registry(providers, None);
    let handle = reg.language_model("native-handle:model").unwrap();

    let response = handle.chat_request(ChatRequest::new(vec![])).await.unwrap();
    assert_eq!(response.content_text(), Some("native-handle-ok"));
}
