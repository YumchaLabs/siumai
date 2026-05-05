use super::*;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
struct BridgeCompletionClient;

#[async_trait::async_trait]
impl crate::traits::CompletionCapability for BridgeCompletionClient {
    async fn complete(
        &self,
        request: crate::types::CompletionRequest,
    ) -> Result<crate::types::CompletionResponse, LlmError> {
        Ok(crate::types::CompletionResponse::new(
            request
                .prompt
                .first()
                .and_then(|message| message.content_text())
                .unwrap_or("bridge completion"),
        ))
    }

    async fn complete_stream(
        &self,
        _request: crate::types::CompletionRequest,
    ) -> Result<crate::streaming::ChatStream, LlmError> {
        let events = vec![Ok(crate::types::ChatStreamEvent::StreamEnd {
            response: crate::types::ChatResponse::empty_with_finish_reason(
                crate::types::FinishReason::Stop,
            ),
        })];
        Ok(Box::pin(futures::stream::iter(events)))
    }
}

impl LlmClient for BridgeCompletionClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("testprov_completion")
    }

    fn supported_models(&self) -> Vec<String> {
        vec!["completion-model".into()]
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new()
            .with_completion()
            .with_streaming()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }

    fn as_completion_capability(&self) -> Option<&dyn crate::traits::CompletionCapability> {
        Some(self)
    }
}

struct BridgeCompletionFactory;

#[async_trait::async_trait]
impl ProviderFactory for BridgeCompletionFactory {
    async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        Ok(Arc::new(BridgeCompletionClient))
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("testprov_completion")
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_completion()
            .with_streaming()
    }
}

#[test]
fn completion_model_handle_implements_model_metadata() {
    let mut providers = HashMap::new();
    providers.insert(
        "testprov_completion".to_string(),
        Arc::new(BridgeCompletionFactory) as Arc<dyn ProviderFactory>,
    );

    let reg = create_provider_registry(providers, None);
    let handle = reg.completion_model("testprov_completion:model").unwrap();

    fn assert_completion_model<M>(model: &M)
    where
        M: siumai_core::completion::CompletionModel + ?Sized,
    {
        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model),
            "testprov_completion"
        );
        assert_eq!(crate::traits::ModelMetadata::model_id(model), "model");
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model),
            crate::traits::ModelSpecVersion::V1
        );
    }

    assert_completion_model(&handle);
}

#[test]
fn completion_model_handle_rejects_provider_without_completion_capability() {
    let mut providers = HashMap::new();
    providers.insert(
        "testprov_chat".to_string(),
        Arc::new(TestProviderFactory::new("testprov_chat")) as Arc<dyn ProviderFactory>,
    );
    let reg = create_provider_registry(providers, None);

    let err = match reg.completion_model("testprov_chat:model") {
        Ok(_) => panic!("completion handle should be rejected without completion capability"),
        Err(err) => err,
    };

    assert!(
        matches!(err, LlmError::UnsupportedOperation(message) if message.contains("completion_model handle"))
    );
}

#[tokio::test]
async fn provider_factory_completion_family_bridge_works() {
    let factory = BridgeCompletionFactory;
    let model = factory
        .completion_model_family("bridged-completion-model")
        .await
        .unwrap();

    assert_eq!(
        crate::traits::ModelMetadata::provider_id(model.as_ref()),
        "testprov_completion"
    );
    assert_eq!(
        crate::traits::ModelMetadata::model_id(model.as_ref()),
        "bridged-completion-model"
    );

    let response = model
        .complete(crate::types::CompletionRequest::new("bridge completion"))
        .await
        .unwrap();
    assert_eq!(response.text(), "bridge completion");
}

#[tokio::test]
async fn provider_factory_native_completion_family_path_works() {
    #[derive(Clone)]
    struct NativeCompletionModel;

    impl crate::traits::ModelMetadata for NativeCompletionModel {
        fn provider_id(&self) -> &str {
            "native-completion"
        }

        fn model_id(&self) -> &str {
            "native-completion-model"
        }
    }

    #[async_trait::async_trait]
    impl siumai_core::completion::CompletionModel for NativeCompletionModel {
        async fn complete(
            &self,
            request: crate::types::CompletionRequest,
        ) -> Result<crate::types::CompletionResponse, LlmError> {
            Ok(crate::types::CompletionResponse::new(
                request.common_params.model.clone(),
            ))
        }

        async fn stream(
            &self,
            _request: crate::types::CompletionRequest,
        ) -> Result<crate::streaming::ChatStream, LlmError> {
            panic!("stream path not needed for this test")
        }

        async fn stream_with_cancel(
            &self,
            _request: crate::types::CompletionRequest,
        ) -> Result<crate::streaming::ChatStreamHandle, LlmError> {
            panic!("stream_with_cancel path not needed for this test")
        }
    }

    struct NativeOnlyCompletionFactory;

    #[async_trait::async_trait]
    impl ProviderFactory for NativeOnlyCompletionFactory {
        async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
            panic!("legacy generic-client path should not be used by native completion-family test")
        }

        async fn completion_model_family_with_ctx(
            &self,
            _model_id: &str,
            _ctx: &BuildContext,
        ) -> Result<Arc<dyn siumai_core::completion::CompletionModel>, LlmError> {
            Ok(Arc::new(NativeCompletionModel))
        }

        fn provider_id(&self) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("native-completion")
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_completion()
        }
    }

    let factory = NativeOnlyCompletionFactory;
    let model = factory
        .completion_model_family("native-completion-model")
        .await
        .unwrap();

    let response = model
        .complete(
            crate::types::CompletionRequest::new("ignored").with_model("native-completion-model"),
        )
        .await
        .unwrap();
    assert_eq!(response.text(), "native-completion-model");
}

#[tokio::test]
async fn completion_model_handle_reuses_cached_family_model() {
    #[derive(Clone)]
    struct CountingCompletionModel;

    impl crate::traits::ModelMetadata for CountingCompletionModel {
        fn provider_id(&self) -> &str {
            "cached-completion"
        }

        fn model_id(&self) -> &str {
            "cached-completion-model"
        }
    }

    #[async_trait::async_trait]
    impl siumai_core::completion::CompletionModel for CountingCompletionModel {
        async fn complete(
            &self,
            _request: crate::types::CompletionRequest,
        ) -> Result<crate::types::CompletionResponse, LlmError> {
            Ok(crate::types::CompletionResponse::new("cached completion"))
        }

        async fn stream(
            &self,
            _request: crate::types::CompletionRequest,
        ) -> Result<crate::streaming::ChatStream, LlmError> {
            panic!("stream path not needed for cache test")
        }

        async fn stream_with_cancel(
            &self,
            _request: crate::types::CompletionRequest,
        ) -> Result<crate::streaming::ChatStreamHandle, LlmError> {
            panic!("stream_with_cancel path not needed for cache test")
        }
    }

    struct CountingCompletionFactory {
        builds: Arc<std::sync::Mutex<usize>>,
    }

    #[async_trait::async_trait]
    impl ProviderFactory for CountingCompletionFactory {
        async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
            panic!("legacy generic-client path should not be used by completion cache test")
        }

        async fn completion_model_family_with_ctx(
            &self,
            _model_id: &str,
            _ctx: &BuildContext,
        ) -> Result<Arc<dyn siumai_core::completion::CompletionModel>, LlmError> {
            *self.builds.lock().unwrap() += 1;
            Ok(Arc::new(CountingCompletionModel))
        }

        fn provider_id(&self) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("cached-completion")
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_completion()
        }
    }

    let builds = Arc::new(std::sync::Mutex::new(0usize));
    let mut providers = HashMap::new();
    providers.insert(
        "cached-completion".to_string(),
        Arc::new(CountingCompletionFactory {
            builds: builds.clone(),
        }) as Arc<dyn ProviderFactory>,
    );
    let reg = create_provider_registry(providers, None);
    let handle = reg.completion_model("cached-completion:model").unwrap();

    let first = siumai_core::completion::CompletionModel::complete(
        &handle,
        crate::types::CompletionRequest::new("one"),
    )
    .await
    .unwrap();
    let second = siumai_core::completion::CompletionModel::complete(
        &handle,
        crate::types::CompletionRequest::new("two"),
    )
    .await
    .unwrap();

    assert_eq!(first.text(), "cached completion");
    assert_eq!(second.text(), "cached completion");
    assert_eq!(*builds.lock().unwrap(), 1);
}
