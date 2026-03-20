use super::*;
use crate::types::{BatchEmbeddingRequest, BatchEmbeddingResponse, EmbeddingRequest};
use std::sync::{Arc, Mutex};

#[tokio::test]
async fn embedding_model_handle_builds_client() {
    let mut providers = HashMap::new();
    providers.insert(
        "testprov_embed".to_string(),
        Arc::new(TestProviderFactory::new("testprov_embed")) as Arc<dyn ProviderFactory>,
    );
    let reg = create_provider_registry(providers, None);
    let handle = reg.embedding_model("testprov_embed:model").unwrap();

    let out = handle.embed(vec!["a".into(), "b".into()]).await.unwrap();
    assert_eq!(out.embeddings[0][0], 2.0);
}

#[test]
fn embedding_model_handle_rejects_provider_without_embedding_capability() {
    let mut providers = HashMap::new();
    providers.insert(
        "testprov_chat".to_string(),
        Arc::new(TestProviderFactory::new("testprov_chat")) as Arc<dyn ProviderFactory>,
    );
    let reg = create_provider_registry(providers, None);

    let err = match reg.embedding_model("testprov_chat:model") {
        Ok(_) => panic!("embedding handle should be rejected without embedding capability"),
        Err(err) => err,
    };

    assert!(
        matches!(err, LlmError::UnsupportedOperation(message) if message.contains("embedding_model handle"))
    );
}

#[test]
fn embedding_model_handle_implements_model_metadata() {
    let mut providers = HashMap::new();
    providers.insert(
        "testprov_embed".to_string(),
        Arc::new(TestProviderFactory::new("testprov_embed")) as Arc<dyn ProviderFactory>,
    );

    let reg = create_provider_registry(providers, None);
    let handle = reg.embedding_model("testprov_embed:model").unwrap();

    fn assert_embedding_model<M>(model: &M)
    where
        M: siumai_core::embedding::EmbeddingModel + ?Sized,
    {
        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model),
            "testprov_embed"
        );
        assert_eq!(crate::traits::ModelMetadata::model_id(model), "model");
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model),
            crate::traits::ModelSpecVersion::V1
        );
    }

    assert_embedding_model(&handle);
}

#[tokio::test]
async fn provider_factory_embedding_family_bridge_works() {
    let factory = TestProviderFactory::new("testprov_embed");
    let model = factory
        .embedding_model_family("bridged-embed-model")
        .await
        .unwrap();

    assert_eq!(
        crate::traits::ModelMetadata::provider_id(model.as_ref()),
        "testprov_embed"
    );
    assert_eq!(
        crate::traits::ModelMetadata::model_id(model.as_ref()),
        "bridged-embed-model"
    );

    let response = model
        .embed(EmbeddingRequest {
            input: vec!["hello".to_string(), "world".to_string()],
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(response.embeddings[0][0], 2.0);
}

#[tokio::test]
async fn provider_factory_native_embedding_family_path_works() {
    #[derive(Clone)]
    struct NativeEmbeddingModel;

    impl crate::traits::ModelMetadata for NativeEmbeddingModel {
        fn provider_id(&self) -> &str {
            "native-embed"
        }

        fn model_id(&self) -> &str {
            "native-embed-model"
        }
    }

    #[async_trait::async_trait]
    impl crate::embedding::EmbeddingModelV3 for NativeEmbeddingModel {
        async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
            Ok(EmbeddingResponse::new(
                vec![vec![request.input.len() as f32]],
                "native-embed-model".to_string(),
            ))
        }

        async fn embed_many(
            &self,
            requests: BatchEmbeddingRequest,
        ) -> Result<BatchEmbeddingResponse, LlmError> {
            Ok(BatchEmbeddingResponse {
                responses: requests
                    .requests
                    .into_iter()
                    .map(|request| {
                        Ok(EmbeddingResponse::new(
                            vec![vec![request.input.len() as f32]],
                            "native-embed-model".to_string(),
                        ))
                    })
                    .collect(),
                metadata: std::collections::HashMap::new(),
            })
        }
    }

    struct NativeOnlyEmbeddingFactory;

    #[async_trait::async_trait]
    impl ProviderFactory for NativeOnlyEmbeddingFactory {
        async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
            panic!("legacy generic-client path should not be used by native embedding-family test")
        }

        async fn embedding_model_family_with_ctx(
            &self,
            _model_id: &str,
            _ctx: &BuildContext,
        ) -> Result<Arc<dyn siumai_core::embedding::EmbeddingModel>, LlmError> {
            Ok(Arc::new(NativeEmbeddingModel))
        }

        fn provider_id(&self) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("native-embed")
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_embedding()
        }
    }

    let factory = NativeOnlyEmbeddingFactory;
    let model = factory
        .embedding_model_family("native-embed-model")
        .await
        .unwrap();

    assert_eq!(
        crate::traits::ModelMetadata::provider_id(model.as_ref()),
        "native-embed"
    );
    assert_eq!(
        crate::traits::ModelMetadata::model_id(model.as_ref()),
        "native-embed-model"
    );

    let response = model
        .embed(EmbeddingRequest {
            input: vec!["hello".to_string()],
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(response.embeddings[0][0], 1.0);
}

#[tokio::test]
async fn embedding_model_handle_uses_native_family_path_when_available() {
    #[derive(Clone)]
    struct NativeEmbeddingModel;

    impl crate::traits::ModelMetadata for NativeEmbeddingModel {
        fn provider_id(&self) -> &str {
            "native-embed-handle"
        }

        fn model_id(&self) -> &str {
            "native-embed-handle-model"
        }
    }

    #[async_trait::async_trait]
    impl siumai_core::embedding::EmbeddingModelV3 for NativeEmbeddingModel {
        async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
            Ok(EmbeddingResponse::new(
                vec![vec![request.input.len() as f32]],
                "native-embed-handle-model".to_string(),
            ))
        }

        async fn embed_many(
            &self,
            requests: BatchEmbeddingRequest,
        ) -> Result<BatchEmbeddingResponse, LlmError> {
            Ok(BatchEmbeddingResponse {
                responses: requests
                    .requests
                    .into_iter()
                    .map(|request| {
                        Ok(EmbeddingResponse::new(
                            vec![vec![request.input.len() as f32]],
                            "native-embed-handle-model".to_string(),
                        ))
                    })
                    .collect(),
                metadata: std::collections::HashMap::new(),
            })
        }
    }

    struct NativeEmbeddingHandleFactory;

    #[async_trait::async_trait]
    impl ProviderFactory for NativeEmbeddingHandleFactory {
        async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
            panic!("legacy generic-client path should not be used by embedding handle")
        }

        async fn embedding_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
            panic!("legacy embedding client path should not be used by embedding handle")
        }

        async fn embedding_model_family_with_ctx(
            &self,
            _model_id: &str,
            _ctx: &BuildContext,
        ) -> Result<Arc<dyn siumai_core::embedding::EmbeddingModel>, LlmError> {
            Ok(Arc::new(NativeEmbeddingModel))
        }

        fn provider_id(&self) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("native-embed-handle")
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_embedding()
        }
    }

    let mut providers = HashMap::new();
    providers.insert(
        "native-embed-handle".to_string(),
        Arc::new(NativeEmbeddingHandleFactory) as Arc<dyn ProviderFactory>,
    );
    let reg = create_provider_registry(providers, None);
    let handle = reg.embedding_model("native-embed-handle:model").unwrap();

    let response = handle
        .embed(vec!["a".to_string(), "b".to_string()])
        .await
        .unwrap();
    assert_eq!(response.embeddings[0][0], 2.0);
}

#[tokio::test]
async fn embedding_model_handle_family_trait_preserves_request_config_on_bridge_path() {
    #[derive(Clone)]
    struct RequestAwareEmbeddingClient {
        seen: Arc<Mutex<Option<EmbeddingRequest>>>,
    }

    #[async_trait::async_trait]
    impl EmbeddingCapability for RequestAwareEmbeddingClient {
        async fn embed(&self, input: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
            Ok(EmbeddingResponse::new(
                vec![vec![input.len() as f32]],
                "fallback".to_string(),
            ))
        }

        fn as_embedding_extensions(&self) -> Option<&dyn EmbeddingExtensions> {
            Some(self)
        }

        fn embedding_dimension(&self) -> usize {
            3
        }
    }

    #[async_trait::async_trait]
    impl EmbeddingExtensions for RequestAwareEmbeddingClient {
        async fn embed_with_config(
            &self,
            request: EmbeddingRequest,
        ) -> Result<EmbeddingResponse, LlmError> {
            *self.seen.lock().expect("request lock") = Some(request.clone());

            Ok(EmbeddingResponse::new(
                vec![vec![request.dimensions.unwrap_or_default() as f32]],
                request
                    .model
                    .clone()
                    .unwrap_or_else(|| "request-aware".to_string()),
            ))
        }
    }

    impl LlmClient for RequestAwareEmbeddingClient {
        fn provider_id(&self) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("request-aware-embed")
        }

        fn supported_models(&self) -> Vec<String> {
            vec!["request-aware-model".to_string()]
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_embedding()
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn clone_box(&self) -> Box<dyn LlmClient> {
            Box::new(self.clone())
        }

        fn as_embedding_capability(&self) -> Option<&dyn EmbeddingCapability> {
            Some(self)
        }

        fn as_embedding_extensions(&self) -> Option<&dyn EmbeddingExtensions> {
            Some(self)
        }
    }

    struct RequestAwareEmbeddingFactory {
        seen: Arc<Mutex<Option<EmbeddingRequest>>>,
    }

    #[async_trait::async_trait]
    impl ProviderFactory for RequestAwareEmbeddingFactory {
        async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
            Ok(Arc::new(RequestAwareEmbeddingClient {
                seen: Arc::clone(&self.seen),
            }))
        }

        fn provider_id(&self) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("request-aware-embed")
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_embedding()
        }
    }

    let seen = Arc::new(Mutex::new(None));
    let mut providers = HashMap::new();
    providers.insert(
        "request-aware-embed".to_string(),
        Arc::new(RequestAwareEmbeddingFactory {
            seen: Arc::clone(&seen),
        }) as Arc<dyn ProviderFactory>,
    );

    let registry = create_provider_registry(providers, None);
    let handle = registry
        .embedding_model("request-aware-embed:model")
        .expect("embedding handle");

    let response = siumai_core::embedding::EmbeddingModelV3::embed(
        &handle,
        EmbeddingRequest::single("hello")
            .with_model("request-model")
            .with_dimensions(64)
            .with_user("user-42")
            .with_header("x-embed-test", "yes"),
    )
    .await
    .expect("embedding response");

    assert_eq!(response.model, "request-model");
    assert_eq!(response.embeddings[0][0], 64.0);

    let seen = seen.lock().expect("request lock");
    let request = seen.as_ref().expect("captured request");
    assert_eq!(request.model.as_deref(), Some("request-model"));
    assert_eq!(request.dimensions, Some(64));
    assert_eq!(request.user.as_deref(), Some("user-42"));
    assert_eq!(
        request
            .http_config
            .as_ref()
            .and_then(|config| config.headers.get("x-embed-test"))
            .map(String::as_str),
        Some("yes")
    );
}

#[tokio::test]
async fn embedding_model_handle_prefers_request_aware_client_extensions_over_lossy_native_family() {
    #[derive(Clone)]
    struct LossyNativeFamilyEmbeddingModel;

    impl crate::traits::ModelMetadata for LossyNativeFamilyEmbeddingModel {
        fn provider_id(&self) -> &str {
            "request-aware-native"
        }

        fn model_id(&self) -> &str {
            "request-aware-native-model"
        }
    }

    #[async_trait::async_trait]
    impl siumai_core::embedding::EmbeddingModelV3 for LossyNativeFamilyEmbeddingModel {
        async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
            Ok(EmbeddingResponse::new(
                vec![vec![request.input.len() as f32]],
                request.model.unwrap_or_else(|| "lossy-native".to_string()),
            ))
        }

        async fn embed_many(
            &self,
            requests: BatchEmbeddingRequest,
        ) -> Result<BatchEmbeddingResponse, LlmError> {
            Ok(BatchEmbeddingResponse {
                responses: requests
                    .requests
                    .into_iter()
                    .map(|request| {
                        Ok(EmbeddingResponse::new(
                            vec![vec![request.input.len() as f32]],
                            request.model.unwrap_or_else(|| "lossy-native".to_string()),
                        ))
                    })
                    .collect(),
                metadata: HashMap::new(),
            })
        }
    }

    #[derive(Clone)]
    struct RequestAwareNativeClient {
        seen: Arc<Mutex<Option<EmbeddingRequest>>>,
    }

    #[async_trait::async_trait]
    impl EmbeddingCapability for RequestAwareNativeClient {
        async fn embed(&self, input: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
            Ok(EmbeddingResponse::new(
                vec![vec![input.len() as f32]],
                "fallback".to_string(),
            ))
        }

        fn as_embedding_extensions(&self) -> Option<&dyn EmbeddingExtensions> {
            Some(self)
        }

        fn embedding_dimension(&self) -> usize {
            3
        }
    }

    #[async_trait::async_trait]
    impl EmbeddingExtensions for RequestAwareNativeClient {
        async fn embed_with_config(
            &self,
            request: EmbeddingRequest,
        ) -> Result<EmbeddingResponse, LlmError> {
            *self.seen.lock().expect("request lock") = Some(request.clone());

            Ok(EmbeddingResponse::new(
                vec![vec![request.dimensions.unwrap_or_default() as f32]],
                request
                    .model
                    .clone()
                    .unwrap_or_else(|| "request-aware-native".to_string()),
            ))
        }
    }

    impl LlmClient for RequestAwareNativeClient {
        fn provider_id(&self) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("request-aware-native")
        }

        fn supported_models(&self) -> Vec<String> {
            vec!["request-aware-native-model".to_string()]
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_embedding()
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn clone_box(&self) -> Box<dyn LlmClient> {
            Box::new(self.clone())
        }

        fn as_embedding_capability(&self) -> Option<&dyn EmbeddingCapability> {
            Some(self)
        }

        fn as_embedding_extensions(&self) -> Option<&dyn EmbeddingExtensions> {
            Some(self)
        }
    }

    #[derive(Clone)]
    struct RequestAwareNativeFactory {
        seen: Arc<Mutex<Option<EmbeddingRequest>>>,
    }

    #[async_trait::async_trait]
    impl ProviderFactory for RequestAwareNativeFactory {
        async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
            Err(LlmError::UnsupportedOperation(
                "language model not used in embedding test".to_string(),
            ))
        }

        async fn embedding_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
            Ok(Arc::new(RequestAwareNativeClient {
                seen: Arc::clone(&self.seen),
            }))
        }

        async fn embedding_model_family_with_ctx(
            &self,
            _model_id: &str,
            _ctx: &BuildContext,
        ) -> Result<Arc<dyn siumai_core::embedding::EmbeddingModel>, LlmError> {
            Ok(Arc::new(LossyNativeFamilyEmbeddingModel))
        }

        fn provider_id(&self) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("request-aware-native")
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_embedding()
        }
    }

    let seen = Arc::new(Mutex::new(None));
    let mut providers = HashMap::new();
    providers.insert(
        "request-aware-native".to_string(),
        Arc::new(RequestAwareNativeFactory {
            seen: Arc::clone(&seen),
        }) as Arc<dyn ProviderFactory>,
    );

    let registry = create_provider_registry(providers, None);
    let handle = registry
        .embedding_model("request-aware-native:model")
        .expect("embedding handle");

    let response = handle
        .embed_with_config(
            EmbeddingRequest::single("hello")
                .with_model("request-model")
                .with_dimensions(256)
                .with_user("user-42"),
        )
        .await
        .expect("embedding response");

    assert_eq!(response.model, "request-model");
    assert_eq!(response.embeddings[0][0], 256.0);

    let seen = seen.lock().expect("request lock");
    let request = seen.as_ref().expect("captured request");
    assert_eq!(request.model.as_deref(), Some("request-model"));
    assert_eq!(request.dimensions, Some(256));
    assert_eq!(request.user.as_deref(), Some("user-42"));
}

#[tokio::test]
async fn embedding_model_handle_embed_many_uses_native_family_batch_path_when_available() {
    #[derive(Clone)]
    struct NativeBatchOnlyEmbeddingModel;

    impl crate::traits::ModelMetadata for NativeBatchOnlyEmbeddingModel {
        fn provider_id(&self) -> &str {
            "native-batch-embed"
        }

        fn model_id(&self) -> &str {
            "native-batch-embed-model"
        }
    }

    #[async_trait::async_trait]
    impl siumai_core::embedding::EmbeddingModelV3 for NativeBatchOnlyEmbeddingModel {
        async fn embed(&self, _request: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
            panic!("native batch test should not fall back to single-request embedding path")
        }

        async fn embed_many(
            &self,
            requests: BatchEmbeddingRequest,
        ) -> Result<BatchEmbeddingResponse, LlmError> {
            Ok(BatchEmbeddingResponse {
                responses: requests
                    .requests
                    .into_iter()
                    .map(|request| {
                        Ok(EmbeddingResponse::new(
                            vec![vec![request.dimensions.unwrap_or_default() as f32]],
                            request
                                .model
                                .unwrap_or_else(|| "native-batch-embed-model".to_string()),
                        ))
                    })
                    .collect(),
                metadata: HashMap::new(),
            })
        }
    }

    struct NativeBatchEmbeddingFactory;

    #[async_trait::async_trait]
    impl ProviderFactory for NativeBatchEmbeddingFactory {
        async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
            panic!("legacy generic-client path should not be used by native batch embedding test")
        }

        async fn embedding_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
            panic!("legacy embedding client path should not be used by native batch embedding test")
        }

        async fn embedding_model_family_with_ctx(
            &self,
            _model_id: &str,
            _ctx: &BuildContext,
        ) -> Result<Arc<dyn siumai_core::embedding::EmbeddingModel>, LlmError> {
            Ok(Arc::new(NativeBatchOnlyEmbeddingModel))
        }

        fn provider_id(&self) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("native-batch-embed")
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_embedding()
        }
    }

    let mut providers = HashMap::new();
    providers.insert(
        "native-batch-embed".to_string(),
        Arc::new(NativeBatchEmbeddingFactory) as Arc<dyn ProviderFactory>,
    );

    let registry = create_provider_registry(providers, None);
    let handle = registry
        .embedding_model("native-batch-embed:model")
        .expect("embedding handle");

    let response = siumai_core::embedding::EmbeddingModelV3::embed_many(
        &handle,
        BatchEmbeddingRequest {
            requests: vec![
                EmbeddingRequest::single("hello")
                    .with_model("batch-model-a")
                    .with_dimensions(8),
                EmbeddingRequest::single("world")
                    .with_model("batch-model-b")
                    .with_dimensions(16),
            ],
            batch_options: crate::types::BatchOptions::default(),
        },
    )
    .await
    .expect("batch embedding response");

    assert_eq!(response.responses.len(), 2);

    let first = response.responses[0].as_ref().expect("first response");
    assert_eq!(first.model, "batch-model-a");
    assert_eq!(first.embeddings[0][0], 8.0);

    let second = response.responses[1].as_ref().expect("second response");
    assert_eq!(second.model, "batch-model-b");
    assert_eq!(second.embeddings[0][0], 16.0);
}
