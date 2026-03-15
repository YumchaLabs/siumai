use super::*;
use crate::types::{BatchEmbeddingRequest, BatchEmbeddingResponse, EmbeddingRequest};

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
