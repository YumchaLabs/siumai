use super::*;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
struct BridgeRerankClient;

#[async_trait::async_trait]
impl crate::traits::RerankCapability for BridgeRerankClient {
    async fn rerank(
        &self,
        request: crate::types::RerankRequest,
    ) -> Result<crate::types::RerankResponse, LlmError> {
        let doc_len = request.documents_len();
        Ok(crate::types::RerankResponse {
            id: "bridge-rerank".to_string(),
            results: (0..doc_len)
                .map(|index| crate::types::RerankResult {
                    document: None,
                    index: index as u32,
                    relevance_score: (doc_len - index) as f64,
                })
                .collect(),
            tokens: crate::types::RerankTokenUsage {
                input_tokens: 1,
                output_tokens: 1,
            },
        })
    }
}

impl LlmClient for BridgeRerankClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("testprov_rerank")
    }

    fn supported_models(&self) -> Vec<String> {
        vec!["rerank-model".into()]
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new().with_rerank()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }

    fn as_rerank_capability(&self) -> Option<&dyn crate::traits::RerankCapability> {
        Some(self)
    }
}

struct BridgeRerankFactory;

#[async_trait::async_trait]
impl ProviderFactory for BridgeRerankFactory {
    async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        Ok(Arc::new(BridgeRerankClient))
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("testprov_rerank")
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_rerank()
    }
}

#[test]
fn reranking_model_handle_implements_model_metadata() {
    let mut providers = HashMap::new();
    providers.insert(
        "testprov_rerank".to_string(),
        Arc::new(BridgeRerankFactory) as Arc<dyn ProviderFactory>,
    );

    let reg = create_provider_registry(providers, None);
    let handle = reg.reranking_model("testprov_rerank:model").unwrap();

    fn assert_reranking_model<M>(model: &M)
    where
        M: siumai_core::rerank::RerankingModel + ?Sized,
    {
        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model),
            "testprov_rerank"
        );
        assert_eq!(crate::traits::ModelMetadata::model_id(model), "model");
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model),
            crate::traits::ModelSpecVersion::V1
        );
    }

    assert_reranking_model(&handle);
}

#[tokio::test]
async fn provider_factory_reranking_family_bridge_works() {
    let factory = BridgeRerankFactory;
    let model = factory
        .reranking_model_family("bridged-rerank-model")
        .await
        .unwrap();

    assert_eq!(
        crate::traits::ModelMetadata::provider_id(model.as_ref()),
        "testprov_rerank"
    );
    assert_eq!(
        crate::traits::ModelMetadata::model_id(model.as_ref()),
        "bridged-rerank-model"
    );

    let response = model
        .rerank(crate::types::RerankRequest::new(
            "bridged-rerank-model".to_string(),
            "query".to_string(),
            vec!["a".to_string(), "b".to_string()],
        ))
        .await
        .unwrap();
    assert_eq!(response.id, "bridge-rerank");
    assert_eq!(response.results.len(), 2);
}

#[tokio::test]
async fn provider_factory_native_reranking_family_path_works() {
    #[derive(Clone)]
    struct NativeRerankingModel;

    impl crate::traits::ModelMetadata for NativeRerankingModel {
        fn provider_id(&self) -> &str {
            "native-rerank"
        }

        fn model_id(&self) -> &str {
            "native-rerank-model"
        }
    }

    #[async_trait::async_trait]
    impl siumai_core::rerank::RerankModelV3 for NativeRerankingModel {
        async fn rerank(
            &self,
            request: crate::types::RerankRequest,
        ) -> Result<crate::types::RerankResponse, LlmError> {
            let doc_len = request.documents_len();
            Ok(crate::types::RerankResponse {
                id: "native-rerank".to_string(),
                results: (0..doc_len)
                    .map(|index| crate::types::RerankResult {
                        document: None,
                        index: index as u32,
                        relevance_score: (doc_len - index) as f64,
                    })
                    .collect(),
                tokens: crate::types::RerankTokenUsage {
                    input_tokens: 1,
                    output_tokens: 1,
                },
            })
        }
    }

    struct NativeOnlyRerankFactory;

    #[async_trait::async_trait]
    impl ProviderFactory for NativeOnlyRerankFactory {
        async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
            panic!("legacy generic-client path should not be used by native rerank-family test")
        }

        async fn reranking_model_family_with_ctx(
            &self,
            _model_id: &str,
            _ctx: &BuildContext,
        ) -> Result<Arc<dyn siumai_core::rerank::RerankingModel>, LlmError> {
            Ok(Arc::new(NativeRerankingModel))
        }

        fn provider_id(&self) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("native-rerank")
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_rerank()
        }
    }

    let factory = NativeOnlyRerankFactory;
    let model = factory
        .reranking_model_family("native-rerank-model")
        .await
        .unwrap();

    assert_eq!(
        crate::traits::ModelMetadata::provider_id(model.as_ref()),
        "native-rerank"
    );
    assert_eq!(
        crate::traits::ModelMetadata::model_id(model.as_ref()),
        "native-rerank-model"
    );

    let response = model
        .rerank(crate::types::RerankRequest::new(
            "native-rerank-model".to_string(),
            "query".to_string(),
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
        ))
        .await
        .unwrap();
    assert_eq!(response.id, "native-rerank");
    assert_eq!(response.results.len(), 3);
}

#[tokio::test]
async fn reranking_model_handle_uses_native_family_path_when_available() {
    #[derive(Clone)]
    struct NativeHandleRerankingModel;

    impl crate::traits::ModelMetadata for NativeHandleRerankingModel {
        fn provider_id(&self) -> &str {
            "native-rerank-handle"
        }

        fn model_id(&self) -> &str {
            "native-rerank-handle-model"
        }
    }

    #[async_trait::async_trait]
    impl siumai_core::rerank::RerankModelV3 for NativeHandleRerankingModel {
        async fn rerank(
            &self,
            request: crate::types::RerankRequest,
        ) -> Result<crate::types::RerankResponse, LlmError> {
            let doc_len = request.documents_len();
            Ok(crate::types::RerankResponse {
                id: "native-handle-rerank".to_string(),
                results: (0..doc_len)
                    .map(|index| crate::types::RerankResult {
                        document: None,
                        index: index as u32,
                        relevance_score: (doc_len - index) as f64,
                    })
                    .collect(),
                tokens: crate::types::RerankTokenUsage {
                    input_tokens: 1,
                    output_tokens: 1,
                },
            })
        }
    }

    struct NativeHandleRerankFactory;

    #[async_trait::async_trait]
    impl ProviderFactory for NativeHandleRerankFactory {
        async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
            panic!("legacy generic-client path should not be used by rerank handle")
        }

        async fn reranking_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
            panic!("legacy reranking client path should not be used by rerank handle")
        }

        async fn reranking_model_family_with_ctx(
            &self,
            _model_id: &str,
            _ctx: &BuildContext,
        ) -> Result<Arc<dyn siumai_core::rerank::RerankingModel>, LlmError> {
            Ok(Arc::new(NativeHandleRerankingModel))
        }

        fn provider_id(&self) -> std::borrow::Cow<'static, str> {
            std::borrow::Cow::Borrowed("native-rerank-handle")
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_rerank()
        }
    }

    let mut providers = HashMap::new();
    providers.insert(
        "native-rerank-handle".to_string(),
        Arc::new(NativeHandleRerankFactory) as Arc<dyn ProviderFactory>,
    );
    let reg = create_provider_registry(providers, None);
    let handle = reg.reranking_model("native-rerank-handle:model").unwrap();

    let response = handle
        .rerank(crate::types::RerankRequest::new(
            "model".to_string(),
            "query".to_string(),
            vec!["a".to_string(), "b".to_string()],
        ))
        .await
        .unwrap();
    assert_eq!(response.id, "native-handle-rerank");
    assert_eq!(response.results.len(), 2);
}
