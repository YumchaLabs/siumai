use super::OpenAiClient;
use crate::error::LlmError;
use crate::traits::RerankCapability;
use crate::types::{RerankRequest, RerankResponse};
use async_trait::async_trait;
use secrecy::ExposeSecret;

#[async_trait]
impl RerankCapability for OpenAiClient {
    async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, LlmError> {
        use crate::execution::executors::rerank::{HttpRerankExecutor, RerankExecutor};
        use crate::execution::http::headers::ProviderHeaders;
        use crate::std_openai::openai::rerank::OpenAiRerankStandard;

        let standard = OpenAiRerankStandard::new();
        #[cfg(feature = "std-openai-external")]
        let transformers = {
            struct ReqBridge(
                std::sync::Arc<dyn siumai_core::execution::rerank::RerankRequestTransformer>,
            );
            impl crate::execution::transformers::rerank_request::RerankRequestTransformer for ReqBridge {
                fn transform(
                    &self,
                    req: &crate::types::RerankRequest,
                ) -> Result<serde_json::Value, LlmError> {
                    let input = siumai_core::execution::rerank::RerankInput {
                        model: Some(req.model.clone()),
                        query: req.query.clone(),
                        documents: req.documents.clone(),
                        top_n: req.top_n,
                        return_documents: req.return_documents,
                        extra: Default::default(),
                    };
                    self.0.transform(&input)
                }
            }
            struct RespBridge(
                std::sync::Arc<dyn siumai_core::execution::rerank::RerankResponseTransformer>,
            );
            impl crate::execution::transformers::rerank_response::RerankResponseTransformer for RespBridge {
                fn transform(
                    &self,
                    raw: serde_json::Value,
                ) -> Result<crate::types::RerankResponse, LlmError> {
                    let out = self.0.transform_response(&raw)?;
                    let results = out
                        .results
                        .into_iter()
                        .map(|i| crate::types::RerankResult {
                            document: i.document.map(|text| crate::types::RerankDocument { text }),
                            index: i.index,
                            relevance_score: i.relevance_score,
                        })
                        .collect::<Vec<_>>();
                    Ok(crate::types::RerankResponse {
                        id: out.id.unwrap_or_default(),
                        results,
                        tokens: crate::types::RerankTokenUsage {
                            input_tokens: out.input_tokens,
                            output_tokens: out.output_tokens,
                        },
                    })
                }
            }
            crate::core::RerankTransformers {
                request: std::sync::Arc::new(ReqBridge(
                    standard.create_request_transformer("openai"),
                )),
                response: std::sync::Arc::new(RespBridge(
                    standard.create_response_transformer("openai"),
                )),
            }
        };
        #[cfg(not(feature = "std-openai-external"))]
        let transformers = standard.create_transformers("openai");

        let url = format!("{}/rerank", self.base_url.trim_end_matches('/'));

        let headers = ProviderHeaders::openai(
            self.api_key.expose_secret(),
            self.organization.as_deref(),
            self.project.as_deref(),
            &self.http_config.headers,
        )?;

        let exec = HttpRerankExecutor {
            provider_id: "openai".to_string(),
            http_client: self.http_client.clone(),
            request_transformer: transformers.request,
            response_transformer: transformers.response,
            policy: crate::execution::ExecutionPolicy::new()
                .with_interceptors(self.http_interceptors.clone())
                .with_retry_options(self.retry_options.clone()),
            url,
            headers,
            before_send: None,
        };
        RerankExecutor::execute(&exec, request).await
    }

    fn max_documents(&self) -> Option<u32> {
        self.rerank_capability.max_documents()
    }

    fn supported_models(&self) -> Vec<String> {
        self.rerank_capability.supported_models()
    }
}
