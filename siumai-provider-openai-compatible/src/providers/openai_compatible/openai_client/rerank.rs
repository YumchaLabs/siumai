use super::OpenAiCompatibleClient;
use crate::error::LlmError;
use crate::execution::executors::rerank::{
    HttpRerankExecutor, RerankExecutor, RerankExecutorBuilder,
};
use crate::traits::RerankCapability;
use crate::types::{RerankRequest, RerankResponse};
use async_trait::async_trait;
use std::sync::Arc;

impl OpenAiCompatibleClient {
    fn resolve_rerank_model_default(&self) -> Option<String> {
        self.resolve_family_model_or_config(super::super::config::get_default_rerank_model(
            &self.config.provider_id,
        ))
    }

    fn ensure_rerank_surface(&self) -> Result<(), LlmError> {
        if !self.config.adapter.capabilities().supports("rerank") {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not support rerank",
                self.config.provider_id
            )));
        }

        Ok(())
    }

    async fn build_rerank_executor(
        &self,
        request: &RerankRequest,
    ) -> Result<Arc<HttpRerankExecutor>, LlmError> {
        let ctx = self.build_context().await?;
        let spec = Arc::new(self.compat_spec());

        let mut builder =
            RerankExecutorBuilder::new(self.config.provider_id.clone(), self.http_client.clone())
                .with_spec(spec)
                .with_context(ctx)
                .with_interceptors(self.http_interceptors.clone());

        if let Some(transport) = self.config.http_transport.clone() {
            builder = builder.with_transport(transport);
        }

        if let Some(retry) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry);
        }

        Ok(builder.build_for_request(request))
    }
}

#[async_trait]
impl RerankCapability for OpenAiCompatibleClient {
    async fn rerank(&self, mut request: RerankRequest) -> Result<RerankResponse, LlmError> {
        self.ensure_rerank_surface()?;

        if request.model.trim().is_empty()
            && let Some(model) = self.resolve_rerank_model_default()
        {
            request.model = model;
        }

        let exec = self.build_rerank_executor(&request).await?;
        RerankExecutor::execute(&*exec, request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::http::transport::{
        HttpTransport, HttpTransportRequest, HttpTransportResponse,
    };
    use crate::providers::openai_compatible::OpenAiCompatibleConfig;
    use crate::standards::openai::compat::provider_registry::{
        ConfigurableAdapter, ProviderConfig, ProviderFieldMappings,
    };
    use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
    use std::sync::Mutex;

    #[derive(Clone, Default)]
    struct NoopInterceptor;

    impl crate::execution::http::interceptor::HttpInterceptor for NoopInterceptor {}

    #[derive(Clone, Default)]
    struct CaptureTransport {
        last: Arc<Mutex<Option<HttpTransportRequest>>>,
    }

    impl CaptureTransport {
        fn take(&self) -> Option<HttpTransportRequest> {
            self.last.lock().unwrap().take()
        }
    }

    #[async_trait]
    impl HttpTransport for CaptureTransport {
        async fn execute_json(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            *self.last.lock().unwrap() = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportResponse {
                status: 401,
                headers,
                body: br#"{"error":{"message":"unauthorized","type":"auth_error","code":"unauthorized"}}"#
                    .to_vec(),
            })
        }
    }

    fn make_jina_rerank_adapter() -> Arc<ConfigurableAdapter> {
        Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "jina".to_string(),
            name: "Jina AI".to_string(),
            base_url: "https://api.jina.ai/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["rerank".to_string()],
            default_model: Some("jina-reranker-m0".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }))
    }

    fn make_voyageai_rerank_adapter() -> Arc<ConfigurableAdapter> {
        Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "voyageai".to_string(),
            name: "VoyageAI".to_string(),
            base_url: "https://api.voyageai.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["rerank".to_string()],
            default_model: Some("rerank-2".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }))
    }

    fn make_siliconflow_rerank_adapter() -> Arc<ConfigurableAdapter> {
        Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "siliconflow".to_string(),
            name: "SiliconFlow".to_string(),
            base_url: "https://api.siliconflow.cn/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["rerank".to_string()],
            default_model: Some("BAAI/bge-reranker-v2-m3".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }))
    }

    #[tokio::test]
    async fn rerank_runtime_jina_missing_model_uses_rerank_family_default() {
        let transport = CaptureTransport::default();
        let cfg = OpenAiCompatibleConfig::new(
            "jina",
            "test-key",
            "https://api.jina.ai/v1",
            make_jina_rerank_adapter(),
        )
        .with_model("jina-embeddings-v2-base-en")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = RerankRequest::new(
            String::new(),
            "query".to_string(),
            vec!["doc-1".to_string(), "doc-2".to_string()],
        )
        .with_top_n(1);

        let _ = client.rerank(request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(
            captured.body["model"],
            serde_json::json!("jina-reranker-m0")
        );
    }

    #[tokio::test]
    async fn rerank_runtime_siliconflow_preserves_request_shape_at_transport_boundary() {
        let transport = CaptureTransport::default();
        let cfg = OpenAiCompatibleConfig::new(
            "siliconflow",
            "test-key",
            "https://api.siliconflow.cn/v1",
            make_siliconflow_rerank_adapter(),
        )
        .with_model("BAAI/bge-reranker-v2-m3")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = RerankRequest::new(
            "BAAI/bge-reranker-v2-m3".to_string(),
            "query".to_string(),
            vec!["doc-1".to_string(), "doc-2".to_string()],
        )
        .with_top_n(1);

        let _ = client.rerank(request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(captured.url, "https://api.siliconflow.cn/v1/rerank");
        assert_eq!(
            captured.body["model"],
            serde_json::json!("BAAI/bge-reranker-v2-m3")
        );
        assert_eq!(captured.body["query"], serde_json::json!("query"));
        assert_eq!(
            captured.body["documents"],
            serde_json::json!(["doc-1", "doc-2"])
        );
        assert_eq!(captured.body["top_n"], serde_json::json!(1));
    }

    #[tokio::test]
    async fn rerank_runtime_jina_preserves_request_shape_at_transport_boundary() {
        let transport = CaptureTransport::default();
        let cfg = OpenAiCompatibleConfig::new(
            "jina",
            "test-key",
            "https://api.jina.ai/v1",
            make_jina_rerank_adapter(),
        )
        .with_model("jina-reranker-m0")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = RerankRequest::new(
            "jina-reranker-m0".to_string(),
            "query".to_string(),
            vec!["doc-1".to_string(), "doc-2".to_string()],
        )
        .with_top_n(1);

        let _ = client.rerank(request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(captured.url, "https://api.jina.ai/v1/rerank");
        assert_eq!(
            captured.body["model"],
            serde_json::json!("jina-reranker-m0")
        );
        assert_eq!(captured.body["query"], serde_json::json!("query"));
        assert_eq!(
            captured.body["documents"],
            serde_json::json!(["doc-1", "doc-2"])
        );
        assert_eq!(captured.body["top_n"], serde_json::json!(1));
    }

    #[tokio::test]
    async fn rerank_runtime_voyageai_preserves_request_shape_at_transport_boundary() {
        let transport = CaptureTransport::default();
        let cfg = OpenAiCompatibleConfig::new(
            "voyageai",
            "test-key",
            "https://api.voyageai.com/v1",
            make_voyageai_rerank_adapter(),
        )
        .with_model("rerank-2")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = RerankRequest::new(
            "rerank-2".to_string(),
            "query".to_string(),
            vec!["doc-1".to_string(), "doc-2".to_string()],
        )
        .with_top_n(1);

        let _ = client.rerank(request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(captured.url, "https://api.voyageai.com/v1/rerank");
        assert_eq!(captured.body["model"], serde_json::json!("rerank-2"));
        assert_eq!(captured.body["query"], serde_json::json!("query"));
        assert_eq!(
            captured.body["documents"],
            serde_json::json!(["doc-1", "doc-2"])
        );
        assert_eq!(captured.body["top_n"], serde_json::json!(1));
    }

    #[tokio::test]
    async fn build_rerank_executor_wires_transport_and_interceptors() {
        let transport = CaptureTransport::default();
        let cfg = OpenAiCompatibleConfig::new(
            "jina",
            "test-key",
            "https://api.jina.ai/v1",
            make_jina_rerank_adapter(),
        )
        .with_model("jina-reranker-m0")
        .with_http_transport(Arc::new(transport));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok")
            .with_http_interceptors(vec![Arc::new(NoopInterceptor)]);

        let request = RerankRequest::new(
            "jina-reranker-m0".to_string(),
            "query".to_string(),
            vec!["doc-1".to_string(), "doc-2".to_string()],
        );

        let exec = client.build_rerank_executor(&request).await.unwrap();
        assert_eq!(exec.policy.interceptors.len(), 1);
        assert!(exec.policy.transport.is_some());
        assert!(exec.policy.before_send.is_none());
        assert!(exec.provider_spec.capabilities().supports("rerank"));
        assert_eq!(exec.provider_context.provider_id, "jina");
    }

    #[test]
    fn rerank_logic_stays_out_of_monolithic_client_module() {
        let source = include_str!("../openai_client.rs");
        for forbidden in [
            "fn resolve_rerank_model_default(",
            "fn ensure_rerank_surface(",
            "fn build_rerank_executor(",
            "impl RerankCapability for OpenAiCompatibleClient",
            "RerankExecutorBuilder::new(",
        ] {
            assert!(
                !source.contains(forbidden),
                "OpenAI-compatible rerank logic should live in openai_client/rerank.rs"
            );
        }
    }
}
