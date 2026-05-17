use super::{OpenAiCompatibleClient, model_slot_is_missing};
use crate::error::LlmError;
use crate::execution::executors::embedding::{
    EmbeddingExecutor, EmbeddingExecutorBuilder, HttpEmbeddingExecutor,
};
use crate::traits::EmbeddingCapability;
use crate::types::{EmbeddingRequest, EmbeddingResponse};
use async_trait::async_trait;
use std::sync::Arc;

impl OpenAiCompatibleClient {
    fn resolve_embedding_model_default(&self) -> Option<String> {
        self.resolve_family_model_or_config(super::super::config::get_default_embedding_model(
            &self.config.provider_id,
        ))
    }

    fn ensure_embedding_surface(&self) -> Result<(), LlmError> {
        if !self.config.adapter.capabilities().embedding {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not support embeddings",
                self.config.provider_id
            )));
        }

        Ok(())
    }

    async fn build_embedding_executor(
        &self,
        request: &EmbeddingRequest,
    ) -> Result<Arc<HttpEmbeddingExecutor>, LlmError> {
        let ctx = self.build_context().await?;
        let spec = Arc::new(self.compat_spec());
        let mut builder = EmbeddingExecutorBuilder::new(
            self.config.provider_id.clone(),
            self.http_client.clone(),
        )
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
impl EmbeddingCapability for OpenAiCompatibleClient {
    async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        self.ensure_embedding_surface()?;
        let mut req = EmbeddingRequest::new(texts);
        if let Some(model) = self.resolve_embedding_model_default() {
            req.model = Some(model);
        }
        let exec = self.build_embedding_executor(&req).await?;
        EmbeddingExecutor::execute(&*exec, req).await
    }

    fn as_embedding_extensions(&self) -> Option<&dyn crate::traits::EmbeddingExtensions> {
        Some(self)
    }

    fn embedding_dimension(&self) -> usize {
        1536
    }
}

#[async_trait]
impl crate::traits::EmbeddingExtensions for OpenAiCompatibleClient {
    async fn embed_with_config(
        &self,
        mut request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LlmError> {
        self.ensure_embedding_surface()?;
        if model_slot_is_missing(request.model.as_deref()) {
            request.model = self.resolve_embedding_model_default();
        }

        let exec = self.build_embedding_executor(&request).await?;
        EmbeddingExecutor::execute(&*exec, request).await
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
    use crate::types::EmbeddingFormat;
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

    fn make_fireworks_embedding_adapter() -> Arc<ConfigurableAdapter> {
        Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "fireworks".to_string(),
            name: "Fireworks AI".to_string(),
            base_url: "https://api.fireworks.ai/inference/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["embedding".to_string()],
            default_model: Some("nomic-ai/nomic-embed-text-v1.5".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }))
    }

    fn make_infini_embedding_adapter() -> Arc<ConfigurableAdapter> {
        Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "infini".to_string(),
            name: "Infini AI".to_string(),
            base_url: "https://cloud.infini-ai.com/maas/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["embedding".to_string()],
            default_model: Some("text-embedding-3-small".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }))
    }

    fn make_together_embedding_adapter() -> Arc<ConfigurableAdapter> {
        Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "together".to_string(),
            name: "Together AI".to_string(),
            base_url: "https://api.together.xyz/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["embedding".to_string()],
            default_model: Some("togethercomputer/m2-bert-80M-8k-retrieval".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }))
    }

    #[tokio::test]
    async fn embed_with_config_runtime_fireworks_uses_inference_boundary_and_preserves_request_shape()
     {
        let transport = CaptureTransport::default();
        let cfg = OpenAiCompatibleConfig::new(
            "fireworks",
            "test-key",
            "https://api.fireworks.ai/inference/v1",
            make_fireworks_embedding_adapter(),
        )
        .with_model("nomic-ai/nomic-embed-text-v1.5")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = EmbeddingRequest::single("hello fireworks embedding")
            .with_model("nomic-ai/nomic-embed-text-v1.5")
            .with_dimensions(256)
            .with_encoding_format(EmbeddingFormat::Base64)
            .with_user("compat-user-1");

        let _ = crate::traits::EmbeddingExtensions::embed_with_config(&client, request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(
            captured.url,
            "https://api.fireworks.ai/inference/v1/embeddings"
        );
        assert_eq!(
            captured.body["model"],
            serde_json::json!("nomic-ai/nomic-embed-text-v1.5")
        );
        assert_eq!(
            captured.body["input"],
            serde_json::json!(["hello fireworks embedding"])
        );
        assert_eq!(captured.body["dimensions"], serde_json::json!(256));
        assert_eq!(
            captured.body["encoding_format"],
            serde_json::json!("base64")
        );
        assert_eq!(captured.body["user"], serde_json::json!("compat-user-1"));
    }

    #[tokio::test]
    async fn embed_with_config_runtime_infini_preserves_request_shape_at_transport_boundary() {
        let transport = CaptureTransport::default();
        let cfg = OpenAiCompatibleConfig::new(
            "infini",
            "test-key",
            "https://cloud.infini-ai.com/maas/v1",
            make_infini_embedding_adapter(),
        )
        .with_model("text-embedding-3-small")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = EmbeddingRequest::single("hello infini embedding")
            .with_model("text-embedding-3-small")
            .with_dimensions(512)
            .with_encoding_format(EmbeddingFormat::Float)
            .with_user("compat-user-7");

        let _ = crate::traits::EmbeddingExtensions::embed_with_config(&client, request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(
            captured.url,
            "https://cloud.infini-ai.com/maas/v1/embeddings"
        );
        assert_eq!(
            captured.body["model"],
            serde_json::json!("text-embedding-3-small")
        );
        assert_eq!(
            captured.body["input"],
            serde_json::json!(["hello infini embedding"])
        );
        assert_eq!(captured.body["dimensions"], serde_json::json!(512));
        assert_eq!(captured.body["encoding_format"], serde_json::json!("float"));
        assert_eq!(captured.body["user"], serde_json::json!("compat-user-7"));
    }

    #[tokio::test]
    async fn embed_with_config_runtime_together_preserves_request_shape_at_transport_boundary() {
        let transport = CaptureTransport::default();
        let cfg = OpenAiCompatibleConfig::new(
            "together",
            "test-key",
            "https://api.together.xyz/v1",
            make_together_embedding_adapter(),
        )
        .with_model("togethercomputer/m2-bert-80M-8k-retrieval")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = EmbeddingRequest::single("hello together embedding")
            .with_model("togethercomputer/m2-bert-80M-8k-retrieval")
            .with_dimensions(384)
            .with_encoding_format(EmbeddingFormat::Float)
            .with_user("compat-user-4");

        let _ = crate::traits::EmbeddingExtensions::embed_with_config(&client, request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(captured.url, "https://api.together.xyz/v1/embeddings");
        assert_eq!(
            captured.body["model"],
            serde_json::json!("togethercomputer/m2-bert-80M-8k-retrieval")
        );
        assert_eq!(
            captured.body["input"],
            serde_json::json!(["hello together embedding"])
        );
        assert_eq!(captured.body["dimensions"], serde_json::json!(384));
        assert_eq!(captured.body["encoding_format"], serde_json::json!("float"));
        assert_eq!(captured.body["user"], serde_json::json!("compat-user-4"));
    }

    #[tokio::test]
    async fn embed_with_config_runtime_together_missing_model_uses_embedding_family_default() {
        let transport = CaptureTransport::default();
        let cfg = OpenAiCompatibleConfig::new(
            "together",
            "test-key",
            "https://api.together.xyz/v1",
            make_together_embedding_adapter(),
        )
        .with_model("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = EmbeddingRequest::single("hello together embedding")
            .with_dimensions(384)
            .with_encoding_format(EmbeddingFormat::Float)
            .with_user("compat-user-4");

        let _ = crate::traits::EmbeddingExtensions::embed_with_config(&client, request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(
            captured.body["model"],
            serde_json::json!("togethercomputer/m2-bert-80M-8k-retrieval")
        );
    }

    #[tokio::test]
    async fn embed_with_config_runtime_together_missing_model_preserves_explicit_config_override() {
        let transport = CaptureTransport::default();
        let cfg = OpenAiCompatibleConfig::new(
            "together",
            "test-key",
            "https://api.together.xyz/v1",
            make_together_embedding_adapter(),
        )
        .with_model("custom-embedding-override")
        .with_http_transport(Arc::new(transport.clone()));

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("client ok");

        let request = EmbeddingRequest::single("hello together embedding");

        let _ = crate::traits::EmbeddingExtensions::embed_with_config(&client, request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(
            captured.body["model"],
            serde_json::json!("custom-embedding-override")
        );
    }

    #[tokio::test]
    async fn build_embedding_executor_wires_before_send_and_interceptors() {
        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "deepseek".to_string(),
            name: "DeepSeek".to_string(),
            base_url: "https://api.test.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
                "embedding".to_string(),
            ],
            default_model: None,
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));

        let cfg =
            OpenAiCompatibleConfig::new("deepseek", "test-key", "https://api.test.com/v1", adapter)
                .with_model("text-embedding-3-small");

        let client = OpenAiCompatibleClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .unwrap()
            .with_http_interceptors(vec![Arc::new(NoopInterceptor)]);

        let req = EmbeddingRequest::single("hi")
            .with_model("text-embedding-3-small")
            .with_provider_option("deepseek", serde_json::json!({ "my_custom": 1 }));

        let exec = client.build_embedding_executor(&req).await.unwrap();
        assert_eq!(exec.policy.interceptors.len(), 1);
        assert!(exec.policy.before_send.is_none());
        assert!(
            exec.provider_spec
                .embedding_before_send(&req, &exec.provider_context)
                .is_some()
        );
    }

    #[test]
    fn embedding_logic_stays_out_of_monolithic_client_module() {
        let source = include_str!("../openai_client.rs");
        for forbidden in [
            "fn resolve_embedding_model_default(",
            "fn ensure_embedding_surface(",
            "fn build_embedding_executor(",
            "impl EmbeddingCapability for OpenAiCompatibleClient",
            "impl crate::traits::EmbeddingExtensions for OpenAiCompatibleClient",
        ] {
            assert!(
                !source.contains(forbidden),
                "OpenAI-compatible embedding logic should live in openai_client/embedding.rs"
            );
        }
    }
}
