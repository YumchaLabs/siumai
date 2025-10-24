use std::sync::Arc;

use siumai::providers::openai_compatible::adapter::ProviderAdapter;
use siumai::providers::openai_compatible::openai_config::OpenAiCompatibleConfig;
use siumai::providers::openai_compatible::types::RequestType;
use siumai::providers::openai_compatible::types::{
    FieldAccessor, FieldMappings, JsonFieldAccessor, ModelConfig,
};
use siumai::traits::EmbeddingCapability;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[derive(Debug, Clone)]
struct MappingAdapter {
    base_url: String,
}

impl ProviderAdapter for MappingAdapter {
    fn provider_id(&self) -> &'static str {
        "mapping-test"
    }

    fn transform_request_params(
        &self,
        params: &mut serde_json::Value,
        _model: &str,
        request_type: RequestType,
    ) -> Result<(), siumai::error::LlmError> {
        if let RequestType::Embedding = request_type {
            if let Some(obj) = params.as_object_mut() {
                obj.insert("granularity".to_string(), serde_json::json!("high"));
            }
        }
        Ok(())
    }

    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn clone_adapter(&self) -> Box<dyn ProviderAdapter> {
        Box::new(self.clone())
    }

    fn get_field_mappings(&self, _model: &str) -> FieldMappings {
        FieldMappings::default()
    }

    fn get_model_config(&self, _model: &str) -> ModelConfig {
        ModelConfig::default()
    }

    fn get_field_accessor(&self) -> Box<dyn FieldAccessor> {
        Box::new(JsonFieldAccessor)
    }

    fn capabilities(&self) -> siumai::traits::ProviderCapabilities {
        siumai::traits::ProviderCapabilities::new().with_embedding()
    }
}

#[tokio::test]
async fn embedding_request_includes_adapter_mapping() {
    let server = MockServer::start().await;

    // Return a minimal OpenAI-compatible embedding response
    let response = serde_json::json!({
        "data": [
            {"embedding": [0.1, 0.2, 0.3], "index": 0}
        ],
        "model": "test-model",
        "usage": {"prompt_tokens": 1, "total_tokens": 1}
    });

    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response))
        .mount(&server)
        .await;

    let adapter = Arc::new(MappingAdapter {
        base_url: server.uri(),
    });
    // Register custom adapter into the global registry so Spec can resolve it
    {
        use siumai::registry::{ProviderRecord, global_registry};
        let mut guard = global_registry().lock().unwrap();
        let record = ProviderRecord {
            id: "mapping-test".to_string(),
            name: "Mapping Test".to_string(),
            base_url: Some(server.uri()),
            capabilities: siumai::traits::ProviderCapabilities::new().with_embedding(),
            adapter: Some(adapter.clone()),
            aliases: vec![],
            model_prefixes: vec![],
            default_model: Some("text-embedding-test".to_string()),
        };
        guard.register(record);
    }
    let config = OpenAiCompatibleConfig::new("mapping-test", "test-key", &server.uri(), adapter)
        .with_model("text-embedding-test");

    let client =
        siumai::providers::openai_compatible::openai_client::OpenAiCompatibleClient::new(config)
            .await
            .unwrap();

    // Issue embedding request
    let _ = client
        .embed(vec!["hello world".to_string()])
        .await
        .expect("embedding response");

    // Assert request body contained our adapter-mapped field
    let reqs = server.received_requests().await.unwrap();
    assert_eq!(reqs.len(), 1);
    let body: serde_json::Value = serde_json::from_slice(&reqs[0].body).unwrap();
    assert_eq!(body["granularity"], "high");
}
