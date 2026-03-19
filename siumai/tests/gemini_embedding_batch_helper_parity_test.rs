#![cfg(feature = "google")]

use async_trait::async_trait;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use serde_json::json;
use siumai::Provider;
use siumai::embedding::{self, BatchEmbeddingRequest, EmbedOptions};
use siumai::experimental::execution::http::transport::{
    HttpTransport, HttpTransportRequest, HttpTransportResponse,
};
use siumai::prelude::unified::registry::{RegistryOptions, create_provider_registry};
use siumai::prelude::unified::{EmbeddingRequest, LlmError, Siumai};
use siumai::registry::ProviderBuildOverrides;
use siumai_core::types::EmbeddingTaskType;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct RecordingTransport {
    calls: Arc<Mutex<Vec<HttpTransportRequest>>>,
    response_body: Arc<Vec<u8>>,
}

impl RecordingTransport {
    fn new(response: serde_json::Value) -> Self {
        Self {
            calls: Arc::new(Mutex::new(Vec::new())),
            response_body: Arc::new(serde_json::to_vec(&response).expect("response json")),
        }
    }

    fn calls(&self) -> Vec<HttpTransportRequest> {
        self.calls.lock().expect("lock calls").clone()
    }
}

#[async_trait]
impl HttpTransport for RecordingTransport {
    async fn execute_json(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        self.calls.lock().expect("lock calls").push(request);

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        Ok(HttpTransportResponse {
            status: 200,
            headers,
            body: self.response_body.as_ref().clone(),
        })
    }
}

fn make_registry(transport: Arc<dyn HttpTransport>) -> siumai::registry::ProviderRegistryHandle {
    let mut providers = HashMap::new();
    providers.insert(
        "gemini".to_string(),
        Arc::new(siumai::registry::factories::GeminiProviderFactory)
            as Arc<dyn siumai::prelude::unified::registry::ProviderFactory>,
    );

    let mut provider_build_overrides = HashMap::new();
    provider_build_overrides.insert(
        "gemini".to_string(),
        ProviderBuildOverrides::default()
            .with_api_key("test-key")
            .with_base_url("https://example.com/v1beta")
            .fetch(transport),
    );

    create_provider_registry(
        providers,
        Some(RegistryOptions {
            separator: ':',
            language_model_middleware: Vec::new(),
            http_interceptors: Vec::new(),
            http_client: None,
            http_transport: None,
            http_config: None,
            api_key: None,
            base_url: None,
            reasoning_enabled: None,
            reasoning_budget: None,
            provider_build_overrides,
            retry_options: None,
            max_cache_entries: None,
            client_ttl: None,
            auto_middleware: true,
        }),
    )
}

fn assert_batch_request_shape(request: &HttpTransportRequest) {
    assert_eq!(
        request.url,
        "https://example.com/v1beta/models/gemini-embedding-001:batchEmbedContents"
    );

    let requests = request.body["requests"]
        .as_array()
        .expect("batch requests array");
    assert_eq!(requests.len(), 2);

    for (index, text) in ["A", "B"].iter().enumerate() {
        let item = &requests[index];
        assert_eq!(
            item["model"],
            serde_json::json!("models/gemini-embedding-001")
        );
        assert_eq!(item["taskType"], serde_json::json!("SEMANTIC_SIMILARITY"));
        assert_eq!(item["outputDimensionality"], serde_json::json!(64));
        assert_eq!(item["content"]["role"], serde_json::json!("user"));
        assert_eq!(item["content"]["parts"][0]["text"], serde_json::json!(text));
    }
}

#[tokio::test]
async fn gemini_embedding_public_helper_coalesces_batch_requests_across_public_paths() {
    let response = json!({
        "embeddings": [
            { "values": [0.1, 0.2] },
            { "values": [0.3, 0.4] }
        ],
        "usageMetadata": {
            "promptTokenCount": 2,
            "totalTokenCount": 2
        }
    });

    let siumai_transport = RecordingTransport::new(response.clone());
    let provider_transport = RecordingTransport::new(response.clone());
    let config_transport = RecordingTransport::new(response.clone());
    let registry_transport = RecordingTransport::new(response);

    let siumai_client = Siumai::builder()
        .gemini()
        .api_key("test-key")
        .base_url("https://example.com/v1beta")
        .model("gemini-embedding-001")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::gemini()
        .api_key("test-key")
        .base_url("https://example.com/v1beta")
        .model("gemini-embedding-001")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::gemini::GeminiClient::from_config(
        siumai::provider_ext::gemini::GeminiConfig::new("test-key")
            .with_base_url("https://example.com/v1beta".to_string())
            .with_model("gemini-embedding-001".to_string())
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()));
    let registry_model = registry
        .embedding_model("gemini:gemini-embedding-001")
        .expect("build registry embedding model");

    let request = BatchEmbeddingRequest::new(vec![
        EmbeddingRequest::single("A")
            .with_dimensions(64)
            .with_task_type(EmbeddingTaskType::SemanticSimilarity),
        EmbeddingRequest::single("B")
            .with_dimensions(64)
            .with_task_type(EmbeddingTaskType::SemanticSimilarity),
    ]);

    let siumai_response =
        embedding::embed_many(&siumai_client, request.clone(), EmbedOptions::default())
            .await
            .expect("siumai batch embedding");
    let provider_response =
        embedding::embed_many(&provider_client, request.clone(), EmbedOptions::default())
            .await
            .expect("provider batch embedding");
    let config_response =
        embedding::embed_many(&config_client, request.clone(), EmbedOptions::default())
            .await
            .expect("config batch embedding");
    let registry_response =
        embedding::embed_many(&registry_model, request, EmbedOptions::default())
            .await
            .expect("registry batch embedding");

    for response in [
        &siumai_response,
        &provider_response,
        &config_response,
        &registry_response,
    ] {
        assert_eq!(response.responses.len(), 2);
        assert!(
            response
                .metadata
                .get("coalesced")
                .and_then(|value| value.as_bool())
                .unwrap_or(false)
        );

        let first = response.responses[0].as_ref().expect("first response");
        assert_eq!(first.model, "gemini-embedding-001");
        assert_eq!(first.embeddings, vec![vec![0.1, 0.2]]);

        let second = response.responses[1].as_ref().expect("second response");
        assert_eq!(second.model, "gemini-embedding-001");
        assert_eq!(second.embeddings, vec![vec![0.3, 0.4]]);
    }

    let siumai_calls = siumai_transport.calls();
    let provider_calls = provider_transport.calls();
    let config_calls = config_transport.calls();
    let registry_calls = registry_transport.calls();

    assert_eq!(siumai_calls.len(), 1);
    assert_eq!(provider_calls.len(), 1);
    assert_eq!(config_calls.len(), 1);
    assert_eq!(registry_calls.len(), 1);

    assert_batch_request_shape(&siumai_calls[0]);
    assert_batch_request_shape(&provider_calls[0]);
    assert_batch_request_shape(&config_calls[0]);
    assert_batch_request_shape(&registry_calls[0]);
}
