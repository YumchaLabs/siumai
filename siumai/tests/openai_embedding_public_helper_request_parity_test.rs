#![cfg(feature = "openai")]

use async_trait::async_trait;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use serde_json::json;
use siumai::embedding::{self, EmbedOptions};
use siumai::experimental::execution::http::transport::{
    HttpTransport, HttpTransportRequest, HttpTransportResponse,
};
use siumai::prelude::unified::registry::{RegistryOptions, create_provider_registry};
use siumai::prelude::unified::{EmbeddingRequest, LlmError};
use siumai::registry::ProviderBuildOverrides;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct JsonSuccessTransport {
    response_body: Arc<Vec<u8>>,
    last: Arc<Mutex<Option<HttpTransportRequest>>>,
}

impl JsonSuccessTransport {
    fn new(response: serde_json::Value) -> Self {
        Self {
            response_body: Arc::new(serde_json::to_vec(&response).expect("response json")),
            last: Arc::new(Mutex::new(None)),
        }
    }

    fn take(&self) -> Option<HttpTransportRequest> {
        self.last.lock().expect("lock request").take()
    }
}

#[async_trait]
impl HttpTransport for JsonSuccessTransport {
    async fn execute_json(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        *self.last.lock().expect("lock request") = Some(request);

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
        "openai".to_string(),
        Arc::new(siumai::registry::factories::OpenAIProviderFactory)
            as Arc<dyn siumai::prelude::unified::registry::ProviderFactory>,
    );

    let mut provider_build_overrides = HashMap::new();
    provider_build_overrides.insert(
        "openai".to_string(),
        ProviderBuildOverrides::default()
            .with_api_key("test-key")
            .with_base_url("https://example.invalid/openai")
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

#[tokio::test]
async fn openai_embedding_public_helper_preserves_request_config_on_registry_handle() {
    let response = json!({
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.1, 0.2, 0.3],
                "index": 0
            }
        ],
        "model": "text-embedding-3-large",
        "usage": {
            "prompt_tokens": 1,
            "total_tokens": 1
        }
    });

    let transport = JsonSuccessTransport::new(response);
    let registry = make_registry(Arc::new(transport.clone()));
    let model = registry
        .embedding_model("openai:text-embedding-3-small")
        .expect("build registry embedding model");

    let request = EmbeddingRequest::single("hello from public helper")
        .with_model("text-embedding-3-large")
        .with_dimensions(1024)
        .with_user("public-user")
        .with_header("x-request-header", "from-request");

    let mut headers = HashMap::new();
    headers.insert("x-helper-header".to_string(), "from-helper".to_string());

    let response = embedding::embed(
        &model,
        request,
        EmbedOptions {
            retry: None,
            timeout: None,
            headers,
        },
    )
    .await
    .expect("embedding response");

    assert_eq!(response.model, "text-embedding-3-large");
    assert_eq!(response.embeddings.len(), 1);

    let sent = transport.take().expect("captured request");
    assert!(sent.url.ends_with("/embeddings"));
    assert_eq!(sent.body["input"], json!(["hello from public helper"]));
    assert_eq!(sent.body["model"], json!("text-embedding-3-large"));
    assert_eq!(sent.body["dimensions"], json!(1024));
    assert_eq!(sent.body["user"], json!("public-user"));
    assert_eq!(
        sent.headers
            .get("x-request-header")
            .and_then(|value| value.to_str().ok()),
        Some("from-request")
    );
    assert_eq!(
        sent.headers
            .get("x-helper-header")
            .and_then(|value| value.to_str().ok()),
        Some("from-helper")
    );
}
