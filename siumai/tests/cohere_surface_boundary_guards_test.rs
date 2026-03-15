#![cfg(all(feature = "openai", feature = "cohere"))]

use async_trait::async_trait;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use siumai::Provider;
use siumai::experimental::execution::http::transport::{
    HttpTransport, HttpTransportRequest, HttpTransportResponse,
};
use siumai::prelude::unified::{EmbeddingExtensions, EmbeddingRequest, LlmError};
use siumai::provider_ext::cohere::CohereConfig;
use siumai::provider_ext::openai_compatible::{
    ConfigurableAdapter, OpenAiCompatibleClient, OpenAiCompatibleConfig, get_provider_config,
    provider_supports_capability,
};
use siumai_core::types::EmbeddingFormat;
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
struct CaptureTransport {
    last: Arc<Mutex<Option<HttpTransportRequest>>>,
}

impl CaptureTransport {
    fn take(&self) -> Option<HttpTransportRequest> {
        self.last.lock().expect("lock").take()
    }
}

#[async_trait]
impl HttpTransport for CaptureTransport {
    async fn execute_json(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        *self.last.lock().expect("lock") = Some(request);

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        Ok(HttpTransportResponse {
            status: 401,
            headers,
            body:
                br#"{"error":{"message":"unauthorized","type":"auth_error","code":"unauthorized"}}"#
                    .to_vec(),
        })
    }
}

async fn make_compat_client(
    provider_id: &str,
    model: &str,
    transport: Arc<dyn HttpTransport>,
) -> OpenAiCompatibleClient {
    let provider = get_provider_config(provider_id).expect("builtin provider config");
    let adapter = Arc::new(ConfigurableAdapter::new(provider.clone()));

    let config = OpenAiCompatibleConfig::new(provider_id, "test-key", &provider.base_url, adapter)
        .with_model(model)
        .with_http_transport(transport);

    OpenAiCompatibleClient::from_config(config)
        .await
        .expect("build compat config client")
}

#[test]
fn cohere_boundary_keeps_native_package_rerank_led() {
    let compat_registry = get_provider_config("cohere").expect("cohere compat config");
    assert!(provider_supports_capability("cohere", "embedding"));

    let native_config = Provider::cohere()
        .api_key("test-key")
        .into_config()
        .expect("native cohere config");

    let compat_config = Provider::openai()
        .compatible("cohere")
        .api_key("test-key")
        .into_config()
        .expect("compat cohere config");

    assert_eq!(native_config.base_url, CohereConfig::DEFAULT_BASE_URL);
    assert_eq!(
        native_config.common_params.model,
        CohereConfig::DEFAULT_MODEL
    );

    assert_eq!(compat_config.provider_id, "cohere");
    assert_eq!(compat_config.base_url, compat_registry.base_url);
    assert_ne!(native_config.base_url, compat_config.base_url);
}

#[tokio::test]
async fn cohere_embedding_stays_on_compat_only_public_story() {
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let provider_client = Provider::openai()
        .compatible("cohere")
        .api_key("test-key")
        .model("embed-english-v3.0")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider compat client");

    let config_client = make_compat_client(
        "cohere",
        "embed-english-v3.0",
        Arc::new(config_transport.clone()),
    )
    .await;

    let request = EmbeddingRequest::single("hello cohere embedding")
        .with_model("embed-english-v3.0")
        .with_dimensions(1024)
        .with_encoding_format(EmbeddingFormat::Float)
        .with_user("compat-user-cohere");

    let _ = provider_client.embed_with_config(request.clone()).await;
    let _ = config_client.embed_with_config(request).await;

    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_eq!(provider_req.url, "https://api.cohere.ai/v1/embeddings");
    assert_eq!(provider_req.url, config_req.url);
    assert_eq!(provider_req.body, config_req.body);
    assert_eq!(
        provider_req.body["model"],
        serde_json::json!("embed-english-v3.0")
    );
    assert_eq!(
        provider_req.body["input"],
        serde_json::json!(["hello cohere embedding"])
    );
    assert_eq!(provider_req.body["dimensions"], serde_json::json!(1024));
    assert_eq!(
        provider_req.body["encoding_format"],
        serde_json::json!("float")
    );
    assert_eq!(
        provider_req.body["user"],
        serde_json::json!("compat-user-cohere")
    );
}
