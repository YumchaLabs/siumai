#![cfg(all(feature = "openai", feature = "cohere"))]

use async_trait::async_trait;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use siumai::compat::Provider;
use siumai::experimental::execution::http::transport::{
    HttpTransport, HttpTransportRequest, HttpTransportResponse,
};
use siumai::prelude::unified::{EmbeddingExtensions, EmbeddingRequest, LlmError};
use siumai::provider_ext::cohere::{CohereClient, CohereConfig};
use siumai::provider_ext::openai_compatible::{
    ConfigurableAdapter, OpenAiCompatibleClient, OpenAiCompatibleConfig, get_provider_config,
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
fn cohere_native_builder_requires_explicit_model() {
    let err = Provider::cohere()
        .api_key("test-key")
        .into_config()
        .expect_err("native cohere should require an explicit model");

    match err {
        LlmError::ConfigurationError(message) => {
            assert!(message.contains("explicit model id"));
        }
        other => panic!("expected ConfigurationError, got: {other:?}"),
    }
}

#[test]
fn cohere_native_and_compat_configs_have_distinct_contracts() {
    let compat_registry = get_provider_config("cohere").expect("cohere compat config");

    let native_config = Provider::cohere()
        .api_key("test-key")
        .language_model("command-a-03-2025")
        .into_config()
        .expect("native cohere config");

    let compat_config = Provider::openai()
        .compatible("cohere")
        .api_key("test-key")
        .into_config()
        .expect("compat cohere config");

    assert_eq!(native_config.base_url, CohereConfig::DEFAULT_BASE_URL);
    assert_eq!(native_config.common_params.model, "command-a-03-2025");

    assert_eq!(compat_config.provider_id, "cohere");
    assert_eq!(compat_config.base_url, compat_registry.base_url);
    assert_ne!(native_config.base_url, compat_config.base_url);
}

#[tokio::test]
async fn cohere_native_v2_embedding_surface_remains_distinct_from_compat_v1_surface() {
    let native_transport = CaptureTransport::default();
    let compat_transport = CaptureTransport::default();

    let native_client = CohereClient::from_config(
        CohereConfig::new("test-key")
            .with_model("embed-v4.0")
            .with_http_transport(Arc::new(native_transport.clone())),
    )
    .expect("build native cohere client");

    let compat_client = make_compat_client(
        "cohere",
        "embed-english-v3.0",
        Arc::new(compat_transport.clone()),
    )
    .await;

    let native_request = EmbeddingRequest::single("hello native embedding")
        .with_model("embed-v4.0")
        .with_dimensions(1024);
    let compat_request = EmbeddingRequest::single("hello compat embedding")
        .with_model("embed-english-v3.0")
        .with_dimensions(1024)
        .with_encoding_format(EmbeddingFormat::Float)
        .with_user("compat-user");

    let _ = native_client.embed_with_config(native_request).await;
    let _ = compat_client.embed_with_config(compat_request).await;

    let native_req = native_transport.take().expect("native request");
    let compat_req = compat_transport.take().expect("compat request");

    assert_eq!(native_req.url, "https://api.cohere.com/v2/embed");
    assert_eq!(compat_req.url, "https://api.cohere.ai/v1/embeddings");
    assert_ne!(native_req.url, compat_req.url);
}
