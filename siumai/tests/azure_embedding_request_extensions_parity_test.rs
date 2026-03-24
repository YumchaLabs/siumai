#![cfg(feature = "azure")]
#![allow(deprecated)]

use async_trait::async_trait;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use serde_json::json;
use siumai::Provider;
use siumai::experimental::execution::http::transport::{
    HttpTransport, HttpTransportRequest, HttpTransportResponse,
};
use siumai::prelude::unified::{EmbeddingExtensions, EmbeddingRequest, LlmError, Siumai};
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

fn assert_requests_equivalent(left: &HttpTransportRequest, right: &HttpTransportRequest) {
    assert_eq!(left.url, right.url);
    assert_eq!(left.body, right.body);
    assert_eq!(
        left.headers
            .get("api-key")
            .and_then(|value| value.to_str().ok()),
        right
            .headers
            .get("api-key")
            .and_then(|value| value.to_str().ok())
    );
    assert_eq!(
        left.headers
            .get("x-azure-embed")
            .and_then(|value| value.to_str().ok()),
        right
            .headers
            .get("x-azure-embed")
            .and_then(|value| value.to_str().ok())
    );
}

#[tokio::test]
async fn azure_embedding_request_extensions_are_equivalent_on_public_paths() {
    let response = json!({
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.1, 0.2, 0.3],
                "index": 0
            }
        ],
        "model": "request-deployment",
        "usage": {
            "prompt_tokens": 1,
            "total_tokens": 1
        }
    });

    let siumai_transport = JsonSuccessTransport::new(response.clone());
    let provider_transport = JsonSuccessTransport::new(response.clone());
    let config_transport = JsonSuccessTransport::new(response);

    let base_url = "https://example.invalid/openai";

    let siumai_client = Siumai::builder()
        .azure()
        .api_key("test-key")
        .base_url(base_url)
        .model("default-deployment")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::azure()
        .api_key("test-key")
        .base_url(base_url)
        .model("default-deployment")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::azure::AzureOpenAiClient::from_config(
        siumai::provider_ext::azure::AzureOpenAiConfig::new("test-key")
            .with_base_url(base_url)
            .with_embedding_model("default-deployment")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = EmbeddingRequest::single("hello azure request-aware embedding")
        .with_model("request-deployment")
        .with_dimensions(256)
        .with_user("azure-user")
        .with_header("x-azure-embed", "yes");

    let siumai_resp = siumai_client
        .embed_with_config(request.clone())
        .await
        .expect("siumai response");
    let provider_resp = provider_client
        .embed_with_config(request.clone())
        .await
        .expect("provider response");
    let config_resp = config_client
        .embed_with_config(request)
        .await
        .expect("config response");

    assert_eq!(siumai_resp.model, "request-deployment");
    assert_eq!(provider_resp.model, "request-deployment");
    assert_eq!(config_resp.model, "request-deployment");

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(
        siumai_req.url,
        "https://example.invalid/openai/v1/embeddings?api-version=v1"
    );
    assert_eq!(siumai_req.body["model"], json!("request-deployment"));
    assert_eq!(
        siumai_req.body["input"],
        json!(["hello azure request-aware embedding"])
    );
    assert_eq!(siumai_req.body["dimensions"], json!(256));
    assert_eq!(siumai_req.body["user"], json!("azure-user"));
}
