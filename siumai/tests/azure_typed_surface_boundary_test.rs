#![cfg(feature = "azure")]
#![allow(deprecated)]

use async_trait::async_trait;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use siumai::experimental::execution::http::transport::{
    HttpTransport, HttpTransportRequest, HttpTransportResponse,
};
use siumai::prelude::unified::*;
use siumai::provider_ext::azure::{AzureOpenAiClient, AzureOpenAiConfig, metadata::*, options::*};
use std::path::Path;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct JsonCaptureTransport {
    response_body: Arc<Vec<u8>>,
    last: Arc<Mutex<Option<HttpTransportRequest>>>,
}

impl JsonCaptureTransport {
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
impl HttpTransport for JsonCaptureTransport {
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

fn fixture_response(case: &str) -> serde_json::Value {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("azure")
        .join("responses")
        .join("response")
        .join(case)
        .join("response.json");
    let text = std::fs::read_to_string(&root).expect("read fixture response");
    serde_json::from_str(&text).expect("parse fixture response")
}

fn make_request() -> ChatRequest {
    ChatRequest::new(vec![ChatMessage::user("hi").build()])
        .with_response_format(ResponseFormat::json_schema(serde_json::json!({
            "type": "object",
            "properties": {
                "answer": { "type": "string" }
            },
            "required": ["answer"],
            "additionalProperties": false
        })))
        .with_azure_options(
            AzureOpenAiOptions::new()
                .with_force_reasoning(true)
                .with_reasoning_effort(AzureReasoningEffort::High)
                .with_strict_json_schema(false)
                .with_responses_reasoning_summary("detailed"),
        )
}

fn assert_request_has_typed_azure_options(request: &HttpTransportRequest) {
    assert_eq!(
        request.url,
        "https://example.invalid/openai/v1/responses?api-version=v1"
    );
    assert_eq!(request.body["model"], serde_json::json!("deployment-id"));
    assert_eq!(
        request.body["reasoning"]["effort"],
        serde_json::json!("high")
    );
    assert_eq!(
        request.body["reasoning"]["summary"],
        serde_json::json!("detailed")
    );
    assert_eq!(
        request.body["text"]["format"]["type"],
        serde_json::json!("json_schema")
    );
    assert_eq!(
        request.body["text"]["format"]["strict"],
        serde_json::json!(false)
    );
    assert_eq!(
        request.body["text"]["format"]["name"],
        serde_json::json!("response")
    );
}

#[tokio::test]
async fn azure_request_ext_shapes_final_responses_body_across_public_paths() {
    let response_json = fixture_response("azure-web-search-preview-tool.1");
    let siumai_transport = JsonCaptureTransport::new(response_json.clone());
    let provider_transport = JsonCaptureTransport::new(response_json.clone());
    let config_transport = JsonCaptureTransport::new(response_json);

    let siumai_client = Siumai::builder()
        .azure()
        .api_key("test-key")
        .base_url("https://example.invalid/openai")
        .model("deployment-id")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = siumai::Provider::azure()
        .api_key("test-key")
        .base_url("https://example.invalid/openai")
        .model("deployment-id")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = AzureOpenAiClient::from_config(
        AzureOpenAiConfig::new("test-key")
            .with_base_url("https://example.invalid/openai")
            .with_model("deployment-id")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let _ = siumai_client
        .chat_request(make_request())
        .await
        .expect("siumai chat");
    let _ = provider_client
        .chat_request(make_request())
        .await
        .expect("provider chat");
    let _ = config_client
        .chat_request(make_request())
        .await
        .expect("config chat");

    let siumai_request = siumai_transport.take().expect("siumai request");
    let provider_request = provider_transport.take().expect("provider request");
    let config_request = config_transport.take().expect("config request");

    assert_request_has_typed_azure_options(&siumai_request);
    assert_eq!(provider_request.url, siumai_request.url);
    assert_eq!(provider_request.body, siumai_request.body);
    assert_eq!(config_request.url, siumai_request.url);
    assert_eq!(config_request.body, siumai_request.body);
}

#[tokio::test]
async fn azure_metadata_ext_reads_default_key_across_public_paths() {
    let response_json = fixture_response("azure-web-search-preview-tool.1");
    let siumai_transport = JsonCaptureTransport::new(response_json.clone());
    let provider_transport = JsonCaptureTransport::new(response_json.clone());
    let config_transport = JsonCaptureTransport::new(response_json);

    let siumai_client = Siumai::builder()
        .azure()
        .api_key("test-key")
        .base_url("https://example.invalid/openai")
        .model("deployment-id")
        .fetch(Arc::new(siumai_transport))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = siumai::Provider::azure()
        .api_key("test-key")
        .base_url("https://example.invalid/openai")
        .model("deployment-id")
        .fetch(Arc::new(provider_transport))
        .build()
        .await
        .expect("build provider client");

    let config_client = AzureOpenAiClient::from_config(
        AzureOpenAiConfig::new("test-key")
            .with_base_url("https://example.invalid/openai")
            .with_model("deployment-id")
            .with_http_transport(Arc::new(config_transport)),
    )
    .expect("build config client");

    for response in [
        siumai_client
            .chat_request(ChatRequest::new(vec![ChatMessage::user("hi").build()]))
            .await
            .expect("siumai chat"),
        provider_client
            .chat_request(ChatRequest::new(vec![ChatMessage::user("hi").build()]))
            .await
            .expect("provider chat"),
        config_client
            .chat_request(ChatRequest::new(vec![ChatMessage::user("hi").build()]))
            .await
            .expect("config chat"),
    ] {
        let provider_metadata = response
            .provider_metadata
            .as_ref()
            .expect("provider metadata present");
        assert!(provider_metadata.contains_key("azure"));

        let typed = response.azure_metadata().expect("typed azure metadata");
        assert!(
            typed
                .sources
                .as_ref()
                .is_some_and(|sources| !sources.is_empty()),
            "expected typed azure sources"
        );
    }
}

#[tokio::test]
async fn azure_metadata_ext_supports_custom_provider_metadata_key() {
    let transport = JsonCaptureTransport::new(fixture_response("azure-web-search-preview-tool.1"));
    let client = AzureOpenAiClient::from_config(
        AzureOpenAiConfig::new("test-key")
            .with_base_url("https://example.invalid/openai")
            .with_model("deployment-id")
            .with_provider_metadata_key("openai")
            .with_http_transport(Arc::new(transport)),
    )
    .expect("build config client");

    let response = client
        .chat_request(ChatRequest::new(vec![ChatMessage::user("hi").build()]))
        .await
        .expect("config chat");

    let provider_metadata = response
        .provider_metadata
        .as_ref()
        .expect("provider metadata present");
    assert!(!provider_metadata.contains_key("azure"));
    assert!(provider_metadata.contains_key("openai"));
    assert!(response.azure_metadata().is_none());
    let typed = response
        .azure_metadata_with_key("openai")
        .expect("typed custom-key azure metadata");
    assert!(
        typed
            .sources
            .as_ref()
            .is_some_and(|sources| !sources.is_empty()),
        "expected typed azure sources under custom key"
    );
}
