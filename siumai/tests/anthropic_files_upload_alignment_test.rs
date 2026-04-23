#![cfg(feature = "anthropic")]

use siumai::files::{self, UploadFileOptions};
use siumai::provider_ext::anthropic::{AnthropicClient, AnthropicConfig};
use siumai_core::error::LlmError;
use siumai_core::execution::http::transport::{
    HttpTransport, HttpTransportMultipartRequest, HttpTransportRequest, HttpTransportResponse,
};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct CaptureTransport {
    multipart_requests: Arc<Mutex<Vec<HttpTransportMultipartRequest>>>,
    responses: Arc<Mutex<VecDeque<HttpTransportResponse>>>,
}

impl CaptureTransport {
    fn new(responses: Vec<HttpTransportResponse>) -> Self {
        Self {
            multipart_requests: Arc::new(Mutex::new(Vec::new())),
            responses: Arc::new(Mutex::new(responses.into_iter().collect())),
        }
    }

    fn take_multipart_requests(&self) -> Vec<HttpTransportMultipartRequest> {
        std::mem::take(&mut *self.multipart_requests.lock().expect("multipart lock"))
    }
}

#[async_trait::async_trait]
impl HttpTransport for CaptureTransport {
    async fn execute_json(
        &self,
        _request: HttpTransportRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "json transport should not be used in anthropic files upload tests".to_string(),
        ))
    }

    async fn execute_multipart(
        &self,
        request: HttpTransportMultipartRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        self.multipart_requests
            .lock()
            .expect("multipart lock")
            .push(request);
        self.responses
            .lock()
            .expect("responses lock")
            .pop_front()
            .ok_or_else(|| LlmError::HttpError("missing multipart response".to_string()))
    }
}

fn make_response(body: serde_json::Value) -> HttpTransportResponse {
    HttpTransportResponse {
        status: 200,
        headers: reqwest::header::HeaderMap::new(),
        body: serde_json::to_vec(&body).expect("serialize response body"),
    }
}

#[tokio::test]
async fn anthropic_upload_helper_uses_shared_file_management_lane_and_preserves_warnings() {
    let transport = CaptureTransport::new(vec![make_response(serde_json::json!({
        "id": "file_abc123",
        "type": "file",
        "filename": "test.pdf",
        "mime_type": "application/pdf",
        "size_bytes": 12345,
        "created_at": "2025-04-14T12:00:00Z",
        "downloadable": true
    }))]);

    let client = AnthropicClient::from_config(
        AnthropicConfig::new("test-key")
            .with_base_url("https://api.anthropic.com/v1")
            .with_model("claude-sonnet-4-5")
            .with_http_transport(Arc::new(transport.clone())),
    )
    .expect("build anthropic client");

    let result = files::upload(
        &client,
        b"%PDF".to_vec(),
        UploadFileOptions::new()
            .with_filename("test.pdf")
            .with_metadata("ignored", "value")
            .with_provider_option("anthropic", serde_json::json!({ "custom": true }))
            .with_header("x-test-header", "yes"),
    )
    .await
    .expect("anthropic upload result");

    assert_eq!(
        result.provider_reference.get("anthropic"),
        Some("file_abc123")
    );
    assert_eq!(result.filename.as_deref(), Some("test.pdf"));
    assert_eq!(result.media_type.as_deref(), Some("application/pdf"));
    assert_eq!(
        result
            .provider_metadata
            .as_ref()
            .and_then(|metadata| metadata.get("anthropic"))
            .and_then(|metadata| metadata.get("mimeType")),
        Some(&serde_json::json!("application/pdf"))
    );
    assert_eq!(
        result
            .provider_metadata
            .as_ref()
            .and_then(|metadata| metadata.get("anthropic"))
            .and_then(|metadata| metadata.get("sizeBytes")),
        Some(&serde_json::json!(12345))
    );
    assert_eq!(
        result.warnings,
        vec![
            siumai_core::types::Warning::compatibility(
                "metadata",
                Some("Anthropic file uploads currently ignore UploadFileOptions.metadata."),
            ),
            siumai_core::types::Warning::compatibility(
                "providerOptions",
                Some("Anthropic file uploads currently ignore UploadFileOptions.provider_options."),
            ),
        ]
    );

    let requests = transport.take_multipart_requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].url, "https://api.anthropic.com/v1/files");
    assert_eq!(
        requests[0]
            .headers
            .get("anthropic-beta")
            .and_then(|value| value.to_str().ok()),
        Some("files-api-2025-04-14")
    );
    assert_eq!(
        requests[0]
            .headers
            .get("x-test-header")
            .and_then(|value| value.to_str().ok()),
        Some("yes")
    );

    let body = String::from_utf8_lossy(&requests[0].body);
    assert!(body.contains("name=\"file\"; filename=\"test.pdf\""));
}
