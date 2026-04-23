#![cfg(feature = "xai")]

use siumai::files::{self, UploadFileOptions};
use siumai::provider_ext::xai::{XaiConfig, options::XaiFilesOptions};
use siumai_core::client::LlmClient;
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
            "json transport should not be used in xai files upload tests".to_string(),
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
async fn xai_upload_helper_uses_provider_owned_files_lane() {
    let transport = CaptureTransport::new(vec![make_response(serde_json::json!({
        "id": "file-123",
        "bytes": 3,
        "created_at": 1712345678,
        "filename": "hello.txt",
        "status": "uploaded"
    }))]);

    let client = siumai::provider_ext::xai::XaiClient::from_config(
        XaiConfig::new("test-key")
            .with_model("grok-4")
            .with_http_transport(Arc::new(transport.clone())),
    )
    .await
    .expect("build xai client");

    assert!(client.as_file_management_capability().is_some());

    let result = files::upload(
        &client,
        b"hey".to_vec(),
        UploadFileOptions::new()
            .with_filename("hello.txt")
            .with_provider_option(
                "xai",
                serde_json::to_value(
                    XaiFilesOptions::new()
                        .with_team_id("team-123")
                        .with_file_path("/uploads/hello.txt"),
                )
                .expect("serialize xai files options"),
            ),
    )
    .await
    .expect("xai upload result");

    assert_eq!(result.provider_reference.get("xai"), Some("file-123"));
    assert_eq!(result.filename.as_deref(), Some("hello.txt"));
    assert_eq!(result.media_type, None);
    assert!(
        result.provider_metadata.is_none(),
        "upload helper should not synthesize xAI provider metadata from generic shared file fields"
    );

    let requests = transport.take_multipart_requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].url, "https://api.x.ai/v1/files");
    let body = String::from_utf8_lossy(&requests[0].body);
    assert!(body.contains("name=\"team_id\""));
    assert!(body.contains("team-123"));
    assert!(body.contains("name=\"file_path\""));
    assert!(body.contains("/uploads/hello.txt"));
    assert!(body.contains("name=\"file\"; filename=\"hello.txt\""));
}
