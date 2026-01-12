#![cfg(feature = "xai")]

use async_trait::async_trait;
use reqwest::header::HeaderMap;
use siumai::experimental::execution::http::transport::{
    HttpTransport, HttpTransportRequest, HttpTransportResponse, HttpTransportStreamResponse,
};
use siumai::prelude::ChatCapability;
use siumai::prelude::unified::{ChatMessage, ChatRequest, CommonParams, LlmError, Siumai};
use std::path::Path;
use std::sync::Arc;

fn fixture_text(path: impl AsRef<Path>) -> String {
    std::fs::read_to_string(path).expect("read fixture")
}

#[derive(Clone)]
struct ErrorTransport {
    body: Vec<u8>,
}

#[async_trait]
impl HttpTransport for ErrorTransport {
    async fn execute_json(
        &self,
        _request: HttpTransportRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        Ok(HttpTransportResponse {
            status: 401,
            headers: HeaderMap::new(),
            body: self.body.clone(),
        })
    }

    async fn execute_stream(
        &self,
        _request: HttpTransportRequest,
    ) -> Result<HttpTransportStreamResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "error transport only supports json".to_string(),
        ))
    }
}

#[test]
fn xai_error_fixture_preserves_message() {
    let body_text = fixture_text(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("xai")
            .join("errors")
            .join("xai-error.1.json"),
    );

    let transport: Arc<dyn HttpTransport> = Arc::new(ErrorTransport {
        body: body_text.as_bytes().to_vec(),
    });

    let client = futures::executor::block_on(async {
        Siumai::builder()
            .xai()
            .api_key("xai-test")
            .base_url("https://api.x.ai/v1")
            .model("grok-3-latest")
            .fetch(transport)
            .build()
            .await
    })
    .expect("build ok");

    let req = ChatRequest {
        messages: vec![ChatMessage::user("hi").build()],
        common_params: CommonParams {
            model: "grok-3-latest".to_string(),
            ..Default::default()
        },
        stream: false,
        ..Default::default()
    };

    let err = futures::executor::block_on(client.chat_request(req)).expect_err("expected error");
    match err {
        LlmError::AuthenticationError(msg) => assert_eq!(msg, "Invalid API key"),
        other => panic!("unexpected error variant: {other:?}"),
    }
}
