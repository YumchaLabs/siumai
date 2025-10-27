#![cfg(feature = "xai")]

use std::sync::Arc;

use reqwest::header::HeaderMap;
use siumai::error::LlmError;
use siumai::execution::http::interceptor::{HttpInterceptor, HttpRequestContext};
use siumai::providers::XaiClient;
use siumai::providers::xai::XaiConfig;
use std::sync::atomic::{AtomicUsize, Ordering};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[derive(Clone, Default)]
struct AddHeaderInterceptor {
    seq: Arc<AtomicUsize>,
}
impl HttpInterceptor for AddHeaderInterceptor {
    fn on_before_send(
        &self,
        _ctx: &HttpRequestContext,
        rb: reqwest::RequestBuilder,
        _body: &serde_json::Value,
        _headers: &HeaderMap,
    ) -> Result<reqwest::RequestBuilder, LlmError> {
        let n = self.seq.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(rb
            .header("x-interceptor", "1")
            .header("x-test-attempt", n.to_string()))
    }
}

#[tokio::test]
async fn xai_deferred_completion_get_retries_401_and_applies_interceptor() {
    // Start mock server
    let server = MockServer::start().await;

    // First attempt: 401 Unauthorized, expect x-retry-attempt=0 and interceptor header
    Mock::given(method("GET"))
        .and(path("/chat/deferred-completion/abc"))
        .and(header("authorization", "Bearer test-key"))
        .and(header("x-test-attempt", "1"))
        .respond_with(ResponseTemplate::new(401).set_body_string("unauthorized"))
        .mount(&server)
        .await;

    // Second attempt: 202 Accepted, expect retry header = 1 and interceptor header still present
    Mock::given(method("GET"))
        .and(path("/chat/deferred-completion/abc"))
        .and(header("authorization", "Bearer test-key"))
        .and(header("x-test-attempt", "2"))
        .respond_with(ResponseTemplate::new(202).set_body_json(serde_json::json!({
            "status": "pending"
        })))
        .mount(&server)
        .await;

    // Build client pointing to mock server base URL
    let cfg = XaiConfig::new("test-key").with_base_url(server.uri());
    let http = reqwest::Client::new();
    let client = XaiClient::with_http_client(cfg, http)
        .await
        .expect("build xai client")
        .with_http_interceptors(vec![Arc::new(AddHeaderInterceptor::default())]);

    // Call get_deferred_completion and expect ApiError with 202 (pending)
    let err = client
        .get_deferred_completion("abc")
        .await
        .expect_err("should return ApiError 202");

    match err {
        LlmError::ApiError { code, .. } => assert_eq!(code, 202, "expected 202 pending"),
        other => panic!("unexpected error: {other:?}"),
    }

    // Verify interceptor header reached server for both attempts
    let reqs = server.received_requests().await.expect("get received reqs");
    assert!(reqs.len() >= 2, "expected two requests (initial + retry)");
    for r in reqs.iter().take(2) {
        let has = r
            .headers
            .iter()
            .any(|(k, v)| k.as_str().eq_ignore_ascii_case("x-interceptor") && v == "1");
        assert!(has, "missing x-interceptor header");
    }
}

#[tokio::test]
async fn xai_deferred_completion_get_returns_unsupported_on_200() {
    // Start mock server
    let server = MockServer::start().await;

    // Single 200 response; assert interceptor header and no retry (x-retry-attempt=0)
    Mock::given(method("GET"))
        .and(path("/chat/deferred-completion/ok"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "status": "done",
            "id": "resp_123"
        })))
        .mount(&server)
        .await;

    // Build client pointing to mock server base URL
    let cfg = XaiConfig::new("test-key").with_base_url(server.uri());
    let http = reqwest::Client::new();
    let client = XaiClient::with_http_client(cfg, http)
        .await
        .expect("build xai client")
        .with_http_interceptors(vec![Arc::new(AddHeaderInterceptor::default())]);

    // Expect UnsupportedOperation (current placeholder behavior)
    let err = client
        .get_deferred_completion("ok")
        .await
        .expect_err("should return UnsupportedOperation for 200 until implemented");

    match err {
        LlmError::UnsupportedOperation(msg) => {
            assert_eq!(msg, "Get deferred completion not implemented yet");
        }
        other => panic!("unexpected error: {other:?}"),
    }

    // Verify interceptor header present
    let reqs = server.received_requests().await.expect("received reqs");
    let r = reqs.first().expect("at least one request");
    let has = r
        .headers
        .iter()
        .any(|(k, v)| k.as_str().eq_ignore_ascii_case("x-interceptor") && v == "1");
    assert!(has, "missing x-interceptor header on 200 path");
}
