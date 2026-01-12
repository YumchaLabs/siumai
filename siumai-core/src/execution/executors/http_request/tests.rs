use super::*;
use crate::error::LlmError;
use std::sync::{Arc, Mutex};

struct FlagInterceptor(Arc<Mutex<bool>>);

impl crate::execution::http::interceptor::HttpInterceptor for FlagInterceptor {
    fn on_error(
        &self,
        _ctx: &crate::execution::http::interceptor::HttpRequestContext,
        _error: &crate::error::LlmError,
    ) {
        *self.0.lock().unwrap() = true;
    }
}

#[derive(Clone)]
struct StaticHeadersSpec {
    headers: reqwest::header::HeaderMap,
}

impl crate::core::ProviderSpec for StaticHeadersSpec {
    fn id(&self) -> &'static str {
        "test"
    }
    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new()
    }
    fn build_headers(
        &self,
        _ctx: &crate::core::ProviderContext,
    ) -> Result<reqwest::header::HeaderMap, LlmError> {
        Ok(self.headers.clone())
    }
    fn chat_url(
        &self,
        _stream: bool,
        _req: &crate::types::ChatRequest,
        _ctx: &crate::core::ProviderContext,
    ) -> String {
        unreachable!()
    }
    fn choose_chat_transformers(
        &self,
        _req: &crate::types::ChatRequest,
        _ctx: &crate::core::ProviderContext,
    ) -> crate::core::ChatTransformers {
        unreachable!()
    }
}

fn test_config(
    client: &reqwest::Client,
    headers: reqwest::header::HeaderMap,
    interceptors: Vec<Arc<dyn crate::execution::http::interceptor::HttpInterceptor>>,
    retry_options: Option<crate::retry_api::RetryOptions>,
) -> HttpExecutionConfig {
    HttpExecutionConfig {
        provider_id: "test".to_string(),
        http_client: client.clone(),
        transport: None,
        provider_spec: Arc::new(StaticHeadersSpec { headers }),
        provider_context: crate::core::ProviderContext::new(
            "test",
            "http://example.invalid",
            None,
            std::collections::HashMap::new(),
        ),
        interceptors,
        retry_options,
    }
}

#[tokio::test]
async fn json_with_headers_propagates_error_and_interceptor() {
    let mut server = mockito::Server::new_async().await;
    let _m = server
        .mock("POST", "/bad")
        .with_status(400)
        .with_body("bad json")
        .create_async()
        .await;

    let url = format!("{}/bad", server.url());
    let client = reqwest::Client::new();
    let body = serde_json::json!({"a":1});
    let flag = Arc::new(Mutex::new(false));
    let it: Arc<dyn crate::execution::http::interceptor::HttpInterceptor> =
        Arc::new(FlagInterceptor(flag.clone()));

    let config = test_config(
        &client,
        reqwest::header::HeaderMap::new(),
        vec![it],
        Some(crate::retry_api::RetryOptions::default()),
    );

    let res = execute_json_request(&config, &url, HttpBody::Json(body), None, false).await;

    assert!(res.is_err(), "expected error for 400 response");
    assert!(*flag.lock().unwrap(), "interceptor not triggered");
}

#[tokio::test]
async fn json_with_headers_retries_401_then_200() {
    let mut server = mockito::Server::new_async().await;
    let _m1 = server
        .mock("POST", "/retry")
        .match_header("x-retry-attempt", "0")
        .with_status(401)
        .with_body("unauthorized")
        .expect(1)
        .create_async()
        .await;

    let _m2 = server
        .mock("POST", "/retry")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body("{\"ok\":true}")
        .create_async()
        .await;

    let url = format!("{}/retry", server.url());
    let client = reqwest::Client::new();
    let body = serde_json::json!({"q":"x"});

    let config = test_config(
        &client,
        reqwest::header::HeaderMap::new(),
        vec![],
        Some(crate::retry_api::RetryOptions::default()),
    );

    let res = execute_json_request(&config, &url, HttpBody::Json(body), None, false)
        .await
        .expect("should succeed after retry");

    assert_eq!(res.status, 200);
    assert_eq!(res.json["ok"], true);
}

#[tokio::test]
async fn json_with_headers_classifies_429_rate_limit() {
    let mut server = mockito::Server::new_async().await;
    let _m = server
        .mock("POST", "/rl")
        .with_status(429)
        .with_header("retry-after", "5")
        .with_body("rate limit")
        .create_async()
        .await;

    let url = format!("{}/rl", server.url());
    let client = reqwest::Client::new();
    let body = serde_json::json!({});
    let config = test_config(
        &client,
        reqwest::header::HeaderMap::new(),
        vec![],
        Some(crate::retry_api::RetryOptions::default()),
    );

    let res = execute_json_request(&config, &url, HttpBody::Json(body), None, false).await;
    match res {
        Err(crate::error::LlmError::RateLimitError(_)) => {}
        other => panic!("expected RateLimitError, got: {:?}", other),
    }
}

#[tokio::test]
async fn json_with_headers_classifies_500_api_error() {
    let mut server = mockito::Server::new_async().await;
    let _m = server
        .mock("POST", "/e500")
        .with_status(500)
        .with_body("server error")
        .create_async()
        .await;

    let url = format!("{}/e500", server.url());
    let client = reqwest::Client::new();
    let body = serde_json::json!({});
    let config = test_config(
        &client,
        reqwest::header::HeaderMap::new(),
        vec![],
        Some(crate::retry_api::RetryOptions::default()),
    );

    let res = execute_json_request(&config, &url, HttpBody::Json(body), None, false).await;
    match res {
        Err(crate::error::LlmError::ApiError { code: 500, .. }) => {}
        other => panic!("expected ApiError(500), got: {:?}", other),
    }
}

#[tokio::test]
async fn multipart_streaming_response_retries_401_then_200() {
    let mut server = mockito::Server::new_async().await;
    let _m1 = server
        .mock("POST", "/mretry")
        .match_header("x-retry-attempt", "0")
        .with_status(401)
        .with_body("unauthorized")
        .expect(1)
        .create_async()
        .await;

    let _m2 = server
        .mock("POST", "/mretry")
        .match_header("x-retry-attempt", "1")
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body("data: ok\n\n")
        .expect(1)
        .create_async()
        .await;

    let url = format!("{}/mretry", server.url());
    let client = reqwest::Client::new();
    let config = test_config(
        &client,
        reqwest::header::HeaderMap::new(),
        vec![],
        Some(crate::retry_api::RetryOptions::default()),
    );

    let resp = execute_multipart_request_streaming_response(
        &config,
        &url,
        || Ok(reqwest::multipart::Form::new().text("a", "b")),
        None,
    )
    .await
    .expect("should succeed after retry");

    assert_eq!(resp.status().as_u16(), 200);
    let text = resp.text().await.expect("read body");
    assert!(text.contains("data: ok"));
}
