use super::*;
use crate::error::LlmError;
use crate::execution::http::transport::{
    HttpTransport, HttpTransportMultipartRequest, HttpTransportRequest, HttpTransportResponse,
    HttpTransportStreamBody, HttpTransportStreamResponse,
};
use async_trait::async_trait;
use std::collections::VecDeque;
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
struct SequenceTransport {
    json_requests: Arc<Mutex<Vec<HttpTransportRequest>>>,
    multipart_requests: Arc<Mutex<Vec<HttpTransportMultipartRequest>>>,
    stream_requests: Arc<Mutex<Vec<HttpTransportRequest>>>,
    multipart_stream_requests: Arc<Mutex<Vec<HttpTransportMultipartRequest>>>,
    responses: Arc<Mutex<VecDeque<HttpTransportResponse>>>,
    stream_responses: Arc<Mutex<VecDeque<HttpTransportStreamResponse>>>,
}

impl SequenceTransport {
    fn new(responses: Vec<HttpTransportResponse>) -> Self {
        Self {
            json_requests: Arc::new(Mutex::new(Vec::new())),
            multipart_requests: Arc::new(Mutex::new(Vec::new())),
            stream_requests: Arc::new(Mutex::new(Vec::new())),
            multipart_stream_requests: Arc::new(Mutex::new(Vec::new())),
            responses: Arc::new(Mutex::new(responses.into_iter().collect())),
            stream_responses: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    fn new_streaming(responses: Vec<HttpTransportStreamResponse>) -> Self {
        Self {
            json_requests: Arc::new(Mutex::new(Vec::new())),
            multipart_requests: Arc::new(Mutex::new(Vec::new())),
            stream_requests: Arc::new(Mutex::new(Vec::new())),
            multipart_stream_requests: Arc::new(Mutex::new(Vec::new())),
            responses: Arc::new(Mutex::new(VecDeque::new())),
            stream_responses: Arc::new(Mutex::new(responses.into_iter().collect())),
        }
    }

    fn take_requests(&self) -> Vec<HttpTransportRequest> {
        std::mem::take(&mut *self.json_requests.lock().unwrap())
    }

    fn take_multipart_requests(&self) -> Vec<HttpTransportMultipartRequest> {
        std::mem::take(&mut *self.multipart_requests.lock().unwrap())
    }

    fn take_stream_requests(&self) -> Vec<HttpTransportRequest> {
        std::mem::take(&mut *self.stream_requests.lock().unwrap())
    }

    fn take_multipart_stream_requests(&self) -> Vec<HttpTransportMultipartRequest> {
        std::mem::take(&mut *self.multipart_stream_requests.lock().unwrap())
    }
}

#[async_trait]
impl HttpTransport for SequenceTransport {
    async fn execute_json(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        self.json_requests.lock().unwrap().push(request);
        self.responses
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| LlmError::HttpError("missing transport response".into()))
    }

    async fn execute_multipart(
        &self,
        request: HttpTransportMultipartRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        self.multipart_requests.lock().unwrap().push(request);
        self.responses
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| LlmError::HttpError("missing transport response".into()))
    }

    async fn execute_stream(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportStreamResponse, LlmError> {
        self.stream_requests.lock().unwrap().push(request);
        self.stream_responses
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| LlmError::HttpError("missing transport stream response".into()))
    }

    async fn execute_multipart_stream(
        &self,
        request: HttpTransportMultipartRequest,
    ) -> Result<HttpTransportStreamResponse, LlmError> {
        self.multipart_stream_requests.lock().unwrap().push(request);
        self.stream_responses
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| LlmError::HttpError("missing transport stream response".into()))
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

fn build_test_multipart_form() -> Result<reqwest::multipart::Form, LlmError> {
    let file_part = reqwest::multipart::Part::bytes(b"audio-bytes".to_vec())
        .file_name("clip.wav")
        .mime_str("audio/wav")
        .map_err(|e| LlmError::InvalidParameter(e.to_string()))?;

    Ok(reqwest::multipart::Form::new()
        .text("model", "whisper-1")
        .part("file", file_part))
}

fn assert_multipart_request_shape(
    request: &HttpTransportMultipartRequest,
    expected_url: &str,
    expected_retry_attempt: &str,
) {
    assert_eq!(request.url, expected_url);
    assert_eq!(
        request
            .headers
            .get("x-retry-attempt")
            .and_then(|value| value.to_str().ok()),
        Some(expected_retry_attempt),
    );

    let content_type = request
        .headers
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .expect("multipart content-type should exist");
    assert!(
        content_type.starts_with("multipart/form-data; boundary="),
        "unexpected content-type: {content_type}"
    );

    let content_length = request
        .headers
        .get(reqwest::header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<usize>().ok())
        .expect("multipart content-length should exist");
    assert_eq!(content_length, request.body.len());

    let body_text = String::from_utf8_lossy(&request.body);
    assert!(body_text.contains("name=\"model\""));
    assert!(body_text.contains("whisper-1"));
    assert!(body_text.contains("name=\"file\"; filename=\"clip.wav\""));
    assert!(body_text.contains("Content-Type: audio/wav"));
    assert!(body_text.contains("audio-bytes"));
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

#[tokio::test]
async fn json_streaming_response_uses_custom_transport_and_returns_stream_body() {
    let client = reqwest::Client::new();
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::CONTENT_TYPE,
        reqwest::header::HeaderValue::from_static("text/event-stream"),
    );
    let transport = SequenceTransport::new_streaming(vec![HttpTransportStreamResponse {
        status: 200,
        headers,
        body: HttpTransportStreamBody::from_stream(futures_util::stream::iter(vec![
            Ok(b"data: ".to_vec()),
            Ok(b"ok\n\n".to_vec()),
        ])),
    }]);
    let mut config = test_config(
        &client,
        reqwest::header::HeaderMap::new(),
        vec![],
        Some(crate::retry_api::RetryOptions::default()),
    );
    config.transport = Some(Arc::new(transport.clone()));

    let body = serde_json::json!({"stream": true, "model": "demo"});
    let url = "http://unused.invalid/v1/stream";
    let resp = execute_json_request_streaming_response(&config, url, body.clone(), None)
        .await
        .expect("streaming json request should use custom transport");

    assert_eq!(resp.status().as_u16(), 200);
    assert_eq!(
        resp.headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream"),
    );
    assert_eq!(resp.text().await.expect("read body"), "data: ok\n\n");

    let requests = transport.take_stream_requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].url, url);
    assert_eq!(requests[0].body, body);
    assert_eq!(
        requests[0]
            .headers
            .get("x-retry-attempt")
            .and_then(|value| value.to_str().ok()),
        Some("0"),
    );
}

#[tokio::test]
async fn json_streaming_response_custom_transport_retries_401_then_200() {
    let client = reqwest::Client::new();
    let mut ok_headers = reqwest::header::HeaderMap::new();
    ok_headers.insert(
        reqwest::header::CONTENT_TYPE,
        reqwest::header::HeaderValue::from_static("text/event-stream"),
    );
    let transport = SequenceTransport::new_streaming(vec![
        HttpTransportStreamResponse {
            status: 401,
            headers: reqwest::header::HeaderMap::new(),
            body: HttpTransportStreamBody::from_bytes(b"unauthorized".to_vec()),
        },
        HttpTransportStreamResponse {
            status: 200,
            headers: ok_headers,
            body: HttpTransportStreamBody::from_bytes(b"data: after retry\n\n".to_vec()),
        },
    ]);
    let mut config = test_config(
        &client,
        reqwest::header::HeaderMap::new(),
        vec![],
        Some(crate::retry_api::RetryOptions::default()),
    );
    config.transport = Some(Arc::new(transport.clone()));

    let body = serde_json::json!({"stream": true, "model": "demo"});
    let url = "http://unused.invalid/v1/stream";
    let resp = execute_json_request_streaming_response(&config, url, body.clone(), None)
        .await
        .expect("streaming json request should retry on 401 through custom transport");

    assert_eq!(resp.status().as_u16(), 200);
    assert_eq!(
        resp.text().await.expect("read body"),
        "data: after retry\n\n"
    );

    let requests = transport.take_stream_requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].body, body);
    assert_eq!(requests[1].body, body);
    assert_eq!(
        requests[0]
            .headers
            .get("x-retry-attempt")
            .and_then(|value| value.to_str().ok()),
        Some("0"),
    );
    assert_eq!(
        requests[1]
            .headers
            .get("x-retry-attempt")
            .and_then(|value| value.to_str().ok()),
        Some("1"),
    );
}

#[tokio::test]
async fn multipart_request_uses_custom_transport_and_returns_json() {
    let client = reqwest::Client::new();
    let transport = SequenceTransport::new(vec![HttpTransportResponse {
        status: 200,
        headers: reqwest::header::HeaderMap::new(),
        body: br#"{"text":"hello"}"#.to_vec(),
    }]);
    let mut config = test_config(
        &client,
        reqwest::header::HeaderMap::new(),
        vec![],
        Some(crate::retry_api::RetryOptions::default()),
    );
    config.transport = Some(Arc::new(transport.clone()));

    let url = "http://unused.invalid/v1/audio/transcriptions";
    let result = execute_multipart_request(&config, url, build_test_multipart_form, None)
        .await
        .expect("multipart request should use custom transport");

    assert_eq!(result.status, 200);
    assert_eq!(result.json["text"], "hello");

    let requests = transport.take_multipart_requests();
    assert_eq!(requests.len(), 1);
    assert_multipart_request_shape(&requests[0], url, "0");
}

#[tokio::test]
async fn multipart_request_custom_transport_retries_401_then_200() {
    let client = reqwest::Client::new();
    let transport = SequenceTransport::new(vec![
        HttpTransportResponse {
            status: 401,
            headers: reqwest::header::HeaderMap::new(),
            body: b"unauthorized".to_vec(),
        },
        HttpTransportResponse {
            status: 200,
            headers: reqwest::header::HeaderMap::new(),
            body: br#"{"text":"after retry"}"#.to_vec(),
        },
    ]);
    let mut config = test_config(
        &client,
        reqwest::header::HeaderMap::new(),
        vec![],
        Some(crate::retry_api::RetryOptions::default()),
    );
    config.transport = Some(Arc::new(transport.clone()));

    let url = "http://unused.invalid/v1/audio/transcriptions";
    let result = execute_multipart_request(&config, url, build_test_multipart_form, None)
        .await
        .expect("multipart request should retry on 401 through custom transport");

    assert_eq!(result.status, 200);
    assert_eq!(result.json["text"], "after retry");

    let requests = transport.take_multipart_requests();
    assert_eq!(requests.len(), 2);
    assert_multipart_request_shape(&requests[0], url, "0");
    assert_multipart_request_shape(&requests[1], url, "1");
}

#[tokio::test]
async fn multipart_streaming_response_uses_custom_transport_and_returns_body() {
    let client = reqwest::Client::new();
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::CONTENT_TYPE,
        reqwest::header::HeaderValue::from_static("text/event-stream"),
    );
    let transport = SequenceTransport::new_streaming(vec![HttpTransportStreamResponse {
        status: 200,
        headers,
        body: HttpTransportStreamBody::from_bytes(b"data: hello\n\n".to_vec()),
    }]);
    let mut config = test_config(
        &client,
        reqwest::header::HeaderMap::new(),
        vec![],
        Some(crate::retry_api::RetryOptions::default()),
    );
    config.transport = Some(Arc::new(transport.clone()));

    let url = "http://unused.invalid/v1/audio/transcriptions";
    let resp =
        execute_multipart_request_streaming_response(&config, url, build_test_multipart_form, None)
            .await
            .expect("multipart streaming request should use custom transport");

    assert_eq!(resp.status().as_u16(), 200);
    assert_eq!(resp.text().await.expect("read body"), "data: hello\n\n");

    let requests = transport.take_multipart_stream_requests();
    assert_eq!(requests.len(), 1);
    assert_multipart_request_shape(&requests[0], url, "0");
}

#[tokio::test]
async fn multipart_streaming_response_custom_transport_retries_401_then_200() {
    let client = reqwest::Client::new();
    let mut ok_headers = reqwest::header::HeaderMap::new();
    ok_headers.insert(
        reqwest::header::CONTENT_TYPE,
        reqwest::header::HeaderValue::from_static("text/event-stream"),
    );
    let transport = SequenceTransport::new_streaming(vec![
        HttpTransportStreamResponse {
            status: 401,
            headers: reqwest::header::HeaderMap::new(),
            body: HttpTransportStreamBody::from_bytes(b"unauthorized".to_vec()),
        },
        HttpTransportStreamResponse {
            status: 200,
            headers: ok_headers,
            body: HttpTransportStreamBody::from_bytes(b"data: after retry\n\n".to_vec()),
        },
    ]);
    let mut config = test_config(
        &client,
        reqwest::header::HeaderMap::new(),
        vec![],
        Some(crate::retry_api::RetryOptions::default()),
    );
    config.transport = Some(Arc::new(transport.clone()));

    let url = "http://unused.invalid/v1/audio/transcriptions";
    let resp =
        execute_multipart_request_streaming_response(&config, url, build_test_multipart_form, None)
            .await
            .expect("multipart streaming request should retry on 401 through custom transport");

    assert_eq!(resp.status().as_u16(), 200);
    assert_eq!(
        resp.text().await.expect("read body"),
        "data: after retry\n\n"
    );

    let requests = transport.take_multipart_stream_requests();
    assert_eq!(requests.len(), 2);
    assert_multipart_request_shape(&requests[0], url, "0");
    assert_multipart_request_shape(&requests[1], url, "1");
}

#[tokio::test]
async fn multipart_bytes_request_uses_custom_transport_and_returns_body() {
    let client = reqwest::Client::new();
    let transport = SequenceTransport::new(vec![HttpTransportResponse {
        status: 200,
        headers: reqwest::header::HeaderMap::new(),
        body: vec![1, 2, 3, 4],
    }]);
    let mut config = test_config(
        &client,
        reqwest::header::HeaderMap::new(),
        vec![],
        Some(crate::retry_api::RetryOptions::default()),
    );
    config.transport = Some(Arc::new(transport.clone()));

    let url = "http://unused.invalid/v1/audio/transcriptions";
    let result = execute_multipart_bytes_request(&config, url, build_test_multipart_form, None)
        .await
        .expect("multipart bytes request should use custom transport");

    assert_eq!(result.status, 200);
    assert_eq!(result.bytes, vec![1, 2, 3, 4]);

    let requests = transport.take_multipart_requests();
    assert_eq!(requests.len(), 1);
    assert_multipart_request_shape(&requests[0], url, "0");
}

#[tokio::test]
async fn multipart_bytes_request_custom_transport_retries_401_then_200() {
    let client = reqwest::Client::new();
    let transport = SequenceTransport::new(vec![
        HttpTransportResponse {
            status: 401,
            headers: reqwest::header::HeaderMap::new(),
            body: b"unauthorized".to_vec(),
        },
        HttpTransportResponse {
            status: 200,
            headers: reqwest::header::HeaderMap::new(),
            body: vec![9, 8, 7],
        },
    ]);
    let mut config = test_config(
        &client,
        reqwest::header::HeaderMap::new(),
        vec![],
        Some(crate::retry_api::RetryOptions::default()),
    );
    config.transport = Some(Arc::new(transport.clone()));

    let url = "http://unused.invalid/v1/audio/transcriptions";
    let result = execute_multipart_bytes_request(&config, url, build_test_multipart_form, None)
        .await
        .expect("multipart bytes request should retry on 401 through custom transport");

    assert_eq!(result.status, 200);
    assert_eq!(result.bytes, vec![9, 8, 7]);

    let requests = transport.take_multipart_requests();
    assert_eq!(requests.len(), 2);
    assert_multipart_request_shape(&requests[0], url, "0");
    assert_multipart_request_shape(&requests[1], url, "1");
}

#[tokio::test]
async fn bytes_request_uses_custom_transport_and_returns_body() {
    let client = reqwest::Client::new();
    let transport = SequenceTransport::new(vec![HttpTransportResponse {
        status: 200,
        headers: reqwest::header::HeaderMap::new(),
        body: vec![1, 2, 3, 4],
    }]);
    let mut config = test_config(
        &client,
        reqwest::header::HeaderMap::new(),
        vec![],
        Some(crate::retry_api::RetryOptions::default()),
    );
    config.transport = Some(Arc::new(transport.clone()));

    let body = serde_json::json!({"text":"hello","voice_id":"eve"});
    let result = execute_bytes_request(
        &config,
        "http://unused.invalid/v1/tts",
        HttpBody::Json(body.clone()),
        None,
    )
    .await
    .expect("bytes request should use custom transport");

    assert_eq!(result.status, 200);
    assert_eq!(result.bytes, vec![1, 2, 3, 4]);

    let requests = transport.take_requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].url, "http://unused.invalid/v1/tts");
    assert_eq!(requests[0].body, body);
    assert_eq!(
        requests[0]
            .headers
            .get("x-retry-attempt")
            .and_then(|value| value.to_str().ok()),
        Some("0"),
    );
}

#[tokio::test]
async fn bytes_request_custom_transport_retries_401_then_200() {
    let client = reqwest::Client::new();
    let transport = SequenceTransport::new(vec![
        HttpTransportResponse {
            status: 401,
            headers: reqwest::header::HeaderMap::new(),
            body: b"unauthorized".to_vec(),
        },
        HttpTransportResponse {
            status: 200,
            headers: reqwest::header::HeaderMap::new(),
            body: vec![9, 8, 7],
        },
    ]);
    let mut config = test_config(
        &client,
        reqwest::header::HeaderMap::new(),
        vec![],
        Some(crate::retry_api::RetryOptions::default()),
    );
    config.transport = Some(Arc::new(transport.clone()));

    let body = serde_json::json!({"text":"retry me","voice_id":"eve"});
    let result = execute_bytes_request(
        &config,
        "http://unused.invalid/v1/tts",
        HttpBody::Json(body.clone()),
        None,
    )
    .await
    .expect("bytes request should retry on 401 through custom transport");

    assert_eq!(result.status, 200);
    assert_eq!(result.bytes, vec![9, 8, 7]);

    let requests = transport.take_requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].body, body);
    assert_eq!(requests[1].body, body);
    assert_eq!(
        requests[0]
            .headers
            .get("x-retry-attempt")
            .and_then(|value| value.to_str().ok()),
        Some("0"),
    );
    assert_eq!(
        requests[1]
            .headers
            .get("x-retry-attempt")
            .and_then(|value| value.to_str().ok()),
        Some("1"),
    );
}
