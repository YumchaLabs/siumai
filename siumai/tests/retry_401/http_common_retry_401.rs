use std::sync::{Arc, Mutex, atomic::AtomicUsize};

use siumai::experimental::execution::executors::common::{
    HttpBody, HttpExecutionConfig, execute_delete_request, execute_get_binary, execute_get_request,
    execute_json_request, execute_multipart_request,
};
use siumai::experimental::execution::http::interceptor::{HttpInterceptor, HttpRequestContext};
use siumai::prelude::unified::LlmError;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// Interceptor counting retry attempts
struct CountingInterceptor {
    retries: Arc<Mutex<usize>>,
}
impl HttpInterceptor for CountingInterceptor {
    fn on_retry(&self, _ctx: &HttpRequestContext, _error: &LlmError, _attempt: usize) {
        let mut g = self.retries.lock().unwrap();
        *g += 1;
    }
}

use crate::support;

#[tokio::test]
async fn json_request_retries_on_401() {
    let server = MockServer::start().await;
    // First 401 for bad token
    Mock::given(method("POST"))
        .and(path("/json"))
        .and(header("authorization", "Bearer bad"))
        .respond_with(ResponseTemplate::new(401).set_body_string("Unauthorized"))
        .expect(1)
        .mount(&server)
        .await;
    // Then 200 for ok token
    Mock::given(method("POST"))
        .and(path("/json"))
        .and(header("authorization", "Bearer ok"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": true})))
        .expect(1)
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let spec = Arc::new(support::FlippingAuthSpec {
        counter: Arc::new(AtomicUsize::new(0)),
    });
    let ctx = siumai::experimental::core::ProviderContext::new(
        "test",
        server.uri(),
        None,
        Default::default(),
    );
    let counter = Arc::new(Mutex::new(0usize));
    let interceptor = Arc::new(CountingInterceptor {
        retries: counter.clone(),
    });
    let config = HttpExecutionConfig {
        provider_id: "test".into(),
        http_client: http,
        provider_spec: spec,
        provider_context: ctx,
        interceptors: vec![interceptor],
        retry_options: Some(siumai::retry_api::RetryOptions::default()),
    };
    let url = format!("{}/json", server.uri());
    let body = serde_json::json!({"k":"v"});
    let res = execute_json_request(&config, &url, HttpBody::Json(body), None, false)
        .await
        .unwrap();
    assert_eq!(res.json["ok"], serde_json::json!(true));
    assert_eq!(*counter.lock().unwrap(), 1);
}

#[tokio::test]
async fn multipart_request_retries_on_401_and_rebuilds_form() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/upload"))
        .and(header("authorization", "Bearer bad"))
        .respond_with(ResponseTemplate::new(401))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/upload"))
        .and(header("authorization", "Bearer ok"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": true})))
        .expect(1)
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let spec = Arc::new(support::FlippingAuthSpec {
        counter: Arc::new(AtomicUsize::new(0)),
    });
    let ctx = siumai::experimental::core::ProviderContext::new(
        "test",
        server.uri(),
        None,
        Default::default(),
    );
    let counter = Arc::new(Mutex::new(0usize));
    let interceptor = Arc::new(CountingInterceptor {
        retries: counter.clone(),
    });
    let config = HttpExecutionConfig {
        provider_id: "test".into(),
        http_client: http,
        provider_spec: spec,
        provider_context: ctx,
        interceptors: vec![interceptor],
        retry_options: Some(siumai::retry_api::RetryOptions::default()),
    };
    let url = format!("{}/upload", server.uri());
    let form_rebuilds = Arc::new(Mutex::new(0usize));
    let form_rebuilds_m = form_rebuilds.clone();
    let build_form = move || {
        let mut g = form_rebuilds_m.lock().unwrap();
        *g += 1;
        Ok(reqwest::multipart::Form::new().text("k", "v"))
    };
    let res = execute_multipart_request(&config, &url, build_form, None)
        .await
        .unwrap();
    assert_eq!(res.json["ok"], serde_json::json!(true));
    assert_eq!(*counter.lock().unwrap(), 1);
    assert_eq!(*form_rebuilds.lock().unwrap(), 2); // initial + retry
}

#[tokio::test]
async fn get_request_retries_on_401() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/get"))
        .and(header("authorization", "Bearer bad"))
        .respond_with(ResponseTemplate::new(401))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/get"))
        .and(header("authorization", "Bearer ok"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": true})))
        .expect(1)
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let spec = Arc::new(support::FlippingAuthSpec {
        counter: Arc::new(AtomicUsize::new(0)),
    });
    let ctx = siumai::experimental::core::ProviderContext::new(
        "test",
        server.uri(),
        None,
        Default::default(),
    );
    let counter = Arc::new(Mutex::new(0usize));
    let interceptor = Arc::new(CountingInterceptor {
        retries: counter.clone(),
    });
    let config = HttpExecutionConfig {
        provider_id: "test".into(),
        http_client: http,
        provider_spec: spec,
        provider_context: ctx,
        interceptors: vec![interceptor],
        retry_options: Some(siumai::retry_api::RetryOptions::default()),
    };
    let url = format!("{}/get", server.uri());
    let res = execute_get_request(&config, &url, None).await.unwrap();
    assert_eq!(res.json["ok"], serde_json::json!(true));
    assert_eq!(*counter.lock().unwrap(), 1);
}

#[tokio::test]
async fn delete_request_retries_on_401() {
    let server = MockServer::start().await;
    Mock::given(method("DELETE"))
        .and(path("/del"))
        .and(header("authorization", "Bearer bad"))
        .respond_with(ResponseTemplate::new(401))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/del"))
        .and(header("authorization", "Bearer ok"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": true})))
        .expect(1)
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let spec = Arc::new(support::FlippingAuthSpec {
        counter: Arc::new(AtomicUsize::new(0)),
    });
    let ctx = siumai::experimental::core::ProviderContext::new(
        "test",
        server.uri(),
        None,
        Default::default(),
    );
    let counter = Arc::new(Mutex::new(0usize));
    let interceptor = Arc::new(CountingInterceptor {
        retries: counter.clone(),
    });
    let config = HttpExecutionConfig {
        provider_id: "test".into(),
        http_client: http,
        provider_spec: spec,
        provider_context: ctx,
        interceptors: vec![interceptor],
        retry_options: Some(siumai::retry_api::RetryOptions::default()),
    };
    let url = format!("{}/del", server.uri());
    let res = execute_delete_request(&config, &url, None).await.unwrap();
    assert_eq!(res.json["ok"], serde_json::json!(true));
    assert_eq!(*counter.lock().unwrap(), 1);
}

#[tokio::test]
async fn get_binary_retries_on_401() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/bin"))
        .and(header("authorization", "Bearer bad"))
        .respond_with(ResponseTemplate::new(401))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/bin"))
        .and(header("authorization", "Bearer ok"))
        .respond_with(ResponseTemplate::new(200).set_body_raw("ABC", "application/octet-stream"))
        .expect(1)
        .mount(&server)
        .await;

    let http = reqwest::Client::new();
    let spec = Arc::new(support::FlippingAuthSpec {
        counter: Arc::new(AtomicUsize::new(0)),
    });
    let ctx = siumai::experimental::core::ProviderContext::new(
        "test",
        server.uri(),
        None,
        Default::default(),
    );
    let counter = Arc::new(Mutex::new(0usize));
    let interceptor = Arc::new(CountingInterceptor {
        retries: counter.clone(),
    });
    let config = HttpExecutionConfig {
        provider_id: "test".into(),
        http_client: http,
        provider_spec: spec,
        provider_context: ctx,
        interceptors: vec![interceptor],
        retry_options: Some(siumai::retry_api::RetryOptions::default()),
    };
    let url = format!("{}/bin", server.uri());
    let res = execute_get_binary(&config, &url, None).await.unwrap();
    assert_eq!(res.bytes, b"ABC".to_vec());
    assert_eq!(*counter.lock().unwrap(), 1);
}
