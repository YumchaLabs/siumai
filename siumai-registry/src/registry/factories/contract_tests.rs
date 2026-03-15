//! Factory contract tests (no network).
//!
//! These tests validate shared construction precedence rules across provider factories:
//! - `ctx.http_client` overrides `ctx.http_config` (so invalid config is ignored when client is provided)
//! - `ctx.api_key` overrides env vars (when env fallback exists)
//! - `ctx.base_url` overrides provider defaults (where applicable)

use crate::error::LlmError;
use crate::execution::http::transport::{
    HttpTransport, HttpTransportMultipartRequest, HttpTransportRequest, HttpTransportResponse,
    HttpTransportStreamBody, HttpTransportStreamResponse,
};
use crate::registry::entry::{BuildContext, ProviderFactory};
use crate::test_support::{ENV_LOCK, EnvGuard};
use crate::types::{ChatMessage, ChatRequest, HttpConfig, RerankRequest};
use async_trait::async_trait;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use std::sync::{Arc, Mutex, MutexGuard};

#[allow(dead_code)]
#[derive(Clone, Default)]
struct CaptureTransport {
    last: Arc<Mutex<Option<HttpTransportRequest>>>,
    last_stream: Arc<Mutex<Option<HttpTransportRequest>>>,
}

impl CaptureTransport {
    #[allow(dead_code)]
    fn take(&self) -> Option<HttpTransportRequest> {
        self.last.lock().unwrap().take()
    }

    #[allow(dead_code)]
    fn take_stream(&self) -> Option<HttpTransportRequest> {
        self.last_stream.lock().unwrap().take()
    }
}

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
        self.last.lock().expect("lock success request").take()
    }
}

#[async_trait]
impl HttpTransport for JsonSuccessTransport {
    async fn execute_json(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        *self.last.lock().expect("lock success request") = Some(request);

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        Ok(HttpTransportResponse {
            status: 200,
            headers,
            body: self.response_body.as_ref().clone(),
        })
    }
}

#[derive(Clone)]
struct SseSuccessTransport {
    response_body: Arc<Vec<u8>>,
    last_stream: Arc<Mutex<Option<HttpTransportRequest>>>,
}

impl SseSuccessTransport {
    fn new(response_body: Vec<u8>) -> Self {
        Self {
            response_body: Arc::new(response_body),
            last_stream: Arc::new(Mutex::new(None)),
        }
    }

    fn take_stream(&self) -> Option<HttpTransportRequest> {
        self.last_stream
            .lock()
            .expect("lock sse success stream")
            .take()
    }
}

#[async_trait]
impl HttpTransport for SseSuccessTransport {
    async fn execute_json(
        &self,
        _request: HttpTransportRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        Ok(HttpTransportResponse {
            status: 501,
            headers,
            body: br#"{"error":{"message":"json unsupported in test","type":"test_error","code":"unsupported"}}"#
                .to_vec(),
        })
    }

    async fn execute_stream(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportStreamResponse, LlmError> {
        *self.last_stream.lock().expect("lock sse success stream") = Some(request);

        let mut headers = HeaderMap::new();
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream; charset=utf-8"),
        );

        Ok(HttpTransportStreamResponse {
            status: 200,
            headers,
            body: HttpTransportStreamBody::from_bytes(self.response_body.as_ref().clone()),
        })
    }
}

#[derive(Clone)]
struct MultipartJsonSuccessTransport {
    response_body: Arc<Vec<u8>>,
    last_multipart: Arc<Mutex<Option<HttpTransportMultipartRequest>>>,
}

impl MultipartJsonSuccessTransport {
    fn new(response: serde_json::Value) -> Self {
        Self {
            response_body: Arc::new(serde_json::to_vec(&response).expect("response json")),
            last_multipart: Arc::new(Mutex::new(None)),
        }
    }

    fn take_multipart(&self) -> Option<HttpTransportMultipartRequest> {
        self.last_multipart
            .lock()
            .expect("lock multipart success request")
            .take()
    }
}

#[derive(Clone)]
struct BytesSuccessTransport {
    response_body: Arc<Vec<u8>>,
    response_content_type: &'static str,
    last: Arc<Mutex<Option<HttpTransportRequest>>>,
}

impl BytesSuccessTransport {
    fn new(response_body: Vec<u8>, response_content_type: &'static str) -> Self {
        Self {
            response_body: Arc::new(response_body),
            response_content_type,
            last: Arc::new(Mutex::new(None)),
        }
    }

    fn take(&self) -> Option<HttpTransportRequest> {
        self.last.lock().expect("lock bytes success request").take()
    }
}

#[async_trait]
impl HttpTransport for BytesSuccessTransport {
    async fn execute_json(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        *self.last.lock().expect("lock bytes success request") = Some(request);

        let mut headers = HeaderMap::new();
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static(self.response_content_type),
        );

        Ok(HttpTransportResponse {
            status: 200,
            headers,
            body: self.response_body.as_ref().clone(),
        })
    }
}

#[async_trait]
impl HttpTransport for MultipartJsonSuccessTransport {
    async fn execute_json(
        &self,
        _request: HttpTransportRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        Ok(HttpTransportResponse {
            status: 501,
            headers,
            body: br#"{"error":{"message":"json unsupported in test","type":"test_error","code":"unsupported"}}"#
                .to_vec(),
        })
    }

    async fn execute_multipart(
        &self,
        request: HttpTransportMultipartRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        *self
            .last_multipart
            .lock()
            .expect("lock multipart success request") = Some(request);

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        Ok(HttpTransportResponse {
            status: 200,
            headers,
            body: self.response_body.as_ref().clone(),
        })
    }
}

#[async_trait]
impl HttpTransport for CaptureTransport {
    async fn execute_json(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        *self.last.lock().unwrap() = Some(request);

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

    async fn execute_stream(
        &self,
        request: HttpTransportRequest,
    ) -> Result<crate::execution::http::transport::HttpTransportStreamResponse, LlmError> {
        *self.last_stream.lock().unwrap() = Some(request);
        Err(LlmError::UnsupportedOperation(
            "capture transport stores streaming requests only".to_string(),
        ))
    }
}

fn lock_env() -> MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[allow(dead_code)]
fn make_chat_request() -> ChatRequest {
    ChatRequest::new(vec![ChatMessage::user("hi").build()])
}

#[allow(dead_code)]
fn make_chat_request_with_model(model: &str) -> ChatRequest {
    let mut req = make_chat_request();
    req.common_params.model = model.to_string();
    req
}

#[allow(dead_code)]
fn assert_embedding_image_rerank_capabilities_absent(caps: &crate::traits::ProviderCapabilities) {
    assert!(!caps.supports("embedding"));
    assert!(!caps.supports("image_generation"));
    assert!(!caps.supports("rerank"));
}

#[allow(dead_code)]
fn assert_no_deferred_capability_leaks(client: &dyn crate::client::LlmClient) {
    assert!(client.as_embedding_capability().is_none());
    assert!(client.as_image_generation_capability().is_none());
    assert!(client.as_rerank_capability().is_none());
}

fn assert_capture_transport_unused(transport: &CaptureTransport) {
    assert!(transport.take().is_none());
    assert!(transport.take_stream().is_none());
}

fn assert_unsupported_operation_contains<T>(result: Result<T, LlmError>, expected: &str) {
    match result {
        Ok(_) => panic!("expected UnsupportedOperation containing `{expected}`"),
        Err(LlmError::UnsupportedOperation(message)) => {
            assert!(
                message.contains(expected),
                "expected `{message}` to contain `{expected}`"
            );
        }
        Err(other) => panic!("expected UnsupportedOperation, got: {other:?}"),
    }
}

#[allow(dead_code)]
fn make_rerank_request(model: &str) -> RerankRequest {
    RerankRequest::new(
        model.to_string(),
        "query".to_string(),
        vec!["doc-1".to_string(), "doc-2".to_string()],
    )
}

#[allow(dead_code)]
fn header_value(req: &HttpTransportRequest, key: &str) -> Option<String> {
    req.headers
        .get(key)
        .and_then(|v| v.to_str().ok())
        .map(ToString::to_string)
}

#[allow(dead_code)]
fn assert_requests_equivalent(left: &HttpTransportRequest, right: &HttpTransportRequest) {
    assert_eq!(left.url, right.url);
    assert_eq!(
        header_value(left, "authorization"),
        header_value(right, "authorization")
    );
    assert_eq!(header_value(left, "accept"), header_value(right, "accept"));
    assert_eq!(left.body, right.body);
}

#[cfg(feature = "azure")]
mod azure_contract {
    use super::*;
    use crate::traits::ChatCapability;

    fn azure_responses_text_stream_body() -> Vec<u8> {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("siumai")
            .join("tests")
            .join("fixtures")
            .join("openai")
            .join("responses-stream")
            .join("text")
            .join("openai-text-deltas.1.chunks.txt");
        let raw = std::fs::read_to_string(&path).unwrap_or_else(|err| {
            panic!("read azure contract stream fixture failed: {path:?}: {err}")
        });

        let mut sse = String::new();
        for line in raw.lines().filter(|line| !line.trim().is_empty()) {
            sse.push_str("data: ");
            sse.push_str(line);
            sse.push_str("\n\n");
        }
        sse.push_str("data: [DONE]\n\n");
        sse.into_bytes()
    }

    fn azure_responses_response_body() -> serde_json::Value {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("siumai")
            .join("tests")
            .join("fixtures")
            .join("openai")
            .join("responses")
            .join("response")
            .join("basic-text")
            .join("response.json");
        let raw = std::fs::read_to_string(&path).unwrap_or_else(|err| {
            panic!("read azure contract response fixture failed: {path:?}: {err}")
        });
        serde_json::from_str(&raw).unwrap_or_else(|err| {
            panic!("parse azure contract response fixture failed: {path:?}: {err}")
        })
    }

    #[tokio::test]
    async fn azure_factory_prefers_ctx_http_client_over_http_config() {
        let _lock = lock_env();

        let factory = crate::registry::factories::AzureOpenAiProviderFactory::default();

        let mut bad = HttpConfig::default();
        bad.proxy = Some("not-a-url".to_string());

        let ctx = BuildContext {
            provider_id: Some("azure".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/openai".to_string()),
            http_client: Some(reqwest::Client::new()),
            http_config: Some(bad),
            ..Default::default()
        };

        factory
            .language_model_with_ctx("test-deployment", &ctx)
            .await
            .expect("factory should prefer ctx.http_client over invalid http_config");
    }

    #[tokio::test]
    async fn azure_factory_uses_env_api_key_when_ctx_missing() {
        let _lock = lock_env();

        let _k = EnvGuard::set("AZURE_API_KEY", "env-key");
        let _r = EnvGuard::set("AZURE_RESOURCE_NAME", "my-azure-resource");

        let factory = crate::registry::factories::AzureOpenAiProviderFactory::default();
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("azure".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("test-deployment", &ctx)
            .await
            .expect("build client via env api key");

        let _ = client.chat_request(make_chat_request()).await;
        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get("api-key").unwrap(), "env-key");
        assert!(
            req.url
                .starts_with("https://my-azure-resource.openai.azure.com/openai/"),
            "unexpected url: {}",
            req.url
        );
    }

    #[tokio::test]
    async fn azure_factory_prefers_ctx_api_key_over_env() {
        let _lock = lock_env();

        let _k = EnvGuard::set("AZURE_API_KEY", "env-key");

        let factory = crate::registry::factories::AzureOpenAiProviderFactory::default();
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("azure".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom/openai".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("test-deployment", &ctx)
            .await
            .expect("build client via ctx api key");

        let _ = client.chat_request(make_chat_request()).await;
        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get("api-key").unwrap(), "ctx-key");
        assert!(req.url.starts_with("https://example.com/custom/openai/"));
    }

    #[tokio::test]
    async fn azure_factory_prefers_ctx_base_url_over_resource_env() {
        let _lock = lock_env();

        let _k = EnvGuard::set("AZURE_API_KEY", "env-key");
        let _r = EnvGuard::set("AZURE_RESOURCE_NAME", "my-azure-resource");

        let factory = crate::registry::factories::AzureOpenAiProviderFactory::default();
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("azure".to_string()),
            base_url: Some("https://example.com/override/openai".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("test-deployment", &ctx)
            .await
            .expect("build client via base_url override");

        let _ = client.chat_request(make_chat_request()).await;
        let req = transport.take().expect("captured request");
        assert!(req.url.starts_with("https://example.com/override/openai/"));
    }

    #[tokio::test]
    async fn azure_factory_supports_native_text_family_path() {
        let _lock = lock_env();

        let factory = crate::registry::factories::AzureOpenAiProviderFactory::default();
        let ctx = BuildContext {
            provider_id: Some("azure".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom/openai".to_string()),
            ..Default::default()
        };

        let model = factory
            .language_model_text_with_ctx("gpt-4o", &ctx)
            .await
            .expect("build native text-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "azure"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "gpt-4o"
        );
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model.as_ref()),
            crate::traits::ModelSpecVersion::V1
        );
    }

    #[tokio::test]
    async fn azure_factory_supports_native_embedding_family_path() {
        let _lock = lock_env();

        let factory = crate::registry::factories::AzureOpenAiProviderFactory::default();
        let ctx = BuildContext {
            provider_id: Some("azure".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom/openai".to_string()),
            ..Default::default()
        };

        let model = factory
            .embedding_model_family_with_ctx("text-embedding-3-small", &ctx)
            .await
            .expect("build native embedding-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "azure"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "text-embedding-3-small"
        );
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model.as_ref()),
            crate::traits::ModelSpecVersion::V1
        );
    }

    #[tokio::test]
    async fn azure_factory_supports_native_image_family_path() {
        let _lock = lock_env();

        let factory = crate::registry::factories::AzureOpenAiProviderFactory::default();
        let ctx = BuildContext {
            provider_id: Some("azure".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom/openai".to_string()),
            ..Default::default()
        };

        let model = factory
            .image_model_family_with_ctx("gpt-image-1", &ctx)
            .await
            .expect("build native image-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "azure"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "gpt-image-1"
        );
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model.as_ref()),
            crate::traits::ModelSpecVersion::V1
        );
    }

    #[tokio::test]
    async fn azure_factory_supports_native_speech_family_path() {
        let _lock = lock_env();

        let factory = crate::registry::factories::AzureOpenAiProviderFactory::default();
        let ctx = BuildContext {
            provider_id: Some("azure".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom/openai".to_string()),
            ..Default::default()
        };

        let model = factory
            .speech_model_family_with_ctx("gpt-4o-mini-tts", &ctx)
            .await
            .expect("build native speech-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "azure"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "gpt-4o-mini-tts"
        );
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model.as_ref()),
            crate::traits::ModelSpecVersion::V1
        );
    }

    #[tokio::test]
    async fn azure_factory_supports_native_transcription_family_path() {
        let _lock = lock_env();

        let factory = crate::registry::factories::AzureOpenAiProviderFactory::default();
        let ctx = BuildContext {
            provider_id: Some("azure".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom/openai".to_string()),
            ..Default::default()
        };

        let model = factory
            .transcription_model_family_with_ctx("gpt-4o-mini-transcribe", &ctx)
            .await
            .expect("build native transcription-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "azure"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "gpt-4o-mini-transcribe"
        );
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model.as_ref()),
            crate::traits::ModelSpecVersion::V1
        );
    }

    #[test]
    fn azure_factory_source_declares_native_audio_family_overrides() {
        let source = include_str!("azure.rs");

        assert!(source.contains("async fn speech_model_family_with_ctx("));
        assert!(source.contains("async fn transcription_model_family_with_ctx("));
    }

    #[test]
    fn azure_factory_source_routes_construction_through_provider_owned_builder() {
        let source = include_str!("azure.rs");

        assert!(source.contains("AzureOpenAiBuilder::new("));
        assert!(source.contains(".with_http_config("));
    }

    #[tokio::test]
    async fn azure_builder_config_registry_chat_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let model = "deployment-id";
        let base_url = "https://example.invalid/openai";

        let builder_client =
            siumai_provider_azure::providers::azure_openai::AzureOpenAiBuilder::new(
                siumai_provider_azure::builder::BuilderBase::default(),
            )
            .api_key("test-key")
            .base_url(base_url)
            .model(model)
            .fetch(Arc::new(builder_transport.clone()))
            .build()
            .expect("build builder client");

        let config_client =
            siumai_provider_azure::providers::azure_openai::AzureOpenAiClient::from_config(
                siumai_provider_azure::providers::azure_openai::AzureOpenAiConfig::new("test-key")
                    .with_base_url(base_url)
                    .with_model(model)
                    .with_http_transport(Arc::new(config_transport.clone())),
            )
            .expect("build config client");

        let factory = crate::registry::factories::AzureOpenAiProviderFactory::default();
        let registry_client = factory
            .language_model_with_ctx(
                model,
                &BuildContext {
                    provider_id: Some("azure".to_string()),
                    api_key: Some("test-key".to_string()),
                    base_url: Some(base_url.to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model(model);

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(
            header_value(&builder_req, "api-key"),
            Some("test-key".to_string())
        );
        assert_eq!(
            header_value(&builder_req, "api-key"),
            header_value(&registry_req, "api-key")
        );
        assert_eq!(
            builder_req.url,
            "https://example.invalid/openai/v1/responses?api-version=v1"
        );
    }

    #[tokio::test]
    async fn azure_builder_config_registry_chat_request_respect_explicit_request_model() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let default_model = "deployment-id";
        let request_model = "override-deployment-id";
        let base_url = "https://example.invalid/openai";

        let builder_client =
            siumai_provider_azure::providers::azure_openai::AzureOpenAiBuilder::new(
                siumai_provider_azure::builder::BuilderBase::default(),
            )
            .api_key("test-key")
            .base_url(base_url)
            .model(default_model)
            .fetch(Arc::new(builder_transport.clone()))
            .build()
            .expect("build builder client");

        let config_client =
            siumai_provider_azure::providers::azure_openai::AzureOpenAiClient::from_config(
                siumai_provider_azure::providers::azure_openai::AzureOpenAiConfig::new("test-key")
                    .with_base_url(base_url)
                    .with_model(default_model)
                    .with_http_transport(Arc::new(config_transport.clone())),
            )
            .expect("build config client");

        let factory = crate::registry::factories::AzureOpenAiProviderFactory::default();
        let registry_client = factory
            .language_model_with_ctx(
                default_model,
                &BuildContext {
                    provider_id: Some("azure".to_string()),
                    api_key: Some("test-key".to_string()),
                    base_url: Some(base_url.to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model(request_model);

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.body["model"], serde_json::json!(request_model));
    }

    #[tokio::test]
    async fn azure_builder_config_registry_chat_stream_request_respect_explicit_request_model() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let default_model = "deployment-id";
        let request_model = "override-deployment-id";
        let base_url = "https://example.invalid/openai";

        let builder_client =
            siumai_provider_azure::providers::azure_openai::AzureOpenAiBuilder::new(
                siumai_provider_azure::builder::BuilderBase::default(),
            )
            .api_key("test-key")
            .base_url(base_url)
            .model(default_model)
            .fetch(Arc::new(builder_transport.clone()))
            .build()
            .expect("build builder client");

        let config_client =
            siumai_provider_azure::providers::azure_openai::AzureOpenAiClient::from_config(
                siumai_provider_azure::providers::azure_openai::AzureOpenAiConfig::new("test-key")
                    .with_base_url(base_url)
                    .with_model(default_model)
                    .with_http_transport(Arc::new(config_transport.clone())),
            )
            .expect("build config client");

        let factory = crate::registry::factories::AzureOpenAiProviderFactory::default();
        let registry_client = factory
            .language_model_with_ctx(
                default_model,
                &BuildContext {
                    provider_id: Some("azure".to_string()),
                    api_key: Some("test-key".to_string()),
                    base_url: Some(base_url.to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model(request_model);

        let _ = builder_client.chat_stream_request(request.clone()).await;
        let _ = config_client.chat_stream_request(request.clone()).await;
        let _ = registry_client.chat_stream_request(request).await;

        let builder_req = builder_transport
            .take_stream()
            .expect("builder stream request");
        let config_req = config_transport
            .take_stream()
            .expect("config stream request");
        let registry_req = registry_transport
            .take_stream()
            .expect("registry stream request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.body["model"], serde_json::json!(request_model));
        assert_eq!(builder_req.body["stream"], serde_json::json!(true));
    }

    #[tokio::test]
    async fn azure_chat_builder_config_registry_chat_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let model = "deployment-id";
        let base_url = "https://example.invalid/openai";

        let builder_client =
            siumai_provider_azure::providers::azure_openai::AzureOpenAiBuilder::new(
                siumai_provider_azure::builder::BuilderBase::default(),
            )
            .api_key("test-key")
            .base_url(base_url)
            .model(model)
            .chat_completions()
            .fetch(Arc::new(builder_transport.clone()))
            .build()
            .expect("build builder client");

        let config_client = siumai_provider_azure::providers::azure_openai::AzureOpenAiClient::from_config(
            siumai_provider_azure::providers::azure_openai::AzureOpenAiConfig::new("test-key")
                .with_base_url(base_url)
                .with_model(model)
                .with_chat_mode(
                    siumai_provider_azure::providers::azure_openai::AzureChatMode::ChatCompletions,
                )
                .with_http_transport(Arc::new(config_transport.clone())),
        )
        .expect("build config client");

        let factory = crate::registry::factories::AzureOpenAiProviderFactory::new(
            siumai_provider_azure::providers::azure_openai::AzureChatMode::ChatCompletions,
        );
        let registry_client = factory
            .language_model_with_ctx(
                model,
                &BuildContext {
                    provider_id: Some("azure".to_string()),
                    api_key: Some("test-key".to_string()),
                    base_url: Some(base_url.to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model(model);

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(
            header_value(&builder_req, "api-key"),
            Some("test-key".to_string())
        );
        assert_eq!(
            header_value(&builder_req, "api-key"),
            header_value(&registry_req, "api-key")
        );
        assert_eq!(
            builder_req.url,
            "https://example.invalid/openai/v1/chat/completions?api-version=v1"
        );
    }

    #[tokio::test]
    async fn azure_registry_override_stream_end_metadata_preserves_vendor_namespace() {
        let _lock = lock_env();

        let stream_body = azure_responses_text_stream_body();

        let global_transport = CaptureTransport::default();
        let azure_transport = SseSuccessTransport::new(stream_body);
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "azure".to_string(),
            Arc::new(crate::registry::factories::AzureOpenAiProviderFactory::default())
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/openai")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "azure",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.invalid/openai")
                    .fetch(Arc::new(azure_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let mut stream = registry
            .language_model("azure:deployment-id")
            .expect("build azure handle")
            .chat_stream_request(make_chat_request_with_model("deployment-id"))
            .await
            .expect("registry stream ok");

        use futures::StreamExt;

        let mut stream_end = None;
        while let Some(event) = stream.next().await {
            if let Ok(crate::streaming::ChatStreamEvent::StreamEnd { response }) = event {
                stream_end = Some(response);
                break;
            }
        }

        let response = stream_end.expect("registry stream end");
        assert_eq!(response.content_text(), Some("answer text"));

        let provider_metadata = response
            .provider_metadata
            .as_ref()
            .expect("registry provider metadata");
        assert!(provider_metadata.get("azure").is_some());
        assert!(provider_metadata.get("openai").is_none());

        let req = azure_transport
            .take_stream()
            .expect("captured azure stream request");
        assert!(global_transport.take().is_none());
        assert!(global_transport.take_stream().is_none());
        assert_eq!(header_value(&req, "api-key"), Some("ctx-key".to_string()));
        assert_eq!(
            header_value(&req, "accept"),
            Some("text/event-stream".to_string())
        );
        assert_eq!(
            req.url,
            "https://example.invalid/openai/v1/responses?api-version=v1"
        );
    }

    #[tokio::test]
    async fn azure_registry_override_chat_response_metadata_preserves_vendor_namespace() {
        let _lock = lock_env();

        let response_json = azure_responses_response_body();

        let global_transport = CaptureTransport::default();
        let azure_transport = JsonSuccessTransport::new(response_json);
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "azure".to_string(),
            Arc::new(crate::registry::factories::AzureOpenAiProviderFactory::default())
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/openai")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "azure",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.invalid/openai")
                    .fetch(Arc::new(azure_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let response = registry
            .language_model("azure:deployment-id")
            .expect("build azure handle")
            .chat_request(make_chat_request_with_model("deployment-id"))
            .await
            .expect("registry response ok");

        assert_eq!(response.content_text(), Some("answer text"));

        let provider_metadata = response
            .provider_metadata
            .as_ref()
            .expect("registry provider metadata");
        assert!(provider_metadata.get("azure").is_some());
        assert!(provider_metadata.get("openai").is_none());

        let req = azure_transport.take().expect("captured azure request");
        assert!(global_transport.take().is_none());
        assert_eq!(header_value(&req, "api-key"), Some("ctx-key".to_string()));
        assert_eq!(
            req.url,
            "https://example.invalid/openai/v1/responses?api-version=v1"
        );
    }
}

#[cfg(feature = "cohere")]
mod cohere_contract {
    use super::*;
    use crate::traits::RerankCapability;
    use reqwest::header::AUTHORIZATION;
    use siumai_provider_cohere::provider_options::CohereRerankOptions;
    use siumai_provider_cohere::providers::cohere::CohereRerankRequestExt;

    #[tokio::test]
    async fn cohere_factory_keeps_non_rerank_capabilities_deferred() {
        let _lock = lock_env();

        let factory = crate::registry::factories::CohereProviderFactory;
        let caps = factory.capabilities();

        assert!(caps.supports("rerank"));
        assert!(!caps.supports("chat"));
        assert!(!caps.supports("embedding"));
        assert!(!caps.supports("image_generation"));
        assert!(!caps.supports("speech"));
        assert!(!caps.supports("transcription"));
        assert!(!caps.supports("audio"));
    }

    #[tokio::test]
    async fn cohere_factory_prefers_ctx_http_client_over_http_config() {
        let _lock = lock_env();

        let factory = crate::registry::factories::CohereProviderFactory;

        let mut bad = HttpConfig::default();
        bad.proxy = Some("not-a-url".to_string());

        let ctx = BuildContext {
            provider_id: Some("cohere".to_string()),
            api_key: Some("ctx-key".to_string()),
            http_client: Some(reqwest::Client::new()),
            http_config: Some(bad),
            ..Default::default()
        };

        factory
            .reranking_model_with_ctx("rerank-english-v3.0", &ctx)
            .await
            .expect("factory should prefer ctx.http_client over invalid http_config");
    }

    #[tokio::test]
    async fn cohere_factory_uses_env_api_key_when_ctx_missing() {
        let _lock = lock_env();

        let _k = EnvGuard::set("COHERE_API_KEY", "env-key");

        let factory = crate::registry::factories::CohereProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("cohere".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .reranking_model_with_ctx("rerank-english-v3.0", &ctx)
            .await
            .expect("build client via env api key");

        let rerank = client
            .as_rerank_capability()
            .expect("cohere client should expose rerank capability");
        let _ = rerank
            .rerank(make_rerank_request("rerank-english-v3.0"))
            .await;

        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer env-key");
        assert_eq!(req.url, "https://api.cohere.com/v2/rerank");
    }

    #[tokio::test]
    async fn cohere_factory_prefers_ctx_api_key_over_env() {
        let _lock = lock_env();

        let _k = EnvGuard::set("COHERE_API_KEY", "env-key");

        let factory = crate::registry::factories::CohereProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("cohere".to_string()),
            api_key: Some("ctx-key".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .reranking_model_with_ctx("rerank-english-v3.0", &ctx)
            .await
            .expect("build client via ctx api key");

        let rerank = client
            .as_rerank_capability()
            .expect("cohere client should expose rerank capability");
        let _ = rerank
            .rerank(make_rerank_request("rerank-english-v3.0"))
            .await;

        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
    }

    #[tokio::test]
    async fn cohere_factory_prefers_ctx_base_url_over_default() {
        let _lock = lock_env();

        let _k = EnvGuard::set("COHERE_API_KEY", "env-key");

        let factory = crate::registry::factories::CohereProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("cohere".to_string()),
            base_url: Some("https://example.com/cohere".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .reranking_model_with_ctx("rerank-english-v3.0", &ctx)
            .await
            .expect("build client via base_url override");

        let rerank = client
            .as_rerank_capability()
            .expect("cohere client should expose rerank capability");
        let _ = rerank
            .rerank(make_rerank_request("rerank-english-v3.0"))
            .await;

        let req = transport.take().expect("captured request");
        assert!(req.url.starts_with("https://example.com/cohere/"));
        assert!(req.url.ends_with("/rerank"));
    }

    #[tokio::test]
    async fn cohere_factory_materializes_provider_owned_typed_client() {
        let _lock = lock_env();

        let factory = crate::registry::factories::CohereProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("cohere".to_string()),
            api_key: Some("ctx-key".to_string()),
            ..Default::default()
        };

        let client = factory
            .reranking_model_with_ctx("rerank-english-v3.0", &ctx)
            .await
            .expect("build cohere typed client");

        assert!(
            client
                .as_any()
                .is::<siumai_provider_cohere::providers::cohere::CohereClient>(),
            "expected provider-owned CohereClient"
        );
    }

    #[tokio::test]
    async fn cohere_factory_rejects_language_model_path_without_chat() {
        let _lock = lock_env();

        let factory = crate::registry::factories::CohereProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("cohere".to_string()),
            api_key: Some("ctx-key".to_string()),
            ..Default::default()
        };

        assert_unsupported_operation_contains(
            factory
                .language_model_with_ctx("rerank-english-v3.0", &ctx)
                .await,
            "language_model/chat family path",
        );
    }

    #[tokio::test]
    async fn cohere_registry_rejects_language_model_handle_without_chat() {
        let _lock = lock_env();

        let transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "cohere".to_string(),
            Arc::new(crate::registry::factories::CohereProviderFactory) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("ctx-key")
            .with_base_url("https://example.com/cohere")
            .fetch(Arc::new(transport.clone()))
            .auto_middleware(false)
            .build()
            .expect("build registry");

        assert_unsupported_operation_contains(
            registry.language_model("cohere:rerank-english-v3.0"),
            "family-specific entries",
        );
        assert_capture_transport_unused(&transport);
    }

    #[tokio::test]
    async fn cohere_builder_config_registry_rerank_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_cohere::providers::cohere::CohereBuilder::new(
            siumai_provider_cohere::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/cohere")
        .reranking_model("rerank-english-v3.0")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .expect("build builder client");

        let config_client = siumai_provider_cohere::providers::cohere::CohereClient::from_config(
            siumai_provider_cohere::providers::cohere::CohereConfig::new("ctx-key")
                .with_base_url("https://example.com/cohere")
                .with_model("rerank-english-v3.0")
                .with_http_transport(Arc::new(config_transport.clone())),
        )
        .expect("build config client");

        let factory = crate::registry::factories::CohereProviderFactory;
        let registry_client = factory
            .reranking_model_with_ctx(
                "rerank-english-v3.0",
                &BuildContext {
                    provider_id: Some("cohere".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/cohere".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_rerank_request("rerank-english-v3.0")
            .with_top_n(1)
            .with_cohere_options(
                CohereRerankOptions::new()
                    .with_max_tokens_per_doc(1000)
                    .with_priority(1),
            );

        let _ = builder_client.rerank(request.clone()).await;
        let _ = config_client.rerank(request.clone()).await;
        let _ = registry_client
            .as_rerank_capability()
            .expect("registry rerank capability")
            .rerank(request)
            .await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(
            builder_req.headers.get(AUTHORIZATION).unwrap(),
            "Bearer ctx-key"
        );
        assert_eq!(builder_req.url, "https://example.com/cohere/rerank");
        assert_eq!(
            builder_req.body["model"],
            serde_json::json!("rerank-english-v3.0")
        );
        assert_eq!(builder_req.body["query"], serde_json::json!("query"));
        assert_eq!(
            builder_req.body["documents"],
            serde_json::json!(["doc-1", "doc-2"])
        );
        assert_eq!(builder_req.body["top_n"], serde_json::json!(1));
        assert_eq!(
            builder_req.body["max_tokens_per_doc"],
            serde_json::json!(1000)
        );
        assert_eq!(builder_req.body["priority"], serde_json::json!(1));
    }

    #[tokio::test]
    async fn cohere_registry_rerank_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let cohere_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "cohere".to_string(),
            Arc::new(crate::registry::factories::CohereProviderFactory) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "cohere",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/cohere")
                    .fetch(Arc::new(cohere_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .reranking_model("cohere:rerank-english-v3.0")
            .expect("build cohere rerank handle");

        let _ = handle
            .rerank(
                make_rerank_request("rerank-english-v3.0")
                    .with_top_n(1)
                    .with_cohere_options(
                        CohereRerankOptions::new()
                            .with_max_tokens_per_doc(1000)
                            .with_priority(1),
                    ),
            )
            .await;

        let req = cohere_transport.take().expect("captured cohere request");
        assert!(global_transport.take().is_none());
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert_eq!(req.url, "https://example.com/cohere/rerank");
        assert_eq!(req.body["top_n"], serde_json::json!(1));
        assert_eq!(req.body["max_tokens_per_doc"], serde_json::json!(1000));
        assert_eq!(req.body["priority"], serde_json::json!(1));
    }
}

#[cfg(feature = "togetherai")]
mod togetherai_contract {
    use super::*;
    use crate::traits::RerankCapability;
    use reqwest::header::AUTHORIZATION;
    use siumai_provider_togetherai::provider_options::TogetherAiRerankOptions;
    use siumai_provider_togetherai::providers::togetherai::TogetherAiRerankRequestExt;

    #[tokio::test]
    async fn togetherai_factory_keeps_non_rerank_capabilities_deferred() {
        let _lock = lock_env();

        let factory = crate::registry::factories::TogetherAiProviderFactory;
        let caps = factory.capabilities();

        assert!(caps.supports("rerank"));
        assert!(!caps.supports("chat"));
        assert!(!caps.supports("embedding"));
        assert!(!caps.supports("image_generation"));
        assert!(!caps.supports("speech"));
        assert!(!caps.supports("transcription"));
        assert!(!caps.supports("audio"));
    }

    #[tokio::test]
    async fn togetherai_factory_prefers_ctx_http_client_over_http_config() {
        let _lock = lock_env();

        let factory = crate::registry::factories::TogetherAiProviderFactory;

        let mut bad = HttpConfig::default();
        bad.proxy = Some("not-a-url".to_string());

        let ctx = BuildContext {
            provider_id: Some("togetherai".to_string()),
            api_key: Some("ctx-key".to_string()),
            http_client: Some(reqwest::Client::new()),
            http_config: Some(bad),
            ..Default::default()
        };

        factory
            .reranking_model_with_ctx("Salesforce/Llama-Rank-v1", &ctx)
            .await
            .expect("factory should prefer ctx.http_client over invalid http_config");
    }

    #[tokio::test]
    async fn togetherai_factory_uses_env_api_key_when_ctx_missing() {
        let _lock = lock_env();

        let _k = EnvGuard::set("TOGETHER_API_KEY", "env-key");

        let factory = crate::registry::factories::TogetherAiProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("togetherai".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .reranking_model_with_ctx("Salesforce/Llama-Rank-v1", &ctx)
            .await
            .expect("build client via env api key");

        let rerank = client
            .as_rerank_capability()
            .expect("togetherai client should expose rerank capability");
        let _ = rerank
            .rerank(make_rerank_request("Salesforce/Llama-Rank-v1"))
            .await;

        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer env-key");
        assert_eq!(req.url, "https://api.together.xyz/v1/rerank");
    }

    #[tokio::test]
    async fn togetherai_factory_prefers_ctx_api_key_over_env() {
        let _lock = lock_env();

        let _k = EnvGuard::set("TOGETHER_API_KEY", "env-key");

        let factory = crate::registry::factories::TogetherAiProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("togetherai".to_string()),
            api_key: Some("ctx-key".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .reranking_model_with_ctx("Salesforce/Llama-Rank-v1", &ctx)
            .await
            .expect("build client via ctx api key");

        let rerank = client
            .as_rerank_capability()
            .expect("togetherai client should expose rerank capability");
        let _ = rerank
            .rerank(make_rerank_request("Salesforce/Llama-Rank-v1"))
            .await;

        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
    }

    #[tokio::test]
    async fn togetherai_factory_prefers_ctx_base_url_over_default() {
        let _lock = lock_env();

        let _k = EnvGuard::set("TOGETHER_API_KEY", "env-key");

        let factory = crate::registry::factories::TogetherAiProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("togetherai".to_string()),
            base_url: Some("https://example.com/together".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .reranking_model_with_ctx("Salesforce/Llama-Rank-v1", &ctx)
            .await
            .expect("build client via base_url override");

        let rerank = client
            .as_rerank_capability()
            .expect("togetherai client should expose rerank capability");
        let _ = rerank
            .rerank(make_rerank_request("Salesforce/Llama-Rank-v1"))
            .await;

        let req = transport.take().expect("captured request");
        assert!(req.url.starts_with("https://example.com/together/"));
        assert!(req.url.ends_with("/rerank"));
    }

    #[tokio::test]
    async fn togetherai_factory_materializes_provider_owned_typed_client() {
        let _lock = lock_env();

        let factory = crate::registry::factories::TogetherAiProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("togetherai".to_string()),
            api_key: Some("ctx-key".to_string()),
            ..Default::default()
        };

        let client = factory
            .reranking_model_with_ctx("Salesforce/Llama-Rank-v1", &ctx)
            .await
            .expect("build togetherai typed client");

        assert!(
            client
                .as_any()
                .is::<siumai_provider_togetherai::providers::togetherai::TogetherAiClient>(),
            "expected provider-owned TogetherAiClient"
        );
    }

    #[tokio::test]
    async fn togetherai_factory_rejects_language_model_path_without_chat() {
        let _lock = lock_env();

        let factory = crate::registry::factories::TogetherAiProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("togetherai".to_string()),
            api_key: Some("ctx-key".to_string()),
            ..Default::default()
        };

        assert_unsupported_operation_contains(
            factory
                .language_model_with_ctx("Salesforce/Llama-Rank-v1", &ctx)
                .await,
            "language_model/chat family path",
        );
    }

    #[tokio::test]
    async fn togetherai_registry_rejects_language_model_handle_without_chat() {
        let _lock = lock_env();

        let transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "togetherai".to_string(),
            Arc::new(crate::registry::factories::TogetherAiProviderFactory)
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("ctx-key")
            .with_base_url("https://example.com/together")
            .fetch(Arc::new(transport.clone()))
            .auto_middleware(false)
            .build()
            .expect("build registry");

        assert_unsupported_operation_contains(
            registry.language_model("togetherai:Salesforce/Llama-Rank-v1"),
            "family-specific entries",
        );
        assert_capture_transport_unused(&transport);
    }

    #[tokio::test]
    async fn togetherai_builder_config_registry_rerank_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client =
            siumai_provider_togetherai::providers::togetherai::TogetherAiBuilder::new(
                siumai_provider_togetherai::builder::BuilderBase::default(),
            )
            .api_key("ctx-key")
            .base_url("https://example.com/together")
            .reranking_model("Salesforce/Llama-Rank-v1")
            .fetch(Arc::new(builder_transport.clone()))
            .build()
            .expect("build builder client");

        let config_client =
            siumai_provider_togetherai::providers::togetherai::TogetherAiClient::from_config(
                siumai_provider_togetherai::providers::togetherai::TogetherAiConfig::new("ctx-key")
                    .with_base_url("https://example.com/together")
                    .with_model("Salesforce/Llama-Rank-v1")
                    .with_http_transport(Arc::new(config_transport.clone())),
            )
            .expect("build config client");

        let factory = crate::registry::factories::TogetherAiProviderFactory;
        let registry_client = factory
            .reranking_model_with_ctx(
                "Salesforce/Llama-Rank-v1",
                &BuildContext {
                    provider_id: Some("togetherai".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/together".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_rerank_request("Salesforce/Llama-Rank-v1")
            .with_top_n(1)
            .with_togetherai_options(
                TogetherAiRerankOptions::new()
                    .with_rank_fields(vec!["title".to_string(), "body".to_string()]),
            );

        let _ = builder_client.rerank(request.clone()).await;
        let _ = config_client.rerank(request.clone()).await;
        let _ = registry_client
            .as_rerank_capability()
            .expect("registry rerank capability")
            .rerank(request)
            .await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(
            builder_req.headers.get(AUTHORIZATION).unwrap(),
            "Bearer ctx-key"
        );
        assert_eq!(builder_req.url, "https://example.com/together/rerank");
        assert_eq!(
            builder_req.body["model"],
            serde_json::json!("Salesforce/Llama-Rank-v1")
        );
        assert_eq!(builder_req.body["query"], serde_json::json!("query"));
        assert_eq!(builder_req.body["top_n"], serde_json::json!(1));
        assert_eq!(
            builder_req.body["documents"],
            serde_json::json!(["doc-1", "doc-2"])
        );
        assert_eq!(
            builder_req.body["return_documents"],
            serde_json::json!(false)
        );
        assert_eq!(
            builder_req.body["rank_fields"],
            serde_json::json!(["title", "body"])
        );
    }

    #[tokio::test]
    async fn togetherai_registry_rerank_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let together_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "togetherai".to_string(),
            Arc::new(crate::registry::factories::TogetherAiProviderFactory)
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "togetherai",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/together")
                    .fetch(Arc::new(together_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .reranking_model("togetherai:Salesforce/Llama-Rank-v1")
            .expect("build togetherai rerank handle");

        let _ = handle
            .rerank(
                make_rerank_request("Salesforce/Llama-Rank-v1")
                    .with_top_n(1)
                    .with_togetherai_options(
                        TogetherAiRerankOptions::new()
                            .with_rank_fields(vec!["title".to_string(), "body".to_string()]),
                    ),
            )
            .await;

        let req = together_transport
            .take()
            .expect("captured togetherai request");
        assert!(global_transport.take().is_none());
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert_eq!(req.url, "https://example.com/together/rerank");
        assert_eq!(req.body["top_n"], serde_json::json!(1));
        assert_eq!(
            req.body["rank_fields"],
            serde_json::json!(["title", "body"])
        );
    }
}

#[cfg(feature = "bedrock")]
mod bedrock_contract {
    use super::*;
    use crate::traits::{ChatCapability, RerankCapability};
    use reqwest::header::AUTHORIZATION;

    #[tokio::test]
    async fn bedrock_factory_keeps_non_chat_rerank_capabilities_deferred() {
        let _lock = lock_env();

        let factory = crate::registry::factories::BedrockProviderFactory;
        let caps = factory.capabilities();

        assert!(caps.supports("chat"));
        assert!(caps.supports("streaming"));
        assert!(caps.supports("tools"));
        assert!(caps.supports("rerank"));
        assert!(!caps.supports("embedding"));
        assert!(!caps.supports("image_generation"));
        assert!(!caps.supports("speech"));
        assert!(!caps.supports("transcription"));
        assert!(!caps.supports("audio"));
    }

    #[tokio::test]
    async fn bedrock_factory_prefers_ctx_http_client_over_http_config() {
        let _lock = lock_env();

        let factory = crate::registry::factories::BedrockProviderFactory;

        let mut bad = HttpConfig::default();
        bad.proxy = Some("not-a-url".to_string());

        let ctx = BuildContext {
            provider_id: Some("bedrock".to_string()),
            api_key: Some("ctx-key".to_string()),
            http_client: Some(reqwest::Client::new()),
            http_config: Some(bad),
            ..Default::default()
        };

        factory
            .language_model_with_ctx("anthropic.claude-3-haiku-20240307-v1:0", &ctx)
            .await
            .expect("factory should prefer ctx.http_client over invalid http_config");
    }

    #[tokio::test]
    async fn bedrock_factory_uses_env_api_key_when_ctx_missing() {
        let _lock = lock_env();

        let _k = EnvGuard::set("BEDROCK_API_KEY", "env-key");
        let _r = EnvGuard::set("AWS_REGION", "us-east-1");

        let factory = crate::registry::factories::BedrockProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("bedrock".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("anthropic.claude-3-haiku-20240307-v1:0", &ctx)
            .await
            .expect("build client via env api key");

        let chat = client
            .as_chat_capability()
            .expect("bedrock client should expose chat capability");
        let _ = chat
            .chat_request(make_chat_request_with_model(
                "anthropic.claude-3-haiku-20240307-v1:0",
            ))
            .await;

        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer env-key");
        assert_eq!(
            req.url,
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1%3A0/converse"
        );
    }

    #[tokio::test]
    async fn bedrock_factory_prefers_ctx_api_key_over_env() {
        let _lock = lock_env();

        let _k = EnvGuard::set("BEDROCK_API_KEY", "env-key");

        let factory = crate::registry::factories::BedrockProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("bedrock".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://bedrock-runtime.us-east-1.amazonaws.com".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("anthropic.claude-3-haiku-20240307-v1:0", &ctx)
            .await
            .expect("build client via ctx api key");

        let chat = client
            .as_chat_capability()
            .expect("bedrock client should expose chat capability");
        let _ = chat
            .chat_request(make_chat_request_with_model(
                "anthropic.claude-3-haiku-20240307-v1:0",
            ))
            .await;

        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
    }

    #[tokio::test]
    async fn bedrock_factory_prefers_ctx_base_url_and_derives_agent_runtime_url() {
        let _lock = lock_env();

        let _k = EnvGuard::set("BEDROCK_API_KEY", "env-key");

        let factory = crate::registry::factories::BedrockProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("bedrock".to_string()),
            base_url: Some("https://bedrock-runtime.us-west-2.amazonaws.com".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .reranking_model_with_ctx("amazon.rerank-v1:0", &ctx)
            .await
            .expect("build client via base_url override");

        let rerank = client
            .as_rerank_capability()
            .expect("bedrock client should expose rerank capability");
        let _ = rerank
            .rerank(make_rerank_request("amazon.rerank-v1:0"))
            .await;

        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer env-key");
        assert_eq!(
            req.url,
            "https://bedrock-agent-runtime.us-west-2.amazonaws.com/rerank"
        );
    }

    #[tokio::test]
    async fn bedrock_factory_materializes_provider_owned_typed_client() {
        let _lock = lock_env();

        let factory = crate::registry::factories::BedrockProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("bedrock".to_string()),
            api_key: Some("ctx-key".to_string()),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("anthropic.claude-3-haiku-20240307-v1:0", &ctx)
            .await
            .expect("build bedrock typed client");

        assert!(
            client
                .as_any()
                .is::<siumai_provider_amazon_bedrock::providers::bedrock::BedrockClient>(),
            "expected provider-owned BedrockClient"
        );
    }

    #[tokio::test]
    async fn bedrock_builder_config_registry_chat_request_are_equivalent() {
        let _lock = lock_env();

        let runtime_base_url = "https://bedrock-runtime.us-east-1.amazonaws.com";
        let model = "anthropic.claude-3-haiku-20240307-v1:0";

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client =
            siumai_provider_amazon_bedrock::providers::bedrock::BedrockBuilder::new(
                siumai_provider_amazon_bedrock::builder::BuilderBase::default(),
            )
            .api_key("ctx-key")
            .base_url(runtime_base_url)
            .model(model)
            .fetch(Arc::new(builder_transport.clone()))
            .build()
            .expect("build builder client");

        let config_client =
            siumai_provider_amazon_bedrock::providers::bedrock::BedrockClient::from_config(
                siumai_provider_amazon_bedrock::providers::bedrock::BedrockConfig::new()
                    .with_api_key("ctx-key")
                    .with_base_url(runtime_base_url)
                    .with_model(model)
                    .with_http_transport(Arc::new(config_transport.clone())),
            )
            .expect("build config client");

        let factory = crate::registry::factories::BedrockProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                model,
                &BuildContext {
                    provider_id: Some("bedrock".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some(runtime_base_url.to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model(model).with_provider_option(
            "bedrock",
            serde_json::json!({
                "additionalModelRequestFields": { "topK": 42 }
            }),
        );

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(
            builder_req.headers.get(AUTHORIZATION).unwrap(),
            "Bearer ctx-key"
        );
        assert_eq!(
            builder_req.url,
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1%3A0/converse"
        );
        assert_eq!(
            builder_req.body["messages"][0]["role"],
            serde_json::json!("user")
        );
        assert_eq!(
            builder_req.body["messages"][0]["content"][0]["text"],
            serde_json::json!("hi")
        );
        assert_eq!(
            builder_req.body["additionalModelRequestFields"]["topK"],
            serde_json::json!(42)
        );
    }

    #[tokio::test]
    async fn bedrock_builder_config_registry_chat_stream_request_are_equivalent() {
        let _lock = lock_env();

        let runtime_base_url = "https://bedrock-runtime.us-east-1.amazonaws.com";
        let model = "anthropic.claude-3-haiku-20240307-v1:0";

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client =
            siumai_provider_amazon_bedrock::providers::bedrock::BedrockBuilder::new(
                siumai_provider_amazon_bedrock::builder::BuilderBase::default(),
            )
            .api_key("ctx-key")
            .base_url(runtime_base_url)
            .model(model)
            .fetch(Arc::new(builder_transport.clone()))
            .build()
            .expect("build builder client");

        let config_client =
            siumai_provider_amazon_bedrock::providers::bedrock::BedrockClient::from_config(
                siumai_provider_amazon_bedrock::providers::bedrock::BedrockConfig::new()
                    .with_api_key("ctx-key")
                    .with_base_url(runtime_base_url)
                    .with_model(model)
                    .with_http_transport(Arc::new(config_transport.clone())),
            )
            .expect("build config client");

        let factory = crate::registry::factories::BedrockProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                model,
                &BuildContext {
                    provider_id: Some("bedrock".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some(runtime_base_url.to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model(model)
            .with_streaming(true)
            .with_provider_option(
                "bedrock",
                serde_json::json!({
                    "additionalModelRequestFields": { "topK": 8 }
                }),
            );

        let _ = builder_client.chat_stream_request(request.clone()).await;
        let _ = config_client.chat_stream_request(request.clone()).await;
        let _ = registry_client.chat_stream_request(request).await;

        let builder_req = builder_transport
            .take_stream()
            .expect("builder stream request");
        let config_req = config_transport
            .take_stream()
            .expect("config stream request");
        let registry_req = registry_transport
            .take_stream()
            .expect("registry stream request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(
            builder_req.url,
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1%3A0/converse-stream"
        );
        assert_eq!(
            builder_req.headers.get(AUTHORIZATION).unwrap(),
            "Bearer ctx-key"
        );
        assert_eq!(
            builder_req.body["additionalModelRequestFields"]["topK"],
            serde_json::json!(8)
        );
    }

    #[tokio::test]
    async fn bedrock_builder_config_registry_stable_request_options_are_equivalent() {
        let _lock = lock_env();

        let runtime_base_url = "https://bedrock-runtime.us-east-1.amazonaws.com";
        let model = "anthropic.claude-3-haiku-20240307-v1:0";
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client =
            siumai_provider_amazon_bedrock::providers::bedrock::BedrockBuilder::new(
                siumai_provider_amazon_bedrock::builder::BuilderBase::default(),
            )
            .api_key("ctx-key")
            .base_url(runtime_base_url)
            .model(model)
            .fetch(Arc::new(builder_transport.clone()))
            .build()
            .expect("build builder client");

        let config_client =
            siumai_provider_amazon_bedrock::providers::bedrock::BedrockClient::from_config(
                siumai_provider_amazon_bedrock::providers::bedrock::BedrockConfig::new()
                    .with_api_key("ctx-key")
                    .with_base_url(runtime_base_url)
                    .with_model(model)
                    .with_http_transport(Arc::new(config_transport.clone())),
            )
            .expect("build config client");

        let factory = crate::registry::factories::BedrockProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                model,
                &BuildContext {
                    provider_id: Some("bedrock".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some(runtime_base_url.to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model(model)
            .with_tools(vec![crate::types::Tool::function(
                "lookup_weather",
                "Look up the weather",
                serde_json::json!({
                    "type": "object",
                    "properties": { "location": { "type": "string" } },
                    "required": ["location"],
                    "additionalProperties": false
                }),
            )])
            .with_tool_choice(crate::types::ToolChoice::Required)
            .with_response_format(crate::types::ResponseFormat::json_schema(schema.clone()))
            .with_provider_option(
                "bedrock",
                serde_json::json!({
                    "additionalModelRequestFields": { "topK": 16 }
                }),
            );

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(
            builder_req.headers.get(AUTHORIZATION).unwrap(),
            "Bearer ctx-key"
        );
        assert_eq!(
            builder_req.url,
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1%3A0/converse"
        );
        assert_eq!(
            builder_req.body["additionalModelRequestFields"],
            serde_json::json!({ "topK": 16 })
        );
        assert_eq!(
            builder_req.body["toolConfig"]["toolChoice"],
            serde_json::json!({ "any": {} })
        );
        let tools = builder_req.body["toolConfig"]["tools"]
            .as_array()
            .expect("tools array");
        assert_eq!(tools.len(), 2);
        assert_eq!(
            tools[0]["toolSpec"]["name"],
            serde_json::json!("lookup_weather")
        );
        assert_eq!(tools[1]["toolSpec"]["name"], serde_json::json!("json"));
        assert_eq!(tools[1]["toolSpec"]["inputSchema"]["json"], schema);
    }

    #[tokio::test]
    async fn bedrock_builder_config_registry_rerank_request_are_equivalent() {
        let _lock = lock_env();

        let runtime_base_url = "https://bedrock-runtime.us-east-1.amazonaws.com";
        let model = "amazon.rerank-v1:0";

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client =
            siumai_provider_amazon_bedrock::providers::bedrock::BedrockBuilder::new(
                siumai_provider_amazon_bedrock::builder::BuilderBase::default(),
            )
            .api_key("ctx-key")
            .base_url(runtime_base_url)
            .model(model)
            .fetch(Arc::new(builder_transport.clone()))
            .build()
            .expect("build builder client");

        let config_client =
            siumai_provider_amazon_bedrock::providers::bedrock::BedrockClient::from_config(
                siumai_provider_amazon_bedrock::providers::bedrock::BedrockConfig::new()
                    .with_api_key("ctx-key")
                    .with_base_url(runtime_base_url)
                    .with_model(model)
                    .with_http_transport(Arc::new(config_transport.clone())),
            )
            .expect("build config client");

        let factory = crate::registry::factories::BedrockProviderFactory;
        let registry_client = factory
            .reranking_model_with_ctx(
                model,
                &BuildContext {
                    provider_id: Some("bedrock".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some(runtime_base_url.to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_rerank_request(model)
            .with_top_n(1)
            .with_provider_option(
                "bedrock",
                serde_json::json!({
                    "region": "us-east-1",
                    "nextToken": "token-1",
                    "additionalModelRequestFields": { "topK": 4 }
                }),
            );

        let _ = builder_client.rerank(request.clone()).await;
        let _ = config_client.rerank(request.clone()).await;
        let _ = registry_client
            .as_rerank_capability()
            .expect("registry rerank capability")
            .rerank(request)
            .await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(
            builder_req.url,
            "https://bedrock-agent-runtime.us-east-1.amazonaws.com/rerank"
        );
        assert_eq!(
            builder_req.headers.get(AUTHORIZATION).unwrap(),
            "Bearer ctx-key"
        );
        assert_eq!(builder_req.body["nextToken"], serde_json::json!("token-1"));
        assert_eq!(
            builder_req.body["rerankingConfiguration"]["bedrockRerankingConfiguration"]["modelConfiguration"]
                ["modelArn"],
            serde_json::json!("arn:aws:bedrock:us-east-1::foundation-model/amazon.rerank-v1:0")
        );
        assert_eq!(
            builder_req.body["rerankingConfiguration"]["bedrockRerankingConfiguration"]["modelConfiguration"]
                ["additionalModelRequestFields"]["topK"],
            serde_json::json!(4)
        );
    }

    #[tokio::test]
    async fn bedrock_registry_rerank_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let model = "amazon.rerank-v1:0";
        let runtime_base_url = "https://bedrock-runtime.us-east-1.amazonaws.com";
        let global_transport = CaptureTransport::default();
        let bedrock_transport = CaptureTransport::default();

        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "bedrock".to_string(),
            Arc::new(crate::registry::factories::BedrockProviderFactory)
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/not-bedrock")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "bedrock",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url(runtime_base_url)
                    .fetch(Arc::new(bedrock_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .reranking_model(&format!("bedrock:{model}"))
            .expect("build bedrock rerank handle");

        let _ = handle
            .rerank(make_rerank_request(model).with_provider_option(
                "bedrock",
                serde_json::json!({
                    "region": "us-east-1",
                    "nextToken": "token-1",
                }),
            ))
            .await;

        let req = bedrock_transport.take().expect("captured bedrock request");
        assert!(global_transport.take().is_none());
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert_eq!(
            req.url,
            "https://bedrock-agent-runtime.us-east-1.amazonaws.com/rerank"
        );
        assert_eq!(req.body["nextToken"], serde_json::json!("token-1"));
    }

    #[tokio::test]
    async fn bedrock_registry_chat_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let model = "anthropic.claude-3-haiku-20240307-v1:0";
        let runtime_base_url = "https://bedrock-runtime.us-east-1.amazonaws.com";
        let global_transport = CaptureTransport::default();
        let bedrock_transport = CaptureTransport::default();

        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "bedrock".to_string(),
            Arc::new(crate::registry::factories::BedrockProviderFactory)
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/not-bedrock")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "bedrock",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url(runtime_base_url)
                    .fetch(Arc::new(bedrock_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .language_model(&format!("bedrock:{model}"))
            .expect("build bedrock language model handle");

        let _ = handle
            .chat_request(make_chat_request_with_model(model).with_provider_option(
                "bedrock",
                serde_json::json!({
                    "additionalModelRequestFields": { "topK": 24 }
                }),
            ))
            .await;

        let req = bedrock_transport.take().expect("captured bedrock request");
        assert!(global_transport.take().is_none());
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert_eq!(
            req.url,
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1%3A0/converse"
        );
        assert_eq!(
            req.body["additionalModelRequestFields"]["topK"],
            serde_json::json!(24)
        );
    }

    #[tokio::test]
    async fn bedrock_registry_chat_stream_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let model = "anthropic.claude-3-haiku-20240307-v1:0";
        let runtime_base_url = "https://bedrock-runtime.us-east-1.amazonaws.com";
        let global_transport = CaptureTransport::default();
        let bedrock_transport = CaptureTransport::default();

        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "bedrock".to_string(),
            Arc::new(crate::registry::factories::BedrockProviderFactory)
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/not-bedrock")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "bedrock",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url(runtime_base_url)
                    .fetch(Arc::new(bedrock_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .language_model(&format!("bedrock:{model}"))
            .expect("build bedrock language model handle");

        let _ = handle
            .chat_stream_request(make_chat_request_with_model(model).with_provider_option(
                "bedrock",
                serde_json::json!({
                    "additionalModelRequestFields": { "topK": 24 }
                }),
            ))
            .await;

        let req = bedrock_transport
            .take_stream()
            .expect("captured bedrock stream request");
        assert!(global_transport.take().is_none());
        assert!(global_transport.take_stream().is_none());
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert_eq!(
            req.url,
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1%3A0/converse-stream"
        );
        assert_eq!(
            req.body["additionalModelRequestFields"]["topK"],
            serde_json::json!(24)
        );
    }
}

#[cfg(feature = "openai")]
mod openai_contract {
    use super::*;
    use crate::registry::factories::{OpenAICompatibleProviderFactory, OpenAIProviderFactory};
    use crate::traits::{
        AudioCapability, ChatCapability, EmbeddingExtensions, ImageGenerationCapability,
        RerankCapability,
    };
    use reqwest::header::AUTHORIZATION;
    use siumai_provider_openai::provider_metadata::openai::{
        OpenAiChatResponseExt, OpenAiSourceExt,
    };
    use siumai_provider_openai_compatible::providers::openai_compatible::{
        OpenRouterChatResponseExt, PerplexityChatResponseExt,
    };

    #[tokio::test]
    async fn openai_factory_prefers_ctx_http_client_over_http_config() {
        let _lock = lock_env();

        let factory = OpenAIProviderFactory;
        let transport = CaptureTransport::default();

        let mut bad = HttpConfig::default();
        bad.proxy = Some("not-a-url".to_string());

        let ctx = BuildContext {
            provider_id: Some("openai".to_string()),
            api_key: Some("ctx-key".to_string()),
            http_client: Some(reqwest::Client::new()),
            http_config: Some(bad),
            http_transport: Some(Arc::new(transport)),
            ..Default::default()
        };

        factory
            .language_model_with_ctx("gpt-4o", &ctx)
            .await
            .expect("factory should prefer ctx.http_client over invalid http_config");
    }

    #[tokio::test]
    async fn openai_factory_keeps_native_rerank_out_of_declared_capabilities() {
        let _lock = lock_env();

        let factory = OpenAIProviderFactory;
        let caps = factory.capabilities();

        assert!(caps.supports("chat"));
        assert!(caps.supports("streaming"));
        assert!(caps.supports("tools"));
        assert!(caps.supports("embedding"));
        assert!(caps.supports("image_generation"));
        assert!(!caps.supports("rerank"));
    }

    #[tokio::test]
    async fn openai_factory_uses_env_api_key_when_ctx_missing() {
        let _lock = lock_env();

        let _g = EnvGuard::set("OPENAI_API_KEY", "env-key");

        let factory = OpenAIProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("openai".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("gpt-4o", &ctx)
            .await
            .expect("build client via env api key");

        let _ = client.chat_request(make_chat_request()).await;
        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer env-key");
        assert!(req.url.ends_with("/responses"));
    }

    #[tokio::test]
    async fn openai_factory_prefers_ctx_api_key_over_env() {
        let _lock = lock_env();

        let _g = EnvGuard::set("OPENAI_API_KEY", "env-key");

        let factory = OpenAIProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("openai".to_string()),
            api_key: Some("ctx-key".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("gpt-4o", &ctx)
            .await
            .expect("build client via ctx api key");

        let _ = client.chat_request(make_chat_request()).await;
        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
    }

    #[tokio::test]
    async fn openai_factory_prefers_ctx_base_url_over_default() {
        let _lock = lock_env();

        let factory = OpenAIProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("openai".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom/v1/".to_string()),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("gpt-4o", &ctx)
            .await
            .expect("build client");

        let typed = client
            .as_any()
            .downcast_ref::<siumai_provider_openai::providers::openai::OpenAiClient>()
            .expect("OpenAiClient");
        assert_eq!(typed.base_url(), "https://example.com/custom/v1");
    }

    #[tokio::test]
    async fn openai_factory_supports_native_text_family_path() {
        let _lock = lock_env();

        let factory = OpenAIProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("openai".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom/v1/".to_string()),
            ..Default::default()
        };

        let model = factory
            .language_model_text_with_ctx("gpt-4o", &ctx)
            .await
            .expect("build native text-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "openai"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "gpt-4o"
        );
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model.as_ref()),
            crate::traits::ModelSpecVersion::V1
        );
    }

    #[tokio::test]
    async fn openai_factory_supports_native_image_family_path() {
        let _lock = lock_env();

        let factory = OpenAIProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("openai".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom/v1/".to_string()),
            ..Default::default()
        };

        let model = factory
            .image_model_family_with_ctx("gpt-image-1", &ctx)
            .await
            .expect("build native image-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "openai"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "gpt-image-1"
        );
    }

    #[tokio::test]
    async fn openai_factory_supports_native_speech_family_path() {
        let _lock = lock_env();

        let factory = OpenAIProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("openai".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom/v1/".to_string()),
            ..Default::default()
        };

        let model = factory
            .speech_model_family_with_ctx("gpt-4o-mini-tts", &ctx)
            .await
            .expect("build native speech-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "openai"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "gpt-4o-mini-tts"
        );
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model.as_ref()),
            crate::traits::ModelSpecVersion::V1
        );
    }

    #[tokio::test]
    async fn openai_factory_supports_native_transcription_family_path() {
        let _lock = lock_env();

        let factory = OpenAIProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("openai".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom/v1/".to_string()),
            ..Default::default()
        };

        let model = factory
            .transcription_model_family_with_ctx("gpt-4o-mini-transcribe", &ctx)
            .await
            .expect("build native transcription-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "openai"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "gpt-4o-mini-transcribe"
        );
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model.as_ref()),
            crate::traits::ModelSpecVersion::V1
        );
    }

    #[test]
    fn openai_factory_source_declares_native_audio_family_overrides() {
        let source = include_str!("openai.rs");

        assert!(source.contains("async fn speech_model_family_with_ctx("));
        assert!(source.contains("async fn transcription_model_family_with_ctx("));
    }

    #[tokio::test]
    async fn openai_registry_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let openai_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "openai".to_string(),
            Arc::new(OpenAIProviderFactory) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "openai",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/openai/v1")
                    .fetch(Arc::new(openai_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .language_model("openai:gpt-4o")
            .expect("build openai handle");

        let _ = handle
            .chat_request(make_chat_request_with_model("gpt-4o"))
            .await;

        let req = openai_transport.take().expect("captured openai request");
        assert!(global_transport.take().is_none());
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert_eq!(req.url, "https://example.com/openai/v1/responses");
        assert_eq!(req.body["model"], serde_json::json!("gpt-4o"));
    }

    #[tokio::test]
    async fn openai_registry_stream_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let openai_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "openai".to_string(),
            Arc::new(OpenAIProviderFactory) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "openai",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/openai/v1")
                    .fetch(Arc::new(openai_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .language_model("openai:gpt-4o")
            .expect("build openai handle");

        let _ = handle
            .chat_stream_request(make_chat_request_with_model("gpt-4o"))
            .await;

        let req = openai_transport
            .take_stream()
            .expect("captured openai stream request");
        assert!(global_transport.take().is_none());
        assert!(global_transport.take_stream().is_none());
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert_eq!(req.url, "https://example.com/openai/v1/responses");
        assert_eq!(
            header_value(&req, "accept"),
            Some("text/event-stream".to_string())
        );
        assert_eq!(req.body["model"], serde_json::json!("gpt-4o"));
    }

    #[tokio::test]
    async fn openai_chat_registry_override_chat_response_metadata_preserves_provider_namespace() {
        let _lock = lock_env();

        let response_json = serde_json::json!({
            "id": "chatcmpl-openai-test",
            "object": "chat.completion",
            "created": 1_718_345_013,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "hello from openai chat"
                    },
                    "finish_reason": "stop",
                    "logprobs": {
                        "content": [
                            {
                                "token": "hello",
                                "logprob": -0.1,
                                "bytes": [104, 101, 108, 108, 111],
                                "top_logprobs": []
                            }
                        ]
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 3,
                "total_tokens": 14
            }
        });

        let global_transport = CaptureTransport::default();
        let openai_transport = JsonSuccessTransport::new(response_json);
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "openai-chat".to_string(),
            Arc::new(OpenAIProviderFactory) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "openai-chat",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/openai/v1")
                    .fetch(Arc::new(openai_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let response = registry
            .language_model("openai-chat:gpt-4o-mini")
            .expect("build openai-chat handle")
            .chat_request(
                make_chat_request_with_model("gpt-4o-mini")
                    .with_provider_option("openai", serde_json::json!({ "logprobs": 3 })),
            )
            .await
            .expect("registry response ok");

        let root = response
            .provider_metadata
            .as_ref()
            .expect("registry provider metadata");
        assert!(root.get("openai").is_some());
        assert!(root.get("azure").is_none());

        let metadata = response.openai_metadata().expect("openai metadata");
        assert_eq!(response.content_text(), Some("hello from openai chat"));
        assert_eq!(
            response.usage.as_ref().map(|usage| usage.total_tokens),
            Some(14)
        );
        assert_eq!(
            metadata
                .logprobs
                .as_ref()
                .and_then(|value| value.as_array().map(Vec::len)),
            Some(1)
        );

        let req = openai_transport.take().expect("captured request");
        assert!(global_transport.take().is_none());
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert_eq!(req.url, "https://example.com/openai/v1/chat/completions");
        assert_eq!(req.body["logprobs"], serde_json::json!(true));
        assert_eq!(req.body["top_logprobs"], serde_json::json!(3));
    }

    #[tokio::test]
    async fn openai_registry_override_stream_end_metadata_preserves_provider_namespace() {
        let _lock = lock_env();

        let stream_body = concat!(
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_sources_2\",\"model\":\"gpt-4.1\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"See attached files.\",\"annotations\":[{\"type\":\"container_file_citation\",\"file_id\":\"file_container_1\",\"container_id\":\"container_42\",\"index\":3,\"filename\":\"bundle.txt\",\"quote\":\"Bundle\"},{\"type\":\"file_path\",\"file_id\":\"file_path_9\",\"index\":5,\"filename\":\"artifact.bin\"}]}]}],\"usage\":{\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3},\"finish_reason\":\"stop\"}}\n\n"
        )
        .as_bytes()
        .to_vec();

        let global_transport = CaptureTransport::default();
        let openai_transport = SseSuccessTransport::new(stream_body);
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "openai".to_string(),
            Arc::new(OpenAIProviderFactory) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "openai",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/openai/v1")
                    .fetch(Arc::new(openai_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let mut stream = registry
            .language_model("openai:gpt-4.1")
            .expect("build openai handle")
            .chat_stream_request(make_chat_request_with_model("gpt-4.1"))
            .await
            .expect("registry stream ok");

        use futures::StreamExt;

        let mut stream_end = None;
        while let Some(event) = stream.next().await {
            if let Ok(crate::streaming::ChatStreamEvent::StreamEnd { response }) = event {
                stream_end = Some(response);
                break;
            }
        }

        let response = stream_end.expect("registry stream end");
        let root = response
            .provider_metadata
            .as_ref()
            .expect("registry provider metadata");
        assert!(root.get("openai").is_some());
        assert!(root.get("azure").is_none());

        let metadata = response.openai_metadata().expect("openai metadata");
        assert_eq!(response.content_text(), Some("See attached files."));
        assert_eq!(
            response.finish_reason,
            Some(crate::types::FinishReason::Stop)
        );
        assert_eq!(metadata.sources.as_ref().map(Vec::len), Some(2));

        let container = metadata
            .sources
            .as_ref()
            .and_then(|sources| {
                sources
                    .iter()
                    .find(|source| source.url == "file_container_1")
            })
            .expect("container source");
        let container_meta = container.openai_metadata().expect("container metadata");
        assert_eq!(container_meta.file_id.as_deref(), Some("file_container_1"));
        assert_eq!(container_meta.container_id.as_deref(), Some("container_42"));
        assert_eq!(container_meta.index, Some(3));

        let file_path = metadata
            .sources
            .as_ref()
            .and_then(|sources| sources.iter().find(|source| source.url == "file_path_9"))
            .expect("file path source");
        let file_path_meta = file_path.openai_metadata().expect("file path metadata");
        assert_eq!(file_path_meta.file_id.as_deref(), Some("file_path_9"));
        assert!(file_path_meta.container_id.is_none());
        assert_eq!(file_path_meta.index, Some(5));

        let req = openai_transport
            .take_stream()
            .expect("captured openai stream request");
        assert!(global_transport.take().is_none());
        assert!(global_transport.take_stream().is_none());
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert_eq!(req.url, "https://example.com/openai/v1/responses");
        assert_eq!(
            header_value(&req, "accept"),
            Some("text/event-stream".to_string())
        );
    }

    #[tokio::test]
    async fn openai_compatible_factory_prefers_ctx_api_key_over_env() {
        let _lock = lock_env();

        let _g = EnvGuard::set("DEEPSEEK_API_KEY", "env-key");

        let factory = OpenAICompatibleProviderFactory::new("deepseek".to_string());
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("deepseek".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/v1".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("deepseek-chat", &ctx)
            .await
            .expect("build openai-compatible client");

        let _ = client.chat_request(make_chat_request()).await;
        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
    }

    #[tokio::test]
    async fn openai_compatible_factory_openrouter_uses_env_api_key_when_ctx_missing() {
        let _lock = lock_env();

        let _g = EnvGuard::set("OPENROUTER_API_KEY", "env-key");

        let factory = OpenAICompatibleProviderFactory::new("openrouter".to_string());
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("openrouter".to_string()),
            base_url: Some("https://example.com/v1".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("openai/gpt-4o", &ctx)
            .await
            .expect("build openai-compatible client via env api key");

        let _ = client
            .chat_request(make_chat_request_with_model("openai/gpt-4o"))
            .await;
        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer env-key");
        assert!(req.url.starts_with("https://example.com/v1"));
    }

    #[tokio::test]
    async fn openai_compatible_factory_openrouter_prefers_ctx_api_key_over_env() {
        let _lock = lock_env();

        let _g = EnvGuard::set("OPENROUTER_API_KEY", "env-key");

        let factory = OpenAICompatibleProviderFactory::new("openrouter".to_string());
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("openrouter".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/v1".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("openai/gpt-4o", &ctx)
            .await
            .expect("build openai-compatible client via ctx api key");

        let _ = client
            .chat_request(make_chat_request_with_model("openai/gpt-4o"))
            .await;
        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert!(req.url.starts_with("https://example.com/v1"));
    }

    #[tokio::test]
    async fn openai_compatible_factory_supports_native_text_family_path() {
        let _lock = lock_env();

        let factory = OpenAICompatibleProviderFactory::new("deepseek".to_string());
        let ctx = BuildContext {
            provider_id: Some("deepseek".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/v1/".to_string()),
            ..Default::default()
        };

        let model = factory
            .language_model_text_with_ctx("deepseek-chat", &ctx)
            .await
            .expect("build native text-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "deepseek"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "deepseek-chat"
        );
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model.as_ref()),
            crate::traits::ModelSpecVersion::V1
        );
    }

    #[tokio::test]
    async fn openai_compatible_factory_supports_native_embedding_family_path() {
        let _lock = lock_env();

        let factory = OpenAICompatibleProviderFactory::new("openrouter".to_string());
        let ctx = BuildContext {
            provider_id: Some("openrouter".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/v1/".to_string()),
            ..Default::default()
        };

        let model = factory
            .embedding_model_family_with_ctx("openai/text-embedding-3-small", &ctx)
            .await
            .expect("build native embedding-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "openrouter"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "openai/text-embedding-3-small"
        );
    }

    #[tokio::test]
    async fn openai_compatible_factory_jina_declares_rerank_capability() {
        let _lock = lock_env();

        let factory = OpenAICompatibleProviderFactory::new("jina".to_string());
        let caps = factory.capabilities();

        assert!(!caps.supports("chat"));
        assert!(!caps.supports("streaming"));
        assert!(caps.supports("rerank"));
    }

    #[tokio::test]
    async fn openai_compatible_factory_jina_rejects_language_model_path_without_chat() {
        let _lock = lock_env();

        let factory = OpenAICompatibleProviderFactory::new("jina".to_string());
        let ctx = BuildContext {
            provider_id: Some("jina".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/v1/".to_string()),
            ..Default::default()
        };

        assert_unsupported_operation_contains(
            factory
                .language_model_with_ctx("jina-embeddings-v2-base-en", &ctx)
                .await,
            "chat",
        );
    }

    #[tokio::test]
    async fn openai_compatible_factory_voyageai_declares_rerank_capability() {
        let _lock = lock_env();

        let factory = OpenAICompatibleProviderFactory::new("voyageai".to_string());
        let caps = factory.capabilities();

        assert!(!caps.supports("chat"));
        assert!(!caps.supports("streaming"));
        assert!(caps.supports("rerank"));
    }

    #[tokio::test]
    async fn openai_compatible_factory_voyageai_rejects_language_model_path_without_chat() {
        let _lock = lock_env();

        let factory = OpenAICompatibleProviderFactory::new("voyageai".to_string());
        let ctx = BuildContext {
            provider_id: Some("voyageai".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/v1/".to_string()),
            ..Default::default()
        };

        assert_unsupported_operation_contains(
            factory.language_model_with_ctx("voyage-3", &ctx).await,
            "chat",
        );
    }

    #[tokio::test]
    async fn openai_compatible_registry_jina_rejects_language_model_handle_without_chat() {
        let _lock = lock_env();

        let transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "jina".to_string(),
            Arc::new(OpenAICompatibleProviderFactory::new("jina".to_string()))
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("ctx-key")
            .with_base_url("https://example.com/v1/")
            .fetch(Arc::new(transport.clone()))
            .auto_middleware(false)
            .build()
            .expect("build registry");

        assert_unsupported_operation_contains(
            registry.language_model("jina:jina-embeddings-v2-base-en"),
            "family-specific entries",
        );
        assert_capture_transport_unused(&transport);
    }

    #[tokio::test]
    async fn openai_compatible_registry_voyageai_rejects_language_model_handle_without_chat() {
        let _lock = lock_env();

        let transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "voyageai".to_string(),
            Arc::new(OpenAICompatibleProviderFactory::new("voyageai".to_string()))
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("ctx-key")
            .with_base_url("https://example.com/v1/")
            .fetch(Arc::new(transport.clone()))
            .auto_middleware(false)
            .build()
            .expect("build registry");

        assert_unsupported_operation_contains(
            registry.language_model("voyageai:voyage-3"),
            "family-specific entries",
        );
        assert_capture_transport_unused(&transport);
    }

    #[tokio::test]
    async fn openai_compatible_factory_infini_keeps_chat_surface_while_exposing_embedding() {
        let _lock = lock_env();

        let factory = OpenAICompatibleProviderFactory::new("infini".to_string());
        let caps = factory.capabilities();

        assert!(caps.supports("chat"));
        assert!(caps.supports("streaming"));
        assert!(caps.supports("embedding"));
        assert!(!caps.supports("rerank"));
    }

    #[tokio::test]
    async fn openai_compatible_factory_infini_keeps_language_model_path_when_chat_supported() {
        let _lock = lock_env();

        let factory = OpenAICompatibleProviderFactory::new("infini".to_string());
        let ctx = BuildContext {
            provider_id: Some("infini".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/v1/".to_string()),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("deepseek-chat", &ctx)
            .await
            .expect("build infini language-model client");

        assert!(client.as_chat_capability().is_some());
        assert!(client.as_embedding_capability().is_some());
    }

    #[tokio::test]
    async fn openai_compatible_registry_infini_keeps_language_model_handle_when_chat_supported() {
        let _lock = lock_env();

        let transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "infini".to_string(),
            Arc::new(OpenAICompatibleProviderFactory::new("infini".to_string()))
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("ctx-key")
            .with_base_url("https://example.com/v1/")
            .fetch(Arc::new(transport.clone()))
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .language_model("infini:deepseek-chat")
            .expect("build infini text handle");

        assert!(crate::client::LlmClient::as_chat_capability(&handle).is_some());
        let _ = handle
            .chat_request(make_chat_request_with_model("deepseek-chat"))
            .await;

        let req = transport.take().expect("captured infini request");
        assert_eq!(req.url, "https://example.com/v1/chat/completions");
        assert_eq!(req.body["model"], serde_json::json!("deepseek-chat"));
    }

    #[tokio::test]
    async fn openai_compatible_factory_openrouter_does_not_declare_rerank_capability() {
        let _lock = lock_env();

        let factory = OpenAICompatibleProviderFactory::new("openrouter".to_string());
        let caps = factory.capabilities();

        assert!(!caps.supports("rerank"));
    }

    #[tokio::test]
    async fn openai_compatible_factory_openrouter_rejects_native_rerank_family_path() {
        let _lock = lock_env();

        let factory = OpenAICompatibleProviderFactory::new("openrouter".to_string());
        let transport = CaptureTransport::default();
        let ctx = BuildContext {
            provider_id: Some("openrouter".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/v1/".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let result = factory
            .reranking_model_family_with_ctx("openai/gpt-4o", &ctx)
            .await;

        assert_unsupported_operation_contains(result, "rerank");
        assert_capture_transport_unused(&transport);
    }

    #[tokio::test]
    async fn openai_compatible_factory_jina_supports_native_rerank_family_path() {
        let _lock = lock_env();

        let factory = OpenAICompatibleProviderFactory::new("jina".to_string());
        let ctx = BuildContext {
            provider_id: Some("jina".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/v1/".to_string()),
            ..Default::default()
        };

        let model = factory
            .reranking_model_family_with_ctx("jina-reranker-m0", &ctx)
            .await
            .expect("build native rerank-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "jina"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "jina-reranker-m0"
        );
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model.as_ref()),
            crate::traits::ModelSpecVersion::V1
        );
    }

    #[tokio::test]
    async fn siliconflow_registry_rerank_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let siliconflow_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "siliconflow".to_string(),
            Arc::new(OpenAICompatibleProviderFactory::new(
                "siliconflow".to_string(),
            )) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "siliconflow",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/siliconflow/v1")
                    .fetch(Arc::new(siliconflow_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .reranking_model("siliconflow:BAAI/bge-reranker-v2-m3")
            .expect("build siliconflow rerank handle");

        let _ = handle
            .rerank(make_rerank_request("BAAI/bge-reranker-v2-m3").with_top_n(1))
            .await;

        let req = siliconflow_transport
            .take()
            .expect("captured siliconflow request");
        assert!(global_transport.take().is_none());
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert_eq!(req.url, "https://example.com/siliconflow/v1/rerank");
        assert_eq!(
            req.body["model"],
            serde_json::json!("BAAI/bge-reranker-v2-m3")
        );
        assert_eq!(req.body["query"], serde_json::json!("query"));
        assert_eq!(req.body["documents"], serde_json::json!(["doc-1", "doc-2"]));
        assert_eq!(req.body["top_n"], serde_json::json!(1));
    }

    #[tokio::test]
    async fn jina_registry_rerank_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let jina_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "jina".to_string(),
            Arc::new(OpenAICompatibleProviderFactory::new("jina".to_string()))
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "jina",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/jina/v1")
                    .fetch(Arc::new(jina_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .reranking_model("jina:jina-reranker-m0")
            .expect("build jina rerank handle");

        let _ = handle
            .rerank(make_rerank_request("jina-reranker-m0").with_top_n(1))
            .await;

        let req = jina_transport.take().expect("captured jina request");
        assert!(global_transport.take().is_none());
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert_eq!(req.url, "https://example.com/jina/v1/rerank");
        assert_eq!(req.body["model"], serde_json::json!("jina-reranker-m0"));
        assert_eq!(req.body["query"], serde_json::json!("query"));
        assert_eq!(req.body["documents"], serde_json::json!(["doc-1", "doc-2"]));
        assert_eq!(req.body["top_n"], serde_json::json!(1));
    }

    #[tokio::test]
    async fn voyageai_registry_rerank_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let voyageai_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "voyageai".to_string(),
            Arc::new(OpenAICompatibleProviderFactory::new("voyageai".to_string()))
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "voyageai",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/voyageai/v1")
                    .fetch(Arc::new(voyageai_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .reranking_model("voyageai:rerank-2")
            .expect("build voyageai rerank handle");

        let _ = handle
            .rerank(make_rerank_request("rerank-2").with_top_n(1))
            .await;

        let req = voyageai_transport
            .take()
            .expect("captured voyageai request");
        assert!(global_transport.take().is_none());
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert_eq!(req.url, "https://example.com/voyageai/v1/rerank");
        assert_eq!(req.body["model"], serde_json::json!("rerank-2"));
        assert_eq!(req.body["query"], serde_json::json!("query"));
        assert_eq!(req.body["documents"], serde_json::json!(["doc-1", "doc-2"]));
        assert_eq!(req.body["top_n"], serde_json::json!(1));
    }

    #[tokio::test]
    async fn openai_compatible_factory_together_declares_image_capability() {
        let _lock = lock_env();

        let factory = OpenAICompatibleProviderFactory::new("together".to_string());
        let caps = factory.capabilities();

        assert!(caps.supports("image_generation"));
    }

    #[tokio::test]
    async fn openai_compatible_factory_siliconflow_declares_image_capability() {
        let _lock = lock_env();

        let factory = OpenAICompatibleProviderFactory::new("siliconflow".to_string());
        let caps = factory.capabilities();

        assert!(caps.supports("image_generation"));
    }

    #[tokio::test]
    async fn openai_compatible_factory_openrouter_does_not_declare_image_capability() {
        let _lock = lock_env();

        let factory = OpenAICompatibleProviderFactory::new("openrouter".to_string());
        let caps = factory.capabilities();

        assert!(!caps.supports("image_generation"));
    }

    #[tokio::test]
    async fn openai_compatible_factory_openrouter_rejects_native_image_family_path() {
        let _lock = lock_env();

        let factory = OpenAICompatibleProviderFactory::new("openrouter".to_string());
        let transport = CaptureTransport::default();
        let ctx = BuildContext {
            provider_id: Some("openrouter".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/v1/".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let result = factory
            .image_model_family_with_ctx("openai/gpt-image-1", &ctx)
            .await;

        assert_unsupported_operation_contains(result, "image_generation");
        assert_capture_transport_unused(&transport);
    }

    #[tokio::test]
    async fn openai_compatible_factory_together_supports_native_image_family_path() {
        let _lock = lock_env();

        let factory = OpenAICompatibleProviderFactory::new("together".to_string());
        let ctx = BuildContext {
            provider_id: Some("together".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/v1/".to_string()),
            ..Default::default()
        };

        let model = factory
            .image_model_family_with_ctx("black-forest-labs/FLUX.1-schnell", &ctx)
            .await
            .expect("build native image-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "together"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "black-forest-labs/FLUX.1-schnell"
        );
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model.as_ref()),
            crate::traits::ModelSpecVersion::V1
        );
    }

    #[tokio::test]
    async fn openai_compatible_factory_together_declares_audio_capabilities() {
        let _lock = lock_env();

        let factory = OpenAICompatibleProviderFactory::new("together".to_string());
        let caps = factory.capabilities();

        assert!(caps.supports("speech"));
        assert!(caps.supports("transcription"));
        assert!(caps.supports("audio"));
    }

    #[tokio::test]
    async fn openai_compatible_factory_siliconflow_declares_audio_capabilities() {
        let _lock = lock_env();

        let factory = OpenAICompatibleProviderFactory::new("siliconflow".to_string());
        let caps = factory.capabilities();

        assert!(caps.supports("speech"));
        assert!(caps.supports("transcription"));
        assert!(caps.supports("audio"));
    }

    #[tokio::test]
    async fn openai_compatible_factory_together_supports_native_speech_family_path() {
        let _lock = lock_env();

        let factory = OpenAICompatibleProviderFactory::new("together".to_string());
        let ctx = BuildContext {
            provider_id: Some("together".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/v1/".to_string()),
            ..Default::default()
        };

        let model = factory
            .speech_model_family_with_ctx("cartesia/sonic-2", &ctx)
            .await
            .expect("build native speech-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "together"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "cartesia/sonic-2"
        );
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model.as_ref()),
            crate::traits::ModelSpecVersion::V1
        );
    }

    #[tokio::test]
    async fn openai_compatible_factory_together_supports_native_transcription_family_path() {
        let _lock = lock_env();

        let factory = OpenAICompatibleProviderFactory::new("together".to_string());
        let ctx = BuildContext {
            provider_id: Some("together".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/v1/".to_string()),
            ..Default::default()
        };

        let model = factory
            .transcription_model_family_with_ctx("openai/whisper-large-v3", &ctx)
            .await
            .expect("build native transcription-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "together"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "openai/whisper-large-v3"
        );
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model.as_ref()),
            crate::traits::ModelSpecVersion::V1
        );
    }

    #[tokio::test]
    async fn together_registry_speech_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = BytesSuccessTransport::new(vec![9, 9, 9], "audio/mpeg");
        let together_transport = BytesSuccessTransport::new(vec![1, 2, 3, 4], "audio/mpeg");
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "together".to_string(),
            Arc::new(OpenAICompatibleProviderFactory::new("together".to_string()))
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "together",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/together/v1")
                    .fetch(Arc::new(together_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .speech_model("together:cartesia/sonic-2")
            .expect("build together speech handle");

        let response = AudioCapability::text_to_speech(
            &handle,
            crate::types::TtsRequest::new("hello from together".to_string())
                .with_model("cartesia/sonic-2".to_string())
                .with_voice("alloy".to_string())
                .with_format("mp3".to_string()),
        )
        .await
        .expect("together speech ok");

        assert_eq!(response.audio_data, vec![1, 2, 3, 4]);

        let req = together_transport
            .take()
            .expect("captured together speech request");
        assert!(global_transport.take().is_none());
        assert_eq!(
            req.headers
                .get(AUTHORIZATION)
                .and_then(|value| value.to_str().ok()),
            Some("Bearer ctx-key")
        );
        assert_eq!(req.url, "https://example.com/together/v1/audio/speech");
        assert_eq!(req.body["model"], serde_json::json!("cartesia/sonic-2"));
        assert_eq!(req.body["input"], serde_json::json!("hello from together"));
        assert_eq!(req.body["voice"], serde_json::json!("alloy"));
        assert_eq!(req.body["response_format"], serde_json::json!("mp3"));
    }

    #[tokio::test]
    async fn together_registry_transcription_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = MultipartJsonSuccessTransport::new(serde_json::json!({
            "text": "hello from global",
            "language": "en"
        }));
        let together_transport = MultipartJsonSuccessTransport::new(serde_json::json!({
            "text": "hello from together",
            "language": "en"
        }));
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "together".to_string(),
            Arc::new(OpenAICompatibleProviderFactory::new("together".to_string()))
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "together",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/together/v1")
                    .fetch(Arc::new(together_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .transcription_model("together:openai/whisper-large-v3")
            .expect("build together transcription handle");

        let mut request = crate::types::SttRequest::from_audio(b"abc".to_vec());
        request.model = Some("openai/whisper-large-v3".to_string());
        request = request.with_media_type("audio/mpeg".to_string());

        let response = handle
            .speech_to_text(request)
            .await
            .expect("together transcription ok");

        assert_eq!(response.text, "hello from together");
        assert_eq!(response.language.as_deref(), Some("en"));

        let req = together_transport
            .take_multipart()
            .expect("captured together transcription request");
        assert!(global_transport.take_multipart().is_none());
        assert_eq!(
            req.headers
                .get(AUTHORIZATION)
                .and_then(|value| value.to_str().ok()),
            Some("Bearer ctx-key")
        );
        assert_eq!(
            req.url,
            "https://example.com/together/v1/audio/transcriptions"
        );
        assert!(
            req.headers
                .get(CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.starts_with("multipart/form-data; boundary="))
        );

        let body_text = String::from_utf8(req.body.clone()).expect("multipart body utf8");
        assert!(body_text.contains("name=\"model\""));
        assert!(body_text.contains("openai/whisper-large-v3"));
        assert!(body_text.contains("name=\"response_format\""));
        assert!(body_text.contains("json"));
        assert!(body_text.contains("name=\"file\"; filename=\"audio.mp3\""));
        assert!(body_text.contains("Content-Type: audio/mpeg"));
        assert!(body_text.contains("abc"));
    }

    #[tokio::test]
    async fn openai_compatible_factory_siliconflow_supports_native_transcription_family_path() {
        let _lock = lock_env();

        let factory = OpenAICompatibleProviderFactory::new("siliconflow".to_string());
        let ctx = BuildContext {
            provider_id: Some("siliconflow".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/v1/".to_string()),
            ..Default::default()
        };

        let model = factory
            .transcription_model_family_with_ctx("FunAudioLLM/SenseVoiceSmall", &ctx)
            .await
            .expect("build native transcription-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "siliconflow"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "FunAudioLLM/SenseVoiceSmall"
        );
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model.as_ref()),
            crate::traits::ModelSpecVersion::V1
        );
    }

    #[tokio::test]
    async fn openai_compatible_factory_siliconflow_supports_native_speech_family_path() {
        let _lock = lock_env();

        let factory = OpenAICompatibleProviderFactory::new("siliconflow".to_string());
        let ctx = BuildContext {
            provider_id: Some("siliconflow".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/v1/".to_string()),
            ..Default::default()
        };

        let model = factory
            .speech_model_family_with_ctx("FunAudioLLM/CosyVoice2-0.5B", &ctx)
            .await
            .expect("build native speech-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "siliconflow"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "FunAudioLLM/CosyVoice2-0.5B"
        );
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model.as_ref()),
            crate::traits::ModelSpecVersion::V1
        );
    }

    #[tokio::test]
    async fn siliconflow_registry_speech_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = BytesSuccessTransport::new(vec![9, 9, 9], "audio/mpeg");
        let siliconflow_transport = BytesSuccessTransport::new(vec![1, 2, 3, 4], "audio/mpeg");
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "siliconflow".to_string(),
            Arc::new(OpenAICompatibleProviderFactory::new(
                "siliconflow".to_string(),
            )) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "siliconflow",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/siliconflow/v1")
                    .fetch(Arc::new(siliconflow_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .speech_model("siliconflow:FunAudioLLM/CosyVoice2-0.5B")
            .expect("build siliconflow speech handle");

        let response = AudioCapability::text_to_speech(
            &handle,
            crate::types::TtsRequest::new("hello from siliconflow".to_string())
                .with_model("FunAudioLLM/CosyVoice2-0.5B".to_string())
                .with_voice("FunAudioLLM/CosyVoice2-0.5B:diana".to_string())
                .with_format("mp3".to_string()),
        )
        .await
        .expect("siliconflow speech ok");

        assert_eq!(response.audio_data, vec![1, 2, 3, 4]);

        let req = siliconflow_transport
            .take()
            .expect("captured siliconflow speech request");
        assert!(global_transport.take().is_none());
        assert_eq!(
            req.headers
                .get(AUTHORIZATION)
                .and_then(|value| value.to_str().ok()),
            Some("Bearer ctx-key")
        );
        assert_eq!(req.url, "https://example.com/siliconflow/v1/audio/speech");
        assert_eq!(
            req.body["model"],
            serde_json::json!("FunAudioLLM/CosyVoice2-0.5B")
        );
        assert_eq!(
            req.body["input"],
            serde_json::json!("hello from siliconflow")
        );
        assert_eq!(
            req.body["voice"],
            serde_json::json!("FunAudioLLM/CosyVoice2-0.5B:diana")
        );
        assert_eq!(req.body["response_format"], serde_json::json!("mp3"));
    }

    #[tokio::test]
    async fn siliconflow_registry_transcription_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = MultipartJsonSuccessTransport::new(serde_json::json!({
            "text": "hello from global",
            "language": "zh"
        }));
        let siliconflow_transport = MultipartJsonSuccessTransport::new(serde_json::json!({
            "text": "hello from siliconflow",
            "language": "zh"
        }));
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "siliconflow".to_string(),
            Arc::new(OpenAICompatibleProviderFactory::new(
                "siliconflow".to_string(),
            )) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "siliconflow",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/siliconflow/v1")
                    .fetch(Arc::new(siliconflow_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .transcription_model("siliconflow:FunAudioLLM/SenseVoiceSmall")
            .expect("build siliconflow transcription handle");

        let mut request = crate::types::SttRequest::from_audio(b"abc".to_vec());
        request.model = Some("FunAudioLLM/SenseVoiceSmall".to_string());
        request = request.with_media_type("audio/mpeg".to_string());

        let response = handle
            .speech_to_text(request)
            .await
            .expect("siliconflow transcription ok");

        assert_eq!(response.text, "hello from siliconflow");
        assert_eq!(response.language.as_deref(), Some("zh"));

        let req = siliconflow_transport
            .take_multipart()
            .expect("captured siliconflow transcription request");
        assert!(global_transport.take_multipart().is_none());
        assert_eq!(
            req.headers
                .get(AUTHORIZATION)
                .and_then(|value| value.to_str().ok()),
            Some("Bearer ctx-key")
        );
        assert_eq!(
            req.url,
            "https://example.com/siliconflow/v1/audio/transcriptions"
        );
        assert!(
            req.headers
                .get(CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.starts_with("multipart/form-data; boundary="))
        );

        let body_text = String::from_utf8_lossy(&req.body);
        assert!(body_text.contains("name=\"model\""));
        assert!(body_text.contains("FunAudioLLM/SenseVoiceSmall"));
        assert!(body_text.contains("name=\"response_format\""));
        assert!(body_text.contains("json"));
        assert!(body_text.contains("name=\"file\"; filename=\"audio.mp3\""));
        assert!(body_text.contains("Content-Type: audio/mpeg"));
        assert!(body_text.contains("abc"));
    }

    #[tokio::test]
    async fn openai_compatible_factory_openrouter_does_not_declare_audio_capabilities() {
        let _lock = lock_env();

        let factory = OpenAICompatibleProviderFactory::new("openrouter".to_string());
        let caps = factory.capabilities();

        assert!(!caps.supports("speech"));
        assert!(!caps.supports("transcription"));
        assert!(!caps.supports("audio"));
    }

    #[tokio::test]
    async fn openai_compatible_factory_openrouter_rejects_native_speech_family_path() {
        let _lock = lock_env();

        let factory = OpenAICompatibleProviderFactory::new("openrouter".to_string());
        let ctx = BuildContext {
            provider_id: Some("openrouter".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/v1/".to_string()),
            ..Default::default()
        };

        let result = factory
            .speech_model_family_with_ctx("openai/gpt-4o", &ctx)
            .await;

        assert_unsupported_operation_contains(result, "'speech' family path");
    }

    #[tokio::test]
    async fn openai_compatible_factory_openrouter_rejects_native_transcription_family_path() {
        let _lock = lock_env();

        let factory = OpenAICompatibleProviderFactory::new("openrouter".to_string());
        let ctx = BuildContext {
            provider_id: Some("openrouter".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/v1/".to_string()),
            ..Default::default()
        };

        let result = factory
            .transcription_model_family_with_ctx("openai/gpt-4o", &ctx)
            .await;

        assert_unsupported_operation_contains(result, "'transcription' family path");
    }

    #[tokio::test]
    async fn openai_compatible_factory_perplexity_does_not_declare_non_text_capabilities() {
        let _lock = lock_env();

        let factory = OpenAICompatibleProviderFactory::new("perplexity".to_string());
        let caps = factory.capabilities();

        assert!(!caps.supports("embedding"));
        assert!(!caps.supports("image_generation"));
        assert!(!caps.supports("rerank"));
        assert!(!caps.supports("speech"));
        assert!(!caps.supports("transcription"));
        assert!(!caps.supports("audio"));
    }

    #[tokio::test]
    async fn openai_compatible_factory_perplexity_rejects_non_text_family_paths() {
        let _lock = lock_env();

        let factory = OpenAICompatibleProviderFactory::new("perplexity".to_string());
        let transport = CaptureTransport::default();
        let ctx = BuildContext {
            provider_id: Some("perplexity".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/v1/".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        assert_unsupported_operation_contains(
            factory.embedding_model_family_with_ctx("sonar", &ctx).await,
            "'embedding' family path",
        );
        assert_unsupported_operation_contains(
            factory.image_model_family_with_ctx("sonar", &ctx).await,
            "'image_generation' family path",
        );
        assert_unsupported_operation_contains(
            factory.reranking_model_family_with_ctx("sonar", &ctx).await,
            "'rerank' family path",
        );
        assert_unsupported_operation_contains(
            factory.speech_model_family_with_ctx("sonar", &ctx).await,
            "'speech' family path",
        );
        assert_unsupported_operation_contains(
            factory
                .transcription_model_family_with_ctx("sonar", &ctx)
                .await,
            "'transcription' family path",
        );
        assert_capture_transport_unused(&transport);
    }

    #[tokio::test]
    async fn openrouter_registry_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let openrouter_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "openrouter".to_string(),
            Arc::new(OpenAICompatibleProviderFactory::new(
                "openrouter".to_string(),
            )) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "openrouter",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/openrouter/v1")
                    .fetch(Arc::new(openrouter_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .language_model("openrouter:openai/gpt-4o")
            .expect("build openrouter handle");

        let _ = handle
            .chat_request(
                make_chat_request_with_model("openai/gpt-4o").with_provider_option(
                    "openrouter",
                    serde_json::json!({
                        "transforms": ["middle-out"],
                    }),
                ),
            )
            .await;

        let req = openrouter_transport
            .take()
            .expect("captured openrouter request");
        assert!(global_transport.take().is_none());
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert_eq!(
            req.url,
            "https://example.com/openrouter/v1/chat/completions"
        );
        assert_eq!(req.body["model"], serde_json::json!("openai/gpt-4o"));
        assert_eq!(req.body["transforms"], serde_json::json!(["middle-out"]));
    }

    #[tokio::test]
    async fn openrouter_registry_stream_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let openrouter_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "openrouter".to_string(),
            Arc::new(OpenAICompatibleProviderFactory::new(
                "openrouter".to_string(),
            )) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "openrouter",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/openrouter/v1")
                    .fetch(Arc::new(openrouter_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .language_model("openrouter:openai/gpt-4o")
            .expect("build openrouter handle");

        let _ = handle
            .chat_stream_request(
                make_chat_request_with_model("openai/gpt-4o").with_provider_option(
                    "openrouter",
                    serde_json::json!({
                        "transforms": ["middle-out"],
                    }),
                ),
            )
            .await;

        let req = openrouter_transport
            .take_stream()
            .expect("captured openrouter stream request");
        assert!(global_transport.take().is_none());
        assert!(global_transport.take_stream().is_none());
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert_eq!(
            header_value(&req, "accept"),
            Some("text/event-stream".to_string())
        );
        assert_eq!(
            req.url,
            "https://example.com/openrouter/v1/chat/completions"
        );
        assert_eq!(req.body["model"], serde_json::json!("openai/gpt-4o"));
        assert_eq!(req.body["stream"], serde_json::json!(true));
        assert_eq!(req.body["transforms"], serde_json::json!(["middle-out"]));
    }

    #[tokio::test]
    async fn openrouter_registry_embedding_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let openrouter_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "openrouter".to_string(),
            Arc::new(OpenAICompatibleProviderFactory::new(
                "openrouter".to_string(),
            )) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "openrouter",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/openrouter/v1")
                    .fetch(Arc::new(openrouter_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .embedding_model("openrouter:openai/text-embedding-3-small")
            .expect("build openrouter embedding handle");

        let _ = handle
            .embed_with_config(
                crate::types::EmbeddingRequest::single("hello openrouter embedding")
                    .with_model("openai/text-embedding-3-small")
                    .with_dimensions(512)
                    .with_encoding_format(crate::types::EmbeddingFormat::Base64)
                    .with_user("compat-user-3"),
            )
            .await;

        let req = openrouter_transport
            .take()
            .expect("captured openrouter embedding request");
        assert!(global_transport.take().is_none());
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert_eq!(req.url, "https://example.com/openrouter/v1/embeddings");
        assert_eq!(
            req.body["model"],
            serde_json::json!("openai/text-embedding-3-small")
        );
        assert_eq!(
            req.body["input"],
            serde_json::json!(["hello openrouter embedding"])
        );
        assert_eq!(req.body["dimensions"], serde_json::json!(512));
        assert_eq!(req.body["encoding_format"], serde_json::json!("base64"));
        assert_eq!(req.body["user"], serde_json::json!("compat-user-3"));
    }

    #[tokio::test]
    async fn jina_registry_embedding_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let jina_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "jina".to_string(),
            Arc::new(OpenAICompatibleProviderFactory::new("jina".to_string()))
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "jina",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/jina/v1")
                    .fetch(Arc::new(jina_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .embedding_model("jina:jina-embeddings-v2-base-en")
            .expect("build jina embedding handle");

        let _ = handle
            .embed_with_config(
                crate::types::EmbeddingRequest::single("hello jina embedding")
                    .with_model("jina-embeddings-v2-base-en")
                    .with_dimensions(768)
                    .with_encoding_format(crate::types::EmbeddingFormat::Float)
                    .with_user("compat-user-5"),
            )
            .await;

        let req = jina_transport
            .take()
            .expect("captured jina embedding request");
        assert!(global_transport.take().is_none());
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert_eq!(req.url, "https://example.com/jina/v1/embeddings");
        assert_eq!(
            req.body["model"],
            serde_json::json!("jina-embeddings-v2-base-en")
        );
        assert_eq!(
            req.body["input"],
            serde_json::json!(["hello jina embedding"])
        );
        assert_eq!(req.body["dimensions"], serde_json::json!(768));
        assert_eq!(req.body["encoding_format"], serde_json::json!("float"));
        assert_eq!(req.body["user"], serde_json::json!("compat-user-5"));
    }

    #[tokio::test]
    async fn voyageai_registry_embedding_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let voyageai_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "voyageai".to_string(),
            Arc::new(OpenAICompatibleProviderFactory::new("voyageai".to_string()))
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "voyageai",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/voyageai/v1")
                    .fetch(Arc::new(voyageai_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .embedding_model("voyageai:voyage-3")
            .expect("build voyageai embedding handle");

        let _ = handle
            .embed_with_config(
                crate::types::EmbeddingRequest::single("hello voyage embedding")
                    .with_model("voyage-3")
                    .with_dimensions(1024)
                    .with_encoding_format(crate::types::EmbeddingFormat::Base64)
                    .with_user("compat-user-6"),
            )
            .await;

        let req = voyageai_transport
            .take()
            .expect("captured voyageai embedding request");
        assert!(global_transport.take().is_none());
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert_eq!(req.url, "https://example.com/voyageai/v1/embeddings");
        assert_eq!(req.body["model"], serde_json::json!("voyage-3"));
        assert_eq!(
            req.body["input"],
            serde_json::json!(["hello voyage embedding"])
        );
        assert_eq!(req.body["dimensions"], serde_json::json!(1024));
        assert_eq!(req.body["encoding_format"], serde_json::json!("base64"));
        assert_eq!(req.body["user"], serde_json::json!("compat-user-6"));
    }

    #[tokio::test]
    async fn infini_registry_embedding_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let infini_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "infini".to_string(),
            Arc::new(OpenAICompatibleProviderFactory::new("infini".to_string()))
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "infini",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/infini/maas/v1")
                    .fetch(Arc::new(infini_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .embedding_model("infini:text-embedding-3-small")
            .expect("build infini embedding handle");

        let _ = handle
            .embed_with_config(
                crate::types::EmbeddingRequest::single("hello infini embedding")
                    .with_model("text-embedding-3-small")
                    .with_dimensions(512)
                    .with_encoding_format(crate::types::EmbeddingFormat::Float)
                    .with_user("compat-user-7"),
            )
            .await;

        let req = infini_transport
            .take()
            .expect("captured infini embedding request");
        assert!(global_transport.take().is_none());
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert_eq!(req.url, "https://example.com/infini/maas/v1/embeddings");
        assert_eq!(
            req.body["model"],
            serde_json::json!("text-embedding-3-small")
        );
        assert_eq!(
            req.body["input"],
            serde_json::json!(["hello infini embedding"])
        );
        assert_eq!(req.body["dimensions"], serde_json::json!(512));
        assert_eq!(req.body["encoding_format"], serde_json::json!("float"));
        assert_eq!(req.body["user"], serde_json::json!("compat-user-7"));
    }

    #[tokio::test]
    async fn mistral_registry_embedding_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let mistral_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "mistral".to_string(),
            Arc::new(OpenAICompatibleProviderFactory::new("mistral".to_string()))
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "mistral",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/mistral/v1")
                    .fetch(Arc::new(mistral_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .embedding_model("mistral:mistral-embed")
            .expect("build mistral embedding handle");

        let _ = handle
            .embed_with_config(
                crate::types::EmbeddingRequest::new(vec!["hello mistral".to_string()])
                    .with_model("mistral-embed"),
            )
            .await;

        let req = mistral_transport
            .take()
            .expect("captured mistral embedding request");
        assert!(global_transport.take().is_none());
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert_eq!(req.url, "https://example.com/mistral/v1/embeddings");
        assert_eq!(req.body["model"], serde_json::json!("mistral-embed"));
        assert_eq!(req.body["input"], serde_json::json!(["hello mistral"]));
    }

    #[tokio::test]
    async fn fireworks_registry_embedding_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let fireworks_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "fireworks".to_string(),
            Arc::new(OpenAICompatibleProviderFactory::new(
                "fireworks".to_string(),
            )) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "fireworks",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/fireworks/inference/v1")
                    .fetch(Arc::new(fireworks_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .embedding_model("fireworks:nomic-ai/nomic-embed-text-v1.5")
            .expect("build fireworks embedding handle");

        let _ = handle
            .embed_with_config(
                crate::types::EmbeddingRequest::single("hello fireworks embedding")
                    .with_model("nomic-ai/nomic-embed-text-v1.5")
                    .with_dimensions(256)
                    .with_encoding_format(crate::types::EmbeddingFormat::Base64)
                    .with_user("compat-user-1"),
            )
            .await;

        let req = fireworks_transport
            .take()
            .expect("captured fireworks embedding request");
        assert!(global_transport.take().is_none());
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert_eq!(
            req.url,
            "https://example.com/fireworks/inference/v1/embeddings"
        );
        assert_eq!(
            req.body["model"],
            serde_json::json!("nomic-ai/nomic-embed-text-v1.5")
        );
        assert_eq!(
            req.body["input"],
            serde_json::json!(["hello fireworks embedding"])
        );
        assert_eq!(req.body["dimensions"], serde_json::json!(256));
        assert_eq!(req.body["encoding_format"], serde_json::json!("base64"));
        assert_eq!(req.body["user"], serde_json::json!("compat-user-1"));
    }

    #[tokio::test]
    async fn siliconflow_registry_embedding_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let siliconflow_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "siliconflow".to_string(),
            Arc::new(OpenAICompatibleProviderFactory::new(
                "siliconflow".to_string(),
            )) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "siliconflow",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/siliconflow/v1")
                    .fetch(Arc::new(siliconflow_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .embedding_model("siliconflow:BAAI/bge-large-zh-v1.5")
            .expect("build siliconflow embedding handle");

        let _ = handle
            .embed_with_config(
                crate::types::EmbeddingRequest::single("hello siliconflow embedding")
                    .with_model("BAAI/bge-large-zh-v1.5")
                    .with_dimensions(768)
                    .with_encoding_format(crate::types::EmbeddingFormat::Float)
                    .with_user("compat-user-2"),
            )
            .await;

        let req = siliconflow_transport
            .take()
            .expect("captured siliconflow embedding request");
        assert!(global_transport.take().is_none());
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert_eq!(req.url, "https://example.com/siliconflow/v1/embeddings");
        assert_eq!(
            req.body["model"],
            serde_json::json!("BAAI/bge-large-zh-v1.5")
        );
        assert_eq!(
            req.body["input"],
            serde_json::json!(["hello siliconflow embedding"])
        );
        assert_eq!(req.body["dimensions"], serde_json::json!(768));
        assert_eq!(req.body["encoding_format"], serde_json::json!("float"));
        assert_eq!(req.body["user"], serde_json::json!("compat-user-2"));
    }

    #[tokio::test]
    async fn together_registry_embedding_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let together_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "together".to_string(),
            Arc::new(OpenAICompatibleProviderFactory::new("together".to_string()))
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "together",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/together/v1")
                    .fetch(Arc::new(together_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .embedding_model("together:togethercomputer/m2-bert-80M-8k-retrieval")
            .expect("build together embedding handle");

        let _ = handle
            .embed_with_config(
                crate::types::EmbeddingRequest::single("hello together embedding")
                    .with_model("togethercomputer/m2-bert-80M-8k-retrieval")
                    .with_dimensions(384)
                    .with_encoding_format(crate::types::EmbeddingFormat::Float)
                    .with_user("compat-user-4"),
            )
            .await;

        let req = together_transport
            .take()
            .expect("captured together embedding request");
        assert!(global_transport.take().is_none());
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert_eq!(req.url, "https://example.com/together/v1/embeddings");
        assert_eq!(
            req.body["model"],
            serde_json::json!("togethercomputer/m2-bert-80M-8k-retrieval")
        );
        assert_eq!(
            req.body["input"],
            serde_json::json!(["hello together embedding"])
        );
        assert_eq!(req.body["dimensions"], serde_json::json!(384));
        assert_eq!(req.body["encoding_format"], serde_json::json!("float"));
        assert_eq!(req.body["user"], serde_json::json!("compat-user-4"));
    }

    #[tokio::test]
    async fn openrouter_registry_override_chat_response_metadata_preserves_vendor_namespace() {
        let _lock = lock_env();

        let response_json = serde_json::json!({
            "id": "chatcmpl-openrouter-test",
            "object": "chat.completion",
            "created": 1_718_345_013,
            "model": "openai/gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "hello from openrouter chat"
                    },
                    "finish_reason": "stop",
                    "logprobs": {
                        "content": [
                            {
                                "token": "hello",
                                "logprob": -0.1,
                                "bytes": [104, 101, 108, 108, 111],
                                "top_logprobs": []
                            }
                        ]
                    }
                }
            ],
            "sources": [
                {
                    "id": "src_1",
                    "source_type": "url",
                    "url": "https://openrouter.ai/docs",
                    "title": "OpenRouter Docs",
                    "provider_metadata": {
                        "openrouter": {
                            "fileId": "file_123",
                            "containerId": "container_456",
                            "index": 1
                        }
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 3,
                "total_tokens": 14
            }
        });

        let global_transport = CaptureTransport::default();
        let openrouter_transport = JsonSuccessTransport::new(response_json);
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "openrouter".to_string(),
            Arc::new(OpenAICompatibleProviderFactory::new(
                "openrouter".to_string(),
            )) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "openrouter",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/openrouter/v1")
                    .fetch(Arc::new(openrouter_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let response = registry
            .language_model("openrouter:openai/gpt-4o")
            .expect("build openrouter handle")
            .chat_request(
                make_chat_request_with_model("openai/gpt-4o")
                    .with_provider_option("openrouter", serde_json::json!({ "logprobs": 3 })),
            )
            .await
            .expect("registry response ok");

        let root = response
            .provider_metadata
            .as_ref()
            .expect("registry provider metadata");
        assert!(root.get("openrouter").is_some());
        assert!(root.get("openai_compatible").is_none());

        let metadata = response.openrouter_metadata().expect("openrouter metadata");
        assert_eq!(response.content_text(), Some("hello from openrouter chat"));
        assert_eq!(
            response.usage.as_ref().map(|usage| usage.total_tokens),
            Some(14)
        );
        assert_eq!(metadata.sources.as_ref().map(Vec::len), Some(1));
        assert_eq!(
            metadata
                .logprobs
                .as_ref()
                .and_then(|value| value.as_array().map(Vec::len)),
            Some(1)
        );

        let req = openrouter_transport.take().expect("captured request");
        assert!(global_transport.take().is_none());
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert_eq!(
            req.url,
            "https://example.com/openrouter/v1/chat/completions"
        );
    }

    #[tokio::test]
    async fn openrouter_registry_override_stream_end_metadata_preserves_vendor_namespace() {
        let _lock = lock_env();

        let stream_body = br#"data: {"id":"1","model":"openai/gpt-4o","created":1718345013,"sources":[{"id":"src_1","source_type":"url","url":"https://openrouter.ai/docs","title":"OpenRouter Docs","provider_metadata":{"openrouter":{"fileId":"file_123","containerId":"container_456","index":1}}}],"choices":[{"index":0,"delta":{"content":"hello","role":"assistant"},"logprobs":{"content":[{"token":"hello","logprob":-0.1,"bytes":[104,101,108,108,111],"top_logprobs":[]}]},"finish_reason":null}]}

data: {"id":"1","model":"openai/gpt-4o","created":1718345013,"choices":[{"index":0,"delta":{"content":" from openrouter","role":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":3,"total_tokens":14}}

data: [DONE]

"#
        .to_vec();

        let global_transport = CaptureTransport::default();
        let openrouter_transport = SseSuccessTransport::new(stream_body);
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "openrouter".to_string(),
            Arc::new(OpenAICompatibleProviderFactory::new(
                "openrouter".to_string(),
            )) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "openrouter",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/openrouter/v1")
                    .fetch(Arc::new(openrouter_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let mut stream = registry
            .language_model("openrouter:openai/gpt-4o")
            .expect("build openrouter handle")
            .chat_stream_request(
                make_chat_request_with_model("openai/gpt-4o")
                    .with_provider_option("openrouter", serde_json::json!({ "logprobs": 3 })),
            )
            .await
            .expect("registry stream ok");

        use futures::StreamExt;
        let mut stream_end = None;
        while let Some(event) = stream.next().await {
            if let Ok(crate::streaming::ChatStreamEvent::StreamEnd { response }) = event {
                stream_end = Some(response);
                break;
            }
        }

        let response = stream_end.expect("registry stream end");
        let root = response
            .provider_metadata
            .as_ref()
            .expect("registry provider metadata");
        assert!(root.get("openrouter").is_some());
        assert!(root.get("openai_compatible").is_none());

        let metadata = response.openrouter_metadata().expect("openrouter metadata");
        assert_eq!(response.content_text(), Some("hello from openrouter"));
        assert_eq!(
            response.usage.as_ref().map(|usage| usage.total_tokens),
            Some(14)
        );
        assert_eq!(metadata.sources.as_ref().map(Vec::len), Some(1));
        assert_eq!(
            metadata
                .logprobs
                .as_ref()
                .and_then(|value| value.as_array().map(Vec::len)),
            Some(1)
        );

        let req = openrouter_transport
            .take_stream()
            .expect("captured stream request");
        assert!(global_transport.take().is_none());
        assert!(global_transport.take_stream().is_none());
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert_eq!(
            req.url,
            "https://example.com/openrouter/v1/chat/completions"
        );
    }

    #[tokio::test]
    async fn perplexity_registry_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let perplexity_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "perplexity".to_string(),
            Arc::new(OpenAICompatibleProviderFactory::new(
                "perplexity".to_string(),
            )) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "perplexity",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/perplexity")
                    .fetch(Arc::new(perplexity_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .language_model("perplexity:sonar")
            .expect("build perplexity handle");

        let _ = handle
            .chat_request(make_chat_request_with_model("sonar").with_provider_option(
                "perplexity",
                serde_json::json!({
                    "search_mode": "academic",
                    "someVendorParam": true,
                }),
            ))
            .await;

        let req = perplexity_transport
            .take()
            .expect("captured perplexity request");
        assert!(global_transport.take().is_none());
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert_eq!(req.url, "https://example.com/perplexity/chat/completions");
        assert_eq!(req.body["model"], serde_json::json!("sonar"));
        assert_eq!(req.body["search_mode"], serde_json::json!("academic"));
        assert_eq!(req.body["someVendorParam"], serde_json::json!(true));
    }

    #[tokio::test]
    async fn perplexity_registry_stream_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let perplexity_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "perplexity".to_string(),
            Arc::new(OpenAICompatibleProviderFactory::new(
                "perplexity".to_string(),
            )) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "perplexity",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/perplexity")
                    .fetch(Arc::new(perplexity_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .language_model("perplexity:sonar")
            .expect("build perplexity handle");

        let _ = handle
            .chat_stream_request(make_chat_request_with_model("sonar").with_provider_option(
                "perplexity",
                serde_json::json!({
                    "search_mode": "academic",
                    "someVendorParam": true,
                }),
            ))
            .await;

        let req = perplexity_transport
            .take_stream()
            .expect("captured perplexity stream request");
        assert!(global_transport.take().is_none());
        assert!(global_transport.take_stream().is_none());
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert_eq!(
            header_value(&req, "accept"),
            Some("text/event-stream".to_string())
        );
        assert_eq!(req.url, "https://example.com/perplexity/chat/completions");
        assert_eq!(req.body["model"], serde_json::json!("sonar"));
        assert_eq!(req.body["stream"], serde_json::json!(true));
        assert_eq!(req.body["search_mode"], serde_json::json!("academic"));
        assert_eq!(req.body["someVendorParam"], serde_json::json!(true));
    }

    #[tokio::test]
    async fn perplexity_registry_override_stream_end_metadata_preserves_vendor_namespace() {
        let _lock = lock_env();

        let stream_body = br#"data: {"id":"1","model":"sonar","created":1718345013,"citations":["https://example.com/rust"],"choices":[{"index":0,"delta":{"content":"Rust","role":"assistant"},"finish_reason":null}]}

data: {"id":"1","model":"sonar","created":1718345013,"choices":[{"index":0,"delta":{"content":" ecosystem","role":null},"finish_reason":"stop"}],"images":[{"image_url":"https://images.example.com/rust.png","origin_url":"https://example.com/rust","height":900,"width":1600}],"usage":{"prompt_tokens":11,"completion_tokens":17,"total_tokens":28,"citation_tokens":7,"num_search_queries":2,"reasoning_tokens":3}}

data: [DONE]

"#
        .to_vec();

        let global_transport = CaptureTransport::default();
        let perplexity_transport = SseSuccessTransport::new(stream_body);
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "perplexity".to_string(),
            Arc::new(OpenAICompatibleProviderFactory::new(
                "perplexity".to_string(),
            )) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "perplexity",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/perplexity")
                    .fetch(Arc::new(perplexity_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let mut stream =
            registry
                .language_model("perplexity:sonar")
                .expect("build perplexity handle")
                .chat_stream_request(make_chat_request_with_model("sonar").with_provider_option(
                    "perplexity",
                    serde_json::json!({ "return_images": true }),
                ))
                .await
                .expect("registry stream ok");

        use futures::StreamExt;
        let mut stream_end = None;
        while let Some(event) = stream.next().await {
            if let Ok(crate::streaming::ChatStreamEvent::StreamEnd { response }) = event {
                stream_end = Some(response);
                break;
            }
        }

        let response = stream_end.expect("registry stream end");
        let root = response
            .provider_metadata
            .as_ref()
            .expect("registry provider metadata");
        assert!(root.get("perplexity").is_some());
        assert!(root.get("openai_compatible").is_none());

        let metadata = response.perplexity_metadata().expect("perplexity metadata");
        assert_eq!(response.content_text(), Some("Rust ecosystem"));
        assert_eq!(
            response.usage.as_ref().map(|usage| usage.total_tokens),
            Some(28)
        );
        assert_eq!(
            metadata.citations.as_ref(),
            Some(&vec!["https://example.com/rust".to_string()])
        );
        assert_eq!(metadata.images.as_ref().map(Vec::len), Some(1));
        assert_eq!(
            metadata
                .usage
                .as_ref()
                .and_then(|usage| usage.citation_tokens),
            Some(7)
        );

        let req = perplexity_transport
            .take_stream()
            .expect("captured stream request");
        assert!(global_transport.take().is_none());
        assert!(global_transport.take_stream().is_none());
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert_eq!(req.url, "https://example.com/perplexity/chat/completions");
    }

    #[tokio::test]
    async fn perplexity_registry_override_chat_response_metadata_preserves_vendor_namespace() {
        let _lock = lock_env();

        let response_json = serde_json::json!({
            "id": "chatcmpl-perplexity-test",
            "object": "chat.completion",
            "created": 1_718_345_013,
            "model": "sonar",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Rust async tooling kept improving across the ecosystem."
                    },
                    "finish_reason": "stop"
                }
            ],
            "citations": ["https://example.com/rust"],
            "images": [
                {
                    "image_url": "https://images.example.com/rust.png",
                    "origin_url": "https://example.com/rust",
                    "height": 900,
                    "width": 1600
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 17,
                "total_tokens": 28,
                "citation_tokens": 7,
                "num_search_queries": 2,
                "reasoning_tokens": 3
            }
        });

        let global_transport = CaptureTransport::default();
        let perplexity_transport = JsonSuccessTransport::new(response_json);
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "perplexity".to_string(),
            Arc::new(OpenAICompatibleProviderFactory::new(
                "perplexity".to_string(),
            )) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "perplexity",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/perplexity")
                    .fetch(Arc::new(perplexity_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let response =
            registry
                .language_model("perplexity:sonar")
                .expect("build perplexity handle")
                .chat_request(make_chat_request_with_model("sonar").with_provider_option(
                    "perplexity",
                    serde_json::json!({ "return_images": true }),
                ))
                .await
                .expect("registry response ok");

        let root = response
            .provider_metadata
            .as_ref()
            .expect("registry provider metadata");
        assert!(root.get("perplexity").is_some());
        assert!(root.get("openai_compatible").is_none());

        let metadata = response.perplexity_metadata().expect("perplexity metadata");
        assert_eq!(
            response.content_text(),
            Some("Rust async tooling kept improving across the ecosystem.")
        );
        assert_eq!(
            response.usage.as_ref().map(|usage| usage.total_tokens),
            Some(28)
        );
        assert_eq!(
            metadata.citations.as_ref(),
            Some(&vec!["https://example.com/rust".to_string()])
        );
        assert_eq!(metadata.images.as_ref().map(Vec::len), Some(1));
        assert_eq!(
            metadata
                .usage
                .as_ref()
                .and_then(|usage| usage.citation_tokens),
            Some(7)
        );

        let req = perplexity_transport.take().expect("captured request");
        assert!(global_transport.take().is_none());
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert_eq!(req.url, "https://example.com/perplexity/chat/completions");
    }

    #[tokio::test]
    async fn openai_compatible_factory_xai_does_not_declare_audio_capabilities() {
        let _lock = lock_env();

        let factory = OpenAICompatibleProviderFactory::new("xai".to_string());
        let caps = factory.capabilities();

        assert!(!caps.supports("speech"));
        assert!(!caps.supports("transcription"));
        assert!(!caps.supports("audio"));
    }

    #[tokio::test]
    async fn openai_compatible_factory_fireworks_declares_transcription_capability() {
        let _lock = lock_env();

        let factory = OpenAICompatibleProviderFactory::new("fireworks".to_string());
        let caps = factory.capabilities();

        assert!(!caps.supports("speech"));
        assert!(caps.supports("transcription"));
        assert!(caps.supports("audio"));
    }

    #[tokio::test]
    async fn openai_compatible_factory_fireworks_supports_native_transcription_family_path() {
        let _lock = lock_env();

        let factory = OpenAICompatibleProviderFactory::new("fireworks".to_string());
        let ctx = BuildContext {
            provider_id: Some("fireworks".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/v1/".to_string()),
            ..Default::default()
        };

        let model = factory
            .transcription_model_family_with_ctx("whisper-v3", &ctx)
            .await
            .expect("build fireworks transcription-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "fireworks"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "whisper-v3"
        );
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model.as_ref()),
            crate::traits::ModelSpecVersion::V1
        );
    }

    #[tokio::test]
    async fn openai_compatible_factory_fireworks_rejects_native_speech_family_path() {
        let _lock = lock_env();

        let factory = OpenAICompatibleProviderFactory::new("fireworks".to_string());
        let ctx = BuildContext {
            provider_id: Some("fireworks".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/v1/".to_string()),
            ..Default::default()
        };

        let result = factory
            .speech_model_family_with_ctx("fireworks-tts-unavailable", &ctx)
            .await;

        match result {
            Err(LlmError::UnsupportedOperation(_)) => {}
            Err(other) => panic!("expected UnsupportedOperation, got: {other:?}"),
            Ok(_) => panic!("fireworks should not expose speech family path"),
        }
    }

    #[tokio::test]
    async fn fireworks_registry_transcription_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();
        use crate::traits::AudioCapability;

        let global_transport = MultipartJsonSuccessTransport::new(serde_json::json!({
            "text": "hello from global",
            "language": "en"
        }));
        let fireworks_transport = MultipartJsonSuccessTransport::new(serde_json::json!({
            "text": "hello from fireworks",
            "language": "en"
        }));

        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "fireworks".to_string(),
            Arc::new(OpenAICompatibleProviderFactory::new(
                "fireworks".to_string(),
            )) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "fireworks",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/fireworks-audio/v1")
                    .fetch(Arc::new(fireworks_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .transcription_model("fireworks:whisper-v3")
            .expect("build fireworks transcription handle");

        let mut request = crate::types::SttRequest::from_audio(b"abc".to_vec());
        request.model = Some("whisper-v3".to_string());
        request = request.with_media_type("audio/mpeg".to_string());

        let response = handle
            .speech_to_text(request)
            .await
            .expect("fireworks transcription ok");

        assert_eq!(response.text, "hello from fireworks");
        assert_eq!(response.language.as_deref(), Some("en"));

        let req = fireworks_transport
            .take_multipart()
            .expect("captured fireworks multipart request");
        assert!(global_transport.take_multipart().is_none());
        assert_eq!(
            req.headers
                .get(AUTHORIZATION)
                .and_then(|value| value.to_str().ok()),
            Some("Bearer ctx-key")
        );
        assert_eq!(
            req.url,
            "https://example.com/fireworks-audio/v1/audio/transcriptions"
        );
        assert!(
            req.headers
                .get(CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.starts_with("multipart/form-data; boundary="))
        );

        let body_text = String::from_utf8_lossy(&req.body);
        assert!(body_text.contains("name=\"model\""));
        assert!(body_text.contains("whisper-v3"));
        assert!(body_text.contains("name=\"response_format\""));
        assert!(body_text.contains("json"));
        assert!(body_text.contains("name=\"file\"; filename=\"audio.mp3\""));
        assert!(body_text.contains("Content-Type: audio/mpeg"));
        assert!(body_text.contains("abc"));
    }

    #[test]
    fn openai_compatible_factory_source_declares_native_rerank_family_override() {
        let source = include_str!("openai_compatible.rs");

        assert!(source.contains("async fn reranking_model_family_with_ctx("));
    }

    #[test]
    fn openai_compatible_factory_source_declares_native_image_family_override() {
        let source = include_str!("openai_compatible.rs");

        assert!(source.contains("async fn image_model_family_with_ctx("));
    }

    #[test]
    fn openai_compatible_factory_source_declares_native_audio_family_overrides() {
        let source = include_str!("openai_compatible.rs");

        assert!(source.contains("async fn speech_model_family_with_ctx("));
        assert!(source.contains("async fn transcription_model_family_with_ctx("));
    }

    #[test]
    fn openai_compatible_factory_source_routes_known_provider_construction_through_builder() {
        let source = include_str!("openai_compatible.rs");

        assert!(source.contains("OpenAiCompatibleBuilder::new("));
        assert!(source.contains(".with_http_config("));
        assert!(source.contains(".with_model_middlewares("));
    }

    #[tokio::test]
    async fn openai_compatible_builder_config_registry_chat_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client =
            siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleBuilder::new(
                siumai_provider_openai_compatible::builder::BuilderBase::default(),
                "deepseek",
            )
            .api_key("ctx-key")
            .base_url("https://example.com/custom/v1/")
            .model("deepseek-chat")
            .temperature(0.4)
            .max_tokens(256)
            .top_p(0.9)
            .stop(vec!["END"])
            .seed(7)
            .reasoning(true)
            .reasoning_budget(2048)
            .fetch(Arc::new(builder_transport.clone()))
            .build()
            .await
            .expect("build builder client");

        let provider =
            siumai_provider_openai_compatible::providers::openai_compatible::get_provider_config(
                "deepseek",
            )
            .expect("provider config");
        let adapter = Arc::new(
            siumai_provider_openai_compatible::providers::openai_compatible::ConfigurableAdapter::new(
                provider,
            ),
        );
        let config_client =
            siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleClient::from_config(
                siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleConfig::new(
                    "deepseek",
                    "ctx-key",
                    "https://example.com/custom/v1/",
                    adapter,
                )
                .with_model("deepseek-chat")
                .with_temperature(0.4)
                .with_max_tokens(256)
                .with_top_p(0.9)
                .with_stop_sequences(vec!["END".to_string()])
                .with_seed(7)
                .with_reasoning(true)
                .with_reasoning_budget(2048)
                .with_http_transport(Arc::new(config_transport.clone())),
            )
            .await
            .expect("build config client");

        let mut common_params = crate::types::CommonParams::default();
        common_params.model = "deepseek-chat".to_string();
        common_params.temperature = Some(0.4);
        common_params.max_tokens = Some(256);
        common_params.top_p = Some(0.9);
        common_params.stop_sequences = Some(vec!["END".to_string()]);
        common_params.seed = Some(7);

        let factory = OpenAICompatibleProviderFactory::new("deepseek".to_string());
        let registry_client = factory
            .language_model_with_ctx(
                "deepseek-chat",
                &BuildContext {
                    provider_id: Some("deepseek".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom/v1/".to_string()),
                    common_params: Some(common_params),
                    reasoning_enabled: Some(true),
                    reasoning_budget: Some(2048),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model("deepseek-chat").with_provider_option(
            "deepseek",
            serde_json::json!({
                "foo": "bar"
            }),
        );

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.body["temperature"], serde_json::json!(0.4));
        assert_eq!(builder_req.body["max_tokens"], serde_json::json!(256));
        assert_eq!(builder_req.body["top_p"], serde_json::json!(0.9));
        assert_eq!(builder_req.body["stop"], serde_json::json!(["END"]));
        assert_eq!(builder_req.body["seed"], serde_json::json!(7));
        assert_eq!(
            builder_req.body["enable_reasoning"],
            serde_json::json!(true)
        );
        assert_eq!(
            builder_req.body["reasoning_budget"],
            serde_json::json!(2048)
        );
        assert_eq!(builder_req.body["foo"], serde_json::json!("bar"));
    }

    #[tokio::test]
    async fn openrouter_builder_config_registry_chat_request_respect_explicit_request_model() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();
        let default_model = "openai/gpt-4o";
        let request_model = "openai/gpt-4.1";

        let builder_client =
            siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleBuilder::new(
                siumai_provider_openai_compatible::builder::BuilderBase::default(),
                "openrouter",
            )
            .api_key("ctx-key")
            .model(default_model)
            .reasoning(true)
            .reasoning_budget(2048)
            .fetch(Arc::new(builder_transport.clone()))
            .build()
            .await
            .expect("build builder client");

        let provider =
            siumai_provider_openai_compatible::providers::openai_compatible::get_provider_config(
                "openrouter",
            )
            .expect("provider config");
        let adapter = Arc::new(
            siumai_provider_openai_compatible::providers::openai_compatible::ConfigurableAdapter::new(
                provider,
            ),
        );
        let config_client =
            siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleClient::from_config(
                siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleConfig::new(
                    "openrouter",
                    "ctx-key",
                    "https://openrouter.ai/api/v1",
                    adapter,
                )
                .with_model(default_model)
                .with_reasoning(true)
                .with_reasoning_budget(2048)
                .with_http_transport(Arc::new(config_transport.clone())),
            )
            .await
            .expect("build config client");

        let factory = OpenAICompatibleProviderFactory::new("openrouter".to_string());
        let registry_client = factory
            .language_model_with_ctx(
                default_model,
                &BuildContext {
                    provider_id: Some("openrouter".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    reasoning_enabled: Some(true),
                    reasoning_budget: Some(2048),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model(request_model).with_provider_option(
            "openrouter",
            serde_json::json!({
                "transforms": ["middle-out"],
                "someVendorParam": true
            }),
        );

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.body["model"], serde_json::json!(request_model));
        assert_eq!(
            builder_req.body["transforms"],
            serde_json::json!(["middle-out"])
        );
        assert_eq!(builder_req.body["someVendorParam"], serde_json::json!(true));
        assert_eq!(
            builder_req.body["enable_reasoning"],
            serde_json::json!(true)
        );
        assert_eq!(
            builder_req.body["reasoning_budget"],
            serde_json::json!(2048)
        );
    }

    #[tokio::test]
    async fn perplexity_builder_config_registry_chat_request_respect_explicit_request_model() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();
        let default_model = "sonar";
        let request_model = "sonar-pro";

        let builder_client =
            siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleBuilder::new(
                siumai_provider_openai_compatible::builder::BuilderBase::default(),
                "perplexity",
            )
            .api_key("ctx-key")
            .model(default_model)
            .fetch(Arc::new(builder_transport.clone()))
            .build()
            .await
            .expect("build builder client");

        let provider =
            siumai_provider_openai_compatible::providers::openai_compatible::get_provider_config(
                "perplexity",
            )
            .expect("provider config");
        let adapter = Arc::new(
            siumai_provider_openai_compatible::providers::openai_compatible::ConfigurableAdapter::new(
                provider,
            ),
        );
        let config_client =
            siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleClient::from_config(
                siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleConfig::new(
                    "perplexity",
                    "ctx-key",
                    "https://api.perplexity.ai",
                    adapter,
                )
                .with_model(default_model)
                .with_http_transport(Arc::new(config_transport.clone())),
            )
            .await
            .expect("build config client");

        let factory = OpenAICompatibleProviderFactory::new("perplexity".to_string());
        let registry_client = factory
            .language_model_with_ctx(
                default_model,
                &BuildContext {
                    provider_id: Some("perplexity".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model(request_model).with_provider_option(
            "perplexity",
            serde_json::json!({
                "search_mode": "academic",
                "return_images": true,
                "someVendorParam": true
            }),
        );

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.body["model"], serde_json::json!(request_model));
        assert_eq!(
            builder_req.body["search_mode"],
            serde_json::json!("academic")
        );
        assert_eq!(builder_req.body["return_images"], serde_json::json!(true));
        assert_eq!(builder_req.body["someVendorParam"], serde_json::json!(true));
    }

    #[tokio::test]
    async fn openai_compatible_builder_config_registry_rerank_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let model = "jina-reranker-m0";

        let builder_client =
            siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleBuilder::new(
                siumai_provider_openai_compatible::builder::BuilderBase::default(),
                "jina",
            )
            .api_key("ctx-key")
            .base_url("https://example.com/v1/")
            .model(model)
            .fetch(Arc::new(builder_transport.clone()))
            .build()
            .await
            .expect("build builder client");

        let provider =
            siumai_provider_openai_compatible::providers::openai_compatible::get_provider_config(
                "jina",
            )
            .expect("provider config");
        let adapter = Arc::new(
            siumai_provider_openai_compatible::providers::openai_compatible::ConfigurableAdapter::new(
                provider,
            ),
        );
        let config_client =
            siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleClient::from_config(
                siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleConfig::new(
                    "jina",
                    "ctx-key",
                    "https://example.com/v1/",
                    adapter,
                )
                .with_model(model)
                .with_http_transport(Arc::new(config_transport.clone())),
            )
            .await
            .expect("build config client");

        let factory = OpenAICompatibleProviderFactory::new("jina".to_string());
        let registry_client = factory
            .reranking_model_with_ctx(
                model,
                &BuildContext {
                    provider_id: Some("jina".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/v1/".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_rerank_request(model).with_top_n(1);

        let _ = builder_client.rerank(request.clone()).await;
        let _ = config_client.rerank(request.clone()).await;
        let _ = registry_client
            .as_rerank_capability()
            .expect("registry rerank capability")
            .rerank(request)
            .await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.url, "https://example.com/v1/rerank");
        assert_eq!(builder_req.body["model"], serde_json::json!(model));
        assert_eq!(builder_req.body["query"], serde_json::json!("query"));
        assert_eq!(
            builder_req.body["documents"],
            serde_json::json!(["doc-1", "doc-2"])
        );
        assert_eq!(builder_req.body["top_n"], serde_json::json!(1));
    }

    #[tokio::test]
    async fn openai_compatible_builder_config_registry_image_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let model = "black-forest-labs/FLUX.1-schnell";

        let builder_client =
            siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleBuilder::new(
                siumai_provider_openai_compatible::builder::BuilderBase::default(),
                "together",
            )
            .api_key("ctx-key")
            .base_url("https://example.com/v1/")
            .model(model)
            .fetch(Arc::new(builder_transport.clone()))
            .build()
            .await
            .expect("build builder client");

        let provider =
            siumai_provider_openai_compatible::providers::openai_compatible::get_provider_config(
                "together",
            )
            .expect("provider config");
        let adapter = Arc::new(
            siumai_provider_openai_compatible::providers::openai_compatible::ConfigurableAdapter::new(
                provider,
            ),
        );
        let config_client =
            siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleClient::from_config(
                siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleConfig::new(
                    "together",
                    "ctx-key",
                    "https://example.com/v1/",
                    adapter,
                )
                .with_model(model)
                .with_http_transport(Arc::new(config_transport.clone())),
            )
            .await
            .expect("build config client");

        let factory = OpenAICompatibleProviderFactory::new("together".to_string());
        let registry_client = factory
            .image_model_with_ctx(
                model,
                &BuildContext {
                    provider_id: Some("together".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/v1/".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = crate::types::ImageGenerationRequest {
            prompt: "a tiny blue robot".to_string(),
            negative_prompt: Some("blurry".to_string()),
            size: Some("1024x1024".to_string()),
            count: 1,
            model: Some(model.to_string()),
            quality: None,
            style: None,
            seed: None,
            steps: None,
            guidance_scale: None,
            enhance_prompt: None,
            response_format: Some("url".to_string()),
            extra_params: Default::default(),
            provider_options_map: Default::default(),
            http_config: None,
        };

        let _ = builder_client.generate_images(request.clone()).await;
        let _ = config_client.generate_images(request.clone()).await;
        let _ = registry_client
            .as_image_generation_capability()
            .expect("registry image capability")
            .generate_images(request)
            .await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.url, "https://example.com/v1/images/generations");
        assert_eq!(builder_req.body["model"], serde_json::json!(model));
        assert_eq!(
            builder_req.body["prompt"],
            serde_json::json!("a tiny blue robot")
        );
        assert_eq!(builder_req.body["size"], serde_json::json!("1024x1024"));
        assert_eq!(builder_req.body["n"], serde_json::json!(1));
        assert_eq!(
            builder_req.body["response_format"],
            serde_json::json!("url")
        );
    }

    #[tokio::test]
    async fn siliconflow_registry_image_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let siliconflow_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "siliconflow".to_string(),
            Arc::new(OpenAICompatibleProviderFactory::new(
                "siliconflow".to_string(),
            )) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "siliconflow",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/siliconflow/v1")
                    .fetch(Arc::new(siliconflow_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .image_model("siliconflow:stability-ai/sdxl")
            .expect("build siliconflow image handle");

        let _ = ImageGenerationCapability::generate_images(
            &handle,
            crate::types::ImageGenerationRequest {
                prompt: "a tiny orange robot".to_string(),
                negative_prompt: Some("blurry".to_string()),
                size: Some("1024x1024".to_string()),
                count: 1,
                model: Some("stability-ai/sdxl".to_string()),
                quality: None,
                style: None,
                seed: None,
                steps: None,
                guidance_scale: None,
                enhance_prompt: None,
                response_format: Some("url".to_string()),
                extra_params: Default::default(),
                provider_options_map: Default::default(),
                http_config: None,
            },
        )
        .await;

        assert!(global_transport.take().is_none());

        let req = siliconflow_transport
            .take()
            .expect("captured siliconflow image request");
        assert_eq!(
            header_value(&req, "authorization"),
            Some("Bearer ctx-key".to_string())
        );
        assert_eq!(
            req.url,
            "https://example.com/siliconflow/v1/images/generations"
        );
        assert_eq!(req.body["model"], serde_json::json!("stability-ai/sdxl"));
        assert_eq!(req.body["prompt"], serde_json::json!("a tiny orange robot"));
        assert_eq!(req.body["size"], serde_json::json!("1024x1024"));
        assert_eq!(req.body["n"], serde_json::json!(1));
        assert_eq!(req.body["response_format"], serde_json::json!("url"));
    }

    #[tokio::test]
    async fn together_registry_image_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let together_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "together".to_string(),
            Arc::new(OpenAICompatibleProviderFactory::new("together".to_string()))
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "together",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/together/v1")
                    .fetch(Arc::new(together_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .image_model("together:black-forest-labs/FLUX.1-schnell")
            .expect("build together image handle");

        let _ = ImageGenerationCapability::generate_images(
            &handle,
            crate::types::ImageGenerationRequest {
                prompt: "a tiny blue robot".to_string(),
                negative_prompt: Some("blurry".to_string()),
                size: Some("1024x1024".to_string()),
                count: 1,
                model: Some("black-forest-labs/FLUX.1-schnell".to_string()),
                quality: None,
                style: None,
                seed: None,
                steps: None,
                guidance_scale: None,
                enhance_prompt: None,
                response_format: Some("url".to_string()),
                extra_params: Default::default(),
                provider_options_map: Default::default(),
                http_config: None,
            },
        )
        .await;

        assert!(global_transport.take().is_none());

        let req = together_transport
            .take()
            .expect("captured together image request");
        assert_eq!(
            header_value(&req, "authorization"),
            Some("Bearer ctx-key".to_string())
        );
        assert_eq!(
            req.url,
            "https://example.com/together/v1/images/generations"
        );
        assert_eq!(
            req.body["model"],
            serde_json::json!("black-forest-labs/FLUX.1-schnell")
        );
        assert_eq!(req.body["prompt"], serde_json::json!("a tiny blue robot"));
        assert_eq!(req.body["size"], serde_json::json!("1024x1024"));
        assert_eq!(req.body["n"], serde_json::json!(1));
        assert_eq!(req.body["response_format"], serde_json::json!("url"));
    }

    #[tokio::test]
    async fn openai_compatible_builder_config_registry_chat_stream_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client =
            siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleBuilder::new(
                siumai_provider_openai_compatible::builder::BuilderBase::default(),
                "deepseek",
            )
            .api_key("ctx-key")
            .base_url("https://example.com/custom/v1/")
            .model("deepseek-chat")
            .temperature(0.4)
            .max_tokens(256)
            .top_p(0.9)
            .stop(vec!["END"])
            .seed(7)
            .reasoning(true)
            .reasoning_budget(2048)
            .fetch(Arc::new(builder_transport.clone()))
            .build()
            .await
            .expect("build builder client");

        let provider =
            siumai_provider_openai_compatible::providers::openai_compatible::get_provider_config(
                "deepseek",
            )
            .expect("provider config");
        let adapter = Arc::new(
            siumai_provider_openai_compatible::providers::openai_compatible::ConfigurableAdapter::new(
                provider,
            ),
        );
        let config_client =
            siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleClient::from_config(
                siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleConfig::new(
                    "deepseek",
                    "ctx-key",
                    "https://example.com/custom/v1/",
                    adapter,
                )
                .with_model("deepseek-chat")
                .with_temperature(0.4)
                .with_max_tokens(256)
                .with_top_p(0.9)
                .with_stop_sequences(vec!["END".to_string()])
                .with_seed(7)
                .with_reasoning(true)
                .with_reasoning_budget(2048)
                .with_http_transport(Arc::new(config_transport.clone())),
            )
            .await
            .expect("build config client");

        let mut common_params = crate::types::CommonParams::default();
        common_params.model = "deepseek-chat".to_string();
        common_params.temperature = Some(0.4);
        common_params.max_tokens = Some(256);
        common_params.top_p = Some(0.9);
        common_params.stop_sequences = Some(vec!["END".to_string()]);
        common_params.seed = Some(7);

        let factory = OpenAICompatibleProviderFactory::new("deepseek".to_string());
        let registry_client = factory
            .language_model_with_ctx(
                "deepseek-chat",
                &BuildContext {
                    provider_id: Some("deepseek".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom/v1/".to_string()),
                    common_params: Some(common_params),
                    reasoning_enabled: Some(true),
                    reasoning_budget: Some(2048),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model("deepseek-chat").with_provider_option(
            "deepseek",
            serde_json::json!({
                "foo": "bar"
            }),
        );

        let _ = builder_client.chat_stream_request(request.clone()).await;
        let _ = config_client.chat_stream_request(request.clone()).await;
        let _ = registry_client.chat_stream_request(request).await;

        let builder_req = builder_transport
            .take_stream()
            .expect("builder stream request");
        let config_req = config_transport
            .take_stream()
            .expect("config stream request");
        let registry_req = registry_transport
            .take_stream()
            .expect("registry stream request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.body["stream"], serde_json::json!(true));
        assert_eq!(
            header_value(&builder_req, "accept"),
            Some("text/event-stream".to_string())
        );
        assert_eq!(builder_req.body["top_p"], serde_json::json!(0.9));
        assert_eq!(builder_req.body["seed"], serde_json::json!(7));
        assert_eq!(
            builder_req.body["enable_reasoning"],
            serde_json::json!(true)
        );
        assert_eq!(
            builder_req.body["reasoning_budget"],
            serde_json::json!(2048)
        );
        assert_eq!(builder_req.body["foo"], serde_json::json!("bar"));
    }

    #[tokio::test]
    async fn openrouter_builder_config_registry_chat_stream_request_respect_explicit_request_model()
    {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();
        let default_model = "openai/gpt-4o";
        let request_model = "openai/gpt-4.1";

        let builder_client =
            siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleBuilder::new(
                siumai_provider_openai_compatible::builder::BuilderBase::default(),
                "openrouter",
            )
            .api_key("ctx-key")
            .model(default_model)
            .reasoning(true)
            .reasoning_budget(2048)
            .fetch(Arc::new(builder_transport.clone()))
            .build()
            .await
            .expect("build builder client");

        let provider =
            siumai_provider_openai_compatible::providers::openai_compatible::get_provider_config(
                "openrouter",
            )
            .expect("provider config");
        let adapter = Arc::new(
            siumai_provider_openai_compatible::providers::openai_compatible::ConfigurableAdapter::new(
                provider,
            ),
        );
        let config_client =
            siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleClient::from_config(
                siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleConfig::new(
                    "openrouter",
                    "ctx-key",
                    "https://openrouter.ai/api/v1",
                    adapter,
                )
                .with_model(default_model)
                .with_reasoning(true)
                .with_reasoning_budget(2048)
                .with_http_transport(Arc::new(config_transport.clone())),
            )
            .await
            .expect("build config client");

        let factory = OpenAICompatibleProviderFactory::new("openrouter".to_string());
        let registry_client = factory
            .language_model_with_ctx(
                default_model,
                &BuildContext {
                    provider_id: Some("openrouter".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    reasoning_enabled: Some(true),
                    reasoning_budget: Some(2048),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model(request_model).with_provider_option(
            "openrouter",
            serde_json::json!({
                "transforms": ["middle-out"],
                "someVendorParam": true
            }),
        );

        let _ = builder_client.chat_stream_request(request.clone()).await;
        let _ = config_client.chat_stream_request(request.clone()).await;
        let _ = registry_client.chat_stream_request(request).await;

        let builder_req = builder_transport
            .take_stream()
            .expect("builder stream request");
        let config_req = config_transport
            .take_stream()
            .expect("config stream request");
        let registry_req = registry_transport
            .take_stream()
            .expect("registry stream request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.body["model"], serde_json::json!(request_model));
        assert_eq!(builder_req.body["stream"], serde_json::json!(true));
        assert_eq!(
            builder_req.body["transforms"],
            serde_json::json!(["middle-out"])
        );
        assert_eq!(builder_req.body["someVendorParam"], serde_json::json!(true));
        assert_eq!(
            builder_req.body["enable_reasoning"],
            serde_json::json!(true)
        );
        assert_eq!(
            builder_req.body["reasoning_budget"],
            serde_json::json!(2048)
        );
    }

    #[tokio::test]
    async fn perplexity_builder_config_registry_chat_stream_request_respect_explicit_request_model()
    {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();
        let default_model = "sonar";
        let request_model = "sonar-pro";

        let builder_client =
            siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleBuilder::new(
                siumai_provider_openai_compatible::builder::BuilderBase::default(),
                "perplexity",
            )
            .api_key("ctx-key")
            .model(default_model)
            .fetch(Arc::new(builder_transport.clone()))
            .build()
            .await
            .expect("build builder client");

        let provider =
            siumai_provider_openai_compatible::providers::openai_compatible::get_provider_config(
                "perplexity",
            )
            .expect("provider config");
        let adapter = Arc::new(
            siumai_provider_openai_compatible::providers::openai_compatible::ConfigurableAdapter::new(
                provider,
            ),
        );
        let config_client =
            siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleClient::from_config(
                siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleConfig::new(
                    "perplexity",
                    "ctx-key",
                    "https://api.perplexity.ai",
                    adapter,
                )
                .with_model(default_model)
                .with_http_transport(Arc::new(config_transport.clone())),
            )
            .await
            .expect("build config client");

        let factory = OpenAICompatibleProviderFactory::new("perplexity".to_string());
        let registry_client = factory
            .language_model_with_ctx(
                default_model,
                &BuildContext {
                    provider_id: Some("perplexity".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model(request_model).with_provider_option(
            "perplexity",
            serde_json::json!({
                "search_mode": "academic",
                "return_images": true,
                "someVendorParam": true
            }),
        );

        let _ = builder_client.chat_stream_request(request.clone()).await;
        let _ = config_client.chat_stream_request(request.clone()).await;
        let _ = registry_client.chat_stream_request(request).await;

        let builder_req = builder_transport
            .take_stream()
            .expect("builder stream request");
        let config_req = config_transport
            .take_stream()
            .expect("config stream request");
        let registry_req = registry_transport
            .take_stream()
            .expect("registry stream request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.body["model"], serde_json::json!(request_model));
        assert_eq!(builder_req.body["stream"], serde_json::json!(true));
        assert_eq!(
            builder_req.body["search_mode"],
            serde_json::json!("academic")
        );
        assert_eq!(builder_req.body["return_images"], serde_json::json!(true));
        assert_eq!(builder_req.body["someVendorParam"], serde_json::json!(true));
    }
}

#[cfg(feature = "deepseek")]
mod deepseek_contract {
    use super::*;
    use reqwest::header::AUTHORIZATION;
    use siumai_core::traits::ChatCapability;
    use siumai_provider_deepseek::provider_metadata::deepseek::DeepSeekChatResponseExt;
    use siumai_provider_deepseek::provider_options::deepseek::DeepSeekOptions;
    use siumai_provider_deepseek::providers::deepseek::ext::request_options::DeepSeekChatRequestExt;

    #[tokio::test]
    async fn deepseek_factory_does_not_advertise_non_text_capabilities() {
        let _lock = lock_env();

        let factory = crate::registry::factories::DeepSeekProviderFactory;
        let caps = factory.capabilities();

        assert!(caps.supports("chat"));
        assert!(caps.supports("streaming"));
        assert!(caps.supports("tools"));
        assert!(caps.supports("vision"));
        assert!(caps.supports("thinking"));
        assert_embedding_image_rerank_capabilities_absent(&caps);
        assert!(!caps.supports("speech"));
        assert!(!caps.supports("transcription"));
        assert!(!caps.supports("audio"));
    }

    #[tokio::test]
    async fn deepseek_factory_prefers_ctx_http_client_over_http_config() {
        let _lock = lock_env();

        let factory = crate::registry::factories::DeepSeekProviderFactory;

        let mut bad = HttpConfig::default();
        bad.proxy = Some("not-a-url".to_string());

        let ctx = BuildContext {
            provider_id: Some("deepseek".to_string()),
            api_key: Some("ctx-key".to_string()),
            http_client: Some(reqwest::Client::new()),
            http_config: Some(bad),
            ..Default::default()
        };

        factory
            .language_model_with_ctx("deepseek-chat", &ctx)
            .await
            .expect("factory should prefer ctx.http_client over invalid http_config");
    }

    #[tokio::test]
    async fn deepseek_factory_uses_env_api_key_when_ctx_missing() {
        let _lock = lock_env();

        let _g = EnvGuard::set("DEEPSEEK_API_KEY", "env-key");

        let factory = crate::registry::factories::DeepSeekProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("deepseek".to_string()),
            base_url: Some("https://example.com/v1".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("deepseek-chat", &ctx)
            .await
            .expect("build client via env api key");

        let _ = client
            .chat_request(make_chat_request_with_model("deepseek-chat"))
            .await;
        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer env-key");
        assert!(req.url.starts_with("https://example.com/v1"));
    }

    #[tokio::test]
    async fn deepseek_factory_prefers_ctx_api_key_over_env() {
        let _lock = lock_env();

        let _g = EnvGuard::set("DEEPSEEK_API_KEY", "env-key");

        let factory = crate::registry::factories::DeepSeekProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("deepseek".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/v1".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("deepseek-chat", &ctx)
            .await
            .expect("build client via ctx api key");

        let _ = client
            .chat_request(make_chat_request_with_model("deepseek-chat"))
            .await;
        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
    }

    #[tokio::test]
    async fn deepseek_factory_prefers_ctx_base_url_over_default() {
        let _lock = lock_env();

        let factory = crate::registry::factories::DeepSeekProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("deepseek".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("deepseek-chat", &ctx)
            .await
            .expect("build client");

        let _ = client
            .chat_request(make_chat_request_with_model("deepseek-chat"))
            .await;
        let req = transport.take().expect("captured request");
        assert!(req.url.starts_with("https://example.com/custom"));
    }

    #[tokio::test]
    async fn deepseek_factory_returns_provider_owned_client() {
        let _lock = lock_env();

        let factory = crate::registry::factories::DeepSeekProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("deepseek".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom/".to_string()),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("deepseek-chat", &ctx)
            .await
            .expect("build provider-owned DeepSeek client");

        assert_no_deferred_capability_leaks(client.as_ref());
        assert!(client.as_speech_capability().is_none());
        assert!(client.as_transcription_capability().is_none());

        let typed = client
            .as_any()
            .downcast_ref::<siumai_provider_deepseek::providers::deepseek::DeepSeekClient>()
            .expect("DeepSeekClient");
        assert_eq!(typed.base_url(), "https://example.com/custom");
        assert_eq!(crate::traits::ModelMetadata::provider_id(typed), "deepseek");
        assert_eq!(
            crate::traits::ModelMetadata::model_id(typed),
            "deepseek-chat"
        );
    }

    #[tokio::test]
    async fn deepseek_factory_supports_native_text_family_path() {
        let _lock = lock_env();

        let factory = crate::registry::factories::DeepSeekProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("deepseek".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/v1/".to_string()),
            ..Default::default()
        };

        let model = factory
            .language_model_text_with_ctx("deepseek-chat", &ctx)
            .await
            .expect("build native text-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "deepseek"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "deepseek-chat"
        );
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model.as_ref()),
            crate::traits::ModelSpecVersion::V1
        );
    }

    #[test]
    fn deepseek_factory_source_routes_construction_through_provider_owned_builder() {
        let source = include_str!("deepseek.rs");

        assert!(source.contains("DeepSeekBuilder::new("));
        assert!(source.contains(".with_http_config("));
        assert!(source.contains(".with_model_middlewares("));
    }

    #[tokio::test]
    async fn deepseek_factory_rejects_deferred_non_text_family_paths() {
        let _lock = lock_env();

        let factory = crate::registry::factories::DeepSeekProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("deepseek".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/v1/".to_string()),
            ..Default::default()
        };

        assert_unsupported_operation_contains(
            factory
                .embedding_model_with_ctx("deepseek-chat", &ctx)
                .await,
            "embedding family path",
        );
        assert_unsupported_operation_contains(
            factory.image_model_with_ctx("deepseek-chat", &ctx).await,
            "image family path",
        );
        assert_unsupported_operation_contains(
            factory.speech_model_with_ctx("deepseek-chat", &ctx).await,
            "speech family path",
        );
        assert_unsupported_operation_contains(
            factory
                .transcription_model_with_ctx("deepseek-chat", &ctx)
                .await,
            "transcription family path",
        );
        assert_unsupported_operation_contains(
            factory
                .reranking_model_with_ctx("deepseek-chat", &ctx)
                .await,
            "reranking family path",
        );
    }

    #[tokio::test]
    async fn deepseek_builder_config_registry_chat_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_deepseek::providers::deepseek::DeepSeekBuilder::new(
            siumai_provider_deepseek::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/custom/v1/")
        .model("deepseek-chat")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client =
            siumai_provider_deepseek::providers::deepseek::DeepSeekClient::from_config(
                siumai_provider_deepseek::providers::deepseek::DeepSeekConfig::new("ctx-key")
                    .with_base_url("https://example.com/custom/v1/")
                    .with_model("deepseek-chat")
                    .with_http_transport(Arc::new(config_transport.clone())),
            )
            .await
            .expect("build config client");

        let factory = crate::registry::factories::DeepSeekProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "deepseek-chat",
                &BuildContext {
                    provider_id: Some("deepseek".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom/v1/".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model("deepseek-chat").with_provider_option(
            "deepseek",
            serde_json::json!({
                "enableReasoning": true,
                "reasoningBudget": 4096,
                "foo": "bar"
            }),
        );

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(
            builder_req.body["enable_reasoning"],
            serde_json::json!(true)
        );
        assert_eq!(
            builder_req.body["reasoning_budget"],
            serde_json::json!(4096)
        );
        assert!(builder_req.body.get("enableReasoning").is_none());
        assert!(builder_req.body.get("reasoningBudget").is_none());
    }

    #[tokio::test]
    async fn deepseek_builder_config_registry_chat_request_respect_explicit_request_model() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();
        let default_model = "deepseek-chat";
        let request_model = "deepseek-reasoner";

        let builder_client = siumai_provider_deepseek::providers::deepseek::DeepSeekBuilder::new(
            siumai_provider_deepseek::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/custom/v1/")
        .model(default_model)
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client =
            siumai_provider_deepseek::providers::deepseek::DeepSeekClient::from_config(
                siumai_provider_deepseek::providers::deepseek::DeepSeekConfig::new("ctx-key")
                    .with_base_url("https://example.com/custom/v1/")
                    .with_model(default_model)
                    .with_http_transport(Arc::new(config_transport.clone())),
            )
            .await
            .expect("build config client");

        let factory = crate::registry::factories::DeepSeekProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                default_model,
                &BuildContext {
                    provider_id: Some("deepseek".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom/v1/".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model(request_model).with_provider_option(
            "deepseek",
            serde_json::json!({
                "enableReasoning": true,
                "reasoningBudget": 4096,
                "foo": "bar"
            }),
        );

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.body["model"], serde_json::json!(request_model));
        assert_eq!(
            builder_req.body["enable_reasoning"],
            serde_json::json!(true)
        );
        assert_eq!(
            builder_req.body["reasoning_budget"],
            serde_json::json!(4096)
        );
    }

    #[tokio::test]
    async fn deepseek_builder_config_registry_stable_request_options_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_deepseek::providers::deepseek::DeepSeekBuilder::new(
            siumai_provider_deepseek::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/custom/v1/")
        .model("deepseek-chat")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client =
            siumai_provider_deepseek::providers::deepseek::DeepSeekClient::from_config(
                siumai_provider_deepseek::providers::deepseek::DeepSeekConfig::new("ctx-key")
                    .with_base_url("https://example.com/custom/v1/")
                    .with_model("deepseek-chat")
                    .with_http_transport(Arc::new(config_transport.clone())),
            )
            .await
            .expect("build config client");

        let factory = crate::registry::factories::DeepSeekProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "deepseek-chat",
                &BuildContext {
                    provider_id: Some("deepseek".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom/v1/".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model("deepseek-chat").with_deepseek_options(
            DeepSeekOptions::new()
                .with_reasoning_budget(4096)
                .with_param("foo", serde_json::json!("bar")),
        );

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(
            builder_req.body["enable_reasoning"],
            serde_json::json!(true)
        );
        assert_eq!(
            builder_req.body["reasoning_budget"],
            serde_json::json!(4096)
        );
        assert_eq!(builder_req.body["foo"], serde_json::json!("bar"));
    }

    #[tokio::test]
    async fn deepseek_builder_config_registry_chat_stream_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_deepseek::providers::deepseek::DeepSeekBuilder::new(
            siumai_provider_deepseek::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/custom/v1/")
        .model("deepseek-chat")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client =
            siumai_provider_deepseek::providers::deepseek::DeepSeekClient::from_config(
                siumai_provider_deepseek::providers::deepseek::DeepSeekConfig::new("ctx-key")
                    .with_base_url("https://example.com/custom/v1/")
                    .with_model("deepseek-chat")
                    .with_http_transport(Arc::new(config_transport.clone())),
            )
            .await
            .expect("build config client");

        let factory = crate::registry::factories::DeepSeekProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "deepseek-chat",
                &BuildContext {
                    provider_id: Some("deepseek".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom/v1/".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model("deepseek-chat").with_provider_option(
            "deepseek",
            serde_json::json!({
                "enableReasoning": true,
                "reasoningBudget": 4096,
                "foo": "bar"
            }),
        );

        let _ = builder_client.chat_stream_request(request.clone()).await;
        let _ = config_client.chat_stream_request(request.clone()).await;
        let _ = registry_client.chat_stream_request(request).await;

        let builder_req = builder_transport
            .take_stream()
            .expect("builder stream request");
        let config_req = config_transport
            .take_stream()
            .expect("config stream request");
        let registry_req = registry_transport
            .take_stream()
            .expect("registry stream request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.body["stream"], serde_json::json!(true));
        assert_eq!(
            header_value(&builder_req, "accept"),
            Some("text/event-stream".to_string())
        );
    }

    #[tokio::test]
    async fn deepseek_builder_config_registry_chat_stream_request_respect_explicit_request_model() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();
        let default_model = "deepseek-chat";
        let request_model = "deepseek-reasoner";

        let builder_client = siumai_provider_deepseek::providers::deepseek::DeepSeekBuilder::new(
            siumai_provider_deepseek::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/custom/v1/")
        .model(default_model)
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client =
            siumai_provider_deepseek::providers::deepseek::DeepSeekClient::from_config(
                siumai_provider_deepseek::providers::deepseek::DeepSeekConfig::new("ctx-key")
                    .with_base_url("https://example.com/custom/v1/")
                    .with_model(default_model)
                    .with_http_transport(Arc::new(config_transport.clone())),
            )
            .await
            .expect("build config client");

        let factory = crate::registry::factories::DeepSeekProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                default_model,
                &BuildContext {
                    provider_id: Some("deepseek".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom/v1/".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model(request_model).with_provider_option(
            "deepseek",
            serde_json::json!({
                "enableReasoning": true,
                "reasoningBudget": 4096,
                "foo": "bar"
            }),
        );

        let _ = builder_client.chat_stream_request(request.clone()).await;
        let _ = config_client.chat_stream_request(request.clone()).await;
        let _ = registry_client.chat_stream_request(request).await;

        let builder_req = builder_transport
            .take_stream()
            .expect("builder stream request");
        let config_req = config_transport
            .take_stream()
            .expect("config stream request");
        let registry_req = registry_transport
            .take_stream()
            .expect("registry stream request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.body["model"], serde_json::json!(request_model));
        assert_eq!(builder_req.body["stream"], serde_json::json!(true));
    }

    #[tokio::test]
    async fn deepseek_builder_config_registry_stable_stream_request_options_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_deepseek::providers::deepseek::DeepSeekBuilder::new(
            siumai_provider_deepseek::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/custom/v1/")
        .model("deepseek-chat")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client =
            siumai_provider_deepseek::providers::deepseek::DeepSeekClient::from_config(
                siumai_provider_deepseek::providers::deepseek::DeepSeekConfig::new("ctx-key")
                    .with_base_url("https://example.com/custom/v1/")
                    .with_model("deepseek-chat")
                    .with_http_transport(Arc::new(config_transport.clone())),
            )
            .await
            .expect("build config client");

        let factory = crate::registry::factories::DeepSeekProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "deepseek-chat",
                &BuildContext {
                    provider_id: Some("deepseek".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom/v1/".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model("deepseek-chat").with_deepseek_options(
            DeepSeekOptions::new()
                .with_reasoning_budget(4096)
                .with_param("foo", serde_json::json!("bar")),
        );

        let _ = builder_client.chat_stream_request(request.clone()).await;
        let _ = config_client.chat_stream_request(request.clone()).await;
        let _ = registry_client.chat_stream_request(request).await;

        let builder_req = builder_transport
            .take_stream()
            .expect("builder stream request");
        let config_req = config_transport
            .take_stream()
            .expect("config stream request");
        let registry_req = registry_transport
            .take_stream()
            .expect("registry stream request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.body["stream"], serde_json::json!(true));
        assert_eq!(
            builder_req.body["enable_reasoning"],
            serde_json::json!(true)
        );
        assert_eq!(
            builder_req.body["reasoning_budget"],
            serde_json::json!(4096)
        );
        assert_eq!(builder_req.body["foo"], serde_json::json!("bar"));
        assert_eq!(
            header_value(&builder_req, "accept"),
            Some("text/event-stream".to_string())
        );
    }

    #[tokio::test]
    async fn deepseek_builder_config_registry_tool_choice_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_deepseek::providers::deepseek::DeepSeekBuilder::new(
            siumai_provider_deepseek::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/custom/v1/")
        .model("deepseek-chat")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client =
            siumai_provider_deepseek::providers::deepseek::DeepSeekClient::from_config(
                siumai_provider_deepseek::providers::deepseek::DeepSeekConfig::new("ctx-key")
                    .with_base_url("https://example.com/custom/v1/")
                    .with_model("deepseek-chat")
                    .with_http_transport(Arc::new(config_transport.clone())),
            )
            .await
            .expect("build config client");

        let factory = crate::registry::factories::DeepSeekProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "deepseek-chat",
                &BuildContext {
                    provider_id: Some("deepseek".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom/v1/".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .model("deepseek-chat")
            .tools(vec![crate::types::Tool::function(
                "get_weather",
                "Get weather",
                serde_json::json!({
                    "type": "object",
                    "properties": { "location": { "type": "string" } },
                    "required": ["location"],
                    "additionalProperties": false
                }),
            )])
            .tool_choice(crate::types::ToolChoice::None)
            .build()
            .with_provider_option(
                "deepseek",
                serde_json::json!({
                    "tool_choice": "auto",
                    "reasoningBudget": 4096
                }),
            );

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.body["tool_choice"], serde_json::json!("none"));
        assert_eq!(
            builder_req.body["reasoning_budget"],
            serde_json::json!(4096)
        );
        assert!(builder_req.body.get("reasoningBudget").is_none());
        assert_eq!(
            builder_req.body["tools"][0]["function"]["name"],
            serde_json::json!("get_weather")
        );
    }

    #[tokio::test]
    async fn deepseek_builder_config_registry_response_format_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_deepseek::providers::deepseek::DeepSeekBuilder::new(
            siumai_provider_deepseek::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/custom/v1/")
        .model("deepseek-chat")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client =
            siumai_provider_deepseek::providers::deepseek::DeepSeekClient::from_config(
                siumai_provider_deepseek::providers::deepseek::DeepSeekConfig::new("ctx-key")
                    .with_base_url("https://example.com/custom/v1/")
                    .with_model("deepseek-chat")
                    .with_http_transport(Arc::new(config_transport.clone())),
            )
            .await
            .expect("build config client");

        let factory = crate::registry::factories::DeepSeekProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "deepseek-chat",
                &BuildContext {
                    provider_id: Some("deepseek".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom/v1/".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let request = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .model("deepseek-chat")
            .response_format(
                crate::types::chat::ResponseFormat::json_schema(schema.clone())
                    .with_name("response"),
            )
            .build()
            .with_provider_option(
                "deepseek",
                serde_json::json!({
                    "response_format": { "type": "json_object" },
                    "reasoningBudget": 4096
                }),
            );

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(
            builder_req.body["response_format"],
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            })
        );
        assert_eq!(
            builder_req.body["reasoning_budget"],
            serde_json::json!(4096)
        );
        assert!(builder_req.body.get("reasoningBudget").is_none());
    }

    #[tokio::test]
    async fn deepseek_registry_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let deepseek_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "deepseek".to_string(),
            Arc::new(crate::registry::factories::DeepSeekProviderFactory)
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1/")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "deepseek",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/deepseek/v1/")
                    .fetch(Arc::new(deepseek_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .language_model("deepseek:deepseek-chat")
            .expect("build deepseek handle");

        let _ = handle
            .chat_request(
                make_chat_request_with_model("deepseek-chat").with_provider_option(
                    "deepseek",
                    serde_json::json!({
                        "enableReasoning": true,
                        "reasoningBudget": 4096,
                    }),
                ),
            )
            .await;

        let req = deepseek_transport
            .take()
            .expect("captured deepseek request");
        assert!(global_transport.take().is_none());
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert_eq!(req.url, "https://example.com/deepseek/v1/chat/completions");
        assert_eq!(req.body["model"], serde_json::json!("deepseek-chat"));
        assert_eq!(req.body["enable_reasoning"], serde_json::json!(true));
        assert_eq!(req.body["reasoning_budget"], serde_json::json!(4096));
    }

    #[tokio::test]
    async fn deepseek_registry_stream_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let deepseek_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "deepseek".to_string(),
            Arc::new(crate::registry::factories::DeepSeekProviderFactory)
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1/")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "deepseek",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/deepseek/v1/")
                    .fetch(Arc::new(deepseek_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .language_model("deepseek:deepseek-chat")
            .expect("build deepseek handle");

        let _ = handle
            .chat_stream_request(
                make_chat_request_with_model("deepseek-chat").with_provider_option(
                    "deepseek",
                    serde_json::json!({
                        "enableReasoning": true,
                        "reasoningBudget": 4096,
                        "foo": "bar",
                    }),
                ),
            )
            .await;

        let req = deepseek_transport
            .take_stream()
            .expect("captured deepseek stream request");
        assert!(global_transport.take().is_none());
        assert!(global_transport.take_stream().is_none());
        assert_eq!(
            header_value(&req, "authorization"),
            Some("Bearer ctx-key".to_string())
        );
        assert_eq!(
            header_value(&req, "accept"),
            Some("text/event-stream".to_string())
        );
        assert_eq!(req.url, "https://example.com/deepseek/v1/chat/completions");
        assert_eq!(req.body["model"], serde_json::json!("deepseek-chat"));
        assert_eq!(req.body["stream"], serde_json::json!(true));
        assert_eq!(req.body["enable_reasoning"], serde_json::json!(true));
        assert_eq!(req.body["reasoning_budget"], serde_json::json!(4096));
        assert_eq!(req.body["foo"], serde_json::json!("bar"));
    }

    #[tokio::test]
    async fn deepseek_registry_override_chat_response_metadata_preserves_vendor_namespace() {
        let _lock = lock_env();

        let response_json = serde_json::json!({
            "id": "chatcmpl-deepseek-test",
            "object": "chat.completion",
            "created": 1_741_392_000,
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "hello from deepseek"
                    },
                    "finish_reason": "stop",
                    "logprobs": {
                        "content": [
                            {
                                "token": "hello",
                                "logprob": -0.1,
                                "bytes": [104, 101, 108, 108, 111],
                                "top_logprobs": []
                            }
                        ]
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 3,
                "total_tokens": 14,
                "reasoning_tokens": 2
            }
        });

        let global_transport = CaptureTransport::default();
        let deepseek_transport = JsonSuccessTransport::new(response_json);
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "deepseek".to_string(),
            Arc::new(crate::registry::factories::DeepSeekProviderFactory)
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1/")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "deepseek",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/deepseek/v1/")
                    .fetch(Arc::new(deepseek_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let response = registry
            .language_model("deepseek:deepseek-chat")
            .expect("build deepseek handle")
            .chat_request(make_chat_request_with_model("deepseek-chat"))
            .await
            .expect("registry response ok");

        let root = response
            .provider_metadata
            .as_ref()
            .expect("registry provider metadata");
        assert!(root.get("deepseek").is_some());

        let metadata = response
            .deepseek_metadata()
            .expect("registry deepseek metadata");
        assert_eq!(response.content_text(), Some("hello from deepseek"));
        assert_eq!(
            response.finish_reason,
            Some(crate::types::FinishReason::Stop)
        );
        assert_eq!(
            response.usage.as_ref().map(|usage| usage.total_tokens),
            Some(14)
        );

        let expected_logprobs = serde_json::json!([
            {
                "token": "hello",
                "logprob": -0.1,
                "bytes": [104, 101, 108, 108, 111],
                "top_logprobs": []
            }
        ]);
        assert_eq!(metadata.logprobs, Some(expected_logprobs));

        let req = deepseek_transport
            .take()
            .expect("captured deepseek request");
        assert!(global_transport.take().is_none());
        assert_eq!(
            header_value(&req, "authorization"),
            Some("Bearer ctx-key".to_string())
        );
        assert_eq!(req.url, "https://example.com/deepseek/v1/chat/completions");
    }

    #[tokio::test]
    async fn deepseek_registry_override_stream_end_metadata_preserves_vendor_namespace() {
        let _lock = lock_env();

        let stream_body = br#"data: {"id":"1","model":"deepseek-chat","created":1718345013,"choices":[{"index":0,"delta":{"content":"hello","role":"assistant"},"finish_reason":null}]}

data: {"id":"1","model":"deepseek-chat","created":1718345013,"choices":[{"index":0,"delta":{"content":" from deepseek","role":null},"finish_reason":"stop","logprobs":{"content":[{"token":"hello","logprob":-0.1,"bytes":[104,101,108,108,111],"top_logprobs":[]}]}}],"usage":{"prompt_tokens":11,"completion_tokens":3,"total_tokens":14,"reasoning_tokens":2}}

data: [DONE]

"#
        .to_vec();

        let global_transport = CaptureTransport::default();
        let deepseek_transport = SseSuccessTransport::new(stream_body);
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "deepseek".to_string(),
            Arc::new(crate::registry::factories::DeepSeekProviderFactory)
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1/")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "deepseek",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/deepseek/v1/")
                    .fetch(Arc::new(deepseek_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let mut stream = registry
            .language_model("deepseek:deepseek-chat")
            .expect("build deepseek handle")
            .chat_stream_request(make_chat_request_with_model("deepseek-chat"))
            .await
            .expect("registry stream ok");

        use futures::StreamExt;

        let mut stream_end = None;
        while let Some(event) = stream.next().await {
            if let Ok(crate::streaming::ChatStreamEvent::StreamEnd { response }) = event {
                stream_end = Some(response);
                break;
            }
        }

        let response = stream_end.expect("registry stream end");
        let root = response
            .provider_metadata
            .as_ref()
            .expect("registry provider metadata");
        assert!(root.get("deepseek").is_some());

        let metadata = response
            .deepseek_metadata()
            .expect("registry deepseek metadata");
        assert_eq!(response.content_text(), Some("hello from deepseek"));
        assert_eq!(
            response.finish_reason,
            Some(crate::types::FinishReason::Stop)
        );
        assert_eq!(
            response.usage.as_ref().map(|usage| usage.total_tokens),
            Some(14)
        );

        let expected_logprobs = serde_json::json!([
            {
                "token": "hello",
                "logprob": -0.1,
                "bytes": [104, 101, 108, 108, 111],
                "top_logprobs": []
            }
        ]);
        assert_eq!(metadata.logprobs, Some(expected_logprobs));

        let req = deepseek_transport
            .take_stream()
            .expect("captured deepseek stream request");
        assert!(global_transport.take().is_none());
        assert!(global_transport.take_stream().is_none());
        assert_eq!(
            header_value(&req, "authorization"),
            Some("Bearer ctx-key".to_string())
        );
        assert_eq!(
            header_value(&req, "accept"),
            Some("text/event-stream".to_string())
        );
        assert_eq!(req.url, "https://example.com/deepseek/v1/chat/completions");
    }

    #[tokio::test]
    async fn deepseek_registry_rejects_unsupported_non_text_handle_construction() {
        let _lock = lock_env();

        let transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "deepseek".to_string(),
            Arc::new(crate::registry::factories::DeepSeekProviderFactory)
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("ctx-key")
            .with_base_url("https://example.com/deepseek/v1/")
            .fetch(Arc::new(transport.clone()))
            .auto_middleware(false)
            .build()
            .expect("build registry");

        assert_unsupported_operation_contains(
            registry.embedding_model("deepseek:deepseek-chat"),
            "embedding_model handle",
        );
        assert_unsupported_operation_contains(
            registry.image_model("deepseek:deepseek-chat"),
            "image_model handle",
        );
        assert_unsupported_operation_contains(
            registry.reranking_model("deepseek:deepseek-chat"),
            "reranking_model handle",
        );

        assert_capture_transport_unused(&transport);
    }
}

#[cfg(feature = "anthropic")]
mod anthropic_contract {
    use super::*;
    use crate::traits::ChatCapability;
    use siumai_provider_anthropic::provider_options::anthropic::{
        AnthropicContainerConfig, AnthropicContainerSkill, AnthropicEffort, AnthropicOptions,
        ThinkingModeConfig,
    };
    use siumai_provider_anthropic::providers::anthropic::ext::request_options::AnthropicChatRequestExt;

    #[tokio::test]
    async fn anthropic_factory_does_not_advertise_non_text_capabilities() {
        let _lock = lock_env();

        let factory = crate::registry::factories::AnthropicProviderFactory;
        let caps = factory.capabilities();

        assert!(caps.supports("chat"));
        assert!(caps.supports("streaming"));
        assert!(caps.supports("tools"));
        assert!(caps.supports("vision"));
        assert!(caps.supports("thinking"));
        assert_embedding_image_rerank_capabilities_absent(&caps);
        assert!(!caps.supports("speech"));
        assert!(!caps.supports("transcription"));
        assert!(!caps.supports("audio"));
    }

    #[tokio::test]
    async fn anthropic_factory_prefers_ctx_http_client_over_http_config() {
        let _lock = lock_env();

        let factory = crate::registry::factories::AnthropicProviderFactory;

        let mut bad = HttpConfig::default();
        bad.proxy = Some("not-a-url".to_string());

        let ctx = BuildContext {
            provider_id: Some("anthropic".to_string()),
            api_key: Some("ctx-key".to_string()),
            http_client: Some(reqwest::Client::new()),
            http_config: Some(bad),
            ..Default::default()
        };

        factory
            .language_model_with_ctx("claude-3-5-sonnet-20241022", &ctx)
            .await
            .expect("factory should prefer ctx.http_client over invalid http_config");
    }

    #[tokio::test]
    async fn anthropic_factory_supports_native_text_family_path() {
        let _lock = lock_env();

        let factory = crate::registry::factories::AnthropicProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("anthropic".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom/".to_string()),
            ..Default::default()
        };

        let model = factory
            .language_model_text_with_ctx("claude-3-5-haiku-20241022", &ctx)
            .await
            .expect("build native text-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "anthropic"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "claude-3-5-haiku-20241022"
        );
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model.as_ref()),
            crate::traits::ModelSpecVersion::V1
        );
    }

    #[tokio::test]
    async fn anthropic_factory_returns_provider_owned_client_without_non_text_leaks() {
        let _lock = lock_env();

        let factory = crate::registry::factories::AnthropicProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("anthropic".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom".to_string()),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("claude-3-5-sonnet-20241022", &ctx)
            .await
            .expect("build provider-owned Anthropic client");

        assert_no_deferred_capability_leaks(client.as_ref());
        assert!(client.as_speech_capability().is_none());
        assert!(client.as_transcription_capability().is_none());

        let typed = client
            .as_any()
            .downcast_ref::<siumai_provider_anthropic::providers::anthropic::AnthropicClient>()
            .expect("AnthropicClient");
        assert_eq!(
            crate::traits::ModelMetadata::provider_id(typed),
            "anthropic"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(typed),
            "claude-3-5-sonnet-20241022"
        );
    }

    #[tokio::test]
    async fn anthropic_factory_rejects_deferred_non_text_family_paths() {
        let _lock = lock_env();

        let factory = crate::registry::factories::AnthropicProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("anthropic".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom".to_string()),
            ..Default::default()
        };

        assert_unsupported_operation_contains(
            factory
                .embedding_model_with_ctx("claude-3-5-sonnet-20241022", &ctx)
                .await,
            "embedding family path",
        );
        assert_unsupported_operation_contains(
            factory
                .image_model_with_ctx("claude-3-5-sonnet-20241022", &ctx)
                .await,
            "image family path",
        );
        assert_unsupported_operation_contains(
            factory
                .speech_model_with_ctx("claude-3-5-sonnet-20241022", &ctx)
                .await,
            "speech family path",
        );
        assert_unsupported_operation_contains(
            factory
                .transcription_model_with_ctx("claude-3-5-sonnet-20241022", &ctx)
                .await,
            "transcription family path",
        );
        assert_unsupported_operation_contains(
            factory
                .reranking_model_with_ctx("claude-3-5-sonnet-20241022", &ctx)
                .await,
            "reranking family path",
        );
    }

    #[tokio::test]
    async fn anthropic_factory_uses_env_api_key_when_ctx_missing() {
        let _lock = lock_env();

        let _g = EnvGuard::set("ANTHROPIC_API_KEY", "env-key");
        let factory = crate::registry::factories::AnthropicProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("anthropic".to_string()),
            base_url: Some("https://example.com".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("claude-3-5-sonnet-20241022", &ctx)
            .await
            .expect("build client via env api key");

        let _ = client
            .chat_request(make_chat_request_with_model("claude-3-5-sonnet-20241022"))
            .await;
        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get("x-api-key").unwrap(), "env-key");
        assert!(req.url.starts_with("https://example.com"));
    }

    #[tokio::test]
    async fn anthropic_factory_prefers_ctx_api_key_over_env() {
        let _lock = lock_env();

        let _g = EnvGuard::set("ANTHROPIC_API_KEY", "env-key");
        let factory = crate::registry::factories::AnthropicProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("anthropic".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("claude-3-5-sonnet-20241022", &ctx)
            .await
            .expect("build client via ctx api key");

        let _ = client
            .chat_request(make_chat_request_with_model("claude-3-5-sonnet-20241022"))
            .await;
        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get("x-api-key").unwrap(), "ctx-key");
        assert!(req.url.starts_with("https://example.com"));
    }

    #[tokio::test]
    async fn anthropic_factory_prefers_ctx_base_url_over_default() {
        let _lock = lock_env();

        let factory = crate::registry::factories::AnthropicProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("anthropic".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("claude-3-5-sonnet-20241022", &ctx)
            .await
            .expect("build client");

        let _ = client
            .chat_request(make_chat_request_with_model("claude-3-5-sonnet-20241022"))
            .await;
        let req = transport.take().expect("captured request");
        assert!(req.url.starts_with("https://example.com/custom"));
    }

    #[tokio::test]
    async fn anthropic_registry_rejects_unsupported_non_text_handle_construction() {
        let _lock = lock_env();

        let transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "anthropic".to_string(),
            Arc::new(crate::registry::factories::AnthropicProviderFactory)
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("ctx-key")
            .with_base_url("https://example.com/anthropic")
            .fetch(Arc::new(transport.clone()))
            .auto_middleware(false)
            .build()
            .expect("build registry");

        assert_unsupported_operation_contains(
            registry.embedding_model("anthropic:claude-3-5-sonnet-20241022"),
            "embedding_model handle",
        );
        assert_unsupported_operation_contains(
            registry.image_model("anthropic:claude-3-5-sonnet-20241022"),
            "image_model handle",
        );
        assert_unsupported_operation_contains(
            registry.reranking_model("anthropic:claude-3-5-sonnet-20241022"),
            "reranking_model handle",
        );

        assert_capture_transport_unused(&transport);
    }

    #[tokio::test]
    async fn anthropic_registry_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let anthropic_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "anthropic".to_string(),
            Arc::new(crate::registry::factories::AnthropicProviderFactory)
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "anthropic",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/anthropic/v1")
                    .fetch(Arc::new(anthropic_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .language_model("anthropic:claude-3-5-sonnet-20241022")
            .expect("build anthropic handle");

        let _ = handle
            .chat_request(make_chat_request_with_model("claude-3-5-sonnet-20241022"))
            .await;

        let req = anthropic_transport
            .take()
            .expect("captured anthropic request");
        assert!(global_transport.take().is_none());
        assert_eq!(req.headers.get("x-api-key").unwrap(), "ctx-key");
        assert_eq!(req.url, "https://example.com/anthropic/v1/messages");
    }

    #[tokio::test]
    async fn anthropic_registry_stream_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let anthropic_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "anthropic".to_string(),
            Arc::new(crate::registry::factories::AnthropicProviderFactory)
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "anthropic",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/anthropic/v1")
                    .fetch(Arc::new(anthropic_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .language_model("anthropic:claude-3-5-sonnet-20241022")
            .expect("build anthropic handle");

        let _ = handle
            .chat_stream_request(make_chat_request_with_model("claude-3-5-sonnet-20241022"))
            .await;

        let req = anthropic_transport
            .take_stream()
            .expect("captured anthropic stream request");
        assert!(global_transport.take().is_none());
        assert!(global_transport.take_stream().is_none());
        assert_eq!(req.headers.get("x-api-key").unwrap(), "ctx-key");
        assert_eq!(req.headers.get("accept").unwrap(), "text/event-stream");
        assert_eq!(req.url, "https://example.com/anthropic/v1/messages");
        assert_eq!(req.body["stream"], serde_json::json!(true));
    }

    #[tokio::test]
    async fn anthropic_builder_config_registry_typed_request_options_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client =
            siumai_provider_anthropic::providers::anthropic::AnthropicBuilder::new(
                siumai_provider_anthropic::builder::BuilderBase::default(),
            )
            .api_key("ctx-key")
            .base_url("https://example.com/custom")
            .model("claude-sonnet-4-5")
            .fetch(Arc::new(builder_transport.clone()))
            .build()
            .await
            .expect("build builder client");

        let config_client =
            siumai_provider_anthropic::providers::anthropic::AnthropicClient::from_config(
                siumai_provider_anthropic::providers::anthropic::AnthropicConfig::new("ctx-key")
                    .with_base_url("https://example.com/custom")
                    .with_model("claude-sonnet-4-5")
                    .with_http_transport(Arc::new(config_transport.clone())),
            )
            .expect("build config client");

        let factory = crate::registry::factories::AnthropicProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "claude-sonnet-4-5",
                &BuildContext {
                    provider_id: Some("anthropic".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let mut request = make_chat_request_with_model("claude-sonnet-4-5").with_anthropic_options(
            AnthropicOptions::new()
                .with_thinking_mode(ThinkingModeConfig {
                    enabled: true,
                    thinking_budget: Some(1000),
                })
                .with_json_object()
                .with_context_management(serde_json::json!({
                    "clear_at_least": 1,
                    "exclude_tools": ["editor"]
                }))
                .with_effort(AnthropicEffort::High)
                .with_container(AnthropicContainerConfig {
                    id: Some("container-1".to_string()),
                    skills: Some(vec![AnthropicContainerSkill {
                        skill_type: "anthropic".to_string(),
                        skill_id: "pptx".to_string(),
                        version: "latest".to_string(),
                    }]),
                }),
        );
        request.common_params.max_tokens = Some(2000);
        request.common_params.temperature = Some(0.5);
        request.common_params.top_p = Some(0.7);

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.url, "https://example.com/custom/v1/messages");
        assert_eq!(
            header_value(&builder_req, "x-api-key"),
            Some("ctx-key".to_string())
        );
        assert_eq!(
            builder_req.body["thinking"],
            serde_json::json!({
                "type": "enabled",
                "budget_tokens": 1000
            })
        );
        assert_eq!(builder_req.body["max_tokens"], serde_json::json!(3000));
        assert!(builder_req.body.get("temperature").is_none());
        assert!(builder_req.body.get("top_p").is_none());
        assert_eq!(
            builder_req.body["output_format"],
            serde_json::json!({
                "type": "json_object"
            })
        );
        assert_eq!(
            builder_req.body["context_management"],
            serde_json::json!({
                "clear_at_least": 1,
                "exclude_tools": ["editor"]
            })
        );
        assert_eq!(
            builder_req.body["output_config"],
            serde_json::json!({
                "effort": "high"
            })
        );
        assert_eq!(
            builder_req.body["container"],
            serde_json::json!({
                "id": "container-1",
                "skills": [{
                    "type": "anthropic",
                    "skill_id": "pptx",
                    "version": "latest"
                }]
            })
        );
    }

    #[tokio::test]
    async fn anthropic_builder_config_registry_typed_stream_request_options_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client =
            siumai_provider_anthropic::providers::anthropic::AnthropicBuilder::new(
                siumai_provider_anthropic::builder::BuilderBase::default(),
            )
            .api_key("ctx-key")
            .base_url("https://example.com/custom")
            .model("claude-sonnet-4-5")
            .fetch(Arc::new(builder_transport.clone()))
            .build()
            .await
            .expect("build builder client");

        let config_client =
            siumai_provider_anthropic::providers::anthropic::AnthropicClient::from_config(
                siumai_provider_anthropic::providers::anthropic::AnthropicConfig::new("ctx-key")
                    .with_base_url("https://example.com/custom")
                    .with_model("claude-sonnet-4-5")
                    .with_http_transport(Arc::new(config_transport.clone())),
            )
            .expect("build config client");

        let factory = crate::registry::factories::AnthropicProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "claude-sonnet-4-5",
                &BuildContext {
                    provider_id: Some("anthropic".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let mut request = make_chat_request_with_model("claude-sonnet-4-5").with_anthropic_options(
            AnthropicOptions::new()
                .with_thinking_mode(ThinkingModeConfig {
                    enabled: true,
                    thinking_budget: Some(1000),
                })
                .with_json_object()
                .with_tool_streaming(false)
                .with_effort(AnthropicEffort::Low),
        );
        request.common_params.max_tokens = Some(2000);
        request.common_params.temperature = Some(0.5);
        request.common_params.top_p = Some(0.7);

        let _ = builder_client.chat_stream_request(request.clone()).await;
        let _ = config_client.chat_stream_request(request.clone()).await;
        let _ = registry_client.chat_stream_request(request).await;

        let builder_req = builder_transport
            .take_stream()
            .expect("builder stream request");
        let config_req = config_transport
            .take_stream()
            .expect("config stream request");
        let registry_req = registry_transport
            .take_stream()
            .expect("registry stream request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.url, "https://example.com/custom/v1/messages");
        assert_eq!(
            header_value(&builder_req, "x-api-key"),
            Some("ctx-key".to_string())
        );
        assert_eq!(
            header_value(&builder_req, "accept"),
            Some("text/event-stream".to_string())
        );
        assert_eq!(builder_req.body["stream"], serde_json::json!(true));
        assert_eq!(
            builder_req.body["thinking"],
            serde_json::json!({
                "type": "enabled",
                "budget_tokens": 1000
            })
        );
        assert_eq!(builder_req.body["max_tokens"], serde_json::json!(3000));
        assert!(builder_req.body.get("temperature").is_none());
        assert!(builder_req.body.get("top_p").is_none());
        assert_eq!(
            builder_req.body["output_format"],
            serde_json::json!({
                "type": "json_object"
            })
        );
        assert_eq!(
            builder_req.body["output_config"],
            serde_json::json!({
                "effort": "low"
            })
        );

        let beta = header_value(&builder_req, "anthropic-beta").unwrap_or_default();
        assert!(
            !beta
                .split(',')
                .any(|token| token.trim() == "fine-grained-tool-streaming-2025-05-14"),
            "unexpected fine-grained-tool-streaming beta token: {beta}"
        );
    }
}

#[cfg(feature = "groq")]
mod groq_contract {
    use super::*;
    use crate::traits::ChatCapability;
    use reqwest::header::AUTHORIZATION;
    use siumai_provider_groq::provider_metadata::groq::GroqChatResponseExt;
    use siumai_provider_groq::provider_options::groq::{
        GroqOptions, GroqReasoningEffort, GroqReasoningFormat, GroqServiceTier,
    };
    use siumai_provider_groq::providers::groq::ext::request_options::GroqChatRequestExt;

    #[tokio::test]
    async fn groq_factory_prefers_ctx_http_client_over_http_config() {
        let _lock = lock_env();

        let factory = crate::registry::factories::GroqProviderFactory;

        let mut bad = HttpConfig::default();
        bad.proxy = Some("not-a-url".to_string());

        let ctx = BuildContext {
            provider_id: Some("groq".to_string()),
            api_key: Some("ctx-key".to_string()),
            http_client: Some(reqwest::Client::new()),
            http_config: Some(bad),
            ..Default::default()
        };

        factory
            .language_model_with_ctx("llama-3.1-70b-versatile", &ctx)
            .await
            .expect("factory should prefer ctx.http_client over invalid http_config");
    }

    #[tokio::test]
    async fn groq_factory_uses_env_api_key_when_ctx_missing() {
        let _lock = lock_env();

        let _g = EnvGuard::set("GROQ_API_KEY", "env-key");
        let factory = crate::registry::factories::GroqProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("groq".to_string()),
            base_url: Some("https://example.com".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("llama-3.1-70b-versatile", &ctx)
            .await
            .expect("build client via env api key");

        let _ = client
            .chat_request(make_chat_request_with_model("llama-3.1-70b-versatile"))
            .await;
        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer env-key");
        assert!(req.url.starts_with("https://example.com/openai/v1"));
    }

    #[tokio::test]
    async fn groq_factory_prefers_ctx_api_key_over_env() {
        let _lock = lock_env();

        let _g = EnvGuard::set("GROQ_API_KEY", "env-key");
        let factory = crate::registry::factories::GroqProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("groq".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("llama-3.1-70b-versatile", &ctx)
            .await
            .expect("build client via ctx api key");

        let _ = client
            .chat_request(make_chat_request_with_model("llama-3.1-70b-versatile"))
            .await;
        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert!(req.url.starts_with("https://example.com/openai/v1"));
    }

    #[tokio::test]
    async fn groq_factory_appends_openai_path_for_root_base_url() {
        let _lock = lock_env();

        let factory = crate::registry::factories::GroqProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("groq".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("llama-3.1-70b-versatile", &ctx)
            .await
            .expect("build client");

        let _ = client
            .chat_request(make_chat_request_with_model("llama-3.1-70b-versatile"))
            .await;
        let req = transport.take().expect("captured request");
        assert!(
            req.url.starts_with("https://example.com/openai/v1"),
            "unexpected url: {}",
            req.url
        );
    }

    #[tokio::test]
    async fn groq_factory_returns_provider_owned_client() {
        let _lock = lock_env();

        let factory = crate::registry::factories::GroqProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("groq".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom/".to_string()),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("llama-3.1-70b-versatile", &ctx)
            .await
            .expect("build provider-owned Groq client");

        assert_no_deferred_capability_leaks(client.as_ref());

        let typed = client
            .as_any()
            .downcast_ref::<siumai_provider_groq::providers::groq::GroqClient>()
            .expect("GroqClient");
        assert_eq!(typed.base_url(), "https://example.com/custom");
        assert_eq!(crate::traits::ModelMetadata::provider_id(typed), "groq");
        assert_eq!(
            crate::traits::ModelMetadata::model_id(typed),
            "llama-3.1-70b-versatile"
        );
    }

    #[tokio::test]
    async fn groq_factory_supports_native_text_family_path() {
        let _lock = lock_env();

        let factory = crate::registry::factories::GroqProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("groq".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://api.groq.com".to_string()),
            ..Default::default()
        };

        let model = factory
            .language_model_text_with_ctx("llama-3.1-70b-versatile", &ctx)
            .await
            .expect("build native text-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "groq"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "llama-3.1-70b-versatile"
        );
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model.as_ref()),
            crate::traits::ModelSpecVersion::V1
        );
    }

    #[tokio::test]
    async fn groq_factory_declares_audio_without_image_or_rerank() {
        let _lock = lock_env();

        let factory = crate::registry::factories::GroqProviderFactory;
        let caps = factory.capabilities();

        assert!(caps.supports("chat"));
        assert!(caps.supports("streaming"));
        assert!(caps.supports("tools"));
        assert!(caps.supports("audio"));
        assert!(caps.supports("speech"));
        assert!(caps.supports("transcription"));
        assert_embedding_image_rerank_capabilities_absent(&caps);
    }

    #[tokio::test]
    async fn groq_factory_supports_native_speech_family_path() {
        let _lock = lock_env();

        let factory = crate::registry::factories::GroqProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("groq".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com".to_string()),
            ..Default::default()
        };

        let model = factory
            .speech_model_family_with_ctx("playai-tts", &ctx)
            .await
            .expect("build native speech-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "groq"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "playai-tts"
        );
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model.as_ref()),
            crate::traits::ModelSpecVersion::V1
        );
    }

    #[tokio::test]
    async fn groq_factory_supports_native_transcription_family_path() {
        let _lock = lock_env();

        let factory = crate::registry::factories::GroqProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("groq".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com".to_string()),
            ..Default::default()
        };

        let model = factory
            .transcription_model_family_with_ctx("whisper-large-v3", &ctx)
            .await
            .expect("build native transcription-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "groq"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "whisper-large-v3"
        );
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model.as_ref()),
            crate::traits::ModelSpecVersion::V1
        );
    }

    #[test]
    fn groq_factory_source_declares_native_audio_family_overrides() {
        let source = include_str!("groq.rs");

        assert!(
            source.contains("async fn speech_model_family_with_ctx("),
            "GroqProviderFactory should override speech_model_family_with_ctx instead of relying on the default ClientBackedSpeechModel bridge"
        );
        assert!(
            source.contains("async fn transcription_model_family_with_ctx("),
            "GroqProviderFactory should override transcription_model_family_with_ctx instead of relying on the default ClientBackedTranscriptionModel bridge"
        );
    }

    #[test]
    fn groq_factory_source_routes_construction_through_provider_owned_builder() {
        let source = include_str!("groq.rs");

        assert!(source.contains("GroqBuilder::new("));
        assert!(source.contains(".with_http_config("));
        assert!(source.contains(".with_model_middlewares("));
    }

    #[tokio::test]
    async fn groq_factory_rejects_native_embedding_family_path() {
        let _lock = lock_env();

        let factory = crate::registry::factories::GroqProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("groq".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://api.groq.com".to_string()),
            ..Default::default()
        };

        match factory
            .embedding_model_with_ctx("text-embedding-test", &ctx)
            .await
        {
            Ok(_) => panic!("expected UnsupportedOperation for groq embedding family path"),
            Err(LlmError::UnsupportedOperation(message)) => {
                assert!(message.contains("embedding family path"));
            }
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn groq_factory_rejects_native_image_family_path() {
        let _lock = lock_env();

        let factory = crate::registry::factories::GroqProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("groq".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://api.groq.com".to_string()),
            ..Default::default()
        };

        match factory
            .image_model_family_with_ctx("groq-image-test", &ctx)
            .await
        {
            Ok(_) => panic!("expected UnsupportedOperation for groq image family path"),
            Err(LlmError::UnsupportedOperation(message)) => {
                assert!(message.contains("image family path"));
            }
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn groq_builder_config_registry_chat_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_groq::providers::groq::GroqBuilder::new(
            siumai_provider_groq::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/custom/")
        .model("llama-3.1-70b-versatile")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client = siumai_provider_groq::providers::groq::GroqClient::from_config(
            siumai_provider_groq::providers::groq::GroqConfig::new("ctx-key")
                .with_base_url("https://example.com/custom/")
                .with_model("llama-3.1-70b-versatile")
                .with_http_transport(Arc::new(config_transport.clone())),
        )
        .await
        .expect("build config client");

        let factory = crate::registry::factories::GroqProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "llama-3.1-70b-versatile",
                &BuildContext {
                    provider_id: Some("groq".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom/".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model("llama-3.1-70b-versatile")
            .with_provider_option("groq", serde_json::json!({ "foo": "bar" }));

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.body["foo"], serde_json::json!("bar"));
    }

    #[tokio::test]
    async fn groq_builder_config_registry_chat_request_respect_explicit_request_model() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();
        let default_model = "llama-3.1-70b-versatile";
        let request_model = "llama-3.3-70b-versatile";

        let builder_client = siumai_provider_groq::providers::groq::GroqBuilder::new(
            siumai_provider_groq::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/custom/")
        .model(default_model)
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client = siumai_provider_groq::providers::groq::GroqClient::from_config(
            siumai_provider_groq::providers::groq::GroqConfig::new("ctx-key")
                .with_base_url("https://example.com/custom/")
                .with_model(default_model)
                .with_http_transport(Arc::new(config_transport.clone())),
        )
        .await
        .expect("build config client");

        let factory = crate::registry::factories::GroqProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                default_model,
                &BuildContext {
                    provider_id: Some("groq".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom/".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model(request_model)
            .with_provider_option("groq", serde_json::json!({ "foo": "bar" }));

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.body["model"], serde_json::json!(request_model));
        assert_eq!(builder_req.body["foo"], serde_json::json!("bar"));
    }

    #[tokio::test]
    async fn groq_builder_config_registry_chat_stream_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_groq::providers::groq::GroqBuilder::new(
            siumai_provider_groq::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/custom/")
        .model("llama-3.1-70b-versatile")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client = siumai_provider_groq::providers::groq::GroqClient::from_config(
            siumai_provider_groq::providers::groq::GroqConfig::new("ctx-key")
                .with_base_url("https://example.com/custom/")
                .with_model("llama-3.1-70b-versatile")
                .with_http_transport(Arc::new(config_transport.clone())),
        )
        .await
        .expect("build config client");

        let factory = crate::registry::factories::GroqProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "llama-3.1-70b-versatile",
                &BuildContext {
                    provider_id: Some("groq".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom/".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model("llama-3.1-70b-versatile")
            .with_provider_option("groq", serde_json::json!({ "foo": "bar" }));

        let _ = builder_client.chat_stream_request(request.clone()).await;
        let _ = config_client.chat_stream_request(request.clone()).await;
        let _ = registry_client.chat_stream_request(request).await;

        let builder_req = builder_transport
            .take_stream()
            .expect("builder stream request");
        let config_req = config_transport
            .take_stream()
            .expect("config stream request");
        let registry_req = registry_transport
            .take_stream()
            .expect("registry stream request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.body["stream"], serde_json::json!(true));
        assert_eq!(
            header_value(&builder_req, "accept"),
            Some("text/event-stream".to_string())
        );
    }

    #[tokio::test]
    async fn groq_builder_config_registry_chat_stream_request_respect_explicit_request_model() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();
        let default_model = "llama-3.1-70b-versatile";
        let request_model = "llama-3.3-70b-versatile";

        let builder_client = siumai_provider_groq::providers::groq::GroqBuilder::new(
            siumai_provider_groq::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/custom/")
        .model(default_model)
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client = siumai_provider_groq::providers::groq::GroqClient::from_config(
            siumai_provider_groq::providers::groq::GroqConfig::new("ctx-key")
                .with_base_url("https://example.com/custom/")
                .with_model(default_model)
                .with_http_transport(Arc::new(config_transport.clone())),
        )
        .await
        .expect("build config client");

        let factory = crate::registry::factories::GroqProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                default_model,
                &BuildContext {
                    provider_id: Some("groq".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom/".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model(request_model)
            .with_provider_option("groq", serde_json::json!({ "foo": "bar" }));

        let _ = builder_client.chat_stream_request(request.clone()).await;
        let _ = config_client.chat_stream_request(request.clone()).await;
        let _ = registry_client.chat_stream_request(request).await;

        let builder_req = builder_transport
            .take_stream()
            .expect("builder stream request");
        let config_req = config_transport
            .take_stream()
            .expect("config stream request");
        let registry_req = registry_transport
            .take_stream()
            .expect("registry stream request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.body["model"], serde_json::json!(request_model));
        assert_eq!(builder_req.body["stream"], serde_json::json!(true));
        assert_eq!(builder_req.body["foo"], serde_json::json!("bar"));
    }

    #[tokio::test]
    async fn groq_builder_config_registry_typed_request_options_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_groq::providers::groq::GroqBuilder::new(
            siumai_provider_groq::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/custom/")
        .model("llama-3.1-70b-versatile")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client = siumai_provider_groq::providers::groq::GroqClient::from_config(
            siumai_provider_groq::providers::groq::GroqConfig::new("ctx-key")
                .with_base_url("https://example.com/custom/")
                .with_model("llama-3.1-70b-versatile")
                .with_http_transport(Arc::new(config_transport.clone())),
        )
        .await
        .expect("build config client");

        let factory = crate::registry::factories::GroqProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "llama-3.1-70b-versatile",
                &BuildContext {
                    provider_id: Some("groq".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom/".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model("llama-3.1-70b-versatile").with_groq_options(
            GroqOptions::new()
                .with_logprobs(true)
                .with_top_logprobs(2)
                .with_service_tier(GroqServiceTier::Flex)
                .with_reasoning_effort(GroqReasoningEffort::Default)
                .with_reasoning_format(GroqReasoningFormat::Parsed)
                .with_param("vendor_extra", serde_json::json!(true)),
        );

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.body["logprobs"], serde_json::json!(true));
        assert_eq!(builder_req.body["top_logprobs"], serde_json::json!(2));
        assert_eq!(builder_req.body["service_tier"], serde_json::json!("flex"));
        assert_eq!(
            builder_req.body["reasoning_effort"],
            serde_json::json!("default")
        );
        assert_eq!(
            builder_req.body["reasoning_format"],
            serde_json::json!("parsed")
        );
        assert_eq!(builder_req.body["vendor_extra"], serde_json::json!(true));
    }

    #[tokio::test]
    async fn groq_builder_config_registry_typed_stream_request_options_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_groq::providers::groq::GroqBuilder::new(
            siumai_provider_groq::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/custom/")
        .model("llama-3.1-70b-versatile")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client = siumai_provider_groq::providers::groq::GroqClient::from_config(
            siumai_provider_groq::providers::groq::GroqConfig::new("ctx-key")
                .with_base_url("https://example.com/custom/")
                .with_model("llama-3.1-70b-versatile")
                .with_http_transport(Arc::new(config_transport.clone())),
        )
        .await
        .expect("build config client");

        let factory = crate::registry::factories::GroqProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "llama-3.1-70b-versatile",
                &BuildContext {
                    provider_id: Some("groq".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom/".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model("llama-3.1-70b-versatile").with_groq_options(
            GroqOptions::new()
                .with_logprobs(true)
                .with_top_logprobs(2)
                .with_service_tier(GroqServiceTier::Flex)
                .with_reasoning_effort(GroqReasoningEffort::Default)
                .with_reasoning_format(GroqReasoningFormat::Parsed)
                .with_param("vendor_extra", serde_json::json!(true)),
        );

        let _ = builder_client.chat_stream_request(request.clone()).await;
        let _ = config_client.chat_stream_request(request.clone()).await;
        let _ = registry_client.chat_stream_request(request).await;

        let builder_req = builder_transport
            .take_stream()
            .expect("builder stream request");
        let config_req = config_transport
            .take_stream()
            .expect("config stream request");
        let registry_req = registry_transport
            .take_stream()
            .expect("registry stream request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.body["stream"], serde_json::json!(true));
        assert_eq!(
            header_value(&builder_req, "accept"),
            Some("text/event-stream".to_string())
        );
        assert_eq!(builder_req.body["logprobs"], serde_json::json!(true));
        assert_eq!(builder_req.body["top_logprobs"], serde_json::json!(2));
        assert_eq!(builder_req.body["service_tier"], serde_json::json!("flex"));
        assert_eq!(
            builder_req.body["reasoning_effort"],
            serde_json::json!("default")
        );
        assert_eq!(
            builder_req.body["reasoning_format"],
            serde_json::json!("parsed")
        );
        assert_eq!(builder_req.body["vendor_extra"], serde_json::json!(true));
    }

    #[tokio::test]
    async fn groq_builder_config_registry_tool_choice_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_groq::providers::groq::GroqBuilder::new(
            siumai_provider_groq::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/custom/")
        .model("llama-3.1-70b-versatile")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client = siumai_provider_groq::providers::groq::GroqClient::from_config(
            siumai_provider_groq::providers::groq::GroqConfig::new("ctx-key")
                .with_base_url("https://example.com/custom/")
                .with_model("llama-3.1-70b-versatile")
                .with_http_transport(Arc::new(config_transport.clone())),
        )
        .await
        .expect("build config client");

        let factory = crate::registry::factories::GroqProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "llama-3.1-70b-versatile",
                &BuildContext {
                    provider_id: Some("groq".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom/".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .model("llama-3.1-70b-versatile")
            .tools(vec![crate::types::Tool::function(
                "get_weather",
                "Get weather",
                serde_json::json!({
                    "type": "object",
                    "properties": { "location": { "type": "string" } },
                    "required": ["location"],
                    "additionalProperties": false
                }),
            )])
            .tool_choice(crate::types::ToolChoice::None)
            .build()
            .with_groq_options(
                GroqOptions::new()
                    .with_reasoning_effort(GroqReasoningEffort::Default)
                    .with_param("tool_choice", serde_json::json!("auto")),
            );

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.body["tool_choice"], serde_json::json!("none"));
        assert_eq!(
            builder_req.body["reasoning_effort"],
            serde_json::json!("default")
        );
        assert_eq!(
            builder_req.body["tools"][0]["function"]["name"],
            serde_json::json!("get_weather")
        );
    }

    #[tokio::test]
    async fn groq_builder_config_registry_response_format_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_groq::providers::groq::GroqBuilder::new(
            siumai_provider_groq::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/custom/")
        .model("llama-3.1-70b-versatile")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client = siumai_provider_groq::providers::groq::GroqClient::from_config(
            siumai_provider_groq::providers::groq::GroqConfig::new("ctx-key")
                .with_base_url("https://example.com/custom/")
                .with_model("llama-3.1-70b-versatile")
                .with_http_transport(Arc::new(config_transport.clone())),
        )
        .await
        .expect("build config client");

        let factory = crate::registry::factories::GroqProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "llama-3.1-70b-versatile",
                &BuildContext {
                    provider_id: Some("groq".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom/".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
            "required": ["value"],
            "additionalProperties": false
        });

        let request = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .model("llama-3.1-70b-versatile")
            .response_format(
                crate::types::chat::ResponseFormat::json_schema(schema.clone())
                    .with_name("response"),
            )
            .build()
            .with_groq_options(
                GroqOptions::new()
                    .with_reasoning_effort(GroqReasoningEffort::Default)
                    .with_param(
                        "response_format",
                        serde_json::json!({
                            "type": "json_object"
                        }),
                    ),
            );

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(
            builder_req.body["response_format"],
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            })
        );
        assert_eq!(
            builder_req.body["reasoning_effort"],
            serde_json::json!("default")
        );
    }

    #[tokio::test]
    async fn groq_registry_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let groq_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "groq".to_string(),
            Arc::new(crate::registry::factories::GroqProviderFactory) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "groq",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com")
                    .fetch(Arc::new(groq_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .language_model("groq:llama-3.1-70b-versatile")
            .expect("build groq handle");

        let _ = handle
            .chat_request(
                make_chat_request_with_model("llama-3.1-70b-versatile")
                    .with_provider_option("groq", serde_json::json!({ "foo": "bar" })),
            )
            .await;

        let req = groq_transport.take().expect("captured groq request");
        assert!(global_transport.take().is_none());
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert_eq!(req.url, "https://example.com/openai/v1/chat/completions");
        assert_eq!(
            req.body["model"],
            serde_json::json!("llama-3.1-70b-versatile")
        );
        assert_eq!(req.body["foo"], serde_json::json!("bar"));
    }

    #[tokio::test]
    async fn groq_registry_stream_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let groq_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "groq".to_string(),
            Arc::new(crate::registry::factories::GroqProviderFactory) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "groq",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com")
                    .fetch(Arc::new(groq_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .language_model("groq:llama-3.1-70b-versatile")
            .expect("build groq handle");

        let _ = handle
            .chat_stream_request(
                make_chat_request_with_model("llama-3.1-70b-versatile")
                    .with_provider_option("groq", serde_json::json!({ "foo": "bar" })),
            )
            .await;

        let req = groq_transport
            .take_stream()
            .expect("captured groq stream request");
        assert!(global_transport.take().is_none());
        assert!(global_transport.take_stream().is_none());
        assert_eq!(
            header_value(&req, "authorization"),
            Some("Bearer ctx-key".to_string())
        );
        assert_eq!(
            header_value(&req, "accept"),
            Some("text/event-stream".to_string())
        );
        assert_eq!(req.url, "https://example.com/openai/v1/chat/completions");
        assert_eq!(req.body["stream"], serde_json::json!(true));
        assert_eq!(
            req.body["model"],
            serde_json::json!("llama-3.1-70b-versatile")
        );
        assert_eq!(req.body["foo"], serde_json::json!("bar"));
    }

    #[tokio::test]
    async fn groq_registry_override_chat_response_metadata_preserves_vendor_namespace() {
        let _lock = lock_env();

        let response_json = serde_json::json!({
            "id": "chatcmpl-groq-test",
            "object": "chat.completion",
            "created": 1_741_392_000,
            "model": "llama-3.3-70b-versatile",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "hello from groq"
                    },
                    "finish_reason": "stop",
                    "logprobs": {
                        "content": [
                            {
                                "token": "hello",
                                "logprob": -0.2,
                                "bytes": [104, 101, 108, 108, 111],
                                "top_logprobs": []
                            }
                        ]
                    }
                }
            ]
        });

        let global_transport = CaptureTransport::default();
        let groq_transport = JsonSuccessTransport::new(response_json);
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "groq".to_string(),
            Arc::new(crate::registry::factories::GroqProviderFactory) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "groq",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com")
                    .fetch(Arc::new(groq_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let response = registry
            .language_model("groq:llama-3.3-70b-versatile")
            .expect("build groq handle")
            .chat_request(make_chat_request_with_model("llama-3.3-70b-versatile"))
            .await
            .expect("registry response ok");

        let root = response
            .provider_metadata
            .as_ref()
            .expect("registry provider metadata");
        assert!(root.get("groq").is_some());

        let metadata = response.groq_metadata().expect("groq metadata");
        assert_eq!(response.content_text(), Some("hello from groq"));
        assert_eq!(
            response.finish_reason,
            Some(crate::types::FinishReason::Stop)
        );

        let expected_logprobs = serde_json::json!([
            {
                "token": "hello",
                "logprob": -0.2,
                "bytes": [104, 101, 108, 108, 111],
                "top_logprobs": []
            }
        ]);
        assert_eq!(metadata.logprobs, Some(expected_logprobs));

        let req = groq_transport.take().expect("captured groq request");
        assert!(global_transport.take().is_none());
        assert_eq!(
            header_value(&req, "authorization"),
            Some("Bearer ctx-key".to_string())
        );
        assert_eq!(req.url, "https://example.com/openai/v1/chat/completions");
    }

    #[tokio::test]
    async fn groq_registry_override_stream_end_metadata_preserves_vendor_namespace() {
        let _lock = lock_env();

        let stream_body = br#"data: {"id":"1","model":"llama-3.3-70b-versatile","created":1718345013,"choices":[{"index":0,"delta":{"content":"hello","role":"assistant"},"finish_reason":null}]}

data: {"id":"1","model":"llama-3.3-70b-versatile","created":1718345013,"choices":[{"index":0,"delta":{"content":" from groq","role":null},"finish_reason":"stop","logprobs":{"content":[{"token":"hello","logprob":-0.2,"bytes":[104,101,108,108,111],"top_logprobs":[]}]}}]}

data: [DONE]

"#
        .to_vec();

        let global_transport = CaptureTransport::default();
        let groq_transport = SseSuccessTransport::new(stream_body);
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "groq".to_string(),
            Arc::new(crate::registry::factories::GroqProviderFactory) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "groq",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com")
                    .fetch(Arc::new(groq_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let mut stream = registry
            .language_model("groq:llama-3.3-70b-versatile")
            .expect("build groq handle")
            .chat_stream_request(make_chat_request_with_model("llama-3.3-70b-versatile"))
            .await
            .expect("registry stream ok");

        use futures::StreamExt;

        let mut stream_end = None;
        while let Some(event) = stream.next().await {
            if let Ok(crate::streaming::ChatStreamEvent::StreamEnd { response }) = event {
                stream_end = Some(response);
                break;
            }
        }

        let response = stream_end.expect("registry stream end");
        let root = response
            .provider_metadata
            .as_ref()
            .expect("registry provider metadata");
        assert!(root.get("groq").is_some());

        let metadata = response.groq_metadata().expect("groq metadata");
        assert_eq!(response.content_text(), Some("hello from groq"));
        assert_eq!(
            response.finish_reason,
            Some(crate::types::FinishReason::Stop)
        );

        let expected_logprobs = serde_json::json!([
            {
                "token": "hello",
                "logprob": -0.2,
                "bytes": [104, 101, 108, 108, 111],
                "top_logprobs": []
            }
        ]);
        assert_eq!(metadata.logprobs, Some(expected_logprobs));

        let req = groq_transport
            .take_stream()
            .expect("captured groq stream request");
        assert!(global_transport.take().is_none());
        assert!(global_transport.take_stream().is_none());
        assert_eq!(
            header_value(&req, "authorization"),
            Some("Bearer ctx-key".to_string())
        );
        assert_eq!(
            header_value(&req, "accept"),
            Some("text/event-stream".to_string())
        );
        assert_eq!(req.url, "https://example.com/openai/v1/chat/completions");
    }
}

#[cfg(feature = "xai")]
mod xai_contract {
    use super::*;
    use crate::traits::ChatCapability;
    use reqwest::header::AUTHORIZATION;
    use siumai_provider_xai::provider_metadata::xai::XaiChatResponseExt;
    use siumai_provider_xai::provider_options::xai::{
        SearchMode, SearchSource, SearchSourceType, XaiOptions, XaiSearchParameters,
    };
    use siumai_provider_xai::providers::xai::ext::request_options::XaiChatRequestExt;

    #[tokio::test]
    async fn xai_factory_prefers_ctx_http_client_over_http_config() {
        let _lock = lock_env();

        let factory = crate::registry::factories::XAIProviderFactory;

        let mut bad = HttpConfig::default();
        bad.proxy = Some("not-a-url".to_string());

        let ctx = BuildContext {
            provider_id: Some("xai".to_string()),
            api_key: Some("ctx-key".to_string()),
            http_client: Some(reqwest::Client::new()),
            http_config: Some(bad),
            ..Default::default()
        };

        factory
            .language_model_with_ctx("grok-beta", &ctx)
            .await
            .expect("factory should prefer ctx.http_client over invalid http_config");
    }

    #[tokio::test]
    async fn xai_factory_uses_env_api_key_when_ctx_missing() {
        let _lock = lock_env();

        let _g = EnvGuard::set("XAI_API_KEY", "env-key");
        let factory = crate::registry::factories::XAIProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("xai".to_string()),
            base_url: Some("https://example.com/v1".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("grok-beta", &ctx)
            .await
            .expect("build client via env api key");

        let _ = client
            .chat_request(make_chat_request_with_model("grok-beta"))
            .await;
        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer env-key");
        assert!(req.url.starts_with("https://example.com/v1"));
    }

    #[tokio::test]
    async fn xai_factory_prefers_ctx_api_key_over_env() {
        let _lock = lock_env();

        let _g = EnvGuard::set("XAI_API_KEY", "env-key");
        let factory = crate::registry::factories::XAIProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("xai".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/v1".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("grok-beta", &ctx)
            .await
            .expect("build client via ctx api key");

        let _ = client
            .chat_request(make_chat_request_with_model("grok-beta"))
            .await;
        let req = transport.take().expect("captured request");
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
    }

    #[tokio::test]
    async fn xai_factory_prefers_ctx_base_url_over_default() {
        let _lock = lock_env();

        let factory = crate::registry::factories::XAIProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("xai".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("grok-beta", &ctx)
            .await
            .expect("build client");

        let _ = client
            .chat_request(make_chat_request_with_model("grok-beta"))
            .await;
        let req = transport.take().expect("captured request");
        assert!(req.url.starts_with("https://example.com/custom"));
    }

    #[tokio::test]
    async fn xai_factory_returns_provider_owned_client() {
        let _lock = lock_env();

        let factory = crate::registry::factories::XAIProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("xai".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom/".to_string()),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("grok-beta", &ctx)
            .await
            .expect("build provider-owned Xai client");

        assert_no_deferred_capability_leaks(client.as_ref());

        let typed = client
            .as_any()
            .downcast_ref::<siumai_provider_xai::providers::xai::XaiClient>()
            .expect("XaiClient");
        assert_eq!(typed.base_url(), "https://example.com/custom");
        assert_eq!(crate::traits::ModelMetadata::provider_id(typed), "xai");
        assert_eq!(crate::traits::ModelMetadata::model_id(typed), "grok-beta");
    }

    #[tokio::test]
    async fn xai_factory_declares_speech_capability() {
        let _lock = lock_env();

        let factory = crate::registry::factories::XAIProviderFactory;
        let caps = factory.capabilities();

        assert!(caps.supports("speech"));
        assert!(caps.supports("audio"));
        assert!(!caps.supports("transcription"));
        assert_embedding_image_rerank_capabilities_absent(&caps);
    }

    #[tokio::test]
    async fn xai_factory_supports_native_speech_family_path() {
        let _lock = lock_env();

        let factory = crate::registry::factories::XAIProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("xai".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom/v1/".to_string()),
            ..Default::default()
        };

        let model = factory
            .speech_model_family_with_ctx("grok-voice-mini", &ctx)
            .await
            .expect("build native speech-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "xai"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "grok-voice-mini"
        );
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model.as_ref()),
            crate::traits::ModelSpecVersion::V1
        );
    }

    #[tokio::test]
    async fn xai_factory_rejects_native_embedding_family_path() {
        let _lock = lock_env();

        let factory = crate::registry::factories::XAIProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("xai".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom/v1/".to_string()),
            ..Default::default()
        };

        match factory
            .embedding_model_with_ctx("grok-embedding-test", &ctx)
            .await
        {
            Ok(_) => panic!("expected UnsupportedOperation for xai embedding family path"),
            Err(LlmError::UnsupportedOperation(message)) => {
                assert!(message.contains("embedding family path"));
            }
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn xai_factory_rejects_native_image_family_path() {
        let _lock = lock_env();

        let factory = crate::registry::factories::XAIProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("xai".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom/v1/".to_string()),
            ..Default::default()
        };

        match factory
            .image_model_family_with_ctx("grok-image-test", &ctx)
            .await
        {
            Ok(_) => panic!("expected UnsupportedOperation for xai image family path"),
            Err(LlmError::UnsupportedOperation(message)) => {
                assert!(message.contains("image family path"));
            }
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn xai_factory_rejects_native_transcription_family_path() {
        let _lock = lock_env();

        let factory = crate::registry::factories::XAIProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("xai".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom/v1/".to_string()),
            ..Default::default()
        };

        match factory
            .transcription_model_family_with_ctx("grok-voice-mini", &ctx)
            .await
        {
            Err(crate::error::LlmError::UnsupportedOperation(message)) => {
                assert!(message.contains("transcription family path"));
            }
            Ok(_) => panic!("expected UnsupportedOperation for xai transcription family path"),
            Err(other) => panic!("expected UnsupportedOperation, got: {other:?}"),
        }
    }

    #[test]
    fn xai_factory_source_declares_native_speech_family_override() {
        let source = include_str!("xai.rs");

        assert!(source.contains("async fn speech_model_family_with_ctx("));
    }

    #[test]
    fn xai_factory_source_routes_construction_through_provider_owned_builder() {
        let source = include_str!("xai.rs");

        assert!(source.contains("XaiBuilder::new("));
        assert!(source.contains(".with_http_config("));
        assert!(source.contains(".with_model_middlewares("));
    }

    #[tokio::test]
    async fn xai_factory_supports_native_text_family_path() {
        let _lock = lock_env();

        let factory = crate::registry::factories::XAIProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("xai".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom/v1/".to_string()),
            ..Default::default()
        };

        let model = factory
            .language_model_text_with_ctx("grok-beta", &ctx)
            .await
            .expect("build native text-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "xai"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "grok-beta"
        );
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model.as_ref()),
            crate::traits::ModelSpecVersion::V1
        );
    }

    #[tokio::test]
    async fn xai_builder_config_registry_chat_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_xai::providers::xai::XaiBuilder::new(
            siumai_provider_xai::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/custom/v1/")
        .model("grok-beta")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client = siumai_provider_xai::providers::xai::XaiClient::from_config(
            siumai_provider_xai::providers::xai::XaiConfig::new("ctx-key")
                .with_base_url("https://example.com/custom/v1/")
                .with_model("grok-beta")
                .with_http_transport(Arc::new(config_transport.clone())),
        )
        .await
        .expect("build config client");

        let factory = crate::registry::factories::XAIProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "grok-beta",
                &BuildContext {
                    provider_id: Some("xai".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom/v1/".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model("grok-beta")
            .with_provider_option("xai", serde_json::json!({ "foo": "bar" }));

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.body["foo"], serde_json::json!("bar"));
    }

    #[tokio::test]
    async fn xai_builder_config_registry_chat_request_respect_explicit_request_model() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();
        let default_model = "grok-beta";
        let request_model = "grok-4";

        let builder_client = siumai_provider_xai::providers::xai::XaiBuilder::new(
            siumai_provider_xai::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/custom/v1/")
        .model(default_model)
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client = siumai_provider_xai::providers::xai::XaiClient::from_config(
            siumai_provider_xai::providers::xai::XaiConfig::new("ctx-key")
                .with_base_url("https://example.com/custom/v1/")
                .with_model(default_model)
                .with_http_transport(Arc::new(config_transport.clone())),
        )
        .await
        .expect("build config client");

        let factory = crate::registry::factories::XAIProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                default_model,
                &BuildContext {
                    provider_id: Some("xai".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom/v1/".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model(request_model)
            .with_provider_option("xai", serde_json::json!({ "foo": "bar" }));

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.body["model"], serde_json::json!(request_model));
        assert_eq!(builder_req.body["foo"], serde_json::json!("bar"));
    }

    #[tokio::test]
    async fn xai_builder_config_registry_chat_stream_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_xai::providers::xai::XaiBuilder::new(
            siumai_provider_xai::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/custom/v1/")
        .model("grok-beta")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client = siumai_provider_xai::providers::xai::XaiClient::from_config(
            siumai_provider_xai::providers::xai::XaiConfig::new("ctx-key")
                .with_base_url("https://example.com/custom/v1/")
                .with_model("grok-beta")
                .with_http_transport(Arc::new(config_transport.clone())),
        )
        .await
        .expect("build config client");

        let factory = crate::registry::factories::XAIProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "grok-beta",
                &BuildContext {
                    provider_id: Some("xai".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom/v1/".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model("grok-beta")
            .with_provider_option("xai", serde_json::json!({ "foo": "bar" }));

        let _ = builder_client.chat_stream_request(request.clone()).await;
        let _ = config_client.chat_stream_request(request.clone()).await;
        let _ = registry_client.chat_stream_request(request).await;

        let builder_req = builder_transport
            .take_stream()
            .expect("builder stream request");
        let config_req = config_transport
            .take_stream()
            .expect("config stream request");
        let registry_req = registry_transport
            .take_stream()
            .expect("registry stream request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.body["stream"], serde_json::json!(true));
        assert_eq!(
            header_value(&builder_req, "accept"),
            Some("text/event-stream".to_string())
        );
    }

    #[tokio::test]
    async fn xai_builder_config_registry_chat_stream_request_respect_explicit_request_model() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();
        let default_model = "grok-beta";
        let request_model = "grok-4";

        let builder_client = siumai_provider_xai::providers::xai::XaiBuilder::new(
            siumai_provider_xai::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/custom/v1/")
        .model(default_model)
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client = siumai_provider_xai::providers::xai::XaiClient::from_config(
            siumai_provider_xai::providers::xai::XaiConfig::new("ctx-key")
                .with_base_url("https://example.com/custom/v1/")
                .with_model(default_model)
                .with_http_transport(Arc::new(config_transport.clone())),
        )
        .await
        .expect("build config client");

        let factory = crate::registry::factories::XAIProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                default_model,
                &BuildContext {
                    provider_id: Some("xai".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom/v1/".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model(request_model)
            .with_provider_option("xai", serde_json::json!({ "foo": "bar" }));

        let _ = builder_client.chat_stream_request(request.clone()).await;
        let _ = config_client.chat_stream_request(request.clone()).await;
        let _ = registry_client.chat_stream_request(request).await;

        let builder_req = builder_transport
            .take_stream()
            .expect("builder stream request");
        let config_req = config_transport
            .take_stream()
            .expect("config stream request");
        let registry_req = registry_transport
            .take_stream()
            .expect("registry stream request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.body["model"], serde_json::json!(request_model));
        assert_eq!(builder_req.body["stream"], serde_json::json!(true));
        assert_eq!(builder_req.body["foo"], serde_json::json!("bar"));
    }

    #[tokio::test]
    async fn xai_builder_config_registry_typed_request_options_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_xai::providers::xai::XaiBuilder::new(
            siumai_provider_xai::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/custom/v1/")
        .model("grok-beta")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client = siumai_provider_xai::providers::xai::XaiClient::from_config(
            siumai_provider_xai::providers::xai::XaiConfig::new("ctx-key")
                .with_base_url("https://example.com/custom/v1/")
                .with_model("grok-beta")
                .with_http_transport(Arc::new(config_transport.clone())),
        )
        .await
        .expect("build config client");

        let factory = crate::registry::factories::XAIProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "grok-beta",
                &BuildContext {
                    provider_id: Some("xai".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom/v1/".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model("grok-beta").with_xai_options(
            XaiOptions::new()
                .with_reasoning_effort("high")
                .with_search(XaiSearchParameters {
                    mode: SearchMode::On,
                    return_citations: Some(true),
                    max_search_results: Some(3),
                    from_date: Some("2025-01-01".to_string()),
                    to_date: Some("2025-01-31".to_string()),
                    sources: Some(vec![SearchSource {
                        source_type: SearchSourceType::Web,
                        country: Some("US".to_string()),
                        allowed_websites: Some(vec!["example.com".to_string()]),
                        excluded_websites: Some(vec!["blocked.example.com".to_string()]),
                        safe_search: Some(true),
                    }]),
                }),
        );

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(
            builder_req.body["reasoning_effort"],
            serde_json::json!("high")
        );
        assert_eq!(
            builder_req.body["search_parameters"]["mode"],
            serde_json::json!("on")
        );
        assert_eq!(
            builder_req.body["search_parameters"]["return_citations"],
            serde_json::json!(true)
        );
        assert_eq!(
            builder_req.body["search_parameters"]["max_search_results"],
            serde_json::json!(3)
        );
        assert_eq!(
            builder_req.body["search_parameters"]["from_date"],
            serde_json::json!("2025-01-01")
        );
        assert_eq!(
            builder_req.body["search_parameters"]["to_date"],
            serde_json::json!("2025-01-31")
        );
        assert_eq!(
            builder_req.body["search_parameters"]["sources"][0]["allowed_websites"],
            serde_json::json!(["example.com"])
        );
        assert_eq!(
            builder_req.body["search_parameters"]["sources"][0]["excluded_websites"],
            serde_json::json!(["blocked.example.com"])
        );
        assert_eq!(
            builder_req.body["search_parameters"]["sources"][0]["safe_search"],
            serde_json::json!(true)
        );
    }

    #[tokio::test]
    async fn xai_builder_config_registry_typed_stream_request_options_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_xai::providers::xai::XaiBuilder::new(
            siumai_provider_xai::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/custom/v1/")
        .model("grok-beta")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client = siumai_provider_xai::providers::xai::XaiClient::from_config(
            siumai_provider_xai::providers::xai::XaiConfig::new("ctx-key")
                .with_base_url("https://example.com/custom/v1/")
                .with_model("grok-beta")
                .with_http_transport(Arc::new(config_transport.clone())),
        )
        .await
        .expect("build config client");

        let factory = crate::registry::factories::XAIProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "grok-beta",
                &BuildContext {
                    provider_id: Some("xai".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom/v1/".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model("grok-beta").with_xai_options(
            XaiOptions::new()
                .with_reasoning_effort("high")
                .with_search(XaiSearchParameters {
                    mode: SearchMode::On,
                    return_citations: Some(true),
                    max_search_results: Some(3),
                    from_date: Some("2025-01-01".to_string()),
                    to_date: Some("2025-01-31".to_string()),
                    sources: Some(vec![SearchSource {
                        source_type: SearchSourceType::Web,
                        country: Some("US".to_string()),
                        allowed_websites: Some(vec!["example.com".to_string()]),
                        excluded_websites: Some(vec!["blocked.example.com".to_string()]),
                        safe_search: Some(true),
                    }]),
                }),
        );

        let _ = builder_client.chat_stream_request(request.clone()).await;
        let _ = config_client.chat_stream_request(request.clone()).await;
        let _ = registry_client.chat_stream_request(request).await;

        let builder_req = builder_transport
            .take_stream()
            .expect("builder stream request");
        let config_req = config_transport
            .take_stream()
            .expect("config stream request");
        let registry_req = registry_transport
            .take_stream()
            .expect("registry stream request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.body["stream"], serde_json::json!(true));
        assert_eq!(
            header_value(&builder_req, "accept"),
            Some("text/event-stream".to_string())
        );
        assert_eq!(
            builder_req.body["reasoning_effort"],
            serde_json::json!("high")
        );
        assert_eq!(
            builder_req.body["search_parameters"]["mode"],
            serde_json::json!("on")
        );
        assert_eq!(
            builder_req.body["search_parameters"]["return_citations"],
            serde_json::json!(true)
        );
        assert_eq!(
            builder_req.body["search_parameters"]["max_search_results"],
            serde_json::json!(3)
        );
        assert_eq!(
            builder_req.body["search_parameters"]["from_date"],
            serde_json::json!("2025-01-01")
        );
        assert_eq!(
            builder_req.body["search_parameters"]["to_date"],
            serde_json::json!("2025-01-31")
        );
        assert_eq!(
            builder_req.body["search_parameters"]["sources"][0]["allowed_websites"],
            serde_json::json!(["example.com"])
        );
        assert_eq!(
            builder_req.body["search_parameters"]["sources"][0]["excluded_websites"],
            serde_json::json!(["blocked.example.com"])
        );
        assert_eq!(
            builder_req.body["search_parameters"]["sources"][0]["safe_search"],
            serde_json::json!(true)
        );
    }

    #[tokio::test]
    async fn xai_builder_config_registry_tool_choice_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_xai::providers::xai::XaiBuilder::new(
            siumai_provider_xai::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/custom/v1/")
        .model("grok-beta")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client = siumai_provider_xai::providers::xai::XaiClient::from_config(
            siumai_provider_xai::providers::xai::XaiConfig::new("ctx-key")
                .with_base_url("https://example.com/custom/v1/")
                .with_model("grok-beta")
                .with_http_transport(Arc::new(config_transport.clone())),
        )
        .await
        .expect("build config client");

        let factory = crate::registry::factories::XAIProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "grok-beta",
                &BuildContext {
                    provider_id: Some("xai".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom/v1/".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .model("grok-beta")
            .tools(vec![crate::types::Tool::function(
                "get_weather",
                "Get weather",
                serde_json::json!({
                    "type": "object",
                    "properties": { "location": { "type": "string" } },
                    "required": ["location"],
                    "additionalProperties": false
                }),
            )])
            .tool_choice(crate::types::ToolChoice::None)
            .build()
            .with_provider_option(
                "xai",
                serde_json::json!({
                    "tool_choice": "auto",
                    "reasoningEffort": "high"
                }),
            );

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.body["tool_choice"], serde_json::json!("none"));
        assert_eq!(
            builder_req.body["reasoning_effort"],
            serde_json::json!("high")
        );
        assert_eq!(
            builder_req.body["tools"][0]["function"]["name"],
            serde_json::json!("get_weather")
        );
    }

    #[tokio::test]
    async fn xai_builder_config_registry_response_format_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_xai::providers::xai::XaiBuilder::new(
            siumai_provider_xai::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/custom/v1/")
        .model("grok-beta")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client = siumai_provider_xai::providers::xai::XaiClient::from_config(
            siumai_provider_xai::providers::xai::XaiConfig::new("ctx-key")
                .with_base_url("https://example.com/custom/v1/")
                .with_model("grok-beta")
                .with_http_transport(Arc::new(config_transport.clone())),
        )
        .await
        .expect("build config client");

        let factory = crate::registry::factories::XAIProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "grok-beta",
                &BuildContext {
                    provider_id: Some("xai".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom/v1/".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let request = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .model("grok-beta")
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                schema.clone(),
            ))
            .build()
            .with_provider_option(
                "xai",
                serde_json::json!({
                    "response_format": { "type": "json_object" },
                    "reasoningEffort": "high"
                }),
            );

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(
            builder_req.body["response_format"],
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            })
        );
        assert_eq!(
            builder_req.body["reasoning_effort"],
            serde_json::json!("high")
        );
    }

    #[tokio::test]
    async fn xai_registry_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let xai_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "xai".to_string(),
            Arc::new(crate::registry::factories::XAIProviderFactory) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1/")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "xai",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/xai/v1/")
                    .fetch(Arc::new(xai_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .language_model("xai:grok-beta")
            .expect("build xai handle");

        let _ = handle
            .chat_request(
                make_chat_request_with_model("grok-beta")
                    .with_provider_option("xai", serde_json::json!({ "foo": "bar" })),
            )
            .await;

        let req = xai_transport.take().expect("captured xai request");
        assert!(global_transport.take().is_none());
        assert_eq!(req.headers.get(AUTHORIZATION).unwrap(), "Bearer ctx-key");
        assert_eq!(req.url, "https://example.com/xai/v1/chat/completions");
        assert_eq!(req.body["model"], serde_json::json!("grok-beta"));
        assert_eq!(req.body["foo"], serde_json::json!("bar"));
    }

    #[tokio::test]
    async fn xai_registry_stream_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let xai_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "xai".to_string(),
            Arc::new(crate::registry::factories::XAIProviderFactory) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1/")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "xai",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/xai/v1/")
                    .fetch(Arc::new(xai_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .language_model("xai:grok-beta")
            .expect("build xai handle");

        let _ = handle
            .chat_stream_request(
                make_chat_request_with_model("grok-beta")
                    .with_provider_option("xai", serde_json::json!({ "foo": "bar" })),
            )
            .await;

        let req = xai_transport
            .take_stream()
            .expect("captured xai stream request");
        assert!(global_transport.take().is_none());
        assert!(global_transport.take_stream().is_none());
        assert_eq!(
            header_value(&req, "authorization"),
            Some("Bearer ctx-key".to_string())
        );
        assert_eq!(
            header_value(&req, "accept"),
            Some("text/event-stream".to_string())
        );
        assert_eq!(req.url, "https://example.com/xai/v1/chat/completions");
        assert_eq!(req.body["stream"], serde_json::json!(true));
        assert_eq!(req.body["model"], serde_json::json!("grok-beta"));
        assert_eq!(req.body["foo"], serde_json::json!("bar"));
    }

    #[tokio::test]
    async fn xai_registry_override_chat_response_metadata_preserves_vendor_namespace() {
        let _lock = lock_env();

        let response_json = serde_json::json!({
            "id": "chatcmpl-xai-test",
            "object": "chat.completion",
            "created": 1_741_392_000,
            "model": "grok-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "hello from xai"
                    },
                    "finish_reason": "stop",
                    "logprobs": {
                        "content": [
                            {
                                "token": "hello",
                                "logprob": -0.1,
                                "bytes": [104, 101, 108, 108, 111],
                                "top_logprobs": []
                            }
                        ]
                    }
                }
            ],
            "sources": [
                {
                    "id": "src_1",
                    "source_type": "url",
                    "url": "https://example.com",
                    "title": "Example"
                }
            ]
        });

        let global_transport = CaptureTransport::default();
        let xai_transport = JsonSuccessTransport::new(response_json);
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "xai".to_string(),
            Arc::new(crate::registry::factories::XAIProviderFactory) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1/")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "xai",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/xai/v1/")
                    .fetch(Arc::new(xai_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let response = registry
            .language_model("xai:grok-4")
            .expect("build xai handle")
            .chat_request(make_chat_request_with_model("grok-4"))
            .await
            .expect("registry response ok");

        let root = response
            .provider_metadata
            .as_ref()
            .expect("registry provider metadata");
        assert!(root.get("xai").is_some());

        let metadata = response.xai_metadata().expect("xai metadata");
        assert_eq!(response.content_text(), Some("hello from xai"));
        assert_eq!(
            response.finish_reason,
            Some(crate::types::FinishReason::Stop)
        );
        assert_eq!(metadata.sources.as_ref().map(Vec::len), Some(1));
        assert_eq!(
            metadata
                .sources
                .as_ref()
                .and_then(|sources| sources.first())
                .map(|source| source.url.as_str()),
            Some("https://example.com")
        );

        let expected_logprobs = serde_json::json!([
            {
                "token": "hello",
                "logprob": -0.1,
                "bytes": [104, 101, 108, 108, 111],
                "top_logprobs": []
            }
        ]);
        assert_eq!(metadata.logprobs, Some(expected_logprobs));

        let req = xai_transport.take().expect("captured xai request");
        assert!(global_transport.take().is_none());
        assert_eq!(
            header_value(&req, "authorization"),
            Some("Bearer ctx-key".to_string())
        );
        assert_eq!(req.url, "https://example.com/xai/v1/chat/completions");
    }

    #[tokio::test]
    async fn xai_registry_override_stream_end_metadata_preserves_vendor_namespace() {
        let _lock = lock_env();

        let stream_body = br#"data: {"id":"1","model":"grok-4","created":1718345013,"sources":[{"id":"src_1","source_type":"url","url":"https://example.com","title":"Example"}],"choices":[{"index":0,"delta":{"content":"hello","role":"assistant"},"finish_reason":null}]}

data: {"id":"1","model":"grok-4","created":1718345013,"choices":[{"index":0,"delta":{"content":" from xai","role":null},"finish_reason":"stop","logprobs":{"content":[{"token":"hello","logprob":-0.1,"bytes":[104,101,108,108,111],"top_logprobs":[]}]}}]}

data: [DONE]

"#
        .to_vec();

        let global_transport = CaptureTransport::default();
        let xai_transport = SseSuccessTransport::new(stream_body);
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "xai".to_string(),
            Arc::new(crate::registry::factories::XAIProviderFactory) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global/v1/")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "xai",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/xai/v1/")
                    .fetch(Arc::new(xai_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let mut stream = registry
            .language_model("xai:grok-4")
            .expect("build xai handle")
            .chat_stream_request(make_chat_request_with_model("grok-4"))
            .await
            .expect("registry stream ok");

        use futures::StreamExt;

        let mut stream_end = None;
        while let Some(event) = stream.next().await {
            if let Ok(crate::streaming::ChatStreamEvent::StreamEnd { response }) = event {
                stream_end = Some(response);
                break;
            }
        }

        let response = stream_end.expect("registry stream end");
        let root = response
            .provider_metadata
            .as_ref()
            .expect("registry provider metadata");
        assert!(root.get("xai").is_some());

        let metadata = response.xai_metadata().expect("xai metadata");
        assert_eq!(response.content_text(), Some("hello from xai"));
        assert_eq!(
            response.finish_reason,
            Some(crate::types::FinishReason::Stop)
        );
        assert_eq!(metadata.sources.as_ref().map(Vec::len), Some(1));
        assert_eq!(
            metadata
                .sources
                .as_ref()
                .and_then(|sources| sources.first())
                .map(|source| source.url.as_str()),
            Some("https://example.com")
        );

        let expected_logprobs = serde_json::json!([
            {
                "token": "hello",
                "logprob": -0.1,
                "bytes": [104, 101, 108, 108, 111],
                "top_logprobs": []
            }
        ]);
        assert_eq!(metadata.logprobs, Some(expected_logprobs));

        let req = xai_transport
            .take_stream()
            .expect("captured xai stream request");
        assert!(global_transport.take().is_none());
        assert!(global_transport.take_stream().is_none());
        assert_eq!(
            header_value(&req, "authorization"),
            Some("Bearer ctx-key".to_string())
        );
        assert_eq!(
            header_value(&req, "accept"),
            Some("text/event-stream".to_string())
        );
        assert_eq!(req.url, "https://example.com/xai/v1/chat/completions");
    }
}

#[cfg(feature = "ollama")]
mod ollama_contract {
    use super::*;
    use crate::traits::ChatCapability;
    use siumai_provider_ollama::provider_options::OllamaOptions;
    use siumai_provider_ollama::providers::ollama::ext::request_options::OllamaChatRequestExt;
    use siumai_provider_ollama::standards::ollama::types::OllamaEmbeddingRequestExt;

    #[tokio::test]
    async fn ollama_factory_keeps_non_embedding_capabilities_deferred() {
        let _lock = lock_env();

        let factory = crate::registry::factories::OllamaProviderFactory;
        let caps = factory.capabilities();

        assert!(caps.supports("chat"));
        assert!(caps.supports("streaming"));
        assert!(caps.supports("tools"));
        assert!(caps.supports("embedding"));
        assert!(!caps.supports("image_generation"));
        assert!(!caps.supports("rerank"));
        assert!(!caps.supports("speech"));
        assert!(!caps.supports("transcription"));
        assert!(!caps.supports("audio"));
    }

    #[tokio::test]
    async fn ollama_factory_prefers_ctx_http_client_over_http_config() {
        let _lock = lock_env();

        let factory = crate::registry::factories::OllamaProviderFactory;

        let mut bad = HttpConfig::default();
        bad.proxy = Some("not-a-url".to_string());

        let ctx = BuildContext {
            provider_id: Some("ollama".to_string()),
            http_client: Some(reqwest::Client::new()),
            http_config: Some(bad),
            ..Default::default()
        };

        factory
            .language_model_with_ctx("llama3.2", &ctx)
            .await
            .expect("factory should prefer ctx.http_client over invalid http_config");
    }

    #[tokio::test]
    async fn ollama_factory_prefers_ctx_base_url_over_default() {
        let _lock = lock_env();

        let factory = crate::registry::factories::OllamaProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("ollama".to_string()),
            base_url: Some("http://example.com:11434".to_string()),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("llama3.2", &ctx)
            .await
            .expect("build client");

        let typed = client
            .as_any()
            .downcast_ref::<siumai_provider_ollama::providers::ollama::OllamaClient>()
            .expect("OllamaClient");
        assert_eq!(typed.base_url(), "http://example.com:11434");
    }

    #[tokio::test]
    async fn ollama_factory_does_not_require_api_key() {
        let _lock = lock_env();

        let factory = crate::registry::factories::OllamaProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("ollama".to_string()),
            ..Default::default()
        };

        factory
            .language_model_with_ctx("llama3.2", &ctx)
            .await
            .expect("ollama should build without api key");
    }

    #[tokio::test]
    async fn ollama_factory_returns_provider_owned_client() {
        let _lock = lock_env();

        let factory = crate::registry::factories::OllamaProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("ollama".to_string()),
            base_url: Some("http://example.com:11434/custom".to_string()),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("llama3.2", &ctx)
            .await
            .expect("build provider-owned Ollama client");

        let typed = client
            .as_any()
            .downcast_ref::<siumai_provider_ollama::providers::ollama::OllamaClient>()
            .expect("OllamaClient");
        assert!(client.as_embedding_capability().is_some());
        assert!(client.as_image_generation_capability().is_none());
        assert!(client.as_rerank_capability().is_none());
        assert!(client.as_speech_capability().is_none());
        assert!(client.as_transcription_capability().is_none());
        assert_eq!(typed.base_url(), "http://example.com:11434/custom");
        assert_eq!(crate::traits::ModelMetadata::provider_id(typed), "ollama");
        assert_eq!(crate::traits::ModelMetadata::model_id(typed), "llama3.2");
    }

    #[tokio::test]
    async fn ollama_factory_supports_native_text_family_path() {
        let _lock = lock_env();

        let factory = crate::registry::factories::OllamaProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("ollama".to_string()),
            base_url: Some("http://example.com:11434/custom/".to_string()),
            ..Default::default()
        };

        let model = factory
            .language_model_text_with_ctx("llama3.2", &ctx)
            .await
            .expect("build native text-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "ollama"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "llama3.2"
        );
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model.as_ref()),
            crate::traits::ModelSpecVersion::V1
        );
    }

    #[tokio::test]
    async fn ollama_factory_supports_native_embedding_family_path() {
        let _lock = lock_env();

        let factory = crate::registry::factories::OllamaProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("ollama".to_string()),
            base_url: Some("http://example.com:11434/custom/".to_string()),
            ..Default::default()
        };

        let model = factory
            .embedding_model_family_with_ctx("nomic-embed-text", &ctx)
            .await
            .expect("build native embedding-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "ollama"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "nomic-embed-text"
        );
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model.as_ref()),
            crate::traits::ModelSpecVersion::V1
        );
    }

    #[tokio::test]
    async fn ollama_builder_config_registry_embedding_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_ollama::providers::ollama::OllamaBuilder::new(
            siumai_provider_ollama::builder::BuilderBase::default(),
        )
        .base_url("http://example.com:11434/")
        .model("nomic-embed-text")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client = siumai_provider_ollama::providers::ollama::OllamaClient::from_config(
            siumai_provider_ollama::providers::ollama::OllamaConfig::builder()
                .base_url("http://example.com:11434/")
                .model("nomic-embed-text")
                .http_transport(Arc::new(config_transport.clone()))
                .build()
                .expect("build ollama config"),
        )
        .expect("build config client");

        let factory = crate::registry::factories::OllamaProviderFactory;
        let registry_client = factory
            .embedding_model_with_ctx(
                "nomic-embed-text",
                &BuildContext {
                    provider_id: Some("ollama".to_string()),
                    base_url: Some("http://example.com:11434/".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request =
            crate::types::EmbeddingRequest::new(vec!["text1".to_string(), "text2".to_string()])
                .with_model("nomic-embed-text")
                .with_ollama_config(
                    siumai_provider_ollama::standards::ollama::types::OllamaEmbeddingOptions::new()
                        .with_truncate(false)
                        .with_keep_alive("5m")
                        .with_option("temperature", serde_json::json!(0.1)),
                );

        let _ =
            crate::traits::EmbeddingExtensions::embed_with_config(&builder_client, request.clone())
                .await;
        let _ =
            crate::traits::EmbeddingExtensions::embed_with_config(&config_client, request.clone())
                .await;
        let typed_registry = registry_client
            .as_any()
            .downcast_ref::<siumai_provider_ollama::providers::ollama::OllamaClient>()
            .expect("OllamaClient");
        let _ =
            crate::traits::EmbeddingExtensions::embed_with_config(typed_registry, request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.url, "http://example.com:11434/api/embed");
        assert_eq!(
            builder_req.body["model"],
            serde_json::json!("nomic-embed-text")
        );
        assert_eq!(
            builder_req.body["input"],
            serde_json::json!(["text1", "text2"])
        );
        assert_eq!(builder_req.body["truncate"], serde_json::json!(false));
        assert_eq!(builder_req.body["keep_alive"], serde_json::json!("5m"));
        assert_eq!(
            builder_req.body["options"]["temperature"],
            serde_json::json!(0.1)
        );
    }

    #[tokio::test]
    async fn ollama_factory_rejects_unsupported_non_embedding_family_paths() {
        let _lock = lock_env();

        let factory = crate::registry::factories::OllamaProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("ollama".to_string()),
            base_url: Some("http://example.com:11434".to_string()),
            ..Default::default()
        };

        assert_unsupported_operation_contains(
            factory.image_model_with_ctx("llama3.2", &ctx).await,
            "image family path",
        );
        assert_unsupported_operation_contains(
            factory.speech_model_with_ctx("llama3.2", &ctx).await,
            "speech family path",
        );
        assert_unsupported_operation_contains(
            factory.transcription_model_with_ctx("llama3.2", &ctx).await,
            "transcription family path",
        );
        assert_unsupported_operation_contains(
            factory.reranking_model_with_ctx("llama3.2", &ctx).await,
            "reranking family path",
        );
    }

    #[tokio::test]
    async fn ollama_builder_config_registry_chat_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_ollama::providers::ollama::OllamaBuilder::new(
            siumai_provider_ollama::builder::BuilderBase::default(),
        )
        .base_url("http://example.com:11434/")
        .model("llama3.2")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client = siumai_provider_ollama::providers::ollama::OllamaClient::from_config(
            siumai_provider_ollama::providers::ollama::OllamaConfig::builder()
                .base_url("http://example.com:11434/")
                .model("llama3.2")
                .http_transport(Arc::new(config_transport.clone()))
                .build()
                .expect("build ollama config"),
        )
        .expect("build config client");

        let factory = crate::registry::factories::OllamaProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "llama3.2",
                &BuildContext {
                    provider_id: Some("ollama".to_string()),
                    base_url: Some("http://example.com:11434/".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "summary": { "type": "string" }
            },
            "required": ["summary"],
            "additionalProperties": false
        });

        let request = make_chat_request_with_model("llama3.2")
            .with_provider_option(
                "ollama",
                serde_json::json!({
                    "keep_alive": "1m",
                    "extra_params": {
                        "think": true,
                        "num_ctx": 4096
                    }
                }),
            )
            .with_response_format(crate::types::ResponseFormat::json_schema(schema.clone()));

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.body["keep_alive"], serde_json::json!("1m"));
        assert_eq!(builder_req.body["think"], serde_json::json!(true));
        assert_eq!(
            builder_req.body["options"]["num_ctx"],
            serde_json::json!(4096)
        );
        assert_eq!(builder_req.body["format"], schema);
    }

    #[tokio::test]
    async fn ollama_builder_config_registry_typed_request_options_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_ollama::providers::ollama::OllamaBuilder::new(
            siumai_provider_ollama::builder::BuilderBase::default(),
        )
        .base_url("http://example.com:11434/")
        .model("llama3.2")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client = siumai_provider_ollama::providers::ollama::OllamaClient::from_config(
            siumai_provider_ollama::providers::ollama::OllamaConfig::builder()
                .base_url("http://example.com:11434/")
                .model("llama3.2")
                .http_transport(Arc::new(config_transport.clone()))
                .build()
                .expect("build ollama config"),
        )
        .expect("build config client");

        let factory = crate::registry::factories::OllamaProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "llama3.2",
                &BuildContext {
                    provider_id: Some("ollama".to_string()),
                    base_url: Some("http://example.com:11434/".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model("llama3.2").with_ollama_options(
            OllamaOptions::new()
                .with_keep_alive("1m")
                .with_format("json")
                .with_raw_mode(true)
                .with_param("think", serde_json::json!(true))
                .with_param("num_ctx", serde_json::json!(4096)),
        );

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.body["keep_alive"], serde_json::json!("1m"));
        assert_eq!(builder_req.body["format"], serde_json::json!("json"));
        assert_eq!(builder_req.body["raw"], serde_json::json!(true));
        assert_eq!(builder_req.body["think"], serde_json::json!(true));
        assert_eq!(
            builder_req.body["options"]["num_ctx"],
            serde_json::json!(4096)
        );
    }

    #[tokio::test]
    async fn ollama_builder_config_registry_chat_stream_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_ollama::providers::ollama::OllamaBuilder::new(
            siumai_provider_ollama::builder::BuilderBase::default(),
        )
        .base_url("http://example.com:11434/")
        .model("llama3.2")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client = siumai_provider_ollama::providers::ollama::OllamaClient::from_config(
            siumai_provider_ollama::providers::ollama::OllamaConfig::builder()
                .base_url("http://example.com:11434/")
                .model("llama3.2")
                .http_transport(Arc::new(config_transport.clone()))
                .build()
                .expect("build ollama config"),
        )
        .expect("build config client");

        let factory = crate::registry::factories::OllamaProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "llama3.2",
                &BuildContext {
                    provider_id: Some("ollama".to_string()),
                    base_url: Some("http://example.com:11434/".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "summary": { "type": "string" }
            },
            "required": ["summary"],
            "additionalProperties": false
        });

        let request = make_chat_request_with_model("llama3.2")
            .with_provider_option(
                "ollama",
                serde_json::json!({
                    "keep_alive": "1m",
                    "extra_params": {
                        "think": true,
                        "num_ctx": 4096
                    }
                }),
            )
            .with_response_format(crate::types::ResponseFormat::json_schema(schema.clone()));

        let _ = builder_client.chat_stream_request(request.clone()).await;
        let _ = config_client.chat_stream_request(request.clone()).await;
        let _ = registry_client.chat_stream_request(request).await;

        let builder_req = builder_transport
            .take_stream()
            .expect("builder stream request");
        let config_req = config_transport
            .take_stream()
            .expect("config stream request");
        let registry_req = registry_transport
            .take_stream()
            .expect("registry stream request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.body["stream"], serde_json::json!(true));
        assert_eq!(builder_req.body["keep_alive"], serde_json::json!("1m"));
        assert_eq!(builder_req.body["think"], serde_json::json!(true));
        assert_eq!(
            builder_req.body["options"]["num_ctx"],
            serde_json::json!(4096)
        );
        assert_eq!(builder_req.body["format"], schema);
    }

    #[tokio::test]
    async fn ollama_builder_config_registry_typed_stream_request_options_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_ollama::providers::ollama::OllamaBuilder::new(
            siumai_provider_ollama::builder::BuilderBase::default(),
        )
        .base_url("http://example.com:11434/")
        .model("llama3.2")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client = siumai_provider_ollama::providers::ollama::OllamaClient::from_config(
            siumai_provider_ollama::providers::ollama::OllamaConfig::builder()
                .base_url("http://example.com:11434/")
                .model("llama3.2")
                .http_transport(Arc::new(config_transport.clone()))
                .build()
                .expect("build ollama config"),
        )
        .expect("build config client");

        let factory = crate::registry::factories::OllamaProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "llama3.2",
                &BuildContext {
                    provider_id: Some("ollama".to_string()),
                    base_url: Some("http://example.com:11434/".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model("llama3.2").with_ollama_options(
            OllamaOptions::new()
                .with_keep_alive("1m")
                .with_format("json")
                .with_raw_mode(true)
                .with_param("think", serde_json::json!(true))
                .with_param("num_ctx", serde_json::json!(4096)),
        );

        let _ = builder_client.chat_stream_request(request.clone()).await;
        let _ = config_client.chat_stream_request(request.clone()).await;
        let _ = registry_client.chat_stream_request(request).await;

        let builder_req = builder_transport
            .take_stream()
            .expect("builder stream request");
        let config_req = config_transport
            .take_stream()
            .expect("config stream request");
        let registry_req = registry_transport
            .take_stream()
            .expect("registry stream request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.body["stream"], serde_json::json!(true));
        assert_eq!(builder_req.body["keep_alive"], serde_json::json!("1m"));
        assert_eq!(builder_req.body["format"], serde_json::json!("json"));
        assert_eq!(builder_req.body["raw"], serde_json::json!(true));
        assert_eq!(builder_req.body["think"], serde_json::json!(true));
        assert_eq!(
            builder_req.body["options"]["num_ctx"],
            serde_json::json!(4096)
        );
    }

    #[tokio::test]
    async fn ollama_registry_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let ollama_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "ollama".to_string(),
            Arc::new(crate::registry::factories::OllamaProviderFactory) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_base_url("http://example.com:11434/global")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "ollama",
                crate::registry::ProviderBuildOverrides::default()
                    .with_base_url("http://example.com:11434/ollama")
                    .fetch(Arc::new(ollama_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .language_model("ollama:llama3.2")
            .expect("build ollama handle");

        let _ = handle
            .chat_request(make_chat_request_with_model("llama3.2"))
            .await;

        let req = ollama_transport.take().expect("captured ollama request");
        assert!(global_transport.take().is_none());
        assert_eq!(req.url, "http://example.com:11434/ollama/api/chat");
        assert_eq!(req.body["model"], serde_json::json!("llama3.2"));
    }

    #[tokio::test]
    async fn ollama_registry_stream_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let ollama_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "ollama".to_string(),
            Arc::new(crate::registry::factories::OllamaProviderFactory) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_base_url("http://example.com:11434/global")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "ollama",
                crate::registry::ProviderBuildOverrides::default()
                    .with_base_url("http://example.com:11434/ollama")
                    .fetch(Arc::new(ollama_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .language_model("ollama:llama3.2")
            .expect("build ollama handle");

        let _ = handle
            .chat_stream_request(make_chat_request_with_model("llama3.2"))
            .await;

        let req = ollama_transport
            .take_stream()
            .expect("captured ollama stream request");
        assert!(global_transport.take().is_none());
        assert!(global_transport.take_stream().is_none());
        assert_eq!(req.url, "http://example.com:11434/ollama/api/chat");
        assert_eq!(req.body["model"], serde_json::json!("llama3.2"));
        assert_eq!(req.body["stream"], serde_json::json!(true));
    }

    #[tokio::test]
    async fn ollama_registry_rejects_unsupported_non_embedding_handle_construction() {
        let _lock = lock_env();

        let transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "ollama".to_string(),
            Arc::new(crate::registry::factories::OllamaProviderFactory) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_base_url("http://example.com:11434")
            .fetch(Arc::new(transport.clone()))
            .auto_middleware(false)
            .build()
            .expect("build registry");

        assert_unsupported_operation_contains(
            registry.image_model("ollama:llama3.2"),
            "image_model handle",
        );
        assert_unsupported_operation_contains(
            registry.reranking_model("ollama:llama3.2"),
            "reranking_model handle",
        );

        assert_capture_transport_unused(&transport);
    }
}

#[cfg(feature = "minimaxi")]
mod minimaxi_contract {
    use super::*;
    use crate::traits::{
        ChatCapability, FileManagementCapability, ImageGenerationCapability,
        MusicGenerationCapability, SpeechCapability, VideoGenerationCapability,
    };
    use siumai_provider_minimaxi::provider_options::MinimaxiOptions;
    use siumai_provider_minimaxi::providers::minimaxi::ext::request_options::MinimaxiChatRequestExt;
    use wiremock::matchers::{method, path, query_param};
    use wiremock::{Mock, MockServer, Request as WiremockRequest, ResponseTemplate};

    #[derive(Clone, Default)]
    struct NoopInterceptor;

    impl crate::execution::http::interceptor::HttpInterceptor for NoopInterceptor {}

    fn wiremock_header_value(req: &WiremockRequest, key: &str) -> Option<String> {
        req.headers
            .get(key)
            .and_then(|v| v.to_str().ok())
            .map(ToString::to_string)
    }

    fn minimaxi_file_object_json() -> serde_json::Value {
        serde_json::json!({
            "file_id": 123,
            "filename": "hello.txt",
            "bytes": 5,
            "created_at": 1_700_000_000i64,
            "purpose": "t2a_async_input"
        })
    }

    fn normalize_wiremock_multipart_body(req: &WiremockRequest) -> String {
        let mut body = String::from_utf8_lossy(&req.body).into_owned();
        if let Some(content_type) = req.headers.get(CONTENT_TYPE).and_then(|v| v.to_str().ok())
            && let Some(boundary) = content_type.split("boundary=").nth(1)
        {
            body = body.replace(boundary.trim(), "<BOUNDARY>");
        }
        body
    }

    #[tokio::test]
    async fn minimaxi_factory_does_not_advertise_embedding_rerank_transcription() {
        let _lock = lock_env();

        let factory = crate::registry::factories::MiniMaxiProviderFactory;
        let caps = factory.capabilities();

        assert!(caps.supports("chat"));
        assert!(caps.supports("streaming"));
        assert!(caps.supports("tools"));
        assert!(caps.supports("image_generation"));
        assert!(caps.supports("speech"));
        assert!(caps.supports("file_management"));
        assert!(caps.supports("video"));
        assert!(caps.supports("music"));
        assert!(!caps.supports("embedding"));
        assert!(!caps.supports("rerank"));
        assert!(!caps.supports("transcription"));
        assert!(caps.supports("audio"));
    }

    #[tokio::test]
    async fn minimaxi_factory_prefers_ctx_http_client_over_http_config() {
        let _lock = lock_env();

        let factory = crate::registry::factories::MiniMaxiProviderFactory;

        let mut bad = HttpConfig::default();
        bad.proxy = Some("not-a-url".to_string());

        let ctx = BuildContext {
            provider_id: Some("minimaxi".to_string()),
            api_key: Some("ctx-key".to_string()),
            http_client: Some(reqwest::Client::new()),
            http_config: Some(bad),
            ..Default::default()
        };

        factory
            .language_model_with_ctx("MiniMax-M2", &ctx)
            .await
            .expect("factory should prefer ctx.http_client over invalid http_config");
    }

    #[tokio::test]
    async fn minimaxi_factory_uses_env_api_key_when_ctx_missing() {
        let _lock = lock_env();

        let _g = EnvGuard::set("MINIMAXI_API_KEY", "env-key");
        let factory = crate::registry::factories::MiniMaxiProviderFactory;

        let ctx = BuildContext {
            provider_id: Some("minimaxi".to_string()),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("MiniMax-M2", &ctx)
            .await
            .expect("build client via env api key");

        let typed = client
            .as_any()
            .downcast_ref::<siumai_provider_minimaxi::providers::minimaxi::client::MinimaxiClient>()
            .expect("MinimaxiClient");
        assert_eq!(typed.config().api_key, "env-key");
    }

    #[tokio::test]
    async fn minimaxi_factory_prefers_ctx_api_key_over_env() {
        let _lock = lock_env();

        let _g = EnvGuard::set("MINIMAXI_API_KEY", "env-key");
        let factory = crate::registry::factories::MiniMaxiProviderFactory;

        let ctx = BuildContext {
            provider_id: Some("minimaxi".to_string()),
            api_key: Some("ctx-key".to_string()),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("MiniMax-M2", &ctx)
            .await
            .expect("build client via ctx api key");

        let typed = client
            .as_any()
            .downcast_ref::<siumai_provider_minimaxi::providers::minimaxi::client::MinimaxiClient>()
            .expect("MinimaxiClient");
        assert_eq!(typed.config().api_key, "ctx-key");
    }

    #[tokio::test]
    async fn minimaxi_factory_prefers_ctx_base_url_over_default() {
        let _lock = lock_env();

        let factory = crate::registry::factories::MiniMaxiProviderFactory;

        let ctx = BuildContext {
            provider_id: Some("minimaxi".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom/".to_string()),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("MiniMax-M2", &ctx)
            .await
            .expect("build client");

        let typed = client
            .as_any()
            .downcast_ref::<siumai_provider_minimaxi::providers::minimaxi::client::MinimaxiClient>()
            .expect("MinimaxiClient");
        assert_eq!(typed.config().base_url, "https://example.com/custom");
        assert!(client.as_embedding_capability().is_none());
        assert!(client.as_rerank_capability().is_none());
        assert!(client.as_transcription_capability().is_none());
        assert!(client.as_image_generation_capability().is_some());
        assert!(client.as_speech_capability().is_some());
    }

    #[tokio::test]
    async fn minimaxi_factory_supports_native_text_family_path() {
        let _lock = lock_env();

        let factory = crate::registry::factories::MiniMaxiProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("minimaxi".to_string()),
            api_key: Some("ctx-key".to_string()),
            ..Default::default()
        };

        let model = factory
            .language_model_text_with_ctx("MiniMax-M2", &ctx)
            .await
            .expect("build native text-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "minimaxi"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "MiniMax-M2"
        );
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model.as_ref()),
            crate::traits::ModelSpecVersion::V1
        );
    }

    #[tokio::test]
    async fn minimaxi_factory_supports_native_image_family_path() {
        let _lock = lock_env();

        let factory = crate::registry::factories::MiniMaxiProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("minimaxi".to_string()),
            api_key: Some("ctx-key".to_string()),
            ..Default::default()
        };

        let model = factory
            .image_model_family_with_ctx("image-01", &ctx)
            .await
            .expect("build native image-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "minimaxi"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "image-01"
        );
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model.as_ref()),
            crate::traits::ModelSpecVersion::V1
        );
    }

    #[tokio::test]
    async fn minimaxi_factory_supports_native_speech_family_path() {
        let _lock = lock_env();

        let factory = crate::registry::factories::MiniMaxiProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("minimaxi".to_string()),
            api_key: Some("ctx-key".to_string()),
            ..Default::default()
        };

        let model = factory
            .speech_model_family_with_ctx("speech-2.6-hd", &ctx)
            .await
            .expect("build native speech-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "minimaxi"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "speech-2.6-hd"
        );
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model.as_ref()),
            crate::traits::ModelSpecVersion::V1
        );
    }

    #[tokio::test]
    async fn minimaxi_factory_rejects_unsupported_embedding_rerank_transcription_family_paths() {
        let _lock = lock_env();

        let factory = crate::registry::factories::MiniMaxiProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("minimaxi".to_string()),
            api_key: Some("ctx-key".to_string()),
            ..Default::default()
        };

        assert_unsupported_operation_contains(
            factory.embedding_model_with_ctx("MiniMax-M2", &ctx).await,
            "embedding family path",
        );
        assert_unsupported_operation_contains(
            factory
                .transcription_model_with_ctx("speech-2.6-hd", &ctx)
                .await,
            "transcription family path",
        );
        assert_unsupported_operation_contains(
            factory.reranking_model_with_ctx("MiniMax-M2", &ctx).await,
            "reranking family path",
        );
    }

    #[test]
    fn minimaxi_factory_source_declares_native_speech_family_override() {
        let source = include_str!("minimaxi.rs");

        assert!(source.contains("async fn speech_model_family_with_ctx("));
    }

    #[tokio::test]
    async fn minimaxi_factory_routes_through_provider_owned_config_surface() {
        let _lock = lock_env();

        let factory = crate::registry::factories::MiniMaxiProviderFactory;
        let transport = Arc::new(CaptureTransport::default());

        let mut http_config = HttpConfig::default();
        http_config.timeout = Some(std::time::Duration::from_secs(9));
        http_config.user_agent = Some("mini-test-agent".to_string());

        let mut common_params = crate::types::CommonParams::default();
        common_params.model = "MiniMax-M2".to_string();
        common_params.temperature = Some(0.42);
        common_params.max_tokens = Some(321);

        let ctx = BuildContext {
            provider_id: Some("minimaxi".to_string()),
            api_key: Some("ctx-key".to_string()),
            http_config: Some(http_config),
            common_params: Some(common_params),
            http_interceptors: vec![Arc::new(NoopInterceptor)],
            http_transport: Some(transport),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("MiniMax-M2", &ctx)
            .await
            .expect("build client");

        let typed = client
            .as_any()
            .downcast_ref::<siumai_provider_minimaxi::providers::minimaxi::client::MinimaxiClient>()
            .expect("MinimaxiClient");

        assert_eq!(typed.config().common_params.temperature, Some(0.42));
        assert_eq!(typed.config().common_params.max_tokens, Some(321));
        assert_eq!(
            typed.config().http_config.timeout,
            Some(std::time::Duration::from_secs(9))
        );
        assert_eq!(
            typed.config().http_config.user_agent.as_deref(),
            Some("mini-test-agent")
        );
        assert_eq!(typed.config().http_interceptors.len(), 1);
        assert!(typed.config().http_transport.is_some());
    }

    #[tokio::test]
    async fn minimaxi_builder_config_registry_chat_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_minimaxi::providers::minimaxi::MinimaxiBuilder::new(
            siumai_provider_minimaxi::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/custom")
        .model("MiniMax-M2")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client =
            siumai_provider_minimaxi::providers::minimaxi::client::MinimaxiClient::from_config(
                siumai_provider_minimaxi::providers::minimaxi::config::MinimaxiConfig::new(
                    "ctx-key",
                )
                .with_base_url("https://example.com/custom")
                .with_model("MiniMax-M2")
                .with_http_transport(Arc::new(config_transport.clone())),
            )
            .expect("build config client");

        let factory = crate::registry::factories::MiniMaxiProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "MiniMax-M2",
                &BuildContext {
                    provider_id: Some("minimaxi".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model("MiniMax-M2");

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(
            header_value(&builder_req, "authorization"),
            Some("Bearer ctx-key".to_string())
        );
        assert_eq!(builder_req.body["model"], serde_json::json!("MiniMax-M2"));
        assert_eq!(
            builder_req.body["messages"][0]["role"],
            serde_json::json!("user")
        );
    }

    #[tokio::test]
    async fn minimaxi_builder_config_registry_typed_request_options_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_minimaxi::providers::minimaxi::MinimaxiBuilder::new(
            siumai_provider_minimaxi::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/custom")
        .model("MiniMax-M2")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client =
            siumai_provider_minimaxi::providers::minimaxi::client::MinimaxiClient::from_config(
                siumai_provider_minimaxi::providers::minimaxi::config::MinimaxiConfig::new(
                    "ctx-key",
                )
                .with_base_url("https://example.com/custom")
                .with_model("MiniMax-M2")
                .with_http_transport(Arc::new(config_transport.clone())),
            )
            .expect("build config client");

        let factory = crate::registry::factories::MiniMaxiProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "MiniMax-M2",
                &BuildContext {
                    provider_id: Some("minimaxi".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model("MiniMax-M2").with_minimaxi_options(
            MinimaxiOptions::new()
                .with_reasoning_budget(4096)
                .with_json_object(),
        );

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(
            builder_req.body["thinking"],
            serde_json::json!({
                "type": "enabled",
                "budget_tokens": 4096
            })
        );
        assert_eq!(
            builder_req.body["output_format"],
            serde_json::json!({
                "type": "json_object"
            })
        );
    }

    #[tokio::test]
    async fn minimaxi_registry_handle_build_overrides_drive_provider_owned_chat_request() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let minimaxi_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "minimaxi".to_string(),
            Arc::new(crate::registry::factories::MiniMaxiProviderFactory)
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "minimaxi",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/custom")
                    .fetch(Arc::new(minimaxi_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .language_model("minimaxi:MiniMax-M2")
            .expect("build minimaxi handle");

        let _ = handle
            .chat_request(make_chat_request_with_model("MiniMax-M2"))
            .await;

        let req = minimaxi_transport.take().expect("captured request");
        assert_eq!(
            header_value(&req, "authorization"),
            Some("Bearer ctx-key".to_string())
        );
        assert!(global_transport.take().is_none());
        assert_eq!(req.body["model"], serde_json::json!("MiniMax-M2"));
        assert_eq!(req.body["messages"][0]["role"], serde_json::json!("user"));
    }

    #[tokio::test]
    async fn minimaxi_registry_file_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        async fn mount_upload_server() -> MockServer {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/v1/files/upload"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "file": minimaxi_file_object_json(),
                    "base_resp": {
                        "status_code": 0,
                        "status_msg": "success"
                    }
                })))
                .mount(&server)
                .await;
            server
        }

        let global_server = mount_upload_server().await;
        let minimaxi_server = mount_upload_server().await;

        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "minimaxi".to_string(),
            Arc::new(crate::registry::factories::MiniMaxiProviderFactory)
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url(format!("{}/anthropic/v1", global_server.uri()))
            .with_provider_build_overrides(
                "minimaxi",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url(format!("{}/anthropic/v1", minimaxi_server.uri())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .language_model("minimaxi:MiniMax-M2")
            .expect("build minimaxi handle");

        let uploaded = handle
            .upload_file(crate::types::FileUploadRequest {
                content: b"hello".to_vec(),
                filename: "hello.txt".to_string(),
                mime_type: Some("text/plain".to_string()),
                purpose: "t2a_async_input".to_string(),
                metadata: std::collections::HashMap::new(),
                http_config: None,
            })
            .await
            .expect("upload ok");

        assert_eq!(uploaded.id, "123");

        let global_requests = global_server
            .received_requests()
            .await
            .expect("global requests");
        assert!(global_requests.is_empty());

        let minimaxi_req = minimaxi_server
            .received_requests()
            .await
            .expect("minimaxi requests")
            .into_iter()
            .next()
            .expect("minimaxi upload request");

        assert_eq!(minimaxi_req.url.path(), "/v1/files/upload");
        assert_eq!(
            wiremock_header_value(&minimaxi_req, "authorization"),
            Some("Bearer ctx-key".to_string())
        );

        let body_text = normalize_wiremock_multipart_body(&minimaxi_req);
        assert!(body_text.contains("name=\"purpose\""));
        assert!(body_text.contains("t2a_async_input"));
        assert!(body_text.contains("name=\"file\"; filename=\"hello.txt\""));
        assert!(body_text.contains("Content-Type: text/plain"));
        assert!(body_text.contains("hello"));
    }

    #[tokio::test]
    async fn minimaxi_registry_list_files_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        async fn mount_list_server() -> MockServer {
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/v1/files/list"))
                .and(query_param("purpose", "t2a_async_input"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "files": [minimaxi_file_object_json()],
                    "base_resp": {
                        "status_code": 0,
                        "status_msg": "success"
                    }
                })))
                .mount(&server)
                .await;
            server
        }

        let global_server = mount_list_server().await;
        let minimaxi_server = mount_list_server().await;

        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "minimaxi".to_string(),
            Arc::new(crate::registry::factories::MiniMaxiProviderFactory)
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url(format!("{}/anthropic/v1", global_server.uri()))
            .with_provider_build_overrides(
                "minimaxi",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url(format!("{}/anthropic/v1", minimaxi_server.uri())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .language_model("minimaxi:MiniMax-M2")
            .expect("build minimaxi handle");

        let listed = handle
            .list_files(Some(crate::types::FileListQuery {
                purpose: Some("t2a_async_input".to_string()),
                limit: None,
                after: None,
                order: None,
                http_config: None,
            }))
            .await
            .expect("list files");

        assert_eq!(listed.files.len(), 1);

        let global_requests = global_server
            .received_requests()
            .await
            .expect("global requests");
        assert!(global_requests.is_empty());

        let minimaxi_req = minimaxi_server
            .received_requests()
            .await
            .expect("minimaxi requests")
            .into_iter()
            .next()
            .expect("minimaxi list request");

        assert_eq!(minimaxi_req.url.path(), "/v1/files/list");
        assert_eq!(minimaxi_req.url.query(), Some("purpose=t2a_async_input"));
        assert_eq!(
            wiremock_header_value(&minimaxi_req, "authorization"),
            Some("Bearer ctx-key".to_string())
        );
    }

    #[tokio::test]
    async fn minimaxi_registry_retrieve_file_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        async fn mount_retrieve_server() -> MockServer {
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/v1/files/retrieve"))
                .and(query_param("file_id", "123"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "file": {
                        "file_id": 123,
                        "filename": "hello.txt",
                        "bytes": 5,
                        "created_at": 1_700_000_000i64,
                        "purpose": "t2a_async_input",
                        "download_url": "https://example.com/download/123"
                    },
                    "base_resp": {
                        "status_code": 0,
                        "status_msg": "success"
                    }
                })))
                .mount(&server)
                .await;
            server
        }

        let global_server = mount_retrieve_server().await;
        let minimaxi_server = mount_retrieve_server().await;

        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "minimaxi".to_string(),
            Arc::new(crate::registry::factories::MiniMaxiProviderFactory)
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url(format!("{}/anthropic/v1", global_server.uri()))
            .with_provider_build_overrides(
                "minimaxi",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url(format!("{}/anthropic/v1", minimaxi_server.uri())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .language_model("minimaxi:MiniMax-M2")
            .expect("build minimaxi handle");

        let file = handle
            .retrieve_file("123".to_string())
            .await
            .expect("retrieve file");

        assert_eq!(file.id, "123");

        let global_requests = global_server
            .received_requests()
            .await
            .expect("global requests");
        assert!(global_requests.is_empty());

        let minimaxi_req = minimaxi_server
            .received_requests()
            .await
            .expect("minimaxi requests")
            .into_iter()
            .next()
            .expect("minimaxi retrieve request");

        assert_eq!(minimaxi_req.url.path(), "/v1/files/retrieve");
        assert_eq!(minimaxi_req.url.query(), Some("file_id=123"));
        assert_eq!(
            wiremock_header_value(&minimaxi_req, "authorization"),
            Some("Bearer ctx-key".to_string())
        );
    }

    #[tokio::test]
    async fn minimaxi_registry_get_file_content_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        async fn mount_content_server() -> MockServer {
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/v1/files/retrieve_content"))
                .and(query_param("file_id", "123"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_bytes(b"hello".to_vec())
                        .insert_header("content-type", "application/octet-stream"),
                )
                .mount(&server)
                .await;
            server
        }

        let global_server = mount_content_server().await;
        let minimaxi_server = mount_content_server().await;

        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "minimaxi".to_string(),
            Arc::new(crate::registry::factories::MiniMaxiProviderFactory)
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url(format!("{}/anthropic/v1", global_server.uri()))
            .with_provider_build_overrides(
                "minimaxi",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url(format!("{}/anthropic/v1", minimaxi_server.uri())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .language_model("minimaxi:MiniMax-M2")
            .expect("build minimaxi handle");

        let content = handle
            .get_file_content("123".to_string())
            .await
            .expect("get file content");

        assert_eq!(content, b"hello");

        let global_requests = global_server
            .received_requests()
            .await
            .expect("global requests");
        assert!(global_requests.is_empty());

        let minimaxi_req = minimaxi_server
            .received_requests()
            .await
            .expect("minimaxi requests")
            .into_iter()
            .next()
            .expect("minimaxi content request");

        assert_eq!(minimaxi_req.url.path(), "/v1/files/retrieve_content");
        assert_eq!(minimaxi_req.url.query(), Some("file_id=123"));
        assert_eq!(
            wiremock_header_value(&minimaxi_req, "authorization"),
            Some("Bearer ctx-key".to_string())
        );
    }

    #[tokio::test]
    async fn minimaxi_registry_delete_file_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        async fn mount_delete_server() -> MockServer {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/v1/files/delete"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "base_resp": {
                        "status_code": 0,
                        "status_msg": "success"
                    }
                })))
                .mount(&server)
                .await;
            server
        }

        let global_server = mount_delete_server().await;
        let minimaxi_server = mount_delete_server().await;

        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "minimaxi".to_string(),
            Arc::new(crate::registry::factories::MiniMaxiProviderFactory)
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url(format!("{}/anthropic/v1", global_server.uri()))
            .with_provider_build_overrides(
                "minimaxi",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url(format!("{}/anthropic/v1", minimaxi_server.uri())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .language_model("minimaxi:MiniMax-M2")
            .expect("build minimaxi handle");

        let deleted = handle
            .delete_file("123:t2a_async_input".to_string())
            .await
            .expect("delete file");

        assert!(deleted.deleted);

        let global_requests = global_server
            .received_requests()
            .await
            .expect("global requests");
        assert!(global_requests.is_empty());

        let minimaxi_req = minimaxi_server
            .received_requests()
            .await
            .expect("minimaxi requests")
            .into_iter()
            .next()
            .expect("minimaxi delete request");

        let body: serde_json::Value =
            serde_json::from_slice(&minimaxi_req.body).expect("minimaxi delete body");
        assert_eq!(minimaxi_req.url.path(), "/v1/files/delete");
        assert_eq!(
            wiremock_header_value(&minimaxi_req, "authorization"),
            Some("Bearer ctx-key".to_string())
        );
        assert_eq!(
            body,
            serde_json::json!({
                "file_id": 123,
                "purpose": "t2a_async_input"
            })
        );
    }

    #[tokio::test]
    async fn minimaxi_registry_video_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        async fn mount_video_server() -> MockServer {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/v1/video_generation"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "task_id": "task-123",
                    "base_resp": {
                        "status_code": 0,
                        "status_msg": "success"
                    }
                })))
                .mount(&server)
                .await;
            server
        }

        let global_server = mount_video_server().await;
        let minimaxi_server = mount_video_server().await;

        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "minimaxi".to_string(),
            Arc::new(crate::registry::factories::MiniMaxiProviderFactory)
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url(format!("{}/anthropic/v1", global_server.uri()))
            .with_provider_build_overrides(
                "minimaxi",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url(format!("{}/anthropic/v1", minimaxi_server.uri())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .language_model("minimaxi:hailuo-2.3")
            .expect("build minimaxi handle");

        let created = handle
            .create_video_task(
                siumai_provider_minimaxi::providers::minimaxi::ext::video::MinimaxiVideoRequestBuilder::new(
                    "hailuo-2.3",
                    "tiny robot in rain",
                )
                .duration(10)
                .resolution("1080P")
                .prompt_optimizer(true)
                .fast_pretreatment(false)
                .callback_url("https://example.com/callback")
                .watermark(false)
                .build(),
            )
            .await
            .expect("create video task");

        assert_eq!(created.task_id, "task-123");

        let global_requests = global_server
            .received_requests()
            .await
            .expect("global requests");
        assert!(global_requests.is_empty());

        let minimaxi_req = minimaxi_server
            .received_requests()
            .await
            .expect("minimaxi requests")
            .into_iter()
            .next()
            .expect("minimaxi video request");

        assert_eq!(minimaxi_req.url.path(), "/v1/video_generation");
        assert_eq!(
            wiremock_header_value(&minimaxi_req, "authorization"),
            Some("Bearer ctx-key".to_string())
        );
        let body: serde_json::Value =
            serde_json::from_slice(&minimaxi_req.body).expect("minimaxi video body");
        assert_eq!(body["model"], serde_json::json!("hailuo-2.3"));
        assert_eq!(body["prompt"], serde_json::json!("tiny robot in rain"));
        assert_eq!(body["duration"], serde_json::json!(10));
        assert_eq!(body["resolution"], serde_json::json!("1080P"));
    }

    #[tokio::test]
    async fn minimaxi_registry_music_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        async fn mount_music_server() -> MockServer {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/v1/music_generation"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "data": {
                        "audio": "48656c6c6f",
                        "status": 2
                    },
                    "extra_info": {
                        "music_duration": 12000,
                        "music_sample_rate": 48000,
                        "music_channel": 2,
                        "bitrate": 320000,
                        "music_size": 5
                    }
                })))
                .mount(&server)
                .await;
            server
        }

        let global_server = mount_music_server().await;
        let minimaxi_server = mount_music_server().await;

        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "minimaxi".to_string(),
            Arc::new(crate::registry::factories::MiniMaxiProviderFactory)
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url(format!("{}/anthropic/v1", global_server.uri()))
            .with_provider_build_overrides(
                "minimaxi",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url(format!("{}/anthropic/v1", minimaxi_server.uri())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .language_model("minimaxi:music-2.0")
            .expect("build minimaxi handle");

        let generated = handle
            .generate_music(
                siumai_provider_minimaxi::providers::minimaxi::ext::music::MinimaxiMusicRequestBuilder::new(
                    "cinematic ambient with piano",
                )
                .lyrics_template()
                .sample_rate(48000)
                .bitrate(320000)
                .format("wav")
                .build(),
            )
            .await
            .expect("generate music");

        assert_eq!(generated.audio_data, b"Hello");

        let global_requests = global_server
            .received_requests()
            .await
            .expect("global requests");
        assert!(global_requests.is_empty());

        let minimaxi_req = minimaxi_server
            .received_requests()
            .await
            .expect("minimaxi requests")
            .into_iter()
            .next()
            .expect("minimaxi music request");

        assert_eq!(minimaxi_req.url.path(), "/v1/music_generation");
        assert_eq!(
            wiremock_header_value(&minimaxi_req, "authorization"),
            Some("Bearer ctx-key".to_string())
        );
        let body: serde_json::Value =
            serde_json::from_slice(&minimaxi_req.body).expect("minimaxi music body");
        assert_eq!(
            body["prompt"],
            serde_json::json!("cinematic ambient with piano")
        );
        assert_eq!(
            body["lyrics"],
            serde_json::json!("[Intro]\n[Main]\n[Outro]")
        );
        assert_eq!(
            body["audio_setting"]["sample_rate"],
            serde_json::json!(48000)
        );
        assert_eq!(body["audio_setting"]["bitrate"], serde_json::json!(320000));
        assert_eq!(body["audio_setting"]["format"], serde_json::json!("wav"));
    }

    #[tokio::test]
    async fn minimaxi_registry_image_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let minimaxi_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "minimaxi".to_string(),
            Arc::new(crate::registry::factories::MiniMaxiProviderFactory)
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "minimaxi",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/custom")
                    .fetch(Arc::new(minimaxi_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .image_model("minimaxi:image-01")
            .expect("build minimaxi image handle");

        let _ = ImageGenerationCapability::generate_images(
            &handle,
            crate::types::ImageGenerationRequest {
                prompt: "a tiny green robot".to_string(),
                negative_prompt: Some("blurry".to_string()),
                size: Some("1024x1024".to_string()),
                count: 1,
                model: Some("image-01".to_string()),
                quality: None,
                style: None,
                seed: None,
                steps: None,
                guidance_scale: None,
                enhance_prompt: None,
                response_format: Some("url".to_string()),
                extra_params: Default::default(),
                provider_options_map: Default::default(),
                http_config: None,
            },
        )
        .await;

        let req = minimaxi_transport.take().expect("captured request");
        assert_eq!(
            header_value(&req, "authorization"),
            Some("Bearer ctx-key".to_string())
        );
        assert!(global_transport.take().is_none());
        assert_eq!(req.url, "https://example.com/custom/v1/image_generation");
        assert_eq!(req.body["model"], serde_json::json!("image-01"));
        assert_eq!(req.body["prompt"], serde_json::json!("a tiny green robot"));
        assert_eq!(req.body["size"], serde_json::json!("1024x1024"));
        assert_eq!(req.body["n"], serde_json::json!(1));
        assert_eq!(req.body["response_format"], serde_json::json!("url"));
    }

    #[tokio::test]
    async fn minimaxi_registry_speech_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let minimaxi_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "minimaxi".to_string(),
            Arc::new(crate::registry::factories::MiniMaxiProviderFactory)
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "minimaxi",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/custom")
                    .fetch(Arc::new(minimaxi_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .speech_model("minimaxi:speech-2.6-hd")
            .expect("build minimaxi speech handle");

        let _ = SpeechCapability::tts(
            &handle,
            crate::types::TtsRequest::new("Hello".to_string())
                .with_voice("male-qn-qingse".to_string())
                .with_format("mp3".to_string())
                .with_provider_option(
                    "minimaxi",
                    serde_json::json!({
                        "emotion": "happy",
                        "pitch": 5,
                        "sample_rate": 32000,
                        "bitrate": 128000,
                        "channel": 1,
                        "subtitle_enable": true
                    }),
                ),
        )
        .await;

        let req = minimaxi_transport.take().expect("captured request");
        assert_eq!(
            header_value(&req, "authorization"),
            Some("Bearer ctx-key".to_string())
        );
        assert!(global_transport.take().is_none());
        assert_eq!(req.url, "https://example.com/custom/v1/t2a_v2");
        assert_eq!(req.body["model"], serde_json::json!("speech-2.6-hd"));
        assert_eq!(req.body["text"], serde_json::json!("Hello"));
        assert_eq!(
            req.body["voice_setting"]["voice_id"],
            serde_json::json!("male-qn-qingse")
        );
        assert_eq!(
            req.body["voice_setting"]["emotion"],
            serde_json::json!("happy")
        );
        assert_eq!(req.body["voice_setting"]["pitch"], serde_json::json!(5));
        assert_eq!(
            req.body["audio_setting"]["sample_rate"],
            serde_json::json!(32000)
        );
        assert_eq!(
            req.body["audio_setting"]["bitrate"],
            serde_json::json!(128000)
        );
        assert_eq!(req.body["subtitle_enable"], serde_json::json!(true));
    }

    #[tokio::test]
    async fn minimaxi_registry_query_video_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        async fn mount_query_server() -> MockServer {
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/v1/query/video_generation"))
                .and(query_param("task_id", "task-123"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "task_id": "task-123",
                    "status": "Success",
                    "file_id": "file-123",
                    "video_width": 1920,
                    "video_height": 1080,
                    "base_resp": {
                        "status_code": 0,
                        "status_msg": "success"
                    }
                })))
                .mount(&server)
                .await;
            server
        }

        let global_server = mount_query_server().await;
        let minimaxi_server = mount_query_server().await;

        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "minimaxi".to_string(),
            Arc::new(crate::registry::factories::MiniMaxiProviderFactory)
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url(format!("{}/anthropic/v1", global_server.uri()))
            .with_provider_build_overrides(
                "minimaxi",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url(format!("{}/anthropic/v1", minimaxi_server.uri())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .language_model("minimaxi:hailuo-2.3")
            .expect("build minimaxi handle");

        let queried =
            crate::traits::VideoGenerationCapability::query_video_task(&handle, "task-123")
                .await
                .expect("query video");

        assert_eq!(queried.task_id, "task-123");
        assert_eq!(queried.file_id.as_deref(), Some("file-123"));

        let global_requests = global_server
            .received_requests()
            .await
            .expect("global requests");
        assert!(global_requests.is_empty());

        let minimaxi_req = minimaxi_server
            .received_requests()
            .await
            .expect("minimaxi requests")
            .into_iter()
            .next()
            .expect("minimaxi query request");

        assert_eq!(minimaxi_req.url.path(), "/v1/query/video_generation");
        assert_eq!(minimaxi_req.url.query(), Some("task_id=task-123"));
        assert_eq!(
            wiremock_header_value(&minimaxi_req, "authorization"),
            Some("Bearer ctx-key".to_string())
        );
    }

    #[tokio::test]
    async fn minimaxi_registry_rejects_unsupported_embedding_rerank_transcription_handle_construction()
     {
        let _lock = lock_env();

        let transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "minimaxi".to_string(),
            Arc::new(crate::registry::factories::MiniMaxiProviderFactory)
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("ctx-key")
            .with_base_url("https://example.com/minimaxi")
            .fetch(Arc::new(transport.clone()))
            .auto_middleware(false)
            .build()
            .expect("build registry");

        assert_unsupported_operation_contains(
            registry.embedding_model("minimaxi:MiniMax-M2"),
            "embedding_model handle",
        );
        assert_unsupported_operation_contains(
            registry.reranking_model("minimaxi:MiniMax-M2"),
            "reranking_model handle",
        );
        assert_unsupported_operation_contains(
            registry.transcription_model("minimaxi:speech-2.6-hd"),
            "transcription_model handle",
        );

        assert_capture_transport_unused(&transport);
    }

    #[tokio::test]
    async fn minimaxi_builder_config_registry_chat_stream_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_minimaxi::providers::minimaxi::MinimaxiBuilder::new(
            siumai_provider_minimaxi::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/custom")
        .model("MiniMax-M2")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client =
            siumai_provider_minimaxi::providers::minimaxi::client::MinimaxiClient::from_config(
                siumai_provider_minimaxi::providers::minimaxi::config::MinimaxiConfig::new(
                    "ctx-key",
                )
                .with_base_url("https://example.com/custom")
                .with_model("MiniMax-M2")
                .with_http_transport(Arc::new(config_transport.clone())),
            )
            .expect("build config client");

        let factory = crate::registry::factories::MiniMaxiProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "MiniMax-M2",
                &BuildContext {
                    provider_id: Some("minimaxi".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model("MiniMax-M2");

        let _ = builder_client.chat_stream_request(request.clone()).await;
        let _ = config_client.chat_stream_request(request.clone()).await;
        let _ = registry_client.chat_stream_request(request).await;

        let builder_req = builder_transport
            .take_stream()
            .expect("builder stream request");
        let config_req = config_transport
            .take_stream()
            .expect("config stream request");
        let registry_req = registry_transport
            .take_stream()
            .expect("registry stream request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(
            header_value(&builder_req, "authorization"),
            Some("Bearer ctx-key".to_string())
        );
        assert_eq!(
            header_value(&builder_req, "accept"),
            Some("text/event-stream".to_string())
        );
        assert_eq!(builder_req.body["stream"], serde_json::json!(true));
        assert_eq!(builder_req.body["model"], serde_json::json!("MiniMax-M2"));
    }

    #[tokio::test]
    async fn minimaxi_builder_config_registry_typed_stream_request_options_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_minimaxi::providers::minimaxi::MinimaxiBuilder::new(
            siumai_provider_minimaxi::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/custom")
        .model("MiniMax-M2")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client =
            siumai_provider_minimaxi::providers::minimaxi::client::MinimaxiClient::from_config(
                siumai_provider_minimaxi::providers::minimaxi::config::MinimaxiConfig::new(
                    "ctx-key",
                )
                .with_base_url("https://example.com/custom")
                .with_model("MiniMax-M2")
                .with_http_transport(Arc::new(config_transport.clone())),
            )
            .expect("build config client");

        let factory = crate::registry::factories::MiniMaxiProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "MiniMax-M2",
                &BuildContext {
                    provider_id: Some("minimaxi".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model("MiniMax-M2").with_minimaxi_options(
            MinimaxiOptions::new()
                .with_reasoning_budget(4096)
                .with_json_object(),
        );

        let _ = builder_client.chat_stream_request(request.clone()).await;
        let _ = config_client.chat_stream_request(request.clone()).await;
        let _ = registry_client.chat_stream_request(request).await;

        let builder_req = builder_transport
            .take_stream()
            .expect("builder stream request");
        let config_req = config_transport
            .take_stream()
            .expect("config stream request");
        let registry_req = registry_transport
            .take_stream()
            .expect("registry stream request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(
            header_value(&builder_req, "accept"),
            Some("text/event-stream".to_string())
        );
        assert_eq!(builder_req.body["stream"], serde_json::json!(true));
        assert_eq!(
            builder_req.body["thinking"],
            serde_json::json!({
                "type": "enabled",
                "budget_tokens": 4096
            })
        );
        assert_eq!(
            builder_req.body["output_format"],
            serde_json::json!({
                "type": "json_object"
            })
        );
    }

    #[tokio::test]
    async fn minimaxi_builder_config_registry_tts_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_minimaxi::providers::minimaxi::MinimaxiBuilder::new(
            siumai_provider_minimaxi::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/custom")
        .model("speech-2.6-hd")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client =
            siumai_provider_minimaxi::providers::minimaxi::client::MinimaxiClient::from_config(
                siumai_provider_minimaxi::providers::minimaxi::config::MinimaxiConfig::new(
                    "ctx-key",
                )
                .with_base_url("https://example.com/custom")
                .with_model("speech-2.6-hd")
                .with_http_transport(Arc::new(config_transport.clone())),
            )
            .expect("build config client");

        let factory = crate::registry::factories::MiniMaxiProviderFactory;
        let registry_client = factory
            .speech_model_with_ctx(
                "speech-2.6-hd",
                &BuildContext {
                    provider_id: Some("minimaxi".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = crate::types::TtsRequest::new("Hello".to_string())
            .with_voice("male-qn-qingse".to_string())
            .with_format("mp3".to_string())
            .with_provider_option(
                "minimaxi",
                serde_json::json!({
                    "emotion": "happy",
                    "pitch": 5,
                    "sample_rate": 32000,
                    "bitrate": 128000,
                    "channel": 1,
                    "subtitle_enable": true
                }),
            );

        let _ = crate::traits::SpeechCapability::tts(&builder_client, request.clone()).await;
        let _ = crate::traits::SpeechCapability::tts(&config_client, request.clone()).await;
        let _ = registry_client
            .as_speech_capability()
            .expect("registry speech capability")
            .tts(request)
            .await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.url, "https://example.com/custom/v1/t2a_v2");
        assert_eq!(
            header_value(&builder_req, "authorization"),
            Some("Bearer ctx-key".to_string())
        );
        assert_eq!(
            builder_req.body["model"],
            serde_json::json!("speech-2.6-hd")
        );
        assert_eq!(builder_req.body["text"], serde_json::json!("Hello"));
        assert_eq!(
            builder_req.body["voice_setting"]["voice_id"],
            serde_json::json!("male-qn-qingse")
        );
        assert_eq!(
            builder_req.body["voice_setting"]["emotion"],
            serde_json::json!("happy")
        );
        assert_eq!(
            builder_req.body["voice_setting"]["pitch"],
            serde_json::json!(5)
        );
        assert_eq!(
            builder_req.body["audio_setting"]["sample_rate"],
            serde_json::json!(32000)
        );
        assert_eq!(
            builder_req.body["audio_setting"]["bitrate"],
            serde_json::json!(128000)
        );
        assert_eq!(builder_req.body["subtitle_enable"], serde_json::json!(true));
    }

    #[tokio::test]
    async fn minimaxi_builder_config_registry_image_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_minimaxi::providers::minimaxi::MinimaxiBuilder::new(
            siumai_provider_minimaxi::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/custom")
        .model("image-01")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client =
            siumai_provider_minimaxi::providers::minimaxi::client::MinimaxiClient::from_config(
                siumai_provider_minimaxi::providers::minimaxi::config::MinimaxiConfig::new(
                    "ctx-key",
                )
                .with_base_url("https://example.com/custom")
                .with_model("image-01")
                .with_http_transport(Arc::new(config_transport.clone())),
            )
            .expect("build config client");

        let factory = crate::registry::factories::MiniMaxiProviderFactory;
        let registry_client = factory
            .image_model_with_ctx(
                "image-01",
                &BuildContext {
                    provider_id: Some("minimaxi".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = crate::types::ImageGenerationRequest {
            prompt: "a tiny green robot".to_string(),
            negative_prompt: Some("blurry".to_string()),
            size: Some("1024x1024".to_string()),
            count: 1,
            model: Some("image-01".to_string()),
            quality: None,
            style: None,
            seed: None,
            steps: None,
            guidance_scale: None,
            enhance_prompt: None,
            response_format: Some("url".to_string()),
            extra_params: Default::default(),
            provider_options_map: Default::default(),
            http_config: None,
        };

        let _ = crate::traits::ImageGenerationCapability::generate_images(
            &builder_client,
            request.clone(),
        )
        .await;
        let _ = crate::traits::ImageGenerationCapability::generate_images(
            &config_client,
            request.clone(),
        )
        .await;
        let _ = registry_client
            .as_image_generation_capability()
            .expect("registry image capability")
            .generate_images(request)
            .await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(
            builder_req.url,
            "https://example.com/custom/v1/image_generation"
        );
        assert_eq!(builder_req.body["model"], serde_json::json!("image-01"));
        assert_eq!(
            builder_req.body["prompt"],
            serde_json::json!("a tiny green robot")
        );
        assert_eq!(builder_req.body["size"], serde_json::json!("1024x1024"));
        assert_eq!(builder_req.body["n"], serde_json::json!(1));
        assert_eq!(
            builder_req.body["response_format"],
            serde_json::json!("url")
        );
    }

    #[tokio::test]
    async fn minimaxi_builder_config_registry_create_video_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_minimaxi::providers::minimaxi::MinimaxiBuilder::new(
            siumai_provider_minimaxi::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/anthropic/v1")
        .model("hailuo-2.3")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client =
            siumai_provider_minimaxi::providers::minimaxi::client::MinimaxiClient::from_config(
                siumai_provider_minimaxi::providers::minimaxi::config::MinimaxiConfig::new(
                    "ctx-key",
                )
                .with_base_url("https://example.com/anthropic/v1")
                .with_model("hailuo-2.3")
                .with_http_transport(Arc::new(config_transport.clone())),
            )
            .expect("build config client");

        let factory = crate::registry::factories::MiniMaxiProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "hailuo-2.3",
                &BuildContext {
                    provider_id: Some("minimaxi".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/anthropic/v1".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = siumai_provider_minimaxi::providers::minimaxi::ext::video::MinimaxiVideoRequestBuilder::new(
            "hailuo-2.3",
            "tiny robot in rain",
        )
        .duration(10)
        .resolution("1080P")
        .prompt_optimizer(true)
        .fast_pretreatment(false)
        .callback_url("https://example.com/callback")
        .watermark(false)
        .build();

        let _ = crate::traits::VideoGenerationCapability::create_video_task(
            &builder_client,
            request.clone(),
        )
        .await;
        let _ = crate::traits::VideoGenerationCapability::create_video_task(
            &config_client,
            request.clone(),
        )
        .await;
        let _ = registry_client
            .as_video_generation_capability()
            .expect("registry video capability")
            .create_video_task(request)
            .await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.url, "https://example.com/v1/video_generation");
        assert_eq!(builder_req.body["model"], serde_json::json!("hailuo-2.3"));
        assert_eq!(
            builder_req.body["prompt"],
            serde_json::json!("tiny robot in rain")
        );
        assert_eq!(builder_req.body["duration"], serde_json::json!(10));
        assert_eq!(builder_req.body["resolution"], serde_json::json!("1080P"));
    }

    #[tokio::test]
    async fn minimaxi_builder_config_registry_generate_music_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_minimaxi::providers::minimaxi::MinimaxiBuilder::new(
            siumai_provider_minimaxi::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/anthropic/v1")
        .model("music-2.0")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client =
            siumai_provider_minimaxi::providers::minimaxi::client::MinimaxiClient::from_config(
                siumai_provider_minimaxi::providers::minimaxi::config::MinimaxiConfig::new(
                    "ctx-key",
                )
                .with_base_url("https://example.com/anthropic/v1")
                .with_model("music-2.0")
                .with_http_transport(Arc::new(config_transport.clone())),
            )
            .expect("build config client");

        let factory = crate::registry::factories::MiniMaxiProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "music-2.0",
                &BuildContext {
                    provider_id: Some("minimaxi".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/anthropic/v1".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = siumai_provider_minimaxi::providers::minimaxi::ext::music::MinimaxiMusicRequestBuilder::new(
            "ambient piano",
        )
        .lyrics_template()
        .sample_rate(48_000)
        .bitrate(320_000)
        .format("wav")
        .build();

        let _ = crate::traits::MusicGenerationCapability::generate_music(
            &builder_client,
            request.clone(),
        )
        .await;
        let _ = crate::traits::MusicGenerationCapability::generate_music(
            &config_client,
            request.clone(),
        )
        .await;
        let _ = registry_client
            .as_music_generation_capability()
            .expect("registry music capability")
            .generate_music(request)
            .await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(builder_req.url, "https://example.com/v1/music_generation");
        assert_eq!(builder_req.body["model"], serde_json::json!("music-2.0"));
        assert_eq!(
            builder_req.body["prompt"],
            serde_json::json!("ambient piano")
        );
        assert_eq!(
            builder_req.body["lyrics"],
            serde_json::json!("[Intro]\n[Main]\n[Outro]")
        );
        assert_eq!(
            builder_req.body["audio_setting"]["sample_rate"],
            serde_json::json!(48_000)
        );
        assert_eq!(
            builder_req.body["audio_setting"]["bitrate"],
            serde_json::json!(320_000)
        );
        assert_eq!(
            builder_req.body["audio_setting"]["format"],
            serde_json::json!("wav")
        );
    }

    #[tokio::test]
    async fn minimaxi_builder_config_registry_query_video_request_are_equivalent() {
        let _lock = lock_env();

        async fn mount_query_server() -> MockServer {
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/v1/query/video_generation"))
                .and(query_param("task_id", "task-123"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "task_id": "task-123",
                    "status": "Success",
                    "file_id": "file-123",
                    "video_width": 1920,
                    "video_height": 1080,
                    "base_resp": {
                        "status_code": 0,
                        "status_msg": "success"
                    }
                })))
                .mount(&server)
                .await;
            server
        }

        let builder_server = mount_query_server().await;
        let config_server = mount_query_server().await;
        let registry_server = mount_query_server().await;

        let builder_client = siumai_provider_minimaxi::providers::minimaxi::MinimaxiBuilder::new(
            siumai_provider_minimaxi::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url(format!("{}/anthropic/v1", builder_server.uri()))
        .model("hailuo-2.3")
        .build()
        .await
        .expect("build builder client");

        let config_client =
            siumai_provider_minimaxi::providers::minimaxi::client::MinimaxiClient::from_config(
                siumai_provider_minimaxi::providers::minimaxi::config::MinimaxiConfig::new(
                    "ctx-key",
                )
                .with_base_url(format!("{}/anthropic/v1", config_server.uri()))
                .with_model("hailuo-2.3"),
            )
            .expect("build config client");

        let factory = crate::registry::factories::MiniMaxiProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "hailuo-2.3",
                &BuildContext {
                    provider_id: Some("minimaxi".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some(format!("{}/anthropic/v1", registry_server.uri())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let _ =
            crate::traits::VideoGenerationCapability::query_video_task(&builder_client, "task-123")
                .await
                .expect("builder query");
        let _ =
            crate::traits::VideoGenerationCapability::query_video_task(&config_client, "task-123")
                .await
                .expect("config query");
        let _ = registry_client
            .as_video_generation_capability()
            .expect("registry video capability")
            .query_video_task("task-123")
            .await
            .expect("registry query");

        let builder_req = builder_server
            .received_requests()
            .await
            .expect("builder requests")
            .into_iter()
            .next()
            .expect("builder query request");
        let config_req = config_server
            .received_requests()
            .await
            .expect("config requests")
            .into_iter()
            .next()
            .expect("config query request");
        let registry_req = registry_server
            .received_requests()
            .await
            .expect("registry requests")
            .into_iter()
            .next()
            .expect("registry query request");

        assert_eq!(builder_req.url.path(), "/v1/query/video_generation");
        assert_eq!(builder_req.url.query(), Some("task_id=task-123"));
        assert_eq!(config_req.url.path(), builder_req.url.path());
        assert_eq!(config_req.url.query(), builder_req.url.query());
        assert_eq!(registry_req.url.path(), builder_req.url.path());
        assert_eq!(registry_req.url.query(), builder_req.url.query());
        assert_eq!(
            wiremock_header_value(&builder_req, "authorization"),
            Some("Bearer ctx-key".to_string())
        );
        assert_eq!(
            wiremock_header_value(&builder_req, "authorization"),
            wiremock_header_value(&config_req, "authorization")
        );
        assert_eq!(
            wiremock_header_value(&builder_req, "authorization"),
            wiremock_header_value(&registry_req, "authorization")
        );
    }

    #[tokio::test]
    async fn minimaxi_builder_config_registry_upload_file_are_equivalent() {
        let _lock = lock_env();

        async fn mount_upload_server() -> MockServer {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/v1/files/upload"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "file": minimaxi_file_object_json(),
                    "base_resp": {
                        "status_code": 0,
                        "status_msg": "success"
                    }
                })))
                .mount(&server)
                .await;
            server
        }

        let builder_server = mount_upload_server().await;
        let config_server = mount_upload_server().await;
        let registry_server = mount_upload_server().await;

        let builder_client = siumai_provider_minimaxi::providers::minimaxi::MinimaxiBuilder::new(
            siumai_provider_minimaxi::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url(format!("{}/anthropic/v1", builder_server.uri()))
        .model("MiniMax-M2")
        .build()
        .await
        .expect("build builder client");

        let config_client =
            siumai_provider_minimaxi::providers::minimaxi::client::MinimaxiClient::from_config(
                siumai_provider_minimaxi::providers::minimaxi::config::MinimaxiConfig::new(
                    "ctx-key",
                )
                .with_base_url(format!("{}/anthropic/v1", config_server.uri()))
                .with_model("MiniMax-M2"),
            )
            .expect("build config client");

        let factory = crate::registry::factories::MiniMaxiProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "MiniMax-M2",
                &BuildContext {
                    provider_id: Some("minimaxi".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some(format!("{}/anthropic/v1", registry_server.uri())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = crate::types::FileUploadRequest {
            content: b"hello".to_vec(),
            filename: "hello.txt".to_string(),
            mime_type: Some("text/plain".to_string()),
            purpose: "t2a_async_input".to_string(),
            metadata: std::collections::HashMap::new(),
            http_config: None,
        };

        let builder_file = builder_client
            .upload_file(request.clone())
            .await
            .expect("builder upload ok");
        let config_file = config_client
            .upload_file(request.clone())
            .await
            .expect("config upload ok");
        let registry_file = registry_client
            .as_file_management_capability()
            .expect("registry file capability")
            .upload_file(request)
            .await
            .expect("registry upload ok");

        assert_eq!(builder_file.id, "123");
        assert_eq!(config_file.id, "123");
        assert_eq!(registry_file.id, "123");

        let builder_req = builder_server
            .received_requests()
            .await
            .expect("builder requests")
            .into_iter()
            .next()
            .expect("builder upload request");
        let config_req = config_server
            .received_requests()
            .await
            .expect("config requests")
            .into_iter()
            .next()
            .expect("config upload request");
        let registry_req = registry_server
            .received_requests()
            .await
            .expect("registry requests")
            .into_iter()
            .next()
            .expect("registry upload request");

        assert_eq!(builder_req.url.path(), "/v1/files/upload");
        assert_eq!(config_req.url.path(), builder_req.url.path());
        assert_eq!(registry_req.url.path(), builder_req.url.path());
        assert_eq!(
            wiremock_header_value(&builder_req, "authorization"),
            Some("Bearer ctx-key".to_string())
        );
        assert_eq!(
            wiremock_header_value(&builder_req, "authorization"),
            wiremock_header_value(&config_req, "authorization")
        );
        assert_eq!(
            wiremock_header_value(&builder_req, "authorization"),
            wiremock_header_value(&registry_req, "authorization")
        );
        assert_eq!(
            normalize_wiremock_multipart_body(&builder_req),
            normalize_wiremock_multipart_body(&config_req)
        );
        assert_eq!(
            normalize_wiremock_multipart_body(&builder_req),
            normalize_wiremock_multipart_body(&registry_req)
        );
    }

    #[tokio::test]
    async fn minimaxi_builder_config_registry_list_files_are_equivalent() {
        let _lock = lock_env();

        async fn mount_list_server() -> MockServer {
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/v1/files/list"))
                .and(query_param("purpose", "t2a_async_input"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "files": [minimaxi_file_object_json()],
                    "base_resp": {
                        "status_code": 0,
                        "status_msg": "success"
                    }
                })))
                .mount(&server)
                .await;
            server
        }

        let builder_server = mount_list_server().await;
        let config_server = mount_list_server().await;
        let registry_server = mount_list_server().await;

        let builder_client = siumai_provider_minimaxi::providers::minimaxi::MinimaxiBuilder::new(
            siumai_provider_minimaxi::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url(format!("{}/anthropic/v1", builder_server.uri()))
        .model("MiniMax-M2")
        .build()
        .await
        .expect("build builder client");

        let config_client =
            siumai_provider_minimaxi::providers::minimaxi::client::MinimaxiClient::from_config(
                siumai_provider_minimaxi::providers::minimaxi::config::MinimaxiConfig::new(
                    "ctx-key",
                )
                .with_base_url(format!("{}/anthropic/v1", config_server.uri()))
                .with_model("MiniMax-M2"),
            )
            .expect("build config client");

        let factory = crate::registry::factories::MiniMaxiProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "MiniMax-M2",
                &BuildContext {
                    provider_id: Some("minimaxi".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some(format!("{}/anthropic/v1", registry_server.uri())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let query = crate::types::FileListQuery {
            purpose: Some("t2a_async_input".to_string()),
            limit: None,
            after: None,
            order: None,
            http_config: None,
        };

        let builder_list = builder_client
            .list_files(Some(query.clone()))
            .await
            .expect("builder list ok");
        let config_list = config_client
            .list_files(Some(query.clone()))
            .await
            .expect("config list ok");
        let registry_list = registry_client
            .as_file_management_capability()
            .expect("registry file capability")
            .list_files(Some(query))
            .await
            .expect("registry list ok");

        assert_eq!(builder_list.files.len(), 1);
        assert_eq!(config_list.files.len(), 1);
        assert_eq!(registry_list.files.len(), 1);

        let builder_req = builder_server
            .received_requests()
            .await
            .expect("builder requests")
            .into_iter()
            .next()
            .expect("builder list request");
        let config_req = config_server
            .received_requests()
            .await
            .expect("config requests")
            .into_iter()
            .next()
            .expect("config list request");
        let registry_req = registry_server
            .received_requests()
            .await
            .expect("registry requests")
            .into_iter()
            .next()
            .expect("registry list request");

        assert_eq!(builder_req.url.path(), "/v1/files/list");
        assert_eq!(builder_req.url.query(), Some("purpose=t2a_async_input"));
        assert_eq!(config_req.url.path(), builder_req.url.path());
        assert_eq!(config_req.url.query(), builder_req.url.query());
        assert_eq!(registry_req.url.path(), builder_req.url.path());
        assert_eq!(registry_req.url.query(), builder_req.url.query());
        assert_eq!(
            wiremock_header_value(&builder_req, "authorization"),
            Some("Bearer ctx-key".to_string())
        );
        assert_eq!(
            wiremock_header_value(&builder_req, "authorization"),
            wiremock_header_value(&registry_req, "authorization")
        );
    }

    #[tokio::test]
    async fn minimaxi_builder_config_registry_retrieve_file_are_equivalent() {
        let _lock = lock_env();

        async fn mount_retrieve_server() -> MockServer {
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/v1/files/retrieve"))
                .and(query_param("file_id", "123"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "file": {
                        "file_id": 123,
                        "filename": "hello.txt",
                        "bytes": 5,
                        "created_at": 1_700_000_000i64,
                        "purpose": "t2a_async_input",
                        "download_url": "https://example.com/download/123"
                    },
                    "base_resp": {
                        "status_code": 0,
                        "status_msg": "success"
                    }
                })))
                .mount(&server)
                .await;
            server
        }

        let builder_server = mount_retrieve_server().await;
        let config_server = mount_retrieve_server().await;
        let registry_server = mount_retrieve_server().await;

        let builder_client = siumai_provider_minimaxi::providers::minimaxi::MinimaxiBuilder::new(
            siumai_provider_minimaxi::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url(format!("{}/anthropic/v1", builder_server.uri()))
        .model("MiniMax-M2")
        .build()
        .await
        .expect("build builder client");

        let config_client =
            siumai_provider_minimaxi::providers::minimaxi::client::MinimaxiClient::from_config(
                siumai_provider_minimaxi::providers::minimaxi::config::MinimaxiConfig::new(
                    "ctx-key",
                )
                .with_base_url(format!("{}/anthropic/v1", config_server.uri()))
                .with_model("MiniMax-M2"),
            )
            .expect("build config client");

        let factory = crate::registry::factories::MiniMaxiProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "MiniMax-M2",
                &BuildContext {
                    provider_id: Some("minimaxi".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some(format!("{}/anthropic/v1", registry_server.uri())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let builder_file = builder_client
            .retrieve_file("123".to_string())
            .await
            .expect("builder retrieve ok");
        let config_file = config_client
            .retrieve_file("123".to_string())
            .await
            .expect("config retrieve ok");
        let registry_file = registry_client
            .as_file_management_capability()
            .expect("registry file capability")
            .retrieve_file("123".to_string())
            .await
            .expect("registry retrieve ok");

        assert_eq!(builder_file.id, "123");
        assert_eq!(config_file.id, "123");
        assert_eq!(registry_file.id, "123");

        let builder_req = builder_server
            .received_requests()
            .await
            .expect("builder requests")
            .into_iter()
            .next()
            .expect("builder retrieve request");
        let config_req = config_server
            .received_requests()
            .await
            .expect("config requests")
            .into_iter()
            .next()
            .expect("config retrieve request");
        let registry_req = registry_server
            .received_requests()
            .await
            .expect("registry requests")
            .into_iter()
            .next()
            .expect("registry retrieve request");

        assert_eq!(builder_req.url.path(), "/v1/files/retrieve");
        assert_eq!(builder_req.url.query(), Some("file_id=123"));
        assert_eq!(config_req.url.path(), builder_req.url.path());
        assert_eq!(config_req.url.query(), builder_req.url.query());
        assert_eq!(registry_req.url.path(), builder_req.url.path());
        assert_eq!(registry_req.url.query(), builder_req.url.query());
        assert_eq!(
            wiremock_header_value(&builder_req, "authorization"),
            Some("Bearer ctx-key".to_string())
        );
        assert_eq!(
            wiremock_header_value(&builder_req, "authorization"),
            wiremock_header_value(&registry_req, "authorization")
        );
    }

    #[tokio::test]
    async fn minimaxi_builder_config_registry_get_file_content_are_equivalent() {
        let _lock = lock_env();

        async fn mount_content_server() -> MockServer {
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/v1/files/retrieve_content"))
                .and(query_param("file_id", "123"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_bytes(b"hello".to_vec())
                        .insert_header("content-type", "application/octet-stream"),
                )
                .mount(&server)
                .await;
            server
        }

        let builder_server = mount_content_server().await;
        let config_server = mount_content_server().await;
        let registry_server = mount_content_server().await;

        let builder_client = siumai_provider_minimaxi::providers::minimaxi::MinimaxiBuilder::new(
            siumai_provider_minimaxi::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url(format!("{}/anthropic/v1", builder_server.uri()))
        .model("MiniMax-M2")
        .build()
        .await
        .expect("build builder client");

        let config_client =
            siumai_provider_minimaxi::providers::minimaxi::client::MinimaxiClient::from_config(
                siumai_provider_minimaxi::providers::minimaxi::config::MinimaxiConfig::new(
                    "ctx-key",
                )
                .with_base_url(format!("{}/anthropic/v1", config_server.uri()))
                .with_model("MiniMax-M2"),
            )
            .expect("build config client");

        let factory = crate::registry::factories::MiniMaxiProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "MiniMax-M2",
                &BuildContext {
                    provider_id: Some("minimaxi".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some(format!("{}/anthropic/v1", registry_server.uri())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let builder_bytes = builder_client
            .get_file_content("123".to_string())
            .await
            .expect("builder content ok");
        let config_bytes = config_client
            .get_file_content("123".to_string())
            .await
            .expect("config content ok");
        let registry_bytes = registry_client
            .as_file_management_capability()
            .expect("registry file capability")
            .get_file_content("123".to_string())
            .await
            .expect("registry content ok");

        assert_eq!(builder_bytes, b"hello");
        assert_eq!(config_bytes, b"hello");
        assert_eq!(registry_bytes, b"hello");

        let builder_req = builder_server
            .received_requests()
            .await
            .expect("builder requests")
            .into_iter()
            .next()
            .expect("builder content request");
        let config_req = config_server
            .received_requests()
            .await
            .expect("config requests")
            .into_iter()
            .next()
            .expect("config content request");
        let registry_req = registry_server
            .received_requests()
            .await
            .expect("registry requests")
            .into_iter()
            .next()
            .expect("registry content request");

        assert_eq!(builder_req.url.path(), "/v1/files/retrieve_content");
        assert_eq!(builder_req.url.query(), Some("file_id=123"));
        assert_eq!(config_req.url.path(), builder_req.url.path());
        assert_eq!(config_req.url.query(), builder_req.url.query());
        assert_eq!(registry_req.url.path(), builder_req.url.path());
        assert_eq!(registry_req.url.query(), builder_req.url.query());
        assert_eq!(
            wiremock_header_value(&builder_req, "authorization"),
            Some("Bearer ctx-key".to_string())
        );
        assert_eq!(
            wiremock_header_value(&builder_req, "authorization"),
            wiremock_header_value(&registry_req, "authorization")
        );
    }

    #[tokio::test]
    async fn minimaxi_builder_config_registry_delete_file_are_equivalent() {
        let _lock = lock_env();

        async fn mount_delete_server() -> MockServer {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/v1/files/delete"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "base_resp": {
                        "status_code": 0,
                        "status_msg": "success"
                    }
                })))
                .mount(&server)
                .await;
            server
        }

        let builder_server = mount_delete_server().await;
        let config_server = mount_delete_server().await;
        let registry_server = mount_delete_server().await;

        let builder_client = siumai_provider_minimaxi::providers::minimaxi::MinimaxiBuilder::new(
            siumai_provider_minimaxi::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url(format!("{}/anthropic/v1", builder_server.uri()))
        .model("MiniMax-M2")
        .build()
        .await
        .expect("build builder client");

        let config_client =
            siumai_provider_minimaxi::providers::minimaxi::client::MinimaxiClient::from_config(
                siumai_provider_minimaxi::providers::minimaxi::config::MinimaxiConfig::new(
                    "ctx-key",
                )
                .with_base_url(format!("{}/anthropic/v1", config_server.uri()))
                .with_model("MiniMax-M2"),
            )
            .expect("build config client");

        let factory = crate::registry::factories::MiniMaxiProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "MiniMax-M2",
                &BuildContext {
                    provider_id: Some("minimaxi".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some(format!("{}/anthropic/v1", registry_server.uri())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let builder_deleted = builder_client
            .delete_file("123:t2a_async_input".to_string())
            .await
            .expect("builder delete ok");
        let config_deleted = config_client
            .delete_file("123:t2a_async_input".to_string())
            .await
            .expect("config delete ok");
        let registry_deleted = registry_client
            .as_file_management_capability()
            .expect("registry file capability")
            .delete_file("123:t2a_async_input".to_string())
            .await
            .expect("registry delete ok");

        assert!(builder_deleted.deleted);
        assert!(config_deleted.deleted);
        assert!(registry_deleted.deleted);

        let builder_req = builder_server
            .received_requests()
            .await
            .expect("builder requests")
            .into_iter()
            .next()
            .expect("builder delete request");
        let config_req = config_server
            .received_requests()
            .await
            .expect("config requests")
            .into_iter()
            .next()
            .expect("config delete request");
        let registry_req = registry_server
            .received_requests()
            .await
            .expect("registry requests")
            .into_iter()
            .next()
            .expect("registry delete request");

        let builder_body: serde_json::Value =
            serde_json::from_slice(&builder_req.body).expect("builder delete body");
        let config_body: serde_json::Value =
            serde_json::from_slice(&config_req.body).expect("config delete body");
        let registry_body: serde_json::Value =
            serde_json::from_slice(&registry_req.body).expect("registry delete body");

        assert_eq!(builder_req.url.path(), "/v1/files/delete");
        assert_eq!(config_req.url.path(), builder_req.url.path());
        assert_eq!(registry_req.url.path(), builder_req.url.path());
        assert_eq!(builder_body, config_body);
        assert_eq!(builder_body, registry_body);
        assert_eq!(
            builder_body,
            serde_json::json!({
                "file_id": 123,
                "purpose": "t2a_async_input"
            })
        );
    }
}

#[cfg(feature = "google-vertex")]
mod anthropic_vertex_contract {
    use super::*;
    use crate::traits::ChatCapability;

    #[tokio::test]
    async fn anthropic_vertex_factory_does_not_advertise_non_text_capabilities() {
        let _lock = lock_env();

        let factory = crate::registry::factories::AnthropicVertexProviderFactory;
        let caps = factory.capabilities();

        assert!(caps.supports("chat"));
        assert!(caps.supports("streaming"));
        assert!(caps.supports("tools"));
        assert!(!caps.supports("vision"));
        assert_embedding_image_rerank_capabilities_absent(&caps);
        assert!(!caps.supports("speech"));
        assert!(!caps.supports("transcription"));
        assert!(!caps.supports("audio"));
    }

    #[tokio::test]
    async fn anthropic_vertex_factory_prefers_ctx_http_client_over_http_config() {
        let _lock = lock_env();

        let factory = crate::registry::factories::AnthropicVertexProviderFactory;

        let mut bad = HttpConfig::default();
        bad.proxy = Some("not-a-url".to_string());

        let ctx = BuildContext {
            provider_id: Some("anthropic-vertex".to_string()),
            base_url: Some("https://example.com/v1".to_string()),
            http_client: Some(reqwest::Client::new()),
            http_config: Some(bad),
            ..Default::default()
        };

        factory
            .language_model_with_ctx("claude-3-5-sonnet-20241022", &ctx)
            .await
            .expect("factory should prefer ctx.http_client over invalid http_config");
    }

    #[tokio::test]
    async fn anthropic_vertex_factory_requires_base_url() {
        let _lock = lock_env();

        let factory = crate::registry::factories::AnthropicVertexProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("anthropic-vertex".to_string()),
            ..Default::default()
        };

        let result = factory
            .language_model_with_ctx("claude-3-5-sonnet-20241022", &ctx)
            .await;
        match result {
            Err(LlmError::ConfigurationError(msg)) => {
                assert!(msg.to_lowercase().contains("base_url"));
            }
            Err(other) => panic!("unexpected error: {other:?}"),
            Ok(_) => panic!("expected base_url to be required"),
        }
    }

    #[tokio::test]
    async fn anthropic_vertex_factory_uses_ctx_token_provider_for_authorization() {
        let _lock = lock_env();

        let factory = crate::registry::factories::AnthropicVertexProviderFactory;
        let transport = CaptureTransport::default();

        let client = factory
            .language_model_with_ctx(
                "claude-3-5-sonnet-20241022",
                &BuildContext {
                    provider_id: Some("anthropic-vertex".to_string()),
                    base_url: Some("https://example.com/custom".to_string()),
                    google_token_provider: Some(Arc::new(
                        siumai_core::auth::StaticTokenProvider::new("ctx-token"),
                    )),
                    http_transport: Some(Arc::new(transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build client");

        let _ = client
            .chat_request(make_chat_request_with_model("claude-3-5-sonnet-20241022"))
            .await;

        let req = transport.take().expect("captured request");
        assert_eq!(
            header_value(&req, "authorization"),
            Some("Bearer ctx-token".to_string())
        );
    }

    #[tokio::test]
    async fn anthropic_vertex_factory_rejects_deferred_non_text_family_paths() {
        let _lock = lock_env();

        let factory = crate::registry::factories::AnthropicVertexProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("anthropic-vertex".to_string()),
            base_url: Some("https://example.com/custom".to_string()),
            ..Default::default()
        };

        assert_unsupported_operation_contains(
            factory
                .embedding_model_with_ctx("claude-3-5-sonnet-20241022", &ctx)
                .await,
            "embedding family path",
        );
        assert_unsupported_operation_contains(
            factory
                .image_model_with_ctx("claude-3-5-sonnet-20241022", &ctx)
                .await,
            "image family path",
        );
        assert_unsupported_operation_contains(
            factory
                .speech_model_with_ctx("claude-3-5-sonnet-20241022", &ctx)
                .await,
            "speech family path",
        );
        assert_unsupported_operation_contains(
            factory
                .transcription_model_with_ctx("claude-3-5-sonnet-20241022", &ctx)
                .await,
            "transcription family path",
        );
        assert_unsupported_operation_contains(
            factory
                .reranking_model_with_ctx("claude-3-5-sonnet-20241022", &ctx)
                .await,
            "reranking family path",
        );
    }

    #[tokio::test]
    async fn anthropic_vertex_builder_config_registry_chat_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client =
            siumai_provider_google_vertex::providers::anthropic_vertex::VertexAnthropicBuilder::new(
                siumai_provider_google_vertex::builder::BuilderBase::default(),
            )
            .base_url("https://example.com/custom")
            .language_model("claude-3-5-sonnet-20241022")
            .bearer_token("ctx-key")
            .fetch(Arc::new(builder_transport.clone()))
            .build()
            .expect("build builder client");

        let config_client =
            siumai_provider_google_vertex::providers::anthropic_vertex::VertexAnthropicClient::from_config(
                siumai_provider_google_vertex::providers::anthropic_vertex::VertexAnthropicConfig::new(
                    "https://example.com/custom",
                    "claude-3-5-sonnet-20241022",
                )
                .with_bearer_token("ctx-key")
                .with_http_transport(Arc::new(config_transport.clone())),
            )
            .expect("build config client");

        let factory = crate::registry::factories::AnthropicVertexProviderFactory;
        let mut http_config = HttpConfig::default();
        http_config
            .headers
            .insert("Authorization".to_string(), "Bearer ctx-key".to_string());
        let registry_client = factory
            .language_model_with_ctx(
                "claude-3-5-sonnet-20241022",
                &BuildContext {
                    provider_id: Some("anthropic-vertex".to_string()),
                    base_url: Some("https://example.com/custom".to_string()),
                    http_config: Some(http_config),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model("claude-3-5-sonnet-20241022");

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(
            header_value(&builder_req, "authorization"),
            Some("Bearer ctx-key".to_string())
        );
        assert!(
            builder_req
                .url
                .contains("/models/claude-3-5-sonnet-20241022:rawPredict"),
            "unexpected url: {}",
            builder_req.url
        );
        assert_eq!(
            builder_req.body["anthropic_version"],
            serde_json::json!("vertex-2023-10-16")
        );
        assert!(builder_req.body.get("model").is_none());
    }

    #[tokio::test]
    async fn anthropic_vertex_builder_config_registry_chat_stream_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client =
            siumai_provider_google_vertex::providers::anthropic_vertex::VertexAnthropicBuilder::new(
                siumai_provider_google_vertex::builder::BuilderBase::default(),
            )
            .base_url("https://example.com/custom")
            .model("claude-3-5-sonnet-20241022")
            .bearer_token("ctx-key")
            .fetch(Arc::new(builder_transport.clone()))
            .build()
            .expect("build builder client");

        let config_client =
            siumai_provider_google_vertex::providers::anthropic_vertex::VertexAnthropicClient::from_config(
                siumai_provider_google_vertex::providers::anthropic_vertex::VertexAnthropicConfig::new(
                    "https://example.com/custom",
                    "claude-3-5-sonnet-20241022",
                )
                .with_bearer_token("ctx-key")
                .with_http_transport(Arc::new(config_transport.clone())),
            )
            .expect("build config client");

        let factory = crate::registry::factories::AnthropicVertexProviderFactory;
        let mut http_config = HttpConfig::default();
        http_config
            .headers
            .insert("Authorization".to_string(), "Bearer ctx-key".to_string());
        let registry_client = factory
            .language_model_with_ctx(
                "claude-3-5-sonnet-20241022",
                &BuildContext {
                    provider_id: Some("anthropic-vertex".to_string()),
                    base_url: Some("https://example.com/custom".to_string()),
                    http_config: Some(http_config),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model("claude-3-5-sonnet-20241022");

        let _ = builder_client.chat_stream_request(request.clone()).await;
        let _ = config_client.chat_stream_request(request.clone()).await;
        let _ = registry_client.chat_stream_request(request).await;

        let builder_req = builder_transport
            .take_stream()
            .expect("builder stream request");
        let config_req = config_transport
            .take_stream()
            .expect("config stream request");
        let registry_req = registry_transport
            .take_stream()
            .expect("registry stream request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert!(
            builder_req
                .url
                .contains("/models/claude-3-5-sonnet-20241022:streamRawPredict"),
            "unexpected url: {}",
            builder_req.url
        );
        assert_eq!(
            header_value(&builder_req, "accept"),
            Some("text/event-stream".to_string())
        );
        assert_eq!(
            builder_req.body["anthropic_version"],
            serde_json::json!("vertex-2023-10-16")
        );
    }

    #[tokio::test]
    async fn anthropic_vertex_builder_config_registry_chat_request_respect_explicit_request_model()
    {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();
        let default_model = "claude-3-5-sonnet-20241022";
        let request_model = "claude-3-7-sonnet-20250219";

        let builder_client =
            siumai_provider_google_vertex::providers::anthropic_vertex::VertexAnthropicBuilder::new(
                siumai_provider_google_vertex::builder::BuilderBase::default(),
            )
            .base_url("https://example.com/custom")
            .language_model(default_model)
            .bearer_token("ctx-key")
            .fetch(Arc::new(builder_transport.clone()))
            .build()
            .expect("build builder client");

        let config_client =
            siumai_provider_google_vertex::providers::anthropic_vertex::VertexAnthropicClient::from_config(
                siumai_provider_google_vertex::providers::anthropic_vertex::VertexAnthropicConfig::new(
                    "https://example.com/custom",
                    default_model,
                )
                .with_bearer_token("ctx-key")
                .with_http_transport(Arc::new(config_transport.clone())),
            )
            .expect("build config client");

        let factory = crate::registry::factories::AnthropicVertexProviderFactory;
        let mut http_config = HttpConfig::default();
        http_config
            .headers
            .insert("Authorization".to_string(), "Bearer ctx-key".to_string());
        let registry_client = factory
            .language_model_with_ctx(
                default_model,
                &BuildContext {
                    provider_id: Some("anthropic-vertex".to_string()),
                    base_url: Some("https://example.com/custom".to_string()),
                    http_config: Some(http_config),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model(request_model);

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert!(
            builder_req
                .url
                .contains("/models/claude-3-7-sonnet-20250219:rawPredict"),
            "unexpected url: {}",
            builder_req.url
        );
    }

    #[tokio::test]
    async fn anthropic_vertex_builder_config_registry_chat_stream_request_respect_explicit_request_model()
     {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();
        let default_model = "claude-3-5-sonnet-20241022";
        let request_model = "claude-3-7-sonnet-20250219";

        let builder_client =
            siumai_provider_google_vertex::providers::anthropic_vertex::VertexAnthropicBuilder::new(
                siumai_provider_google_vertex::builder::BuilderBase::default(),
            )
            .base_url("https://example.com/custom")
            .model(default_model)
            .bearer_token("ctx-key")
            .fetch(Arc::new(builder_transport.clone()))
            .build()
            .expect("build builder client");

        let config_client =
            siumai_provider_google_vertex::providers::anthropic_vertex::VertexAnthropicClient::from_config(
                siumai_provider_google_vertex::providers::anthropic_vertex::VertexAnthropicConfig::new(
                    "https://example.com/custom",
                    default_model,
                )
                .with_bearer_token("ctx-key")
                .with_http_transport(Arc::new(config_transport.clone())),
            )
            .expect("build config client");

        let factory = crate::registry::factories::AnthropicVertexProviderFactory;
        let mut http_config = HttpConfig::default();
        http_config
            .headers
            .insert("Authorization".to_string(), "Bearer ctx-key".to_string());
        let registry_client = factory
            .language_model_with_ctx(
                default_model,
                &BuildContext {
                    provider_id: Some("anthropic-vertex".to_string()),
                    base_url: Some("https://example.com/custom".to_string()),
                    http_config: Some(http_config),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model(request_model);

        let _ = builder_client.chat_stream_request(request.clone()).await;
        let _ = config_client.chat_stream_request(request.clone()).await;
        let _ = registry_client.chat_stream_request(request).await;

        let builder_req = builder_transport
            .take_stream()
            .expect("builder stream request");
        let config_req = config_transport
            .take_stream()
            .expect("config stream request");
        let registry_req = registry_transport
            .take_stream()
            .expect("registry stream request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert!(
            builder_req
                .url
                .contains("/models/claude-3-7-sonnet-20250219:streamRawPredict"),
            "unexpected url: {}",
            builder_req.url
        );
    }

    #[tokio::test]
    async fn anthropic_vertex_registry_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let vertex_transport = CaptureTransport::default();

        let mut global_http_config = HttpConfig::default();
        global_http_config.headers.insert(
            "Authorization".to_string(),
            "Bearer global-token".to_string(),
        );
        global_http_config
            .headers
            .insert("X-Global-Header".to_string(), "keep-me".to_string());

        let mut provider_http_config = HttpConfig::empty();
        provider_http_config
            .headers
            .insert("Authorization".to_string(), "Bearer ctx-key".to_string());

        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "anthropic-vertex".to_string(),
            Arc::new(crate::registry::factories::AnthropicVertexProviderFactory)
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_base_url("https://example.com/global")
            .with_http_config(global_http_config)
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "anthropic-vertex",
                crate::registry::ProviderBuildOverrides::default()
                    .with_base_url("https://example.com/custom")
                    .with_http_config(provider_http_config)
                    .fetch(Arc::new(vertex_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .language_model("anthropic-vertex:claude-3-5-sonnet-20241022")
            .expect("build anthropic-vertex handle");

        let _ = handle
            .chat_request(make_chat_request_with_model("claude-3-5-sonnet-20241022"))
            .await;

        let req = vertex_transport.take().expect("captured request");
        assert!(global_transport.take().is_none());
        assert_eq!(
            header_value(&req, "authorization"),
            Some("Bearer ctx-key".to_string())
        );
        assert_eq!(
            header_value(&req, "x-global-header"),
            Some("keep-me".to_string())
        );
        assert!(
            req.url.starts_with(
                "https://example.com/custom/models/claude-3-5-sonnet-20241022:rawPredict"
            ),
            "unexpected url: {}",
            req.url
        );
        assert_eq!(
            req.body["anthropic_version"],
            serde_json::json!("vertex-2023-10-16")
        );
    }

    #[tokio::test]
    async fn anthropic_vertex_registry_rejects_unsupported_non_text_handle_construction() {
        let _lock = lock_env();

        let transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "anthropic-vertex".to_string(),
            Arc::new(crate::registry::factories::AnthropicVertexProviderFactory)
                as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_base_url("https://example.com/custom")
            .fetch(Arc::new(transport.clone()))
            .auto_middleware(false)
            .build()
            .expect("build registry");

        assert_unsupported_operation_contains(
            registry.embedding_model("anthropic-vertex:claude-3-5-sonnet-20241022"),
            "embedding_model handle",
        );
        assert_unsupported_operation_contains(
            registry.image_model("anthropic-vertex:claude-3-5-sonnet-20241022"),
            "image_model handle",
        );
        assert_unsupported_operation_contains(
            registry.reranking_model("anthropic-vertex:claude-3-5-sonnet-20241022"),
            "reranking_model handle",
        );

        assert_capture_transport_unused(&transport);
    }
}

#[cfg(feature = "google")]
mod gemini_contract {
    use super::*;
    use crate::registry::factories::GeminiProviderFactory;
    use crate::traits::ChatCapability;
    use crate::traits::EmbeddingExtensions;
    use siumai_provider_gemini::provider_options::gemini::{GeminiOptions, GeminiThinkingConfig};
    use siumai_provider_gemini::providers::gemini::ext::GeminiChatRequestExt;

    #[tokio::test]
    async fn gemini_factory_prefers_ctx_http_client_over_http_config() {
        let _lock = lock_env();

        let factory = GeminiProviderFactory;

        let mut bad = HttpConfig::default();
        bad.proxy = Some("not-a-url".to_string());

        let ctx = BuildContext {
            provider_id: Some("gemini".to_string()),
            api_key: Some("ctx-key".to_string()),
            http_client: Some(reqwest::Client::new()),
            http_config: Some(bad),
            ..Default::default()
        };

        factory
            .language_model_with_ctx("gemini-2.5-flash", &ctx)
            .await
            .expect("factory should prefer ctx.http_client over invalid http_config");
    }

    #[tokio::test]
    async fn gemini_factory_uses_env_api_key_when_ctx_missing() {
        let _lock = lock_env();

        let _g = EnvGuard::set("GEMINI_API_KEY", "env-key");
        let factory = GeminiProviderFactory;

        let ctx = BuildContext {
            provider_id: Some("gemini".to_string()),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("gemini-2.5-flash", &ctx)
            .await
            .expect("build client via env api key");

        let typed = client
            .as_any()
            .downcast_ref::<siumai_provider_gemini::providers::gemini::GeminiClient>()
            .expect("GeminiClient");
        assert_eq!(typed.api_key(), "env-key");
    }

    #[tokio::test]
    async fn gemini_factory_prefers_ctx_api_key_over_env() {
        let _lock = lock_env();

        let _g = EnvGuard::set("GEMINI_API_KEY", "env-key");
        let factory = GeminiProviderFactory;

        let ctx = BuildContext {
            provider_id: Some("gemini".to_string()),
            api_key: Some("ctx-key".to_string()),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("gemini-2.5-flash", &ctx)
            .await
            .expect("build client via ctx api key");

        let typed = client
            .as_any()
            .downcast_ref::<siumai_provider_gemini::providers::gemini::GeminiClient>()
            .expect("GeminiClient");
        assert_eq!(typed.api_key(), "ctx-key");
    }

    #[tokio::test]
    async fn gemini_factory_accepts_root_base_url() {
        let _lock = lock_env();

        let _g = EnvGuard::set("GEMINI_API_KEY", "env-key");
        let factory = GeminiProviderFactory;

        let ctx = BuildContext {
            provider_id: Some("gemini".to_string()),
            base_url: Some("https://generativelanguage.googleapis.com".to_string()),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("gemini-2.5-flash", &ctx)
            .await
            .expect("build client");

        let typed = client
            .as_any()
            .downcast_ref::<siumai_provider_gemini::providers::gemini::GeminiClient>()
            .expect("GeminiClient");
        assert_eq!(
            typed.base_url(),
            "https://generativelanguage.googleapis.com/v1beta"
        );
    }

    #[tokio::test]
    async fn gemini_registry_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let gemini_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "gemini".to_string(),
            Arc::new(GeminiProviderFactory) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "gemini",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/v1beta")
                    .fetch(Arc::new(gemini_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .language_model("gemini:gemini-2.5-flash")
            .expect("build gemini handle");

        let _ = handle
            .chat_request(make_chat_request_with_model("gemini-2.5-flash"))
            .await;

        let req = gemini_transport.take().expect("captured gemini request");
        assert!(global_transport.take().is_none());
        assert_eq!(req.headers.get("x-goog-api-key").unwrap(), "ctx-key");
        assert_eq!(
            req.url,
            "https://example.com/v1beta/models/gemini-2.5-flash:generateContent"
        );
        assert_eq!(
            req.body["contents"][0]["parts"][0]["text"],
            serde_json::json!("hi")
        );
    }

    #[tokio::test]
    async fn gemini_builder_config_registry_chat_stream_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_gemini::providers::gemini::GeminiBuilder::new(
            siumai_provider_gemini::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/v1beta")
        .model("gemini-2.5-flash")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client = siumai_provider_gemini::providers::gemini::GeminiClient::from_config(
            siumai_provider_gemini::providers::gemini::GeminiConfig::new("ctx-key")
                .with_base_url("https://example.com/v1beta".to_string())
                .with_model("gemini-2.5-flash".to_string())
                .with_http_transport(Arc::new(config_transport.clone())),
        )
        .expect("build config client");

        let factory = GeminiProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "gemini-2.5-flash",
                &BuildContext {
                    provider_id: Some("gemini".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/v1beta".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model("gemini-2.5-flash");

        let _ = builder_client.chat_stream_request(request.clone()).await;
        let _ = config_client.chat_stream_request(request.clone()).await;
        let _ = registry_client.chat_stream_request(request).await;

        let builder_req = builder_transport
            .take_stream()
            .expect("builder stream request");
        let config_req = config_transport
            .take_stream()
            .expect("config stream request");
        let registry_req = registry_transport
            .take_stream()
            .expect("registry stream request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(
            builder_req.url,
            "https://example.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
        );
        assert_eq!(
            header_value(&builder_req, "x-goog-api-key"),
            Some("ctx-key".to_string())
        );
        assert_eq!(
            header_value(&builder_req, "accept"),
            Some("text/event-stream".to_string())
        );
        assert_eq!(
            builder_req.body["contents"][0]["parts"][0]["text"],
            serde_json::json!("hi")
        );
    }

    #[tokio::test]
    async fn gemini_registry_stream_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let gemini_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "gemini".to_string(),
            Arc::new(GeminiProviderFactory) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "gemini",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/v1beta")
                    .fetch(Arc::new(gemini_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .language_model("gemini:gemini-2.5-flash")
            .expect("build gemini handle");

        let _ = handle
            .chat_stream_request(make_chat_request_with_model("gemini-2.5-flash"))
            .await;

        let req = gemini_transport
            .take_stream()
            .expect("captured gemini stream request");
        assert!(global_transport.take().is_none());
        assert!(global_transport.take_stream().is_none());
        assert_eq!(req.headers.get("x-goog-api-key").unwrap(), "ctx-key");
        assert_eq!(
            req.url,
            "https://example.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
        );
        assert_eq!(
            header_value(&req, "accept"),
            Some("text/event-stream".to_string())
        );
        assert_eq!(
            req.body["contents"][0]["parts"][0]["text"],
            serde_json::json!("hi")
        );
    }

    #[tokio::test]
    async fn gemini_builder_config_registry_stable_request_options_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let builder_client = siumai_provider_gemini::providers::gemini::GeminiBuilder::new(
            siumai_provider_gemini::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/v1beta")
        .model("gemini-2.5-flash")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client = siumai_provider_gemini::providers::gemini::GeminiClient::from_config(
            siumai_provider_gemini::providers::gemini::GeminiConfig::new("ctx-key")
                .with_base_url("https://example.com/v1beta".to_string())
                .with_model("gemini-2.5-flash".to_string())
                .with_http_transport(Arc::new(config_transport.clone())),
        )
        .expect("build config client");

        let factory = GeminiProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "gemini-2.5-flash",
                &BuildContext {
                    provider_id: Some("gemini".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/v1beta".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model("gemini-2.5-flash")
            .with_tools(vec![crate::types::Tool::function(
                "get_weather",
                "Get weather",
                serde_json::json!({
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"]
                }),
            )])
            .with_tool_choice(crate::types::ToolChoice::None)
            .with_response_format(crate::types::ResponseFormat::json_schema(schema.clone()))
            .with_gemini_options(
                GeminiOptions::new()
                    .with_thinking_config(
                        GeminiThinkingConfig::new()
                            .with_thinking_budget(2048)
                            .with_include_thoughts(true),
                    )
                    .with_structured_outputs(true),
            );

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(
            builder_req.url,
            "https://example.com/v1beta/models/gemini-2.5-flash:generateContent"
        );
        assert_eq!(
            builder_req.body["generationConfig"]["thinkingConfig"],
            serde_json::json!({
                "thinkingBudget": 2048,
                "includeThoughts": true
            })
        );
        assert_eq!(
            builder_req.body["generationConfig"]["responseMimeType"],
            serde_json::json!("application/json")
        );
        assert!(
            builder_req.body["generationConfig"]
                .get("responseSchema")
                .is_some()
        );
        assert!(
            builder_req.body["generationConfig"]
                .get("responseJsonSchema")
                .is_none()
        );
        assert_eq!(
            builder_req.body["toolConfig"],
            serde_json::json!({
                "functionCallingConfig": { "mode": "NONE" }
            })
        );
    }

    #[tokio::test]
    async fn gemini_factory_does_not_require_api_key_with_authorization_header() {
        let _lock = lock_env();

        let _g = EnvGuard::remove("GEMINI_API_KEY");
        let factory = GeminiProviderFactory;

        let mut http_config = HttpConfig::default();
        http_config
            .headers
            .insert("Authorization".to_string(), "Bearer token".to_string());

        let ctx = BuildContext {
            provider_id: Some("gemini".to_string()),
            http_config: Some(http_config),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("gemini-2.5-flash", &ctx)
            .await
            .expect("build client without api key when auth header is present");

        let typed = client
            .as_any()
            .downcast_ref::<siumai_provider_gemini::providers::gemini::GeminiClient>()
            .expect("GeminiClient");
        assert_eq!(typed.api_key(), "");
    }

    #[tokio::test]
    async fn gemini_factory_supports_native_text_family_path() {
        let _lock = lock_env();

        let factory = GeminiProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("gemini".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://generativelanguage.googleapis.com/v1beta/".to_string()),
            ..Default::default()
        };

        let model = factory
            .language_model_text_with_ctx("gemini-2.5-flash", &ctx)
            .await
            .expect("build native text-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "gemini"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "gemini-2.5-flash"
        );
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model.as_ref()),
            crate::traits::ModelSpecVersion::V1
        );
    }

    #[tokio::test]
    async fn gemini_factory_supports_native_image_family_path() {
        let _lock = lock_env();

        let factory = GeminiProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("gemini".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://generativelanguage.googleapis.com/v1beta/".to_string()),
            ..Default::default()
        };

        let model = factory
            .image_model_family_with_ctx("imagen-3.0-generate-002", &ctx)
            .await
            .expect("build native image-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "gemini"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "imagen-3.0-generate-002"
        );
    }

    #[tokio::test]
    async fn gemini_factory_supports_native_embedding_family_path() {
        let _lock = lock_env();

        let factory = GeminiProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("gemini".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://generativelanguage.googleapis.com/v1beta/".to_string()),
            ..Default::default()
        };

        let model = factory
            .embedding_model_family_with_ctx("text-embedding-004", &ctx)
            .await
            .expect("build native embedding-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "gemini"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "text-embedding-004"
        );
    }

    #[tokio::test]
    async fn gemini_factory_rejects_audio_family_paths_without_audio_family_support() {
        let _lock = lock_env();

        let factory = GeminiProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("gemini".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://generativelanguage.googleapis.com/v1beta/".to_string()),
            ..Default::default()
        };

        assert_unsupported_operation_contains(
            factory
                .speech_model_with_ctx("gemini-2.5-flash-preview-native-audio-dialog", &ctx)
                .await,
            "speech family path",
        );
        assert_unsupported_operation_contains(
            factory
                .transcription_model_with_ctx("gemini-2.5-flash-preview-native-audio-dialog", &ctx)
                .await,
            "transcription family path",
        );
    }

    #[tokio::test]
    async fn gemini_builder_config_registry_batch_embedding_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client = siumai_provider_gemini::providers::gemini::GeminiBuilder::new(
            siumai_provider_gemini::builder::BuilderBase::default(),
        )
        .api_key("ctx-key")
        .base_url("https://example.com/v1beta")
        .model("gemini-embedding-001")
        .fetch(Arc::new(builder_transport.clone()))
        .build()
        .await
        .expect("build builder client");

        let config_client = siumai_provider_gemini::providers::gemini::GeminiClient::from_config(
            siumai_provider_gemini::providers::gemini::GeminiConfig::new("ctx-key")
                .with_base_url("https://example.com/v1beta".to_string())
                .with_model("gemini-embedding-001".to_string())
                .with_http_transport(Arc::new(config_transport.clone())),
        )
        .expect("build config client");

        let factory = GeminiProviderFactory;
        let registry_client = factory
            .embedding_model_with_ctx(
                "gemini-embedding-001",
                &BuildContext {
                    provider_id: Some("gemini".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/v1beta".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = crate::types::EmbeddingRequest::new(vec!["A".to_string(), "B".to_string()])
            .with_model("gemini-embedding-001")
            .with_dimensions(64)
            .with_task_type(crate::types::EmbeddingTaskType::SemanticSimilarity);

        let _ = builder_client.embed_with_config(request.clone()).await;
        let _ = config_client.embed_with_config(request.clone()).await;
        let typed_registry = registry_client
            .as_any()
            .downcast_ref::<siumai_provider_gemini::providers::gemini::GeminiClient>()
            .expect("GeminiClient");
        let _ = typed_registry.embed_with_config(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert_eq!(
            builder_req.url,
            "https://example.com/v1beta/models/gemini-embedding-001:batchEmbedContents"
        );
        let requests = builder_req.body["requests"]
            .as_array()
            .expect("requests array");
        assert_eq!(requests.len(), 2);
        for (index, text) in ["A", "B"].iter().enumerate() {
            let item = &requests[index];
            assert_eq!(
                item["model"],
                serde_json::json!("models/gemini-embedding-001")
            );
            assert_eq!(item["taskType"], serde_json::json!("SEMANTIC_SIMILARITY"));
            assert_eq!(item["outputDimensionality"], serde_json::json!(64));
            assert_eq!(item["content"]["role"], serde_json::json!("user"));
            assert_eq!(item["content"]["parts"][0]["text"], serde_json::json!(text));
        }
    }
}

#[cfg(feature = "google-vertex")]
mod vertex_contract {
    use super::*;
    use crate::registry::factories::GoogleVertexProviderFactory;
    use crate::traits::ChatCapability;
    use crate::types::{EmbeddingRequest, ImageGenerationRequest};
    use siumai_provider_google_vertex::provider_options::vertex::{
        VertexEmbeddingOptions, VertexImagenOptions,
    };
    use siumai_provider_google_vertex::providers::vertex::{
        VertexEmbeddingRequestExt, VertexImagenRequestExt,
    };

    fn make_image_request() -> ImageGenerationRequest {
        ImageGenerationRequest {
            prompt: "hi".to_string(),
            count: 1,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn vertex_factory_prefers_ctx_http_client_over_http_config() {
        let _lock = lock_env();

        let factory = GoogleVertexProviderFactory;

        let mut bad = HttpConfig::default();
        bad.proxy = Some("not-a-url".to_string());

        let ctx = BuildContext {
            provider_id: Some("vertex".to_string()),
            api_key: Some("ctx-key".to_string()),
            http_client: Some(reqwest::Client::new()),
            http_config: Some(bad),
            ..Default::default()
        };

        factory
            .language_model_with_ctx("imagen-4.0-generate-001", &ctx)
            .await
            .expect("factory should prefer ctx.http_client over invalid http_config");
    }

    #[tokio::test]
    async fn vertex_factory_uses_express_base_url_when_ctx_api_key_present() {
        let _lock = lock_env();

        let factory = GoogleVertexProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("vertex".to_string()),
            api_key: Some("ctx-key".to_string()),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("imagen-4.0-generate-001", &ctx)
            .await
            .expect("build client");

        let typed = client
            .as_any()
            .downcast_ref::<siumai_provider_google_vertex::providers::vertex::GoogleVertexClient>()
            .expect("GoogleVertexClient");
        assert_eq!(
            typed.base_url(),
            crate::utils::vertex::GOOGLE_VERTEX_EXPRESS_BASE_URL
        );
    }

    #[tokio::test]
    async fn vertex_factory_uses_env_project_location_when_no_api_key_or_base_url() {
        let _lock = lock_env();

        let _k = EnvGuard::remove("GOOGLE_VERTEX_API_KEY");
        let _p = EnvGuard::set("GOOGLE_VERTEX_PROJECT", "test-project");
        let _l = EnvGuard::set("GOOGLE_VERTEX_LOCATION", "us-central1");

        let factory = GoogleVertexProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("vertex".to_string()),
            http_client: Some(reqwest::Client::new()),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("imagen-4.0-generate-001", &ctx)
            .await
            .expect("build client via env project/location");

        let typed = client
            .as_any()
            .downcast_ref::<siumai_provider_google_vertex::providers::vertex::GoogleVertexClient>()
            .expect("GoogleVertexClient");
        assert_eq!(
            typed.base_url(),
            crate::utils::vertex::google_vertex_base_url("test-project", "us-central1")
        );
    }

    #[tokio::test]
    async fn vertex_factory_prefers_ctx_base_url_over_express_default() {
        let _lock = lock_env();

        let factory = GoogleVertexProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("vertex".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom".to_string()),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("imagen-4.0-generate-001", &ctx)
            .await
            .expect("build client");

        let typed = client
            .as_any()
            .downcast_ref::<siumai_provider_google_vertex::providers::vertex::GoogleVertexClient>()
            .expect("GoogleVertexClient");
        assert_eq!(typed.base_url(), "https://example.com/custom");
    }

    #[tokio::test]
    async fn vertex_factory_prefers_ctx_api_key_over_env_for_express_query() {
        let _lock = lock_env();

        let _g = EnvGuard::set("GOOGLE_VERTEX_API_KEY", "env-key");
        let factory = GoogleVertexProviderFactory;
        let transport = CaptureTransport::default();

        let ctx = BuildContext {
            provider_id: Some("vertex".to_string()),
            api_key: Some("ctx-key".to_string()),
            http_transport: Some(Arc::new(transport.clone())),
            ..Default::default()
        };

        let client = factory
            .language_model_with_ctx("imagen-4.0-generate-001", &ctx)
            .await
            .expect("build client");

        let cap = client
            .as_image_generation_capability()
            .expect("image generation capability");
        let _ = cap.generate_images(make_image_request()).await;

        let req = transport.take().expect("captured request");
        assert!(
            req.url.contains("key=ctx-key"),
            "unexpected url: {}",
            req.url
        );
    }

    #[tokio::test]
    async fn vertex_factory_embedding_model_with_ctx_preserves_ctx_build_inputs() {
        let _lock = lock_env();

        let factory = GoogleVertexProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("vertex".to_string()),
            api_key: Some("ctx-key".to_string()),
            base_url: Some("https://example.com/custom".to_string()),
            ..Default::default()
        };

        let client = factory
            .embedding_model_with_ctx("text-embedding-004", &ctx)
            .await
            .expect("build embedding client");

        let typed = client
            .as_any()
            .downcast_ref::<siumai_provider_google_vertex::providers::vertex::GoogleVertexClient>()
            .expect("GoogleVertexClient");
        assert_eq!(typed.base_url(), "https://example.com/custom");
        assert_eq!(
            crate::traits::ModelMetadata::model_id(typed),
            "text-embedding-004"
        );
    }

    #[tokio::test]
    async fn vertex_factory_supports_native_text_family_path() {
        let _lock = lock_env();

        let factory = GoogleVertexProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("vertex".to_string()),
            api_key: Some("ctx-key".to_string()),
            ..Default::default()
        };

        let model = factory
            .language_model_text_with_ctx("gemini-2.5-flash", &ctx)
            .await
            .expect("build native text-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "vertex"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "gemini-2.5-flash"
        );
        assert_eq!(
            crate::traits::ModelMetadata::specification_version(model.as_ref()),
            crate::traits::ModelSpecVersion::V1
        );
    }

    #[tokio::test]
    async fn vertex_factory_supports_native_image_family_path() {
        let _lock = lock_env();

        let factory = GoogleVertexProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("vertex".to_string()),
            api_key: Some("ctx-key".to_string()),
            ..Default::default()
        };

        let model = factory
            .image_model_family_with_ctx("imagen-3.0-generate-002", &ctx)
            .await
            .expect("build native image-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "vertex"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "imagen-3.0-generate-002"
        );
    }

    #[tokio::test]
    async fn vertex_factory_supports_native_embedding_family_path() {
        let _lock = lock_env();

        let factory = GoogleVertexProviderFactory;
        let ctx = BuildContext {
            provider_id: Some("vertex".to_string()),
            api_key: Some("ctx-key".to_string()),
            ..Default::default()
        };

        let model = factory
            .embedding_model_family_with_ctx("text-embedding-004", &ctx)
            .await
            .expect("build native embedding-family model");

        assert_eq!(
            crate::traits::ModelMetadata::provider_id(model.as_ref()),
            "vertex"
        );
        assert_eq!(
            crate::traits::ModelMetadata::model_id(model.as_ref()),
            "text-embedding-004"
        );
    }

    #[tokio::test]
    async fn vertex_builder_config_registry_image_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client =
            siumai_provider_google_vertex::providers::vertex::GoogleVertexBuilder::new(
                siumai_provider_google_vertex::builder::BuilderBase::default(),
            )
            .api_key("ctx-key")
            .base_url("https://example.com/custom")
            .model("imagen-4.0-generate-001")
            .fetch(Arc::new(builder_transport.clone()))
            .build()
            .expect("build builder client");

        let config_client =
            siumai_provider_google_vertex::providers::vertex::GoogleVertexClient::from_config(
                siumai_provider_google_vertex::providers::vertex::GoogleVertexConfig::new(
                    "https://example.com/custom",
                    "imagen-4.0-generate-001",
                )
                .with_api_key("ctx-key")
                .with_http_transport(Arc::new(config_transport.clone())),
            )
            .expect("build config client");

        let factory = GoogleVertexProviderFactory;
        let registry_client = factory
            .image_model_with_ctx(
                "imagen-4.0-generate-001",
                &BuildContext {
                    provider_id: Some("vertex".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = ImageGenerationRequest {
            prompt: "a tiny orange robot".to_string(),
            negative_prompt: None,
            size: Some("1024x1024".to_string()),
            count: 1,
            model: Some("imagen-4.0-generate-001".to_string()),
            quality: None,
            style: None,
            seed: Some(7),
            steps: None,
            guidance_scale: None,
            enhance_prompt: Some(true),
            response_format: Some("b64_json".to_string()),
            extra_params: Default::default(),
            provider_options_map: Default::default(),
            http_config: None,
        }
        .with_vertex_imagen_options(VertexImagenOptions::new().with_negative_prompt("blurry"));

        let _ = crate::traits::ImageGenerationCapability::generate_images(
            &builder_client,
            request.clone(),
        )
        .await;
        let _ = crate::traits::ImageGenerationCapability::generate_images(
            &config_client,
            request.clone(),
        )
        .await;
        let _ = registry_client
            .as_image_generation_capability()
            .expect("registry image capability")
            .generate_images(request)
            .await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert!(
            builder_req
                .url
                .contains("/models/imagen-4.0-generate-001:predict?key=ctx-key"),
            "unexpected url: {}",
            builder_req.url
        );
        assert_eq!(
            builder_req.body["instances"][0]["prompt"],
            serde_json::json!("a tiny orange robot")
        );
        assert_eq!(
            builder_req.body["parameters"]["sampleCount"],
            serde_json::json!(1)
        );
        assert_eq!(builder_req.body["parameters"]["seed"], serde_json::json!(7));
        assert_eq!(
            builder_req.body["parameters"]["negativePrompt"],
            serde_json::json!("blurry")
        );
    }

    #[tokio::test]
    async fn vertex_builder_config_registry_embedding_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client =
            siumai_provider_google_vertex::providers::vertex::GoogleVertexBuilder::new(
                siumai_provider_google_vertex::builder::BuilderBase::default(),
            )
            .api_key("ctx-key")
            .base_url("https://example.com/custom")
            .embedding_model("text-embedding-004")
            .fetch(Arc::new(builder_transport.clone()))
            .build()
            .expect("build builder client");

        let config_client =
            siumai_provider_google_vertex::providers::vertex::GoogleVertexClient::from_config(
                siumai_provider_google_vertex::providers::vertex::GoogleVertexConfig::new(
                    "https://example.com/custom",
                    "text-embedding-004",
                )
                .with_api_key("ctx-key")
                .with_http_transport(Arc::new(config_transport.clone())),
            )
            .expect("build config client");

        let factory = GoogleVertexProviderFactory;
        let registry_client = factory
            .embedding_model_with_ctx(
                "text-embedding-004",
                &BuildContext {
                    provider_id: Some("vertex".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = EmbeddingRequest::new(vec![
            "hello vertex".to_string(),
            "embedding parity".to_string(),
        ])
        .with_model("text-embedding-004")
        .with_task_type(crate::types::EmbeddingTaskType::RetrievalDocument)
        .with_title("vertex-doc")
        .with_vertex_embedding_options(VertexEmbeddingOptions {
            output_dimensionality: Some(256),
            auto_truncate: Some(true),
            ..Default::default()
        });

        let _ =
            crate::traits::EmbeddingExtensions::embed_with_config(&builder_client, request.clone())
                .await;
        let _ =
            crate::traits::EmbeddingExtensions::embed_with_config(&config_client, request.clone())
                .await;

        let typed_registry = registry_client
            .as_any()
            .downcast_ref::<siumai_provider_google_vertex::providers::vertex::GoogleVertexClient>()
            .expect("GoogleVertexClient");
        let _ =
            crate::traits::EmbeddingExtensions::embed_with_config(typed_registry, request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert!(
            builder_req
                .url
                .contains("/models/text-embedding-004:predict?key=ctx-key"),
            "unexpected url: {}",
            builder_req.url
        );
        assert_eq!(
            builder_req.body["instances"][0]["content"],
            serde_json::json!("hello vertex")
        );
        assert_eq!(
            builder_req.body["instances"][0]["task_type"],
            serde_json::json!("RETRIEVAL_DOCUMENT")
        );
        assert_eq!(
            builder_req.body["instances"][0]["title"],
            serde_json::json!("vertex-doc")
        );
        assert_eq!(
            builder_req.body["parameters"]["outputDimensionality"],
            serde_json::json!(256)
        );
        assert_eq!(
            builder_req.body["parameters"]["autoTruncate"],
            serde_json::json!(true)
        );
    }

    #[tokio::test]
    async fn vertex_builder_config_registry_chat_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client =
            siumai_provider_google_vertex::providers::vertex::GoogleVertexBuilder::new(
                siumai_provider_google_vertex::builder::BuilderBase::default(),
            )
            .api_key("ctx-key")
            .base_url("https://example.com/custom")
            .model("gemini-2.5-flash")
            .fetch(Arc::new(builder_transport.clone()))
            .build()
            .expect("build builder client");

        let config_client =
            siumai_provider_google_vertex::providers::vertex::GoogleVertexClient::from_config(
                siumai_provider_google_vertex::providers::vertex::GoogleVertexConfig::new(
                    "https://example.com/custom",
                    "gemini-2.5-flash",
                )
                .with_api_key("ctx-key")
                .with_http_transport(Arc::new(config_transport.clone())),
            )
            .expect("build config client");

        let factory = GoogleVertexProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "gemini-2.5-flash",
                &BuildContext {
                    provider_id: Some("vertex".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model("gemini-2.5-flash");

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert!(
            builder_req
                .url
                .contains("/models/gemini-2.5-flash:generateContent?key=ctx-key"),
            "unexpected url: {}",
            builder_req.url
        );
        assert_eq!(
            builder_req.body["contents"][0]["role"],
            serde_json::json!("user")
        );
        assert_eq!(
            builder_req.body["contents"][0]["parts"][0]["text"],
            serde_json::json!("hi")
        );
    }

    #[tokio::test]
    async fn vertex_builder_config_registry_chat_stream_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client =
            siumai_provider_google_vertex::providers::vertex::GoogleVertexBuilder::new(
                siumai_provider_google_vertex::builder::BuilderBase::default(),
            )
            .api_key("ctx-key")
            .base_url("https://example.com/custom")
            .model("gemini-2.5-flash")
            .fetch(Arc::new(builder_transport.clone()))
            .build()
            .expect("build builder client");

        let config_client =
            siumai_provider_google_vertex::providers::vertex::GoogleVertexClient::from_config(
                siumai_provider_google_vertex::providers::vertex::GoogleVertexConfig::new(
                    "https://example.com/custom",
                    "gemini-2.5-flash",
                )
                .with_api_key("ctx-key")
                .with_http_transport(Arc::new(config_transport.clone())),
            )
            .expect("build config client");

        let factory = GoogleVertexProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "gemini-2.5-flash",
                &BuildContext {
                    provider_id: Some("vertex".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model("gemini-2.5-flash");

        let _ = builder_client.chat_stream_request(request.clone()).await;
        let _ = config_client.chat_stream_request(request.clone()).await;
        let _ = registry_client.chat_stream_request(request).await;

        let builder_req = builder_transport
            .take_stream()
            .expect("builder stream request");
        let config_req = config_transport
            .take_stream()
            .expect("config stream request");
        let registry_req = registry_transport
            .take_stream()
            .expect("registry stream request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert!(
            builder_req
                .url
                .contains("/models/gemini-2.5-flash:streamGenerateContent?alt=sse&key=ctx-key"),
            "unexpected url: {}",
            builder_req.url
        );
        assert_eq!(
            header_value(&builder_req, "accept"),
            Some("text/event-stream".to_string())
        );
        assert_eq!(
            builder_req.body["contents"][0]["parts"][0]["text"],
            serde_json::json!("hi")
        );
    }

    #[tokio::test]
    async fn vertex_builder_config_registry_stable_request_options_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let builder_client =
            siumai_provider_google_vertex::providers::vertex::GoogleVertexBuilder::new(
                siumai_provider_google_vertex::builder::BuilderBase::default(),
            )
            .api_key("ctx-key")
            .base_url("https://example.com/custom")
            .model("gemini-2.5-flash")
            .fetch(Arc::new(builder_transport.clone()))
            .build()
            .expect("build builder client");

        let config_client =
            siumai_provider_google_vertex::providers::vertex::GoogleVertexClient::from_config(
                siumai_provider_google_vertex::providers::vertex::GoogleVertexConfig::new(
                    "https://example.com/custom",
                    "gemini-2.5-flash",
                )
                .with_api_key("ctx-key")
                .with_http_transport(Arc::new(config_transport.clone())),
            )
            .expect("build config client");

        let factory = GoogleVertexProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "gemini-2.5-flash",
                &BuildContext {
                    provider_id: Some("vertex".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model("gemini-2.5-flash")
            .with_response_format(crate::types::ResponseFormat::json_schema(schema.clone()))
            .with_provider_option(
                "vertex",
                serde_json::json!({
                    "thinkingConfig": {
                        "thinkingBudget": 2048,
                        "includeThoughts": true
                    },
                    "structuredOutputs": true
                }),
            );

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert!(
            builder_req
                .url
                .contains("/models/gemini-2.5-flash:generateContent?key=ctx-key"),
            "unexpected url: {}",
            builder_req.url
        );
        assert_eq!(
            builder_req.body["generationConfig"]["thinkingConfig"],
            serde_json::json!({
                "thinkingBudget": 2048,
                "includeThoughts": true
            })
        );
        assert_eq!(
            builder_req.body["generationConfig"]["responseMimeType"],
            serde_json::json!("application/json")
        );
        assert!(
            builder_req.body["generationConfig"]
                .get("responseSchema")
                .is_some()
        );
        assert!(
            builder_req.body["generationConfig"]
                .get("responseJsonSchema")
                .is_none()
        );
    }

    #[tokio::test]
    async fn vertex_builder_config_registry_tool_choice_request_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();

        let builder_client =
            siumai_provider_google_vertex::providers::vertex::GoogleVertexBuilder::new(
                siumai_provider_google_vertex::builder::BuilderBase::default(),
            )
            .api_key("ctx-key")
            .base_url("https://example.com/custom")
            .model("gemini-2.5-flash")
            .fetch(Arc::new(builder_transport.clone()))
            .build()
            .expect("build builder client");

        let config_client =
            siumai_provider_google_vertex::providers::vertex::GoogleVertexClient::from_config(
                siumai_provider_google_vertex::providers::vertex::GoogleVertexConfig::new(
                    "https://example.com/custom",
                    "gemini-2.5-flash",
                )
                .with_api_key("ctx-key")
                .with_http_transport(Arc::new(config_transport.clone())),
            )
            .expect("build config client");

        let factory = GoogleVertexProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "gemini-2.5-flash",
                &BuildContext {
                    provider_id: Some("vertex".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model("gemini-2.5-flash")
            .with_tools(vec![crate::types::Tool::function(
                "lookup_weather",
                "Look up the weather",
                serde_json::json!({
                    "type": "object",
                    "properties": { "location": { "type": "string" } },
                    "required": ["location"],
                    "additionalProperties": false
                }),
            )])
            .with_tool_choice(crate::types::ToolChoice::None);

        let _ = builder_client.chat_request(request.clone()).await;
        let _ = config_client.chat_request(request.clone()).await;
        let _ = registry_client.chat_request(request).await;

        let builder_req = builder_transport.take().expect("builder request");
        let config_req = config_transport.take().expect("config request");
        let registry_req = registry_transport.take().expect("registry request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert!(
            builder_req
                .url
                .contains("/models/gemini-2.5-flash:generateContent?key=ctx-key"),
            "unexpected url: {}",
            builder_req.url
        );
        assert_eq!(
            builder_req.body["toolConfig"],
            serde_json::json!({
                "functionCallingConfig": { "mode": "NONE" }
            })
        );
    }

    #[tokio::test]
    async fn vertex_builder_config_registry_stream_stable_request_options_are_equivalent() {
        let _lock = lock_env();

        let builder_transport = CaptureTransport::default();
        let config_transport = CaptureTransport::default();
        let registry_transport = CaptureTransport::default();
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let builder_client =
            siumai_provider_google_vertex::providers::vertex::GoogleVertexBuilder::new(
                siumai_provider_google_vertex::builder::BuilderBase::default(),
            )
            .api_key("ctx-key")
            .base_url("https://example.com/custom")
            .model("gemini-2.5-flash")
            .fetch(Arc::new(builder_transport.clone()))
            .build()
            .expect("build builder client");

        let config_client =
            siumai_provider_google_vertex::providers::vertex::GoogleVertexClient::from_config(
                siumai_provider_google_vertex::providers::vertex::GoogleVertexConfig::new(
                    "https://example.com/custom",
                    "gemini-2.5-flash",
                )
                .with_api_key("ctx-key")
                .with_http_transport(Arc::new(config_transport.clone())),
            )
            .expect("build config client");

        let factory = GoogleVertexProviderFactory;
        let registry_client = factory
            .language_model_with_ctx(
                "gemini-2.5-flash",
                &BuildContext {
                    provider_id: Some("vertex".to_string()),
                    api_key: Some("ctx-key".to_string()),
                    base_url: Some("https://example.com/custom".to_string()),
                    http_transport: Some(Arc::new(registry_transport.clone())),
                    ..Default::default()
                },
            )
            .await
            .expect("build registry client");

        let request = make_chat_request_with_model("gemini-2.5-flash")
            .with_tools(vec![crate::types::Tool::function(
                "lookup_weather",
                "Look up the weather",
                serde_json::json!({
                    "type": "object",
                    "properties": { "location": { "type": "string" } },
                    "required": ["location"],
                    "additionalProperties": false
                }),
            )])
            .with_tool_choice(crate::types::ToolChoice::None)
            .with_response_format(crate::types::ResponseFormat::json_schema(schema.clone()))
            .with_provider_option(
                "vertex",
                serde_json::json!({
                    "thinkingConfig": {
                        "thinkingBudget": 2048,
                        "includeThoughts": true
                    },
                    "structuredOutputs": true
                }),
            );

        let _ = builder_client.chat_stream_request(request.clone()).await;
        let _ = config_client.chat_stream_request(request.clone()).await;
        let _ = registry_client.chat_stream_request(request).await;

        let builder_req = builder_transport
            .take_stream()
            .expect("builder stream request");
        let config_req = config_transport
            .take_stream()
            .expect("config stream request");
        let registry_req = registry_transport
            .take_stream()
            .expect("registry stream request");

        assert_requests_equivalent(&builder_req, &config_req);
        assert_requests_equivalent(&builder_req, &registry_req);
        assert!(
            builder_req
                .url
                .contains("/models/gemini-2.5-flash:streamGenerateContent?alt=sse&key=ctx-key"),
            "unexpected url: {}",
            builder_req.url
        );
        assert_eq!(
            header_value(&builder_req, "accept"),
            Some("text/event-stream".to_string())
        );
        assert_eq!(
            builder_req.body["generationConfig"]["thinkingConfig"],
            serde_json::json!({
                "thinkingBudget": 2048,
                "includeThoughts": true
            })
        );
        assert_eq!(
            builder_req.body["generationConfig"]["responseMimeType"],
            serde_json::json!("application/json")
        );
        assert!(
            builder_req.body["generationConfig"]
                .get("responseSchema")
                .is_some()
        );
        assert!(
            builder_req.body["generationConfig"]
                .get("responseJsonSchema")
                .is_none()
        );
        assert_eq!(
            builder_req.body["toolConfig"],
            serde_json::json!({
                "functionCallingConfig": { "mode": "NONE" }
            })
        );
        assert_eq!(
            builder_req.body["tools"][0]["functionDeclarations"][0]["name"],
            serde_json::json!("lookup_weather")
        );
    }

    #[tokio::test]
    async fn vertex_registry_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let vertex_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "vertex".to_string(),
            Arc::new(GoogleVertexProviderFactory) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "vertex",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/custom")
                    .fetch(Arc::new(vertex_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .language_model("vertex:gemini-2.5-flash")
            .expect("build vertex handle");

        let _ = handle
            .chat_request(make_chat_request_with_model("gemini-2.5-flash"))
            .await;

        let req = vertex_transport.take().expect("captured vertex request");
        assert!(global_transport.take().is_none());
        assert!(
            req.url.starts_with("https://example.com/custom"),
            "unexpected url: {}",
            req.url
        );
        assert!(
            req.url
                .contains("/models/gemini-2.5-flash:generateContent?key=ctx-key"),
            "unexpected url: {}",
            req.url
        );
        assert_eq!(
            req.body["contents"][0]["parts"][0]["text"],
            serde_json::json!("hi")
        );
    }

    #[tokio::test]
    async fn vertex_registry_stream_handle_prefers_provider_specific_build_overrides() {
        let _lock = lock_env();

        let global_transport = CaptureTransport::default();
        let vertex_transport = CaptureTransport::default();
        let mut providers = std::collections::HashMap::new();
        providers.insert(
            "vertex".to_string(),
            Arc::new(GoogleVertexProviderFactory) as Arc<dyn ProviderFactory>,
        );

        let registry = crate::registry::builder::RegistryBuilder::new(providers)
            .with_api_key("global-key")
            .with_base_url("https://example.com/global")
            .fetch(Arc::new(global_transport.clone()))
            .with_provider_build_overrides(
                "vertex",
                crate::registry::ProviderBuildOverrides::default()
                    .with_api_key("ctx-key")
                    .with_base_url("https://example.com/custom")
                    .fetch(Arc::new(vertex_transport.clone())),
            )
            .auto_middleware(false)
            .build()
            .expect("build registry");

        let handle = registry
            .language_model("vertex:gemini-2.5-flash")
            .expect("build vertex handle");

        let _ = handle
            .chat_stream_request(make_chat_request_with_model("gemini-2.5-flash"))
            .await;

        let req = vertex_transport
            .take_stream()
            .expect("captured vertex stream request");
        assert!(global_transport.take().is_none());
        assert!(global_transport.take_stream().is_none());
        assert!(
            req.url.starts_with("https://example.com/custom"),
            "unexpected url: {}",
            req.url
        );
        assert!(
            req.url
                .contains("/models/gemini-2.5-flash:streamGenerateContent?alt=sse&key=ctx-key"),
            "unexpected url: {}",
            req.url
        );
        assert_eq!(
            header_value(&req, "accept"),
            Some("text/event-stream".to_string())
        );
        assert_eq!(
            req.body["contents"][0]["parts"][0]["text"],
            serde_json::json!("hi")
        );
    }
}
