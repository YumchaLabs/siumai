#![cfg(any(
    feature = "openai",
    feature = "azure",
    feature = "anthropic",
    feature = "xai",
    feature = "groq",
    feature = "cohere",
    feature = "togetherai",
    feature = "deepinfra",
    feature = "bedrock",
    feature = "deepseek",
    feature = "ollama",
    feature = "minimaxi",
    feature = "google",
    feature = "google-vertex"
))]
#![allow(dead_code)]
#![allow(deprecated)]

use async_trait::async_trait;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use siumai::compat::Provider;
use siumai::experimental::execution::http::transport::{
    HttpTransport, HttpTransportMultipartRequest, HttpTransportRequest, HttpTransportResponse,
    HttpTransportStreamBody, HttpTransportStreamResponse,
};
#[allow(unused_imports)]
use siumai::extensions::{AudioCapability, ImageExtras};
use siumai::prelude::compat::Siumai;
#[allow(unused_imports)]
use siumai::prelude::unified::{
    ChatCapability, ChatMessage, ChatRequest, CompletionCapability, CompletionRequest,
    EmbeddingModel, EmbeddingRequest, ImageGenerationCapability, ImageGenerationRequest, LlmError,
    RerankRequest, RerankingModel, SttRequest, TtsRequest,
};
#[cfg(feature = "google-vertex")]
use siumai::provider_ext::anthropic_vertex::{
    AnthropicChatResponseExt, VertexAnthropicChatRequestExt,
};
#[allow(unused_imports)]
use siumai_registry::compat::client::LlmClient;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy)]
struct BuiltInProviderRegistryHarness<'a> {
    registry_provider_id: &'a str,
    built_in_provider_id: &'a str,
}

impl<'a> BuiltInProviderRegistryHarness<'a> {
    const fn new(registry_provider_id: &'a str, built_in_provider_id: &'a str) -> Self {
        Self {
            registry_provider_id,
            built_in_provider_id,
        }
    }

    const fn same(provider_id: &'a str) -> Self {
        Self::new(provider_id, provider_id)
    }

    fn factory(self) -> Arc<dyn siumai::registry::ProviderFactory> {
        siumai::registry::builtin_provider_factory(self.built_in_provider_id).unwrap_or_else(
            |err| {
                panic!(
                    "{} built-in provider factory: {err:?}",
                    self.built_in_provider_id
                )
            },
        )
    }

    fn providers(self) -> HashMap<String, Arc<dyn siumai::registry::ProviderFactory>> {
        let mut providers = HashMap::new();
        providers.insert(self.registry_provider_id.to_string(), self.factory());
        providers
    }

    fn registry_builder(self) -> siumai::registry::builder::RegistryBuilder {
        siumai::registry::builder::RegistryBuilder::new(self.providers())
    }
}

fn built_in_registry_factory(provider_id: &str) -> Arc<dyn siumai::registry::ProviderFactory> {
    BuiltInProviderRegistryHarness::same(provider_id).factory()
}

fn built_in_registry_providers(
    registry_provider_id: &str,
    built_in_provider_id: &str,
) -> HashMap<String, Arc<dyn siumai::registry::ProviderFactory>> {
    BuiltInProviderRegistryHarness::new(registry_provider_id, built_in_provider_id).providers()
}

fn built_in_registry_builder(
    registry_provider_id: &str,
    built_in_provider_id: &str,
) -> siumai::registry::builder::RegistryBuilder {
    BuiltInProviderRegistryHarness::new(registry_provider_id, built_in_provider_id)
        .registry_builder()
}

#[allow(dead_code)]
#[derive(Clone, Default)]
struct CaptureTransport {
    last: Arc<Mutex<Option<HttpTransportRequest>>>,
    last_stream: Arc<Mutex<Option<HttpTransportRequest>>>,
}

impl CaptureTransport {
    #[allow(dead_code)]
    fn take(&self) -> Option<HttpTransportRequest> {
        self.last.lock().expect("lock").take()
    }

    #[allow(dead_code)]
    fn take_stream(&self) -> Option<HttpTransportRequest> {
        self.last_stream.lock().expect("lock").take()
    }
}

#[allow(dead_code)]
fn remember_stream_tool_call_id(current: &mut Option<String>, next: &str) {
    if current.is_none() {
        *current = Some(next.to_string());
    } else {
        assert_eq!(current.as_deref(), Some(next));
    }
}

#[allow(dead_code)]
fn record_streamed_tool_part(
    event: &siumai::prelude::unified::ChatStreamEvent,
    tool_call_id: &mut Option<String>,
    tool_name: &mut Option<String>,
    arguments: &mut String,
) {
    match event.part_ref() {
        Some(siumai::prelude::unified::ChatStreamPart::ToolInputStart {
            id,
            tool_name: name,
            ..
        }) => {
            remember_stream_tool_call_id(tool_call_id, id);
            if tool_name.is_none() {
                *tool_name = Some(name.clone());
            }
        }
        Some(siumai::prelude::unified::ChatStreamPart::ToolInputDelta { id, delta, .. }) => {
            remember_stream_tool_call_id(tool_call_id, id);
            arguments.push_str(delta);
        }
        Some(siumai::prelude::unified::ChatStreamPart::ToolCall(call)) => {
            remember_stream_tool_call_id(tool_call_id, &call.tool_call_id);
            if tool_name.is_none() {
                *tool_name = Some(call.tool_name.clone());
            }
            *arguments = call.input.clone();
        }
        _ => {}
    }
}

#[async_trait]
impl HttpTransport for CaptureTransport {
    async fn execute_json(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        *self.last.lock().expect("lock") = Some(request);

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
    ) -> Result<HttpTransportStreamResponse, LlmError> {
        *self.last_stream.lock().expect("lock") = Some(request);

        let mut headers = HeaderMap::new();
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/x-ndjson"),
        );

        Ok(HttpTransportStreamResponse {
            status: 200,
            headers,
            body: HttpTransportStreamBody::from_bytes(b"data: [DONE]\n\n".to_vec()),
        })
    }
}

#[allow(dead_code)]
#[derive(Clone, Default)]
struct MixedCaptureTransport {
    last: Arc<Mutex<Option<HttpTransportRequest>>>,
    last_stream: Arc<Mutex<Option<HttpTransportRequest>>>,
    last_multipart: Arc<Mutex<Option<HttpTransportMultipartRequest>>>,
}

impl MixedCaptureTransport {
    fn take(&self) -> Option<HttpTransportRequest> {
        self.last.lock().expect("lock mixed request").take()
    }

    fn take_stream(&self) -> Option<HttpTransportRequest> {
        self.last_stream
            .lock()
            .expect("lock mixed stream request")
            .take()
    }

    fn take_multipart(&self) -> Option<HttpTransportMultipartRequest> {
        self.last_multipart
            .lock()
            .expect("lock mixed multipart request")
            .take()
    }
}

#[async_trait]
impl HttpTransport for MixedCaptureTransport {
    async fn execute_json(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        *self.last.lock().expect("lock mixed request") = Some(request);

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
    ) -> Result<HttpTransportStreamResponse, LlmError> {
        *self.last_stream.lock().expect("lock mixed stream request") = Some(request);

        let mut headers = HeaderMap::new();
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/x-ndjson"),
        );

        Ok(HttpTransportStreamResponse {
            status: 200,
            headers,
            body: HttpTransportStreamBody::from_bytes(b"data: [DONE]\n\n".to_vec()),
        })
    }

    async fn execute_multipart(
        &self,
        request: HttpTransportMultipartRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        *self
            .last_multipart
            .lock()
            .expect("lock mixed multipart request") = Some(request);

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        Ok(HttpTransportResponse {
            status: 200,
            headers,
            body: br#"{"text":"should not be used"}"#.to_vec(),
        })
    }
}

#[allow(dead_code)]
fn make_chat_request_with_model(model: &str) -> ChatRequest {
    let mut req = ChatRequest::new(vec![ChatMessage::user("hi").build()]);
    req.common_params.model = model.to_string();
    req
}

fn make_completion_request_with_model(model: &str) -> siumai::prelude::unified::CompletionRequest {
    siumai::prelude::unified::CompletionRequest::new("hi").with_model(model)
}

#[allow(dead_code)]
fn make_rerank_request_with_model(model: &str) -> RerankRequest {
    RerankRequest::new(
        model.to_string(),
        "query".to_string(),
        vec!["doc-1".to_string(), "doc-2".to_string()],
    )
}

fn make_image_request_with_model(model: &str) -> ImageGenerationRequest {
    ImageGenerationRequest {
        prompt: "a tiny purple robot".to_string(),
        negative_prompt: Some("blurry".to_string()),
        size: Some("1024x1024".to_string()),
        aspect_ratio: None,
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
    }
}

fn make_data_url_image_edit_request_with_model(
    model: &str,
) -> siumai::extensions::types::ImageEditRequest {
    siumai::extensions::types::ImageEditRequest {
        images: vec![siumai::extensions::types::ImageEditInput::url(
            "data:image/png;base64,aW1hZ2Utb25l",
        )],
        mask: Some(siumai::extensions::types::ImageEditInput::url(
            "data:image/png;base64,bWFzay1vbmU=",
        )),
        prompt: "replace the background with a neon skyline".to_string(),
        model: Some(model.to_string()),
        count: Some(1),
        size: Some("1024x1024".to_string()),
        aspect_ratio: None,
        seed: None,
        response_format: Some("b64_json".to_string()),
        extra_params: Default::default(),
        provider_options_map: Default::default(),
        http_config: None,
    }
}

fn make_data_url_image_variation_request_with_model(
    model: &str,
) -> siumai::extensions::types::ImageVariationRequest {
    siumai::extensions::types::ImageVariationRequest {
        image: siumai::extensions::types::ImageEditInput::url("data:image/png;base64,aW1hZ2Utb25l"),
        model: Some(model.to_string()),
        count: Some(2),
        size: None,
        aspect_ratio: Some("16:9".to_string()),
        seed: Some(7),
        response_format: Some("b64_json".to_string()),
        extra_params: std::collections::HashMap::from([(
            "prompt".to_string(),
            serde_json::json!("keep the subject and explore new backgrounds"),
        )]),
        provider_options_map: Default::default(),
        http_config: None,
    }
}

fn assert_unsupported_operation(err: &LlmError) {
    assert!(matches!(err, LlmError::UnsupportedOperation(_)));
}

fn assert_capture_transports_unused(transports: &[&CaptureTransport]) {
    for transport in transports {
        assert!(transport.take().is_none());
        assert!(transport.take_stream().is_none());
    }
}

fn assert_mixed_capture_transports_unused(transports: &[&MixedCaptureTransport]) {
    for transport in transports {
        assert!(transport.take().is_none());
        assert!(transport.take_stream().is_none());
        assert!(transport.take_multipart().is_none());
    }
}

fn assert_no_deferred_capability_leaks(client: &dyn siumai::experimental::client::LlmClient) {
    assert!(client.as_embedding_capability().is_none());
    assert!(client.as_image_generation_capability().is_none());
    assert!(client.as_rerank_capability().is_none());
}

fn header_value(req: &HttpTransportRequest, key: &str) -> Option<String> {
    req.headers
        .get(key)
        .and_then(|v| v.to_str().ok())
        .map(ToString::to_string)
}

fn assert_requests_equivalent(left: &HttpTransportRequest, right: &HttpTransportRequest) {
    assert_eq!(left.url, right.url);
    assert_eq!(
        header_value(left, "authorization"),
        header_value(right, "authorization")
    );
    assert_eq!(header_value(left, "accept"), header_value(right, "accept"));
    assert_eq!(left.body, right.body);
}

fn normalize_multipart_body(req: &HttpTransportMultipartRequest) -> String {
    let mut body = String::from_utf8_lossy(&req.body).into_owned();
    if let Some(content_type) = req.headers.get(CONTENT_TYPE).and_then(|v| v.to_str().ok())
        && let Some(boundary) = content_type.split("boundary=").nth(1)
    {
        body = body.replace(boundary.trim(), "<BOUNDARY>");
    }
    body
}

fn assert_multipart_requests_equivalent(
    left: &HttpTransportMultipartRequest,
    right: &HttpTransportMultipartRequest,
) {
    assert_eq!(left.url, right.url);
    assert_eq!(
        left.headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .map(ToString::to_string),
        right
            .headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .map(ToString::to_string)
    );
    assert!(
        left.headers
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.starts_with("multipart/form-data; boundary="))
    );
    assert!(
        right
            .headers
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.starts_with("multipart/form-data; boundary="))
    );
    assert_eq!(
        normalize_multipart_body(left),
        normalize_multipart_body(right)
    );
}

#[allow(dead_code)]
#[derive(Clone)]
struct MultipartCaptureTransport {
    response_body: Arc<Vec<u8>>,
    last: Arc<Mutex<Option<HttpTransportMultipartRequest>>>,
}

impl MultipartCaptureTransport {
    fn new(response: serde_json::Value) -> Self {
        Self {
            response_body: Arc::new(serde_json::to_vec(&response).expect("response json")),
            last: Arc::new(Mutex::new(None)),
        }
    }

    fn take(&self) -> Option<HttpTransportMultipartRequest> {
        self.last.lock().expect("lock multipart request").take()
    }
}

#[async_trait]
impl HttpTransport for MultipartCaptureTransport {
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
        *self.last.lock().expect("lock multipart request") = Some(request);

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        Ok(HttpTransportResponse {
            status: 200,
            headers,
            body: self.response_body.as_ref().clone(),
        })
    }
}

#[allow(dead_code)]
#[derive(Clone)]
struct BinaryCaptureTransport {
    response_body: Arc<Vec<u8>>,
    response_content_type: &'static str,
    last: Arc<Mutex<Option<HttpTransportRequest>>>,
}

impl BinaryCaptureTransport {
    fn new(response_body: Vec<u8>, response_content_type: &'static str) -> Self {
        Self {
            response_body: Arc::new(response_body),
            response_content_type,
            last: Arc::new(Mutex::new(None)),
        }
    }

    fn take(&self) -> Option<HttpTransportRequest> {
        self.last.lock().expect("lock binary request").take()
    }
}

#[async_trait]
impl HttpTransport for BinaryCaptureTransport {
    async fn execute_json(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        *self.last.lock().expect("lock binary request") = Some(request);

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

#[allow(dead_code)]
#[derive(Clone)]
struct JsonSuccessTransport {
    response_body: Arc<Vec<u8>>,
    last: Arc<Mutex<Option<HttpTransportRequest>>>,
}

#[allow(dead_code)]
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

#[allow(dead_code)]
#[derive(Clone)]
struct JsonStreamSuccessTransport {
    response_body: Arc<Vec<u8>>,
    last_stream: Arc<Mutex<Option<HttpTransportRequest>>>,
}

#[allow(dead_code)]
impl JsonStreamSuccessTransport {
    fn new(response_body: Vec<u8>) -> Self {
        Self {
            response_body: Arc::new(response_body),
            last_stream: Arc::new(Mutex::new(None)),
        }
    }

    fn take_stream(&self) -> Option<HttpTransportRequest> {
        self.last_stream.lock().expect("lock success stream").take()
    }
}

#[async_trait]
impl HttpTransport for JsonStreamSuccessTransport {
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
        *self.last_stream.lock().expect("lock success stream") = Some(request);

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

#[allow(dead_code)]
#[derive(Clone)]
struct SseSuccessTransport {
    response_body: Arc<Vec<u8>>,
    last_stream: Arc<Mutex<Option<HttpTransportRequest>>>,
}

#[allow(dead_code)]
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

#[allow(dead_code)]
#[derive(Clone)]
struct MultipartSseSuccessTransport {
    response_body: Arc<Vec<u8>>,
    last_stream: Arc<Mutex<Option<HttpTransportMultipartRequest>>>,
}

#[allow(dead_code)]
impl MultipartSseSuccessTransport {
    fn new(response_body: Vec<u8>) -> Self {
        Self {
            response_body: Arc::new(response_body),
            last_stream: Arc::new(Mutex::new(None)),
        }
    }

    fn take_stream(&self) -> Option<HttpTransportMultipartRequest> {
        self.last_stream
            .lock()
            .expect("lock multipart sse success stream")
            .take()
    }
}

#[async_trait]
impl HttpTransport for MultipartSseSuccessTransport {
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

    async fn execute_multipart_stream(
        &self,
        request: HttpTransportMultipartRequest,
    ) -> Result<HttpTransportStreamResponse, LlmError> {
        *self
            .last_stream
            .lock()
            .expect("lock multipart sse success stream") = Some(request);

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

#[cfg(feature = "openai")]
#[path = "provider_public_path_parity/openai_public_path.rs"]
mod openai_public_path;

#[cfg(feature = "azure")]
#[path = "provider_public_path_parity/azure_public_path.rs"]
mod azure_public_path;

#[cfg(feature = "google")]
#[path = "provider_public_path_parity/gemini_public_path.rs"]
mod gemini_public_path;

#[cfg(feature = "cohere")]
#[path = "provider_public_path_parity/cohere_public_path.rs"]
mod cohere_public_path;

#[cfg(feature = "togetherai")]
#[path = "provider_public_path_parity/togetherai_public_path.rs"]
mod togetherai_public_path;

#[cfg(feature = "deepinfra")]
#[path = "provider_public_path_parity/deepinfra_public_path.rs"]
mod deepinfra_public_path;

#[cfg(feature = "google-vertex")]
#[path = "provider_public_path_parity/vertex_maas_public_path.rs"]
mod vertex_maas_public_path;

#[cfg(feature = "google-vertex")]
#[path = "provider_public_path_parity/google_vertex_xai_public_path.rs"]
mod google_vertex_xai_public_path;

#[cfg(feature = "deepseek")]
#[path = "provider_public_path_parity/deepseek_public_path.rs"]
mod deepseek_public_path;

#[cfg(feature = "openai")]
#[path = "provider_public_path_parity/openai_compatible_audio_public_path.rs"]
mod openai_compatible_audio_public_path;

#[cfg(feature = "groq")]
#[path = "provider_public_path_parity/groq_public_path.rs"]
mod groq_public_path;

#[cfg(feature = "ollama")]
#[path = "provider_public_path_parity/ollama_public_path.rs"]
mod ollama_public_path;

#[cfg(feature = "minimaxi")]
#[path = "provider_public_path_parity/minimaxi_public_path.rs"]
mod minimaxi_public_path;

#[cfg(feature = "bedrock")]
#[path = "provider_public_path_parity/bedrock_public_path.rs"]
mod bedrock_public_path;

#[cfg(feature = "anthropic")]
#[path = "provider_public_path_parity/anthropic_public_path.rs"]
mod anthropic_public_path;

#[cfg(feature = "google-vertex")]
#[path = "provider_public_path_parity/vertex_public_path.rs"]
mod vertex_public_path;

#[cfg(feature = "xai")]
#[path = "provider_public_path_parity/xai_public_path.rs"]
mod xai_public_path;
