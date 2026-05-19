use super::*;
use futures_util::StreamExt;
use siumai::experimental::client::LlmClient;
use siumai::extensions::types::{FileListQuery, FileUploadRequest};
use siumai::extensions::{
    FileManagementCapability, MusicGenerationCapability, SpeechExtras, VideoGenerationCapability,
};
use siumai::prelude::unified::{
    EmbeddingExtensions, EmbeddingRequest, FinishReason, Tool, ToolChoice,
};
use siumai::provider_ext::minimaxi::ext::music::MinimaxiMusicRequestBuilder;
use siumai::provider_ext::minimaxi::ext::video::MinimaxiVideoRequestBuilder;
use siumai::provider_ext::minimaxi::{
    MinimaxiChatRequestExt, MinimaxiChatResponseExt, MinimaxiOptions, MinimaxiTtsOptions,
    MinimaxiTtsRequestExt, MinimaxiVideoOptions, MinimaxiVideoRequestExt,
};
use std::collections::HashMap;
use wiremock::matchers::{method, path, query_param};
use wiremock::{Mock, MockServer, Request as WiremockRequest, ResponseTemplate};

#[derive(Clone, Default)]
struct MinimaxiCaptureTransport {
    last: Arc<Mutex<Option<HttpTransportRequest>>>,
    last_stream: Arc<Mutex<Option<HttpTransportRequest>>>,
}

impl MinimaxiCaptureTransport {
    fn take(&self) -> Option<HttpTransportRequest> {
        self.last.lock().expect("lock").take()
    }

    fn take_stream(&self) -> Option<HttpTransportRequest> {
        self.last_stream.lock().expect("lock").take()
    }
}

#[derive(Clone)]
struct MinimaxiJsonSuccessTransport {
    response_body: Arc<Vec<u8>>,
    last: Arc<Mutex<Option<HttpTransportRequest>>>,
}

impl MinimaxiJsonSuccessTransport {
    fn new(response: serde_json::Value) -> Self {
        Self {
            response_body: Arc::new(serde_json::to_vec(&response).expect("response json")),
            last: Arc::new(Mutex::new(None)),
        }
    }

    fn take(&self) -> Option<HttpTransportRequest> {
        self.last.lock().expect("lock success").take()
    }
}

fn minimaxi_stream_bytes() -> Vec<u8> {
    concat!(
            r#"data: {"type":"message_start","message":{"id":"msg_test","model":"MiniMax-M2","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}}

"#,
            r#"data: {"type":"message_stop"}

"#
        )
        .as_bytes()
        .to_vec()
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

fn wiremock_header_value(req: &WiremockRequest, key: &str) -> Option<String> {
    req.headers
        .get(key)
        .and_then(|v| v.to_str().ok())
        .map(ToString::to_string)
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

#[async_trait]
impl HttpTransport for MinimaxiCaptureTransport {
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
                body: br#"{"type":"error","error":{"type":"authentication_error","message":"unauthorized"}}"#
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
            HeaderValue::from_static("text/event-stream; charset=utf-8"),
        );

        Ok(HttpTransportStreamResponse {
            status: 200,
            headers,
            body: HttpTransportStreamBody::from_bytes(minimaxi_stream_bytes()),
        })
    }
}

#[async_trait]
impl HttpTransport for MinimaxiJsonSuccessTransport {
    async fn execute_json(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        *self.last.lock().expect("lock success") = Some(request);

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        Ok(HttpTransportResponse {
            status: 200,
            headers,
            body: self.response_body.as_ref().clone(),
        })
    }
}

fn minimaxi_registry_providers() -> HashMap<String, Arc<dyn siumai::registry::ProviderFactory>> {
    built_in_registry_providers("minimaxi", "minimaxi")
}

fn minimaxi_registry_builder() -> siumai::registry::builder::RegistryBuilder {
    built_in_registry_builder("minimaxi", "minimaxi")
}

fn minimaxi_wiremock_base_url(server: &MockServer) -> String {
    format!("{}/anthropic/v1", server.uri())
}

fn make_minimaxi_server_override_registry(
    global_server: &MockServer,
    minimaxi_server: &MockServer,
) -> siumai::registry::ProviderRegistryHandle {
    minimaxi_registry_builder()
        .with_api_key("global-key")
        .with_base_url(minimaxi_wiremock_base_url(global_server))
        .with_provider_api_key_base_url(
            "minimaxi",
            "ctx-key",
            minimaxi_wiremock_base_url(minimaxi_server),
        )
        .auto_middleware(false)
        .build()
        .expect("build registry")
}

fn make_minimaxi_transport_override_registry(
    global_transport: Arc<dyn HttpTransport>,
    minimaxi_transport: Arc<dyn HttpTransport>,
) -> siumai::registry::ProviderRegistryHandle {
    minimaxi_registry_builder()
        .with_api_key("global-key")
        .with_base_url("https://example.com/global")
        .fetch(global_transport)
        .with_provider_api_key_base_url_fetch(
            "minimaxi",
            "ctx-key",
            "https://example.com/custom",
            minimaxi_transport,
        )
        .auto_middleware(false)
        .build()
        .expect("build registry")
}

fn make_registry(
    transport: Arc<dyn HttpTransport>,
    base_url: &str,
) -> siumai::registry::ProviderRegistryHandle {
    minimaxi_registry_builder()
        .with_provider_api_key_base_url_fetch("minimaxi", "test-key", base_url, transport)
        .build()
        .expect("build registry")
}

fn make_registry_without_transport(base_url: &str) -> siumai::registry::ProviderRegistryHandle {
    minimaxi_registry_builder()
        .with_provider_api_key_base_url("minimaxi", "test-key", base_url)
        .build()
        .expect("build registry")
}

fn assert_minimaxi_default_options_request(
    req: &HttpTransportRequest,
    base_url: &str,
    model: &str,
) {
    assert_eq!(req.url, format!("{base_url}/v1/messages"));
    assert_eq!(req.body["model"], serde_json::json!(model));
    assert_eq!(
        req.body["thinking"],
        serde_json::json!({
            "type": "enabled",
            "budget_tokens": 1024
        })
    );
    assert_eq!(
        req.body["output_format"],
        serde_json::json!({
            "type": "json_object"
        })
    );
    assert_eq!(
        header_value(req, "authorization"),
        Some("Bearer test-key".to_string())
    );
}

fn assert_minimaxi_default_options_stream_request(
    req: &HttpTransportRequest,
    base_url: &str,
    model: &str,
) {
    assert_minimaxi_default_options_request(req, base_url, model);
    assert_eq!(req.body["stream"], serde_json::json!(true));
    assert_eq!(
        header_value(req, "accept"),
        Some("text/event-stream".to_string())
    );
}

#[tokio::test]
async fn minimaxi_siumai_provider_config_chat_request_are_equivalent() {
    let siumai_transport = MinimaxiCaptureTransport::default();
    let provider_transport = MinimaxiCaptureTransport::default();
    let config_transport = MinimaxiCaptureTransport::default();

    let siumai_client = Siumai::builder()
        .minimaxi()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model("MiniMax-M2")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::minimaxi()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model("MiniMax-M2")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::minimaxi::MinimaxiClient::from_config(
        siumai::provider_ext::minimaxi::MinimaxiConfig::new("test-key")
            .with_base_url("https://example.com/custom")
            .with_model("MiniMax-M2")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = make_chat_request_with_model("MiniMax-M2");

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(siumai_req.body["model"], serde_json::json!("MiniMax-M2"));
    assert_eq!(
        header_value(&siumai_req, "authorization"),
        Some("Bearer test-key".to_string())
    );
}

#[tokio::test]
async fn minimaxi_default_options_match_public_request_shape() {
    let model = "MiniMax-M2";
    let base_url = "https://example.com/custom";
    let siumai_transport = MinimaxiCaptureTransport::default();
    let provider_transport = MinimaxiCaptureTransport::default();
    let config_transport = MinimaxiCaptureTransport::default();

    let siumai_client = Siumai::builder()
        .minimaxi()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .with_minimaxi_reasoning_budget(1024)
        .with_minimaxi_json_object()
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::minimaxi()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .with_reasoning_budget(1024)
        .with_json_object()
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::minimaxi::MinimaxiClient::from_config(
        siumai::provider_ext::minimaxi::MinimaxiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_reasoning_budget(1024)
            .with_json_object()
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = make_chat_request_with_model(model);

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_minimaxi_default_options_request(&siumai_req, base_url, model);
}

#[tokio::test]
async fn minimaxi_siumai_provider_config_stable_request_options_are_equivalent() {
    let siumai_transport = MinimaxiCaptureTransport::default();
    let provider_transport = MinimaxiCaptureTransport::default();
    let config_transport = MinimaxiCaptureTransport::default();

    let model = "MiniMax-M2";
    let base_url = "https://example.com/custom";

    let siumai_client = Siumai::builder()
        .minimaxi()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::minimaxi()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::minimaxi::MinimaxiClient::from_config(
        siumai::provider_ext::minimaxi::MinimaxiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = make_chat_request_with_model(model).with_minimaxi_options(
        MinimaxiOptions::new()
            .with_reasoning_budget(4096)
            .with_json_object(),
    );

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(
        siumai_req.body["thinking"],
        serde_json::json!({
            "type": "enabled",
            "budget_tokens": 4096
        })
    );
    assert_eq!(
        siumai_req.body["output_format"],
        serde_json::json!({
            "type": "json_object"
        })
    );
}

#[tokio::test]
async fn minimaxi_registry_stable_request_options_match_config_path() {
    let config_transport = MinimaxiCaptureTransport::default();
    let registry_transport = MinimaxiCaptureTransport::default();

    let model = "MiniMax-M2";
    let base_url = "https://example.com/custom";

    let config_client = siumai::provider_ext::minimaxi::MinimaxiClient::from_config(
        siumai::provider_ext::minimaxi::MinimaxiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("minimaxi:MiniMax-M2")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model).with_minimaxi_options(
        MinimaxiOptions::new()
            .with_reasoning_budget(4096)
            .with_json_object(),
    );

    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(
        registry_req.body["thinking"],
        serde_json::json!({
            "type": "enabled",
            "budget_tokens": 4096
        })
    );
    assert_eq!(
        registry_req.body["output_format"],
        serde_json::json!({
            "type": "json_object"
        })
    );
}

#[tokio::test]
async fn minimaxi_siumai_provider_config_chat_stream_request_are_equivalent() {
    let siumai_transport = MinimaxiCaptureTransport::default();
    let provider_transport = MinimaxiCaptureTransport::default();
    let config_transport = MinimaxiCaptureTransport::default();

    let siumai_client = Siumai::builder()
        .minimaxi()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model("MiniMax-M2")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::minimaxi()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model("MiniMax-M2")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::minimaxi::MinimaxiClient::from_config(
        siumai::provider_ext::minimaxi::MinimaxiConfig::new("test-key")
            .with_base_url("https://example.com/custom")
            .with_model("MiniMax-M2")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = make_chat_request_with_model("MiniMax-M2");

    let mut siumai_stream = siumai_client
        .chat_stream_request(request.clone())
        .await
        .expect("siumai stream ok");
    let mut provider_stream = provider_client
        .chat_stream_request(request.clone())
        .await
        .expect("provider stream ok");
    let mut config_stream = config_client
        .chat_stream_request(request)
        .await
        .expect("config stream ok");

    use futures_util::StreamExt;
    let _ = siumai_stream.next().await;
    let _ = provider_stream.next().await;
    let _ = config_stream.next().await;

    let siumai_req = siumai_transport
        .take_stream()
        .expect("siumai stream request");
    let provider_req = provider_transport
        .take_stream()
        .expect("provider stream request");
    let config_req = config_transport
        .take_stream()
        .expect("config stream request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(siumai_req.body["model"], serde_json::json!("MiniMax-M2"));
    assert_eq!(siumai_req.body["stream"], serde_json::json!(true));
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
}

#[tokio::test]
async fn minimaxi_default_options_match_public_stream_request_shape() {
    let model = "MiniMax-M2";
    let base_url = "https://example.com/custom";
    let siumai_transport = MinimaxiCaptureTransport::default();
    let provider_transport = MinimaxiCaptureTransport::default();
    let config_transport = MinimaxiCaptureTransport::default();

    let siumai_client = Siumai::builder()
        .minimaxi()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .with_minimaxi_reasoning_budget(1024)
        .with_minimaxi_json_object()
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::minimaxi()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .with_reasoning_budget(1024)
        .with_json_object()
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::minimaxi::MinimaxiClient::from_config(
        siumai::provider_ext::minimaxi::MinimaxiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_reasoning_budget(1024)
            .with_json_object()
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = make_chat_request_with_model(model);

    let mut siumai_stream = siumai_client
        .chat_stream_request(request.clone())
        .await
        .expect("siumai stream ok");
    let mut provider_stream = provider_client
        .chat_stream_request(request.clone())
        .await
        .expect("provider stream ok");
    let mut config_stream = config_client
        .chat_stream_request(request)
        .await
        .expect("config stream ok");

    let _ = siumai_stream.next().await;
    let _ = provider_stream.next().await;
    let _ = config_stream.next().await;

    let siumai_req = siumai_transport
        .take_stream()
        .expect("siumai stream request");
    let provider_req = provider_transport
        .take_stream()
        .expect("provider stream request");
    let config_req = config_transport
        .take_stream()
        .expect("config stream request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_minimaxi_default_options_stream_request(&siumai_req, base_url, model);
}

#[tokio::test]
async fn minimaxi_siumai_provider_config_stable_stream_request_options_are_equivalent() {
    let siumai_transport = MinimaxiCaptureTransport::default();
    let provider_transport = MinimaxiCaptureTransport::default();
    let config_transport = MinimaxiCaptureTransport::default();

    let model = "MiniMax-M2";
    let base_url = "https://example.com/custom";

    let siumai_client = Siumai::builder()
        .minimaxi()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::minimaxi()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::minimaxi::MinimaxiClient::from_config(
        siumai::provider_ext::minimaxi::MinimaxiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = make_chat_request_with_model(model).with_minimaxi_options(
        MinimaxiOptions::new()
            .with_reasoning_budget(4096)
            .with_json_object(),
    );

    let mut siumai_stream = siumai_client
        .chat_stream_request(request.clone())
        .await
        .expect("siumai stream ok");
    let mut provider_stream = provider_client
        .chat_stream_request(request.clone())
        .await
        .expect("provider stream ok");
    let mut config_stream = config_client
        .chat_stream_request(request)
        .await
        .expect("config stream ok");

    let _ = siumai_stream.next().await;
    let _ = provider_stream.next().await;
    let _ = config_stream.next().await;

    let siumai_req = siumai_transport
        .take_stream()
        .expect("siumai stream request");
    let provider_req = provider_transport
        .take_stream()
        .expect("provider stream request");
    let config_req = config_transport
        .take_stream()
        .expect("config stream request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(siumai_req.body["stream"], serde_json::json!(true));
    assert_eq!(
        siumai_req.body["thinking"],
        serde_json::json!({
            "type": "enabled",
            "budget_tokens": 4096
        })
    );
    assert_eq!(
        siumai_req.body["output_format"],
        serde_json::json!({
            "type": "json_object"
        })
    );
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
}

#[tokio::test]
async fn minimaxi_registry_stable_stream_request_options_match_config_path() {
    let config_transport = MinimaxiCaptureTransport::default();
    let registry_transport = MinimaxiCaptureTransport::default();

    let model = "MiniMax-M2";
    let base_url = "https://example.com/custom";

    let config_client = siumai::provider_ext::minimaxi::MinimaxiClient::from_config(
        siumai::provider_ext::minimaxi::MinimaxiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("minimaxi:MiniMax-M2")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model).with_minimaxi_options(
        MinimaxiOptions::new()
            .with_reasoning_budget(4096)
            .with_json_object(),
    );

    let mut config_stream = config_client
        .chat_stream_request(request.clone())
        .await
        .expect("config stream ok");
    let mut registry_stream = registry_model
        .chat_stream_request(request)
        .await
        .expect("registry stream ok");

    let _ = config_stream.next().await;
    let _ = registry_stream.next().await;

    let config_req = config_transport
        .take_stream()
        .expect("config stream request");
    let registry_req = registry_transport
        .take_stream()
        .expect("registry stream request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(registry_req.body["stream"], serde_json::json!(true));
    assert_eq!(
        registry_req.body["thinking"],
        serde_json::json!({
            "type": "enabled",
            "budget_tokens": 4096
        })
    );
    assert_eq!(
        registry_req.body["output_format"],
        serde_json::json!({
            "type": "json_object"
        })
    );
    assert_eq!(
        header_value(&registry_req, "accept"),
        Some("text/event-stream".to_string())
    );
}

#[tokio::test]
async fn minimaxi_siumai_provider_config_chat_response_metadata_are_equivalent() {
    let model = "MiniMax-M2";
    let base_url = "https://example.com/custom";
    let response_json = serde_json::json!({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [
            {
                "type": "text",
                "text": "hello minimaxi"
            }
        ],
        "container": {
            "id": "container_test",
            "expires_at": "2025-10-20T12:27:25.107823Z"
        },
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 15,
            "output_tokens": 42
        }
    });

    let siumai_transport = JsonSuccessTransport::new(response_json.clone());
    let provider_transport = JsonSuccessTransport::new(response_json.clone());
    let config_transport = JsonSuccessTransport::new(response_json);

    let siumai_client = Siumai::builder()
        .minimaxi()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::minimaxi()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::minimaxi::MinimaxiClient::from_config(
        siumai::provider_ext::minimaxi::MinimaxiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = make_chat_request_with_model(model);

    let siumai_resp = siumai_client
        .chat_request(request.clone())
        .await
        .expect("siumai response ok");
    let provider_resp = provider_client
        .chat_request(request.clone())
        .await
        .expect("provider response ok");
    let config_resp = config_client
        .chat_request(request)
        .await
        .expect("config response ok");

    let siumai_root = siumai_resp
        .provider_metadata
        .as_ref()
        .expect("siumai provider metadata");
    let provider_root = provider_resp
        .provider_metadata
        .as_ref()
        .expect("provider provider metadata");
    let config_root = config_resp
        .provider_metadata
        .as_ref()
        .expect("config provider metadata");

    assert!(siumai_root.get("minimaxi").is_some());
    assert!(provider_root.get("minimaxi").is_some());
    assert!(config_root.get("minimaxi").is_some());
    assert!(siumai_root.get("anthropic").is_none());
    assert!(provider_root.get("anthropic").is_none());
    assert!(config_root.get("anthropic").is_none());

    let siumai_meta = siumai_resp
        .minimaxi_metadata()
        .expect("siumai minimaxi metadata");
    let provider_meta = provider_resp
        .minimaxi_metadata()
        .expect("provider minimaxi metadata");
    let config_meta = config_resp
        .minimaxi_metadata()
        .expect("config minimaxi metadata");

    assert_eq!(siumai_resp.content_text(), Some("hello minimaxi"));
    assert_eq!(provider_resp.content_text(), Some("hello minimaxi"));
    assert_eq!(config_resp.content_text(), Some("hello minimaxi"));
    assert_eq!(siumai_resp.finish_reason, Some(FinishReason::Stop));
    assert_eq!(provider_resp.finish_reason, Some(FinishReason::Stop));
    assert_eq!(config_resp.finish_reason, Some(FinishReason::Stop));
    assert_eq!(
        siumai_meta
            .container
            .as_ref()
            .and_then(|container| container.id.as_deref()),
        Some("container_test")
    );
    assert_eq!(
        provider_meta
            .container
            .as_ref()
            .and_then(|container| container.id.as_deref()),
        Some("container_test")
    );
    assert_eq!(
        config_meta
            .container
            .as_ref()
            .and_then(|container| container.id.as_deref()),
        Some("container_test")
    );

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
}

#[tokio::test]
async fn minimaxi_siumai_provider_config_stream_end_metadata_are_equivalent() {
    let model = "MiniMax-M2";
    let base_url = "https://example.com/custom";
    let stream_body = concat!(
            r#"data: {"type":"message_start","message":{"id":"msg_test","model":"MiniMax-M2","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}}

"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null,"context_management":{"applied_edits":[{"type":"clear_tool_uses_20250919","cleared_tool_uses":5,"cleared_input_tokens":10000}]}},"usage":{"input_tokens":1,"output_tokens":1}}

"#
        )
        .as_bytes()
        .to_vec();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body);

    let siumai_client = Siumai::builder()
        .minimaxi()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::minimaxi()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::minimaxi::MinimaxiClient::from_config(
        siumai::provider_ext::minimaxi::MinimaxiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = make_chat_request_with_model(model);

    let mut siumai_stream = siumai_client
        .chat_stream_request(request.clone())
        .await
        .expect("siumai stream ok");
    let mut provider_stream = provider_client
        .chat_stream_request(request.clone())
        .await
        .expect("provider stream ok");
    let mut config_stream = config_client
        .chat_stream_request(request)
        .await
        .expect("config stream ok");

    use futures_util::StreamExt;

    let mut siumai_end = None;
    let mut provider_end = None;
    let mut config_end = None;

    while let Some(event) = siumai_stream.next().await {
        if let Ok(siumai::prelude::unified::ChatStreamEvent::StreamEnd { response }) = event {
            siumai_end = Some(response);
            break;
        }
    }
    while let Some(event) = provider_stream.next().await {
        if let Ok(siumai::prelude::unified::ChatStreamEvent::StreamEnd { response }) = event {
            provider_end = Some(response);
            break;
        }
    }
    while let Some(event) = config_stream.next().await {
        if let Ok(siumai::prelude::unified::ChatStreamEvent::StreamEnd { response }) = event {
            config_end = Some(response);
            break;
        }
    }

    let siumai_resp = siumai_end.expect("siumai stream end");
    let provider_resp = provider_end.expect("provider stream end");
    let config_resp = config_end.expect("config stream end");

    let siumai_root = siumai_resp
        .provider_metadata
        .as_ref()
        .expect("siumai provider metadata");
    let provider_root = provider_resp
        .provider_metadata
        .as_ref()
        .expect("provider provider metadata");
    let config_root = config_resp
        .provider_metadata
        .as_ref()
        .expect("config provider metadata");

    assert!(siumai_root.get("minimaxi").is_some());
    assert!(provider_root.get("minimaxi").is_some());
    assert!(config_root.get("minimaxi").is_some());
    assert!(siumai_root.get("anthropic").is_none());
    assert!(provider_root.get("anthropic").is_none());
    assert!(config_root.get("anthropic").is_none());

    let siumai_meta = siumai_resp
        .minimaxi_metadata()
        .expect("siumai minimaxi metadata");
    let provider_meta = provider_resp
        .minimaxi_metadata()
        .expect("provider minimaxi metadata");
    let config_meta = config_resp
        .minimaxi_metadata()
        .expect("config minimaxi metadata");

    assert_eq!(siumai_resp.finish_reason, Some(FinishReason::Stop));
    assert_eq!(provider_resp.finish_reason, Some(FinishReason::Stop));
    assert_eq!(config_resp.finish_reason, Some(FinishReason::Stop));
    assert!(siumai_meta.context_management.is_some());
    assert!(provider_meta.context_management.is_some());
    assert!(config_meta.context_management.is_some());

    let siumai_req = siumai_transport
        .take_stream()
        .expect("siumai stream request");
    let provider_req = provider_transport
        .take_stream()
        .expect("provider stream request");
    let config_req = config_transport
        .take_stream()
        .expect("config stream request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
}

#[tokio::test]
async fn minimaxi_registry_chat_response_metadata_match_config_path() {
    let model = "MiniMax-M2";
    let base_url = "https://example.com/custom";
    let response_json = serde_json::json!({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [
            {
                "type": "text",
                "text": "hello minimaxi"
            }
        ],
        "container": {
            "id": "container_test",
            "expires_at": "2025-10-20T12:27:25.107823Z"
        },
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 15,
            "output_tokens": 42
        }
    });

    let config_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let config_client = siumai::provider_ext::minimaxi::MinimaxiClient::from_config(
        siumai::provider_ext::minimaxi::MinimaxiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("minimaxi:MiniMax-M2")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model);

    let config_resp = config_client
        .chat_request(request.clone())
        .await
        .expect("config response ok");
    let registry_resp = registry_model
        .chat_request(request)
        .await
        .expect("registry response ok");

    let config_root = config_resp
        .provider_metadata
        .as_ref()
        .expect("config provider metadata");
    let registry_root = registry_resp
        .provider_metadata
        .as_ref()
        .expect("registry provider metadata");

    assert!(config_root.get("minimaxi").is_some());
    assert!(registry_root.get("minimaxi").is_some());
    assert!(config_root.get("anthropic").is_none());
    assert!(registry_root.get("anthropic").is_none());

    let config_meta = config_resp
        .minimaxi_metadata()
        .expect("config minimaxi metadata");
    let registry_meta = registry_resp
        .minimaxi_metadata()
        .expect("registry minimaxi metadata");

    assert_eq!(config_resp.content_text(), Some("hello minimaxi"));
    assert_eq!(registry_resp.content_text(), Some("hello minimaxi"));
    assert_eq!(config_resp.finish_reason, Some(FinishReason::Stop));
    assert_eq!(registry_resp.finish_reason, Some(FinishReason::Stop));
    assert_eq!(
        config_meta
            .container
            .as_ref()
            .and_then(|container| container.id.as_deref()),
        Some("container_test")
    );
    assert_eq!(
        registry_meta
            .container
            .as_ref()
            .and_then(|container| container.id.as_deref()),
        Some("container_test")
    );

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
}

#[tokio::test]
async fn minimaxi_registry_stream_end_metadata_match_config_path() {
    let model = "MiniMax-M2";
    let base_url = "https://example.com/custom";
    let stream_body = concat!(
            r#"data: {"type":"message_start","message":{"id":"msg_test","model":"MiniMax-M2","type":"message","role":"assistant","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}}

"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null,"context_management":{"applied_edits":[{"type":"clear_tool_uses_20250919","cleared_tool_uses":5,"cleared_input_tokens":10000}]}},"usage":{"input_tokens":1,"output_tokens":1}}

"#
        )
        .as_bytes()
        .to_vec();

    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let config_client = siumai::provider_ext::minimaxi::MinimaxiClient::from_config(
        siumai::provider_ext::minimaxi::MinimaxiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("minimaxi:MiniMax-M2")
        .expect("build registry language model");

    let request = make_chat_request_with_model(model);

    let mut config_stream = config_client
        .chat_stream_request(request.clone())
        .await
        .expect("config stream ok");
    let mut registry_stream = registry_model
        .chat_stream_request(request)
        .await
        .expect("registry stream ok");

    let mut config_end = None;
    let mut registry_end = None;

    while let Some(event) = config_stream.next().await {
        if let Ok(siumai::prelude::unified::ChatStreamEvent::StreamEnd { response }) = event {
            config_end = Some(response);
            break;
        }
    }
    while let Some(event) = registry_stream.next().await {
        if let Ok(siumai::prelude::unified::ChatStreamEvent::StreamEnd { response }) = event {
            registry_end = Some(response);
            break;
        }
    }

    let config_resp = config_end.expect("config stream end");
    let registry_resp = registry_end.expect("registry stream end");

    let config_root = config_resp
        .provider_metadata
        .as_ref()
        .expect("config provider metadata");
    let registry_root = registry_resp
        .provider_metadata
        .as_ref()
        .expect("registry provider metadata");

    assert!(config_root.get("minimaxi").is_some());
    assert!(registry_root.get("minimaxi").is_some());
    assert!(config_root.get("anthropic").is_none());
    assert!(registry_root.get("anthropic").is_none());

    let config_meta = config_resp
        .minimaxi_metadata()
        .expect("config minimaxi metadata");
    let registry_meta = registry_resp
        .minimaxi_metadata()
        .expect("registry minimaxi metadata");

    assert_eq!(config_resp.finish_reason, Some(FinishReason::Stop));
    assert_eq!(registry_resp.finish_reason, Some(FinishReason::Stop));
    assert!(config_meta.context_management.is_some());
    assert!(registry_meta.context_management.is_some());

    let config_req = config_transport
        .take_stream()
        .expect("config stream request");
    let registry_req = registry_transport
        .take_stream()
        .expect("registry stream request");

    assert_requests_equivalent(&config_req, &registry_req);
}

#[tokio::test]
async fn minimaxi_siumai_provider_config_embedding_rerank_are_intentionally_unsupported() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "MiniMax-M2";
    let base_url = "https://example.com/custom";

    let siumai_client = Siumai::builder()
        .minimaxi()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::minimaxi()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::minimaxi::MinimaxiClient::from_config(
        siumai::provider_ext::minimaxi::MinimaxiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let embedding_request = EmbeddingRequest::single("hello minimaxi embedding").with_model(model);
    let rerank_request = make_rerank_request_with_model(model).with_top_n(1);

    let siumai_embedding_err = siumai_client
        .embed_with_config(embedding_request.clone())
        .await
        .expect_err("minimaxi embedding should be unsupported");
    let siumai_rerank_err = siumai_client
        .rerank(rerank_request.clone())
        .await
        .expect_err("minimaxi rerank should be unsupported");

    assert_unsupported_operation(&siumai_embedding_err);
    assert_unsupported_operation(&siumai_rerank_err);

    assert!(siumai_client.as_embedding_capability().is_none());
    assert!(provider_client.as_embedding_capability().is_none());
    assert!(config_client.as_embedding_capability().is_none());
    assert!(siumai_client.as_rerank_capability().is_none());
    assert!(provider_client.as_rerank_capability().is_none());
    assert!(config_client.as_rerank_capability().is_none());
    assert!(siumai_client.as_image_generation_capability().is_some());
    assert!(provider_client.as_image_generation_capability().is_some());
    assert!(config_client.as_image_generation_capability().is_some());
    assert!(siumai_client.as_speech_capability().is_some());
    assert!(provider_client.as_speech_capability().is_some());
    assert!(config_client.as_speech_capability().is_some());
    assert!(siumai_client.as_transcription_capability().is_none());
    assert!(provider_client.as_transcription_capability().is_none());
    assert!(config_client.as_transcription_capability().is_none());

    assert_capture_transports_unused(&[&siumai_transport, &provider_transport, &config_transport]);
}

#[tokio::test]
async fn minimaxi_registry_embedding_rerank_are_intentionally_unsupported() {
    let registry_transport = CaptureTransport::default();
    let base_url = "https://example.com/custom";
    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let embedding_err = match registry.embedding_model("minimaxi:MiniMax-M2") {
        Ok(_) => panic!("build registry embedding model should be unsupported"),
        Err(err) => err,
    };
    let rerank_err = match registry.reranking_model("minimaxi:MiniMax-M2") {
        Ok(_) => panic!("build registry rerank handle should be unsupported"),
        Err(err) => err,
    };

    assert_unsupported_operation(&embedding_err);
    assert_unsupported_operation(&rerank_err);
    assert_capture_transports_unused(&[&registry_transport]);
}

#[tokio::test]
async fn minimaxi_siumai_provider_config_stt_request_is_intentionally_unsupported() {
    let siumai_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "should not be used"
    }));
    let provider_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "should not be used"
    }));
    let config_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "should not be used"
    }));

    let model = "MiniMax-Speech-02";
    let base_url = "https://example.com/custom";

    let siumai_client = Siumai::builder()
        .minimaxi()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::minimaxi()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::minimaxi::MinimaxiClient::from_config(
        siumai::provider_ext::minimaxi::MinimaxiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let mut request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg");
    request.model = Some(model.to_string());
    request = request.with_media_type("audio/mpeg".to_string());

    let siumai_err = siumai_client
        .speech_to_text(request.clone())
        .await
        .expect_err("minimaxi speech-to-text should be unsupported");
    let provider_err = provider_client
        .speech_to_text(request.clone())
        .await
        .expect_err("minimaxi speech-to-text should be unsupported");
    let config_err = config_client
        .speech_to_text(request)
        .await
        .expect_err("minimaxi speech-to-text should be unsupported");

    assert_unsupported_operation(&siumai_err);
    assert_unsupported_operation(&provider_err);
    assert_unsupported_operation(&config_err);
    assert!(siumai_client.as_audio_capability().is_some());
    assert!(provider_client.as_audio_capability().is_some());
    assert!(config_client.as_audio_capability().is_some());
    assert!(siumai_client.as_speech_capability().is_some());
    assert!(provider_client.as_speech_capability().is_some());
    assert!(config_client.as_speech_capability().is_some());
    assert!(siumai_client.as_transcription_capability().is_none());
    assert!(provider_client.as_transcription_capability().is_none());
    assert!(config_client.as_transcription_capability().is_none());
    assert!(siumai_transport.take().is_none());
    assert!(provider_transport.take().is_none());
    assert!(config_transport.take().is_none());
}

#[tokio::test]
async fn minimaxi_registry_stt_request_is_intentionally_unsupported() {
    let registry_transport = MultipartCaptureTransport::new(serde_json::json!({
        "text": "should not be used"
    }));
    let base_url = "https://example.com/custom";
    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_err = match registry.transcription_model("minimaxi:MiniMax-Speech-02") {
        Ok(_) => panic!("minimaxi registry transcription handle should be unsupported"),
        Err(err) => err,
    };

    assert_unsupported_operation(&registry_err);
    assert!(registry_transport.take().is_none());
}

#[tokio::test]
async fn minimaxi_siumai_provider_config_speech_extras_are_intentionally_unavailable() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "speech-2.5-hd";
    let base_url = "https://example.com/custom";

    let siumai_client = Siumai::builder()
        .minimaxi()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::minimaxi()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::minimaxi::MinimaxiClient::from_config(
        siumai::provider_ext::minimaxi::MinimaxiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .speech_model("minimaxi:speech-2.5-hd")
        .expect("build registry speech model");

    assert!(siumai_client.as_speech_capability().is_some());
    assert!(provider_client.as_speech_capability().is_some());
    assert!(config_client.as_speech_capability().is_some());
    assert!(siumai_client.as_speech_extras().is_none());
    assert!(provider_client.as_speech_extras().is_none());
    assert!(config_client.as_speech_extras().is_none());

    assert!(siumai_client.as_transcription_capability().is_none());
    assert!(provider_client.as_transcription_capability().is_none());
    assert!(config_client.as_transcription_capability().is_none());
    assert!(siumai_client.as_transcription_extras().is_none());
    assert!(provider_client.as_transcription_extras().is_none());
    assert!(config_client.as_transcription_extras().is_none());

    let registry_err = match registry_model
        .tts_stream(
            TtsRequest::new("hello from minimaxi".to_string())
                .with_voice("Wise_Woman".to_string())
                .with_format("mp3".to_string()),
        )
        .await
    {
        Ok(_) => panic!("minimaxi registry speech extras should be unavailable"),
        Err(err) => err,
    };

    assert_unsupported_operation(&registry_err);
    assert_capture_transports_unused(&[
        &siumai_transport,
        &provider_transport,
        &config_transport,
        &registry_transport,
    ]);
}

#[tokio::test]
async fn minimaxi_siumai_provider_config_tool_choice_request_are_equivalent() {
    let siumai_transport = MinimaxiCaptureTransport::default();
    let provider_transport = MinimaxiCaptureTransport::default();
    let config_transport = MinimaxiCaptureTransport::default();

    let siumai_client = Siumai::builder()
        .minimaxi()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model("MiniMax-M2")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::minimaxi()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model("MiniMax-M2")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::minimaxi::MinimaxiClient::from_config(
        siumai::provider_ext::minimaxi::MinimaxiConfig::new("test-key")
            .with_base_url("https://example.com/custom")
            .with_model("MiniMax-M2")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = ChatRequest::builder()
        .model("MiniMax-M2")
        .messages(vec![ChatMessage::user("hi").build()])
        .tools(vec![Tool::function(
            "lookup_weather",
            "Look up the weather",
            serde_json::json!({
                "type": "object",
                "properties": { "location": { "type": "string" } },
                "required": ["location"],
                "additionalProperties": false
            }),
        )])
        .tool_choice(ToolChoice::Required)
        .build();

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(siumai_req.url, "https://example.com/custom/v1/messages");
    assert_eq!(
        siumai_req.body["tool_choice"]["type"],
        serde_json::json!("any")
    );
    assert_eq!(
        siumai_req.body["tools"][0]["name"],
        serde_json::json!("lookup_weather")
    );
}

#[tokio::test]
async fn minimaxi_siumai_provider_config_tts_request_are_equivalent() {
    let tts_response = serde_json::json!({
        "data": {
            "audio": "48656c6c6f",
            "status": 2
        },
        "extra_info": {
            "audio_length": 1000,
            "audio_sample_rate": 32000,
            "audio_size": 5,
            "bitrate": 128000,
            "word_count": 1,
            "usage_characters": 5,
            "audio_format": "mp3",
            "audio_channel": 1
        },
        "base_resp": {
            "status_code": 0,
            "status_msg": "success"
        }
    });

    let siumai_transport = MinimaxiJsonSuccessTransport::new(tts_response.clone());
    let provider_transport = MinimaxiJsonSuccessTransport::new(tts_response.clone());
    let config_transport = MinimaxiJsonSuccessTransport::new(tts_response);

    let siumai_client = Siumai::builder()
        .minimaxi()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model("speech-2.6-hd")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::minimaxi()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model("speech-2.6-hd")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::minimaxi::MinimaxiClient::from_config(
        siumai::provider_ext::minimaxi::MinimaxiConfig::new("test-key")
            .with_base_url("https://example.com/custom")
            .with_model("speech-2.6-hd")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = TtsRequest::new("Hello".to_string())
        .with_voice("male-qn-qingse".to_string())
        .with_format("mp3".to_string())
        .with_minimaxi_tts_options(
            MinimaxiTtsOptions::new()
                .with_emotion("happy")
                .with_pitch(5)
                .with_sample_rate(32000)
                .with_bitrate(128000)
                .with_channel(1)
                .with_subtitle_enable(true),
        );

    let siumai_resp = siumai_client
        .text_to_speech(request.clone())
        .await
        .expect("siumai tts ok");
    let provider_resp = provider_client
        .text_to_speech(request.clone())
        .await
        .expect("provider tts ok");
    let config_resp = config_client
        .text_to_speech(request)
        .await
        .expect("config tts ok");

    assert_eq!(siumai_resp.audio_data, b"Hello");
    assert_eq!(provider_resp.audio_data, b"Hello");
    assert_eq!(config_resp.audio_data, b"Hello");

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(siumai_req.url, "https://example.com/custom/v1/t2a_v2");
    assert_eq!(siumai_req.body["model"], serde_json::json!("speech-2.6-hd"));
    assert_eq!(siumai_req.body["text"], serde_json::json!("Hello"));
    assert_eq!(
        siumai_req.body["voice_setting"]["voice_id"],
        serde_json::json!("male-qn-qingse")
    );
    assert_eq!(
        siumai_req.body["voice_setting"]["emotion"],
        serde_json::json!("happy")
    );
    assert_eq!(
        siumai_req.body["voice_setting"]["pitch"],
        serde_json::json!(5)
    );
    assert_eq!(
        siumai_req.body["audio_setting"]["sample_rate"],
        serde_json::json!(32000)
    );
    assert_eq!(
        siumai_req.body["audio_setting"]["bitrate"],
        serde_json::json!(128000)
    );
    assert_eq!(siumai_req.body["subtitle_enable"], serde_json::json!(true));
}

#[tokio::test]
async fn minimaxi_registry_tts_request_options_match_config_path() {
    let tts_response = serde_json::json!({
        "data": {
            "audio": "48656c6c6f",
            "status": 2
        },
        "extra_info": {
            "audio_length": 1000,
            "audio_sample_rate": 32000,
            "audio_size": 5,
            "bitrate": 128000,
            "word_count": 1,
            "usage_characters": 5,
            "audio_format": "mp3",
            "audio_channel": 1
        },
        "base_resp": {
            "status_code": 0,
            "status_msg": "success"
        }
    });

    let config_transport = MinimaxiJsonSuccessTransport::new(tts_response.clone());
    let registry_transport = MinimaxiJsonSuccessTransport::new(tts_response);
    let model = "speech-2.6-hd";
    let base_url = "https://example.com/custom";

    let config_client = siumai::provider_ext::minimaxi::MinimaxiClient::from_config(
        siumai::provider_ext::minimaxi::MinimaxiConfig::new("test-key")
            .with_base_url(base_url)
            .with_model(model)
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .speech_model("minimaxi:speech-2.6-hd")
        .expect("build registry speech model");

    let request = TtsRequest::new("Hello".to_string())
        .with_voice("male-qn-qingse".to_string())
        .with_format("mp3".to_string())
        .with_minimaxi_tts_options(
            MinimaxiTtsOptions::new()
                .with_emotion("happy")
                .with_pitch(5)
                .with_sample_rate(32000)
                .with_bitrate(128000)
                .with_channel(1)
                .with_subtitle_enable(true),
        );

    let config_resp = config_client
        .text_to_speech(request.clone())
        .await
        .expect("config tts ok");
    let registry_resp = siumai::speech::SpeechModel::synthesize(&registry_model, request)
        .await
        .expect("registry tts ok");

    assert_eq!(config_resp.audio_data, b"Hello");
    assert_eq!(registry_resp.audio_data, b"Hello");

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(registry_req.url, "https://example.com/custom/v1/t2a_v2");
    assert_eq!(registry_req.body["model"], serde_json::json!(model));
    assert_eq!(registry_req.body["text"], serde_json::json!("Hello"));
    assert_eq!(
        registry_req.body["voice_setting"]["voice_id"],
        serde_json::json!("male-qn-qingse")
    );
    assert_eq!(
        registry_req.body["voice_setting"]["emotion"],
        serde_json::json!("happy")
    );
    assert_eq!(
        registry_req.body["voice_setting"]["pitch"],
        serde_json::json!(5)
    );
    assert_eq!(
        registry_req.body["audio_setting"]["sample_rate"],
        serde_json::json!(32000)
    );
    assert_eq!(
        registry_req.body["audio_setting"]["bitrate"],
        serde_json::json!(128000)
    );
    assert_eq!(
        registry_req.body["subtitle_enable"],
        serde_json::json!(true)
    );
}

#[tokio::test]
async fn minimaxi_siumai_provider_config_registry_image_request_are_equivalent() {
    let image_response = serde_json::json!({
        "data": {
            "image_urls": [
                "https://example.com/generated.png"
            ]
        },
        "base_resp": {
            "status_code": 0,
            "status_msg": "success"
        }
    });

    let siumai_transport = MinimaxiJsonSuccessTransport::new(image_response.clone());
    let provider_transport = MinimaxiJsonSuccessTransport::new(image_response.clone());
    let config_transport = MinimaxiJsonSuccessTransport::new(image_response.clone());
    let registry_transport = MinimaxiJsonSuccessTransport::new(image_response);

    let siumai_client = Siumai::builder()
        .minimaxi()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model("image-01")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::minimaxi()
        .api_key("test-key")
        .base_url("https://example.com/custom")
        .model("image-01")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::minimaxi::MinimaxiClient::from_config(
        siumai::provider_ext::minimaxi::MinimaxiConfig::new("test-key")
            .with_base_url("https://example.com/custom")
            .with_model("image-01")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(
        Arc::new(registry_transport.clone()),
        "https://example.com/custom",
    );
    let registry_client = registry
        .image_model("minimaxi:image-01")
        .expect("build registry image model");

    let request = ImageGenerationRequest {
        prompt: "a tiny green robot".to_string(),
        negative_prompt: Some("blurry".to_string()),
        size: Some("1024x1024".to_string()),
        aspect_ratio: None,
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

    let siumai_resp = siumai_client
        .generate_images(request.clone())
        .await
        .expect("siumai image ok");
    let provider_resp = provider_client
        .generate_images(request.clone())
        .await
        .expect("provider image ok");
    let config_resp = config_client
        .generate_images(request.clone())
        .await
        .expect("config image ok");
    let registry_resp = registry_client
        .generate_images(request)
        .await
        .expect("registry image ok");

    assert_eq!(
        siumai_resp.images[0].url.as_deref(),
        Some("https://example.com/generated.png")
    );
    assert_eq!(
        provider_resp.images[0].url.as_deref(),
        Some("https://example.com/generated.png")
    );
    assert_eq!(
        config_resp.images[0].url.as_deref(),
        Some("https://example.com/generated.png")
    );
    assert_eq!(
        registry_resp.images[0].url.as_deref(),
        Some("https://example.com/generated.png")
    );

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(
        siumai_req.url,
        "https://example.com/custom/v1/image_generation"
    );
    assert_eq!(siumai_req.body["model"], serde_json::json!("image-01"));
    assert_eq!(
        siumai_req.body["prompt"],
        serde_json::json!("a tiny green robot")
    );
    assert_eq!(siumai_req.body["size"], serde_json::json!("1024x1024"));
    assert_eq!(siumai_req.body["n"], serde_json::json!(1));
    assert_eq!(siumai_req.body["response_format"], serde_json::json!("url"));
}

#[tokio::test]
async fn minimaxi_siumai_provider_config_registry_create_video_task_are_equivalent() {
    let video_response = serde_json::json!({
        "task_id": "task-123",
        "base_resp": {
            "status_code": 0,
            "status_msg": "success"
        }
    });

    let siumai_transport = MinimaxiJsonSuccessTransport::new(video_response.clone());
    let provider_transport = MinimaxiJsonSuccessTransport::new(video_response.clone());
    let config_transport = MinimaxiJsonSuccessTransport::new(video_response.clone());
    let registry_transport = MinimaxiJsonSuccessTransport::new(video_response);

    let siumai_client = Siumai::builder()
        .minimaxi()
        .api_key("test-key")
        .base_url("https://example.com/anthropic/v1")
        .model("hailuo-2.3")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::minimaxi()
        .api_key("test-key")
        .base_url("https://example.com/anthropic/v1")
        .model("hailuo-2.3")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::minimaxi::MinimaxiClient::from_config(
        siumai::provider_ext::minimaxi::MinimaxiConfig::new("test-key")
            .with_base_url("https://example.com/anthropic/v1")
            .with_model("hailuo-2.3")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(
        Arc::new(registry_transport.clone()),
        "https://example.com/anthropic/v1",
    );
    let registry_client = registry
        .video_model("minimaxi:hailuo-2.3")
        .expect("build registry video model");

    let request = siumai::prelude::extensions::types::VideoGenerationRequest::new(
        "hailuo-2.3",
        "a tiny robot walking in rain",
    )
    .with_duration(10)
    .with_resolution("1080P")
    .with_minimaxi_video_options(
        MinimaxiVideoOptions::new()
            .with_prompt_optimizer(true)
            .with_fast_pretreatment(false)
            .with_callback_url("https://example.com/callback")
            .with_aigc_watermark(false),
    );

    let siumai_resp = siumai_client
        .create_video_task(request.clone())
        .await
        .expect("siumai create video ok");
    let provider_resp = provider_client
        .create_video_task(request.clone())
        .await
        .expect("provider create video ok");
    let config_resp = config_client
        .create_video_task(request.clone())
        .await
        .expect("config create video ok");
    let registry_resp = registry_client
        .create_video_task(request)
        .await
        .expect("registry create video ok");

    assert_eq!(siumai_resp.task_id, "task-123");
    assert_eq!(provider_resp.task_id, "task-123");
    assert_eq!(config_resp.task_id, "task-123");
    assert_eq!(registry_resp.task_id, "task-123");

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(siumai_req.url, "https://example.com/v1/video_generation");
    assert_eq!(siumai_req.body["model"], serde_json::json!("hailuo-2.3"));
    assert_eq!(
        siumai_req.body["prompt"],
        serde_json::json!("a tiny robot walking in rain")
    );
    assert_eq!(siumai_req.body["duration"], serde_json::json!(10));
    assert_eq!(siumai_req.body["resolution"], serde_json::json!("1080P"));
    assert_eq!(siumai_req.body["prompt_optimizer"], serde_json::json!(true));
    assert_eq!(
        siumai_req.body["fast_pretreatment"],
        serde_json::json!(false)
    );
    assert_eq!(
        siumai_req.body["callback_url"],
        serde_json::json!("https://example.com/callback")
    );
    assert_eq!(siumai_req.body["aigc_watermark"], serde_json::json!(false));
}

#[tokio::test]
async fn minimaxi_siumai_provider_config_registry_query_video_task_are_equivalent() {
    async fn mount_video_query_server() -> MockServer {
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

    let siumai_server = mount_video_query_server().await;
    let provider_server = mount_video_query_server().await;
    let config_server = mount_video_query_server().await;
    let registry_server = mount_video_query_server().await;

    let siumai_client = Siumai::builder()
        .minimaxi()
        .api_key("test-key")
        .base_url(format!("{}/anthropic/v1", siumai_server.uri()))
        .model("hailuo-2.3")
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::minimaxi()
        .api_key("test-key")
        .base_url(format!("{}/anthropic/v1", provider_server.uri()))
        .model("hailuo-2.3")
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::minimaxi::MinimaxiClient::from_config(
        siumai::provider_ext::minimaxi::MinimaxiConfig::new("test-key")
            .with_base_url(format!("{}/anthropic/v1", config_server.uri()))
            .with_model("hailuo-2.3"),
    )
    .expect("build config client");

    let registry = minimaxi_registry_builder()
        .with_provider_api_key_base_url(
            "minimaxi",
            "test-key",
            minimaxi_wiremock_base_url(&registry_server),
        )
        .build()
        .expect("build registry");
    let registry_client = registry
        .video_model("minimaxi:hailuo-2.3")
        .expect("build registry video model");

    let siumai_resp = siumai_client
        .query_video_task("task-123")
        .await
        .expect("siumai query video ok");
    let provider_resp = provider_client
        .query_video_task("task-123")
        .await
        .expect("provider query video ok");
    let config_resp = config_client
        .query_video_task("task-123")
        .await
        .expect("config query video ok");
    let registry_resp = registry_client
        .query_video_task("task-123")
        .await
        .expect("registry query video ok");

    assert_eq!(siumai_resp.task_id, "task-123");
    assert_eq!(provider_resp.task_id, "task-123");
    assert_eq!(config_resp.task_id, "task-123");
    assert_eq!(registry_resp.task_id, "task-123");
    assert_eq!(siumai_resp.file_id.as_deref(), Some("file-123"));
    assert_eq!(provider_resp.file_id.as_deref(), Some("file-123"));
    assert_eq!(config_resp.file_id.as_deref(), Some("file-123"));
    assert_eq!(registry_resp.file_id.as_deref(), Some("file-123"));

    let siumai_req = siumai_server
        .received_requests()
        .await
        .expect("recorded siumai requests")
        .into_iter()
        .next()
        .expect("siumai query request");
    let provider_req = provider_server
        .received_requests()
        .await
        .expect("recorded provider requests")
        .into_iter()
        .next()
        .expect("provider query request");
    let config_req = config_server
        .received_requests()
        .await
        .expect("recorded config requests")
        .into_iter()
        .next()
        .expect("config query request");
    let registry_req = registry_server
        .received_requests()
        .await
        .expect("recorded registry requests")
        .into_iter()
        .next()
        .expect("registry query request");

    assert_eq!(siumai_req.url.path(), "/v1/query/video_generation");
    assert_eq!(siumai_req.url.query(), Some("task_id=task-123"));
    assert_eq!(provider_req.url.path(), siumai_req.url.path());
    assert_eq!(provider_req.url.query(), siumai_req.url.query());
    assert_eq!(config_req.url.path(), siumai_req.url.path());
    assert_eq!(config_req.url.query(), siumai_req.url.query());
    assert_eq!(registry_req.url.path(), siumai_req.url.path());
    assert_eq!(registry_req.url.query(), siumai_req.url.query());
    assert_eq!(
        wiremock_header_value(&siumai_req, "authorization"),
        Some("Bearer test-key".to_string())
    );
    assert_eq!(
        wiremock_header_value(&siumai_req, "authorization"),
        wiremock_header_value(&provider_req, "authorization")
    );
    assert_eq!(
        wiremock_header_value(&siumai_req, "authorization"),
        wiremock_header_value(&config_req, "authorization")
    );
    assert_eq!(
        wiremock_header_value(&siumai_req, "authorization"),
        wiremock_header_value(&registry_req, "authorization")
    );
}

#[tokio::test]
async fn minimaxi_siumai_provider_config_registry_generate_music_are_equivalent() {
    let music_response = serde_json::json!({
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
    });

    let siumai_transport = MinimaxiJsonSuccessTransport::new(music_response.clone());
    let provider_transport = MinimaxiJsonSuccessTransport::new(music_response.clone());
    let config_transport = MinimaxiJsonSuccessTransport::new(music_response.clone());
    let registry_transport = MinimaxiJsonSuccessTransport::new(music_response);

    let siumai_client = Siumai::builder()
        .minimaxi()
        .api_key("test-key")
        .base_url("https://example.com/anthropic/v1")
        .model("music-2.0")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::minimaxi()
        .api_key("test-key")
        .base_url("https://example.com/anthropic/v1")
        .model("music-2.0")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::minimaxi::MinimaxiClient::from_config(
        siumai::provider_ext::minimaxi::MinimaxiConfig::new("test-key")
            .with_base_url("https://example.com/anthropic/v1")
            .with_model("music-2.0")
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(
        Arc::new(registry_transport.clone()),
        "https://example.com/anthropic/v1",
    );
    let registry_client = registry
        .language_model("minimaxi:music-2.0")
        .expect("build registry language model");

    let request = MinimaxiMusicRequestBuilder::new("cinematic ambient with piano")
        .lyrics_template()
        .sample_rate(48000)
        .bitrate(320000)
        .format("wav")
        .build();

    let siumai_resp = siumai_client
        .generate_music(request.clone())
        .await
        .expect("siumai music ok");
    let provider_resp = provider_client
        .generate_music(request.clone())
        .await
        .expect("provider music ok");
    let config_resp = config_client
        .generate_music(request.clone())
        .await
        .expect("config music ok");
    let registry_resp = registry_client
        .generate_music(request)
        .await
        .expect("registry music ok");

    assert_eq!(siumai_resp.audio_data, b"Hello");
    assert_eq!(provider_resp.audio_data, b"Hello");
    assert_eq!(config_resp.audio_data, b"Hello");
    assert_eq!(registry_resp.audio_data, b"Hello");

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(siumai_req.url, "https://example.com/v1/music_generation");
    assert_eq!(siumai_req.body["model"], serde_json::json!("music-2.0"));
    assert_eq!(
        siumai_req.body["prompt"],
        serde_json::json!("cinematic ambient with piano")
    );
    assert_eq!(
        siumai_req.body["lyrics"],
        serde_json::json!("[Intro]\n[Main]\n[Outro]")
    );
    assert_eq!(
        siumai_req.body["audio_setting"]["sample_rate"],
        serde_json::json!(48000)
    );
    assert_eq!(
        siumai_req.body["audio_setting"]["bitrate"],
        serde_json::json!(320000)
    );
    assert_eq!(
        siumai_req.body["audio_setting"]["format"],
        serde_json::json!("wav")
    );
}

#[tokio::test]
async fn minimaxi_siumai_provider_config_registry_upload_file_are_equivalent() {
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

    let siumai_server = mount_upload_server().await;
    let provider_server = mount_upload_server().await;
    let config_server = mount_upload_server().await;
    let registry_server = mount_upload_server().await;

    let siumai_client = Siumai::builder()
        .minimaxi()
        .api_key("test-key")
        .base_url(format!("{}/anthropic/v1", siumai_server.uri()))
        .model("MiniMax-M2")
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::minimaxi()
        .api_key("test-key")
        .base_url(format!("{}/anthropic/v1", provider_server.uri()))
        .model("MiniMax-M2")
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::minimaxi::MinimaxiClient::from_config(
        siumai::provider_ext::minimaxi::MinimaxiConfig::new("test-key")
            .with_base_url(format!("{}/anthropic/v1", config_server.uri()))
            .with_model("MiniMax-M2"),
    )
    .expect("build config client");
    let registry =
        make_registry_without_transport(&format!("{}/anthropic/v1", registry_server.uri()));
    let registry_client = registry
        .language_model("minimaxi:MiniMax-M2")
        .expect("build registry language model");

    let request = FileUploadRequest {
        content: b"hello".to_vec(),
        filename: Some("hello.txt".to_string()),
        mime_type: Some("text/plain".to_string()),
        purpose: "t2a_async_input".to_string(),
        metadata: HashMap::new(),
        provider_options: Default::default(),
        http_config: None,
    };

    let siumai_file = siumai_client
        .upload_file(request.clone())
        .await
        .expect("siumai upload ok");
    let provider_file = provider_client
        .upload_file(request.clone())
        .await
        .expect("provider upload ok");
    let config_file = config_client
        .upload_file(request.clone())
        .await
        .expect("config upload ok");
    let registry_file = registry_client
        .upload_file(request)
        .await
        .expect("registry upload ok");

    assert_eq!(siumai_file.id, "123");
    assert_eq!(provider_file.id, "123");
    assert_eq!(config_file.id, "123");
    assert_eq!(registry_file.id, "123");

    let siumai_req = siumai_server
        .received_requests()
        .await
        .expect("recorded siumai requests")
        .into_iter()
        .next()
        .expect("siumai upload request");
    let provider_req = provider_server
        .received_requests()
        .await
        .expect("recorded provider requests")
        .into_iter()
        .next()
        .expect("provider upload request");
    let config_req = config_server
        .received_requests()
        .await
        .expect("recorded config requests")
        .into_iter()
        .next()
        .expect("config upload request");
    let registry_req = registry_server
        .received_requests()
        .await
        .expect("recorded registry requests")
        .into_iter()
        .next()
        .expect("registry upload request");

    assert_eq!(siumai_req.url.path(), "/v1/files/upload");
    assert_eq!(provider_req.url.path(), "/v1/files/upload");
    assert_eq!(config_req.url.path(), "/v1/files/upload");
    assert_eq!(registry_req.url.path(), "/v1/files/upload");
    assert_eq!(
        wiremock_header_value(&siumai_req, "authorization"),
        Some("Bearer test-key".to_string())
    );
    assert_eq!(
        wiremock_header_value(&siumai_req, "authorization"),
        wiremock_header_value(&provider_req, "authorization")
    );
    assert_eq!(
        wiremock_header_value(&siumai_req, "authorization"),
        wiremock_header_value(&config_req, "authorization")
    );
    assert_eq!(
        wiremock_header_value(&siumai_req, "authorization"),
        wiremock_header_value(&registry_req, "authorization")
    );
    assert_eq!(
        normalize_wiremock_multipart_body(&siumai_req),
        normalize_wiremock_multipart_body(&provider_req)
    );
    assert_eq!(
        normalize_wiremock_multipart_body(&siumai_req),
        normalize_wiremock_multipart_body(&config_req)
    );
    assert_eq!(
        normalize_wiremock_multipart_body(&siumai_req),
        normalize_wiremock_multipart_body(&registry_req)
    );
}

#[tokio::test]
async fn minimaxi_siumai_provider_config_registry_list_files_are_equivalent() {
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

    let siumai_server = mount_list_server().await;
    let provider_server = mount_list_server().await;
    let config_server = mount_list_server().await;
    let registry_server = mount_list_server().await;

    let siumai_client = Siumai::builder()
        .minimaxi()
        .api_key("test-key")
        .base_url(format!("{}/anthropic/v1", siumai_server.uri()))
        .model("MiniMax-M2")
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::minimaxi()
        .api_key("test-key")
        .base_url(format!("{}/anthropic/v1", provider_server.uri()))
        .model("MiniMax-M2")
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::minimaxi::MinimaxiClient::from_config(
        siumai::provider_ext::minimaxi::MinimaxiConfig::new("test-key")
            .with_base_url(format!("{}/anthropic/v1", config_server.uri()))
            .with_model("MiniMax-M2"),
    )
    .expect("build config client");
    let registry =
        make_registry_without_transport(&format!("{}/anthropic/v1", registry_server.uri()));
    let registry_client = registry
        .language_model("minimaxi:MiniMax-M2")
        .expect("build registry language model");

    let query = FileListQuery {
        purpose: Some("t2a_async_input".to_string()),
        limit: None,
        after: None,
        order: None,
        http_config: None,
    };

    let siumai_list = siumai_client
        .list_files(Some(query.clone()))
        .await
        .expect("siumai list ok");
    let provider_list = provider_client
        .list_files(Some(query.clone()))
        .await
        .expect("provider list ok");
    let config_list = config_client
        .list_files(Some(query.clone()))
        .await
        .expect("config list ok");
    let registry_list = registry_client
        .list_files(Some(query))
        .await
        .expect("registry list ok");

    assert_eq!(siumai_list.files.len(), 1);
    assert_eq!(provider_list.files.len(), 1);
    assert_eq!(config_list.files.len(), 1);
    assert_eq!(registry_list.files.len(), 1);

    let siumai_req = siumai_server
        .received_requests()
        .await
        .expect("recorded siumai requests")
        .into_iter()
        .next()
        .expect("siumai list request");
    let provider_req = provider_server
        .received_requests()
        .await
        .expect("recorded provider requests")
        .into_iter()
        .next()
        .expect("provider list request");
    let config_req = config_server
        .received_requests()
        .await
        .expect("recorded config requests")
        .into_iter()
        .next()
        .expect("config list request");
    let registry_req = registry_server
        .received_requests()
        .await
        .expect("recorded registry requests")
        .into_iter()
        .next()
        .expect("registry list request");

    assert_eq!(siumai_req.url.path(), "/v1/files/list");
    assert_eq!(siumai_req.url.query(), Some("purpose=t2a_async_input"));
    assert_eq!(provider_req.url.path(), siumai_req.url.path());
    assert_eq!(provider_req.url.query(), siumai_req.url.query());
    assert_eq!(config_req.url.path(), siumai_req.url.path());
    assert_eq!(config_req.url.query(), siumai_req.url.query());
    assert_eq!(registry_req.url.path(), siumai_req.url.path());
    assert_eq!(registry_req.url.query(), siumai_req.url.query());
    assert_eq!(
        wiremock_header_value(&siumai_req, "authorization"),
        Some("Bearer test-key".to_string())
    );
    assert_eq!(
        wiremock_header_value(&siumai_req, "authorization"),
        wiremock_header_value(&registry_req, "authorization")
    );
}

#[tokio::test]
async fn minimaxi_siumai_provider_config_registry_retrieve_file_are_equivalent() {
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

    let siumai_server = mount_retrieve_server().await;
    let provider_server = mount_retrieve_server().await;
    let config_server = mount_retrieve_server().await;
    let registry_server = mount_retrieve_server().await;

    let siumai_client = Siumai::builder()
        .minimaxi()
        .api_key("test-key")
        .base_url(format!("{}/anthropic/v1", siumai_server.uri()))
        .model("MiniMax-M2")
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::minimaxi()
        .api_key("test-key")
        .base_url(format!("{}/anthropic/v1", provider_server.uri()))
        .model("MiniMax-M2")
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::minimaxi::MinimaxiClient::from_config(
        siumai::provider_ext::minimaxi::MinimaxiConfig::new("test-key")
            .with_base_url(format!("{}/anthropic/v1", config_server.uri()))
            .with_model("MiniMax-M2"),
    )
    .expect("build config client");
    let registry =
        make_registry_without_transport(&format!("{}/anthropic/v1", registry_server.uri()));
    let registry_client = registry
        .language_model("minimaxi:MiniMax-M2")
        .expect("build registry language model");

    let siumai_file = siumai_client
        .retrieve_file("123".to_string())
        .await
        .expect("siumai retrieve ok");
    let provider_file = provider_client
        .retrieve_file("123".to_string())
        .await
        .expect("provider retrieve ok");
    let config_file = config_client
        .retrieve_file("123".to_string())
        .await
        .expect("config retrieve ok");
    let registry_file = registry_client
        .retrieve_file("123".to_string())
        .await
        .expect("registry retrieve ok");

    assert_eq!(siumai_file.id, "123");
    assert_eq!(provider_file.id, "123");
    assert_eq!(config_file.id, "123");
    assert_eq!(registry_file.id, "123");

    let siumai_req = siumai_server
        .received_requests()
        .await
        .expect("recorded siumai requests")
        .into_iter()
        .next()
        .expect("siumai retrieve request");
    let provider_req = provider_server
        .received_requests()
        .await
        .expect("recorded provider requests")
        .into_iter()
        .next()
        .expect("provider retrieve request");
    let config_req = config_server
        .received_requests()
        .await
        .expect("recorded config requests")
        .into_iter()
        .next()
        .expect("config retrieve request");
    let registry_req = registry_server
        .received_requests()
        .await
        .expect("recorded registry requests")
        .into_iter()
        .next()
        .expect("registry retrieve request");

    assert_eq!(siumai_req.url.path(), "/v1/files/retrieve");
    assert_eq!(siumai_req.url.query(), Some("file_id=123"));
    assert_eq!(provider_req.url.path(), siumai_req.url.path());
    assert_eq!(provider_req.url.query(), siumai_req.url.query());
    assert_eq!(config_req.url.path(), siumai_req.url.path());
    assert_eq!(config_req.url.query(), siumai_req.url.query());
    assert_eq!(registry_req.url.path(), siumai_req.url.path());
    assert_eq!(registry_req.url.query(), siumai_req.url.query());
    assert_eq!(
        wiremock_header_value(&siumai_req, "authorization"),
        Some("Bearer test-key".to_string())
    );
    assert_eq!(
        wiremock_header_value(&siumai_req, "authorization"),
        wiremock_header_value(&registry_req, "authorization")
    );
}

#[tokio::test]
async fn minimaxi_siumai_provider_config_registry_get_file_content_are_equivalent() {
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

    let siumai_server = mount_content_server().await;
    let provider_server = mount_content_server().await;
    let config_server = mount_content_server().await;
    let registry_server = mount_content_server().await;

    let siumai_client = Siumai::builder()
        .minimaxi()
        .api_key("test-key")
        .base_url(format!("{}/anthropic/v1", siumai_server.uri()))
        .model("MiniMax-M2")
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::minimaxi()
        .api_key("test-key")
        .base_url(format!("{}/anthropic/v1", provider_server.uri()))
        .model("MiniMax-M2")
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::minimaxi::MinimaxiClient::from_config(
        siumai::provider_ext::minimaxi::MinimaxiConfig::new("test-key")
            .with_base_url(format!("{}/anthropic/v1", config_server.uri()))
            .with_model("MiniMax-M2"),
    )
    .expect("build config client");
    let registry =
        make_registry_without_transport(&format!("{}/anthropic/v1", registry_server.uri()));
    let registry_client = registry
        .language_model("minimaxi:MiniMax-M2")
        .expect("build registry language model");

    let siumai_bytes = siumai_client
        .get_file_content("123".to_string())
        .await
        .expect("siumai content ok");
    let provider_bytes = provider_client
        .get_file_content("123".to_string())
        .await
        .expect("provider content ok");
    let config_bytes = config_client
        .get_file_content("123".to_string())
        .await
        .expect("config content ok");
    let registry_bytes = registry_client
        .get_file_content("123".to_string())
        .await
        .expect("registry content ok");

    assert_eq!(siumai_bytes, b"hello");
    assert_eq!(provider_bytes, b"hello");
    assert_eq!(config_bytes, b"hello");
    assert_eq!(registry_bytes, b"hello");

    let siumai_req = siumai_server
        .received_requests()
        .await
        .expect("recorded siumai requests")
        .into_iter()
        .next()
        .expect("siumai content request");
    let provider_req = provider_server
        .received_requests()
        .await
        .expect("recorded provider requests")
        .into_iter()
        .next()
        .expect("provider content request");
    let config_req = config_server
        .received_requests()
        .await
        .expect("recorded config requests")
        .into_iter()
        .next()
        .expect("config content request");
    let registry_req = registry_server
        .received_requests()
        .await
        .expect("recorded registry requests")
        .into_iter()
        .next()
        .expect("registry content request");

    assert_eq!(siumai_req.url.path(), "/v1/files/retrieve_content");
    assert_eq!(siumai_req.url.query(), Some("file_id=123"));
    assert_eq!(provider_req.url.path(), siumai_req.url.path());
    assert_eq!(provider_req.url.query(), siumai_req.url.query());
    assert_eq!(config_req.url.path(), siumai_req.url.path());
    assert_eq!(config_req.url.query(), siumai_req.url.query());
    assert_eq!(registry_req.url.path(), siumai_req.url.path());
    assert_eq!(registry_req.url.query(), siumai_req.url.query());
    assert_eq!(
        wiremock_header_value(&siumai_req, "authorization"),
        Some("Bearer test-key".to_string())
    );
    assert_eq!(
        wiremock_header_value(&siumai_req, "authorization"),
        wiremock_header_value(&registry_req, "authorization")
    );
}

#[tokio::test]
async fn minimaxi_siumai_provider_config_registry_delete_file_are_equivalent() {
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

    let siumai_server = mount_delete_server().await;
    let provider_server = mount_delete_server().await;
    let config_server = mount_delete_server().await;
    let registry_server = mount_delete_server().await;

    let siumai_client = Siumai::builder()
        .minimaxi()
        .api_key("test-key")
        .base_url(format!("{}/anthropic/v1", siumai_server.uri()))
        .model("MiniMax-M2")
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::minimaxi()
        .api_key("test-key")
        .base_url(format!("{}/anthropic/v1", provider_server.uri()))
        .model("MiniMax-M2")
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::minimaxi::MinimaxiClient::from_config(
        siumai::provider_ext::minimaxi::MinimaxiConfig::new("test-key")
            .with_base_url(format!("{}/anthropic/v1", config_server.uri()))
            .with_model("MiniMax-M2"),
    )
    .expect("build config client");
    let registry =
        make_registry_without_transport(&format!("{}/anthropic/v1", registry_server.uri()));
    let registry_client = registry
        .language_model("minimaxi:MiniMax-M2")
        .expect("build registry language model");

    let siumai_deleted = siumai_client
        .delete_file("123:t2a_async_input".to_string())
        .await
        .expect("siumai delete ok");
    let provider_deleted = provider_client
        .delete_file("123:t2a_async_input".to_string())
        .await
        .expect("provider delete ok");
    let config_deleted = config_client
        .delete_file("123:t2a_async_input".to_string())
        .await
        .expect("config delete ok");
    let registry_deleted = registry_client
        .delete_file("123:t2a_async_input".to_string())
        .await
        .expect("registry delete ok");

    assert!(siumai_deleted.deleted);
    assert!(provider_deleted.deleted);
    assert!(config_deleted.deleted);
    assert!(registry_deleted.deleted);

    let siumai_req = siumai_server
        .received_requests()
        .await
        .expect("recorded siumai requests")
        .into_iter()
        .next()
        .expect("siumai delete request");
    let provider_req = provider_server
        .received_requests()
        .await
        .expect("recorded provider requests")
        .into_iter()
        .next()
        .expect("provider delete request");
    let config_req = config_server
        .received_requests()
        .await
        .expect("recorded config requests")
        .into_iter()
        .next()
        .expect("config delete request");
    let registry_req = registry_server
        .received_requests()
        .await
        .expect("recorded registry requests")
        .into_iter()
        .next()
        .expect("registry delete request");

    let siumai_body: serde_json::Value =
        serde_json::from_slice(&siumai_req.body).expect("siumai delete body");
    let provider_body: serde_json::Value =
        serde_json::from_slice(&provider_req.body).expect("provider delete body");
    let config_body: serde_json::Value =
        serde_json::from_slice(&config_req.body).expect("config delete body");
    let registry_body: serde_json::Value =
        serde_json::from_slice(&registry_req.body).expect("registry delete body");

    assert_eq!(siumai_req.url.path(), "/v1/files/delete");
    assert_eq!(provider_req.url.path(), siumai_req.url.path());
    assert_eq!(config_req.url.path(), siumai_req.url.path());
    assert_eq!(registry_req.url.path(), siumai_req.url.path());
    assert_eq!(siumai_body, provider_body);
    assert_eq!(siumai_body, config_body);
    assert_eq!(siumai_body, registry_body);
    assert_eq!(
        siumai_body,
        serde_json::json!({
            "file_id": 123,
            "purpose": "t2a_async_input"
        })
    );
}

#[tokio::test]
async fn minimaxi_registry_file_handle_prefers_provider_specific_build_overrides() {
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

    let registry = make_minimaxi_server_override_registry(&global_server, &minimaxi_server);

    let handle = registry
        .language_model("minimaxi:MiniMax-M2")
        .expect("build registry language model");

    let uploaded = handle
        .upload_file(FileUploadRequest {
            content: b"hello".to_vec(),
            filename: Some("hello.txt".to_string()),
            mime_type: Some("text/plain".to_string()),
            purpose: "t2a_async_input".to_string(),
            metadata: HashMap::new(),
            provider_options: Default::default(),
            http_config: None,
        })
        .await
        .expect("upload file through registry handle");

    assert_eq!(uploaded.id, "123");

    let global_requests = global_server
        .received_requests()
        .await
        .expect("global requests");
    assert!(
        global_requests.is_empty(),
        "global override lane should stay unused: {global_requests:?}"
    );

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

    let registry = make_minimaxi_server_override_registry(&global_server, &minimaxi_server);

    let handle = registry
        .language_model("minimaxi:MiniMax-M2")
        .expect("build registry language model");

    let listed = handle
        .list_files(Some(FileListQuery {
            purpose: Some("t2a_async_input".to_string()),
            limit: None,
            after: None,
            order: None,
            http_config: None,
        }))
        .await
        .expect("list files through registry handle");

    assert_eq!(listed.files.len(), 1);

    let global_requests = global_server
        .received_requests()
        .await
        .expect("global requests");
    assert!(
        global_requests.is_empty(),
        "global override lane should stay unused: {global_requests:?}"
    );

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

    let registry = make_minimaxi_server_override_registry(&global_server, &minimaxi_server);

    let handle = registry
        .language_model("minimaxi:MiniMax-M2")
        .expect("build registry language model");

    let file = handle
        .retrieve_file("123".to_string())
        .await
        .expect("retrieve file through registry handle");

    assert_eq!(file.id, "123");

    let global_requests = global_server
        .received_requests()
        .await
        .expect("global requests");
    assert!(
        global_requests.is_empty(),
        "global override lane should stay unused: {global_requests:?}"
    );

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

    let registry = make_minimaxi_server_override_registry(&global_server, &minimaxi_server);

    let handle = registry
        .language_model("minimaxi:MiniMax-M2")
        .expect("build registry language model");

    let content = handle
        .get_file_content("123".to_string())
        .await
        .expect("get file content through registry handle");

    assert_eq!(content, b"hello");

    let global_requests = global_server
        .received_requests()
        .await
        .expect("global requests");
    assert!(
        global_requests.is_empty(),
        "global override lane should stay unused: {global_requests:?}"
    );

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

    let registry = make_minimaxi_server_override_registry(&global_server, &minimaxi_server);

    let handle = registry
        .language_model("minimaxi:MiniMax-M2")
        .expect("build registry language model");

    let deleted = handle
        .delete_file("123:t2a_async_input".to_string())
        .await
        .expect("delete file through registry handle");

    assert!(deleted.deleted);

    let global_requests = global_server
        .received_requests()
        .await
        .expect("global requests");
    assert!(
        global_requests.is_empty(),
        "global override lane should stay unused: {global_requests:?}"
    );

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

    let registry = make_minimaxi_server_override_registry(&global_server, &minimaxi_server);

    let handle = registry
        .video_model("minimaxi:hailuo-2.3")
        .expect("build registry video model");

    let created = handle
        .create_video_task(
            MinimaxiVideoRequestBuilder::new("hailuo-2.3", "tiny robot in rain")
                .duration(10)
                .resolution("1080P")
                .prompt_optimizer(true)
                .fast_pretreatment(false)
                .callback_url("https://example.com/callback")
                .watermark(false)
                .build(),
        )
        .await
        .expect("create video task through registry handle");

    assert_eq!(created.task_id, "task-123");

    let global_requests = global_server
        .received_requests()
        .await
        .expect("global requests");
    assert!(
        global_requests.is_empty(),
        "global override lane should stay unused: {global_requests:?}"
    );

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

    let registry = make_minimaxi_server_override_registry(&global_server, &minimaxi_server);

    let handle = registry
        .language_model("minimaxi:music-2.0")
        .expect("build registry language model");

    let generated = handle
        .generate_music(
            MinimaxiMusicRequestBuilder::new("cinematic ambient with piano")
                .lyrics_template()
                .sample_rate(48000)
                .bitrate(320000)
                .format("wav")
                .build(),
        )
        .await
        .expect("generate music through registry handle");

    assert_eq!(generated.audio_data, b"Hello");

    let global_requests = global_server
        .received_requests()
        .await
        .expect("global requests");
    assert!(
        global_requests.is_empty(),
        "global override lane should stay unused: {global_requests:?}"
    );

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
    let image_response = serde_json::json!({
        "data": {
            "image_urls": [
                "https://example.com/generated.png"
            ]
        },
        "base_resp": {
            "status_code": 0,
            "status_msg": "success"
        }
    });

    let global_transport = MinimaxiJsonSuccessTransport::new(image_response.clone());
    let minimaxi_transport = MinimaxiJsonSuccessTransport::new(image_response);

    let registry = make_minimaxi_transport_override_registry(
        Arc::new(global_transport.clone()),
        Arc::new(minimaxi_transport.clone()),
    );

    let handle = registry
        .image_model("minimaxi:image-01")
        .expect("build registry image model");

    let generated = handle
        .generate_images(ImageGenerationRequest {
            prompt: "a tiny green robot".to_string(),
            negative_prompt: Some("blurry".to_string()),
            size: Some("1024x1024".to_string()),
            aspect_ratio: None,
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
        })
        .await
        .expect("generate images through registry handle");

    assert_eq!(
        generated.images[0].url.as_deref(),
        Some("https://example.com/generated.png")
    );
    assert!(global_transport.take().is_none());

    let req = minimaxi_transport.take().expect("captured request");
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
    assert_eq!(req.url, "https://example.com/custom/v1/image_generation");
    assert_eq!(req.body["model"], serde_json::json!("image-01"));
    assert_eq!(req.body["prompt"], serde_json::json!("a tiny green robot"));
    assert_eq!(req.body["size"], serde_json::json!("1024x1024"));
    assert_eq!(req.body["n"], serde_json::json!(1));
    assert_eq!(req.body["response_format"], serde_json::json!("url"));
}

#[tokio::test]
async fn minimaxi_registry_speech_handle_prefers_provider_specific_build_overrides() {
    let tts_response = serde_json::json!({
        "data": {
            "audio": "48656c6c6f",
            "status": 2
        },
        "extra_info": {
            "audio_length": 1000,
            "audio_sample_rate": 32000,
            "audio_size": 5,
            "bitrate": 128000,
            "word_count": 1,
            "usage_characters": 5,
            "audio_format": "mp3",
            "audio_channel": 1
        },
        "base_resp": {
            "status_code": 0,
            "status_msg": "success"
        }
    });

    let global_transport = MinimaxiJsonSuccessTransport::new(tts_response.clone());
    let minimaxi_transport = MinimaxiJsonSuccessTransport::new(tts_response);

    let registry = make_minimaxi_transport_override_registry(
        Arc::new(global_transport.clone()),
        Arc::new(minimaxi_transport.clone()),
    );

    let handle = registry
        .speech_model("minimaxi:speech-2.6-hd")
        .expect("build registry speech model");

    let generated = siumai::speech::SpeechModel::synthesize(
        &handle,
        TtsRequest::new("Hello".to_string())
            .with_voice("male-qn-qingse".to_string())
            .with_format("mp3".to_string())
            .with_minimaxi_tts_options(
                MinimaxiTtsOptions::new()
                    .with_emotion("happy")
                    .with_pitch(5)
                    .with_sample_rate(32000)
                    .with_bitrate(128000)
                    .with_channel(1)
                    .with_subtitle_enable(true),
            ),
    )
    .await
    .expect("text to speech through registry handle");

    assert_eq!(generated.audio_data, b"Hello");
    assert!(global_transport.take().is_none());

    let req = minimaxi_transport.take().expect("captured request");
    assert_eq!(
        header_value(&req, "authorization"),
        Some("Bearer ctx-key".to_string())
    );
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
    async fn mount_video_query_server() -> MockServer {
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

    let global_server = mount_video_query_server().await;
    let minimaxi_server = mount_video_query_server().await;

    let registry = make_minimaxi_server_override_registry(&global_server, &minimaxi_server);

    let handle = registry
        .video_model("minimaxi:hailuo-2.3")
        .expect("build registry video model");

    let queried = handle
        .query_video_task("task-123")
        .await
        .expect("query video through registry handle");

    assert_eq!(queried.task_id, "task-123");
    assert_eq!(queried.file_id.as_deref(), Some("file-123"));

    let global_requests = global_server
        .received_requests()
        .await
        .expect("global requests");
    assert!(
        global_requests.is_empty(),
        "global override lane should stay unused: {global_requests:?}"
    );

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
