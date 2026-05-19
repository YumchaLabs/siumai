use super::*;
use siumai::experimental::client::LlmClient;
use siumai::experimental::execution::http::transport::HttpTransportGetRequest;
use siumai::extensions::types::FileListQuery;
use siumai::extensions::{FileManagementCapability, VideoGenerationCapability};
use siumai::prelude::unified::{
    EmbeddingExtensions, EmbeddingRequest, ResponseFormat, Tool, ToolChoice,
};
use siumai::provider_ext::gemini::{
    GeminiChatRequestExt, GeminiChatResponseExt, GeminiContentPartExt, GeminiOptions,
    GeminiThinkingConfig, GoogleChatRequestExt, GoogleEmbeddingContentPart,
    GoogleEmbeddingInlineData, GoogleEmbeddingModelOptions, GoogleEmbeddingRequestExt,
    GoogleInteractionsAgentConfig, GoogleInteractionsModelInput,
    GoogleInteractionsResponseFormatEntry, GoogleLanguageModelInteractionsOptions,
    GoogleLanguageModelOptions,
};
use siumai_core::types::EmbeddingTaskType;
use std::collections::VecDeque;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request as WiremockRequest, ResponseTemplate};

fn gemini_registry_builder() -> siumai::registry::builder::RegistryBuilder {
    built_in_registry_builder("gemini", "gemini")
}

fn make_gemini_override_registry(
    global_transport: Arc<dyn HttpTransport>,
    provider_transport: Arc<dyn HttpTransport>,
) -> siumai::registry::ProviderRegistryHandle {
    gemini_registry_builder()
        .with_api_key("global-key")
        .with_base_url("https://example.com/global")
        .fetch(global_transport)
        .with_provider_api_key_base_url_fetch(
            "gemini",
            "ctx-key",
            "https://example.com/v1beta",
            provider_transport,
        )
        .auto_middleware(false)
        .build()
        .expect("build gemini override registry")
}

fn make_registry(
    transport: Arc<dyn HttpTransport>,
    base_url: &str,
) -> siumai::registry::ProviderRegistryHandle {
    gemini_registry_builder()
        .with_provider_api_key_base_url_fetch("gemini", "test-key", base_url, transport)
        .build()
        .expect("build gemini registry")
}

fn gemini_response_fixture(name: &str) -> serde_json::Value {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("google")
        .join("generative-ai")
        .join(name)
        .join("response.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("read gemini response fixture failed: {path:?}: {err}"));
    serde_json::from_str(&raw)
        .unwrap_or_else(|err| panic!("parse gemini response fixture failed: {path:?}: {err}"))
}

fn gemini_reasoning_stream_body() -> Vec<u8> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("google")
        .join("generative-ai-stream")
        .join("google-thought-signature-reasoning.1.chunks.txt");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("read gemini stream fixture failed: {path:?}: {err}"));

    let mut sse = String::new();
    for line in raw.lines().filter(|line| !line.trim().is_empty()) {
        sse.push_str("data: ");
        sse.push_str(line);
        sse.push_str("\n\n");
    }
    sse.push_str("data: [DONE]\n\n");
    sse.into_bytes()
}

#[derive(Clone)]
struct GoogleInteractionsCaptureTransport {
    post_responses: Arc<Mutex<VecDeque<serde_json::Value>>>,
    post_stream_responses: Arc<Mutex<VecDeque<String>>>,
    get_stream_responses: Arc<Mutex<VecDeque<String>>>,
    posts: Arc<Mutex<Vec<HttpTransportRequest>>>,
    post_streams: Arc<Mutex<Vec<HttpTransportRequest>>>,
    get_streams: Arc<Mutex<Vec<HttpTransportGetRequest>>>,
}

impl GoogleInteractionsCaptureTransport {
    fn new(post_responses: Vec<serde_json::Value>) -> Self {
        Self {
            post_responses: Arc::new(Mutex::new(post_responses.into())),
            post_stream_responses: Arc::new(Mutex::new(VecDeque::new())),
            get_stream_responses: Arc::new(Mutex::new(VecDeque::new())),
            posts: Arc::new(Mutex::new(Vec::new())),
            post_streams: Arc::new(Mutex::new(Vec::new())),
            get_streams: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn with_post_stream_response(self, response: String) -> Self {
        self.post_stream_responses
            .lock()
            .expect("lock google interactions post stream responses")
            .push_back(response);
        self
    }

    fn with_get_stream_response(self, response: String) -> Self {
        self.get_stream_responses
            .lock()
            .expect("lock google interactions get stream responses")
            .push_back(response);
        self
    }

    fn take_posts(&self) -> Vec<HttpTransportRequest> {
        std::mem::take(&mut *self.posts.lock().expect("lock google interactions posts"))
    }

    fn take_post_streams(&self) -> Vec<HttpTransportRequest> {
        std::mem::take(
            &mut *self
                .post_streams
                .lock()
                .expect("lock google interactions post streams"),
        )
    }

    fn take_get_streams(&self) -> Vec<HttpTransportGetRequest> {
        std::mem::take(
            &mut *self
                .get_streams
                .lock()
                .expect("lock google interactions get streams"),
        )
    }

    fn json_response(value: serde_json::Value) -> HttpTransportResponse {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        HttpTransportResponse {
            status: 200,
            headers,
            body: serde_json::to_vec(&value).expect("serialize interactions json response"),
        }
    }

    fn stream_response(body: String) -> HttpTransportStreamResponse {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));

        HttpTransportStreamResponse {
            status: 200,
            headers,
            body: HttpTransportStreamBody::from_bytes(body.into_bytes()),
        }
    }
}

#[async_trait]
impl HttpTransport for GoogleInteractionsCaptureTransport {
    async fn execute_json(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportResponse, LlmError> {
        self.posts
            .lock()
            .expect("lock google interactions posts")
            .push(request);
        let response = self
            .post_responses
            .lock()
            .expect("lock google interactions post responses")
            .pop_front()
            .unwrap_or_else(|| {
                serde_json::json!({
                    "id": "iact_default",
                    "status": "completed",
                    "model": siumai::provider_ext::google::interactions::GEMINI_2_5_FLASH,
                    "steps": []
                })
            });
        Ok(Self::json_response(response))
    }

    async fn execute_stream(
        &self,
        request: HttpTransportRequest,
    ) -> Result<HttpTransportStreamResponse, LlmError> {
        self.post_streams
            .lock()
            .expect("lock google interactions post streams")
            .push(request);
        let response = self
            .post_stream_responses
            .lock()
            .expect("lock google interactions post stream responses")
            .pop_front()
            .unwrap_or_else(|| "data: [DONE]\n\n".to_string());
        Ok(Self::stream_response(response))
    }

    async fn execute_get_stream(
        &self,
        request: HttpTransportGetRequest,
    ) -> Result<HttpTransportStreamResponse, LlmError> {
        self.get_streams
            .lock()
            .expect("lock google interactions get streams")
            .push(request);
        let response = self
            .get_stream_responses
            .lock()
            .expect("lock google interactions get stream responses")
            .pop_front()
            .unwrap_or_else(|| "data: [DONE]\n\n".to_string());
        Ok(Self::stream_response(response))
    }
}

fn google_interactions_completed_response(id: &str, model: &str, text: &str) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "status": "completed",
        "model": model,
        "service_tier": "priority",
        "steps": [{
            "type": "model_output",
            "content": [{ "type": "text", "text": text }]
        }],
        "usage": {
            "total_input_tokens": 3,
            "total_output_tokens": 2,
            "total_tokens": 5
        }
    })
}

fn google_interactions_sse_event(value: serde_json::Value) -> String {
    format!(
        "data: {}\n\n",
        serde_json::to_string(&value).expect("serialize google interactions SSE event")
    )
}

fn google_interactions_text_stream_body(id: &str, model: &str, text: &str) -> String {
    format!(
        "{}{}{}{}{}",
        google_interactions_sse_event(serde_json::json!({
            "event_type": "interaction.created",
            "event_id": "evt_1",
            "interaction": {
                "id": id,
                "model": model
            }
        })),
        google_interactions_sse_event(serde_json::json!({
            "event_type": "step.start",
            "event_id": "evt_2",
            "index": 0,
            "step": { "type": "model_output" }
        })),
        google_interactions_sse_event(serde_json::json!({
            "event_type": "step.delta",
            "event_id": "evt_3",
            "index": 0,
            "delta": { "type": "text", "text": text }
        })),
        google_interactions_sse_event(serde_json::json!({
            "event_type": "step.stop",
            "event_id": "evt_4",
            "index": 0
        })),
        google_interactions_sse_event(serde_json::json!({
            "event_type": "interaction.completed",
            "event_id": "evt_5",
            "interaction": {
                "id": id,
                "status": "completed"
            }
        })),
    )
}

fn google_interactions_agent_created_stream_body(id: &str, agent: &str) -> String {
    google_interactions_sse_event(serde_json::json!({
        "event_type": "interaction.created",
        "event_id": "evt_1",
        "interaction": {
            "id": id,
            "agent": agent
        }
    }))
}

fn google_interactions_agent_completed_stream_body(id: &str, text: &str) -> String {
    format!(
        "{}{}{}{}",
        google_interactions_sse_event(serde_json::json!({
            "event_type": "step.start",
            "event_id": "evt_2",
            "index": 0,
            "step": { "type": "model_output" }
        })),
        google_interactions_sse_event(serde_json::json!({
            "event_type": "step.delta",
            "event_id": "evt_3",
            "index": 0,
            "delta": { "type": "text", "text": text }
        })),
        google_interactions_sse_event(serde_json::json!({
            "event_type": "step.stop",
            "event_id": "evt_4",
            "index": 0
        })),
        google_interactions_sse_event(serde_json::json!({
            "event_type": "interaction.completed",
            "event_id": "evt_5",
            "interaction": {
                "id": id,
                "status": "completed"
            }
        })),
    )
}

fn custom_events_by_type(
    events: &[siumai::prelude::unified::ChatStreamEvent],
    ty: &str,
) -> Vec<serde_json::Value> {
    let stable_parts: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            siumai::prelude::unified::ChatStreamEvent::Part { part }
            | siumai::prelude::unified::ChatStreamEvent::PartWithReplay { part, .. } => {
                serde_json::to_value(part).ok()
            }
            _ => None,
        })
        .filter(|data| data.get("type") == Some(&serde_json::Value::String(ty.to_string())))
        .collect();
    if !stable_parts.is_empty() {
        return stable_parts;
    }

    events
        .iter()
        .filter_map(|event| match event {
            siumai::prelude::unified::ChatStreamEvent::Custom { data, .. } => Some(data.clone()),
            _ => None,
        })
        .filter(|data| data.get("type") == Some(&serde_json::Value::String(ty.to_string())))
        .collect()
}

async fn collect_stream_events(
    stream: &mut siumai::prelude::unified::ChatStream,
) -> Vec<siumai::prelude::unified::ChatStreamEvent> {
    use futures_util::StreamExt;

    let mut events = Vec::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(event) => events.push(event),
            Err(err) => panic!("collect gemini public-path stream event failed: {err:?}"),
        }
    }
    events
}

#[tokio::test]
async fn gemini_siumai_provider_config_embedding_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .gemini()
        .api_key("test-key")
        .base_url("https://example.com/v1beta")
        .model("gemini-embedding-001")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::gemini()
        .api_key("test-key")
        .base_url("https://example.com/v1beta")
        .model("gemini-embedding-001")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::gemini::GeminiClient::from_config(
        siumai::provider_ext::gemini::GeminiConfig::new("test-key")
            .with_base_url("https://example.com/v1beta".to_string())
            .with_model("gemini-embedding-001".to_string())
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = EmbeddingRequest::query("hello gemini embedding".to_string())
        .with_model("gemini-embedding-001")
        .with_title("Search Context")
        .with_dimensions(768);

    let _ = siumai_client.embed_with_config(request.clone()).await;
    let _ = provider_client.embed_with_config(request.clone()).await;
    let _ = config_client.embed_with_config(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(
        siumai_req.url,
        "https://example.com/v1beta/models/gemini-embedding-001:embedContent"
    );
    assert_eq!(
        siumai_req.body["model"],
        serde_json::json!("models/gemini-embedding-001")
    );
    assert_eq!(
        siumai_req.body["taskType"],
        serde_json::json!("RETRIEVAL_QUERY")
    );
    assert_eq!(
        siumai_req.body["title"],
        serde_json::json!("Search Context")
    );
    assert_eq!(
        siumai_req.body["outputDimensionality"],
        serde_json::json!(768)
    );
}

#[tokio::test]
async fn gemini_siumai_provider_config_batch_embedding_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let siumai_client = Siumai::builder()
        .gemini()
        .api_key("test-key")
        .base_url("https://example.com/v1beta")
        .model("gemini-embedding-001")
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::gemini()
        .api_key("test-key")
        .base_url("https://example.com/v1beta")
        .model("gemini-embedding-001")
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::gemini::GeminiClient::from_config(
        siumai::provider_ext::gemini::GeminiConfig::new("test-key")
            .with_base_url("https://example.com/v1beta".to_string())
            .with_model("gemini-embedding-001".to_string())
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = EmbeddingRequest::new(vec!["A".to_string(), "B".to_string()])
        .with_model("gemini-embedding-001")
        .with_dimensions(64)
        .with_task_type(EmbeddingTaskType::SemanticSimilarity);

    let _ = siumai_client.embed_with_config(request.clone()).await;
    let _ = provider_client.embed_with_config(request.clone()).await;
    let _ = config_client.embed_with_config(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(
        siumai_req.url,
        "https://example.com/v1beta/models/gemini-embedding-001:batchEmbedContents"
    );
    let requests = siumai_req.body["requests"]
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

fn wiremock_header_value(req: &WiremockRequest, key: &str) -> Option<String> {
    req.headers
        .get(key)
        .and_then(|v| v.to_str().ok())
        .map(ToString::to_string)
}

#[tokio::test]
async fn gemini_provider_builder_files_list_request_matches_provider_client_paths() {
    async fn mount_list_server() -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1beta/files"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "files": [{
                    "name": "files/123",
                    "display_name": "hello.txt",
                    "mime_type": "text/plain",
                    "size_bytes": "5",
                    "create_time": "2026-04-22T00:00:00Z",
                    "state": "ACTIVE",
                    "uri": "https://example.com/files/123"
                }],
                "next_page_token": "page-2"
            })))
            .mount(&server)
            .await;
        server
    }

    let provider_client_server = mount_list_server().await;
    let provider_builder_server = mount_list_server().await;
    let config_server = mount_list_server().await;

    let provider_client = Provider::google()
        .api_key("test-key")
        .base_url(format!("{}/v1beta", provider_client_server.uri()))
        .model("gemini-2.5-flash")
        .build()
        .await
        .expect("build provider client");

    let provider_files = Provider::google()
        .api_key("test-key")
        .base_url(format!("{}/v1beta", provider_builder_server.uri()))
        .files()
        .expect("build provider files");

    let config_client = siumai::provider_ext::google::GeminiClient::from_config(
        siumai::provider_ext::google::GeminiConfig::new("test-key")
            .with_base_url(format!("{}/v1beta", config_server.uri())),
    )
    .expect("build config client");

    let query = FileListQuery {
        limit: Some(20),
        after: Some("page123".to_string()),
        ..Default::default()
    };

    let provider_client_list = provider_client
        .list_files(Some(query.clone()))
        .await
        .expect("provider client list ok");
    let provider_builder_list = provider_files
        .list_files(Some(query.clone()))
        .await
        .expect("provider builder list ok");
    let config_list = config_client
        .files()
        .list_files(Some(query))
        .await
        .expect("config client list ok");

    assert_eq!(provider_client_list.files.len(), 1);
    assert_eq!(provider_builder_list.files.len(), 1);
    assert_eq!(config_list.files.len(), 1);
    assert_eq!(provider_client_list.files[0].id, "123");
    assert_eq!(provider_builder_list.files[0].id, "123");
    assert_eq!(config_list.files[0].id, "123");

    let provider_client_req = provider_client_server
        .received_requests()
        .await
        .expect("recorded provider client requests")
        .into_iter()
        .next()
        .expect("provider client list-files request");
    let provider_builder_req = provider_builder_server
        .received_requests()
        .await
        .expect("recorded provider builder requests")
        .into_iter()
        .next()
        .expect("provider builder list-files request");
    let config_req = config_server
        .received_requests()
        .await
        .expect("recorded config requests")
        .into_iter()
        .next()
        .expect("config list-files request");

    assert_eq!(provider_client_req.url.path(), "/v1beta/files");
    assert_eq!(provider_builder_req.url.path(), "/v1beta/files");
    assert_eq!(config_req.url.path(), "/v1beta/files");
    assert_eq!(
        provider_client_req.url.query(),
        Some("pageSize=20&pageToken=page123")
    );
    assert_eq!(
        provider_client_req.url.query(),
        provider_builder_req.url.query()
    );
    assert_eq!(provider_client_req.url.query(), config_req.url.query());
    assert_eq!(
        wiremock_header_value(&provider_client_req, "x-goog-api-key"),
        Some("test-key".to_string())
    );
    assert_eq!(
        wiremock_header_value(&provider_client_req, "x-goog-api-key"),
        wiremock_header_value(&provider_builder_req, "x-goog-api-key")
    );
    assert_eq!(
        wiremock_header_value(&provider_client_req, "x-goog-api-key"),
        wiremock_header_value(&config_req, "x-goog-api-key")
    );
}

#[tokio::test]
async fn gemini_google_and_package_settings_preserve_provider_name_boundaries() {
    let google_client = Provider::google()
        .api_key("test-key")
        .model("gemini-2.5-flash")
        .build()
        .await
        .expect("build google client");
    let gemini_client = Provider::gemini()
        .api_key("test-key")
        .model("gemini-2.5-flash")
        .build()
        .await
        .expect("build gemini client");
    let custom_client = siumai::provider_ext::google::GoogleProviderSettings::new()
        .with_api_key("test-key")
        .with_name("my-gemini-proxy")
        .into_builder_for_model("gemini-2.5-flash")
        .build()
        .await
        .expect("build custom google client");

    assert_eq!(google_client.provider_id().as_ref(), "gemini");
    assert_eq!(google_client.provider_name(), "google.generative-ai");
    assert_eq!(
        google_client.files().provider_name(),
        "google.generative-ai"
    );

    assert_eq!(gemini_client.provider_id().as_ref(), "gemini");
    assert_eq!(gemini_client.provider_name(), "gemini");
    assert_eq!(gemini_client.files().provider_name(), "gemini");

    assert_eq!(custom_client.provider_id().as_ref(), "gemini");
    assert_eq!(custom_client.provider_name(), "my-gemini-proxy");
    assert_eq!(custom_client.files().provider_name(), "my-gemini-proxy");
}

#[tokio::test]
async fn gemini_public_paths_preserve_google_embedding_provider_options() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let base_url = "https://example.com/v1beta";
    let model = "gemini-embedding-001";

    let siumai_client = Siumai::builder()
        .gemini()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::gemini()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::gemini::GeminiClient::from_config(
        siumai::provider_ext::gemini::GeminiConfig::new("test-key")
            .with_base_url(base_url.to_string())
            .with_model(model.to_string())
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .embedding_model("gemini:gemini-embedding-001")
        .expect("build registry embedding model");

    let request = EmbeddingRequest::new(vec!["A".to_string(), "B".to_string()])
        .with_model(model)
        .with_google_embedding_options(
            GoogleEmbeddingModelOptions::new()
                .with_output_dimensionality(64)
                .with_task_type(EmbeddingTaskType::SemanticSimilarity)
                .with_content(vec![
                    Some(vec![GoogleEmbeddingContentPart::InlineData {
                        inline_data: GoogleEmbeddingInlineData {
                            mime_type: "image/png".to_string(),
                            data: "Zm9v".to_string(),
                        },
                    }]),
                    None,
                ]),
        );

    let _ = siumai_client.embed_with_config(request.clone()).await;
    let _ = provider_client.embed_with_config(request.clone()).await;
    let _ = config_client.embed_with_config(request.clone()).await;
    let _ = registry_model.embed_with_config(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);

    let requests = siumai_req.body["requests"]
        .as_array()
        .expect("requests array");
    assert_eq!(requests.len(), 2);
    assert_eq!(
        requests[0]["taskType"],
        serde_json::json!("SEMANTIC_SIMILARITY")
    );
    assert_eq!(requests[0]["outputDimensionality"], serde_json::json!(64));
    assert_eq!(
        requests[0]["content"]["parts"][0]["text"],
        serde_json::json!("A")
    );
    assert_eq!(
        requests[0]["content"]["parts"][1]["inlineData"]["mimeType"],
        serde_json::json!("image/png")
    );
    assert_eq!(
        requests[0]["content"]["parts"][1]["inlineData"]["data"],
        serde_json::json!("Zm9v")
    );
    assert_eq!(
        requests[1]["content"]["parts"][0]["text"],
        serde_json::json!("B")
    );
    assert_eq!(
        requests[1]["content"]["parts"]
            .as_array()
            .map(|parts| parts.len()),
        Some(1)
    );
}

#[tokio::test]
async fn gemini_public_paths_preserve_google_language_model_options() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "gemini-2.5-flash";
    let base_url = "https://example.com/v1beta";

    let siumai_client = Siumai::builder()
        .gemini()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::gemini()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::gemini::GeminiClient::from_config(
        siumai::provider_ext::gemini::GeminiConfig::new("test-key")
            .with_base_url(base_url.to_string())
            .with_model(model.to_string())
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");
    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("gemini:gemini-2.5-flash")
        .expect("build registry language model");

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .build()
        .with_google_options(
            GoogleLanguageModelOptions::new()
                .with_service_tier("flex")
                .with_structured_outputs(true),
        );

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(siumai_req.body["serviceTier"], serde_json::json!("flex"));
}

#[tokio::test]
async fn google_interactions_model_public_paths_execute_non_stream_runtime() {
    let provider_transport =
        GoogleInteractionsCaptureTransport::new(vec![google_interactions_completed_response(
            "iact_model_public",
            siumai::provider_ext::google::interactions::GEMINI_2_5_FLASH,
            "hello from interactions",
        )]);
    let package_transport =
        GoogleInteractionsCaptureTransport::new(vec![google_interactions_completed_response(
            "iact_model_public",
            siumai::provider_ext::google::interactions::GEMINI_2_5_FLASH,
            "hello from interactions",
        )]);
    let direct_transport =
        GoogleInteractionsCaptureTransport::new(vec![google_interactions_completed_response(
            "iact_model_public",
            siumai::provider_ext::google::interactions::GEMINI_2_5_FLASH,
            "hello from interactions",
        )]);

    let request = ChatRequest::builder()
        .model(siumai::provider_ext::google::interactions::GEMINI_2_5_FLASH)
        .messages(vec![ChatMessage::user("hi").build()])
        .build()
        .with_google_interactions_options(
            GoogleLanguageModelInteractionsOptions::new()
                .with_previous_interaction_id("iact_123")
                .with_store(true)
                .with_response_format(vec![
                    GoogleInteractionsResponseFormatEntry::json_schema(serde_json::json!({
                        "type": "object"
                    })),
                    GoogleInteractionsResponseFormatEntry::image()
                        .with_mime_type("image/png")
                        .with_aspect_ratio("16:9")
                        .with_image_size("1K"),
                ])
                .with_media_resolution("high")
                .with_response_modalities(["text", "image"])
                .with_service_tier("priority")
                .with_system_instruction("be concise")
                .with_signature("sig_123")
                .with_interaction_id("iact_456")
                .with_polling_timeout_ms(30_000),
        );

    let options = request
        .provider_option("google")
        .expect("google provider options");
    assert_eq!(
        options["previousInteractionId"],
        serde_json::json!("iact_123")
    );
    assert!(options.get("agent").is_none());
    assert!(options.get("agentConfig").is_none());
    assert_eq!(
        options["responseFormat"][1]["aspectRatio"],
        serde_json::json!("16:9")
    );
    assert_eq!(options["serviceTier"], serde_json::json!("priority"));
    assert_eq!(options["pollingTimeoutMs"], serde_json::json!(30_000));

    let provider_model = Provider::google()
        .api_key("test-key")
        .base_url("https://example.com/v1beta")
        .fetch(Arc::new(provider_transport.clone()))
        .interactions(GoogleInteractionsModelInput::model(
            siumai::provider_ext::google::interactions::GEMINI_2_5_FLASH,
        ))
        .expect("build Provider::google interactions model");
    let package_model = siumai::provider_ext::google::create_google()
        .api_key("test-key")
        .base_url("https://example.com/v1beta")
        .fetch(Arc::new(package_transport.clone()))
        .interactions(GoogleInteractionsModelInput::model(
            siumai::provider_ext::google::interactions::GEMINI_2_5_FLASH,
        ))
        .expect("build provider_ext::google interactions model");
    let direct_model = siumai::provider_ext::google::GoogleInteractionsLanguageModel::new(
        siumai::provider_ext::google::GeminiConfig::new("test-key")
            .with_provider_name("google.generative-ai")
            .with_base_url("https://example.com/v1beta".to_string())
            .with_http_transport(Arc::new(direct_transport.clone())),
        GoogleInteractionsModelInput::model(
            siumai::provider_ext::google::interactions::GEMINI_2_5_FLASH,
        ),
    );

    assert_eq!(
        provider_model.provider(),
        "google.generative-ai.interactions"
    );
    assert_eq!(
        package_model.provider(),
        "google.generative-ai.interactions"
    );
    assert_eq!(direct_model.provider(), "google.generative-ai.interactions");

    let provider_response = provider_model
        .chat_request(request.clone())
        .await
        .expect("Provider::google interactions non-stream response");
    let package_response = package_model
        .chat_request(request.clone())
        .await
        .expect("provider_ext::google interactions non-stream response");
    let direct_response = direct_model
        .chat_request(request)
        .await
        .expect("direct interactions non-stream response");

    for response in [&provider_response, &package_response, &direct_response] {
        assert_eq!(response.id.as_deref(), Some("iact_model_public"));
        assert_eq!(response.text().as_deref(), Some("hello from interactions"));
        assert_eq!(response.service_tier.as_deref(), Some("priority"));
        assert_eq!(
            response.get_metadata("google", "interactionId"),
            Some(&serde_json::json!("iact_model_public"))
        );
    }

    let provider_req = provider_transport
        .take_posts()
        .pop()
        .expect("provider interactions POST request");
    let package_req = package_transport
        .take_posts()
        .pop()
        .expect("package interactions POST request");
    let direct_req = direct_transport
        .take_posts()
        .pop()
        .expect("direct interactions POST request");

    assert_requests_equivalent(&provider_req, &package_req);
    assert_requests_equivalent(&provider_req, &direct_req);
    assert_eq!(provider_req.url, "https://example.com/v1beta/interactions");
    assert_eq!(
        header_value(&provider_req, "x-goog-api-key"),
        Some("test-key".to_string())
    );
    assert_eq!(
        header_value(&provider_req, "api-revision"),
        Some("2026-05-20".to_string())
    );
    assert_eq!(
        provider_req.body["model"],
        serde_json::json!(siumai::provider_ext::google::interactions::GEMINI_2_5_FLASH)
    );
    assert_eq!(
        provider_req.body["previous_interaction_id"],
        serde_json::json!("iact_123")
    );
    assert_eq!(provider_req.body["store"], serde_json::json!(true));
    assert_eq!(
        provider_req.body["response_format"][1]["aspect_ratio"],
        serde_json::json!("16:9")
    );
    assert_eq!(
        provider_req.body["service_tier"],
        serde_json::json!("priority")
    );
    assert_eq!(
        provider_req.body["system_instruction"],
        serde_json::json!("be concise")
    );
    assert!(provider_req.body.get("background").is_none());
    assert!(provider_req.body.get("stream").is_none());
    assert!(provider_transport.take_post_streams().is_empty());
    assert!(package_transport.take_post_streams().is_empty());
    assert!(direct_transport.take_post_streams().is_empty());
}

#[tokio::test]
async fn google_interactions_model_public_paths_execute_stream_runtime() {
    let provider_transport = GoogleInteractionsCaptureTransport::new(vec![])
        .with_post_stream_response(google_interactions_text_stream_body(
            "iact_stream_public",
            siumai::provider_ext::google::interactions::GEMINI_2_5_FLASH,
            "stream text",
        ));
    let package_transport = GoogleInteractionsCaptureTransport::new(vec![])
        .with_post_stream_response(google_interactions_text_stream_body(
            "iact_stream_public",
            siumai::provider_ext::google::interactions::GEMINI_2_5_FLASH,
            "stream text",
        ));
    let direct_transport = GoogleInteractionsCaptureTransport::new(vec![])
        .with_post_stream_response(google_interactions_text_stream_body(
            "iact_stream_public",
            siumai::provider_ext::google::interactions::GEMINI_2_5_FLASH,
            "stream text",
        ));

    let provider_model = Provider::google()
        .api_key("test-key")
        .base_url("https://example.com/v1beta")
        .fetch(Arc::new(provider_transport.clone()))
        .interactions(GoogleInteractionsModelInput::model(
            siumai::provider_ext::google::interactions::GEMINI_2_5_FLASH,
        ))
        .expect("build Provider::google interactions stream model");
    let package_model = siumai::provider_ext::google::google()
        .api_key("test-key")
        .base_url("https://example.com/v1beta")
        .fetch(Arc::new(package_transport.clone()))
        .interactions(GoogleInteractionsModelInput::model(
            siumai::provider_ext::google::interactions::GEMINI_2_5_FLASH,
        ))
        .expect("build provider_ext::google interactions stream model");
    let direct_model = siumai::provider_ext::google::GoogleInteractionsLanguageModel::new(
        siumai::provider_ext::google::GeminiConfig::new("test-key")
            .with_provider_name("google.generative-ai")
            .with_base_url("https://example.com/v1beta".to_string())
            .with_http_transport(Arc::new(direct_transport.clone())),
        GoogleInteractionsModelInput::model(
            siumai::provider_ext::google::interactions::GEMINI_2_5_FLASH,
        ),
    );

    let request = ChatRequest::new(vec![ChatMessage::user("hi").build()]);
    let mut provider_stream = provider_model
        .chat_stream_request(request.clone())
        .await
        .expect("Provider::google interactions stream");
    let mut package_stream = package_model
        .chat_stream_request(request.clone())
        .await
        .expect("provider_ext::google interactions stream");
    let mut direct_stream = direct_model
        .chat_stream_request(request)
        .await
        .expect("direct interactions stream");

    let provider_events = collect_stream_events(&mut provider_stream).await;
    let package_events = collect_stream_events(&mut package_stream).await;
    let direct_events = collect_stream_events(&mut direct_stream).await;

    for events in [&provider_events, &package_events, &direct_events] {
        assert_eq!(
            events
                .iter()
                .filter_map(|event| event.text_delta())
                .collect::<Vec<_>>(),
            vec!["stream text"]
        );
        assert!(events.iter().any(|event| matches!(
            event.part_ref(),
            Some(siumai::prelude::unified::ChatStreamPart::Finish { finish_reason, .. })
                if finish_reason.raw.as_deref() == Some("completed")
        )));
    }

    let provider_req = provider_transport
        .take_post_streams()
        .pop()
        .expect("provider interactions stream POST request");
    let package_req = package_transport
        .take_post_streams()
        .pop()
        .expect("package interactions stream POST request");
    let direct_req = direct_transport
        .take_post_streams()
        .pop()
        .expect("direct interactions stream POST request");

    assert_requests_equivalent(&provider_req, &package_req);
    assert_requests_equivalent(&provider_req, &direct_req);
    assert_eq!(provider_req.url, "https://example.com/v1beta/interactions");
    assert_eq!(provider_req.body["stream"], serde_json::json!(true));
    assert_eq!(
        provider_req.body["model"],
        serde_json::json!(siumai::provider_ext::google::interactions::GEMINI_2_5_FLASH)
    );
    assert!(provider_transport.take_posts().is_empty());
    assert!(provider_transport.take_get_streams().is_empty());
}

#[tokio::test]
async fn google_interactions_agent_public_paths_execute_background_get_stream_runtime() {
    let agent = siumai::provider_ext::google::agents::DEEP_RESEARCH_PREVIEW_04_2026;
    let provider_transport = GoogleInteractionsCaptureTransport::new(vec![serde_json::json!({
        "id": "iact_agent_public",
        "status": "in_progress",
        "agent": agent,
        "steps": []
    })])
    .with_get_stream_response(google_interactions_agent_created_stream_body(
        "iact_agent_public",
        agent,
    ))
    .with_get_stream_response(google_interactions_agent_completed_stream_body(
        "iact_agent_public",
        "agent stream text",
    ));
    let package_transport = GoogleInteractionsCaptureTransport::new(vec![serde_json::json!({
        "id": "iact_agent_public",
        "status": "in_progress",
        "agent": agent,
        "steps": []
    })])
    .with_get_stream_response(google_interactions_agent_created_stream_body(
        "iact_agent_public",
        agent,
    ))
    .with_get_stream_response(google_interactions_agent_completed_stream_body(
        "iact_agent_public",
        "agent stream text",
    ));
    let direct_transport = GoogleInteractionsCaptureTransport::new(vec![serde_json::json!({
        "id": "iact_agent_public",
        "status": "in_progress",
        "agent": agent,
        "steps": []
    })])
    .with_get_stream_response(google_interactions_agent_created_stream_body(
        "iact_agent_public",
        agent,
    ))
    .with_get_stream_response(google_interactions_agent_completed_stream_body(
        "iact_agent_public",
        "agent stream text",
    ));

    let provider_model = Provider::google()
        .api_key("test-key")
        .base_url("https://example.com/v1beta")
        .fetch(Arc::new(provider_transport.clone()))
        .interactions(GoogleInteractionsModelInput::agent(agent))
        .expect("build Provider::google interactions agent");
    let package_model = siumai::provider_ext::google::create_google()
        .api_key("test-key")
        .base_url("https://example.com/v1beta")
        .fetch(Arc::new(package_transport.clone()))
        .interactions(GoogleInteractionsModelInput::agent(agent))
        .expect("build provider_ext::google interactions agent");
    let direct_model = siumai::provider_ext::google::GoogleInteractionsLanguageModel::new(
        siumai::provider_ext::google::GeminiConfig::new("test-key")
            .with_provider_name("google.generative-ai")
            .with_base_url("https://example.com/v1beta".to_string())
            .with_http_transport(Arc::new(direct_transport.clone())),
        GoogleInteractionsModelInput::agent(agent),
    );

    let request = ChatRequest::new(vec![ChatMessage::user("research").build()])
        .with_google_interactions_options(
            GoogleLanguageModelInteractionsOptions::new().with_agent_config(
                GoogleInteractionsAgentConfig::deep_research()
                    .with_thinking_summaries("auto")
                    .with_visualization("auto")
                    .with_collaborative_planning(true),
            ),
        );

    let mut provider_stream = provider_model
        .chat_stream_request(request.clone())
        .await
        .expect("Provider::google interactions agent stream");
    let mut package_stream = package_model
        .chat_stream_request(request.clone())
        .await
        .expect("provider_ext::google interactions agent stream");
    let mut direct_stream = direct_model
        .chat_stream_request(request)
        .await
        .expect("direct interactions agent stream");

    let provider_events = collect_stream_events(&mut provider_stream).await;
    let package_events = collect_stream_events(&mut package_stream).await;
    let direct_events = collect_stream_events(&mut direct_stream).await;

    for events in [&provider_events, &package_events, &direct_events] {
        assert_eq!(
            events
                .iter()
                .filter_map(|event| event.text_delta())
                .collect::<Vec<_>>(),
            vec!["agent stream text"]
        );
    }

    let provider_post = provider_transport
        .take_posts()
        .pop()
        .expect("provider interactions agent POST request");
    let package_post = package_transport
        .take_posts()
        .pop()
        .expect("package interactions agent POST request");
    let direct_post = direct_transport
        .take_posts()
        .pop()
        .expect("direct interactions agent POST request");

    assert_requests_equivalent(&provider_post, &package_post);
    assert_requests_equivalent(&provider_post, &direct_post);
    assert_eq!(provider_post.body["agent"], serde_json::json!(agent));
    assert_eq!(provider_post.body["background"], serde_json::json!(true));
    assert!(provider_post.body.get("model").is_none());
    assert!(provider_post.body.get("stream").is_none());
    assert_eq!(
        provider_post.body["agent_config"]["type"],
        serde_json::json!("deep-research")
    );

    let provider_get_streams = provider_transport.take_get_streams();
    let package_get_streams = package_transport.take_get_streams();
    let direct_get_streams = direct_transport.take_get_streams();

    assert_eq!(provider_get_streams.len(), 2);
    assert_eq!(package_get_streams.len(), 2);
    assert_eq!(direct_get_streams.len(), 2);
    assert_eq!(
        provider_get_streams[0].url,
        "https://example.com/v1beta/interactions/iact_agent_public?stream=true"
    );
    assert_eq!(
        provider_get_streams[1].url,
        "https://example.com/v1beta/interactions/iact_agent_public?stream=true&last_event_id=evt_1"
    );
    assert_eq!(provider_get_streams[0].url, package_get_streams[0].url);
    assert_eq!(provider_get_streams[1].url, package_get_streams[1].url);
    assert_eq!(provider_get_streams[0].url, direct_get_streams[0].url);
    assert_eq!(provider_get_streams[1].url, direct_get_streams[1].url);
    assert_eq!(
        provider_get_streams[1]
            .headers
            .get("api-revision")
            .and_then(|value| value.to_str().ok()),
        Some("2026-05-20")
    );
}

#[tokio::test]
async fn gemini_siumai_provider_config_chat_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "gemini-2.5-flash";
    let base_url = "https://example.com/v1beta";

    let siumai_client = Siumai::builder()
        .gemini()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::gemini()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::gemini::GeminiClient::from_config(
        siumai::provider_ext::gemini::GeminiConfig::new("test-key")
            .with_base_url(base_url.to_string())
            .with_model(model.to_string())
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .build();

    let _ = siumai_client.chat_request(request.clone()).await;
    let _ = provider_client.chat_request(request.clone()).await;
    let _ = config_client.chat_request(request).await;

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_eq!(
        siumai_req.url,
        "https://example.com/v1beta/models/gemini-2.5-flash:generateContent"
    );
    assert_eq!(
        header_value(&siumai_req, "x-goog-api-key"),
        Some("test-key".to_string())
    );
    assert_eq!(
        siumai_req.body["contents"][0]["parts"][0]["text"],
        serde_json::json!("hi")
    );
}

#[tokio::test]
async fn gemini_siumai_provider_config_chat_stream_request_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "gemini-2.5-flash";
    let base_url = "https://example.com/v1beta";

    let siumai_client = Siumai::builder()
        .gemini()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::gemini()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::gemini::GeminiClient::from_config(
        siumai::provider_ext::gemini::GeminiConfig::new("test-key")
            .with_base_url(base_url.to_string())
            .with_model(model.to_string())
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .build();

    let _ = siumai_client.chat_stream_request(request.clone()).await;
    let _ = provider_client.chat_stream_request(request.clone()).await;
    let _ = config_client.chat_stream_request(request).await;

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
    assert_eq!(
        siumai_req.url,
        "https://example.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
    );
    assert_eq!(
        header_value(&siumai_req, "x-goog-api-key"),
        Some("test-key".to_string())
    );
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
    assert_eq!(
        siumai_req.body["contents"][0]["parts"][0]["text"],
        serde_json::json!("hi")
    );
}

#[tokio::test]
async fn gemini_registry_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let gemini_transport = CaptureTransport::default();

    let registry = make_gemini_override_registry(
        Arc::new(global_transport.clone()),
        Arc::new(gemini_transport.clone()),
    );

    let handle = registry
        .language_model("gemini:gemini-2.5-flash")
        .expect("build gemini handle");

    let _ = handle
        .chat_request(make_chat_request_with_model("gemini-2.5-flash"))
        .await;

    let req = gemini_transport.take().expect("captured gemini request");
    assert!(global_transport.take().is_none());
    assert_eq!(
        header_value(&req, "x-goog-api-key"),
        Some("ctx-key".to_string())
    );
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
async fn gemini_registry_stream_handle_prefers_provider_specific_build_overrides() {
    let global_transport = CaptureTransport::default();
    let gemini_transport = CaptureTransport::default();

    let registry = make_gemini_override_registry(
        Arc::new(global_transport.clone()),
        Arc::new(gemini_transport.clone()),
    );

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
    assert_eq!(
        header_value(&req, "x-goog-api-key"),
        Some("ctx-key".to_string())
    );
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
async fn gemini_siumai_provider_config_audio_family_requests_are_intentionally_unsupported() {
    let siumai_transport = MixedCaptureTransport::default();
    let provider_transport = MixedCaptureTransport::default();
    let config_transport = MixedCaptureTransport::default();

    let model = "gemini-2.5-flash-preview-native-audio-dialog";
    let base_url = "https://example.com/v1beta";

    let siumai_client = Siumai::builder()
        .gemini()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::gemini()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::gemini::GeminiClient::from_config(
        siumai::provider_ext::gemini::GeminiConfig::new("test-key")
            .with_base_url(base_url.to_string())
            .with_model(model.to_string())
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let tts_request = TtsRequest::new("hello gemini audio".to_string())
        .with_voice("alloy".to_string())
        .with_format("mp3".to_string());
    let stt_request = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg");

    let siumai_tts_err = siumai_client
        .text_to_speech(tts_request.clone())
        .await
        .expect_err("gemini text-to-speech should be unsupported");

    let siumai_stt_err = siumai_client
        .speech_to_text(stt_request.clone())
        .await
        .expect_err("gemini speech-to-text should be unsupported");

    for err in [siumai_tts_err, siumai_stt_err] {
        assert_unsupported_operation(&err);
    }

    assert!(siumai_client.as_speech_capability().is_none());
    assert!(provider_client.as_speech_capability().is_none());
    assert!(config_client.as_speech_capability().is_none());
    assert!(siumai_client.as_transcription_capability().is_none());
    assert!(provider_client.as_transcription_capability().is_none());
    assert!(config_client.as_transcription_capability().is_none());

    assert_mixed_capture_transports_unused(&[
        &siumai_transport,
        &provider_transport,
        &config_transport,
    ]);
}

#[tokio::test]
async fn gemini_registry_audio_family_requests_are_intentionally_unsupported() {
    let registry_transport = MixedCaptureTransport::default();
    let base_url = "https://example.com/v1beta";
    let model = "gemini:gemini-2.5-flash-preview-native-audio-dialog";
    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let tts_err = match registry.speech_model(model) {
        Ok(_) => panic!("build gemini registry speech model should be unsupported"),
        Err(err) => err,
    };
    let stt_err = match registry.transcription_model(model) {
        Ok(_) => panic!("build gemini registry transcription model should be unsupported"),
        Err(err) => err,
    };

    assert_unsupported_operation(&tts_err);
    assert_unsupported_operation(&stt_err);
    assert_mixed_capture_transports_unused(&[&registry_transport]);
}

#[tokio::test]
async fn gemini_siumai_provider_config_rerank_request_is_intentionally_unsupported() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "gemini-2.5-flash";
    let base_url = "https://example.com/v1beta";

    let siumai_client = Siumai::builder()
        .gemini()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::gemini()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::gemini::GeminiClient::from_config(
        siumai::provider_ext::gemini::GeminiConfig::new("test-key")
            .with_base_url(base_url.to_string())
            .with_model(model.to_string())
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let err = siumai_client
        .rerank(make_rerank_request_with_model(model).with_top_n(1))
        .await
        .expect_err("gemini rerank should be unsupported");

    assert_unsupported_operation(&err);
    assert!(siumai_client.as_rerank_capability().is_none());
    assert!(provider_client.as_rerank_capability().is_none());
    assert!(config_client.as_rerank_capability().is_none());
    assert_capture_transports_unused(&[&siumai_transport, &provider_transport, &config_transport]);
}

#[tokio::test]
async fn gemini_registry_rerank_request_is_intentionally_unsupported() {
    let registry_transport = CaptureTransport::default();
    let base_url = "https://example.com/v1beta";
    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let err = match registry.reranking_model("gemini:gemini-2.5-flash") {
        Ok(_) => panic!("gemini registry rerank handle should be unsupported"),
        Err(err) => err,
    };

    assert_unsupported_operation(&err);
    assert_capture_transports_unused(&[&registry_transport]);
}

#[tokio::test]
async fn gemini_siumai_provider_config_chat_response_metadata_are_equivalent() {
    let response_json = gemini_response_fixture("google-thought-signature-text-and-reasoning.1");

    let siumai_transport = JsonSuccessTransport::new(response_json.clone());
    let provider_transport = JsonSuccessTransport::new(response_json.clone());
    let config_transport = JsonSuccessTransport::new(response_json.clone());
    let registry_transport = JsonSuccessTransport::new(response_json);

    let model = "gemini-pro";
    let base_url = "https://example.com/v1beta";

    let siumai_client = Siumai::builder()
        .gemini()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::gemini()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::gemini::GeminiClient::from_config(
        siumai::provider_ext::gemini::GeminiConfig::new("test-key")
            .with_base_url(base_url.to_string())
            .with_model(model.to_string())
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");
    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("gemini:gemini-pro")
        .expect("build registry language model");

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("Hello").build()])
        .build();

    let siumai_resp = siumai_client
        .chat_request(request.clone())
        .await
        .expect("siumai response");
    let provider_resp = provider_client
        .chat_request(request.clone())
        .await
        .expect("provider response");
    let config_resp = config_client
        .chat_request(request)
        .await
        .expect("config response");
    let registry_resp = registry_model
        .chat_request(
            ChatRequest::builder()
                .model(model)
                .messages(vec![ChatMessage::user("Hello").build()])
                .build(),
        )
        .await
        .expect("registry response");

    for response in [&siumai_resp, &provider_resp, &config_resp, &registry_resp] {
        let siumai::prelude::unified::MessageContent::MultiModal(parts) = &response.content else {
            panic!("expected multimodal content");
        };
        assert_eq!(parts.len(), 3);
        assert_eq!(
            response.reasoning(),
            vec!["This is a thought process.".to_string()]
        );

        assert_eq!(
            parts[0]
                .gemini_metadata()
                .and_then(|meta| meta.thought_signature),
            Some("sig1".to_string())
        );
        assert_eq!(
            parts[1]
                .gemini_metadata()
                .and_then(|meta| meta.thought_signature),
            Some("sig2".to_string())
        );
        assert_eq!(
            parts[2]
                .gemini_metadata()
                .and_then(|meta| meta.thought_signature),
            Some("sig3".to_string())
        );

        let meta = response
            .gemini_metadata()
            .expect("expected gemini metadata");
        assert_eq!(meta.safety_ratings.as_ref().map(Vec::len), Some(1));
        assert_eq!(
            meta.safety_ratings
                .as_ref()
                .and_then(|ratings| ratings.first())
                .map(|rating| rating.category.as_str()),
            Some("HARM_CATEGORY_DEROGATORY")
        );

        let provider_metadata = response
            .provider_metadata
            .as_ref()
            .expect("expected provider_metadata");
        assert!(provider_metadata.contains_key("google"));
    }

    let siumai_req = siumai_transport.take().expect("siumai request");
    let provider_req = provider_transport.take().expect("provider request");
    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(
        siumai_req.url,
        "https://example.com/v1beta/models/gemini-pro:generateContent"
    );
}

#[tokio::test]
async fn gemini_siumai_provider_config_stream_end_metadata_are_equivalent() {
    let stream_body = gemini_reasoning_stream_body();

    let siumai_transport = SseSuccessTransport::new(stream_body.clone());
    let provider_transport = SseSuccessTransport::new(stream_body.clone());
    let config_transport = SseSuccessTransport::new(stream_body.clone());
    let registry_transport = SseSuccessTransport::new(stream_body);

    let model = "gemini-pro";
    let base_url = "https://example.com/v1beta";

    let siumai_client = Siumai::builder()
        .gemini()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::gemini()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::gemini::GeminiClient::from_config(
        siumai::provider_ext::gemini::GeminiConfig::new("test-key")
            .with_base_url(base_url.to_string())
            .with_model(model.to_string())
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");
    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("gemini:gemini-pro")
        .expect("build registry language model");

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("Hello").build()])
        .build();

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
    let mut registry_stream = registry_model
        .chat_stream_request(
            ChatRequest::builder()
                .model(model)
                .messages(vec![ChatMessage::user("Hello").build()])
                .build(),
        )
        .await
        .expect("registry stream ok");

    let siumai_events = collect_stream_events(&mut siumai_stream).await;
    let provider_events = collect_stream_events(&mut provider_stream).await;
    let config_events = collect_stream_events(&mut config_stream).await;
    let registry_events = collect_stream_events(&mut registry_stream).await;

    let siumai_starts = custom_events_by_type(&siumai_events, "reasoning-start");
    let provider_starts = custom_events_by_type(&provider_events, "reasoning-start");
    let config_starts = custom_events_by_type(&config_events, "reasoning-start");
    let registry_starts = custom_events_by_type(&registry_events, "reasoning-start");
    assert_eq!(
        siumai_starts.len(),
        1,
        "expected one siumai reasoning-start"
    );
    assert_eq!(
        provider_starts.len(),
        1,
        "expected one provider reasoning-start"
    );
    assert_eq!(
        config_starts.len(),
        1,
        "expected one config reasoning-start"
    );
    assert_eq!(
        registry_starts.len(),
        1,
        "expected one registry reasoning-start"
    );

    for event in [
        &siumai_starts[0],
        &provider_starts[0],
        &config_starts[0],
        &registry_starts[0],
    ] {
        assert_eq!(
            event
                .get("providerMetadata")
                .and_then(|meta| meta.get("google"))
                .and_then(|meta| meta.get("thoughtSignature"))
                .and_then(|value| value.as_str()),
            Some("stream_sig")
        );
        assert!(
            event
                .get("providerMetadata")
                .and_then(|meta| meta.get("vertex"))
                .is_none(),
            "did not expect providerMetadata.vertex on gemini path"
        );
    }

    let collect_reasoning = |events: &[siumai::prelude::unified::ChatStreamEvent]| {
        events
            .iter()
            .filter_map(siumai::prelude::unified::ChatStreamEvent::reasoning_delta)
            .collect::<String>()
    };

    for reasoning in [
        collect_reasoning(&siumai_events),
        collect_reasoning(&provider_events),
        collect_reasoning(&config_events),
        collect_reasoning(&registry_events),
    ] {
        assert_eq!(reasoning, "thinking...");
    }

    let siumai_end = siumai_events
        .iter()
        .find_map(|event| match event {
            siumai::prelude::unified::ChatStreamEvent::StreamEnd { response } => Some(response),
            _ => None,
        })
        .expect("expected siumai StreamEnd");
    let provider_end = provider_events
        .iter()
        .find_map(|event| match event {
            siumai::prelude::unified::ChatStreamEvent::StreamEnd { response } => Some(response),
            _ => None,
        })
        .expect("expected provider StreamEnd");
    let config_end = config_events
        .iter()
        .find_map(|event| match event {
            siumai::prelude::unified::ChatStreamEvent::StreamEnd { response } => Some(response),
            _ => None,
        })
        .expect("expected config StreamEnd");
    let registry_end = registry_events
        .iter()
        .find_map(|event| match event {
            siumai::prelude::unified::ChatStreamEvent::StreamEnd { response } => Some(response),
            _ => None,
        })
        .expect("expected registry StreamEnd");

    for response in [siumai_end, provider_end, config_end, registry_end] {
        assert_eq!(
            response.finish_reason,
            Some(siumai::prelude::unified::FinishReason::Stop)
        );

        let meta = response
            .gemini_metadata()
            .expect("expected gemini metadata");
        assert_eq!(meta.safety_ratings.as_ref().map(Vec::len), Some(1));
        assert_eq!(
            meta.safety_ratings
                .as_ref()
                .and_then(|ratings| ratings.first())
                .map(|rating| rating.category.as_str()),
            Some("HARM_CATEGORY_DEROGATORY")
        );

        let provider_metadata = response
            .provider_metadata
            .as_ref()
            .expect("expected provider_metadata");
        assert!(provider_metadata.contains_key("google"));
        assert!(!provider_metadata.contains_key("vertex"));
    }

    let siumai_req = siumai_transport
        .take_stream()
        .expect("siumai stream request");
    let provider_req = provider_transport
        .take_stream()
        .expect("provider stream request");
    let config_req = config_transport
        .take_stream()
        .expect("config stream request");
    let registry_req = registry_transport
        .take_stream()
        .expect("registry stream request");

    assert_requests_equivalent(&siumai_req, &provider_req);
    assert_requests_equivalent(&siumai_req, &config_req);
    assert_requests_equivalent(&siumai_req, &registry_req);
    assert_eq!(
        siumai_req.url,
        "https://example.com/v1beta/models/gemini-pro:streamGenerateContent?alt=sse"
    );
    assert_eq!(
        header_value(&siumai_req, "accept"),
        Some("text/event-stream".to_string())
    );
}

#[tokio::test]
async fn gemini_siumai_provider_config_stable_request_options_are_equivalent() {
    let siumai_transport = CaptureTransport::default();
    let provider_transport = CaptureTransport::default();
    let config_transport = CaptureTransport::default();

    let model = "gemini-2.5-flash";
    let base_url = "https://example.com/v1beta";
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false
    });

    let siumai_client = Siumai::builder()
        .gemini()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(siumai_transport.clone()))
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::gemini()
        .api_key("test-key")
        .base_url(base_url)
        .model(model)
        .fetch(Arc::new(provider_transport.clone()))
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::gemini::GeminiClient::from_config(
        siumai::provider_ext::gemini::GeminiConfig::new("test-key")
            .with_base_url(base_url.to_string())
            .with_model(model.to_string())
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .tools(vec![Tool::function(
            "get_weather".to_string(),
            "Get weather".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            }),
        )])
        .tool_choice(ToolChoice::None)
        .response_format(ResponseFormat::json_schema(schema.clone()))
        .build()
        .with_gemini_options(
            GeminiOptions::new()
                .with_thinking_config(
                    GeminiThinkingConfig::new()
                        .with_thinking_budget(2048)
                        .with_include_thoughts(true),
                )
                .with_structured_outputs(true),
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
        siumai_req.url,
        "https://example.com/v1beta/models/gemini-2.5-flash:generateContent"
    );
    assert_eq!(
        siumai_req.body["generationConfig"]["thinkingConfig"],
        serde_json::json!({
            "thinkingBudget": 2048,
            "includeThoughts": true
        })
    );
    assert_eq!(
        siumai_req.body["generationConfig"]["responseMimeType"],
        serde_json::json!("application/json")
    );
    assert!(
        siumai_req.body["generationConfig"]
            .get("responseSchema")
            .is_some()
    );
    assert!(
        siumai_req.body["generationConfig"]
            .get("responseJsonSchema")
            .is_none()
    );
}

#[tokio::test]
async fn gemini_registry_stable_request_options_match_config_path() {
    let config_transport = CaptureTransport::default();
    let registry_transport = CaptureTransport::default();

    let model = "gemini-2.5-flash";
    let base_url = "https://example.com/v1beta";
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false
    });

    let config_client = siumai::provider_ext::gemini::GeminiClient::from_config(
        siumai::provider_ext::gemini::GeminiConfig::new("test-key")
            .with_base_url(base_url.to_string())
            .with_model(model.to_string())
            .with_http_transport(Arc::new(config_transport.clone())),
    )
    .expect("build config client");

    let registry = make_registry(Arc::new(registry_transport.clone()), base_url);
    let registry_model = registry
        .language_model("gemini:gemini-2.5-flash")
        .expect("build registry language model");

    let request = ChatRequest::builder()
        .model(model)
        .messages(vec![ChatMessage::user("hi").build()])
        .tools(vec![Tool::function(
            "get_weather".to_string(),
            "Get weather".to_string(),
            serde_json::json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            }),
        )])
        .tool_choice(ToolChoice::None)
        .response_format(ResponseFormat::json_schema(schema.clone()))
        .build()
        .with_gemini_options(
            GeminiOptions::new()
                .with_thinking_config(
                    GeminiThinkingConfig::new()
                        .with_thinking_budget(2048)
                        .with_include_thoughts(true),
                )
                .with_structured_outputs(true),
        );

    let _ = config_client.chat_request(request.clone()).await;
    let _ = registry_model.chat_request(request).await;

    let config_req = config_transport.take().expect("config request");
    let registry_req = registry_transport.take().expect("registry request");

    assert_requests_equivalent(&config_req, &registry_req);
    assert_eq!(
        registry_req.url,
        "https://example.com/v1beta/models/gemini-2.5-flash:generateContent"
    );
    assert_eq!(
        registry_req.body["generationConfig"]["thinkingConfig"],
        serde_json::json!({
            "thinkingBudget": 2048,
            "includeThoughts": true
        })
    );
    assert_eq!(
        registry_req.body["generationConfig"]["responseMimeType"],
        serde_json::json!("application/json")
    );
    assert!(
        registry_req.body["generationConfig"]
            .get("responseSchema")
            .is_some()
    );
    assert!(
        registry_req.body["generationConfig"]
            .get("responseJsonSchema")
            .is_none()
    );
    assert_eq!(
        registry_req.body["toolConfig"],
        serde_json::json!({
            "functionCallingConfig": { "mode": "NONE" }
        })
    );
}

#[tokio::test]
async fn gemini_siumai_provider_config_registry_query_video_task_are_equivalent() {
    async fn mount_video_query_server() -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1beta/operations/test-video-123"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "name": "operations/test-video-123",
                "done": true,
                "response": {
                    "generateVideoResponse": {
                        "generatedSamples": [
                            {
                                "video": {
                                    "uri": "https://example.com/generated/video.mp4",
                                    "mimeType": "video/mp4"
                                }
                            }
                        ]
                    }
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

    let model = "veo-3.1-generate-preview";
    let expected_uri = "https://example.com/generated/video.mp4";

    let siumai_client = Siumai::builder()
        .gemini()
        .api_key("test-key")
        .base_url(format!("{}/v1beta", siumai_server.uri()))
        .model(model)
        .build()
        .await
        .expect("build siumai client");

    let provider_client = Provider::gemini()
        .api_key("test-key")
        .base_url(format!("{}/v1beta", provider_server.uri()))
        .model(model)
        .build()
        .await
        .expect("build provider client");

    let config_client = siumai::provider_ext::gemini::GeminiClient::from_config(
        siumai::provider_ext::gemini::GeminiConfig::new("test-key")
            .with_base_url(format!("{}/v1beta", config_server.uri()))
            .with_model(model.to_string()),
    )
    .expect("build config client");

    let registry = gemini_registry_builder()
        .with_provider_api_key_base_url(
            "gemini",
            "test-key",
            format!("{}/v1beta", registry_server.uri()),
        )
        .build()
        .expect("build gemini registry");
    let registry_client = registry
        .video_model("gemini:veo-3.1-generate-preview")
        .expect("build registry video model");

    let siumai_resp = siumai_client
        .query_video_task("operations/test-video-123")
        .await
        .expect("siumai query video ok");
    let provider_resp = provider_client
        .query_video_task("operations/test-video-123")
        .await
        .expect("provider query video ok");
    let config_resp = config_client
        .query_video_task("operations/test-video-123")
        .await
        .expect("config query video ok");
    let registry_resp = registry_client
        .query_video_task("operations/test-video-123")
        .await
        .expect("registry query video ok");

    for response in [&siumai_resp, &provider_resp, &config_resp, &registry_resp] {
        assert_eq!(response.status.to_string(), "Success");
        assert_eq!(response.file_id.as_deref(), Some(expected_uri));
        assert_eq!(response.video_url, None);
        assert_eq!(
            response
                .provider_reference()
                .and_then(|reference| reference.get("gemini")),
            Some(expected_uri)
        );
        assert_eq!(
            response
                .provider_reference()
                .and_then(|reference| reference.get("google")),
            Some(expected_uri)
        );
        assert_eq!(
            response
                .metadata
                .get("gemini")
                .and_then(|value| value.get("videos"))
                .and_then(|value| value.as_array())
                .map(Vec::len),
            Some(1)
        );
    }

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

    assert_eq!(siumai_req.url.path(), "/v1beta/operations/test-video-123");
    assert_eq!(siumai_req.url.query(), None);
    assert_eq!(provider_req.url.path(), siumai_req.url.path());
    assert_eq!(provider_req.url.query(), siumai_req.url.query());
    assert_eq!(config_req.url.path(), siumai_req.url.path());
    assert_eq!(config_req.url.query(), siumai_req.url.query());
    assert_eq!(registry_req.url.path(), siumai_req.url.path());
    assert_eq!(registry_req.url.query(), siumai_req.url.query());
    assert_eq!(
        wiremock_header_value(&siumai_req, "x-goog-api-key"),
        Some("test-key".to_string())
    );
    assert_eq!(
        wiremock_header_value(&siumai_req, "x-goog-api-key"),
        wiremock_header_value(&provider_req, "x-goog-api-key")
    );
    assert_eq!(
        wiremock_header_value(&siumai_req, "x-goog-api-key"),
        wiremock_header_value(&config_req, "x-goog-api-key")
    );
    assert_eq!(
        wiremock_header_value(&siumai_req, "x-goog-api-key"),
        wiremock_header_value(&registry_req, "x-goog-api-key")
    );
}
