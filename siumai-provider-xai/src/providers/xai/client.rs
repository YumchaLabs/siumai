//! `xAI` client.
//!
//! Provider-owned client wrapper around the OpenAI-compatible backend.

use crate::client::LlmClient;
use crate::core::ProviderContext;
use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::http::transport::HttpTransport;
use crate::retry_api::RetryOptions;
use crate::streaming::ChatStream;
use crate::traits::{
    AudioCapability, ChatCapability, EmbeddingCapability, FileManagementCapability, ImageExtras,
    ImageGenerationCapability, ModelListingCapability, RerankCapability, SpeechCapability,
    SpeechExtras, TranscriptionCapability, TranscriptionExtras, VideoGenerationCapability,
};
use crate::types::{
    ChatMessage, ChatRequest, ChatResponse, EmbeddingRequest, EmbeddingResponse,
    FileDeleteResponse, FileListQuery, FileListResponse, FileObject, FileUploadRequest,
    ImageEditRequest, ImageGenerationRequest, ImageGenerationResponse, ImageVariationRequest,
    ModelInfo, ProviderOptionsMap, RerankRequest, RerankResponse, Tool, VideoGenerationRequest,
    VideoGenerationResponse, VideoTaskStatusResponse,
};
use async_trait::async_trait;
use siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleClient;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct XaiClient {
    inner: OpenAiCompatibleClient,
    default_provider_options_map: ProviderOptionsMap,
}

impl XaiClient {
    pub fn new(
        inner: OpenAiCompatibleClient,
        default_provider_options_map: ProviderOptionsMap,
    ) -> Self {
        Self {
            inner,
            default_provider_options_map,
        }
    }

    /// Construct an `XaiClient` from `XaiConfig`.
    pub async fn from_config(config: super::XaiConfig) -> Result<Self, LlmError> {
        let default_provider_options_map = config.default_provider_options_map.clone();
        let inner = OpenAiCompatibleClient::from_config(config.into_compatible_config()?).await?;
        Ok(Self::new(inner, default_provider_options_map))
    }

    /// Construct an `XaiClient` from `XaiConfig` with a caller-supplied HTTP client.
    pub async fn with_http_client(
        config: super::XaiConfig,
        http_client: reqwest::Client,
    ) -> Result<Self, LlmError> {
        let default_provider_options_map = config.default_provider_options_map.clone();
        let inner =
            OpenAiCompatibleClient::with_http_client(config.into_compatible_config()?, http_client)
                .await?;
        Ok(Self::new(inner, default_provider_options_map))
    }

    /// Construct an `XaiClient` from env (`XAI_API_KEY`).
    pub async fn from_env(model: Option<&str>) -> Result<Self, LlmError> {
        let model = model.unwrap_or(super::models::grok_2::GROK_2_1212);
        let config = super::XaiConfig::from_env()?.with_model(model);
        Self::from_config(config).await
    }

    /// Alias kept for parity with provider-owned wrappers.
    pub async fn from_builtin_env(model: Option<&str>) -> Result<Self, LlmError> {
        Self::from_env(model).await
    }

    pub fn inner(&self) -> &OpenAiCompatibleClient {
        &self.inner
    }

    pub fn provider_context(&self) -> ProviderContext {
        self.inner.provider_context()
    }

    pub fn base_url(&self) -> String {
        self.provider_context().base_url
    }

    pub fn http_client(&self) -> reqwest::Client {
        self.inner.http_client()
    }

    pub fn retry_options(&self) -> Option<RetryOptions> {
        self.inner.retry_options()
    }

    pub fn http_interceptors(&self) -> Vec<Arc<dyn HttpInterceptor>> {
        self.inner.http_interceptors()
    }

    pub fn http_transport(&self) -> Option<Arc<dyn HttpTransport>> {
        self.inner.http_transport()
    }

    pub(crate) fn files(&self) -> super::files::XaiFiles {
        super::files::XaiFiles::new(
            self.provider_context(),
            self.http_client(),
            self.http_transport(),
            self.http_interceptors(),
            self.retry_options(),
        )
    }

    pub fn set_retry_options(&mut self, options: Option<RetryOptions>) {
        self.inner.set_retry_options(options);
    }

    pub(crate) fn merge_default_provider_options_map_non_chat(
        &self,
        request_map: &mut ProviderOptionsMap,
    ) {
        let mut merged = self.default_provider_options_map.clone();
        merged.merge_overrides(request_map.clone());
        *request_map = merged;
    }
}

impl crate::traits::ModelMetadata for XaiClient {
    fn provider_id(&self) -> &str {
        "xai"
    }

    fn model_id(&self) -> &str {
        self.inner.model()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::http::transport::{
        HttpTransport, HttpTransportMultipartRequest, HttpTransportRequest, HttpTransportResponse,
        HttpTransportStreamBody, HttpTransportStreamResponse,
    };
    use crate::provider_metadata::xai::XaiChatResponseExt;
    use crate::provider_options::{
        RssSearchSource, SearchMode, WebSearchSource, XSearchSource, XaiOptions,
        XaiResponsesOptions, XaiSearchParameters,
    };
    use crate::providers::xai::ext::XaiChatRequestExt;
    use crate::types::ToolChoice;
    use async_trait::async_trait;
    use futures_util::StreamExt;
    use reqwest::header::{ACCEPT, CONTENT_TYPE, HeaderMap, HeaderValue};
    use std::sync::Mutex;

    #[derive(Clone, Default)]
    struct CaptureTransport {
        last: Arc<Mutex<Option<HttpTransportRequest>>>,
    }

    impl CaptureTransport {
        fn take(&self) -> Option<HttpTransportRequest> {
            self.last.lock().unwrap().take()
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
                body: br#"{"error":{"message":"unauthorized","type":"auth_error","code":"unauthorized"}}"#
                    .to_vec(),
            })
        }
    }

    #[derive(Clone)]
    struct JsonResponseTransport {
        body: serde_json::Value,
        last: Arc<Mutex<Option<HttpTransportRequest>>>,
    }

    impl JsonResponseTransport {
        fn new(body: serde_json::Value) -> Self {
            Self {
                body,
                last: Arc::new(Mutex::new(None)),
            }
        }

        fn take(&self) -> Option<HttpTransportRequest> {
            self.last.lock().unwrap().take()
        }
    }

    #[async_trait]
    impl HttpTransport for JsonResponseTransport {
        async fn execute_json(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            *self.last.lock().unwrap() = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportResponse {
                status: 200,
                headers,
                body: serde_json::to_vec(&self.body).expect("serialize response body"),
            })
        }
    }

    #[derive(Clone, Default)]
    struct MultipartResponseTransport {
        last: Arc<Mutex<Option<HttpTransportMultipartRequest>>>,
    }

    impl MultipartResponseTransport {
        fn take(&self) -> Option<HttpTransportMultipartRequest> {
            self.last.lock().unwrap().take()
        }
    }

    #[async_trait]
    impl HttpTransport for MultipartResponseTransport {
        async fn execute_json(
            &self,
            _request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            Err(LlmError::UnsupportedOperation(
                "json transport should not be used in xAI file upload tests".to_string(),
            ))
        }

        async fn execute_multipart(
            &self,
            request: HttpTransportMultipartRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            *self.last.lock().unwrap() = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
            Ok(HttpTransportResponse {
                status: 200,
                headers,
                body: serde_json::to_vec(&serde_json::json!({
                    "id": "file-123",
                    "bytes": 3,
                    "created_at": 1,
                    "filename": "hello.txt",
                    "status": "uploaded"
                }))
                .expect("serialize response body"),
            })
        }
    }

    #[derive(Clone)]
    struct SseResponseTransport {
        body: Arc<Vec<u8>>,
        last_stream: Arc<Mutex<Option<HttpTransportRequest>>>,
    }

    impl SseResponseTransport {
        fn new(body: impl Into<Vec<u8>>) -> Self {
            Self {
                body: Arc::new(body.into()),
                last_stream: Arc::new(Mutex::new(None)),
            }
        }

        fn take_stream(&self) -> Option<HttpTransportRequest> {
            self.last_stream.lock().unwrap().take()
        }
    }

    #[async_trait]
    impl HttpTransport for SseResponseTransport {
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
            *self.last_stream.lock().unwrap() = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));

            Ok(HttpTransportStreamResponse {
                status: 200,
                headers,
                body: HttpTransportStreamBody::from_bytes(self.body.as_ref().clone()),
            })
        }
    }

    #[tokio::test]
    async fn file_upload_merges_default_provider_options_map() {
        let transport = Arc::new(MultipartResponseTransport::default());
        let mut provider_options = ProviderOptionsMap::default();
        provider_options.insert(
            "xai",
            serde_json::json!({
                "teamId": "team-default",
                "filePath": "/defaults/hello.txt"
            }),
        );
        let config = super::super::XaiConfig::new("test-api-key")
            .with_model("grok-2")
            .with_http_transport(transport.clone())
            .with_provider_options_map(provider_options);
        let client = XaiClient::from_config(config).await.expect("xai client");

        let request = FileUploadRequest {
            content: b"hey".to_vec(),
            filename: Some("hello.txt".to_string()),
            mime_type: Some("text/plain".to_string()),
            purpose: "assistants".to_string(),
            metadata: Default::default(),
            provider_options: Default::default(),
            http_config: None,
        };

        let result = client.upload_file(request).await.expect("upload result");
        assert_eq!(result.id, "file-123");

        let request = transport.take().expect("multipart request");
        let body = String::from_utf8_lossy(&request.body);
        assert!(body.contains("name=\"team_id\""));
        assert!(body.contains("team-default"));
        assert!(body.contains("name=\"file_path\""));
        assert!(body.contains("/defaults/hello.txt"));
    }

    #[tokio::test]
    async fn xai_client_from_config_builds_inner_client() {
        let cfg = super::super::config::XaiConfig::new("test-key").with_model("grok-4");
        let client = XaiClient::from_config(cfg).await.expect("from_config ok");
        assert_eq!(client.provider_id(), std::borrow::Cow::Borrowed("xai"));
    }

    #[tokio::test]
    async fn xai_client_with_http_client_preserves_provider_context() {
        let cfg = super::super::config::XaiConfig::new("test-key")
            .with_base_url("https://example.com/custom/")
            .with_model("grok-4");
        let client = XaiClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("with_http_client ok");
        assert_eq!(client.base_url(), "https://example.com/custom/");
    }

    #[tokio::test]
    async fn xai_client_chat_request_preserves_provider_options_and_normalization() {
        let transport = CaptureTransport::default();
        let cfg = super::super::config::XaiConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_model("grok-4")
            .with_http_transport(Arc::new(transport.clone()));
        let client = XaiClient::from_config(cfg).await.expect("from_config ok");

        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()]).with_provider_option(
            "xai",
            serde_json::json!({
                "reasoningEffort": "high",
                "searchParameters": {
                    "returnCitations": true,
                    "maxSearchResults": 3,
                    "sources": [
                        {
                            "type": "web",
                            "allowedWebsites": ["example.com"],
                            "safeSearch": true
                        }
                    ]
                }
            }),
        );

        let _ = client.chat_request(request).await;
        let captured = transport.take().expect("captured request");
        assert_eq!(captured.body["reasoning_effort"], serde_json::json!("high"));
        assert_eq!(
            captured.body["search_parameters"]["return_citations"],
            serde_json::json!(true)
        );
        assert_eq!(
            captured.body["search_parameters"]["max_search_results"],
            serde_json::json!(3)
        );
        assert_eq!(
            captured.body["search_parameters"]["sources"][0]["allowed_websites"],
            serde_json::json!(["example.com"])
        );
        assert_eq!(
            captured.body["search_parameters"]["sources"][0]["safe_search"],
            serde_json::json!(true)
        );
        assert!(captured.body.get("reasoningEffort").is_none());
        assert!(captured.body.get("searchParameters").is_none());
    }

    #[tokio::test]
    async fn xai_client_chat_request_typed_options_normalize_on_request_path() {
        let transport = CaptureTransport::default();
        let cfg = super::super::config::XaiConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_model("grok-4")
            .with_http_transport(Arc::new(transport.clone()));
        let client = XaiClient::from_config(cfg).await.expect("from_config ok");

        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()]).with_xai_options(
            XaiOptions::new()
                .with_reasoning_effort("high")
                .with_parallel_function_calling(false)
                .with_search(XaiSearchParameters {
                    mode: SearchMode::On,
                    return_citations: Some(true),
                    max_search_results: Some(3),
                    from_date: Some("2025-01-01".to_string()),
                    to_date: Some("2025-01-31".to_string()),
                    sources: Some(vec![
                        WebSearchSource {
                            country: Some("US".to_string()),
                            allowed_websites: Some(vec!["example.com".to_string()]),
                            excluded_websites: None,
                            safe_search: Some(true),
                        }
                        .into(),
                    ]),
                }),
        );

        let _ = client.chat_request(request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(captured.body["reasoning_effort"], serde_json::json!("high"));
        assert_eq!(
            captured.body["parallel_function_calling"],
            serde_json::json!(false)
        );
        assert_eq!(
            captured.body["search_parameters"]["mode"],
            serde_json::json!("on")
        );
        assert_eq!(
            captured.body["search_parameters"]["return_citations"],
            serde_json::json!(true)
        );
        assert_eq!(
            captured.body["search_parameters"]["max_search_results"],
            serde_json::json!(3)
        );
        assert_eq!(
            captured.body["search_parameters"]["from_date"],
            serde_json::json!("2025-01-01")
        );
        assert_eq!(
            captured.body["search_parameters"]["to_date"],
            serde_json::json!("2025-01-31")
        );
        assert_eq!(
            captured.body["search_parameters"]["sources"][0]["allowed_websites"],
            serde_json::json!(["example.com"])
        );
        assert_eq!(
            captured.body["search_parameters"]["sources"][0]["safe_search"],
            serde_json::json!(true)
        );
        assert!(captured.body.get("reasoningEffort").is_none());
        assert!(captured.body.get("searchParameters").is_none());
    }

    #[tokio::test]
    async fn xai_client_chat_request_typed_search_source_variants_serialize() {
        let transport = CaptureTransport::default();
        let cfg = super::super::config::XaiConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_model("grok-4")
            .with_http_transport(Arc::new(transport.clone()));
        let client = XaiClient::from_config(cfg).await.expect("from_config ok");

        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()]).with_xai_options(
            XaiOptions::new().with_search(XaiSearchParameters {
                mode: SearchMode::On,
                return_citations: Some(true),
                max_search_results: Some(3),
                from_date: None,
                to_date: None,
                sources: Some(vec![
                    XSearchSource {
                        excluded_x_handles: Some(vec!["spam".to_string()]),
                        included_x_handles: Some(vec![
                            "openai".to_string(),
                            "deepmind".to_string(),
                        ]),
                        post_favorite_count: Some(10),
                        post_view_count: Some(99),
                    }
                    .into(),
                    RssSearchSource::new(["https://example.com/feed.xml"]).into(),
                ]),
            }),
        );

        let _ = client.chat_request(request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(
            captured.body["search_parameters"]["sources"][0]["type"],
            serde_json::json!("x")
        );
        assert_eq!(
            captured.body["search_parameters"]["sources"][0]["included_x_handles"],
            serde_json::json!(["openai", "deepmind"])
        );
        assert_eq!(
            captured.body["search_parameters"]["sources"][0]["excluded_x_handles"],
            serde_json::json!(["spam"])
        );
        assert_eq!(
            captured.body["search_parameters"]["sources"][0]["post_favorite_count"],
            serde_json::json!(10)
        );
        assert_eq!(
            captured.body["search_parameters"]["sources"][0]["post_view_count"],
            serde_json::json!(99)
        );
        assert!(
            captured.body["search_parameters"]["sources"][0]
                .get("x_handles")
                .is_none()
        );
        assert_eq!(
            captured.body["search_parameters"]["sources"][1]["type"],
            serde_json::json!("rss")
        );
        assert_eq!(
            captured.body["search_parameters"]["sources"][1]["links"],
            serde_json::json!(["https://example.com/feed.xml"])
        );
    }

    #[tokio::test]
    async fn xai_client_chat_request_normalizes_legacy_x_handles_alias() {
        let transport = CaptureTransport::default();
        let cfg = super::super::config::XaiConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_model("grok-4")
            .with_http_transport(Arc::new(transport.clone()));
        let client = XaiClient::from_config(cfg).await.expect("from_config ok");

        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()]).with_provider_option(
            "xai",
            serde_json::json!({
                "searchParameters": {
                    "mode": "on",
                    "sources": [
                        {
                            "type": "x",
                            "xHandles": ["openai", "deepmind"],
                            "excludedXHandles": ["grok"]
                        }
                    ]
                }
            }),
        );

        let _ = client.chat_request(request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(
            captured.body["search_parameters"]["sources"][0]["included_x_handles"],
            serde_json::json!(["openai", "deepmind"])
        );
        assert_eq!(
            captured.body["search_parameters"]["sources"][0]["excluded_x_handles"],
            serde_json::json!(["grok"])
        );
        assert!(
            captured.body["search_parameters"]["sources"][0]
                .get("x_handles")
                .is_none()
        );
    }

    #[tokio::test]
    async fn xai_client_chat_request_normalizes_logprobs_and_drops_responses_only_fields() {
        let transport = CaptureTransport::default();
        let cfg = super::super::config::XaiConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_model("grok-4")
            .with_http_transport(Arc::new(transport.clone()));
        let client = XaiClient::from_config(cfg).await.expect("from_config ok");

        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()]).with_xai_options(
            XaiResponsesOptions::new()
                .with_reasoning_summary("detailed")
                .with_top_logprobs(2)
                .with_store(false)
                .with_previous_response("resp_prev_123")
                .with_include(["file_search_call.results"]),
        );

        let _ = client.chat_request(request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(captured.body["top_logprobs"], serde_json::json!(2));
        assert_eq!(captured.body["logprobs"], serde_json::json!(true));
        assert!(captured.body.get("reasoning_summary").is_none());
        assert!(captured.body.get("previous_response_id").is_none());
        assert!(captured.body.get("include").is_none());
        assert!(captured.body.get("store").is_none());
    }

    #[tokio::test]
    async fn xai_client_chat_request_drops_stop_and_preserves_stable_response_format() {
        let transport = CaptureTransport::default();
        let cfg = super::super::config::XaiConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_model("grok-4")
            .with_http_transport(Arc::new(transport.clone()));
        let client = XaiClient::from_config(cfg).await.expect("from_config ok");

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let request = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .model("grok-4")
            .stop_sequences(vec!["END".to_string()])
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

        let _ = client.chat_request(request).await;
        let captured = transport.take().expect("captured request");

        assert!(captured.body.get("stop").is_none());
        assert_eq!(captured.body["reasoning_effort"], serde_json::json!("high"));
        assert_eq!(
            captured.body["response_format"],
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            })
        );
    }

    #[tokio::test]
    async fn xai_client_chat_request_keeps_stable_tool_choice_over_raw_provider_options() {
        let transport = CaptureTransport::default();
        let cfg = super::super::config::XaiConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_model("grok-4")
            .with_http_transport(Arc::new(transport.clone()));
        let client = XaiClient::from_config(cfg).await.expect("from_config ok");

        let request = ChatRequest::builder()
            .messages(vec![ChatMessage::user("hi").build()])
            .model("grok-4")
            .tools(vec![Tool::function(
                "get_weather",
                "Get weather",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .tool_choice(ToolChoice::None)
            .build()
            .with_provider_option(
                "xai",
                serde_json::json!({
                    "tool_choice": "auto",
                    "reasoningEffort": "high"
                }),
            );

        let _ = client.chat_request(request).await;
        let captured = transport.take().expect("captured request");

        assert_eq!(captured.body["tool_choice"], serde_json::json!("none"));
        assert_eq!(captured.body["reasoning_effort"], serde_json::json!("high"));
    }

    #[tokio::test]
    async fn xai_client_chat_stream_request_preserves_typed_search_options_and_stable_fields_at_transport_boundary()
     {
        let transport = SseResponseTransport::new(b"data: [DONE]\n\n".to_vec());
        let cfg = super::super::config::XaiConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_model("grok-4")
            .with_http_transport(Arc::new(transport.clone()));
        let client = XaiClient::from_config(cfg).await.expect("from_config ok");

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let mut request = ChatRequest::builder()
            .model("grok-4")
            .messages(vec![ChatMessage::user("hi").build()])
            .tools(vec![Tool::function(
                "get_weather",
                "Get weather",
                serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }),
            )])
            .tool_choice(ToolChoice::None)
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                schema.clone(),
            ))
            .build()
            .with_xai_options(XaiOptions::new().with_reasoning_effort("high").with_search(
                XaiSearchParameters {
                    mode: SearchMode::Auto,
                    return_citations: Some(true),
                    max_search_results: Some(3),
                    from_date: Some("2025-01-01".to_string()),
                    to_date: Some("2025-01-31".to_string()),
                    sources: Some(vec![WebSearchSource {
                        country: Some("US".to_string()),
                        allowed_websites: Some(vec!["example.com".to_string()]),
                        excluded_websites: Some(vec!["blocked.example.com".to_string()]),
                        safe_search: Some(true),
                    }
                    .into()]),
                },
            ));

        request
            .provider_options_map
            .0
            .get_mut("xai")
            .and_then(|value| value.as_object_mut())
            .expect("xai provider options object")
            .extend([
                ("tool_choice".to_string(), serde_json::json!("auto")),
                (
                    "response_format".to_string(),
                    serde_json::json!({ "type": "json_object" }),
                ),
            ]);

        let _stream = client
            .chat_stream_request(request)
            .await
            .expect("stream ok");
        let captured = transport.take_stream().expect("captured stream request");

        assert_eq!(
            captured
                .headers
                .get(ACCEPT)
                .and_then(|value| value.to_str().ok()),
            Some("text/event-stream")
        );
        assert_eq!(captured.body["stream"], serde_json::json!(true));
        assert_eq!(
            captured.body["stream_options"],
            serde_json::json!({ "include_usage": true })
        );
        assert_eq!(captured.body["reasoning_effort"], serde_json::json!("high"));
        assert_eq!(
            captured.body["search_parameters"]["mode"],
            serde_json::json!("auto")
        );
        assert_eq!(
            captured.body["search_parameters"]["from_date"],
            serde_json::json!("2025-01-01")
        );
        assert_eq!(
            captured.body["search_parameters"]["to_date"],
            serde_json::json!("2025-01-31")
        );
        assert_eq!(
            captured.body["search_parameters"]["sources"][0]["allowed_websites"],
            serde_json::json!(["example.com"])
        );
        assert_eq!(
            captured.body["search_parameters"]["sources"][0]["excluded_websites"],
            serde_json::json!(["blocked.example.com"])
        );
        assert_eq!(captured.body["tool_choice"], serde_json::json!("none"));
        assert_eq!(
            captured.body["response_format"],
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            })
        );
    }

    #[tokio::test]
    async fn xai_client_from_config_merges_typed_provider_defaults() {
        let transport = JsonResponseTransport::new(serde_json::json!({
            "id": "chatcmpl-xai-test",
            "object": "chat.completion",
            "created": 1_741_392_000,
            "model": "grok-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "hello"
                    },
                    "finish_reason": "stop"
                }
            ]
        }));
        let cfg = super::super::config::XaiConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_model("grok-4")
            .with_http_transport(Arc::new(transport.clone()))
            .with_reasoning(true)
            .with_reasoning_budget(2048)
            .with_reasoning_effort("high")
            .with_default_search();
        let client = XaiClient::from_config(cfg).await.expect("from_config ok");

        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()]);
        let _ = client.chat_request(request).await.expect("chat ok");
        let captured = transport.take().expect("captured request");

        assert_eq!(captured.body["reasoning_effort"], serde_json::json!("high"));
        assert!(captured.body.get("enable_reasoning").is_none());
        assert!(captured.body.get("reasoning_budget").is_none());
        assert!(captured.body.get("enable_thinking").is_none());
        assert!(captured.body.get("thinking_budget").is_none());
        assert_eq!(
            captured.body["search_parameters"]["mode"],
            serde_json::json!("auto")
        );
        assert_eq!(
            captured.body["search_parameters"]["return_citations"],
            serde_json::json!(true)
        );
        assert_eq!(
            captured.body["search_parameters"]["max_search_results"],
            serde_json::json!(20)
        );
    }

    #[tokio::test]
    async fn xai_client_exposes_typed_response_metadata() {
        let transport = JsonResponseTransport::new(serde_json::json!({
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
        }));
        let cfg = super::super::config::XaiConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_model("grok-4")
            .with_http_transport(Arc::new(transport.clone()));
        let client = XaiClient::from_config(cfg).await.expect("from_config ok");

        let request = ChatRequest::builder()
            .model("grok-4")
            .messages(vec![ChatMessage::user("hi").build()])
            .build();

        let response = client.chat_request(request).await.expect("response ok");
        let captured = transport.take().expect("captured request");

        assert_eq!(captured.body["model"], serde_json::json!("grok-4"));
        assert_eq!(response.content_text(), Some("hello from xai"));

        let meta = response.xai_metadata().expect("xai metadata");
        assert_eq!(meta.sources.as_ref().map(Vec::len), Some(1));
        assert_eq!(
            meta.sources
                .as_ref()
                .and_then(|sources| sources.first())
                .map(|source| source.url.as_str()),
            Some("https://example.com")
        );
        assert_eq!(
            meta.logprobs,
            Some(serde_json::json!([
                {
                    "token": "hello",
                    "logprob": -0.1,
                    "bytes": [104, 101, 108, 108, 111],
                    "top_logprobs": []
                }
            ]))
        );
    }

    #[tokio::test]
    async fn xai_client_chat_stream_exposes_typed_response_metadata() {
        let transport = SseResponseTransport::new(
            br#"data: {"id":"1","model":"grok-4","created":1718345013,"sources":[{"id":"src_1","source_type":"url","url":"https://example.com","title":"Example"}],"choices":[{"index":0,"delta":{"content":"hello","role":"assistant"},"finish_reason":null}]}


data: {"id":"1","model":"grok-4","created":1718345013,"choices":[{"index":0,"delta":{"content":" from xai","role":null},"finish_reason":"stop","logprobs":{"content":[{"token":"hello","logprob":-0.1,"bytes":[104,101,108,108,111],"top_logprobs":[]}]}}]}


data: [DONE]

"#
                .to_vec(),
        );
        let cfg = super::super::config::XaiConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_model("grok-4")
            .with_http_transport(Arc::new(transport.clone()));
        let client = XaiClient::from_config(cfg).await.expect("from_config ok");

        let request = ChatRequest::builder()
            .model("grok-4")
            .messages(vec![ChatMessage::user("hi").build()])
            .build();

        let stream = client
            .chat_stream_request(request)
            .await
            .expect("stream ok");
        let events = stream.collect::<Vec<_>>().await;
        let captured = transport.take_stream().expect("captured stream request");

        assert_eq!(captured.body["model"], serde_json::json!("grok-4"));
        assert_eq!(
            captured.body["stream_options"],
            serde_json::json!({ "include_usage": true })
        );

        let end = events
            .iter()
            .find_map(|event| match event {
                Ok(crate::types::ChatStreamEvent::StreamEnd { response }) => Some(response),
                _ => None,
            })
            .expect("stream end event");
        assert_eq!(end.content_text(), Some("hello from xai"));

        let meta = end.xai_metadata().expect("xai metadata");
        assert_eq!(meta.sources.as_ref().map(Vec::len), Some(1));
        assert_eq!(
            meta.sources
                .as_ref()
                .and_then(|sources| sources.first())
                .map(|source| source.url.as_str()),
            Some("https://example.com")
        );
        assert_eq!(
            meta.logprobs,
            Some(serde_json::json!([
                {
                    "token": "hello",
                    "logprob": -0.1,
                    "bytes": [104, 101, 108, 108, 111],
                    "top_logprobs": []
                }
            ]))
        );
    }
}

#[async_trait]
impl ChatCapability for XaiClient {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        self.inner.chat_with_tools(messages, tools).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        self.inner.chat_stream(messages, tools).await
    }

    async fn chat_request(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        self.inner.chat_request(request).await
    }

    async fn chat_stream_request(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        self.inner.chat_stream_request(request).await
    }
}

#[async_trait]
impl EmbeddingCapability for XaiClient {
    async fn embed(&self, _texts: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "xAI does not currently expose embedding on the provider-owned xAI client path"
                .to_string(),
        ))
    }

    fn embedding_dimension(&self) -> usize {
        0
    }
}

#[async_trait]
impl crate::traits::EmbeddingExtensions for XaiClient {
    async fn embed_with_config(
        &self,
        _request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "xAI does not currently expose embedding on the provider-owned xAI client path"
                .to_string(),
        ))
    }
}

#[async_trait]
impl RerankCapability for XaiClient {
    async fn rerank(&self, _request: RerankRequest) -> Result<RerankResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "xAI does not currently expose rerank on the provider-owned xAI client path"
                .to_string(),
        ))
    }
}

#[async_trait]
impl ModelListingCapability for XaiClient {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.inner.list_models().await
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        self.inner.get_model(model_id).await
    }
}

#[async_trait]
impl FileManagementCapability for XaiClient {
    async fn upload_file(&self, mut request: FileUploadRequest) -> Result<FileObject, LlmError> {
        self.merge_default_provider_options_map_non_chat(&mut request.provider_options);
        self.files().upload_file(request).await
    }

    async fn list_files(&self, query: Option<FileListQuery>) -> Result<FileListResponse, LlmError> {
        self.files().list_files(query).await
    }

    async fn retrieve_file(&self, file_id: String) -> Result<FileObject, LlmError> {
        self.files().retrieve_file(file_id).await
    }

    async fn delete_file(&self, file_id: String) -> Result<FileDeleteResponse, LlmError> {
        self.files().delete_file(file_id).await
    }

    async fn get_file_content(&self, file_id: String) -> Result<Vec<u8>, LlmError> {
        self.files().get_file_content(file_id).await
    }
}

#[async_trait]
impl ImageGenerationCapability for XaiClient {
    async fn generate_images(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        super::image::generate_images(self, request).await
    }

    fn max_images_per_call(&self) -> Option<u32> {
        Some(3)
    }
}

#[async_trait]
impl ImageExtras for XaiClient {
    async fn edit_image(
        &self,
        request: ImageEditRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        super::image::edit_image(self, request).await
    }

    async fn create_variation(
        &self,
        request: ImageVariationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        super::image::create_variation(self, request).await
    }

    fn get_supported_sizes(&self) -> Vec<String> {
        Vec::new()
    }

    fn get_supported_formats(&self) -> Vec<String> {
        vec!["b64_json".to_string()]
    }

    fn supports_image_editing(&self) -> bool {
        true
    }

    fn supports_image_variations(&self) -> bool {
        false
    }
}

#[async_trait]
impl VideoGenerationCapability for XaiClient {
    async fn create_video_task(
        &self,
        request: VideoGenerationRequest,
    ) -> Result<VideoGenerationResponse, LlmError> {
        super::video::create_video_task(self, request).await
    }

    async fn query_video_task(&self, task_id: &str) -> Result<VideoTaskStatusResponse, LlmError> {
        super::video::query_video_task(self, task_id).await
    }

    fn polling_options(
        &self,
        request: &VideoGenerationRequest,
    ) -> Result<siumai_core::video::VideoPollingOptions, LlmError> {
        super::video::polling_options(self, request)
    }

    fn max_videos_per_call(&self) -> Option<u32> {
        Some(1)
    }

    fn get_supported_models(&self) -> Vec<String> {
        super::video::supported_models()
    }

    fn get_supported_resolutions(&self, model: &str) -> Vec<String> {
        super::video::supported_resolutions(model)
    }

    fn get_supported_durations(&self, model: &str) -> Vec<u32> {
        super::video::supported_durations(model)
    }
}

impl LlmClient for XaiClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        crate::client::LlmClient::provider_id(&self.inner)
    }

    fn supported_models(&self) -> Vec<String> {
        crate::client::LlmClient::supported_models(&self.inner)
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_vision()
            .with_image_generation()
            .with_speech()
            .with_file_management()
            .with_custom_feature("video", true)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }

    fn as_chat_capability(&self) -> Option<&dyn ChatCapability> {
        if self.inner.as_chat_capability().is_some() {
            Some(self)
        } else {
            None
        }
    }

    fn as_embedding_capability(&self) -> Option<&dyn EmbeddingCapability> {
        None
    }

    fn as_audio_capability(&self) -> Option<&dyn AudioCapability> {
        Some(self)
    }

    fn as_speech_capability(&self) -> Option<&dyn SpeechCapability> {
        Some(self)
    }

    fn as_speech_extras(&self) -> Option<&dyn SpeechExtras> {
        None
    }

    fn as_transcription_capability(&self) -> Option<&dyn TranscriptionCapability> {
        None
    }

    fn as_transcription_extras(&self) -> Option<&dyn TranscriptionExtras> {
        None
    }

    fn as_image_generation_capability(&self) -> Option<&dyn ImageGenerationCapability> {
        Some(self)
    }

    fn as_image_extras(&self) -> Option<&dyn ImageExtras> {
        Some(self)
    }

    fn as_file_management_capability(&self) -> Option<&dyn FileManagementCapability> {
        Some(self)
    }

    fn as_rerank_capability(&self) -> Option<&dyn RerankCapability> {
        None
    }

    fn as_model_listing_capability(&self) -> Option<&dyn ModelListingCapability> {
        Some(self)
    }

    fn as_video_generation_capability(&self) -> Option<&dyn VideoGenerationCapability> {
        Some(self)
    }
}
