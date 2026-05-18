use async_trait::async_trait;
use std::sync::Arc;

use crate::LlmError;
use crate::streaming::ChatStream;
use crate::traits::ChatCapability;
use crate::types::{ChatMessage, ChatRequest, ChatResponse, Tool};

use super::{GeminiConfig, SharedIdGenerator};
use request::{GoogleInteractionsPreparedRequest, build_interactions_request_body};
use runtime::execute_interactions_non_stream;

mod request;
mod response;
mod runtime;

/// Model selector for the Google Interactions API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GoogleInteractionsModelInput {
    Model(String),
    Agent(String),
}

impl GoogleInteractionsModelInput {
    pub fn model(model_id: impl Into<String>) -> Self {
        Self::Model(model_id.into())
    }

    pub fn agent(agent_name: impl Into<String>) -> Self {
        Self::Agent(agent_name.into())
    }

    pub fn id(&self) -> &str {
        match self {
            Self::Model(model_id) | Self::Agent(model_id) => model_id,
        }
    }

    pub const fn is_agent(&self) -> bool {
        matches!(self, Self::Agent(_))
    }
}

impl From<String> for GoogleInteractionsModelInput {
    fn from(value: String) -> Self {
        Self::Model(value)
    }
}

impl From<&str> for GoogleInteractionsModelInput {
    fn from(value: &str) -> Self {
        Self::Model(value.to_string())
    }
}

/// Provider-owned handle for `google.interactions(...)`.
///
/// This type is intentionally not wired into ordinary Gemini chat execution.
/// The Interactions API uses `/v1beta/interactions`, background agent polling,
/// per-block signatures, and interaction ids, so it owns its request/response
/// runtime instead of reusing `:generateContent`.
#[derive(Clone)]
pub struct GoogleInteractionsLanguageModel {
    config: GeminiConfig,
    model_input: GoogleInteractionsModelInput,
    http_client: reqwest::Client,
    retry_options: Option<crate::retry_api::RetryOptions>,
}

impl std::fmt::Debug for GoogleInteractionsLanguageModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GoogleInteractionsLanguageModel")
            .field("provider", &self.provider())
            .field("base_url", &self.config.base_url)
            .field("model_input", &self.model_input)
            .finish()
    }
}

impl GoogleInteractionsLanguageModel {
    pub fn new(config: GeminiConfig, model_input: impl Into<GoogleInteractionsModelInput>) -> Self {
        Self {
            config,
            model_input: model_input.into(),
            http_client: reqwest::Client::new(),
            retry_options: None,
        }
    }

    pub(crate) fn with_runtime(
        mut self,
        http_client: reqwest::Client,
        retry_options: Option<crate::retry_api::RetryOptions>,
    ) -> Self {
        self.http_client = http_client;
        self.retry_options = retry_options;
        self
    }

    pub fn config(&self) -> &GeminiConfig {
        &self.config
    }

    pub fn model_input(&self) -> &GoogleInteractionsModelInput {
        &self.model_input
    }

    pub fn model_id(&self) -> &str {
        self.model_input.id()
    }

    pub fn agent(&self) -> Option<&str> {
        match &self.model_input {
            GoogleInteractionsModelInput::Agent(agent) => Some(agent),
            GoogleInteractionsModelInput::Model(_) => None,
        }
    }

    pub fn provider(&self) -> String {
        format!("{}.interactions", self.config.provider_name())
    }

    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }

    pub fn generate_id(&self) -> String {
        self.config.generate_id()
    }

    pub(crate) fn prepare_request_body(
        &self,
        request: &ChatRequest,
        stream: bool,
    ) -> Result<GoogleInteractionsPreparedRequest, LlmError> {
        build_interactions_request_body(&self.model_input, request, stream)
    }

    fn unsupported_streaming_runtime_error(&self) -> LlmError {
        LlmError::UnsupportedOperation(
            "google.interactions streaming runtime is not implemented yet; use chat_request for the non-streaming /interactions path"
                .to_string(),
        )
    }
}

#[async_trait]
impl ChatCapability for GoogleInteractionsLanguageModel {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        let mut request = ChatRequest::new(messages);
        if let Some(tools) = tools {
            request.tools = Some(tools);
        }
        self.chat_request(request).await
    }

    async fn chat_stream(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        Err(self.unsupported_streaming_runtime_error())
    }

    async fn chat_request(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        execute_interactions_non_stream(
            self,
            request,
            self.http_client.clone(),
            self.retry_options.clone(),
        )
        .await
    }

    async fn chat_stream_request(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
        Err(self.unsupported_streaming_runtime_error())
    }
}

/// Curated model-id constants for the audited Google Interactions package surface.
#[allow(clippy::module_inception)]
pub mod interactions {
    pub const GEMINI_2_5_COMPUTER_USE_PREVIEW_10_2025: &str =
        "gemini-2.5-computer-use-preview-10-2025";
    pub const GEMINI_2_5_FLASH: &str = "gemini-2.5-flash";
    pub const GEMINI_2_5_FLASH_IMAGE: &str = "gemini-2.5-flash-image";
    pub const GEMINI_2_5_FLASH_LITE: &str = "gemini-2.5-flash-lite";
    pub const GEMINI_2_5_FLASH_LITE_PREVIEW_09_2025: &str = "gemini-2.5-flash-lite-preview-09-2025";
    pub const GEMINI_2_5_FLASH_NATIVE_AUDIO_PREVIEW_12_2025: &str =
        "gemini-2.5-flash-native-audio-preview-12-2025";
    pub const GEMINI_2_5_FLASH_PREVIEW_09_2025: &str = "gemini-2.5-flash-preview-09-2025";
    pub const GEMINI_2_5_FLASH_PREVIEW_TTS: &str = "gemini-2.5-flash-preview-tts";
    pub const GEMINI_2_5_PRO: &str = "gemini-2.5-pro";
    pub const GEMINI_2_5_PRO_PREVIEW_TTS: &str = "gemini-2.5-pro-preview-tts";
    pub const GEMINI_3_FLASH_PREVIEW: &str = "gemini-3-flash-preview";
    pub const GEMINI_3_PRO_IMAGE_PREVIEW: &str = "gemini-3-pro-image-preview";
    pub const GEMINI_3_PRO_PREVIEW: &str = "gemini-3-pro-preview";
    pub const GEMINI_3_1_PRO_PREVIEW: &str = "gemini-3.1-pro-preview";
    pub const GEMINI_3_1_FLASH_IMAGE_PREVIEW: &str = "gemini-3.1-flash-image-preview";
    pub const GEMINI_3_1_FLASH_LITE_PREVIEW: &str = "gemini-3.1-flash-lite-preview";
    pub const GEMINI_3_1_FLASH_TTS_PREVIEW: &str = "gemini-3.1-flash-tts-preview";
    pub const LYRIA_3_CLIP_PREVIEW: &str = "lyria-3-clip-preview";
    pub const LYRIA_3_PRO_PREVIEW: &str = "lyria-3-pro-preview";

    pub const ALL: &[&str] = &[
        GEMINI_2_5_COMPUTER_USE_PREVIEW_10_2025,
        GEMINI_2_5_FLASH,
        GEMINI_2_5_FLASH_IMAGE,
        GEMINI_2_5_FLASH_LITE,
        GEMINI_2_5_FLASH_LITE_PREVIEW_09_2025,
        GEMINI_2_5_FLASH_NATIVE_AUDIO_PREVIEW_12_2025,
        GEMINI_2_5_FLASH_PREVIEW_09_2025,
        GEMINI_2_5_FLASH_PREVIEW_TTS,
        GEMINI_2_5_PRO,
        GEMINI_2_5_PRO_PREVIEW_TTS,
        GEMINI_3_FLASH_PREVIEW,
        GEMINI_3_PRO_IMAGE_PREVIEW,
        GEMINI_3_PRO_PREVIEW,
        GEMINI_3_1_PRO_PREVIEW,
        GEMINI_3_1_FLASH_IMAGE_PREVIEW,
        GEMINI_3_1_FLASH_LITE_PREVIEW,
        GEMINI_3_1_FLASH_TTS_PREVIEW,
        LYRIA_3_CLIP_PREVIEW,
        LYRIA_3_PRO_PREVIEW,
    ];
}

/// Curated agent-name constants for the audited Google Interactions package surface.
pub mod agents {
    pub const DEEP_RESEARCH_PRO_PREVIEW_12_2025: &str = "deep-research-pro-preview-12-2025";
    pub const DEEP_RESEARCH_PREVIEW_04_2026: &str = "deep-research-preview-04-2026";
    pub const DEEP_RESEARCH_MAX_PREVIEW_04_2026: &str = "deep-research-max-preview-04-2026";

    pub const ALL: &[&str] = &[
        DEEP_RESEARCH_PRO_PREVIEW_12_2025,
        DEEP_RESEARCH_PREVIEW_04_2026,
        DEEP_RESEARCH_MAX_PREVIEW_04_2026,
    ];
}

pub(super) fn interactions_config_from_builder_parts(
    mut config: GeminiConfig,
    model_input: GoogleInteractionsModelInput,
    generate_id: Option<SharedIdGenerator>,
    http_client: reqwest::Client,
    retry_options: Option<crate::retry_api::RetryOptions>,
) -> GoogleInteractionsLanguageModel {
    if let Some(generate_id) = generate_id {
        config = config.with_shared_generate_id(generate_id);
    }
    GoogleInteractionsLanguageModel::new(config, model_input)
        .with_runtime(http_client, retry_options)
}

pub(super) fn clone_shared_id_generator(
    generate_id: &Option<SharedIdGenerator>,
) -> Option<SharedIdGenerator> {
    generate_id.as_ref().map(Arc::clone)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::http::transport::{
        HttpTransport, HttpTransportGetRequest, HttpTransportRequest, HttpTransportResponse,
        HttpTransportStreamBody, HttpTransportStreamResponse,
    };
    use crate::provider_options::gemini::{
        GoogleInteractionsResponseFormatEntry, GoogleLanguageModelInteractionsOptions,
    };
    use crate::providers::gemini::ext::request_options::GoogleChatRequestExt;
    use crate::traits::ChatCapability;
    use crate::types::{
        ContentPart, MessageContent, MessageMetadata, MessageRole, ProviderMetadataMap,
        ProviderOptionsMap, ResponseFormat, Tool, ToolChoice, ToolResultContentPart, Warning,
    };
    use async_trait::async_trait;
    use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct InteractionsCaptureTransport {
        post_responses: Arc<Mutex<VecDeque<serde_json::Value>>>,
        get_responses: Arc<Mutex<VecDeque<serde_json::Value>>>,
        posts: Arc<Mutex<Vec<HttpTransportRequest>>>,
        gets: Arc<Mutex<Vec<HttpTransportGetRequest>>>,
    }

    impl InteractionsCaptureTransport {
        fn new(post_responses: Vec<serde_json::Value>) -> Self {
            Self {
                post_responses: Arc::new(Mutex::new(post_responses.into())),
                get_responses: Arc::new(Mutex::new(VecDeque::new())),
                posts: Arc::new(Mutex::new(Vec::new())),
                gets: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn with_get_responses(self, get_responses: Vec<serde_json::Value>) -> Self {
            *self.get_responses.lock().expect("lock get responses") = get_responses.into();
            self
        }

        fn take_posts(&self) -> Vec<HttpTransportRequest> {
            std::mem::take(&mut *self.posts.lock().expect("lock posts"))
        }

        fn take_gets(&self) -> Vec<HttpTransportGetRequest> {
            std::mem::take(&mut *self.gets.lock().expect("lock gets"))
        }

        fn json_response(value: serde_json::Value) -> HttpTransportResponse {
            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
            headers.insert("x-test-response", HeaderValue::from_static("captured"));

            HttpTransportResponse {
                status: 200,
                headers,
                body: serde_json::to_vec(&value).expect("serialize transport response"),
            }
        }
    }

    #[async_trait]
    impl HttpTransport for InteractionsCaptureTransport {
        async fn execute_json(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            self.posts.lock().expect("lock posts").push(request);
            let response = self
                .post_responses
                .lock()
                .expect("lock post responses")
                .pop_front()
                .unwrap_or_else(|| {
                    serde_json::json!({
                        "id": "iact_default",
                        "status": "completed",
                        "model": interactions::GEMINI_2_5_FLASH,
                        "steps": []
                    })
                });
            Ok(Self::json_response(response))
        }

        async fn execute_get(
            &self,
            request: HttpTransportGetRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            self.gets.lock().expect("lock gets").push(request);
            let response = self
                .get_responses
                .lock()
                .expect("lock get responses")
                .pop_front()
                .unwrap_or_else(|| {
                    serde_json::json!({
                        "id": "iact_default",
                        "status": "completed",
                        "agent": agents::DEEP_RESEARCH_PREVIEW_04_2026,
                        "steps": []
                    })
                });
            Ok(Self::json_response(response))
        }

        async fn execute_stream(
            &self,
            _request: HttpTransportRequest,
        ) -> Result<HttpTransportStreamResponse, LlmError> {
            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
            Ok(HttpTransportStreamResponse {
                status: 200,
                headers,
                body: HttpTransportStreamBody::from_bytes(b"data: [DONE]\n\n".to_vec()),
            })
        }
    }

    fn model_handle() -> GoogleInteractionsLanguageModel {
        GoogleInteractionsLanguageModel::new(
            GeminiConfig::new("test-key").with_provider_name("google.generative-ai"),
            GoogleInteractionsModelInput::model(interactions::GEMINI_2_5_FLASH),
        )
    }

    fn model_handle_with_transport(
        transport: InteractionsCaptureTransport,
    ) -> GoogleInteractionsLanguageModel {
        GoogleInteractionsLanguageModel::new(
            GeminiConfig::new("test-key")
                .with_provider_name("google.generative-ai")
                .with_base_url("https://example.com/v1beta".to_string())
                .with_http_transport(Arc::new(transport)),
            GoogleInteractionsModelInput::model(interactions::GEMINI_2_5_FLASH),
        )
    }

    fn agent_handle_with_transport(
        transport: InteractionsCaptureTransport,
    ) -> GoogleInteractionsLanguageModel {
        GoogleInteractionsLanguageModel::new(
            GeminiConfig::new("test-key")
                .with_provider_name("google.generative-ai")
                .with_base_url("https://example.com/v1beta".to_string())
                .with_http_transport(Arc::new(transport)),
            GoogleInteractionsModelInput::agent(agents::DEEP_RESEARCH_PREVIEW_04_2026),
        )
    }

    fn prepared_body_json(
        handle: &GoogleInteractionsLanguageModel,
        request: &ChatRequest,
    ) -> serde_json::Value {
        serde_json::to_value(
            handle
                .prepare_request_body(request, false)
                .expect("prepare interactions request")
                .body,
        )
        .expect("serialize interactions request body")
    }

    #[tokio::test]
    async fn interactions_streaming_runtime_is_explicitly_deferred() {
        let model = GoogleInteractionsLanguageModel::new(
            GeminiConfig::new("test-key").with_provider_name("google.generative-ai"),
            GoogleInteractionsModelInput::agent(agents::DEEP_RESEARCH_PREVIEW_04_2026),
        );

        assert_eq!(model.provider(), "google.generative-ai.interactions");
        assert_eq!(model.agent(), Some(agents::DEEP_RESEARCH_PREVIEW_04_2026));

        let result = model
            .chat_stream_request(ChatRequest::new(vec![ChatMessage::user("hi").build()]))
            .await;
        let Err(err) = result else {
            panic!("streaming runtime should be deferred");
        };

        match err {
            LlmError::UnsupportedOperation(message) => {
                assert!(message.contains("google.interactions streaming runtime"));
            }
            other => panic!("expected UnsupportedOperation, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn google_interactions_non_stream_posts_model_request_and_parses_response() {
        let transport = InteractionsCaptureTransport::new(vec![serde_json::json!({
            "id": "iact_model",
            "status": "completed",
            "model": interactions::GEMINI_2_5_FLASH,
            "service_tier": "priority",
            "steps": [
                {
                    "type": "model_output",
                    "content": [{ "type": "text", "text": "hello" }]
                }
            ],
            "usage": {
                "total_input_tokens": 3,
                "total_output_tokens": 2,
                "total_tokens": 5
            }
        })]);
        let model = model_handle_with_transport(transport.clone());

        let response = model
            .chat_request(ChatRequest::new(vec![ChatMessage::user("hi").build()]))
            .await
            .expect("execute model interactions request");

        assert_eq!(response.id.as_deref(), Some("iact_model"));
        assert_eq!(response.text().as_deref(), Some("hello"));
        assert_eq!(response.service_tier.as_deref(), Some("priority"));
        assert_eq!(
            response.get_metadata("google", "interactionId"),
            Some(&serde_json::json!("iact_model"))
        );
        assert_eq!(
            response
                .response
                .as_ref()
                .and_then(|info| info.headers.get("x-test-response")),
            Some(&"captured".to_string())
        );
        assert!(
            response
                .request
                .as_ref()
                .and_then(|info| info.body.as_ref())
                .is_some_and(|body| body.contains("\"model\":\"gemini-2.5-flash\""))
        );

        let posts = transport.take_posts();
        assert_eq!(posts.len(), 1);
        assert_eq!(posts[0].url, "https://example.com/v1beta/interactions");
        assert_eq!(
            posts[0]
                .headers
                .get("x-goog-api-key")
                .and_then(|value| value.to_str().ok()),
            Some("test-key")
        );
        assert_eq!(
            posts[0]
                .headers
                .get("api-revision")
                .and_then(|value| value.to_str().ok()),
            Some("2026-05-20")
        );
        assert_eq!(posts[0].body["model"], interactions::GEMINI_2_5_FLASH);
        assert!(posts[0].body.get("background").is_none());
        assert!(transport.take_gets().is_empty());
    }

    #[tokio::test]
    async fn google_interactions_non_stream_polls_agent_until_terminal() {
        let transport = InteractionsCaptureTransport::new(vec![serde_json::json!({
            "id": "iact_agent",
            "status": "in_progress",
            "agent": agents::DEEP_RESEARCH_PREVIEW_04_2026,
            "steps": []
        })])
        .with_get_responses(vec![serde_json::json!({
            "id": "iact_agent",
            "status": "completed",
            "agent": agents::DEEP_RESEARCH_PREVIEW_04_2026,
            "steps": [
                {
                    "type": "model_output",
                    "content": [{ "type": "text", "text": "research done" }]
                }
            ]
        })]);
        let model = agent_handle_with_transport(transport.clone());

        let response = model
            .chat_request(
                ChatRequest::new(vec![ChatMessage::user("research").build()])
                    .with_google_interactions_options(
                        GoogleLanguageModelInteractionsOptions::new()
                            .with_polling_timeout_ms(5_000),
                    ),
            )
            .await
            .expect("execute agent interactions request");

        assert_eq!(response.id.as_deref(), Some("iact_agent"));
        assert_eq!(response.text().as_deref(), Some("research done"));

        let posts = transport.take_posts();
        assert_eq!(posts.len(), 1);
        assert_eq!(
            posts[0].body["agent"],
            agents::DEEP_RESEARCH_PREVIEW_04_2026
        );
        assert_eq!(posts[0].body["background"], serde_json::json!(true));

        let gets = transport.take_gets();
        assert_eq!(gets.len(), 1);
        assert_eq!(
            gets[0].url,
            "https://example.com/v1beta/interactions/iact_agent"
        );
        assert_eq!(
            gets[0]
                .headers
                .get("api-revision")
                .and_then(|value| value.to_str().ok()),
            Some("2026-05-20")
        );
    }

    #[tokio::test]
    async fn google_interactions_non_stream_errors_when_polling_without_id() {
        let transport = InteractionsCaptureTransport::new(vec![serde_json::json!({
            "status": "in_progress",
            "agent": agents::DEEP_RESEARCH_PREVIEW_04_2026,
            "steps": []
        })]);
        let model = agent_handle_with_transport(transport.clone());

        let err = model
            .chat_request(ChatRequest::new(vec![
                ChatMessage::user("research").build(),
            ]))
            .await
            .expect_err("polling without id should fail");

        match err {
            LlmError::ParseError(message) => {
                assert!(message.contains("cannot poll a background interaction without an id"));
            }
            other => panic!("expected ParseError, got {other:?}"),
        }
        assert!(transport.take_gets().is_empty());
    }

    #[tokio::test]
    async fn google_interactions_non_stream_times_out_polling_non_terminal_agent() {
        let transport = InteractionsCaptureTransport::new(vec![serde_json::json!({
            "id": "iact_slow",
            "status": "in_progress",
            "agent": agents::DEEP_RESEARCH_PREVIEW_04_2026,
            "steps": []
        })])
        .with_get_responses(vec![serde_json::json!({
            "id": "iact_slow",
            "status": "in_progress",
            "agent": agents::DEEP_RESEARCH_PREVIEW_04_2026,
            "steps": []
        })]);
        let model = agent_handle_with_transport(transport.clone());

        let err = model
            .chat_request(
                ChatRequest::new(vec![ChatMessage::user("research").build()])
                    .with_google_interactions_options(
                        GoogleLanguageModelInteractionsOptions::new().with_polling_timeout_ms(1),
                    ),
            )
            .await
            .expect_err("non-terminal polling should time out");

        match err {
            LlmError::TimeoutError(message) => {
                assert!(message.contains("timed out polling interaction iact_slow"));
            }
            other => panic!("expected TimeoutError, got {other:?}"),
        }
        assert_eq!(transport.take_gets().len(), 1);
    }

    #[test]
    fn google_interactions_request_model_mode_converts_system_and_user_text() {
        let model = model_handle();
        let request = ChatRequest::new(vec![
            ChatMessage::system("You are concise.").build(),
            ChatMessage::user("Summarize this.").build(),
        ]);

        let body = prepared_body_json(&model, &request);

        assert_eq!(
            body["model"],
            serde_json::json!(interactions::GEMINI_2_5_FLASH)
        );
        assert_eq!(
            body["system_instruction"],
            serde_json::json!("You are concise.")
        );
        assert_eq!(
            body["input"],
            serde_json::json!([
                {
                    "type": "user_input",
                    "content": [
                        { "type": "text", "text": "Summarize this." }
                    ]
                }
            ])
        );
        assert!(body.get("agent").is_none());
        assert!(body.get("stream").is_none());
    }

    #[test]
    fn google_interactions_request_maps_model_options_and_response_format() {
        let model = model_handle();
        let request = ChatRequest::builder()
            .message(ChatMessage::user("return json and image").build())
            .temperature(0.2)
            .top_p(0.7)
            .seed(42)
            .max_tokens(128)
            .stop_sequences(vec!["STOP".to_string()])
            .response_format(ResponseFormat::json_schema(serde_json::json!({
                "type": "object",
                "properties": { "answer": { "type": "string" } }
            })))
            .build()
            .with_google_interactions_options(
                GoogleLanguageModelInteractionsOptions::new()
                    .with_store(true)
                    .with_previous_interaction_id("iact_prev")
                    .with_media_resolution("high")
                    .with_response_modalities(["text", "image"])
                    .with_service_tier("priority")
                    .with_thinking_level("high")
                    .with_thinking_summaries("auto")
                    .with_response_format(vec![
                        GoogleInteractionsResponseFormatEntry::image()
                            .with_mime_type("image/png")
                            .with_aspect_ratio("16:9")
                            .with_image_size("1K"),
                    ]),
            );

        let prepared = model
            .prepare_request_body(&request, true)
            .expect("prepare interactions request");
        let body = serde_json::to_value(prepared.body).expect("serialize body");

        assert_eq!(body["store"], serde_json::json!(true));
        assert_eq!(
            body["previous_interaction_id"],
            serde_json::json!("iact_prev")
        );
        assert_eq!(body["service_tier"], serde_json::json!("priority"));
        assert_eq!(
            body["response_modalities"],
            serde_json::json!(["text", "image"])
        );
        assert_eq!(body["stream"], serde_json::json!(true));
        assert_eq!(
            body["generation_config"],
            serde_json::json!({
                "temperature": 0.2,
                "top_p": 0.7,
                "seed": 42,
                "stop_sequences": ["STOP"],
                "max_output_tokens": 128,
                "thinking_level": "high",
                "thinking_summaries": "auto"
            })
        );
        assert_eq!(
            body["response_format"],
            serde_json::json!([
                {
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": {
                        "type": "object",
                        "properties": { "answer": { "type": "string" } }
                    }
                },
                {
                    "type": "image",
                    "mime_type": "image/png",
                    "aspect_ratio": "16:9",
                    "image_size": "1K"
                }
            ])
        );
    }

    #[test]
    fn google_interactions_request_maps_tools_and_tool_choice() {
        let model = model_handle();
        let request = ChatRequest::new(vec![ChatMessage::user("use tools").build()])
            .with_tools(vec![
                Tool::function(
                    "weather",
                    "Get weather",
                    serde_json::json!({
                        "type": "object",
                        "properties": { "city": { "type": "string" } }
                    }),
                ),
                crate::tools::google::google_search().with_args(serde_json::json!({
                    "searchTypes": { "webSearch": true, "imageSearch": true }
                })),
                crate::tools::google::file_search(vec!["stores/main".to_string()]).with_args(
                    serde_json::json!({
                        "fileSearchStoreNames": ["stores/main"],
                        "topK": 5,
                        "metadataFilter": "lang = 'en'"
                    }),
                ),
            ])
            .with_tool_choice(ToolChoice::tool("weather"));

        let body = prepared_body_json(&model, &request);

        assert_eq!(
            body["tools"],
            serde_json::json!([
                {
                    "type": "function",
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": { "city": { "type": "string" } }
                    }
                },
                {
                    "type": "google_search",
                    "search_types": ["web_search", "image_search"]
                },
                {
                    "type": "file_search",
                    "file_search_store_names": ["stores/main"],
                    "top_k": 5,
                    "metadata_filter": "lang = 'en'"
                }
            ])
        );
        assert_eq!(
            body["generation_config"]["tool_choice"],
            serde_json::json!({
                "allowed_tools": {
                    "mode": "validated",
                    "tools": ["weather"]
                }
            })
        );
    }

    #[test]
    fn google_interactions_request_warns_for_unsupported_provider_tool() {
        let model = model_handle();
        let request = ChatRequest::new(vec![ChatMessage::user("search").build()])
            .with_tools(vec![crate::tools::openai::web_search()]);

        let prepared = model
            .prepare_request_body(&request, false)
            .expect("prepare request");
        assert_eq!(prepared.warnings.len(), 1);
        assert!(matches!(prepared.warnings[0], Warning::Unsupported { .. }));
        let body = serde_json::to_value(prepared.body).expect("serialize body");
        assert!(body.get("tools").is_none());
        assert!(body.get("generation_config").is_none());
    }

    #[test]
    fn google_interactions_request_maps_deprecated_image_config_fallback() {
        let model = model_handle();
        let request = ChatRequest::new(vec![ChatMessage::user("make image").build()])
            .with_google_interactions_options(
                GoogleLanguageModelInteractionsOptions::new().with_image_config(
                    crate::provider_options::gemini::GoogleInteractionsImageConfig::new()
                        .with_aspect_ratio("1:1")
                        .with_image_size("512"),
                ),
            );

        let prepared = model
            .prepare_request_body(&request, false)
            .expect("prepare request");
        assert_eq!(prepared.warnings.len(), 1);
        assert!(matches!(prepared.warnings[0], Warning::Deprecated { .. }));
        let body = serde_json::to_value(prepared.body).expect("serialize body");
        assert_eq!(
            body["response_format"],
            serde_json::json!([
                {
                    "type": "image",
                    "mime_type": "image/png",
                    "aspect_ratio": "1:1",
                    "image_size": "512"
                }
            ])
        );
    }

    #[test]
    fn google_interactions_request_converts_files_with_media_resolution() {
        let model = model_handle();
        let request = ChatRequest::new(vec![
            ChatMessage::user("inspect")
                .with_image("https://example.com/cat.png".to_string(), None)
                .with_file_base64(
                    "JVBERi0x",
                    "application/pdf",
                    Some("report.pdf".to_string()),
                )
                .with_file_url("gs://bucket/movie.mp4", "video/mp4")
                .build(),
        ])
        .with_google_interactions_options(
            GoogleLanguageModelInteractionsOptions::new().with_media_resolution("high"),
        );

        let body = prepared_body_json(&model, &request);

        assert_eq!(
            body["input"][0]["content"],
            serde_json::json!([
                { "type": "text", "text": "inspect" },
                {
                    "type": "image",
                    "uri": "https://example.com/cat.png",
                    "resolution": "high"
                },
                {
                    "type": "document",
                    "data": "JVBERi0x",
                    "mime_type": "application/pdf"
                },
                {
                    "type": "video",
                    "uri": "gs://bucket/movie.mp4",
                    "mime_type": "video/mp4",
                    "resolution": "high"
                }
            ])
        );
    }

    #[test]
    fn google_interactions_request_roundtrips_assistant_tool_and_reasoning_signatures() {
        let model = model_handle();
        let google_meta = ProviderMetadataMap::from([(
            "google".to_string(),
            serde_json::json!({
                "signature": "sig_tool",
                "interactionId": "iact_current"
            }),
        )]);
        let reasoning_meta = ProviderMetadataMap::from([(
            "google".to_string(),
            serde_json::json!({
                "signature": "sig_reasoning"
            }),
        )]);
        let assistant = ChatMessage::assistant_with_content(vec![
            ContentPart::text("I will call weather."),
            ContentPart::Reasoning {
                text: "Need live weather.".to_string(),
                provider_options: Default::default(),
                provider_metadata: Some(reasoning_meta),
            },
            ContentPart::ToolCall {
                tool_call_id: "call_weather".to_string(),
                tool_name: "weather".to_string(),
                arguments: serde_json::json!({ "city": "Shanghai" }),
                provider_executed: None,
                dynamic: None,
                invalid: None,
                error: None,
                title: None,
                provider_options: Default::default(),
                provider_metadata: Some(google_meta.clone()),
            },
        ])
        .build();
        let tool = ChatMessage {
            role: MessageRole::Tool,
            content: MessageContent::MultiModal(vec![
                ContentPart::tool_result_content(
                    "call_weather",
                    "weather",
                    vec![
                        ToolResultContentPart::text("22C"),
                        ToolResultContentPart::image_url("https://example.com/weather.png"),
                    ],
                )
                .with_provider_option(
                    "google",
                    serde_json::json!({
                        "signature": "sig_result"
                    }),
                ),
            ]),
            provider_options: ProviderOptionsMap::default(),
            metadata: MessageMetadata::default(),
        };
        let request = ChatRequest::new(vec![assistant, tool]);

        let body = prepared_body_json(&model, &request);

        assert_eq!(
            body["input"],
            serde_json::json!([
                {
                    "type": "model_output",
                    "content": [
                        { "type": "text", "text": "I will call weather." }
                    ]
                },
                {
                    "type": "thought",
                    "signature": "sig_reasoning",
                    "summary": [
                        { "type": "text", "text": "Need live weather." }
                    ]
                },
                {
                    "type": "function_call",
                    "id": "call_weather",
                    "name": "weather",
                    "arguments": { "city": "Shanghai" },
                    "signature": "sig_tool"
                },
                {
                    "type": "user_input",
                    "content": [
                        {
                            "type": "function_result",
                            "call_id": "call_weather",
                            "name": "weather",
                            "result": [
                                { "type": "text", "text": "22C" },
                                {
                                    "type": "image",
                                    "uri": "https://example.com/weather.png"
                                }
                            ],
                            "signature": "sig_result"
                        }
                    ]
                }
            ])
        );
    }

    #[test]
    fn google_interactions_request_compacts_previous_interaction_history() {
        let model = model_handle();
        let linked_meta = ProviderMetadataMap::from([(
            "google".to_string(),
            serde_json::json!({ "interactionId": "iact_prev" }),
        )]);
        let linked_assistant = ChatMessage::assistant_with_content(vec![ContentPart::ToolCall {
            tool_call_id: "call_prev".to_string(),
            tool_name: "search".to_string(),
            arguments: serde_json::json!({ "q": "old" }),
            provider_executed: None,
            dynamic: None,
            invalid: None,
            error: None,
            title: None,
            provider_options: Default::default(),
            provider_metadata: Some(linked_meta),
        }])
        .build();
        let linked_tool =
            ChatMessage::tool_result_text("call_prev", "search", "old result").build();
        let new_user = ChatMessage::user("continue").build();
        let request = ChatRequest::new(vec![linked_assistant, linked_tool, new_user])
            .with_google_interactions_options(
                GoogleLanguageModelInteractionsOptions::new()
                    .with_previous_interaction_id("iact_prev")
                    .with_store(true),
            );

        let body = prepared_body_json(&model, &request);

        assert_eq!(
            body["input"],
            serde_json::json!([
                {
                    "type": "user_input",
                    "content": [
                        { "type": "text", "text": "continue" }
                    ]
                }
            ])
        );
        assert_eq!(
            body["previous_interaction_id"],
            serde_json::json!("iact_prev")
        );
    }

    #[test]
    fn google_interactions_request_warns_and_keeps_history_when_store_false_conflicts() {
        let model = model_handle();
        let request = ChatRequest::new(vec![ChatMessage::user("continue").build()])
            .with_google_interactions_options(
                GoogleLanguageModelInteractionsOptions::new()
                    .with_previous_interaction_id("iact_prev")
                    .with_store(false),
            );

        let prepared = model
            .prepare_request_body(&request, false)
            .expect("prepare request");
        assert_eq!(prepared.warnings.len(), 1);
        assert!(matches!(prepared.warnings[0], Warning::Other { .. }));
        let body = serde_json::to_value(prepared.body).expect("serialize body");
        assert_eq!(body["store"], serde_json::json!(false));
        assert_eq!(
            body["previous_interaction_id"],
            serde_json::json!("iact_prev")
        );
        assert_eq!(
            body["input"][0]["content"][0]["text"],
            serde_json::json!("continue")
        );
    }

    #[test]
    fn google_interactions_request_prefers_prompt_system_over_option_system_instruction() {
        let model = model_handle();
        let request = ChatRequest::new(vec![
            ChatMessage::system("prompt system").build(),
            ChatMessage::user("hi").build(),
        ])
        .with_google_interactions_options(
            GoogleLanguageModelInteractionsOptions::new().with_system_instruction("option system"),
        );

        let prepared = model
            .prepare_request_body(&request, false)
            .expect("prepare request");
        assert_eq!(prepared.warnings.len(), 1);
        let body = serde_json::to_value(prepared.body).expect("serialize body");
        assert_eq!(
            body["system_instruction"],
            serde_json::json!("prompt system")
        );
    }

    #[test]
    fn google_interactions_agent_sets_agent_background_and_agent_config() {
        let model = GoogleInteractionsLanguageModel::new(
            GeminiConfig::new("test-key"),
            GoogleInteractionsModelInput::agent(agents::DEEP_RESEARCH_PREVIEW_04_2026),
        );
        let request = ChatRequest::new(vec![ChatMessage::user("research").build()])
            .with_google_interactions_options(
                GoogleLanguageModelInteractionsOptions::new().with_agent_config(
                    crate::provider_options::gemini::GoogleInteractionsAgentConfig::deep_research()
                        .with_thinking_summaries("auto")
                        .with_visualization("auto")
                        .with_collaborative_planning(true),
                ),
            );

        let prepared = model
            .prepare_request_body(&request, false)
            .expect("prepare agent request");
        assert!(prepared.warnings.is_empty());
        let body = serde_json::to_value(prepared.body).expect("serialize body");

        assert!(body.get("model").is_none());
        assert_eq!(
            body["agent"],
            serde_json::json!(agents::DEEP_RESEARCH_PREVIEW_04_2026)
        );
        assert_eq!(body["background"], serde_json::json!(true));
        assert!(body.get("stream").is_none());
        assert!(body.get("generation_config").is_none());
        assert_eq!(
            body["agent_config"],
            serde_json::json!({
                "type": "deep-research",
                "thinking_summaries": "auto",
                "visualization": "auto",
                "collaborative_planning": true
            })
        );
    }

    #[test]
    fn google_interactions_agent_warns_and_drops_model_only_fields() {
        let model = GoogleInteractionsLanguageModel::new(
            GeminiConfig::new("test-key"),
            GoogleInteractionsModelInput::agent(agents::DEEP_RESEARCH_PREVIEW_04_2026),
        );
        let request = ChatRequest::builder()
            .message(ChatMessage::user("research").build())
            .temperature(0.7)
            .top_p(0.8)
            .seed(7)
            .max_tokens(256)
            .stop_sequences(vec!["STOP".to_string()])
            .response_format(ResponseFormat::json_schema(serde_json::json!({
                "type": "object"
            })))
            .tools(vec![Tool::function(
                "lookup",
                "Lookup data",
                serde_json::json!({ "type": "object" }),
            )])
            .build()
            .with_google_interactions_options(
                GoogleLanguageModelInteractionsOptions::new()
                    .with_thinking_level("high")
                    .with_thinking_summaries("auto")
                    .with_response_format(vec![GoogleInteractionsResponseFormatEntry::image()])
                    .with_image_config(
                        crate::provider_options::gemini::GoogleInteractionsImageConfig::new()
                            .with_aspect_ratio("1:1"),
                    ),
            )
            .with_tool_choice(ToolChoice::Required);

        let prepared = model
            .prepare_request_body(&request, true)
            .expect("prepare agent request");
        assert_eq!(prepared.warnings.len(), 4);
        let body = serde_json::to_value(prepared.body).expect("serialize body");

        assert_eq!(
            body["agent"],
            serde_json::json!(agents::DEEP_RESEARCH_PREVIEW_04_2026)
        );
        assert_eq!(body["background"], serde_json::json!(true));
        assert!(body.get("generation_config").is_none());
        assert!(body.get("response_format").is_none());
        assert!(body.get("tools").is_none());
        assert!(body.get("stream").is_none());
    }
}
