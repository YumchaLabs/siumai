use async_trait::async_trait;
use std::sync::Arc;

use crate::LlmError;
use crate::streaming::{ChatStream, ChatStreamHandle};
use crate::traits::ChatCapability;
use crate::types::{ChatMessage, ChatRequest, ChatResponse, Tool};

use super::{GeminiConfig, SharedIdGenerator};
use request::{GoogleInteractionsPreparedRequest, build_interactions_request_body};
use runtime::execute_interactions_non_stream;
use stream::{
    execute_interactions_agent_stream, execute_interactions_agent_stream_handle,
    execute_interactions_stream,
};

mod request;
mod response;
mod runtime;
mod stream;

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
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        let mut request = ChatRequest::new(messages);
        if let Some(tools) = tools {
            request.tools = Some(tools);
        }
        self.chat_stream_request(request).await
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

    async fn chat_stream_request(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        if self.model_input().is_agent() {
            return execute_interactions_agent_stream(
                self,
                request,
                self.http_client.clone(),
                self.retry_options.clone(),
                None,
            )
            .await;
        }

        execute_interactions_stream(
            self,
            request,
            self.http_client.clone(),
            self.retry_options.clone(),
        )
        .await
    }

    async fn chat_stream_with_cancel(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStreamHandle, LlmError> {
        let mut request = ChatRequest::new(messages);
        if let Some(tools) = tools {
            request.tools = Some(tools);
        }
        self.chat_stream_request_with_cancel(request).await
    }

    async fn chat_stream_request_with_cancel(
        &self,
        request: ChatRequest,
    ) -> Result<ChatStreamHandle, LlmError> {
        if self.model_input().is_agent() {
            return execute_interactions_agent_stream_handle(
                self,
                request,
                self.http_client.clone(),
                self.retry_options.clone(),
            )
            .await;
        }

        let stream = self.chat_stream_request(request).await?;
        let (cancellable, cancel) = crate::utils::cancel::make_cancellable_stream(stream);
        Ok(ChatStreamHandle {
            stream: cancellable,
            cancel,
        })
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
    use futures_util::StreamExt;
    use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct InteractionsCaptureTransport {
        post_responses: Arc<Mutex<VecDeque<serde_json::Value>>>,
        stream_responses: Arc<Mutex<VecDeque<String>>>,
        get_stream_responses: Arc<Mutex<VecDeque<String>>>,
        get_responses: Arc<Mutex<VecDeque<serde_json::Value>>>,
        posts: Arc<Mutex<Vec<HttpTransportRequest>>>,
        streams: Arc<Mutex<Vec<HttpTransportRequest>>>,
        gets: Arc<Mutex<Vec<HttpTransportGetRequest>>>,
        get_streams: Arc<Mutex<Vec<HttpTransportGetRequest>>>,
    }

    impl InteractionsCaptureTransport {
        fn new(post_responses: Vec<serde_json::Value>) -> Self {
            Self {
                post_responses: Arc::new(Mutex::new(post_responses.into())),
                stream_responses: Arc::new(Mutex::new(VecDeque::new())),
                get_stream_responses: Arc::new(Mutex::new(VecDeque::new())),
                get_responses: Arc::new(Mutex::new(VecDeque::new())),
                posts: Arc::new(Mutex::new(Vec::new())),
                streams: Arc::new(Mutex::new(Vec::new())),
                gets: Arc::new(Mutex::new(Vec::new())),
                get_streams: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn with_get_responses(self, get_responses: Vec<serde_json::Value>) -> Self {
            *self.get_responses.lock().expect("lock get responses") = get_responses.into();
            self
        }

        fn with_stream_response(self, stream_response: String) -> Self {
            self.stream_responses
                .lock()
                .expect("lock stream responses")
                .push_back(stream_response);
            self
        }

        fn with_get_stream_response(self, stream_response: String) -> Self {
            self.get_stream_responses
                .lock()
                .expect("lock get stream responses")
                .push_back(stream_response);
            self
        }

        fn take_posts(&self) -> Vec<HttpTransportRequest> {
            std::mem::take(&mut *self.posts.lock().expect("lock posts"))
        }

        fn take_streams(&self) -> Vec<HttpTransportRequest> {
            std::mem::take(&mut *self.streams.lock().expect("lock streams"))
        }

        fn take_gets(&self) -> Vec<HttpTransportGetRequest> {
            std::mem::take(&mut *self.gets.lock().expect("lock gets"))
        }

        fn take_get_streams(&self) -> Vec<HttpTransportGetRequest> {
            std::mem::take(&mut *self.get_streams.lock().expect("lock get streams"))
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

        async fn execute_get_stream(
            &self,
            request: HttpTransportGetRequest,
        ) -> Result<HttpTransportStreamResponse, LlmError> {
            self.get_streams
                .lock()
                .expect("lock get streams")
                .push(request);
            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
            headers.insert("x-test-stream", HeaderValue::from_static("captured"));
            let body = self
                .get_stream_responses
                .lock()
                .expect("lock get stream responses")
                .pop_front()
                .unwrap_or_else(|| "data: [DONE]\n\n".to_string());
            Ok(HttpTransportStreamResponse {
                status: 200,
                headers,
                body: HttpTransportStreamBody::from_bytes(body.into_bytes()),
            })
        }

        async fn execute_stream(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportStreamResponse, LlmError> {
            self.streams.lock().expect("lock streams").push(request);
            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
            headers.insert("x-test-stream", HeaderValue::from_static("captured"));
            let body = self
                .stream_responses
                .lock()
                .expect("lock stream responses")
                .pop_front()
                .unwrap_or_else(|| "data: [DONE]\n\n".to_string());
            Ok(HttpTransportStreamResponse {
                status: 200,
                headers,
                body: HttpTransportStreamBody::from_bytes(body.into_bytes()),
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
    async fn google_interactions_stream_agent_posts_background_and_streams() {
        let transport = InteractionsCaptureTransport::new(vec![serde_json::json!({
            "id": "iact_agent_direct",
            "status": "in_progress",
            "agent": agents::DEEP_RESEARCH_PREVIEW_04_2026,
            "steps": []
        })])
        .with_get_stream_response(sse_event(serde_json::json!({
            "event_type": "interaction.created",
            "event_id": "evt_1",
            "interaction": {
                "id": "iact_agent_direct",
                "agent": agents::DEEP_RESEARCH_PREVIEW_04_2026
            }
        })))
        .with_get_stream_response(format!(
            "{}{}{}",
            sse_event(serde_json::json!({
                "event_type": "step.start",
                "event_id": "evt_2",
                "index": 0,
                "step": { "type": "model_output" }
            })),
            sse_event(serde_json::json!({
                "event_type": "step.delta",
                "event_id": "evt_3",
                "index": 0,
                "delta": { "type": "text", "text": "hello agent" }
            })),
            sse_event(serde_json::json!({
                "event_type": "interaction.completed",
                "event_id": "evt_4",
                "interaction": {
                    "id": "iact_agent_direct",
                    "status": "completed"
                }
            })),
        ));
        let model = agent_handle_with_transport(transport.clone());

        let events = collect_stream_events(
            model
                .chat_stream_request(ChatRequest::new(vec![ChatMessage::user("hi").build()]))
                .await
                .expect("execute agent interactions stream"),
        )
        .await;

        assert_eq!(
            events
                .iter()
                .filter_map(|event| event.text_delta())
                .collect::<Vec<_>>(),
            vec!["hello agent"]
        );
        let posts = transport.take_posts();
        assert_eq!(posts.len(), 1);
        assert_eq!(
            posts[0].body["agent"],
            agents::DEEP_RESEARCH_PREVIEW_04_2026
        );
        assert_eq!(posts[0].body["background"], serde_json::json!(true));
        assert!(posts[0].body.get("stream").is_none());
        assert_eq!(transport.take_get_streams().len(), 2);
    }

    async fn collect_stream_events(mut stream: ChatStream) -> Vec<crate::types::ChatStreamEvent> {
        use futures_util::StreamExt;

        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.expect("collect google interactions stream event"));
        }
        events
    }

    fn sse_event(value: serde_json::Value) -> String {
        format!(
            "data: {}\n\n",
            serde_json::to_string(&value).expect("serialize SSE event")
        )
    }

    #[tokio::test]
    async fn google_interactions_stream_posts_model_request_and_converts_text_finish() {
        let transport = InteractionsCaptureTransport::new(vec![]).with_stream_response(format!(
            "{}{}{}{}{}",
            sse_event(serde_json::json!({
                "event_type": "interaction.created",
                "event_id": "evt_1",
                "interaction": {
                    "id": "iact_stream",
                    "created": "2026-05-18T00:00:00Z",
                    "model": interactions::GEMINI_2_5_FLASH
                }
            })),
            sse_event(serde_json::json!({
                "event_type": "step.start",
                "event_id": "evt_2",
                "index": 0,
                "step": { "type": "model_output" }
            })),
            sse_event(serde_json::json!({
                "event_type": "step.delta",
                "event_id": "evt_3",
                "index": 0,
                "delta": { "type": "text", "text": "hello" }
            })),
            sse_event(serde_json::json!({
                "event_type": "step.stop",
                "event_id": "evt_4",
                "index": 0
            })),
            sse_event(serde_json::json!({
                "event_type": "interaction.completed",
                "event_id": "evt_5",
                "interaction": {
                    "id": "iact_stream",
                    "status": "completed",
                    "usage": {
                        "total_input_tokens": 3,
                        "total_output_tokens": 2,
                        "total_tokens": 5
                    },
                    "service_tier": "priority"
                }
            })),
        ));
        let model = model_handle_with_transport(transport.clone());

        let events = collect_stream_events(
            model
                .chat_stream_request(ChatRequest::new(vec![ChatMessage::user("hi").build()]))
                .await
                .expect("execute interactions stream"),
        )
        .await;

        assert_eq!(
            events
                .iter()
                .filter_map(|event| event.text_delta())
                .collect::<Vec<_>>(),
            vec!["hello"]
        );
        assert!(events.iter().any(|event| {
            matches!(
                event.part_ref(),
                Some(crate::types::ChatStreamPart::TextStart { id, .. })
                    if id == "iact_stream:0"
            )
        }));
        assert!(events.iter().any(|event| {
            matches!(
                event.part_ref(),
                Some(crate::types::ChatStreamPart::TextEnd {
                    id,
                    provider_metadata,
                }) if id == "iact_stream:0"
                    && provider_metadata
                        .as_ref()
                        .and_then(|metadata| metadata.get("google"))
                        .and_then(|google| google.get("interactionId"))
                        == Some(&serde_json::json!("iact_stream"))
            )
        }));
        assert!(events.iter().any(|event| {
            matches!(
                event.part_ref(),
                Some(crate::types::ChatStreamPart::Finish {
                    usage,
                    finish_reason,
                    provider_metadata,
                }) if usage.prompt_tokens() == Some(3)
                    && usage.completion_tokens() == Some(2)
                    && finish_reason.unified == crate::types::FinishReason::Stop
                    && finish_reason.raw.as_deref() == Some("completed")
                    && provider_metadata
                        .as_ref()
                        .and_then(|metadata| metadata.get("google"))
                        .and_then(|google| google.get("serviceTier"))
                        == Some(&serde_json::json!("priority"))
            )
        }));
        let end = events
            .iter()
            .find_map(|event| match event {
                crate::types::ChatStreamEvent::StreamEnd { response } => Some(response),
                _ => None,
            })
            .expect("stream end");
        assert_eq!(end.id.as_deref(), Some("iact_stream"));
        assert_eq!(end.finish_reason, Some(crate::types::FinishReason::Stop));
        assert!(end.request.as_ref().is_some_and(|info| {
            info.body
                .as_deref()
                .is_some_and(|body| body.contains("\"stream\":true"))
        }));
        assert_eq!(
            end.response
                .as_ref()
                .and_then(|info| info.headers.get("x-test-stream")),
            Some(&"captured".to_string())
        );

        let streams = transport.take_streams();
        assert_eq!(streams.len(), 1);
        assert_eq!(streams[0].url, "https://example.com/v1beta/interactions");
        assert_eq!(
            streams[0]
                .headers
                .get("api-revision")
                .and_then(|value| value.to_str().ok()),
            Some("2026-05-20")
        );
        assert_eq!(streams[0].body["stream"], serde_json::json!(true));
        assert_eq!(streams[0].body["model"], interactions::GEMINI_2_5_FLASH);
        assert!(transport.take_posts().is_empty());
    }

    #[tokio::test]
    async fn google_interactions_stream_converts_reasoning_tools_sources_and_images() {
        let transport = InteractionsCaptureTransport::new(vec![]).with_stream_response(format!(
            "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",
            sse_event(serde_json::json!({
                "event_type": "interaction.created",
                "interaction": {
                    "id": "iact_rich",
                    "model": interactions::GEMINI_2_5_FLASH_IMAGE
                }
            })),
            sse_event(serde_json::json!({
                "event_type": "step.start",
                "index": 0,
                "step": {
                    "type": "thought",
                    "signature": "sig_start",
                    "summary": [{ "type": "text", "text": "plan" }]
                }
            })),
            sse_event(serde_json::json!({
                "event_type": "step.delta",
                "index": 0,
                "delta": {
                    "type": "thought_summary",
                    "content": { "type": "text", "text": " call weather" }
                }
            })),
            sse_event(serde_json::json!({
                "event_type": "step.delta",
                "index": 0,
                "delta": { "type": "thought_signature", "signature": "sig_final" }
            })),
            sse_event(serde_json::json!({
                "event_type": "step.stop",
                "index": 0
            })),
            sse_event(serde_json::json!({
                "event_type": "step.start",
                "index": 1,
                "step": {
                    "type": "function_call",
                    "id": "call_weather",
                    "name": "weather"
                }
            })),
            sse_event(serde_json::json!({
                "event_type": "step.delta",
                "index": 1,
                "delta": {
                    "type": "arguments_delta",
                    "arguments": "{\"city\":\"Sh",
                    "signature": "sig_call"
                }
            })),
            sse_event(serde_json::json!({
                "event_type": "step.delta",
                "index": 1,
                "delta": {
                    "type": "arguments_delta",
                    "arguments": "anghai\"}"
                }
            })),
            sse_event(serde_json::json!({
                "event_type": "step.stop",
                "index": 1
            })),
            sse_event(serde_json::json!({
                "event_type": "step.start",
                "index": 2,
                "step": {
                    "type": "google_search_result",
                    "call_id": "search_1",
                    "result": [
                        { "url": "https://example.com/a", "title": "A" },
                        { "url": "https://example.com/a", "title": "A duplicate" }
                    ]
                }
            })),
            sse_event(serde_json::json!({
                "event_type": "step.stop",
                "index": 2
            })),
            sse_event(serde_json::json!({
                "event_type": "step.start",
                "index": 3,
                "step": { "type": "model_output" }
            })),
            sse_event(serde_json::json!({
                "event_type": "step.delta",
                "index": 3,
                "delta": {
                    "type": "image",
                    "mime_type": "image/png",
                    "data": "aGVsbG8="
                }
            })),
            sse_event(serde_json::json!({
                "event_type": "step.stop",
                "index": 3
            })),
            sse_event(serde_json::json!({
                "event_type": "interaction.completed",
                "interaction": {
                    "id": "iact_rich",
                    "status": "completed"
                }
            })),
            "data: [DONE]\n\n"
        ));
        let model = model_handle_with_transport(transport);

        let events = collect_stream_events(
            model
                .chat_stream_request(ChatRequest::new(vec![
                    ChatMessage::user("use tools").build(),
                ]))
                .await
                .expect("execute interactions stream"),
        )
        .await;

        assert_eq!(
            events
                .iter()
                .filter_map(|event| event.reasoning_delta())
                .collect::<Vec<_>>(),
            vec!["plan", " call weather"]
        );
        assert!(events.iter().any(|event| {
            matches!(
                event.part_ref(),
                Some(crate::types::ChatStreamPart::ReasoningEnd {
                    provider_metadata,
                    ..
                }) if provider_metadata
                    .as_ref()
                    .and_then(|metadata| metadata.get("google"))
                    .and_then(|google| google.get("signature"))
                    == Some(&serde_json::json!("sig_final"))
            )
        }));
        assert_eq!(
            events
                .iter()
                .filter_map(|event| match event.part_ref() {
                    Some(crate::types::ChatStreamPart::ToolInputDelta { delta, .. }) => {
                        Some(delta.as_str())
                    }
                    _ => None,
                })
                .collect::<Vec<_>>(),
            vec!["{\"city\":\"Sh", "anghai\"}"]
        );
        assert!(events.iter().any(|event| {
            matches!(
                event.part_ref(),
                Some(crate::types::ChatStreamPart::ToolCall(call))
                    if call.tool_call_id == "call_weather"
                        && call.tool_name == "weather"
                        && call.input == "{\"city\":\"Shanghai\"}"
                        && call.provider_metadata
                            .as_ref()
                            .and_then(|metadata| metadata.get("google"))
                            .and_then(|google| google.get("signature"))
                            == Some(&serde_json::json!("sig_call"))
            )
        }));
        assert!(events.iter().any(|event| {
            matches!(
                event.part_ref(),
                Some(crate::types::ChatStreamPart::ToolResult(result))
                    if result.tool_call_id == "search_1"
                        && result.tool_name == "google_search"
                        && result.result.is_array()
            )
        }));
        let sources = events
            .iter()
            .filter_map(|event| match event.part_ref() {
                Some(crate::types::ChatStreamPart::Source { id, source, .. }) => {
                    Some((id.as_str(), source))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(sources.len(), 1);
        assert!(matches!(
            sources[0].1,
            crate::types::SourcePart::Url { url, title }
                if url == "https://example.com/a" && title.as_deref() == Some("A")
        ));
        assert!(events.iter().any(|event| {
            matches!(
                event.part_ref(),
                Some(crate::types::ChatStreamPart::File(file))
                    if file.media_type == "image/png"
                        && matches!(
                            &file.data,
                            crate::types::ChatStreamFileData::Base64(data) if data == "aGVsbG8="
                        )
            )
        }));
        assert!(events.iter().any(|event| {
            matches!(
                event.part_ref(),
                Some(crate::types::ChatStreamPart::Finish { finish_reason, .. })
                    if finish_reason.unified == crate::types::FinishReason::ToolCalls
            )
        }));
    }

    #[tokio::test]
    async fn google_interactions_stream_reconnects_agent_with_last_event_id() {
        let transport = InteractionsCaptureTransport::new(vec![serde_json::json!({
            "id": "iact_agent_stream",
            "status": "in_progress",
            "agent": agents::DEEP_RESEARCH_PREVIEW_04_2026,
            "steps": []
        })])
        .with_get_stream_response(sse_event(serde_json::json!({
            "event_type": "interaction.created",
            "event_id": "evt_1",
            "interaction": {
                "id": "iact_agent_stream",
                "agent": agents::DEEP_RESEARCH_PREVIEW_04_2026
            }
        })))
        .with_get_stream_response(format!(
            "{}{}{}{}",
            sse_event(serde_json::json!({
                "event_type": "step.start",
                "event_id": "evt_2",
                "index": 0,
                "step": { "type": "model_output" }
            })),
            sse_event(serde_json::json!({
                "event_type": "step.delta",
                "event_id": "evt_3",
                "index": 0,
                "delta": { "type": "text", "text": "done" }
            })),
            sse_event(serde_json::json!({
                "event_type": "step.stop",
                "event_id": "evt_4",
                "index": 0
            })),
            sse_event(serde_json::json!({
                "event_type": "interaction.completed",
                "event_id": "evt_5",
                "interaction": {
                    "id": "iact_agent_stream",
                    "status": "completed"
                }
            })),
        ));
        let model = agent_handle_with_transport(transport.clone());

        let events = collect_stream_events(
            model
                .chat_stream_request(ChatRequest::new(vec![
                    ChatMessage::user("research").build(),
                ]))
                .await
                .expect("execute agent interactions stream"),
        )
        .await;

        assert_eq!(
            events
                .iter()
                .filter_map(|event| event.text_delta())
                .collect::<Vec<_>>(),
            vec!["done"]
        );
        let posts = transport.take_posts();
        assert_eq!(posts.len(), 1);
        assert_eq!(
            posts[0].body["agent"],
            agents::DEEP_RESEARCH_PREVIEW_04_2026
        );
        assert_eq!(posts[0].body["background"], serde_json::json!(true));
        assert!(posts[0].body.get("stream").is_none());

        let get_streams = transport.take_get_streams();
        assert_eq!(get_streams.len(), 2);
        assert_eq!(
            get_streams[0].url,
            "https://example.com/v1beta/interactions/iact_agent_stream?stream=true"
        );
        assert_eq!(
            get_streams[1].url,
            "https://example.com/v1beta/interactions/iact_agent_stream?stream=true&last_event_id=evt_1"
        );
        assert_eq!(
            get_streams[1]
                .headers
                .get("api-revision")
                .and_then(|value| value.to_str().ok()),
            Some("2026-05-20")
        );
    }

    #[tokio::test]
    async fn google_interactions_stream_empty_agent_get_obeys_retry_budget() {
        let transport = InteractionsCaptureTransport::new(vec![serde_json::json!({
            "id": "iact_empty",
            "status": "in_progress",
            "agent": agents::DEEP_RESEARCH_PREVIEW_04_2026,
            "steps": []
        })])
        .with_get_stream_response(String::new())
        .with_get_stream_response(String::new())
        .with_get_stream_response(String::new());
        let model = agent_handle_with_transport(transport.clone());

        let mut stream = model
            .chat_stream_request(ChatRequest::new(vec![
                ChatMessage::user("research").build(),
            ]))
            .await
            .expect("open agent interactions stream");

        let mut last_error = None;
        while let Some(item) = stream.next().await {
            if let Err(err) = item {
                last_error = Some(err);
                break;
            }
        }

        let err = last_error.expect("empty stream should error after retry budget");
        match err {
            LlmError::StreamError(message) => {
                assert!(message.contains("closed without producing any events"));
            }
            other => panic!("expected StreamError, got {other:?}"),
        }
        assert_eq!(transport.take_get_streams().len(), 3);
    }

    #[tokio::test]
    async fn google_interactions_stream_cancel_posts_best_effort_cancel() {
        let pending_stream = futures_util::stream::pending::<Result<Vec<u8>, LlmError>>();
        let transport = InteractionsCaptureTransport::new(vec![serde_json::json!({
            "id": "iact_cancel",
            "status": "in_progress",
            "agent": agents::DEEP_RESEARCH_PREVIEW_04_2026,
            "steps": []
        })]);
        transport
            .get_stream_responses
            .lock()
            .expect("lock get stream responses")
            .push_back(String::new());
        let model = agent_handle_with_transport(transport.clone());

        let mut handle = model
            .chat_stream_request_with_cancel(ChatRequest::new(vec![
                ChatMessage::user("research").build(),
            ]))
            .await
            .expect("open cancellable agent stream");

        handle.cancel.cancel();
        let _ = tokio::time::timeout(std::time::Duration::from_millis(200), handle.stream.next())
            .await
            .expect("cancel should wake stream");

        drop(pending_stream);
        let posts = transport.take_posts();
        assert_eq!(posts.len(), 2);
        assert_eq!(
            posts[1].url,
            "https://example.com/v1beta/interactions/iact_cancel/cancel"
        );
        assert_eq!(posts[1].body, serde_json::json!({}));
    }

    #[tokio::test]
    async fn google_interactions_stream_errors_when_agent_post_lacks_id() {
        let transport = InteractionsCaptureTransport::new(vec![serde_json::json!({
            "status": "in_progress",
            "agent": agents::DEEP_RESEARCH_PREVIEW_04_2026,
            "steps": []
        })]);
        let model = agent_handle_with_transport(transport.clone());

        let result = model
            .chat_stream_request(ChatRequest::new(vec![
                ChatMessage::user("research").build(),
            ]))
            .await;
        let Err(err) = result else {
            panic!("missing interaction id should fail");
        };

        match err {
            LlmError::ParseError(message) => {
                assert!(
                    message.contains("background POST response did not include an interaction id")
                );
            }
            other => panic!("expected ParseError, got {other:?}"),
        }
        assert!(transport.take_get_streams().is_empty());
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
