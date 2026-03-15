//! `Groq` hybrid client implementation.
//!
//! - Chat/streaming/tools reuse the OpenAI-compatible vendor client.
//! - Audio (TTS/STT) uses Groq's OpenAI-like audio endpoints via `GroqSpec`.

use crate::client::LlmClient;
use crate::core::ProviderContext;
use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::http::transport::HttpTransport;
use crate::retry_api::RetryOptions;
use crate::streaming::ChatStream;
use crate::traits::{
    AudioCapability, ChatCapability, EmbeddingCapability, ImageExtras, ImageGenerationCapability,
    ModelListingCapability, RerankCapability,
};
use crate::types::{ChatMessage, ChatRequest, ChatResponse, ModelInfo, Tool};
use async_trait::async_trait;
use siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleClient;
use siumai_provider_openai_compatible::providers::openai_compatible::{
    ConfigurableAdapter, OpenAiCompatibleConfig, ProviderAdapter, get_provider_config,
};
use std::sync::Arc;

mod audio;

#[derive(Clone, Debug)]
pub struct GroqClient {
    inner: OpenAiCompatibleClient,
}

impl GroqClient {
    pub fn new(inner: OpenAiCompatibleClient) -> Self {
        Self { inner }
    }

    /// Construct a `GroqClient` from a `GroqConfig` (config-first construction).
    pub async fn from_config(config: super::GroqConfig) -> Result<Self, LlmError> {
        config.validate()?;

        let http_interceptors = config.http_interceptors.clone();
        let model_middlewares = config.model_middlewares.clone();
        let provider_specific_config = config.provider_specific_config.clone();

        let provider = get_provider_config("groq").ok_or_else(|| {
            LlmError::ConfigurationError("OpenAI-compatible provider config not found: groq".into())
        })?;
        let mut adapter: Box<dyn ProviderAdapter> = Box::new(ConfigurableAdapter::new(provider));
        if !provider_specific_config.is_empty() {
            adapter = Box::new(
                siumai_provider_openai_compatible::providers::openai_compatible::adapter::ParamMergingAdapter::new(
                    adapter,
                    provider_specific_config,
                ),
            );
        }
        let adapter: Arc<dyn ProviderAdapter> = Arc::from(adapter);

        use secrecy::ExposeSecret;
        let mut openai_cfg = OpenAiCompatibleConfig::new(
            "groq",
            config.api_key.expose_secret(),
            &config.base_url,
            adapter,
        )
        .with_http_config(config.http_config.clone())
        .with_common_params(config.common_params.clone())
        .with_model(&config.common_params.model)
        .with_http_interceptors(http_interceptors)
        .with_model_middlewares(model_middlewares);

        if let Some(transport) = config.http_transport.clone() {
            openai_cfg = openai_cfg.with_http_transport(transport);
        }

        let inner = OpenAiCompatibleClient::from_config(openai_cfg).await?;
        Ok(Self::new(inner))
    }

    /// Construct a `GroqClient` from a `GroqConfig` with a caller-supplied HTTP client.
    pub async fn with_http_client(
        config: super::GroqConfig,
        http_client: reqwest::Client,
    ) -> Result<Self, LlmError> {
        config.validate()?;

        let http_interceptors = config.http_interceptors.clone();
        let model_middlewares = config.model_middlewares.clone();
        let provider_specific_config = config.provider_specific_config.clone();

        let provider = get_provider_config("groq").ok_or_else(|| {
            LlmError::ConfigurationError("OpenAI-compatible provider config not found: groq".into())
        })?;
        let mut adapter: Box<dyn ProviderAdapter> = Box::new(ConfigurableAdapter::new(provider));
        if !provider_specific_config.is_empty() {
            adapter = Box::new(
                siumai_provider_openai_compatible::providers::openai_compatible::adapter::ParamMergingAdapter::new(
                    adapter,
                    provider_specific_config,
                ),
            );
        }
        let adapter: Arc<dyn ProviderAdapter> = Arc::from(adapter);

        use secrecy::ExposeSecret;
        let mut openai_cfg = OpenAiCompatibleConfig::new(
            "groq",
            config.api_key.expose_secret(),
            &config.base_url,
            adapter,
        )
        .with_http_config(config.http_config.clone())
        .with_common_params(config.common_params.clone())
        .with_model(&config.common_params.model)
        .with_http_interceptors(http_interceptors)
        .with_model_middlewares(model_middlewares);

        if let Some(transport) = config.http_transport.clone() {
            openai_cfg = openai_cfg.with_http_transport(transport);
        }

        let inner = OpenAiCompatibleClient::with_http_client(openai_cfg, http_client).await?;
        Ok(Self::new(inner))
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

    pub fn set_retry_options(&mut self, options: Option<RetryOptions>) {
        self.inner.set_retry_options(options);
    }
}

impl crate::traits::ModelMetadata for GroqClient {
    fn provider_id(&self) -> &str {
        "groq"
    }

    fn model_id(&self) -> &str {
        self.inner.model()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::http::transport::{
        HttpTransport, HttpTransportRequest, HttpTransportResponse, HttpTransportStreamBody,
        HttpTransportStreamResponse,
    };
    use crate::provider_metadata::groq::GroqChatResponseExt;
    use crate::provider_options::{
        GroqOptions, GroqReasoningEffort, GroqReasoningFormat, GroqServiceTier,
    };
    use crate::providers::groq::GroqConfig;
    use crate::providers::groq::ext::GroqChatRequestExt;
    use crate::types::{ToolChoice, chat::ResponseFormat};
    use async_trait::async_trait;
    use futures_util::StreamExt;
    use reqwest::header::{ACCEPT, CONTENT_TYPE, HeaderMap, HeaderValue};
    use std::sync::{Arc, Mutex};

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
    async fn groq_client_from_config_builds_inner_client() {
        let cfg = GroqConfig::new("test-key").with_model("llama-3.3-70b-versatile");
        let client = GroqClient::from_config(cfg).await.expect("from_config ok");
        assert_eq!(client.provider_id(), std::borrow::Cow::Borrowed("groq"));
        assert_eq!(crate::traits::ModelMetadata::provider_id(&client), "groq");
        assert_eq!(
            crate::traits::ModelMetadata::model_id(&client),
            "llama-3.3-70b-versatile"
        );
    }

    #[tokio::test]
    async fn groq_client_does_not_expose_audio_extras_without_provider_owned_support() {
        let cfg = GroqConfig::new("test-key").with_model("whisper-large-v3");
        let client = GroqClient::from_config(cfg).await.expect("from_config ok");

        assert!(client.as_speech_capability().is_some());
        assert!(client.as_speech_extras().is_none());
        assert!(client.as_transcription_capability().is_some());
        assert!(client.as_transcription_extras().is_none());
    }

    #[tokio::test]
    async fn groq_client_with_http_client_preserves_provider_context() {
        let cfg = GroqConfig::new("test-key")
            .with_base_url("https://example.com/custom/")
            .with_model("llama-3.3-70b-versatile");
        let client = GroqClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("with_http_client ok");
        assert_eq!(client.base_url(), "https://example.com/custom");
    }

    #[tokio::test]
    async fn groq_client_chat_request_preserves_provider_options() {
        let transport = CaptureTransport::default();
        let cfg = GroqConfig::new("test-key")
            .with_model("llama-3.3-70b-versatile")
            .with_http_transport(Arc::new(transport.clone()));
        let client = GroqClient::from_config(cfg).await.expect("from_config ok");

        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_provider_option("groq", serde_json::json!({ "foo": "bar" }));

        let _ = client.chat_request(request).await;
        let captured = transport.take().expect("captured request");
        assert_eq!(captured.body["foo"], serde_json::json!("bar"));
    }

    #[tokio::test]
    async fn groq_client_chat_stream_request_preserves_typed_options_and_stable_fields_at_transport_boundary()
     {
        let transport = SseResponseTransport::new(b"data: [DONE]\n\n".to_vec());
        let cfg = GroqConfig::new("test-key")
            .with_model("llama-3.3-70b-versatile")
            .with_http_transport(Arc::new(transport.clone()));
        let client = GroqClient::from_config(cfg).await.expect("from_config ok");

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
            "required": ["value"],
            "additionalProperties": false
        });

        let request = ChatRequest::builder()
            .model("llama-3.3-70b-versatile")
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
            .response_format(ResponseFormat::json_schema(schema.clone()).with_name("response"))
            .build()
            .with_groq_options(
                GroqOptions::new()
                    .with_logprobs(true)
                    .with_top_logprobs(2)
                    .with_service_tier(GroqServiceTier::Flex)
                    .with_reasoning_effort(GroqReasoningEffort::Default)
                    .with_reasoning_format(GroqReasoningFormat::Parsed)
                    .with_param(
                        "response_format",
                        serde_json::json!({
                            "type": "json_object"
                        }),
                    )
                    .with_param("tool_choice", serde_json::json!("auto")),
            );

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
        assert_eq!(captured.body["logprobs"], serde_json::json!(true));
        assert_eq!(captured.body["top_logprobs"], serde_json::json!(2));
        assert_eq!(captured.body["service_tier"], serde_json::json!("flex"));
        assert_eq!(
            captured.body["reasoning_effort"],
            serde_json::json!("default")
        );
        assert_eq!(
            captured.body["reasoning_format"],
            serde_json::json!("parsed")
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
    async fn groq_client_from_config_merges_typed_provider_defaults() {
        let transport = JsonResponseTransport::new(serde_json::json!({
            "id": "chatcmpl-groq-test",
            "object": "chat.completion",
            "created": 1_741_392_000,
            "model": "llama-3.3-70b-versatile",
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
        let cfg = GroqConfig::new("test-key")
            .with_model("llama-3.3-70b-versatile")
            .with_http_transport(Arc::new(transport.clone()))
            .with_logprobs(true)
            .with_top_logprobs(2)
            .with_service_tier(GroqServiceTier::Flex)
            .with_reasoning_effort(GroqReasoningEffort::Default)
            .with_reasoning_format(GroqReasoningFormat::Parsed);
        let client = GroqClient::from_config(cfg).await.expect("from_config ok");

        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()]);
        let _ = client.chat_request(request).await.expect("chat ok");
        let captured = transport.take().expect("captured request");

        assert_eq!(captured.body["logprobs"], serde_json::json!(true));
        assert_eq!(captured.body["top_logprobs"], serde_json::json!(2));
        assert_eq!(captured.body["service_tier"], serde_json::json!("flex"));
        assert_eq!(
            captured.body["reasoning_effort"],
            serde_json::json!("default")
        );
        assert_eq!(
            captured.body["reasoning_format"],
            serde_json::json!("parsed")
        );
    }

    #[tokio::test]
    async fn groq_client_exposes_typed_response_metadata() {
        let transport = JsonResponseTransport::new(serde_json::json!({
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
        }));
        let cfg = GroqConfig::new("test-key")
            .with_model("llama-3.3-70b-versatile")
            .with_http_transport(Arc::new(transport.clone()));
        let client = GroqClient::from_config(cfg).await.expect("from_config ok");

        let request = ChatRequest::builder()
            .model("llama-3.3-70b-versatile")
            .messages(vec![ChatMessage::user("hi").build()])
            .build();

        let response = client.chat_request(request).await.expect("response ok");
        let captured = transport.take().expect("captured request");

        assert_eq!(
            captured.body["model"],
            serde_json::json!("llama-3.3-70b-versatile")
        );
        assert_eq!(response.content_text(), Some("hello from groq"));

        let meta = response.groq_metadata().expect("groq metadata");
        assert_eq!(
            meta.logprobs,
            Some(serde_json::json!([
                {
                    "token": "hello",
                    "logprob": -0.2,
                    "bytes": [104, 101, 108, 108, 111],
                    "top_logprobs": []
                }
            ]))
        );
    }

    #[tokio::test]
    async fn groq_client_chat_stream_exposes_typed_response_metadata() {
        let transport = SseResponseTransport::new(
            br#"data: {"id":"1","model":"llama-3.3-70b-versatile","created":1718345013,"choices":[{"index":0,"delta":{"content":"hello","role":"assistant"},"finish_reason":null}]}


data: {"id":"1","model":"llama-3.3-70b-versatile","created":1718345013,"choices":[{"index":0,"delta":{"content":" from groq","role":null},"finish_reason":"stop","logprobs":{"content":[{"token":"hello","logprob":-0.2,"bytes":[104,101,108,108,111],"top_logprobs":[]}]}}]}


data: [DONE]

"#
                .to_vec(),
        );
        let cfg = GroqConfig::new("test-key")
            .with_model("llama-3.3-70b-versatile")
            .with_http_transport(Arc::new(transport.clone()));
        let client = GroqClient::from_config(cfg).await.expect("from_config ok");

        let request = ChatRequest::builder()
            .model("llama-3.3-70b-versatile")
            .messages(vec![ChatMessage::user("hi").build()])
            .build();

        let stream = client
            .chat_stream_request(request)
            .await
            .expect("stream ok");
        let events = stream.collect::<Vec<_>>().await;
        let captured = transport.take_stream().expect("captured stream request");

        assert_eq!(
            captured.body["model"],
            serde_json::json!("llama-3.3-70b-versatile")
        );

        let end = events
            .iter()
            .find_map(|event| match event {
                Ok(crate::types::ChatStreamEvent::StreamEnd { response }) => Some(response),
                _ => None,
            })
            .expect("stream end event");
        assert_eq!(end.content_text(), Some("hello from groq"));

        let meta = end.groq_metadata().expect("groq metadata");
        assert_eq!(
            meta.logprobs,
            Some(serde_json::json!([
                {
                    "token": "hello",
                    "logprob": -0.2,
                    "bytes": [104, 101, 108, 108, 111],
                    "top_logprobs": []
                }
            ]))
        );
    }
}

#[async_trait]
impl ChatCapability for GroqClient {
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

impl LlmClient for GroqClient {
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
            .with_audio()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }

    fn as_chat_capability(&self) -> Option<&dyn ChatCapability> {
        Some(self)
    }

    fn as_embedding_capability(&self) -> Option<&dyn EmbeddingCapability> {
        None
    }

    fn as_audio_capability(&self) -> Option<&dyn AudioCapability> {
        Some(self)
    }

    fn as_speech_capability(&self) -> Option<&dyn crate::traits::SpeechCapability> {
        Some(self)
    }

    fn as_speech_extras(&self) -> Option<&dyn crate::traits::SpeechExtras> {
        None
    }

    fn as_transcription_capability(&self) -> Option<&dyn crate::traits::TranscriptionCapability> {
        Some(self)
    }

    fn as_transcription_extras(&self) -> Option<&dyn crate::traits::TranscriptionExtras> {
        None
    }

    fn as_image_generation_capability(&self) -> Option<&dyn ImageGenerationCapability> {
        None
    }

    fn as_image_extras(&self) -> Option<&dyn ImageExtras> {
        None
    }

    fn as_rerank_capability(&self) -> Option<&dyn RerankCapability> {
        None
    }

    fn as_model_listing_capability(&self) -> Option<&dyn ModelListingCapability> {
        Some(self)
    }
}

#[async_trait]
impl ModelListingCapability for GroqClient {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.inner.list_models().await
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        self.inner.get_model(model_id).await
    }
}
