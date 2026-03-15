//! `DeepSeek` client.
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
    ChatCapability, EmbeddingCapability, ImageExtras, ImageGenerationCapability,
    ModelListingCapability, RerankCapability,
};
use crate::types::{
    ChatMessage, ChatRequest, ChatResponse, EmbeddingRequest, EmbeddingResponse, ImageEditRequest,
    ImageGenerationRequest, ImageGenerationResponse, ImageVariationRequest, ModelInfo,
    RerankRequest, RerankResponse, Tool,
};
use async_trait::async_trait;
use siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleClient;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct DeepSeekClient {
    inner: OpenAiCompatibleClient,
}

impl DeepSeekClient {
    pub fn new(inner: OpenAiCompatibleClient) -> Self {
        Self { inner }
    }

    /// Construct a `DeepSeekClient` from `DeepSeekConfig`.
    pub async fn from_config(config: super::DeepSeekConfig) -> Result<Self, LlmError> {
        let inner = OpenAiCompatibleClient::from_config(config.into_compatible_config()?).await?;
        Ok(Self::new(inner))
    }

    /// Construct a `DeepSeekClient` from `DeepSeekConfig` with a caller-supplied HTTP client.
    pub async fn with_http_client(
        config: super::DeepSeekConfig,
        http_client: reqwest::Client,
    ) -> Result<Self, LlmError> {
        let inner =
            OpenAiCompatibleClient::with_http_client(config.into_compatible_config()?, http_client)
                .await?;
        Ok(Self::new(inner))
    }

    /// Construct a `DeepSeekClient` from env (`DEEPSEEK_API_KEY`).
    pub async fn from_env(model: Option<&str>) -> Result<Self, LlmError> {
        let model = model.unwrap_or(super::models::CHAT);
        let config = super::DeepSeekConfig::from_env()?.with_model(model);
        Self::from_config(config).await
    }

    /// Alias kept for parity with OpenAI-compatible wrapper patterns.
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

    pub fn set_retry_options(&mut self, options: Option<RetryOptions>) {
        self.inner.set_retry_options(options);
    }

    fn deepseek_spec(&self) -> Arc<dyn crate::core::ProviderSpec> {
        Arc::new(super::spec::DeepSeekSpec::new(self.inner.adapter()))
    }
}

impl crate::traits::ModelMetadata for DeepSeekClient {
    fn provider_id(&self) -> &str {
        "deepseek"
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
    use crate::provider_metadata::deepseek::DeepSeekChatResponseExt;
    use crate::types::chat::ResponseFormat;
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
    async fn deepseek_client_from_config_builds_inner_client() {
        let cfg = super::super::config::DeepSeekConfig::new("test-key").with_model("deepseek-chat");
        let client = DeepSeekClient::from_config(cfg)
            .await
            .expect("from_config ok");
        assert_eq!(client.provider_id(), std::borrow::Cow::Borrowed("deepseek"));
    }

    #[tokio::test]
    async fn deepseek_client_with_http_client_preserves_provider_context() {
        let cfg = super::super::config::DeepSeekConfig::new("test-key")
            .with_base_url("https://example.com/custom/")
            .with_model("deepseek-chat");
        let client = DeepSeekClient::with_http_client(cfg, reqwest::Client::new())
            .await
            .expect("with_http_client ok");
        assert_eq!(client.base_url(), "https://example.com/custom/");
    }

    #[tokio::test]
    async fn deepseek_client_chat_request_uses_deepseek_spec_normalization() {
        let transport = CaptureTransport::default();
        let cfg = super::super::config::DeepSeekConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_model("deepseek-chat")
            .with_http_transport(Arc::new(transport.clone()));
        let client = DeepSeekClient::from_config(cfg)
            .await
            .expect("from_config ok");

        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()]).with_provider_option(
            "deepseek",
            serde_json::json!({
                "enableReasoning": true,
                "reasoningBudget": 4096,
                "foo": "bar"
            }),
        );

        let _ = client.chat_request(request).await;
        let captured = transport.take().expect("captured request");
        assert_eq!(captured.body["enable_reasoning"], serde_json::json!(true));
        assert_eq!(captured.body["reasoning_budget"], serde_json::json!(4096));
        assert_eq!(captured.body["foo"], serde_json::json!("bar"));
        assert!(captured.body.get("enableReasoning").is_none());
        assert!(captured.body.get("reasoningBudget").is_none());
    }

    #[tokio::test]
    async fn deepseek_client_preserves_stable_response_format_against_raw_provider_options() {
        let transport = CaptureTransport::default();
        let cfg = super::super::config::DeepSeekConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_model("deepseek-chat")
            .with_http_transport(Arc::new(transport.clone()));
        let client = DeepSeekClient::from_config(cfg)
            .await
            .expect("from_config ok");

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let request = ChatRequest::builder()
            .model("deepseek-chat")
            .messages(vec![ChatMessage::user("hi").build()])
            .response_format(ResponseFormat::json_schema(schema.clone()).with_name("response"))
            .build()
            .with_provider_option(
                "deepseek",
                serde_json::json!({
                    "response_format": { "type": "json_object" },
                    "reasoningBudget": 2048
                }),
            );

        let _ = client.chat_request(request).await;
        let captured = transport.take().expect("captured request");
        assert_eq!(captured.body["reasoning_budget"], serde_json::json!(2048));
        assert!(captured.body.get("reasoningBudget").is_none());
        assert_eq!(
            captured.body.get("response_format"),
            Some(&serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            }))
        );
    }

    #[tokio::test]
    async fn deepseek_client_chat_stream_request_preserves_stable_fields_and_reasoning_normalization_at_transport_boundary()
     {
        let transport = SseResponseTransport::new(b"data: [DONE]\n\n".to_vec());
        let cfg = super::super::config::DeepSeekConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_model("deepseek-chat")
            .with_http_transport(Arc::new(transport.clone()));
        let client = DeepSeekClient::from_config(cfg)
            .await
            .expect("from_config ok");

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let request = ChatRequest::builder()
            .model("deepseek-chat")
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
            .tool_choice(crate::types::ToolChoice::None)
            .response_format(ResponseFormat::json_schema(schema.clone()).with_name("response"))
            .build()
            .with_provider_option(
                "deepseek",
                serde_json::json!({
                    "response_format": { "type": "json_object" },
                    "tool_choice": "auto",
                    "reasoningBudget": 2048
                }),
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
        assert_eq!(captured.body["reasoning_budget"], serde_json::json!(2048));
        assert!(captured.body.get("reasoningBudget").is_none());
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
    async fn deepseek_client_exposes_typed_response_metadata() {
        let transport = JsonResponseTransport::new(serde_json::json!({
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
        }));

        let cfg = super::super::config::DeepSeekConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_model("deepseek-chat")
            .with_http_transport(Arc::new(transport.clone()));
        let client = DeepSeekClient::from_config(cfg)
            .await
            .expect("from_config ok");

        let request = ChatRequest::builder()
            .model("deepseek-chat")
            .messages(vec![ChatMessage::user("hi").build()])
            .build();

        let response = client.chat_request(request).await.expect("response ok");
        let captured = transport.take().expect("captured request");

        assert_eq!(captured.body["model"], serde_json::json!("deepseek-chat"));
        assert_eq!(response.content_text(), Some("hello from deepseek"));
        let meta = response.deepseek_metadata().expect("deepseek metadata");
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
    async fn deepseek_client_chat_stream_exposes_typed_response_metadata() {
        let transport = SseResponseTransport::new(
            br#"data: {"id":"1","model":"deepseek-chat","created":1718345013,"choices":[{"index":0,"delta":{"content":"hello","role":"assistant"},"finish_reason":null}]}


data: {"id":"1","model":"deepseek-chat","created":1718345013,"choices":[{"index":0,"delta":{"content":" from deepseek","role":null},"finish_reason":"stop","logprobs":{"content":[{"token":"hello","logprob":-0.1,"bytes":[104,101,108,108,111],"top_logprobs":[]}]}}],"usage":{"prompt_tokens":11,"completion_tokens":3,"total_tokens":14,"reasoning_tokens":2}}


data: [DONE]

"#
                .to_vec(),
        );
        let cfg = super::super::config::DeepSeekConfig::new("test-key")
            .with_base_url("https://example.com/v1")
            .with_model("deepseek-chat")
            .with_http_transport(Arc::new(transport.clone()));
        let client = DeepSeekClient::from_config(cfg)
            .await
            .expect("from_config ok");

        let request = ChatRequest::builder()
            .model("deepseek-chat")
            .messages(vec![ChatMessage::user("hi").build()])
            .build();

        let stream = client
            .chat_stream_request(request)
            .await
            .expect("stream ok");
        let events = stream.collect::<Vec<_>>().await;
        let captured = transport.take_stream().expect("captured stream request");

        assert_eq!(captured.body["model"], serde_json::json!("deepseek-chat"));

        let end = events
            .iter()
            .find_map(|event| match event {
                Ok(crate::types::ChatStreamEvent::StreamEnd { response }) => Some(response),
                _ => None,
            })
            .expect("stream end event");
        assert_eq!(end.content_text(), Some("hello from deepseek"));
        assert_eq!(end.usage.as_ref().map(|usage| usage.total_tokens), Some(14));

        let meta = end.deepseek_metadata().expect("deepseek metadata");
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
impl ChatCapability for DeepSeekClient {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        let mut builder = ChatRequest::builder()
            .messages(messages)
            .common_params(self.inner.common_params())
            .http_config(self.inner.http_config());
        if let Some(ts) = tools {
            builder = builder.tools(ts);
        }
        self.chat_request(builder.build()).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        let mut builder = ChatRequest::builder()
            .messages(messages)
            .common_params(self.inner.common_params())
            .http_config(self.inner.http_config())
            .stream(true);
        if let Some(ts) = tools {
            builder = builder.tools(ts);
        }
        self.chat_stream_request(builder.build()).await
    }

    async fn chat_request(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        self.inner
            .chat_request_with_spec(request, self.deepseek_spec())
            .await
    }

    async fn chat_stream_request(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        self.inner
            .chat_stream_request_with_spec(request, self.deepseek_spec())
            .await
    }
}

#[async_trait]
impl EmbeddingCapability for DeepSeekClient {
    async fn embed(&self, _texts: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "DeepSeek does not currently expose embedding on the provider-owned DeepSeek client path"
                .to_string(),
        ))
    }

    fn embedding_dimension(&self) -> usize {
        0
    }
}

#[async_trait]
impl crate::traits::EmbeddingExtensions for DeepSeekClient {
    async fn embed_with_config(
        &self,
        _request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "DeepSeek does not currently expose embedding on the provider-owned DeepSeek client path"
                .to_string(),
        ))
    }
}

#[async_trait]
impl RerankCapability for DeepSeekClient {
    async fn rerank(&self, _request: RerankRequest) -> Result<RerankResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "DeepSeek does not currently expose rerank on the provider-owned DeepSeek client path"
                .to_string(),
        ))
    }
}

#[async_trait]
impl ModelListingCapability for DeepSeekClient {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.inner.list_models().await
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        self.inner.get_model(model_id).await
    }
}

#[async_trait]
impl ImageGenerationCapability for DeepSeekClient {
    async fn generate_images(
        &self,
        _request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "DeepSeek does not currently expose image generation on the provider-owned DeepSeek client path"
                .to_string(),
        ))
    }
}

#[async_trait]
impl ImageExtras for DeepSeekClient {
    async fn edit_image(
        &self,
        _request: ImageEditRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "DeepSeek does not currently expose image editing on the provider-owned DeepSeek client path"
                .to_string(),
        ))
    }

    async fn create_variation(
        &self,
        _request: ImageVariationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "DeepSeek does not currently expose image variation on the provider-owned DeepSeek client path"
                .to_string(),
        ))
    }

    fn get_supported_sizes(&self) -> Vec<String> {
        Vec::new()
    }

    fn get_supported_formats(&self) -> Vec<String> {
        Vec::new()
    }

    fn supports_image_editing(&self) -> bool {
        false
    }

    fn supports_image_variations(&self) -> bool {
        false
    }
}

impl LlmClient for DeepSeekClient {
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
            .with_custom_feature("thinking", true)
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
