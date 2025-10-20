//! Anthropic on Vertex AI client
//!
//! This client reuses Anthropic transformers but targets Vertex AI publisher endpoints,
//! authenticating via `Authorization: Bearer <token>` headers.

use async_trait::async_trait;
use reqwest::Client as HttpClient;
use std::sync::Arc;

use crate::error::LlmError;
use crate::executors::chat::{ChatExecutor, HttpChatExecutor};
use crate::middleware::language_model::LanguageModelMiddleware;
use crate::stream::ChatStream;
use crate::traits::{ChatCapability, ModelListingCapability};
use crate::types::{ChatMessage, ChatRequest, ChatResponse, ModelInfo};
use crate::utils::http_interceptor::HttpInterceptor;

/// Minimal config for Vertex Anthropic client (delegate to SiumaiBuilder for common params)
#[derive(Debug, Clone)]
pub struct VertexAnthropicConfig {
    pub base_url: String,
    pub model: String,
    pub http_config: crate::types::HttpConfig,
}

#[derive(Clone)]
pub struct VertexAnthropicClient {
    http_client: HttpClient,
    config: VertexAnthropicConfig,
    /// Optional HTTP interceptors applied to chat requests
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Optional model-level middlewares
    model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
}

impl VertexAnthropicClient {
    pub fn new(config: VertexAnthropicConfig, http_client: HttpClient) -> Self {
        Self {
            http_client,
            config,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
        }
    }

    fn build_headers_fn(
        &self,
    ) -> impl Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<reqwest::header::HeaderMap, LlmError>> + Send>> + Send + Sync + 'static {
        let extra = self.config.http_config.headers.clone();
        move || {
            let extra = extra.clone();
            Box::pin(async move {
                let mut headers = crate::utils::http_headers::ProviderHeaders::vertex_bearer(&extra)?;
                crate::utils::http_headers::inject_tracing_headers(&mut headers);
                Ok(headers)
            }) as std::pin::Pin<Box<dyn std::future::Future<Output = Result<reqwest::header::HeaderMap, LlmError>> + Send>>
        }
    }

    fn build_url_fn(&self, _stream: bool) -> impl Fn(bool) -> String + Send + Sync + 'static {
        let base = self.config.base_url.clone();
        let model = self.config.model.clone();
        move |is_stream| {
            if is_stream {
                crate::utils::url::join_url(
                    &base,
                    &format!("models/{}:streamRawPredict?alt=sse", model),
                )
            } else {
                crate::utils::url::join_url(&base, &format!("models/{}:rawPredict", model))
            }
        }
    }

    fn models_url(&self) -> String {
        crate::utils::url::join_url(&self.config.base_url, "models")
    }

    fn parse_model_id(name: &str) -> String {
        // Vertex model resource typically ends with "/models/{id}"; extract the trailing id.
        match name.rsplit_once("/models/") {
            Some((_, id)) => id.to_string(),
            None => name.to_string(),
        }
    }
}

#[async_trait]
impl ChatCapability for VertexAnthropicClient {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<crate::types::Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        let req = ChatRequest {
            messages,
            tools,
            common_params: crate::types::CommonParams {
                model: self.config.model.clone(),
                ..Default::default()
            },
            provider_params: None,
            http_config: Some(self.config.http_config.clone()),
            web_search: None,
            stream: false,
            telemetry: None,
        };

        let req_tx =
            crate::providers::anthropic::transformers::AnthropicRequestTransformer::new(None);
        let resp_tx = crate::providers::anthropic::transformers::AnthropicResponseTransformer;

        let exec = HttpChatExecutor {
            provider_id: "anthropic-vertex".to_string(),
            http_client: self.http_client.clone(),
            request_transformer: Arc::new(req_tx),
            response_transformer: Arc::new(resp_tx),
            stream_transformer: None,
            json_stream_converter: None,
            stream_disable_compression: self.config.http_config.stream_disable_compression,
            interceptors: self.http_interceptors.clone(),
            middlewares: self.model_middlewares.clone(),
            build_url: Box::new(self.build_url_fn(false)),
            build_headers: std::sync::Arc::new(self.build_headers_fn()),
            before_send: None,
        };
        exec.execute(req).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<crate::types::Tool>>,
    ) -> Result<ChatStream, LlmError> {
        let req = ChatRequest {
            messages,
            tools,
            common_params: crate::types::CommonParams {
                model: self.config.model.clone(),
                ..Default::default()
            },
            provider_params: None,
            http_config: Some(self.config.http_config.clone()),
            web_search: None,
            stream: true,
            telemetry: None,
        };

        let req_tx =
            crate::providers::anthropic::transformers::AnthropicRequestTransformer::new(None);
        let resp_tx = crate::providers::anthropic::transformers::AnthropicResponseTransformer;
        let stream_tx =
            crate::providers::anthropic::transformers::AnthropicStreamChunkTransformer {
                provider_id: "anthropic-vertex".to_string(),
                inner: crate::providers::anthropic::streaming::AnthropicEventConverter::new(
                    crate::params::AnthropicParams::default(),
                ),
            };

        let exec = HttpChatExecutor {
            provider_id: "anthropic-vertex".to_string(),
            http_client: self.http_client.clone(),
            request_transformer: Arc::new(req_tx),
            response_transformer: Arc::new(resp_tx),
            stream_transformer: Some(Arc::new(stream_tx)),
            json_stream_converter: None,
            stream_disable_compression: self.config.http_config.stream_disable_compression,
            interceptors: self.http_interceptors.clone(),
            middlewares: self.model_middlewares.clone(),
            build_url: Box::new(self.build_url_fn(true)),
            build_headers: std::sync::Arc::new(self.build_headers_fn()),
            before_send: None,
        };
        exec.execute_stream(req).await
    }
}

impl VertexAnthropicClient {
    /// Install HTTP interceptors for all chat requests.
    pub fn with_http_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.http_interceptors = interceptors;
        self
    }

    /// Install model-level middlewares for chat requests.
    pub fn with_model_middlewares(
        mut self,
        middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    ) -> Self {
        self.model_middlewares = middlewares;
        self
    }
}

#[async_trait]
impl ModelListingCapability for VertexAnthropicClient {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        let url = self.models_url();
        let headers = (self.build_headers_fn())().await?;
        let resp = self
            .http_client
            .get(url.clone())
            .headers(headers)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;
        let resp = if !resp.status().is_success() {
            let status = resp.status();
            if status.as_u16() == 401 {
                let headers = (self.build_headers_fn())().await?;
                self.http_client
                    .get(url)
                    .headers(headers)
                    .send()
                    .await
                    .map_err(|e| LlmError::HttpError(e.to_string()))?
            } else {
                let headers_map = resp.headers().clone();
                let text = resp.text().await.unwrap_or_default();
                return Err(crate::retry_api::classify_http_error(
                    "anthropic-vertex",
                    status.as_u16(),
                    &text,
                    &headers_map,
                    None,
                ));
            }
        } else {
            resp
        };
        let text = resp
            .text()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;
        let json: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| LlmError::ParseError(format!("Failed to parse models JSON: {e}")))?;
        let mut out = Vec::new();
        if let Some(arr) = json.get("models").and_then(|v| v.as_array()) {
            for m in arr {
                let raw_name = m.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let id = if raw_name.is_empty() {
                    self.config.model.clone()
                } else {
                    Self::parse_model_id(raw_name)
                };
                let display = m
                    .get("displayName")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                out.push(ModelInfo {
                    id,
                    name: display,
                    description: None,
                    owned_by: "Anthropic@Vertex".to_string(),
                    created: None,
                    capabilities: vec![
                        "chat".to_string(),
                        "streaming".to_string(),
                        "tools".to_string(),
                    ],
                    context_window: None,
                    max_output_tokens: None,
                    input_cost_per_token: None,
                    output_cost_per_token: None,
                });
            }
        }
        if out.is_empty() {
            // Fallback to configured model when list is empty
            out.push(ModelInfo {
                id: self.config.model.clone(),
                name: None,
                description: None,
                owned_by: "Anthropic@Vertex".to_string(),
                created: None,
                capabilities: vec![
                    "chat".to_string(),
                    "streaming".to_string(),
                    "tools".to_string(),
                ],
                context_window: None,
                max_output_tokens: None,
                input_cost_per_token: None,
                output_cost_per_token: None,
            });
        }
        Ok(out)
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        // Attempt to fetch a specific model; if endpoint is unavailable, fallback to minimal info
        let url =
            crate::utils::url::join_url(&self.config.base_url, &format!("models/{}", model_id));
        let headers = (self.build_headers_fn())().await?;
        let resp = self
            .http_client
            .get(url.clone())
            .headers(headers)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;
        if !resp.status().is_success() {
            // Fallback minimal info
            return Ok(ModelInfo {
                id: model_id,
                name: None,
                description: None,
                owned_by: "Anthropic@Vertex".to_string(),
                created: None,
                capabilities: vec![
                    "chat".to_string(),
                    "streaming".to_string(),
                    "tools".to_string(),
                ],
                context_window: None,
                max_output_tokens: None,
                input_cost_per_token: None,
                output_cost_per_token: None,
            });
        }
        let text = resp
            .text()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;
        let json: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| LlmError::ParseError(format!("Failed to parse model JSON: {e}")))?;
        let raw_name = json.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let id = if raw_name.is_empty() {
            model_id
        } else {
            Self::parse_model_id(raw_name)
        };
        let display = json
            .get("displayName")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        Ok(ModelInfo {
            id,
            name: display,
            description: None,
            owned_by: "Anthropic@Vertex".to_string(),
            created: None,
            capabilities: vec![
                "chat".to_string(),
                "streaming".to_string(),
                "tools".to_string(),
            ],
            context_window: None,
            max_output_tokens: None,
            input_cost_per_token: None,
            output_cost_per_token: None,
        })
    }
}

impl crate::client::LlmClient for VertexAnthropicClient {
    fn provider_name(&self) -> &'static str {
        "anthropic"
    }

    fn supported_models(&self) -> Vec<String> {
        vec![self.config.model.clone()]
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn crate::client::LlmClient> {
        Box::new(self.clone())
    }
}
