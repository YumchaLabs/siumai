//! OpenAI Compatible Client
//!
//! This module provides a client implementation for OpenAI-compatible providers.

use super::openai_config::OpenAiCompatibleConfig;
use crate::client::LlmClient;
use crate::core::{ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::executors::chat::{ChatExecutor, HttpChatExecutor};
use crate::execution::executors::embedding::{EmbeddingExecutor, HttpEmbeddingExecutor};
use crate::execution::executors::image::{HttpImageExecutor, ImageExecutor};
// use crate::providers::openai_compatible::RequestType; // no longer needed here
use crate::retry_api::RetryOptions;
use crate::streaming::ChatStream;
use crate::traits::{
    ChatCapability, EmbeddingCapability, ImageGenerationCapability, ModelListingCapability,
    RerankCapability,
};
// use crate::execution::transformers::request::RequestTransformer; // unused
use crate::types::*;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
// removed: HashMap import not needed after legacy removal
use crate::execution::http::headers::ProviderHeaders;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::middleware::language_model::LanguageModelMiddleware;

/// OpenAI Compatible Chat Response with provider-specific fields
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiCompatibleChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAiCompatibleChoice>,
    pub usage: Option<OpenAiCompatibleUsage>,
}

/// OpenAI Compatible Choice with provider-specific fields
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiCompatibleChoice {
    pub index: u32,
    pub message: OpenAiCompatibleMessage,
    pub finish_reason: Option<String>,
}

/// OpenAI Compatible Message with provider-specific fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiCompatibleMessage {
    pub role: String,
    pub content: Option<serde_json::Value>,
    pub tool_calls: Option<Vec<OpenAiCompatibleToolCall>>,
    pub tool_call_id: Option<String>,

    // Provider-specific thinking/reasoning fields
    pub thinking: Option<String>,          // Standard thinking field
    pub reasoning_content: Option<String>, // DeepSeek reasoning field
    pub reasoning: Option<String>,         // Alternative reasoning field
}

/// OpenAI Compatible Tool Call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiCompatibleToolCall {
    pub id: String,
    pub r#type: String,
    pub function: Option<OpenAiCompatibleFunction>,
}

/// OpenAI Compatible Function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiCompatibleFunction {
    pub name: String,
    pub arguments: String,
}

/// OpenAI Compatible Usage
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiCompatibleUsage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
}

/// OpenAI compatible client
///
/// This is a separate client implementation that uses the adapter system
/// to handle provider-specific differences without modifying the core OpenAI client.
#[derive(Clone)]
pub struct OpenAiCompatibleClient {
    config: OpenAiCompatibleConfig,
    http_client: reqwest::Client,
    retry_options: Option<RetryOptions>,
    /// Optional HTTP interceptors applied to all chat requests
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Optional model-level middlewares
    model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
}

impl OpenAiCompatibleClient {
    /// Execute a non-stream chat via ProviderSpec
    async fn chat_request_via_spec(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        let spec = crate::providers::openai_compatible::spec::OpenAiCompatibleSpec;
        // Merge custom headers from HttpConfig + config.custom_headers + adapter.custom_headers
        let mut extra_headers: std::collections::HashMap<String, String> =
            self.config.http_config.headers.clone();
        let merge_header_map = |dst: &mut std::collections::HashMap<String, String>,
                                hm: &reqwest::header::HeaderMap| {
            for (name, value) in hm.iter() {
                if let Ok(val) = value.to_str() {
                    dst.insert(name.as_str().to_string(), val.to_string());
                }
            }
        };
        merge_header_map(&mut extra_headers, &self.config.custom_headers);
        merge_header_map(&mut extra_headers, &self.config.adapter.custom_headers());

        let ctx = ProviderContext::new(
            self.config.provider_id.clone(),
            self.config.base_url.clone(),
            Some(self.config.api_key.clone()),
            extra_headers,
        );
        let spec = Arc::new(spec);
        let bundle = spec.choose_chat_transformers(&request, &ctx);
        let http = self.http_client.clone();

        let exec = std::sync::Arc::new(HttpChatExecutor {
            provider_id: self.config.provider_id.clone(),
            http_client: http,
            request_transformer: bundle.request,
            response_transformer: bundle.response,
            stream_transformer: bundle.stream,
            json_stream_converter: bundle.json,
            stream_disable_compression: self.config.http_config.stream_disable_compression,
            interceptors: self.http_interceptors.clone(),
            middlewares: self.model_middlewares.clone(),
            provider_spec: spec,
            provider_context: ctx,
            before_send: None,
            retry_options: None,
        });
        if let Some(opts) = &self.retry_options {
            crate::retry_api::retry_with(
                || {
                    let reqc = request.clone();
                    let exec = exec.clone();
                    async move { exec.execute(reqc).await }
                },
                opts.clone(),
            )
            .await
        } else {
            exec.execute(request).await
        }
    }

    /// Execute a stream chat via ProviderSpec
    async fn chat_stream_request_via_spec(
        &self,
        request: ChatRequest,
    ) -> Result<ChatStream, LlmError> {
        let spec = crate::providers::openai_compatible::spec::OpenAiCompatibleSpec;
        let mut extra_headers: std::collections::HashMap<String, String> =
            self.config.http_config.headers.clone();
        let merge_header_map = |dst: &mut std::collections::HashMap<String, String>,
                                hm: &reqwest::header::HeaderMap| {
            for (name, value) in hm.iter() {
                if let Ok(val) = value.to_str() {
                    dst.insert(name.as_str().to_string(), val.to_string());
                }
            }
        };
        merge_header_map(&mut extra_headers, &self.config.custom_headers);
        merge_header_map(&mut extra_headers, &self.config.adapter.custom_headers());

        let ctx = ProviderContext::new(
            self.config.provider_id.clone(),
            self.config.base_url.clone(),
            Some(self.config.api_key.clone()),
            extra_headers,
        );
        let spec = Arc::new(spec);
        let bundle = spec.choose_chat_transformers(&request, &ctx);
        let http = self.http_client.clone();

        let exec = std::sync::Arc::new(HttpChatExecutor {
            provider_id: self.config.provider_id.clone(),
            http_client: http,
            request_transformer: bundle.request,
            response_transformer: bundle.response,
            stream_transformer: bundle.stream,
            json_stream_converter: bundle.json,
            stream_disable_compression: self.config.http_config.stream_disable_compression,
            interceptors: self.http_interceptors.clone(),
            middlewares: self.model_middlewares.clone(),
            provider_spec: spec,
            provider_context: ctx,
            before_send: None,
            retry_options: None,
        });
        if let Some(opts) = &self.retry_options {
            crate::retry_api::retry_with(
                || {
                    let reqc = request.clone();
                    let exec = exec.clone();
                    async move { exec.execute_stream(reqc).await }
                },
                opts.clone(),
            )
            .await
        } else {
            exec.execute_stream(request).await
        }
    }
    /// Create a new OpenAI compatible client
    pub async fn new(config: OpenAiCompatibleConfig) -> Result<Self, LlmError> {
        // Validate configuration
        config.validate()?;

        // Validate model with adapter
        if !config.model.is_empty() {
            config.adapter.validate_model(&config.model)?;
        }

        // Create HTTP client with configuration
        let http_client = Self::build_http_client(&config)?;

        Ok(Self {
            config,
            http_client,
            retry_options: None,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
        })
    }

    /// Create a new OpenAI compatible client with custom HTTP client
    pub async fn with_http_client(
        config: OpenAiCompatibleConfig,
        http_client: reqwest::Client,
    ) -> Result<Self, LlmError> {
        // Validate configuration
        config.validate()?;

        // Validate model with adapter
        if !config.model.is_empty() {
            config.adapter.validate_model(&config.model)?;
        }

        Ok(Self {
            config,
            http_client,
            retry_options: None,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
        })
    }

    // Removed legacy build_headers; headers are created in executor closures

    /// Set unified retry options
    pub fn set_retry_options(&mut self, options: Option<RetryOptions>) {
        self.retry_options = options;
    }

    /// Install HTTP interceptors for all chat requests.
    pub fn with_http_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.http_interceptors = interceptors;
        self
    }

    /// Install model-level middlewares
    pub fn with_model_middlewares(
        mut self,
        middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    ) -> Self {
        self.model_middlewares = middlewares;
        self
    }

    /// Build unified JSON headers for OpenAI-compatible providers
    fn build_json_headers(
        api_key: &str,
        http_extra: &std::collections::HashMap<String, String>,
        config_headers: &reqwest::header::HeaderMap,
        adapter_headers: &reqwest::header::HeaderMap,
    ) -> Result<reqwest::header::HeaderMap, LlmError> {
        let mut headers = ProviderHeaders::openai(api_key, None, None, http_extra)?;
        // Merge config.custom_headers
        for (k, v) in config_headers.iter() {
            headers.insert(k, v.clone());
        }
        // Merge adapter.custom_headers
        for (k, v) in adapter_headers.iter() {
            headers.insert(k, v.clone());
        }
        Ok(headers)
    }

    /// Build HTTP client with configuration
    fn build_http_client(config: &OpenAiCompatibleConfig) -> Result<reqwest::Client, LlmError> {
        // Use unified HTTP client builder
        crate::execution::http::client::build_http_client_from_config(&config.http_config)
    }

    /// Get the provider ID
    pub fn provider_id(&self) -> &str {
        &self.config.provider_id
    }

    /// Get the current model
    pub fn model(&self) -> &str {
        &self.config.model
    }

    // Removed legacy build_chat_request; executors use transformers directly

    // Removed legacy parse_chat_response; response transformer handles mapping

    // removed legacy send_request; executors handle requests
}

// Removed legacy chat_with_tools_inner; ChatCapability now uses HttpChatExecutor

#[async_trait]
impl ChatCapability for OpenAiCompatibleClient {
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        // Build unified ChatRequest
        let req = ChatRequest {
            messages,
            tools,
            common_params: self.config.common_params.clone(),
            http_config: Some(self.config.http_config.clone()),
            ..Default::default()
        };

        // Execute via ProviderSpec
        self.chat_request_via_spec(req).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        // Unified ChatRequest
        let request = crate::types::ChatRequest {
            messages,
            tools,
            common_params: self.config.common_params.clone(),
            http_config: Some(self.config.http_config.clone()),
            stream: true,
            ..Default::default()
        };

        // Execute via ProviderSpec
        self.chat_stream_request_via_spec(request).await
    }
}

#[async_trait]
impl EmbeddingCapability for OpenAiCompatibleClient {
    async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        use crate::core::{ProviderContext, ProviderSpec};
        let req = crate::types::EmbeddingRequest::new(texts).with_model(self.config.model.clone());

        // Build merged extra headers (HttpConfig + custom + adapter)
        let mut extra_headers = self.config.http_config.headers.clone();
        let merge_header_map = |dst: &mut std::collections::HashMap<String, String>,
                                hm: &reqwest::header::HeaderMap| {
            for (name, value) in hm.iter() {
                if let Ok(val) = value.to_str() {
                    dst.insert(name.as_str().to_string(), val.to_string());
                }
            }
        };
        merge_header_map(&mut extra_headers, &self.config.custom_headers);
        merge_header_map(&mut extra_headers, &self.config.adapter.custom_headers());

        let ctx = ProviderContext::new(
            self.config.provider_id.clone(),
            self.config.base_url.clone(),
            Some(self.config.api_key.clone()),
            extra_headers,
        );
        let spec = Arc::new(crate::providers::openai_compatible::spec::OpenAiCompatibleSpec);
        let bundle = spec.choose_embedding_transformers(&req, &ctx);

        if let Some(opts) = &self.retry_options {
            let http = self.http_client.clone();
            let spec_clone = spec.clone();
            let ctx_clone = ctx.clone();
            let provider_id = self.config.provider_id.clone();
            crate::retry_api::retry_with(
                || {
                    let rqc = req.clone();
                    let http = http.clone();
                    let spec = spec_clone.clone();
                    let ctx = ctx_clone.clone();
                    let provider_id = provider_id.clone();
                    let req_tx_arc = bundle.request.clone();
                    let resp_tx_arc = bundle.response.clone();
                    async move {
                        let exec = HttpEmbeddingExecutor {
                            provider_id,
                            http_client: http,
                            request_transformer: req_tx_arc.clone(),
                            response_transformer: resp_tx_arc.clone(),
                            provider_spec: spec,
                            provider_context: ctx,
                            interceptors: vec![],
                            before_send: None,
                            retry_options: None,
                        };
                        EmbeddingExecutor::execute(&exec, rqc).await
                    }
                },
                opts.clone(),
            )
            .await
        } else {
            let exec = HttpEmbeddingExecutor {
                provider_id: self.config.provider_id.clone(),
                http_client: self.http_client.clone(),
                request_transformer: bundle.request,
                response_transformer: bundle.response,
                provider_spec: spec,
                provider_context: ctx,
                interceptors: vec![],
                before_send: None,
                retry_options: None,
            };
            exec.execute(req).await
        }
    }

    fn embedding_dimension(&self) -> usize {
        // Default dimension, could be made configurable per model
        1536
    }
}

#[async_trait]
impl RerankCapability for OpenAiCompatibleClient {
    async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, LlmError> {
        // Use OpenAI Rerank Standard with provider-aware adapter
        struct LocalRerankAdapter {
            provider_id: String,
        }
        impl crate::standards::openai::rerank::OpenAiRerankAdapter for LocalRerankAdapter {
            fn transform_request(
                &self,
                _req: &RerankRequest,
                _body: &mut serde_json::Value,
            ) -> Result<(), LlmError> {
                Ok(())
            }
            fn transform_response(&self, resp: &mut serde_json::Value) -> Result<(), LlmError> {
                if self.provider_id == "siliconflow" {
                    if let Some(meta) = resp.get_mut("meta") {
                        if let Some(tokens) = meta.get("tokens").cloned() {
                            if let Some(obj) = resp.as_object_mut() {
                                obj.insert("usage".to_string(), tokens);
                            }
                        }
                    }
                }
                Ok(())
            }
        }

        let adapter = std::sync::Arc::new(LocalRerankAdapter {
            provider_id: self.config.provider_id.clone(),
        });
        let standard =
            crate::standards::openai::rerank::OpenAiRerankStandard::with_adapter(adapter);
        let transformers = standard.create_transformers(&self.config.provider_id);

        // Build URL via adapter route mapping
        let endpoint = self
            .config
            .adapter
            .route_for(super::types::RequestType::Rerank);
        let url = format!(
            "{}/{}",
            self.config.base_url.trim_end_matches('/'),
            endpoint
        );

        // Build headers similar to JSON request path
        let adapter_headers = self.config.adapter.custom_headers();
        let headers = Self::build_json_headers(
            &self.config.api_key,
            &self.config.http_config.headers,
            &self.config.custom_headers,
            &adapter_headers,
        )?;

        let exec = crate::execution::executors::rerank::HttpRerankExecutor {
            provider_id: self.config.provider_id.clone(),
            http_client: self.http_client.clone(),
            request_transformer: transformers.request,
            response_transformer: transformers.response,
            interceptors: self.http_interceptors.clone(),
            retry_options: self.retry_options.clone(),
            url,
            headers,
            before_send: None,
        };
        crate::execution::executors::rerank::RerankExecutor::execute(&exec, request).await
    }
}

impl OpenAiCompatibleClient {
    /// List available models from the provider
    async fn list_models_internal(&self) -> Result<Vec<ModelInfo>, LlmError> {
        let url = format!("{}/models", self.config.base_url.trim_end_matches('/'));
        let adapter_headers = self.config.adapter.custom_headers();
        let headers = Self::build_json_headers(
            &self.config.api_key,
            &self.config.http_config.headers,
            &self.config.custom_headers,
            &adapter_headers,
        )?;

        let response = self.http_client.get(&url).headers(headers).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!(
                    "{} Models API error: {}",
                    self.config.provider_id, error_text
                ),
                details: serde_json::from_str(&error_text).ok(),
            });
        }

        let models_response: serde_json::Value = response.json().await?;

        // Parse OpenAI-compatible models response
        let models = models_response
            .get("data")
            .and_then(|data| data.as_array())
            .ok_or_else(|| LlmError::ParseError("Invalid models response format".to_string()))?;

        let mut model_infos = Vec::new();
        for model in models {
            if let Some(model_id) = model.get("id").and_then(|id| id.as_str()) {
                let model_info = ModelInfo {
                    id: model_id.to_string(),
                    name: Some(model_id.to_string()),
                    description: model
                        .get("description")
                        .and_then(|d| d.as_str())
                        .map(|s| s.to_string()),
                    owned_by: model
                        .get("owned_by")
                        .and_then(|o| o.as_str())
                        .unwrap_or(&self.config.provider_id)
                        .to_string(),
                    created: model.get("created").and_then(|c| c.as_u64()),
                    capabilities: self.determine_model_capabilities(model_id),
                    context_window: None, // Not typically provided by OpenAI-compatible APIs
                    max_output_tokens: None, // Not typically provided by OpenAI-compatible APIs
                    input_cost_per_token: None, // Not typically provided by OpenAI-compatible APIs
                    output_cost_per_token: None, // Not typically provided by OpenAI-compatible APIs
                };
                model_infos.push(model_info);
            }
        }

        Ok(model_infos)
    }

    /// Determine model capabilities based on model ID
    fn determine_model_capabilities(&self, model_id: &str) -> Vec<String> {
        let mut capabilities = vec!["chat".to_string()];

        // Add capabilities based on model name patterns
        if model_id.contains("embed") || model_id.contains("embedding") {
            capabilities.push("embedding".to_string());
        }

        if model_id.contains("rerank") || model_id.contains("bge-reranker") {
            capabilities.push("rerank".to_string());
        }

        if model_id.contains("flux")
            || model_id.contains("stable-diffusion")
            || model_id.contains("kolors")
        {
            capabilities.push("image_generation".to_string());
        }

        // Add thinking capability for supported models
        if self
            .config
            .adapter
            .get_model_config(model_id)
            .supports_thinking
        {
            capabilities.push("thinking".to_string());
        }

        capabilities
    }

    /// Get detailed information about a specific model
    async fn get_model_internal(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        // First try to get from list
        let models = self.list_models_internal().await?;

        if let Some(model) = models.into_iter().find(|m| m.id == model_id) {
            return Ok(model);
        }

        // If not found in list, create a basic model info
        Ok(ModelInfo {
            id: model_id.clone(),
            name: Some(model_id.clone()),
            description: Some(format!("{} model: {}", self.config.provider_id, model_id)),
            owned_by: self.config.provider_id.clone(),
            created: None,
            capabilities: self.determine_model_capabilities(&model_id),
            context_window: None,
            max_output_tokens: None,
            input_cost_per_token: None,
            output_cost_per_token: None,
        })
    }
}

#[async_trait]
impl ModelListingCapability for OpenAiCompatibleClient {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.list_models_internal().await
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        self.get_model_internal(model_id).await
    }
}

#[async_trait]
impl ImageGenerationCapability for OpenAiCompatibleClient {
    async fn generate_images(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        if !self.config.adapter.supports_image_generation() {
            return Err(LlmError::UnsupportedOperation(format!(
                "Provider '{}' does not support image generation",
                self.config.provider_id
            )));
        }
        use crate::core::{ProviderContext, ProviderSpec};

        // Build merged extra headers (HttpConfig + custom + adapter)
        let mut extra_headers = self.config.http_config.headers.clone();
        let merge_header_map = |dst: &mut std::collections::HashMap<String, String>,
                                hm: &reqwest::header::HeaderMap| {
            for (name, value) in hm.iter() {
                if let Ok(val) = value.to_str() {
                    dst.insert(name.as_str().to_string(), val.to_string());
                }
            }
        };
        merge_header_map(&mut extra_headers, &self.config.custom_headers);
        merge_header_map(&mut extra_headers, &self.config.adapter.custom_headers());

        let spec = Arc::new(crate::providers::openai_compatible::spec::OpenAiCompatibleSpec);
        let ctx = ProviderContext::new(
            self.config.provider_id.clone(),
            self.config.base_url.clone(),
            Some(self.config.api_key.clone()),
            extra_headers,
        );
        let bundle = spec.choose_image_transformers(&request, &ctx);

        if let Some(opts) = &self.retry_options {
            let http = self.http_client.clone();
            let spec_clone = spec.clone();
            let ctx_clone = ctx.clone();
            let provider_id = self.config.provider_id.clone();
            crate::retry_api::retry_with(
                || {
                    let rqc = request.clone();
                    let http = http.clone();
                    let spec = spec_clone.clone();
                    let ctx = ctx_clone.clone();
                    let provider_id = provider_id.clone();
                    let req_tx_arc = bundle.request.clone();
                    let resp_tx_arc = bundle.response.clone();
                    async move {
                        let exec = HttpImageExecutor {
                            provider_id,
                            http_client: http,
                            request_transformer: req_tx_arc.clone(),
                            response_transformer: resp_tx_arc.clone(),
                            provider_spec: spec,
                            provider_context: ctx,
                            interceptors: vec![],
                            before_send: None,
                            retry_options: None,
                        };
                        ImageExecutor::execute(&exec, rqc).await
                    }
                },
                opts.clone(),
            )
            .await
        } else {
            let exec = HttpImageExecutor {
                provider_id: self.config.provider_id.clone(),
                http_client: self.http_client.clone(),
                request_transformer: bundle.request,
                response_transformer: bundle.response,
                provider_spec: spec,
                provider_context: ctx,
                interceptors: vec![],
                before_send: None,
                retry_options: None,
            };
            exec.execute(request).await
        }
    }

    fn get_supported_sizes(&self) -> Vec<String> {
        self.config.adapter.get_supported_image_sizes()
    }

    fn get_supported_formats(&self) -> Vec<String> {
        self.config.adapter.get_supported_image_formats()
    }

    fn supports_image_editing(&self) -> bool {
        self.config.adapter.supports_image_editing()
    }

    fn supports_image_variations(&self) -> bool {
        self.config.adapter.supports_image_variations()
    }
}

impl LlmClient for OpenAiCompatibleClient {
    fn provider_name(&self) -> &'static str {
        self.config.adapter.provider_id()
    }

    fn supported_models(&self) -> Vec<String> {
        // Return a basic list - could be enhanced with adapter-specific models
        vec![self.config.model.clone()]
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        let adapter_caps = self.config.adapter.capabilities();

        // Convert adapter capabilities to library capabilities
        let mut caps = crate::traits::ProviderCapabilities::new();

        if adapter_caps.chat {
            caps = caps.with_chat();
        }
        if adapter_caps.streaming {
            caps = caps.with_streaming();
        }
        if adapter_caps.embedding {
            caps = caps.with_embedding();
        }
        if adapter_caps.tools {
            caps = caps.with_tools();
        }
        if adapter_caps.vision {
            caps = caps.with_vision();
        }

        caps
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new((*self).clone())
    }

    fn as_embedding_capability(&self) -> Option<&dyn EmbeddingCapability> {
        if self.config.adapter.capabilities().embedding {
            Some(self)
        } else {
            None
        }
    }

    fn as_image_generation_capability(&self) -> Option<&dyn ImageGenerationCapability> {
        if self.config.adapter.supports_image_generation() {
            Some(self)
        } else {
            None
        }
    }

    fn as_rerank_capability(&self) -> Option<&dyn RerankCapability> {
        if self.config.adapter.capabilities().supports("rerank") {
            Some(self)
        } else if self.config.model.contains("rerank") || self.config.model.contains("bge-reranker")
        {
            Some(self)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::openai_compatible::registry::ConfigurableAdapter;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_client_creation() {
        let provider_config = crate::providers::openai_compatible::registry::ProviderConfig {
            id: "test".to_string(),
            name: "Test Provider".to_string(),
            base_url: "https://api.test.com/v1".to_string(),
            field_mappings:
                crate::providers::openai_compatible::registry::ProviderFieldMappings::default(),
            capabilities: vec!["chat".to_string()],
            default_model: Some("test-model".to_string()),
            supports_reasoning: false,
        };

        let config = OpenAiCompatibleConfig::new(
            "test",
            "test-key",
            "https://api.test.com/v1",
            Arc::new(ConfigurableAdapter::new(provider_config)),
        )
        .with_model("test-model");

        let client = OpenAiCompatibleClient::new(config).await.unwrap();
        assert_eq!(client.provider_id(), "test");
        assert_eq!(client.model(), "test-model");
    }

    #[tokio::test]
    async fn test_client_validation() {
        let provider_config = crate::providers::openai_compatible::registry::ProviderConfig {
            id: "test".to_string(),
            name: "Test Provider".to_string(),
            base_url: "https://api.test.com/v1".to_string(),
            field_mappings:
                crate::providers::openai_compatible::registry::ProviderFieldMappings::default(),
            capabilities: vec!["chat".to_string()],
            default_model: Some("test-model".to_string()),
            supports_reasoning: false,
        };

        // Invalid config should fail
        let config = OpenAiCompatibleConfig::new(
            "",
            "test-key",
            "https://api.test.com/v1",
            Arc::new(ConfigurableAdapter::new(provider_config)),
        );

        assert!(OpenAiCompatibleClient::new(config).await.is_err());
    }
}

// Removed legacy rerank parsing; rerank now routed through OpenAI Rerank Standard
