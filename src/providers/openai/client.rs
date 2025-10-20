//! `OpenAI` Client Implementation
//!
//! Main client structure that aggregates all `OpenAI` capabilities.

use async_trait::async_trait;
use secrecy::ExposeSecret;

use crate::client::LlmClient;
use crate::error::LlmError;
use crate::params::OpenAiParams;
use crate::stream::ChatStream;
use crate::traits::*;
use crate::types::*;

use super::models::OpenAiModels;
use super::rerank::OpenAiRerank;
use super::types::OpenAiSpecificParams;
use super::utils::get_default_models;
use crate::executors::chat::ChatExecutor;
use crate::middleware::language_model::LanguageModelMiddleware;
use crate::retry_api::RetryOptions;

/// `OpenAI` Client
pub struct OpenAiClient {
    /// API key and endpoint configuration
    api_key: secrecy::SecretString,
    base_url: String,
    organization: Option<String>,
    project: Option<String>,
    http_config: HttpConfig,
    /// Models capability implementation
    models_capability: OpenAiModels,
    /// Rerank capability implementation
    rerank_capability: OpenAiRerank,
    /// Common parameters
    common_params: CommonParams,
    /// OpenAI-specific parameters
    openai_params: OpenAiParams,
    /// OpenAI-specific configuration
    specific_params: OpenAiSpecificParams,
    /// HTTP client for making requests
    http_client: reqwest::Client,
    /// Tracing configuration
    tracing_config: Option<crate::tracing::TracingConfig>,
    /// Tracing guard to keep tracing system active (not cloned)
    #[allow(dead_code)]
    _tracing_guard: Option<tracing_appender::non_blocking::WorkerGuard>,
    /// Responses API toggle
    use_responses_api: bool,
    /// Previous response id for chaining
    previous_response_id: Option<String>,
    /// Built-in tools for Responses API
    built_in_tools: Vec<crate::types::OpenAiBuiltInTool>,
    /// Web search config
    web_search_config: crate::types::WebSearchConfig,
    /// Unified retry options for chat
    retry_options: Option<RetryOptions>,
    /// Optional HTTP interceptors applied to all chat requests
    http_interceptors: Vec<std::sync::Arc<dyn crate::utils::http_interceptor::HttpInterceptor>>,
    /// Optional model-level middlewares applied before provider mapping
    model_middlewares: Vec<std::sync::Arc<dyn LanguageModelMiddleware>>,
}

impl Clone for OpenAiClient {
    fn clone(&self) -> Self {
        Self {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            organization: self.organization.clone(),
            project: self.project.clone(),
            http_config: self.http_config.clone(),
            models_capability: self.models_capability.clone(),
            rerank_capability: self.rerank_capability.clone(),
            common_params: self.common_params.clone(),
            openai_params: self.openai_params.clone(),
            specific_params: self.specific_params.clone(),
            http_client: self.http_client.clone(),
            tracing_config: self.tracing_config.clone(),
            _tracing_guard: None, // Don't clone the tracing guard
            use_responses_api: self.use_responses_api,
            previous_response_id: self.previous_response_id.clone(),
            built_in_tools: self.built_in_tools.clone(),
            web_search_config: self.web_search_config.clone(),
            retry_options: self.retry_options.clone(),
            http_interceptors: self.http_interceptors.clone(),
            model_middlewares: Vec::new(),
        }
    }
}

impl std::fmt::Debug for OpenAiClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("OpenAiClient");

        debug_struct
            .field("provider_name", &"openai")
            .field("model", &self.common_params.model)
            .field("base_url", &self.base_url)
            .field("temperature", &self.common_params.temperature)
            .field("max_tokens", &self.common_params.max_tokens)
            .field("top_p", &self.common_params.top_p)
            .field("seed", &self.common_params.seed)
            .field("use_responses_api", &self.use_responses_api)
            .field("has_tracing", &self.tracing_config.is_some())
            .field("built_in_tools_count", &self.built_in_tools.len());

        // Only show organization/project if they exist (but don't show the actual values)
        if self.specific_params.organization.is_some() {
            debug_struct.field("has_organization", &true);
        }
        if self.specific_params.project.is_some() {
            debug_struct.field("has_project", &true);
        }

        debug_struct.finish()
    }
}

impl OpenAiClient {
    /// Convenience: configure structured outputs using a JSON object schema.
    /// Only applied to Responses API requests. For chat/completions, this is ignored.
    pub fn with_json_object_schema(mut self, schema: serde_json::Value, strict: bool) -> Self {
        let fmt = serde_json::json!({
            "type": "json_object",
            "json_schema": {
                "schema": schema,
                "strict": strict
            }
        });
        self.specific_params.response_format = Some(fmt);
        self
    }

    /// Convenience: configure structured outputs using a named JSON schema.
    /// Only applied to Responses API requests.
    pub fn with_json_named_schema<S: Into<String>>(
        mut self,
        name: S,
        schema: serde_json::Value,
        strict: bool,
    ) -> Self {
        let fmt = serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": name.into(),
                "schema": schema,
                "strict": strict
            }
        });
        self.specific_params.response_format = Some(fmt);
        self
    }
    /// Build standard OpenAI JSON headers with optional org/project and tracing
    fn build_openai_headers(
        api_key: &secrecy::SecretString,
        organization: &Option<String>,
        project: &Option<String>,
        custom_headers: &std::collections::HashMap<String, String>,
    ) -> Result<reqwest::header::HeaderMap, LlmError> {
        let mut headers = crate::utils::http_headers::ProviderHeaders::openai(
            api_key.expose_secret(),
            organization.as_deref(),
            project.as_deref(),
            custom_headers,
        )?;
        crate::utils::http_headers::inject_tracing_headers(&mut headers);
        Ok(headers)
    }
    /// Creates a new `OpenAI` client with configuration and HTTP client
    pub fn new(config: super::OpenAiConfig, http_client: reqwest::Client) -> Self {
        let specific_params = OpenAiSpecificParams {
            organization: config.organization.clone(),
            project: config.project.clone(),
            ..Default::default()
        };

        let models_capability = OpenAiModels::new(
            config.api_key.clone(),
            config.base_url.clone(),
            http_client.clone(),
            config.organization.clone(),
            config.project.clone(),
            config.http_config.clone(),
        );

        let rerank_capability = OpenAiRerank::new(
            config.api_key.clone(),
            config.base_url.clone(),
            http_client.clone(),
            config.organization.clone(),
            config.project.clone(),
            config.http_config.clone(),
        );

        Self {
            api_key: config.api_key,
            base_url: config.base_url,
            organization: config.organization,
            project: config.project,
            http_config: config.http_config,
            models_capability,
            rerank_capability,
            common_params: config.common_params,
            openai_params: config.openai_params,
            specific_params,
            http_client,
            tracing_config: None,
            _tracing_guard: None,
            use_responses_api: config.use_responses_api,
            previous_response_id: config.previous_response_id,
            built_in_tools: config.built_in_tools,
            web_search_config: config.web_search_config,
            retry_options: None,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
        }
    }

    /// Set the tracing guard to keep tracing system active
    pub(crate) fn set_tracing_guard(
        &mut self,
        guard: Option<tracing_appender::non_blocking::WorkerGuard>,
    ) {
        self._tracing_guard = guard;
    }

    /// Set unified retry options
    pub fn set_retry_options(&mut self, options: Option<RetryOptions>) {
        self.retry_options = options;
    }

    /// Creates a new `OpenAI` client with configuration (for OpenAI-compatible providers)
    pub fn new_with_config(config: super::OpenAiConfig) -> Self {
        let http_client = reqwest::Client::new();
        Self::new(config, http_client)
    }

    /// Set whether to use the Responses API (builder-style)
    /// Decide whether to use Responses API for current client config (auto routes gpt-5*)
    pub(crate) fn should_use_responses(&self) -> bool {
        let cfg = super::config::OpenAiConfig {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            organization: self.organization.clone(),
            project: self.project.clone(),
            common_params: self.common_params.clone(),
            openai_params: self.openai_params.clone(),
            http_config: self.http_config.clone(),
            web_search_config: self.web_search_config.clone(),
            use_responses_api: self.use_responses_api,
            previous_response_id: self.previous_response_id.clone(),
            built_in_tools: self.built_in_tools.clone(),
        };
        super::utils::should_route_responses(&cfg)
    }

    /// Creates a new `OpenAI` client (legacy constructor for backward compatibility)
    #[allow(clippy::too_many_arguments)]
    pub fn new_legacy(
        api_key: String,
        base_url: String,
        http_client: reqwest::Client,
        common_params: CommonParams,
        openai_params: OpenAiParams,
        http_config: HttpConfig,
        organization: Option<String>,
        project: Option<String>,
    ) -> Self {
        let config = super::OpenAiConfig {
            api_key: secrecy::SecretString::from(api_key),
            base_url,
            organization,
            project,
            common_params,
            openai_params,
            http_config,
            web_search_config: crate::types::WebSearchConfig::default(),
            use_responses_api: false,
            previous_response_id: None,
            built_in_tools: Vec::new(),
        };
        let mut client = Self::new(config, http_client);
        // Default: no interceptors
        client.http_interceptors = Vec::new();
        client
    }

    /// Get OpenAI-specific parameters
    pub const fn specific_params(&self) -> &OpenAiSpecificParams {
        &self.specific_params
    }

    /// Get common parameters (for testing and debugging)
    pub const fn common_params(&self) -> &CommonParams {
        &self.common_params
    }

    /// Set previous response id for Responses API chaining.
    pub fn with_previous_response_id<S: Into<String>>(mut self, id: S) -> Self {
        self.previous_response_id = Some(id.into());
        self
    }

    /// Add a single built-in tool for Responses API.
    pub fn with_built_in_tool(mut self, tool: crate::types::OpenAiBuiltInTool) -> Self {
        self.built_in_tools.push(tool);
        self
    }

    /// Add multiple built-in tools for Responses API.
    pub fn with_built_in_tools(mut self, tools: Vec<crate::types::OpenAiBuiltInTool>) -> Self {
        self.built_in_tools.extend(tools);
        self
    }

    /// Enable or disable the OpenAI Responses API routing.
    /// This toggles whether chat requests use `/responses` (when true)
    /// or `/chat/completions` (when false).
    pub fn with_responses_api(mut self, enabled: bool) -> Self {
        self.use_responses_api = enabled;
        self
    }

    /// Install HTTP interceptors for all chat requests.
    pub fn with_http_interceptors(
        mut self,
        interceptors: Vec<std::sync::Arc<dyn crate::utils::http_interceptor::HttpInterceptor>>,
    ) -> Self {
        self.http_interceptors = interceptors;
        self
    }

    /// Install model-level middlewares for chat requests.
    pub fn with_model_middlewares(
        mut self,
        middlewares: Vec<std::sync::Arc<dyn LanguageModelMiddleware>>,
    ) -> Self {
        self.model_middlewares = middlewares;
        self
    }

    // chat_capability removed after executors migration

    /// Update OpenAI-specific parameters
    pub fn with_specific_params(mut self, params: OpenAiSpecificParams) -> Self {
        self.specific_params = params;
        self
    }

    /// Set organization
    pub fn with_organization(mut self, organization: String) -> Self {
        self.specific_params.organization = Some(organization);
        self
    }

    /// Set project
    pub fn with_project(mut self, project: String) -> Self {
        self.specific_params.project = Some(project);
        self
    }

    /// Set response format for structured output
    pub fn with_response_format(mut self, format: serde_json::Value) -> Self {
        self.specific_params.response_format = Some(format);
        self
    }

    /// Set logit bias
    pub fn with_logit_bias(mut self, bias: serde_json::Value) -> Self {
        self.specific_params.logit_bias = Some(bias);
        self
    }

    /// Enable logprobs
    pub const fn with_logprobs(mut self, enabled: bool, top_logprobs: Option<u32>) -> Self {
        self.specific_params.logprobs = Some(enabled);
        self.specific_params.top_logprobs = top_logprobs;
        self
    }

    /// Set presence penalty
    pub const fn with_presence_penalty(mut self, penalty: f32) -> Self {
        self.specific_params.presence_penalty = Some(penalty);
        self
    }

    /// Set frequency penalty
    pub const fn with_frequency_penalty(mut self, penalty: f32) -> Self {
        self.specific_params.frequency_penalty = Some(penalty);
        self
    }

    /// Set user identifier
    pub fn with_user(mut self, user: String) -> Self {
        self.specific_params.user = Some(user);
        self
    }
}

impl OpenAiClient {
    async fn chat_with_tools_inner(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        if self.should_use_responses() {
            // Unified executors path for OpenAI Responses API
            use crate::executors::chat::HttpChatExecutor;
            let request = ChatRequest {
                messages,
                tools,
                common_params: self.common_params.clone(),
                provider_params: Some(ProviderParams::from_openai(self.openai_params.clone())),
                http_config: None,
                web_search: None,
                stream: false,
                telemetry: None,
            };
            let http = self.http_client.clone();
            let base = self.base_url.clone();
            let api_key = self.api_key.clone();
            let org = self.organization.clone();
            let proj = self.project.clone();
            let req_tx = super::transformers::OpenAiResponsesRequestTransformer;
            let resp_tx = super::transformers::OpenAiResponsesResponseTransformer;
            let extra_headers = self.http_config.headers.clone();
            let api_key_clone = api_key.clone();
            let org_clone = org.clone();
            let proj_clone = proj.clone();
            let extra_headers_clone = extra_headers.clone();
            let headers_builder = move || {
                let api_key = api_key_clone.clone();
                let org = org_clone.clone();
                let proj = proj_clone.clone();
                let extra_headers = extra_headers_clone.clone();
                Box::pin(async move {
                    Self::build_openai_headers(&api_key, &org, &proj, &extra_headers)
                }) as std::pin::Pin<Box<dyn std::future::Future<Output = Result<reqwest::header::HeaderMap, crate::error::LlmError>> + Send>>
            };
            let builtins = self.built_in_tools.clone();
            let prev_id = self.previous_response_id.clone();
            let response_format = self.specific_params.response_format.clone();
            let before_send: Option<crate::executors::BeforeSendHook> = if !builtins.is_empty()
                || prev_id.is_some()
                || response_format.is_some()
            {
                let hook = move |body: &serde_json::Value| -> Result<serde_json::Value, LlmError> {
                    let mut out = body.clone();
                    // Merge built-in tools
                    if !builtins.is_empty() {
                        let mut arr = out
                            .get("tools")
                            .and_then(|v| v.as_array().cloned())
                            .unwrap_or_default();
                        for t in &builtins {
                            arr.push(t.to_json());
                        }
                        // Deduplicate by key: function -> keep all; file_search -> include vector_store_ids + max_num_results (or legacy max_results) + extras; others -> by type
                        let mut dedup = Vec::new();
                        let mut seen = std::collections::HashSet::new();
                        for item in arr.into_iter() {
                            let typ = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
                            if typ == "function" {
                                dedup.push(item);
                                continue;
                            }
                            let key = if typ == "file_search" {
                                let mut parts = Vec::new();
                                if let Some(ids) =
                                    item.get("vector_store_ids").and_then(|v| v.as_array())
                                {
                                    let mut s: Vec<String> = ids
                                        .iter()
                                        .filter_map(|x| x.as_str().map(|s| s.to_string()))
                                        .collect();
                                    s.sort();
                                    parts.push(format!("ids={}", s.join("|")));
                                }
                                if let Some(mr) = item
                                    .get("max_num_results")
                                    .and_then(|v| v.as_u64())
                                {
                                    parts.push(format!("mr={}", mr));
                                }
                                if let Some(obj) = item.as_object() {
                                    let mut keys: Vec<&String> = obj.keys().collect();
                                    keys.sort();
                                    for k in keys {
                                        if k == "type"
                                            || k == "vector_store_ids"
                                            || k == "max_num_results"
                                        {
                                            continue;
                                        }
                                        let val = obj.get(k).unwrap();
                                        parts.push(format!("{}={}", k, val));
                                    }
                                }
                                format!("file_search:{}", parts.join(","))
                            } else {
                                typ.to_string()
                            };
                            if seen.insert(key) {
                                dedup.push(item);
                            }
                        }
                        out["tools"] = serde_json::Value::Array(dedup);
                    }
                    // Previous response id (chaining)
                    if let Some(id) = &prev_id {
                        out["previous_response_id"] = serde_json::Value::String(id.clone());
                    }
                    // Structured output response format (if configured)
                    if let Some(fmt) = &response_format {
                        out["response_format"] = fmt.clone();
                    }
                    Ok(out)
                };
                Some(std::sync::Arc::new(hook))
            } else {
                None
            };
            let exec = HttpChatExecutor {
                provider_id: "openai_responses".to_string(),
                http_client: http,
                request_transformer: std::sync::Arc::new(req_tx),
                response_transformer: std::sync::Arc::new(resp_tx),
                stream_transformer: None,
                json_stream_converter: None,
                stream_disable_compression: self.http_config.stream_disable_compression,
                interceptors: self.http_interceptors.clone(),
                middlewares: self.model_middlewares.clone(),
                build_url: Box::new(move |_stream| format!("{}/responses", base)),
                build_headers: std::sync::Arc::new(headers_builder),
                before_send,
            };
            exec.execute(request).await
        } else {
            // Unified executors path for non-streaming chat
            use crate::executors::chat::HttpChatExecutor;
            let request = ChatRequest {
                messages,
                tools,
                common_params: self.common_params.clone(),
                provider_params: Some(ProviderParams::from_openai(self.openai_params.clone())),
                http_config: None,
                web_search: None,
                stream: false,
                telemetry: None,
            };
            let http = self.http_client.clone();
            let base = self.base_url.clone();
            let api_key = self.api_key.clone();
            let org = self.organization.clone();
            let proj = self.project.clone();
            let req_tx = super::transformers::OpenAiRequestTransformer;
            let resp_tx = super::transformers::OpenAiResponseTransformer;
            let extra_headers = self.http_config.headers.clone();
            let api_key_clone = api_key.clone();
            let org_clone = org.clone();
            let proj_clone = proj.clone();
            let extra_headers_clone = extra_headers.clone();
            let headers_builder = move || {
                let api_key = api_key_clone.clone();
                let org = org_clone.clone();
                let proj = proj_clone.clone();
                let extra_headers = extra_headers_clone.clone();
                Box::pin(async move {
                    Self::build_openai_headers(&api_key, &org, &proj, &extra_headers)
                }) as std::pin::Pin<Box<dyn std::future::Future<Output = Result<reqwest::header::HeaderMap, crate::error::LlmError>> + Send>>
            };
            let exec = HttpChatExecutor {
                provider_id: "openai".to_string(),
                http_client: http,
                request_transformer: std::sync::Arc::new(req_tx),
                response_transformer: std::sync::Arc::new(resp_tx),
                stream_transformer: None,
                json_stream_converter: None,
                stream_disable_compression: self.http_config.stream_disable_compression,
                interceptors: self.http_interceptors.clone(),
                middlewares: self.model_middlewares.clone(),
                build_url: Box::new(move |_stream| format!("{}/chat/completions", base)),
                build_headers: std::sync::Arc::new(headers_builder),
                before_send: None,
            };
            exec.execute(request).await
        }
    }
}

#[async_trait]
impl ChatCapability for OpenAiClient {
    /// Chat with tools implementation
    async fn chat_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        if let Some(opts) = &self.retry_options {
            crate::retry_api::retry_with(
                || {
                    let m = messages.clone();
                    let t = tools.clone();
                    async move { self.chat_with_tools_inner(m, t).await }
                },
                opts.clone(),
            )
            .await
        } else {
            self.chat_with_tools_inner(messages, tools).await
        }
    }

    /// Streaming chat with tools
    async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatStream, LlmError> {
        if self.should_use_responses() {
            use crate::executors::chat::HttpChatExecutor;
            let request = ChatRequest {
                messages,
                tools,
                common_params: self.common_params.clone(),
                provider_params: Some(ProviderParams::from_openai(self.openai_params.clone())),
                http_config: None,
                web_search: None,
                stream: true,
                telemetry: None,
            };
            let http = self.http_client.clone();
            let base = self.base_url.clone();
            let api_key = self.api_key.clone();
            let org = self.organization.clone();
            let proj = self.project.clone();
            let req_tx = super::transformers::OpenAiResponsesRequestTransformer;
            let resp_tx = super::transformers::OpenAiResponsesResponseTransformer;
            let converter =
                crate::providers::openai::responses::OpenAiResponsesEventConverter::new();
            let stream_tx = super::transformers::OpenAiResponsesStreamChunkTransformer {
                provider_id: "openai_responses".to_string(),
                inner: converter,
            };
            let extra_headers = self.http_config.headers.clone();
            let api_key_clone = api_key.clone();
            let org_clone = org.clone();
            let proj_clone = proj.clone();
            let extra_headers_clone = extra_headers.clone();
            let headers_builder = move || {
                let api_key = api_key_clone.clone();
                let org = org_clone.clone();
                let proj = proj_clone.clone();
                let extra_headers = extra_headers_clone.clone();
                Box::pin(async move {
                    Self::build_openai_headers(&api_key, &org, &proj, &extra_headers)
                }) as std::pin::Pin<Box<dyn std::future::Future<Output = Result<reqwest::header::HeaderMap, crate::error::LlmError>> + Send>>
            };
            let builtins = self.built_in_tools.clone();
            let prev_id = self.previous_response_id.clone();
            let response_format = self.specific_params.response_format.clone();
            let before_send: Option<crate::executors::BeforeSendHook> = if !builtins.is_empty()
                || prev_id.is_some()
                || response_format.is_some()
            {
                let hook = move |body: &serde_json::Value| -> Result<serde_json::Value, LlmError> {
                    let mut out = body.clone();
                    if !builtins.is_empty() {
                        let mut arr = out
                            .get("tools")
                            .and_then(|v| v.as_array().cloned())
                            .unwrap_or_default();
                        for t in &builtins {
                            arr.push(t.to_json());
                        }
                        // Deduplicate by key (same rule as non-stream)
                        let mut dedup = Vec::new();
                        let mut seen = std::collections::HashSet::new();
                        for item in arr.into_iter() {
                            let typ = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
                            if typ == "function" {
                                dedup.push(item);
                                continue;
                            }
                            let key = if typ == "file_search" {
                                if let Some(ids) =
                                    item.get("vector_store_ids").and_then(|v| v.as_array())
                                {
                                    let mut s: Vec<String> = ids
                                        .iter()
                                        .filter_map(|x| x.as_str().map(|s| s.to_string()))
                                        .collect();
                                    s.sort();
                                    format!("file_search:{}", s.join(","))
                                } else {
                                    typ.to_string()
                                }
                            } else {
                                typ.to_string()
                            };
                            if seen.insert(key) {
                                dedup.push(item);
                            }
                        }
                        out["tools"] = serde_json::Value::Array(dedup);
                    }
                    if let Some(id) = &prev_id {
                        out["previous_response_id"] = serde_json::Value::String(id.clone());
                    }
                    if let Some(fmt) = &response_format {
                        out["response_format"] = fmt.clone();
                    }
                    Ok(out)
                };
                Some(std::sync::Arc::new(hook))
            } else {
                None
            };
            let exec = HttpChatExecutor {
                provider_id: "openai_responses".to_string(),
                http_client: http,
                request_transformer: std::sync::Arc::new(req_tx),
                response_transformer: std::sync::Arc::new(resp_tx),
                stream_transformer: Some(std::sync::Arc::new(stream_tx)),
                json_stream_converter: None,
                stream_disable_compression: self.http_config.stream_disable_compression,
                interceptors: self.http_interceptors.clone(),
                middlewares: self.model_middlewares.clone(),
                build_url: Box::new(move |_stream| format!("{}/responses", base)),
                build_headers: std::sync::Arc::new(headers_builder),
                before_send,
            };
            exec.execute_stream(request).await
        } else {
            // Streaming via HttpChatExecutor and OpenAI stream transformer
            use crate::executors::chat::HttpChatExecutor;
            let request = ChatRequest {
                messages,
                tools,
                common_params: self.common_params.clone(),
                provider_params: Some(ProviderParams::from_openai(self.openai_params.clone())),
                http_config: None,
                web_search: None,
                stream: true,
                telemetry: None,
            };

            let http = self.http_client.clone();
            let base = self.base_url.clone();
            let api_key = self.api_key.clone();
            let org = self.organization.clone();
            let proj = self.project.clone();
            let req_tx = super::transformers::OpenAiRequestTransformer;
            let resp_tx = super::transformers::OpenAiResponseTransformer;

            // Build compat-based event converter via shared OpenAI adapter
            let adapter: std::sync::Arc<
                dyn crate::providers::openai_compatible::adapter::ProviderAdapter,
            > = std::sync::Arc::new(crate::providers::openai::adapter::OpenAiStandardAdapter {
                base_url: base.clone(),
            });
            let compat_cfg =
                crate::providers::openai_compatible::openai_config::OpenAiCompatibleConfig::new(
                    "openai",
                    api_key.expose_secret(),
                    &base,
                    adapter.clone(),
                )
                .with_model(&self.common_params.model);
            let inner =
                crate::providers::openai_compatible::streaming::OpenAiCompatibleEventConverter::new(
                    compat_cfg, adapter,
                );
            let stream_tx = super::transformers::OpenAiStreamChunkTransformer {
                provider_id: "openai".to_string(),
                inner,
            };

            let extra_headers = self.http_config.headers.clone();
            let api_key_clone = api_key.clone();
            let org_clone = org.clone();
            let proj_clone = proj.clone();
            let extra_headers_clone = extra_headers.clone();
            let headers_builder = move || {
                let api_key = api_key_clone.clone();
                let org = org_clone.clone();
                let proj = proj_clone.clone();
                let extra_headers = extra_headers_clone.clone();
                Box::pin(async move {
                    Self::build_openai_headers(&api_key, &org, &proj, &extra_headers)
                }) as std::pin::Pin<Box<dyn std::future::Future<Output = Result<reqwest::header::HeaderMap, crate::error::LlmError>> + Send>>
            };
            let exec = HttpChatExecutor {
                provider_id: "openai".to_string(),
                http_client: http,
                request_transformer: std::sync::Arc::new(req_tx),
                response_transformer: std::sync::Arc::new(resp_tx),
                stream_transformer: Some(std::sync::Arc::new(stream_tx)),
                json_stream_converter: None,
                stream_disable_compression: self.http_config.stream_disable_compression,
                interceptors: self.http_interceptors.clone(),
                middlewares: self.model_middlewares.clone(),
                build_url: Box::new(move |_stream| format!("{}/chat/completions", base)),
                build_headers: std::sync::Arc::new(headers_builder),
                before_send: None,
            };
            exec.execute_stream(request).await
        }
    }

    async fn chat_request(&self, request: ChatRequest) -> Result<ChatResponse, LlmError> {
        use crate::executors::chat::HttpChatExecutor;
        if self.should_use_responses() {
            let http = self.http_client.clone();
            let base = self.base_url.clone();
            let api_key = self.api_key.clone();
            let org = self.organization.clone();
            let proj = self.project.clone();
            let req_tx = super::transformers::OpenAiResponsesRequestTransformer;
            let resp_tx = super::transformers::OpenAiResponsesResponseTransformer;
            let extra_headers = self.http_config.headers.clone();
            let api_key_clone = api_key.clone();
            let org_clone = org.clone();
            let proj_clone = proj.clone();
            let extra_headers_clone = extra_headers.clone();
            let headers_builder = move || {
                let api_key = api_key_clone.clone();
                let org = org_clone.clone();
                let proj = proj_clone.clone();
                let extra_headers = extra_headers_clone.clone();
                Box::pin(async move {
                    Self::build_openai_headers(&api_key, &org, &proj, &extra_headers)
                }) as std::pin::Pin<Box<dyn std::future::Future<Output = Result<reqwest::header::HeaderMap, crate::error::LlmError>> + Send>>
            };
            let builtins = self.built_in_tools.clone();
            let prev_id = self.previous_response_id.clone();
            let response_format = self.specific_params.response_format.clone();
            let before_send: Option<crate::executors::BeforeSendHook> = if !builtins.is_empty()
                || prev_id.is_some()
                || response_format.is_some()
            {
                let hook = move |body: &serde_json::Value| -> Result<serde_json::Value, LlmError> {
                    let mut out = body.clone();
                    if !builtins.is_empty() {
                        let mut arr = out
                            .get("tools")
                            .and_then(|v| v.as_array().cloned())
                            .unwrap_or_default();
                        for t in &builtins { arr.push(t.to_json()); }
                        // simple de-dup by type for responses path (file_search handled elsewhere)
                        let mut seen = std::collections::HashSet::new();
                        let mut dedup = Vec::new();
                        for item in arr.into_iter() {
                            let typ = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
                            if typ == "function" || seen.insert(typ.to_string()) { dedup.push(item); }
                        }
                        out["tools"] = serde_json::Value::Array(dedup);
                    }
                    if let Some(id) = &prev_id { out["previous_response_id"] = serde_json::json!(id); }
                    if let Some(fmt) = &response_format { out["response_format"] = fmt.clone(); }
                    Ok(out)
                };
                Some(std::sync::Arc::new(hook))
            } else { None };
            let exec = HttpChatExecutor {
                provider_id: "openai_responses".to_string(),
                http_client: http,
                request_transformer: std::sync::Arc::new(req_tx),
                response_transformer: std::sync::Arc::new(resp_tx),
                stream_transformer: None,
                json_stream_converter: None,
                stream_disable_compression: self.http_config.stream_disable_compression,
                interceptors: self.http_interceptors.clone(),
                middlewares: self.model_middlewares.clone(),
                build_url: Box::new(move |_stream| format!("{}/responses", base)),
                build_headers: std::sync::Arc::new(headers_builder),
                before_send,
            };
            exec.execute(request).await
        } else {
            let http = self.http_client.clone();
            let base = self.base_url.clone();
            let api_key = self.api_key.clone();
            let org = self.organization.clone();
            let proj = self.project.clone();
            let req_tx = super::transformers::OpenAiRequestTransformer;
            let resp_tx = super::transformers::OpenAiResponseTransformer;
            let extra_headers = self.http_config.headers.clone();
            let api_key_clone = api_key.clone();
            let org_clone = org.clone();
            let proj_clone = proj.clone();
            let extra_headers_clone = extra_headers.clone();
            let headers_builder = move || {
                let api_key = api_key_clone.clone();
                let org = org_clone.clone();
                let proj = proj_clone.clone();
                let extra_headers = extra_headers_clone.clone();
                Box::pin(async move {
                    Self::build_openai_headers(&api_key, &org, &proj, &extra_headers)
                }) as std::pin::Pin<Box<dyn std::future::Future<Output = Result<reqwest::header::HeaderMap, crate::error::LlmError>> + Send>>
            };
            let exec = HttpChatExecutor {
                provider_id: "openai".to_string(),
                http_client: http,
                request_transformer: std::sync::Arc::new(req_tx),
                response_transformer: std::sync::Arc::new(resp_tx),
                stream_transformer: None,
                json_stream_converter: None,
                stream_disable_compression: self.http_config.stream_disable_compression,
                interceptors: self.http_interceptors.clone(),
                middlewares: self.model_middlewares.clone(),
                build_url: Box::new(move |_stream| format!("{}/chat/completions", base)),
                build_headers: std::sync::Arc::new(headers_builder),
                before_send: None,
            };
            exec.execute(request).await
        }
    }

    async fn chat_stream_request(&self, request: ChatRequest) -> Result<ChatStream, LlmError> {
        use crate::executors::chat::HttpChatExecutor;
        if self.should_use_responses() {
            let http = self.http_client.clone();
            let base = self.base_url.clone();
            let api_key = self.api_key.clone();
            let org = self.organization.clone();
            let proj = self.project.clone();
            let req_tx = super::transformers::OpenAiResponsesRequestTransformer;
            let resp_tx = super::transformers::OpenAiResponsesResponseTransformer;
            let converter =
                crate::providers::openai::responses::OpenAiResponsesEventConverter::new();
            let stream_tx = super::transformers::OpenAiResponsesStreamChunkTransformer {
                provider_id: "openai_responses".to_string(),
                inner: converter,
            };
            let extra_headers = self.http_config.headers.clone();
            let api_key_clone = api_key.clone();
            let org_clone = org.clone();
            let proj_clone = proj.clone();
            let extra_headers_clone = extra_headers.clone();
            let headers_builder = move || {
                let api_key = api_key_clone.clone();
                let org = org_clone.clone();
                let proj = proj_clone.clone();
                let extra_headers = extra_headers_clone.clone();
                Box::pin(async move {
                    Self::build_openai_headers(&api_key, &org, &proj, &extra_headers)
                }) as std::pin::Pin<Box<dyn std::future::Future<Output = Result<reqwest::header::HeaderMap, crate::error::LlmError>> + Send>>
            };
            let builtins = self.built_in_tools.clone();
            let prev_id = self.previous_response_id.clone();
            let response_format = self.specific_params.response_format.clone();
            let before_send: Option<crate::executors::BeforeSendHook> = if !builtins.is_empty()
                || prev_id.is_some()
                || response_format.is_some()
            {
                let hook = move |body: &serde_json::Value| -> Result<serde_json::Value, LlmError> {
                    let mut out = body.clone();
                    if !builtins.is_empty() {
                        let mut arr = out
                            .get("tools")
                            .and_then(|v| v.as_array().cloned())
                            .unwrap_or_default();
                        for t in &builtins { arr.push(t.to_json()); }
                        let mut seen = std::collections::HashSet::new();
                        let mut dedup = Vec::new();
                        for item in arr.into_iter() {
                            let typ = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
                            if typ == "function" || seen.insert(typ.to_string()) { dedup.push(item); }
                        }
                        out["tools"] = serde_json::Value::Array(dedup);
                    }
                    if let Some(id) = &prev_id { out["previous_response_id"] = serde_json::json!(id); }
                    if let Some(fmt) = &response_format { out["response_format"] = fmt.clone(); }
                    Ok(out)
                };
                Some(std::sync::Arc::new(hook))
            } else { None };
            let exec = HttpChatExecutor {
                provider_id: "openai_responses".to_string(),
                http_client: http,
                request_transformer: std::sync::Arc::new(req_tx),
                response_transformer: std::sync::Arc::new(resp_tx),
                stream_transformer: Some(std::sync::Arc::new(stream_tx)),
                json_stream_converter: None,
                stream_disable_compression: self.http_config.stream_disable_compression,
                interceptors: self.http_interceptors.clone(),
                middlewares: self.model_middlewares.clone(),
                build_url: Box::new(move |_stream| format!("{}/responses", base)),
                build_headers: std::sync::Arc::new(headers_builder),
                before_send,
            };
            exec.execute_stream(request).await
        } else {
            let http = self.http_client.clone();
            let base = self.base_url.clone();
            let api_key = self.api_key.clone();
            let org = self.organization.clone();
            let proj = self.project.clone();
            let req_tx = super::transformers::OpenAiRequestTransformer;
            let resp_tx = super::transformers::OpenAiResponseTransformer;
            // Build compat-based event converter via shared OpenAI adapter
            let adapter: std::sync::Arc<
                dyn crate::providers::openai_compatible::adapter::ProviderAdapter,
            > = std::sync::Arc::new(crate::providers::openai::adapter::OpenAiStandardAdapter {
                base_url: base.clone(),
            });
            let compat_cfg =
                crate::providers::openai_compatible::openai_config::OpenAiCompatibleConfig::new(
                    "openai",
                    api_key.expose_secret(),
                    &base,
                    adapter.clone(),
                )
                .with_model(&self.common_params.model);
            let inner =
                crate::providers::openai_compatible::streaming::OpenAiCompatibleEventConverter::new(
                    compat_cfg, adapter,
                );
            let stream_tx = super::transformers::OpenAiStreamChunkTransformer {
                provider_id: "openai".to_string(),
                inner,
            };
            let extra_headers = self.http_config.headers.clone();
            let api_key_clone = api_key.clone();
            let org_clone = org.clone();
            let proj_clone = proj.clone();
            let extra_headers_clone = extra_headers.clone();
            let headers_builder = move || {
                let api_key = api_key_clone.clone();
                let org = org_clone.clone();
                let proj = proj_clone.clone();
                let extra_headers = extra_headers_clone.clone();
                Box::pin(async move {
                    Self::build_openai_headers(&api_key, &org, &proj, &extra_headers)
                }) as std::pin::Pin<Box<dyn std::future::Future<Output = Result<reqwest::header::HeaderMap, crate::error::LlmError>> + Send>>
            };
            let exec = HttpChatExecutor {
                provider_id: "openai".to_string(),
                http_client: http,
                request_transformer: std::sync::Arc::new(req_tx),
                response_transformer: std::sync::Arc::new(resp_tx),
                stream_transformer: Some(std::sync::Arc::new(stream_tx)),
                json_stream_converter: None,
                stream_disable_compression: self.http_config.stream_disable_compression,
                interceptors: self.http_interceptors.clone(),
                middlewares: self.model_middlewares.clone(),
                build_url: Box::new(move |_stream| format!("{}/chat/completions", base)),
                build_headers: std::sync::Arc::new(headers_builder),
                before_send: None,
            };
            exec.execute_stream(request).await
        }
    }
}

#[async_trait]
impl ModelListingCapability for OpenAiClient {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.models_capability.list_models().await
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        self.models_capability.get_model(model_id).await
    }
}

#[async_trait]
impl EmbeddingCapability for OpenAiClient {
    async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingResponse, LlmError> {
        use crate::executors::embedding::{EmbeddingExecutor, HttpEmbeddingExecutor};
        let req0 = EmbeddingRequest::new(texts).with_model(self.common_params.model.clone());
        if let Some(opts) = &self.retry_options {
            let http0 = self.http_client.clone();
            let base0 = self.base_url.clone();
            let api_key0 = self.api_key.clone();
            let org0 = self.organization.clone();
            let proj0 = self.project.clone();
            crate::retry_api::retry_with(
                || {
                    let req = req0.clone();
                    let http = http0.clone();
                    let base = base0.clone();
                    let api_key = api_key0.clone();
                    let org = org0.clone();
                    let proj = proj0.clone();
                    async move {
                        let req_tx = super::transformers::OpenAiRequestTransformer;
                        let resp_tx = super::transformers::OpenAiResponseTransformer;
                        let extra_headers = self.http_config.headers.clone();
                        let api_key_clone = api_key.clone();
                        let org_clone = org.clone();
                        let proj_clone = proj.clone();
                        let extra_headers_clone = extra_headers.clone();
                        let headers_builder = move || {
                            let api_key = api_key_clone.clone();
                            let org = org_clone.clone();
                            let proj = proj_clone.clone();
                            let extra_headers = extra_headers_clone.clone();
                            Box::pin(async move {
                                Self::build_openai_headers(&api_key, &org, &proj, &extra_headers)
                            }) as std::pin::Pin<Box<dyn std::future::Future<Output = Result<reqwest::header::HeaderMap, crate::error::LlmError>> + Send>>
                        };
                        let exec = HttpEmbeddingExecutor {
                            provider_id: "openai".to_string(),
                            http_client: http,
                            request_transformer: std::sync::Arc::new(req_tx),
                            response_transformer: std::sync::Arc::new(resp_tx),
                            build_url: Box::new(move |_r| format!("{}/embeddings", base)),
                            build_headers: Box::new(headers_builder),
                            before_send: None,
                        };
                        EmbeddingExecutor::execute(&exec, req).await
                    }
                },
                opts.clone(),
            )
            .await
        } else {
            let http = self.http_client.clone();
            let base = self.base_url.clone();
            let api_key = self.api_key.clone();
            let org = self.organization.clone();
            let proj = self.project.clone();
            let req_tx = super::transformers::OpenAiRequestTransformer;
            let resp_tx = super::transformers::OpenAiResponseTransformer;
            let extra_headers = self.http_config.headers.clone();
            let api_key_clone = api_key.clone();
            let org_clone = org.clone();
            let proj_clone = proj.clone();
            let extra_headers_clone = extra_headers.clone();
            let headers_builder = move || {
                let api_key = api_key_clone.clone();
                let org = org_clone.clone();
                let proj = proj_clone.clone();
                let extra_headers = extra_headers_clone.clone();
                Box::pin(async move {
                    Self::build_openai_headers(&api_key, &org, &proj, &extra_headers)
                }) as std::pin::Pin<Box<dyn std::future::Future<Output = Result<reqwest::header::HeaderMap, crate::error::LlmError>> + Send>>
            };
            let exec = HttpEmbeddingExecutor {
                provider_id: "openai".to_string(),
                http_client: http,
                request_transformer: std::sync::Arc::new(req_tx),
                response_transformer: std::sync::Arc::new(resp_tx),
                build_url: Box::new(move |_r| format!("{}/embeddings", base)),
                build_headers: Box::new(headers_builder),
                before_send: None,
            };
            exec.execute(req0).await
        }
    }

    fn embedding_dimension(&self) -> usize {
        // Return dimension based on model
        let model = if !self.common_params.model.is_empty() {
            &self.common_params.model
        } else {
            "text-embedding-3-small"
        };

        match model {
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            "text-embedding-ada-002" => 1536,
            _ => 1536, // Default fallback
        }
    }

    fn max_tokens_per_embedding(&self) -> usize {
        8192 // OpenAI's current limit
    }

    fn supported_embedding_models(&self) -> Vec<String> {
        vec![
            "text-embedding-3-small".to_string(),
            "text-embedding-3-large".to_string(),
            "text-embedding-ada-002".to_string(),
        ]
    }
}

// Provide extended embedding APIs that accept EmbeddingRequest directly
#[async_trait]
impl EmbeddingExtensions for OpenAiClient {
    async fn embed_with_config(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LlmError> {
        use crate::executors::embedding::{EmbeddingExecutor, HttpEmbeddingExecutor};

        if let Some(opts) = &self.retry_options {
            let http0 = self.http_client.clone();
            let base0 = self.base_url.clone();
            let api_key0 = self.api_key.clone();
            let org0 = self.organization.clone();
            let proj0 = self.project.clone();
            crate::retry_api::retry_with(
                || {
                    let rq = request.clone();
                    let http = http0.clone();
                    let base = base0.clone();
                    let api_key = api_key0.clone();
                    let org = org0.clone();
                    let proj = proj0.clone();
                    async move {
                        let req_tx = super::transformers::OpenAiRequestTransformer;
                        let resp_tx = super::transformers::OpenAiResponseTransformer;
                        let extra_headers = self.http_config.headers.clone();
                        let api_key_clone = api_key.clone();
                        let org_clone = org.clone();
                        let proj_clone = proj.clone();
                        let extra_headers_clone = extra_headers.clone();
                        let headers_builder = move || {
                            let api_key = api_key_clone.clone();
                            let org = org_clone.clone();
                            let proj = proj_clone.clone();
                            let extra_headers = extra_headers_clone.clone();
                            Box::pin(async move {
                                Self::build_openai_headers(&api_key, &org, &proj, &extra_headers)
                            }) as std::pin::Pin<Box<dyn std::future::Future<Output = Result<reqwest::header::HeaderMap, crate::error::LlmError>> + Send>>
                        };
                        let exec = HttpEmbeddingExecutor {
                            provider_id: "openai".to_string(),
                            http_client: http,
                            request_transformer: std::sync::Arc::new(req_tx),
                            response_transformer: std::sync::Arc::new(resp_tx),
                            build_url: Box::new(move |_r| format!("{}/embeddings", base)),
                            build_headers: Box::new(headers_builder),
                            before_send: None,
                        };
                        EmbeddingExecutor::execute(&exec, rq).await
                    }
                },
                opts.clone(),
            )
            .await
        } else {
            let http = self.http_client.clone();
            let base = self.base_url.clone();
            let api_key = self.api_key.clone();
            let org = self.organization.clone();
            let proj = self.project.clone();
            let req_tx = super::transformers::OpenAiRequestTransformer;
            let resp_tx = super::transformers::OpenAiResponseTransformer;
            let extra_headers = self.http_config.headers.clone();
            let api_key_clone = api_key.clone();
            let org_clone = org.clone();
            let proj_clone = proj.clone();
            let extra_headers_clone = extra_headers.clone();
            let headers_builder = move || {
                let api_key = api_key_clone.clone();
                let org = org_clone.clone();
                let proj = proj_clone.clone();
                let extra_headers = extra_headers_clone.clone();
                Box::pin(async move {
                    Self::build_openai_headers(&api_key, &org, &proj, &extra_headers)
                }) as std::pin::Pin<Box<dyn std::future::Future<Output = Result<reqwest::header::HeaderMap, crate::error::LlmError>> + Send>>
            };
            let exec = HttpEmbeddingExecutor {
                provider_id: "openai".to_string(),
                http_client: http,
                request_transformer: std::sync::Arc::new(req_tx),
                response_transformer: std::sync::Arc::new(resp_tx),
                build_url: Box::new(move |_r| format!("{}/embeddings", base)),
                build_headers: Box::new(headers_builder),
                before_send: None,
            };
            exec.execute(request).await
        }
    }
}

#[async_trait]
impl AudioCapability for OpenAiClient {
    fn supported_features(&self) -> &[crate::types::AudioFeature] {
        use crate::types::AudioFeature::*;
        const FEATURES: &[crate::types::AudioFeature] = &[TextToSpeech, SpeechToText];
        FEATURES
    }

    async fn text_to_speech(
        &self,
        request: crate::types::TtsRequest,
    ) -> Result<crate::types::TtsResponse, LlmError> {
        use crate::executors::audio::{AudioExecutor, HttpAudioExecutor};
        let http = self.http_client.clone();
        let base = self.base_url.clone();
        let api_key = self.api_key.clone();
        let org = self.organization.clone();
        let proj = self.project.clone();
        #[derive(Clone)]
        struct OpenAiAudioTransformer;
        impl crate::transformers::audio::AudioTransformer for OpenAiAudioTransformer {
            fn provider_id(&self) -> &str {
                "openai"
            }
            fn build_tts_body(
                &self,
                req: &crate::types::TtsRequest,
            ) -> Result<crate::transformers::audio::AudioHttpBody, LlmError> {
                let model = req.model.clone().unwrap_or_else(|| "tts-1".to_string());
                let voice = req.voice.clone().unwrap_or_else(|| "alloy".to_string());
                let format = req.format.clone().unwrap_or_else(|| "mp3".to_string());
                let mut json = serde_json::json!({
                    "model": model,
                    "input": req.text,
                    "voice": voice,
                    "response_format": format,
                });
                if let Some(s) = req.speed {
                    json["speed"] = serde_json::json!(s);
                }
                if let Some(instr) = req
                    .extra_params
                    .get("instructions")
                    .and_then(|v| v.as_str())
                {
                    json["instructions"] = serde_json::json!(instr);
                }
                Ok(crate::transformers::audio::AudioHttpBody::Json(json))
            }
            fn build_stt_body(
                &self,
                req: &crate::types::SttRequest,
            ) -> Result<crate::transformers::audio::AudioHttpBody, LlmError> {
                let model = req.model.clone().unwrap_or_else(|| "whisper-1".to_string());
                let audio = req.audio_data.clone().ok_or_else(|| {
                    LlmError::InvalidInput("audio_data required for STT".to_string())
                })?;
                let mut form = reqwest::multipart::Form::new()
                    .part(
                        "file",
                        reqwest::multipart::Part::bytes(audio).file_name("audio.mp3"),
                    )
                    .text("model", model)
                    .text("response_format", "json");
                if let Some(lang) = &req.language {
                    form = form.text("language", lang.clone());
                }
                if let Some(grans) = &req.timestamp_granularities {
                    for g in grans {
                        form = form.text("timestamp_granularities[]", g.clone());
                    }
                }
                Ok(crate::transformers::audio::AudioHttpBody::Multipart(form))
            }
            fn tts_endpoint(&self) -> &str {
                "/audio/speech"
            }
            fn stt_endpoint(&self) -> &str {
                "/audio/transcriptions"
            }
            fn parse_stt_response(&self, json: &serde_json::Value) -> Result<String, LlmError> {
                let text = json
                    .get("text")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| LlmError::ParseError("missing 'text' field".to_string()))?;
                Ok(text.to_string())
            }
        }
        let base_clone = base.clone();
        let result_bytes = if let Some(opts) = &self.retry_options {
            crate::retry_api::retry_with(
                || {
                    let rq = request.clone();
                    let http = http.clone();
                    let base = base_clone.clone();
                    let api_key = api_key.clone();
                    let org = org.clone();
                    let proj = proj.clone();
                    async move {
                        let extra_headers = self.http_config.headers.clone();
                        let api_key_clone = api_key.clone();
                        let org_clone = org.clone();
                        let proj_clone = proj.clone();
                        let extra_headers_clone = extra_headers.clone();
                        let headers_builder = move || {
                            let api_key = api_key_clone.clone();
                            let org = org_clone.clone();
                            let proj = proj_clone.clone();
                            let extra_headers = extra_headers_clone.clone();
                            Box::pin(async move {
                                Self::build_openai_headers(&api_key, &org, &proj, &extra_headers)
                            }) as std::pin::Pin<Box<dyn std::future::Future<Output = Result<reqwest::header::HeaderMap, crate::error::LlmError>> + Send>>
                        };
                        let exec = HttpAudioExecutor {
                            provider_id: "openai".to_string(),
                            http_client: http,
                            transformer: std::sync::Arc::new(OpenAiAudioTransformer),
                            build_base_url: Box::new(move || base.clone()),
                            build_headers: Box::new(headers_builder),
                        };
                        AudioExecutor::tts(&exec, rq).await
                    }
                },
                opts.clone(),
            )
            .await?
        } else {
            let extra_headers = self.http_config.headers.clone();
            let api_key_clone = api_key.clone();
            let org_clone = org.clone();
            let proj_clone = proj.clone();
            let extra_headers_clone = extra_headers.clone();
            let headers_builder = move || {
                let api_key = api_key_clone.clone();
                let org = org_clone.clone();
                let proj = proj_clone.clone();
                let extra_headers = extra_headers_clone.clone();
                Box::pin(async move {
                    Self::build_openai_headers(&api_key, &org, &proj, &extra_headers)
                }) as std::pin::Pin<Box<dyn std::future::Future<Output = Result<reqwest::header::HeaderMap, crate::error::LlmError>> + Send>>
            };
            let exec = HttpAudioExecutor {
                provider_id: "openai".to_string(),
                http_client: http,
                transformer: std::sync::Arc::new(OpenAiAudioTransformer),
                build_base_url: Box::new(move || base.clone()),
                build_headers: Box::new(headers_builder),
            };
            AudioExecutor::tts(&exec, request.clone()).await?
        };
        Ok(crate::types::TtsResponse {
            audio_data: result_bytes,
            format: request.format.unwrap_or_else(|| "mp3".to_string()),
            duration: None,
            sample_rate: None,
            metadata: std::collections::HashMap::new(),
        })
    }

    async fn speech_to_text(
        &self,
        request: crate::types::SttRequest,
    ) -> Result<crate::types::SttResponse, LlmError> {
        use crate::executors::audio::{AudioExecutor, HttpAudioExecutor};
        let http = self.http_client.clone();
        let base = self.base_url.clone();
        let api_key = self.api_key.clone();
        let org = self.organization.clone();
        let proj = self.project.clone();
        #[derive(Clone)]
        struct OpenAiAudioTransformer;
        impl crate::transformers::audio::AudioTransformer for OpenAiAudioTransformer {
            fn provider_id(&self) -> &str {
                "openai"
            }
            fn build_tts_body(
                &self,
                _req: &crate::types::TtsRequest,
            ) -> Result<crate::transformers::audio::AudioHttpBody, LlmError> {
                unreachable!()
            }
            fn build_stt_body(
                &self,
                req: &crate::types::SttRequest,
            ) -> Result<crate::transformers::audio::AudioHttpBody, LlmError> {
                let model = req.model.clone().unwrap_or_else(|| "whisper-1".to_string());
                let audio = req.audio_data.clone().ok_or_else(|| {
                    LlmError::InvalidInput("audio_data required for STT".to_string())
                })?;
                let mut form = reqwest::multipart::Form::new()
                    .part(
                        "file",
                        reqwest::multipart::Part::bytes(audio).file_name("audio.mp3"),
                    )
                    .text("model", model)
                    .text("response_format", "json");
                if let Some(lang) = &req.language {
                    form = form.text("language", lang.clone());
                }
                if let Some(grans) = &req.timestamp_granularities {
                    for g in grans {
                        form = form.text("timestamp_granularities[]", g.clone());
                    }
                }
                Ok(crate::transformers::audio::AudioHttpBody::Multipart(form))
            }
            fn tts_endpoint(&self) -> &str {
                "/audio/speech"
            }
            fn stt_endpoint(&self) -> &str {
                "/audio/transcriptions"
            }
            fn parse_stt_response(&self, json: &serde_json::Value) -> Result<String, LlmError> {
                let text = json
                    .get("text")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| LlmError::ParseError("missing 'text' field".to_string()))?;
                Ok(text.to_string())
            }
        }
        let base_clone = base.clone();
        let text = if let Some(opts) = &self.retry_options {
            crate::retry_api::retry_with(
                || {
                    let rq = request.clone();
                    let http = http.clone();
                    let base = base_clone.clone();
                    let api_key = api_key.clone();
                    let org = org.clone();
                    let proj = proj.clone();
                    async move {
                        let api_key_clone = api_key.clone();
                        let org_clone = org.clone();
                        let proj_clone = proj.clone();
                        let headers_builder = move || {
                            let api_key = api_key_clone.clone();
                            let org = org_clone.clone();
                            let proj = proj_clone.clone();
                            Box::pin(async move {
                                let mut headers = reqwest::header::HeaderMap::new();
                                headers.insert(
                                    reqwest::header::AUTHORIZATION,
                                    reqwest::header::HeaderValue::from_str(&format!(
                                        "Bearer {}",
                                        api_key.expose_secret()
                                    ))
                                    .map_err(|e| LlmError::ConfigurationError(e.to_string()))?,
                                );
                                if let Some(org) = &org
                                    && !org.is_empty()
                                {
                                    headers.insert(
                                        "OpenAI-Organization",
                                        reqwest::header::HeaderValue::from_str(org)
                                            .map_err(|e| LlmError::ConfigurationError(e.to_string()))?,
                                    );
                                }
                                if let Some(proj) = &proj
                                    && !proj.is_empty()
                                {
                                    headers.insert(
                                        "OpenAI-Project",
                                        reqwest::header::HeaderValue::from_str(proj)
                                            .map_err(|e| LlmError::ConfigurationError(e.to_string()))?,
                                    );
                                }
                                Ok(headers)
                            }) as std::pin::Pin<Box<dyn std::future::Future<Output = Result<reqwest::header::HeaderMap, crate::error::LlmError>> + Send>>
                        };
                        let exec = HttpAudioExecutor {
                            provider_id: "openai".to_string(),
                            http_client: http,
                            transformer: std::sync::Arc::new(OpenAiAudioTransformer),
                            build_base_url: Box::new(move || base.clone()),
                            build_headers: Box::new(headers_builder),
                        };
                        AudioExecutor::stt(&exec, rq).await
                    }
                },
                opts.clone(),
            )
            .await?
        } else {
            let extra_headers = self.http_config.headers.clone();
            let api_key_clone = api_key.clone();
            let org_clone = org.clone();
            let proj_clone = proj.clone();
            let extra_headers_clone = extra_headers.clone();
            let headers_builder = move || {
                let api_key = api_key_clone.clone();
                let org = org_clone.clone();
                let proj = proj_clone.clone();
                let extra_headers = extra_headers_clone.clone();
                Box::pin(async move {
                    let mut headers = crate::utils::http_headers::ProviderHeaders::openai(
                        api_key.expose_secret(),
                        org.as_deref(),
                        proj.as_deref(),
                        &extra_headers,
                    )?;
                    crate::utils::http_headers::inject_tracing_headers(&mut headers);
                    Ok(headers)
                }) as std::pin::Pin<Box<dyn std::future::Future<Output = Result<reqwest::header::HeaderMap, crate::error::LlmError>> + Send>>
            };
            let exec = HttpAudioExecutor {
                provider_id: "openai".to_string(),
                http_client: http,
                transformer: std::sync::Arc::new(OpenAiAudioTransformer),
                build_base_url: Box::new(move || base.clone()),
                build_headers: Box::new(headers_builder),
            };
            AudioExecutor::stt(&exec, request).await?
        };
        Ok(crate::types::SttResponse {
            text,
            language: None,
            confidence: None,
            words: None,
            duration: None,
            metadata: std::collections::HashMap::new(),
        })
    }
}

impl LlmProvider for OpenAiClient {
    fn provider_name(&self) -> &'static str {
        "openai"
    }

    fn supported_models(&self) -> Vec<String> {
        get_default_models()
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_vision()
            .with_audio()
            .with_embedding()
            .with_custom_feature("structured_output", true)
            .with_custom_feature("batch_processing", true)
            .with_custom_feature("rerank", true)
    }

    fn http_client(&self) -> &reqwest::Client {
        &self.http_client
    }
}

impl LlmClient for OpenAiClient {
    fn provider_name(&self) -> &'static str {
        LlmProvider::provider_name(self)
    }

    fn supported_models(&self) -> Vec<String> {
        LlmProvider::supported_models(self)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        LlmProvider::capabilities(self)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }

    fn as_embedding_capability(&self) -> Option<&dyn EmbeddingCapability> {
        Some(self)
    }

    fn as_audio_capability(&self) -> Option<&dyn AudioCapability> {
        // Provide audio via executor-backed implementation
        Some(self)
    }

    fn as_image_generation_capability(&self) -> Option<&dyn ImageGenerationCapability> {
        // Return the image generation capability
        Some(self)
    }

    fn as_file_management_capability(
        &self,
    ) -> Option<&dyn crate::traits::FileManagementCapability> {
        Some(self)
    }
}

#[async_trait]
impl RerankCapability for OpenAiClient {
    /// Rerank documents based on their relevance to a query
    async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, LlmError> {
        self.rerank_capability.rerank(request).await
    }

    /// Get the maximum number of documents that can be reranked
    fn max_documents(&self) -> Option<u32> {
        self.rerank_capability.max_documents()
    }

    /// Get supported rerank models for this provider
    fn supported_models(&self) -> Vec<String> {
        self.rerank_capability.supported_models()
    }
}

#[async_trait]
impl ImageGenerationCapability for OpenAiClient {
    /// Generate images from text prompts.
    async fn generate_images(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        use crate::executors::image::{HttpImageExecutor, ImageExecutor};
        if let Some(opts) = &self.retry_options {
            let http0 = self.http_client.clone();
            let base0 = self.base_url.clone();
            let api_key0 = self.api_key.clone();
            let org0 = self.organization.clone();
            let proj0 = self.project.clone();
            crate::retry_api::retry_with(
                || {
                    let rq = request.clone();
                    let http = http0.clone();
                    let base = base0.clone();
                    let api_key = api_key0.clone();
                    let org = org0.clone();
                    let proj = proj0.clone();
                    async move {
                        let req_tx = super::transformers::OpenAiRequestTransformer;
                        let resp_tx = super::transformers::OpenAiResponseTransformer;
                        let api_key_clone = api_key.clone();
                        let org_clone = org.clone();
                        let proj_clone = proj.clone();
                        let headers_builder = move || {
                            let api_key = api_key_clone.clone();
                            let org = org_clone.clone();
                            let proj = proj_clone.clone();
                            Box::pin(async move {
                                let mut headers = reqwest::header::HeaderMap::new();
                                headers.insert(
                                    reqwest::header::AUTHORIZATION,
                                    reqwest::header::HeaderValue::from_str(&format!(
                                        "Bearer {}",
                                        api_key.expose_secret()
                                    ))
                                    .map_err(|e| LlmError::ConfigurationError(e.to_string()))?,
                                );
                                headers.insert(
                                    reqwest::header::CONTENT_TYPE,
                                    reqwest::header::HeaderValue::from_static("application/json"),
                                );
                                if let Some(org) = &org
                                    && !org.is_empty()
                                {
                                    headers.insert(
                                        "OpenAI-Organization",
                                        reqwest::header::HeaderValue::from_str(org)
                                            .map_err(|e| LlmError::ConfigurationError(e.to_string()))?,
                                    );
                                }
                                if let Some(proj) = &proj
                                    && !proj.is_empty()
                                {
                                    headers.insert(
                                        "OpenAI-Project",
                                        reqwest::header::HeaderValue::from_str(proj)
                                            .map_err(|e| LlmError::ConfigurationError(e.to_string()))?,
                                    );
                                }
                                Ok(headers)
                            }) as std::pin::Pin<Box<dyn std::future::Future<Output = Result<reqwest::header::HeaderMap, crate::error::LlmError>> + Send>>
                        };
                        let exec = HttpImageExecutor {
                            provider_id: "openai".to_string(),
                            http_client: http,
                            request_transformer: std::sync::Arc::new(req_tx),
                            response_transformer: std::sync::Arc::new(resp_tx),
                            build_url: Box::new(move || format!("{}/images/generations", base)),
                            build_headers: Box::new(headers_builder),
                            before_send: None,
                        };
                        ImageExecutor::execute(&exec, rq).await
                    }
                },
                opts.clone(),
            )
            .await
        } else {
            let http = self.http_client.clone();
            let base = self.base_url.clone();
            let api_key = self.api_key.clone();
            let org = self.organization.clone();
            let proj = self.project.clone();
            let req_tx = super::transformers::OpenAiRequestTransformer;
            let resp_tx = super::transformers::OpenAiResponseTransformer;
            let api_key_clone = api_key.clone();
            let org_clone = org.clone();
            let proj_clone = proj.clone();
            let headers_builder = move || {
                let api_key = api_key_clone.clone();
                let org = org_clone.clone();
                let proj = proj_clone.clone();
                Box::pin(async move {
                    let mut headers = reqwest::header::HeaderMap::new();
                    headers.insert(
                        reqwest::header::AUTHORIZATION,
                        reqwest::header::HeaderValue::from_str(&format!(
                            "Bearer {}",
                            api_key.expose_secret()
                        ))
                        .map_err(|e| LlmError::ConfigurationError(e.to_string()))?,
                    );
                    headers.insert(
                        reqwest::header::CONTENT_TYPE,
                        reqwest::header::HeaderValue::from_static("application/json"),
                    );
                    if let Some(org) = &org
                        && !org.is_empty()
                    {
                        headers.insert(
                            "OpenAI-Organization",
                            reqwest::header::HeaderValue::from_str(org)
                                .map_err(|e| LlmError::ConfigurationError(e.to_string()))?,
                        );
                    }
                    if let Some(proj) = &proj
                        && !proj.is_empty()
                    {
                        headers.insert(
                            "OpenAI-Project",
                            reqwest::header::HeaderValue::from_str(proj)
                                .map_err(|e| LlmError::ConfigurationError(e.to_string()))?,
                        );
                    }
                    Ok(headers)
                }) as std::pin::Pin<Box<dyn std::future::Future<Output = Result<reqwest::header::HeaderMap, crate::error::LlmError>> + Send>>
            };
            let exec = HttpImageExecutor {
                provider_id: "openai".to_string(),
                http_client: http,
                request_transformer: std::sync::Arc::new(req_tx),
                response_transformer: std::sync::Arc::new(resp_tx),
                build_url: Box::new(move || format!("{}/images/generations", base)),
                build_headers: Box::new(headers_builder),
                before_send: None,
            };
            exec.execute(request).await
        }
    }

    /// Edit an existing image with a text prompt.
    async fn edit_image(
        &self,
        request: ImageEditRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        use crate::executors::image::{HttpImageExecutor, ImageExecutor};
        let http = self.http_client.clone();
        let base = format!("{}/images/edits", self.base_url);
        let api_key = self.api_key.clone();
        let org = self.organization.clone();
        let proj = self.project.clone();
        let req_tx = super::transformers::OpenAiRequestTransformer;
        let resp_tx = super::transformers::OpenAiResponseTransformer;
        let base_clone = base.clone();
        if let Some(opts) = &self.retry_options {
            crate::retry_api::retry_with(
                || {
                    let rq = request.clone();
                    let http = http.clone();
                    let base = base_clone.clone();
                    let api_key = api_key.clone();
                    let org = org.clone();
                    let proj = proj.clone();
                    async move {
                        let req_tx = super::transformers::OpenAiRequestTransformer;
                        let resp_tx = super::transformers::OpenAiResponseTransformer;
                        let api_key_clone = api_key.clone();
                        let org_clone = org.clone();
                        let proj_clone = proj.clone();
                        let headers_builder = move || {
                            let api_key = api_key_clone.clone();
                            let org = org_clone.clone();
                            let proj = proj_clone.clone();
                            Box::pin(async move {
                                let mut headers = reqwest::header::HeaderMap::new();
                                headers.insert(
                                    reqwest::header::AUTHORIZATION,
                                    reqwest::header::HeaderValue::from_str(&format!(
                                        "Bearer {}",
                                        api_key.expose_secret()
                                    ))
                                    .map_err(|e| LlmError::ConfigurationError(e.to_string()))?,
                                );
                                if let Some(org) = &org
                                    && !org.is_empty()
                                {
                                    headers.insert(
                                        "OpenAI-Organization",
                                        reqwest::header::HeaderValue::from_str(org)
                                            .map_err(|e| LlmError::ConfigurationError(e.to_string()))?,
                                    );
                                }
                                if let Some(proj) = &proj
                                    && !proj.is_empty()
                                {
                                    headers.insert(
                                        "OpenAI-Project",
                                        reqwest::header::HeaderValue::from_str(proj)
                                            .map_err(|e| LlmError::ConfigurationError(e.to_string()))?,
                                    );
                                }
                                Ok(headers)
                            }) as std::pin::Pin<Box<dyn std::future::Future<Output = Result<reqwest::header::HeaderMap, crate::error::LlmError>> + Send>>
                        };
                        let exec = HttpImageExecutor {
                            provider_id: "openai".to_string(),
                            http_client: http,
                            request_transformer: std::sync::Arc::new(req_tx),
                            response_transformer: std::sync::Arc::new(resp_tx),
                            build_url: Box::new(move || base.clone()),
                            build_headers: Box::new(headers_builder),
                            before_send: None,
                        };
                        ImageExecutor::execute_edit(&exec, rq).await
                    }
                },
                opts.clone(),
            )
            .await
        } else {
            let api_key_clone = api_key.clone();
            let org_clone = org.clone();
            let proj_clone = proj.clone();
            let headers_builder = move || {
                let api_key = api_key_clone.clone();
                let org = org_clone.clone();
                let proj = proj_clone.clone();
                Box::pin(async move {
                    let mut headers = reqwest::header::HeaderMap::new();
                    headers.insert(
                        reqwest::header::AUTHORIZATION,
                        reqwest::header::HeaderValue::from_str(&format!(
                            "Bearer {}",
                            api_key.expose_secret()
                        ))
                        .map_err(|e| LlmError::ConfigurationError(e.to_string()))?,
                    );
                    if let Some(org) = &org
                        && !org.is_empty()
                    {
                        headers.insert(
                            "OpenAI-Organization",
                            reqwest::header::HeaderValue::from_str(org)
                                .map_err(|e| LlmError::ConfigurationError(e.to_string()))?,
                        );
                    }
                    if let Some(proj) = &proj
                        && !proj.is_empty()
                    {
                        headers.insert(
                            "OpenAI-Project",
                            reqwest::header::HeaderValue::from_str(proj)
                                .map_err(|e| LlmError::ConfigurationError(e.to_string()))?,
                        );
                    }
                    Ok(headers)
                }) as std::pin::Pin<Box<dyn std::future::Future<Output = Result<reqwest::header::HeaderMap, crate::error::LlmError>> + Send>>
            };
            let exec = HttpImageExecutor {
                provider_id: "openai".to_string(),
                http_client: http,
                request_transformer: std::sync::Arc::new(req_tx),
                response_transformer: std::sync::Arc::new(resp_tx),
                build_url: Box::new(move || base.clone()),
                build_headers: Box::new(headers_builder),
                before_send: None,
            };
            exec.execute_edit(request).await
        }
    }

    /// Create variations of an existing image.
    async fn create_variation(
        &self,
        request: ImageVariationRequest,
    ) -> Result<ImageGenerationResponse, LlmError> {
        use crate::executors::image::{HttpImageExecutor, ImageExecutor};
        let http = self.http_client.clone();
        let base = format!("{}/images/variations", self.base_url);
        let api_key = self.api_key.clone();
        let org = self.organization.clone();
        let proj = self.project.clone();
        let req_tx = super::transformers::OpenAiRequestTransformer;
        let resp_tx = super::transformers::OpenAiResponseTransformer;
        let base_clone = base.clone();
        if let Some(opts) = &self.retry_options {
            crate::retry_api::retry_with(
                || {
                    let rq = request.clone();
                    let http = http.clone();
                    let base = base_clone.clone();
                    let api_key = api_key.clone();
                    let org = org.clone();
                    let proj = proj.clone();
                    async move {
                        let req_tx = super::transformers::OpenAiRequestTransformer;
                        let resp_tx = super::transformers::OpenAiResponseTransformer;
                        let api_key_clone = api_key.clone();
                        let org_clone = org.clone();
                        let proj_clone = proj.clone();
                        let headers_builder = move || {
                            let api_key = api_key_clone.clone();
                            let org = org_clone.clone();
                            let proj = proj_clone.clone();
                            Box::pin(async move {
                                let mut headers = reqwest::header::HeaderMap::new();
                                headers.insert(
                                    reqwest::header::AUTHORIZATION,
                                    reqwest::header::HeaderValue::from_str(&format!(
                                        "Bearer {}",
                                        api_key.expose_secret()
                                    ))
                                    .map_err(|e| LlmError::ConfigurationError(e.to_string()))?,
                                );
                                if let Some(org) = &org
                                    && !org.is_empty()
                                {
                                    headers.insert(
                                        "OpenAI-Organization",
                                        reqwest::header::HeaderValue::from_str(org)
                                            .map_err(|e| LlmError::ConfigurationError(e.to_string()))?,
                                    );
                                }
                                if let Some(proj) = &proj
                                    && !proj.is_empty()
                                {
                                    headers.insert(
                                        "OpenAI-Project",
                                        reqwest::header::HeaderValue::from_str(proj)
                                            .map_err(|e| LlmError::ConfigurationError(e.to_string()))?,
                                    );
                                }
                                Ok(headers)
                            }) as std::pin::Pin<Box<dyn std::future::Future<Output = Result<reqwest::header::HeaderMap, crate::error::LlmError>> + Send>>
                        };
                        let exec = HttpImageExecutor {
                            provider_id: "openai".to_string(),
                            http_client: http,
                            request_transformer: std::sync::Arc::new(req_tx),
                            response_transformer: std::sync::Arc::new(resp_tx),
                            build_url: Box::new(move || base.clone()),
                            build_headers: Box::new(headers_builder),
                            before_send: None,
                        };
                        ImageExecutor::execute_variation(&exec, rq).await
                    }
                },
                opts.clone(),
            )
            .await
        } else {
            let api_key_clone = api_key.clone();
            let org_clone = org.clone();
            let proj_clone = proj.clone();
            let headers_builder = move || {
                let api_key = api_key_clone.clone();
                let org = org_clone.clone();
                let proj = proj_clone.clone();
                Box::pin(async move {
                    let mut headers = reqwest::header::HeaderMap::new();
                    headers.insert(
                        reqwest::header::AUTHORIZATION,
                        reqwest::header::HeaderValue::from_str(&format!(
                            "Bearer {}",
                            api_key.expose_secret()
                        ))
                        .map_err(|e| LlmError::ConfigurationError(e.to_string()))?,
                    );
                    if let Some(org) = &org
                        && !org.is_empty()
                    {
                        headers.insert(
                            "OpenAI-Organization",
                            reqwest::header::HeaderValue::from_str(org)
                                .map_err(|e| LlmError::ConfigurationError(e.to_string()))?,
                        );
                    }
                    if let Some(proj) = &proj
                        && !proj.is_empty()
                    {
                        headers.insert(
                            "OpenAI-Project",
                            reqwest::header::HeaderValue::from_str(proj)
                                .map_err(|e| LlmError::ConfigurationError(e.to_string()))?,
                        );
                    }
                    Ok(headers)
                }) as std::pin::Pin<Box<dyn std::future::Future<Output = Result<reqwest::header::HeaderMap, crate::error::LlmError>> + Send>>
            };
            let exec = HttpImageExecutor {
                provider_id: "openai".to_string(),
                http_client: http,
                request_transformer: std::sync::Arc::new(req_tx),
                response_transformer: std::sync::Arc::new(resp_tx),
                build_url: Box::new(move || base.clone()),
                build_headers: Box::new(headers_builder),
                before_send: None,
            };
            exec.execute_variation(request).await
        }
    }

    /// Get supported image sizes for this provider.
    fn get_supported_sizes(&self) -> Vec<String> {
        // SiliconFlow supports additional sizes
        if self.base_url.contains("siliconflow.cn") {
            vec![
                "1024x1024".to_string(),
                "960x1280".to_string(),
                "768x1024".to_string(),
                "720x1440".to_string(),
                "720x1280".to_string(),
            ]
        } else {
            // OpenAI supported sizes
            vec![
                "256x256".to_string(),
                "512x512".to_string(),
                "1024x1024".to_string(),
                "1792x1024".to_string(),
                "1024x1792".to_string(),
                "2048x2048".to_string(),
            ]
        }
    }

    /// Get supported response formats for this provider.
    fn get_supported_formats(&self) -> Vec<String> {
        if self.base_url.contains("siliconflow.cn") {
            vec!["url".to_string()]
        } else {
            vec!["url".to_string(), "b64_json".to_string()]
        }
    }

    /// Check if the provider supports image editing.
    fn supports_image_editing(&self) -> bool {
        // SiliconFlow doesn't support editing/variations via OpenAI-compatible paths
        !self.base_url.contains("siliconflow.cn")
    }

    /// Check if the provider supports image variations.
    fn supports_image_variations(&self) -> bool {
        // SiliconFlow doesn't support editing/variations via OpenAI-compatible paths
        !self.base_url.contains("siliconflow.cn")
    }
}

#[async_trait::async_trait]
impl crate::traits::FileManagementCapability for OpenAiClient {
    async fn upload_file(
        &self,
        request: crate::types::FileUploadRequest,
    ) -> Result<crate::types::FileObject, LlmError> {
        let cfg = super::OpenAiConfig {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            organization: self.organization.clone(),
            project: self.project.clone(),
            common_params: self.common_params.clone(),
            openai_params: self.openai_params.clone(),
            http_config: self.http_config.clone(),
            web_search_config: self.web_search_config.clone(),
            use_responses_api: self.use_responses_api,
            previous_response_id: self.previous_response_id.clone(),
            built_in_tools: self.built_in_tools.clone(),
        };
        let files = super::files::OpenAiFiles::new(cfg, self.http_client.clone());
        files.upload_file(request).await
    }

    async fn list_files(
        &self,
        query: Option<crate::types::FileListQuery>,
    ) -> Result<crate::types::FileListResponse, LlmError> {
        let cfg = super::OpenAiConfig {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            organization: self.organization.clone(),
            project: self.project.clone(),
            common_params: self.common_params.clone(),
            openai_params: self.openai_params.clone(),
            http_config: self.http_config.clone(),
            web_search_config: self.web_search_config.clone(),
            use_responses_api: self.use_responses_api,
            previous_response_id: self.previous_response_id.clone(),
            built_in_tools: self.built_in_tools.clone(),
        };
        let files = super::files::OpenAiFiles::new(cfg, self.http_client.clone());
        files.list_files(query).await
    }

    async fn retrieve_file(&self, file_id: String) -> Result<crate::types::FileObject, LlmError> {
        let cfg = super::OpenAiConfig {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            organization: self.organization.clone(),
            project: self.project.clone(),
            common_params: self.common_params.clone(),
            openai_params: self.openai_params.clone(),
            http_config: self.http_config.clone(),
            web_search_config: self.web_search_config.clone(),
            use_responses_api: self.use_responses_api,
            previous_response_id: self.previous_response_id.clone(),
            built_in_tools: self.built_in_tools.clone(),
        };
        let files = super::files::OpenAiFiles::new(cfg, self.http_client.clone());
        files.retrieve_file(file_id).await
    }

    async fn delete_file(
        &self,
        file_id: String,
    ) -> Result<crate::types::FileDeleteResponse, LlmError> {
        let cfg = super::OpenAiConfig {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            organization: self.organization.clone(),
            project: self.project.clone(),
            common_params: self.common_params.clone(),
            openai_params: self.openai_params.clone(),
            http_config: self.http_config.clone(),
            web_search_config: self.web_search_config.clone(),
            use_responses_api: self.use_responses_api,
            previous_response_id: self.previous_response_id.clone(),
            built_in_tools: self.built_in_tools.clone(),
        };
        let files = super::files::OpenAiFiles::new(cfg, self.http_client.clone());
        files.delete_file(file_id).await
    }

    async fn get_file_content(&self, file_id: String) -> Result<Vec<u8>, LlmError> {
        let cfg = super::OpenAiConfig {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            organization: self.organization.clone(),
            project: self.project.clone(),
            common_params: self.common_params.clone(),
            openai_params: self.openai_params.clone(),
            http_config: self.http_config.clone(),
            web_search_config: self.web_search_config.clone(),
            use_responses_api: self.use_responses_api,
            previous_response_id: self.previous_response_id.clone(),
            built_in_tools: self.built_in_tools.clone(),
        };
        let files = super::files::OpenAiFiles::new(cfg, self.http_client.clone());
        files.get_file_content(file_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::openai::OpenAiConfig;
    use crate::providers::openai::transformers;
    use crate::transformers::request::RequestTransformer;
    use crate::utils::http_interceptor::{HttpInterceptor, HttpRequestContext};
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_openai_client_creation() {
        let config = OpenAiConfig::new("test-key");
        let client = OpenAiClient::new(config, reqwest::Client::new());

        assert_eq!(LlmProvider::provider_name(&client), "openai");
        assert!(!LlmProvider::supported_models(&client).is_empty());
    }

    #[test]
    fn test_openai_client_with_specific_params() {
        let config = OpenAiConfig::new("test-key")
            .with_organization("org-123")
            .with_project("proj-456");
        let client = OpenAiClient::new(config, reqwest::Client::new())
            .with_presence_penalty(0.5)
            .with_frequency_penalty(0.3);

        assert_eq!(
            client.specific_params().organization,
            Some("org-123".to_string())
        );
        assert_eq!(
            client.specific_params().project,
            Some("proj-456".to_string())
        );
        assert_eq!(client.specific_params().presence_penalty, Some(0.5));
        assert_eq!(client.specific_params().frequency_penalty, Some(0.3));
    }

    #[test]
    fn test_openai_client_legacy_constructor() {
        let client = OpenAiClient::new_legacy(
            "test-key".to_string(),
            "https://api.openai.com/v1".to_string(),
            reqwest::Client::new(),
            CommonParams::default(),
            OpenAiParams::default(),
            HttpConfig::default(),
            None,
            None,
        );

        assert_eq!(LlmProvider::provider_name(&client), "openai");
        assert!(!LlmProvider::supported_models(&client).is_empty());
    }

    #[test]
    fn test_openai_client_uses_builder_model() {
        let config = OpenAiConfig::new("test-key").with_model("gpt-4");
        let client = OpenAiClient::new(config, reqwest::Client::new());

        // Verify that the client stores the model from the builder
        assert_eq!(client.common_params.model, "gpt-4");
    }

    #[tokio::test]
    async fn test_openai_chat_request_uses_client_model() {
        use crate::types::{ChatMessage, MessageContent, MessageMetadata, MessageRole};

        let config = OpenAiConfig::new("test-key").with_model("gpt-4-test");
        let client = OpenAiClient::new(config, reqwest::Client::new());

        // Create a test message
        let message = ChatMessage {
            role: MessageRole::User,
            content: MessageContent::Text("Hello".to_string()),
            metadata: MessageMetadata::default(),
            tool_calls: None,
            tool_call_id: None,
        };

        // Create a ChatRequest to test the legacy chat method
        let request = ChatRequest {
            messages: vec![message],
            tools: None,
            common_params: client.common_params.clone(),
            provider_params: None,
            http_config: None,
            web_search: None,
            stream: false,
            telemetry: None,
        };

        // Test that the request body includes the correct model (via transformers)
        let tx = transformers::OpenAiRequestTransformer;
        let body = tx.transform_chat(&request).unwrap();
        assert_eq!(body["model"], "gpt-4-test");
    }

    #[test]
    fn responses_builtins_and_previous_id_injected_non_stream() {
        // Build config enabling Responses API with built-in web search and previous_response_id
        let cfg = OpenAiConfig::new("test-key")
            .with_model("gpt-4.1-mini")
            .with_responses_api(true)
            .with_built_in_tool(crate::types::OpenAiBuiltInTool::WebSearch)
            .with_previous_response_id("resp_123");
        let http = reqwest::Client::new();
        let client = OpenAiClient::new(cfg, http);

        // Interceptor to capture transformed JSON body and abort before HTTP send
        struct Capture(Arc<Mutex<Option<serde_json::Value>>>);
        impl HttpInterceptor for Capture {
            fn on_before_send(
                &self,
                _ctx: &HttpRequestContext,
                rb: reqwest::RequestBuilder,
                body: &serde_json::Value,
                _headers: &reqwest::header::HeaderMap,
            ) -> Result<reqwest::RequestBuilder, LlmError> {
                *self.0.lock().unwrap() = Some(body.clone());
                Err(LlmError::InvalidParameter("stop".into()))
            }
        }

        let captured = Arc::new(Mutex::new(None));
        let cap = Capture(captured.clone());
        let client = client.with_http_interceptors(vec![Arc::new(cap)]);

        // Invoke non-stream chat, which should hit interceptor and abort
        let req = vec![crate::types::ChatMessage::user("hi").build()];
        let err = futures::executor::block_on(client.chat(req)).unwrap_err();
        match err {
            LlmError::InvalidParameter(s) => assert_eq!(s, "stop"),
            other => panic!("unexpected: {:?}", other),
        }

        // Assert captured body contains previous_response_id and built-in tool
        let body = captured.lock().unwrap().clone().expect("captured body");
        assert_eq!(
            body.get("previous_response_id")
                .and_then(|v| v.as_str())
                .unwrap_or(""),
            "resp_123"
        );
        let tools = body
            .get("tools")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        assert!(
            tools
                .iter()
                .any(|t| t.get("type").and_then(|s| s.as_str()) == Some("web_search_preview"))
        );
    }

    #[test]
    fn responses_builtins_dedup_non_stream() {
        // Duplicate built-ins should be deduplicated by type
        let cfg = OpenAiConfig::new("test-key")
            .with_model("gpt-4.1-mini")
            .with_responses_api(true)
            .with_built_in_tools(vec![
                crate::types::OpenAiBuiltInTool::WebSearch,
                crate::types::OpenAiBuiltInTool::WebSearch,
            ]);
        let http = reqwest::Client::new();
        let client = OpenAiClient::new(cfg, http);

        struct Capture(std::sync::Arc<std::sync::Mutex<Option<serde_json::Value>>>);
        impl HttpInterceptor for Capture {
            fn on_before_send(
                &self,
                _ctx: &HttpRequestContext,
                rb: reqwest::RequestBuilder,
                body: &serde_json::Value,
                _headers: &reqwest::header::HeaderMap,
            ) -> Result<reqwest::RequestBuilder, LlmError> {
                *self.0.lock().unwrap() = Some(body.clone());
                Err(LlmError::InvalidParameter("stop".into()))
            }
        }
        let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
        let cap = Capture(captured.clone());
        let client = client.with_http_interceptors(vec![std::sync::Arc::new(cap)]);
        let _ = futures::executor::block_on(
            client.chat(vec![crate::types::ChatMessage::user("hi").build()]),
        );
        let body = captured.lock().unwrap().clone().expect("captured body");
        let tools = body
            .get("tools")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let web_count = tools
            .iter()
            .filter(|t| t.get("type").and_then(|s| s.as_str()) == Some("web_search_preview"))
            .count();
        assert_eq!(web_count, 1, "duplicate built-ins must be deduplicated");
    }

    #[test]
    fn responses_file_search_key_includes_max_num_results() {
        // Two file_search entries with same ids but different max_num_results should NOT be deduplicated
        let cfg = OpenAiConfig::new("test-key")
            .with_model("gpt-4.1-mini")
            .with_responses_api(true)
            .with_built_in_tools(vec![
                crate::types::OpenAiBuiltInTool::FileSearchOptions {
                    vector_store_ids: Some(vec!["vs1".into()]),
                    max_num_results: Some(10),
                    ranking_options: None,
                    filters: None,
                },
                crate::types::OpenAiBuiltInTool::FileSearchOptions {
                    vector_store_ids: Some(vec!["vs1".into()]),
                    max_num_results: Some(20),
                    ranking_options: None,
                    filters: None,
                },
            ]);
        let http = reqwest::Client::new();
        let client = OpenAiClient::new(cfg, http);

        struct Capture(std::sync::Arc<std::sync::Mutex<Option<serde_json::Value>>>);
        impl HttpInterceptor for Capture {
            fn on_before_send(
                &self,
                _ctx: &HttpRequestContext,
                _rb: reqwest::RequestBuilder,
                body: &serde_json::Value,
                _headers: &reqwest::header::HeaderMap,
            ) -> Result<reqwest::RequestBuilder, LlmError> {
                *self.0.lock().unwrap() = Some(body.clone());
                Err(LlmError::InvalidParameter("stop".into()))
            }
        }

        let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
        let cap = Capture(captured.clone());
        let client = client.with_http_interceptors(vec![std::sync::Arc::new(cap)]);
        let _ = futures::executor::block_on(
            client.chat(vec![crate::types::ChatMessage::user("hi").build()]),
        );
        let body = captured.lock().unwrap().clone().expect("captured body");
        let tools = body
            .get("tools")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let files: Vec<_> = tools
            .iter()
            .filter(|t| t.get("type").and_then(|s| s.as_str()) == Some("file_search"))
            .collect();
        assert_eq!(
            files.len(),
            2,
            "file_search entries with different max_num_results must both remain"
        );
        let mut maxes: Vec<u64> = files
            .iter()
            .filter_map(|t| t.get("max_num_results").and_then(|v| v.as_u64()))
            .collect();
        maxes.sort();
        assert_eq!(maxes, vec![10, 20]);
    }

    #[test]
    fn responses_file_search_dedup_respects_ranking_options() {
        // Two file_search entries with same ids and max_num_results but different ranking_options should NOT be deduplicated
        let cfg = OpenAiConfig::new("test-key")
            .with_model("gpt-4.1-mini")
            .with_responses_api(true)
            .with_built_in_tools(vec![
                crate::types::OpenAiBuiltInTool::FileSearchOptions {
                    vector_store_ids: Some(vec!["vs1".into()]),
                    max_num_results: Some(10),
                    ranking_options: Some(crate::types::FileSearchRankingOptions { ranker: Some("semantic".into()), score_threshold: Some(0.6) }),
                    filters: None,
                },
                crate::types::OpenAiBuiltInTool::FileSearchOptions {
                    vector_store_ids: Some(vec!["vs1".into()]),
                    max_num_results: Some(10),
                    ranking_options: Some(crate::types::FileSearchRankingOptions { ranker: Some("bm25".into()), score_threshold: Some(0.2) }),
                    filters: None,
                },
            ]);
        let http = reqwest::Client::new();
        let client = OpenAiClient::new(cfg, http);

        struct Capture(std::sync::Arc<std::sync::Mutex<Option<serde_json::Value>>>);
        impl HttpInterceptor for Capture {
            fn on_before_send(
                &self,
                _ctx: &HttpRequestContext,
                _rb: reqwest::RequestBuilder,
                body: &serde_json::Value,
                _headers: &reqwest::header::HeaderMap,
            ) -> Result<reqwest::RequestBuilder, LlmError> {
                *self.0.lock().unwrap() = Some(body.clone());
                Err(LlmError::InvalidParameter("stop".into()))
            }
        }

        let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
        let cap = Capture(captured.clone());
        let client = client.with_http_interceptors(vec![std::sync::Arc::new(cap)]);
        let _ = futures::executor::block_on(
            client.chat(vec![crate::types::ChatMessage::user("hi").build()]),
        );
        let body = captured.lock().unwrap().clone().expect("captured body");
        let tools = body
            .get("tools")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let files: Vec<_> = tools
            .iter()
            .filter(|t| t.get("type").and_then(|s| s.as_str()) == Some("file_search"))
            .collect();
        assert_eq!(files.len(), 2, "file_search with different ranking_options must both remain");
    }

    #[test]
    fn responses_file_search_dedup_respects_filters() {
        // Two file_search entries with same ids/max_num_results but different filters should NOT be deduplicated
        let cfg = OpenAiConfig::new("test-key")
            .with_model("gpt-4.1-mini")
            .with_responses_api(true)
            .with_built_in_tools(vec![
                crate::types::OpenAiBuiltInTool::FileSearchOptions {
                    vector_store_ids: Some(vec!["vs1".into()]),
                    max_num_results: Some(10),
                    ranking_options: None,
                    filters: Some(crate::types::FileSearchFilter::Eq { key: "doctype".into(), value: serde_json::json!("pdf") }),
                },
                crate::types::OpenAiBuiltInTool::FileSearchOptions {
                    vector_store_ids: Some(vec!["vs1".into()]),
                    max_num_results: Some(10),
                    ranking_options: None,
                    filters: Some(crate::types::FileSearchFilter::Eq { key: "doctype".into(), value: serde_json::json!("md") }),
                },
            ]);
        let http = reqwest::Client::new();
        let client = OpenAiClient::new(cfg, http);

        struct Capture(std::sync::Arc<std::sync::Mutex<Option<serde_json::Value>>>);
        impl HttpInterceptor for Capture {
            fn on_before_send(
                &self,
                _ctx: &HttpRequestContext,
                _rb: reqwest::RequestBuilder,
                body: &serde_json::Value,
                _headers: &reqwest::header::HeaderMap,
            ) -> Result<reqwest::RequestBuilder, LlmError> {
                *self.0.lock().unwrap() = Some(body.clone());
                Err(LlmError::InvalidParameter("stop".into()))
            }
        }

        let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
        let cap = Capture(captured.clone());
        let client = client.with_http_interceptors(vec![std::sync::Arc::new(cap)]);
        let _ = futures::executor::block_on(
            client.chat(vec![crate::types::ChatMessage::user("hi").build()]),
        );
        let body = captured.lock().unwrap().clone().expect("captured body");
        let tools = body
            .get("tools")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let files: Vec<_> = tools
            .iter()
            .filter(|t| t.get("type").and_then(|s| s.as_str()) == Some("file_search"))
            .collect();
        assert_eq!(files.len(), 2, "file_search with different filters must both remain");
    }

    #[test]
    fn responses_response_format_injected_non_stream() {
        let cfg = OpenAiConfig::new("test-key")
            .with_model("gpt-4.1-mini")
            .with_responses_api(true);
        let http = reqwest::Client::new();
        let client = OpenAiClient::new(cfg, http).with_json_object_schema(
            serde_json::json!({
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"]
            }),
            true,
        );

        struct Capture(std::sync::Arc<std::sync::Mutex<Option<serde_json::Value>>>);
        impl HttpInterceptor for Capture {
            fn on_before_send(
                &self,
                _ctx: &HttpRequestContext,
                _rb: reqwest::RequestBuilder,
                body: &serde_json::Value,
                _headers: &reqwest::header::HeaderMap,
            ) -> Result<reqwest::RequestBuilder, LlmError> {
                *self.0.lock().unwrap() = Some(body.clone());
                Err(LlmError::InvalidParameter("stop".into()))
            }
        }
        let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
        let cap = Capture(captured.clone());
        let client = client.with_http_interceptors(vec![std::sync::Arc::new(cap)]);
        let _ = futures::executor::block_on(
            client.chat(vec![crate::types::ChatMessage::user("hi").build()]),
        );
        let body = captured.lock().unwrap().clone().expect("captured body");
        let rf = body
            .get("response_format")
            .cloned()
            .expect("has response_format");
        assert_eq!(
            rf.get("type").and_then(|v| v.as_str()).unwrap_or(""),
            "json_object"
        );
        let sch = rf
            .get("json_schema")
            .and_then(|v| v.get("schema"))
            .cloned()
            .expect("schema present");
        assert_eq!(
            sch.get("type").and_then(|v| v.as_str()).unwrap_or(""),
            "object"
        );
    }

    #[test]
    fn responses_response_format_injected_stream() {
        let cfg = OpenAiConfig::new("test-key")
            .with_model("gpt-4.1-mini")
            .with_responses_api(true);
        let http = reqwest::Client::new();
        let client = OpenAiClient::new(cfg, http).with_json_named_schema(
            "User",
            serde_json::json!({
                "type": "object",
                "properties": {"age": {"type": "integer"}},
                "required": ["age"]
            }),
            true,
        );

        struct Capture(std::sync::Arc<std::sync::Mutex<Option<serde_json::Value>>>);
        impl HttpInterceptor for Capture {
            fn on_before_send(
                &self,
                _ctx: &HttpRequestContext,
                _rb: reqwest::RequestBuilder,
                body: &serde_json::Value,
                _headers: &reqwest::header::HeaderMap,
            ) -> Result<reqwest::RequestBuilder, LlmError> {
                *self.0.lock().unwrap() = Some(body.clone());
                Err(LlmError::InvalidParameter("stop".into()))
            }
        }
        let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
        let cap = Capture(captured.clone());
        let client = client.with_http_interceptors(vec![std::sync::Arc::new(cap)]);
        // trigger stream path
        let _ = futures::executor::block_on(
            client.chat_stream(vec![crate::types::ChatMessage::user("hi").build()], None),
        );
        let body = captured.lock().unwrap().clone().expect("captured body");
        let rf = body
            .get("response_format")
            .cloned()
            .expect("has response_format");
        assert_eq!(
            rf.get("type").and_then(|v| v.as_str()).unwrap_or(""),
            "json_schema"
        );
        let name = rf
            .get("json_schema")
            .and_then(|v| v.get("name"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert_eq!(name, "User");
    }
}
