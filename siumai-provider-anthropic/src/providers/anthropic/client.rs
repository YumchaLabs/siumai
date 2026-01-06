//! Anthropic Client Implementation
//!
//! Main client structure that aggregates all Anthropic capabilities.

use async_trait::async_trait;
use backoff::ExponentialBackoffBuilder;
use secrecy::SecretString;
use std::time::Duration;

use crate::client::LlmClient;
use crate::error::LlmError;
use crate::execution::executors::chat::HttpChatExecutor;
// use crate::execution::transformers::{request::RequestTransformer, response::ResponseTransformer};
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::params::AnthropicParams;

use crate::execution::http::interceptor::HttpInterceptor;
use crate::retry_api::RetryOptions;
use crate::traits::*;
use crate::types::*;
use std::sync::Arc;

use super::models::AnthropicModels;
use super::types::AnthropicSpecificParams;
use super::utils::get_default_models;

// Split capability implementations into submodules (no public API changes)
mod chat;

/// Anthropic Client
pub struct AnthropicClient {
    /// API key and endpoint configuration (securely stored)
    api_key: SecretString,
    base_url: String,
    http_client: reqwest::Client,
    http_config: HttpConfig,
    /// Models capability implementation
    models_capability: AnthropicModels,
    /// Common parameters
    common_params: CommonParams,
    /// Anthropic-specific parameters
    anthropic_params: AnthropicParams,
    /// Anthropic-specific configuration
    specific_params: AnthropicSpecificParams,
    /// Tracing configuration
    tracing_config: Option<crate::observability::tracing::TracingConfig>,
    /// Tracing guard to keep tracing system active (retained but not read)
    /// NOTE: Tracing subscriber functionality has been moved to siumai-extras
    #[allow(dead_code)]
    _tracing_guard: Option<()>,
    /// Unified retry options for chat
    retry_options: Option<RetryOptions>,
    /// Optional HTTP interceptors applied to all chat requests
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Optional model-level middlewares applied before provider mapping
    model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
}

impl Clone for AnthropicClient {
    fn clone(&self) -> Self {
        Self {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            http_client: self.http_client.clone(),
            http_config: self.http_config.clone(),
            models_capability: self.models_capability.clone(),
            common_params: self.common_params.clone(),
            anthropic_params: self.anthropic_params.clone(),
            specific_params: self.specific_params.clone(),
            tracing_config: self.tracing_config.clone(),
            _tracing_guard: None, // Don't clone the tracing guard
            retry_options: self.retry_options.clone(),
            http_interceptors: self.http_interceptors.clone(),
            model_middlewares: self.model_middlewares.clone(),
        }
    }
}

impl std::fmt::Debug for AnthropicClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicClient")
            .field("provider_id", &"anthropic")
            .field("model", &self.common_params.model)
            .field("base_url", &self.base_url)
            .field("temperature", &self.common_params.temperature)
            .field("max_tokens", &self.common_params.max_tokens)
            .field("top_p", &self.common_params.top_p)
            .field("seed", &self.common_params.seed)
            .field(
                "beta_features_count",
                &self.specific_params.beta_features.len(),
            )
            .field(
                "thinking_enabled",
                &self
                    .specific_params
                    .thinking_config
                    .as_ref()
                    .map(|c| c.is_enabled())
                    .unwrap_or(false),
            )
            .field(
                "cache_control_enabled",
                &self.specific_params.cache_control.is_some(),
            )
            .field("has_tracing", &self.tracing_config.is_some())
            .finish()
    }
}

impl AnthropicClient {
    /// Creates a new Anthropic client
    pub fn new(
        api_key: String,
        base_url: String,
        http_client: reqwest::Client,
        common_params: CommonParams,
        anthropic_params: AnthropicParams,
        http_config: HttpConfig,
    ) -> Self {
        let api_key = SecretString::from(api_key);
        let specific_params = AnthropicSpecificParams::default();

        let models_capability = AnthropicModels::new(
            api_key.clone(),
            base_url.clone(),
            http_client.clone(),
            http_config.clone(),
        );

        Self {
            api_key,
            base_url,
            http_client,
            http_config,
            models_capability,
            common_params,
            anthropic_params,
            specific_params,
            tracing_config: None,
            _tracing_guard: None,
            retry_options: None,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
        }
    }

    /// Get Anthropic-specific parameters
    pub const fn specific_params(&self) -> &AnthropicSpecificParams {
        &self.specific_params
    }

    /// Get common parameters (for testing and debugging)
    pub const fn common_params(&self) -> &CommonParams {
        &self.common_params
    }

    // Chat capability getter removed after executors migration

    /// Set the tracing configuration
    #[doc(hidden)]
    pub fn set_tracing_config(
        &mut self,
        config: Option<crate::observability::tracing::TracingConfig>,
    ) {
        self.tracing_config = config;
    }

    /// Set unified retry options
    pub fn set_retry_options(&mut self, options: Option<RetryOptions>) {
        self.retry_options = options.map(normalize_anthropic_retry_options);
    }

    /// Install HTTP interceptors for all chat requests.
    pub fn with_http_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.http_interceptors = interceptors;
        self
    }

    /// Install model-level middlewares for chat requests (parameter transforms, etc.).
    pub fn with_model_middlewares(
        mut self,
        middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    ) -> Self {
        self.model_middlewares = middlewares;
        self
    }

    /// Update Anthropic-specific parameters
    pub fn with_specific_params(mut self, params: AnthropicSpecificParams) -> Self {
        self.specific_params = params;
        self
    }

    /// Enable beta features
    pub fn with_beta_features(mut self, features: Vec<String>) -> Self {
        self.specific_params.beta_features = features;
        self
    }

    /// Enable prompt caching
    pub fn with_cache_control(mut self, cache_control: super::cache::CacheControl) -> Self {
        self.specific_params.cache_control = Some(cache_control);
        self
    }

    /// Enable thinking mode with specified budget tokens
    pub fn with_thinking_mode(mut self, budget_tokens: Option<u32>) -> Self {
        let config = budget_tokens.map(super::thinking::ThinkingConfig::enabled);
        self.specific_params.thinking_config = config;
        self
    }

    /// Enable thinking mode with default budget (10k tokens)
    pub fn with_thinking_enabled(mut self) -> Self {
        self.specific_params.thinking_config =
            Some(super::thinking::ThinkingConfig::enabled(10000));
        self
    }

    /// Set custom metadata
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.specific_params.metadata = Some(metadata);
        self
    }

    /// Add a beta feature
    pub fn add_beta_feature(mut self, feature: String) -> Self {
        self.specific_params.beta_features.push(feature);
        self
    }

    /// Enable prompt caching with ephemeral type
    pub fn with_ephemeral_cache(self) -> Self {
        self.with_cache_control(super::cache::CacheControl::ephemeral())
    }
}

fn normalize_anthropic_retry_options(mut options: RetryOptions) -> RetryOptions {
    if matches!(options.backend, crate::retry_api::RetryBackend::Backoff)
        && options.backoff_executor.is_none()
    {
        options.backoff_executor = Some(anthropic_backoff_executor());
    }
    options
}

fn anthropic_backoff_executor() -> crate::retry_api::BackoffRetryExecutor {
    let backoff = ExponentialBackoffBuilder::new()
        .with_initial_interval(Duration::from_millis(1000))
        .with_max_interval(Duration::from_secs(60))
        .with_multiplier(1.5)
        .with_max_elapsed_time(Some(Duration::from_secs(300)))
        .build();
    crate::retry_api::BackoffRetryExecutor::with_backoff(backoff)
}

impl AnthropicClient {
    /// Create provider context for this client
    fn build_context(&self) -> crate::core::ProviderContext {
        use secrecy::ExposeSecret;
        let mut headers = self.http_config.headers.clone();

        // Apply configured beta features as the global `anthropic-beta` header.
        // This makes `with_beta_features` / `add_beta_feature` effective.
        if !self.specific_params.beta_features.is_empty() {
            let mut existing_values: Vec<String> = Vec::new();
            let keys: Vec<String> = headers
                .keys()
                .filter(|k| k.eq_ignore_ascii_case("anthropic-beta"))
                .cloned()
                .collect();
            for k in keys {
                if let Some(v) = headers.remove(&k) {
                    existing_values.push(v);
                }
            }

            let mut merged = String::new();
            if !existing_values.is_empty() {
                merged.push_str(&existing_values.join(","));
            }
            if !merged.is_empty() {
                merged.push(',');
            }
            merged.push_str(&self.specific_params.beta_features.join(","));

            headers.insert("anthropic-beta".to_string(), merged);
        }

        crate::core::ProviderContext::new(
            "anthropic",
            self.base_url.clone(),
            Some(self.api_key.expose_secret().to_string()),
            headers,
        )
    }

    /// Create chat executor using the builder pattern
    fn build_chat_executor(&self, request: &ChatRequest) -> Arc<HttpChatExecutor> {
        use crate::core::ProviderSpec;
        use crate::execution::executors::chat::ChatExecutorBuilder;

        let ctx = self.build_context();
        let spec = Arc::new(crate::providers::anthropic::spec::AnthropicSpec::new());
        let bundle = spec.choose_chat_transformers(request, &ctx);
        let before_send_hook = spec.chat_before_send(request, &ctx);

        // Ensure required Anthropic beta headers are present when using provider-hosted tools.
        let mut middlewares = self.model_middlewares.clone();
        middlewares.insert(0, Arc::new(AnthropicAutoBetaHeadersMiddleware));

        let mut builder = ChatExecutorBuilder::new("anthropic", self.http_client.clone())
            .with_spec(spec)
            .with_context(ctx)
            .with_transformer_bundle(bundle)
            .with_stream_disable_compression(self.http_config.stream_disable_compression)
            .with_interceptors(self.http_interceptors.clone())
            .with_middlewares(middlewares);

        if let Some(hook) = before_send_hook {
            builder = builder.with_before_send(hook);
        }

        if let Some(retry) = self.retry_options.clone() {
            builder = builder.with_retry_options(retry);
        }

        builder.build()
    }
}

#[derive(Clone, Default)]
struct AnthropicAutoBetaHeadersMiddleware;

impl AnthropicAutoBetaHeadersMiddleware {
    fn required_beta_features(req: &ChatRequest) -> Vec<&'static str> {
        let mut out: Vec<&'static str> = Vec::new();

        // Provider-hosted tools -> required betas.
        if let Some(tools) = &req.tools {
            for tool in tools {
                let Tool::ProviderDefined(t) = tool else {
                    continue;
                };
                if t.provider() != Some("anthropic") {
                    continue;
                }
                match t.tool_type() {
                    Some("web_fetch_20250910") => out.push("web-fetch-2025-09-10"),
                    Some("code_execution_20250522") => out.push("code-execution-2025-05-22"),
                    Some("code_execution_20250825") => out.push("code-execution-2025-08-25"),
                    Some("tool_search_regex_20251119") | Some("tool_search_bm25_20251119") => {
                        out.push("advanced-tool-use-2025-11-20")
                    }
                    Some("memory_20250818") => out.push("context-management-2025-06-27"),
                    _ => {}
                }
            }
        }

        // Structured output format -> beta header (Vercel-aligned).
        if matches!(
            req.response_format,
            Some(crate::types::chat::ResponseFormat::Json { .. })
        ) {
            let model = req.common_params.model.as_str();
            let supports_structured_outputs = model.starts_with("claude-sonnet-4-5")
                || model.starts_with("claude-opus-4-5")
                || model.starts_with("claude-haiku-4-5");

            if supports_structured_outputs {
                out.push("structured-outputs-2025-11-13");
            }
        }

        // PDF documents -> required beta (Vercel-aligned).
        let uses_pdf = req.messages.iter().any(|m| match &m.content {
            crate::types::MessageContent::MultiModal(parts) => parts.iter().any(|p| {
                matches!(
                    p,
                    crate::types::ContentPart::File {
                        media_type,
                        ..
                    } if media_type == "application/pdf"
                )
            }),
            _ => false,
        });
        if uses_pdf {
            out.push("pdfs-2024-09-25");
        }

        // Agent skills (Vercel-aligned): requires both `skills` and `files` betas.
        let has_agent_skills = req
            .provider_options_map
            .get("anthropic")
            .and_then(|v| v.as_object())
            .and_then(|o| o.get("container"))
            .and_then(|v| v.as_object())
            .and_then(|o| o.get("skills"))
            .and_then(|v| v.as_array())
            .is_some_and(|arr| !arr.is_empty());
        if has_agent_skills {
            out.push("skills-2025-10-02");
            out.push("files-api-2025-04-14");
        }

        out
    }

    fn merge_beta_values(existing: &str, required: &[&'static str]) -> String {
        use std::collections::HashSet;

        let mut seen: HashSet<String> = HashSet::new();
        let mut out: Vec<String> = Vec::new();

        for raw in existing.split(',') {
            let token = raw.trim();
            if token.is_empty() {
                continue;
            }
            if seen.insert(token.to_string()) {
                out.push(token.to_string());
            }
        }

        for token in required {
            if seen.insert((*token).to_string()) {
                out.push((*token).to_string());
            }
        }

        out.join(",")
    }
}

impl LanguageModelMiddleware for AnthropicAutoBetaHeadersMiddleware {
    fn transform_params(&self, mut req: ChatRequest) -> ChatRequest {
        let required = Self::required_beta_features(&req);
        if required.is_empty() {
            return req;
        }

        let http = req.http_config.get_or_insert_with(HttpConfig::default);

        // Normalize existing header key casing and merge values.
        let mut existing_values: Vec<String> = Vec::new();
        let keys: Vec<String> = http
            .headers
            .keys()
            .filter(|k| k.eq_ignore_ascii_case("anthropic-beta"))
            .cloned()
            .collect();
        for k in keys {
            if let Some(v) = http.headers.remove(&k) {
                existing_values.push(v);
            }
        }
        let existing = existing_values.join(",");
        let merged = Self::merge_beta_values(&existing, &required);
        http.headers.insert("anthropic-beta".to_string(), merged);

        req
    }

    fn post_generate(
        &self,
        req: &ChatRequest,
        mut resp: ChatResponse,
    ) -> Result<ChatResponse, LlmError> {
        let has_agent_skills = req
            .provider_options_map
            .get("anthropic")
            .and_then(|v| v.as_object())
            .and_then(|o| o.get("container"))
            .and_then(|v| v.as_object())
            .and_then(|o| o.get("skills"))
            .and_then(|v| v.as_array())
            .is_some_and(|arr| !arr.is_empty());

        if !has_agent_skills {
            return Ok(resp);
        }

        let has_code_execution_tool = req
            .tools
            .as_ref()
            .map(|tools| {
                tools.iter().any(|tool| {
                    let Tool::ProviderDefined(t) = tool else {
                        return false;
                    };
                    t.provider() == Some("anthropic")
                        && t.tool_type().is_some_and(|ty| {
                            ty == "code_execution_20250522" || ty == "code_execution_20250825"
                        })
                })
            })
            .unwrap_or(false);

        if has_code_execution_tool {
            return Ok(resp);
        }

        let warning = Warning::other("code execution tool is required when using skills");
        match resp.warnings.as_mut() {
            Some(warnings) => warnings.push(warning),
            None => resp.warnings = Some(vec![warning]),
        }

        Ok(resp)
    }
}

#[async_trait]
impl ModelListingCapability for AnthropicClient {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.models_capability.list_models().await
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        self.models_capability.get_model(model_id).await
    }
}

impl LlmClient for AnthropicClient {
    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("anthropic")
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
            .with_custom_feature("prompt_caching", true)
            .with_custom_feature("thinking_mode", true)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn LlmClient> {
        Box::new(self.clone())
    }

    fn as_model_listing_capability(&self) -> Option<&dyn crate::traits::ModelListingCapability> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anthropic_client_creation() {
        let client = AnthropicClient::new(
            "test-key".to_string(),
            "https://api.anthropic.com".to_string(),
            reqwest::Client::new(),
            CommonParams::default(),
            AnthropicParams::default(),
            HttpConfig::default(),
        );

        assert_eq!(
            client.provider_id(),
            std::borrow::Cow::Borrowed("anthropic")
        );
        assert!(!client.supported_models().is_empty());
    }

    #[test]
    fn test_anthropic_client_with_specific_params() {
        let client = AnthropicClient::new(
            "test-key".to_string(),
            "https://api.anthropic.com".to_string(),
            reqwest::Client::new(),
            CommonParams::default(),
            AnthropicParams::default(),
            HttpConfig::default(),
        )
        .with_beta_features(vec!["feature1".to_string(), "feature2".to_string()])
        .with_thinking_enabled()
        .with_ephemeral_cache();

        assert_eq!(client.specific_params().beta_features.len(), 2);
        assert!(client.specific_params().thinking_config.is_some());
        assert!(
            client
                .specific_params()
                .thinking_config
                .as_ref()
                .unwrap()
                .is_enabled()
        );
        assert!(client.specific_params().cache_control.is_some());
    }

    #[test]
    fn test_anthropic_client_beta_features() {
        let client = AnthropicClient::new(
            "test-key".to_string(),
            "https://api.anthropic.com".to_string(),
            reqwest::Client::new(),
            CommonParams::default(),
            AnthropicParams::default(),
            HttpConfig::default(),
        )
        .add_beta_feature("computer-use-2024-10-22".to_string())
        .add_beta_feature("prompt-caching-2024-07-31".to_string());

        assert_eq!(client.specific_params().beta_features.len(), 2);
        assert!(
            client
                .specific_params()
                .beta_features
                .contains(&"computer-use-2024-10-22".to_string())
        );
        assert!(
            client
                .specific_params()
                .beta_features
                .contains(&"prompt-caching-2024-07-31".to_string())
        );
    }

    #[test]
    fn applies_beta_features_to_provider_context_headers() {
        let client = AnthropicClient::new(
            "test-key".to_string(),
            "https://api.anthropic.com".to_string(),
            reqwest::Client::new(),
            CommonParams::default(),
            AnthropicParams::default(),
            HttpConfig::default(),
        )
        .with_beta_features(vec![
            "web-fetch-2025-09-10".to_string(),
            "advanced-tool-use-2025-11-20".to_string(),
        ]);

        let ctx = client.build_context();
        assert_eq!(
            ctx.http_extra_headers
                .get("anthropic-beta")
                .map(|s| s.as_str()),
            Some("web-fetch-2025-09-10,advanced-tool-use-2025-11-20")
        );
    }

    #[test]
    fn beta_middleware_injects_required_headers_for_hosted_tools() {
        let mw = AnthropicAutoBetaHeadersMiddleware;

        let req = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_tools(vec![
                Tool::provider_defined("anthropic.tool_search_regex_20251119", "tool_search"),
                Tool::provider_defined("anthropic.code_execution_20250522", "code_execution"),
            ])
            .with_http_config(
                HttpConfig::builder()
                    .header("Anthropic-Beta", "web-fetch-2025-09-10")
                    .build(),
            );

        let out = mw.transform_params(req);
        let val = out
            .http_config
            .as_ref()
            .and_then(|hc| hc.headers.get("anthropic-beta"))
            .cloned()
            .unwrap_or_default();

        assert_eq!(
            val,
            "web-fetch-2025-09-10,advanced-tool-use-2025-11-20,code-execution-2025-05-22"
        );
    }

    #[test]
    fn beta_middleware_injects_skills_and_files_betas_for_agent_skills() {
        let mw = AnthropicAutoBetaHeadersMiddleware;

        let req = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_tools(vec![Tool::provider_defined(
                "anthropic.code_execution_20250825",
                "code_execution",
            )])
            .provider_option(
                "anthropic",
                serde_json::json!({
                    "container": {
                        "skills": [
                            { "type": "anthropic", "skillId": "pptx", "version": "latest" }
                        ]
                    }
                }),
            );

        let out = mw.transform_params(req);
        let val = out
            .http_config
            .as_ref()
            .and_then(|hc| hc.headers.get("anthropic-beta"))
            .cloned()
            .unwrap_or_default();

        assert_eq!(
            val,
            "code-execution-2025-08-25,skills-2025-10-02,files-api-2025-04-14"
        );
    }

    #[test]
    fn adds_warning_when_agent_skills_used_without_code_execution_tool() {
        let mw = AnthropicAutoBetaHeadersMiddleware;

        let req = ChatRequest::new(vec![ChatMessage::user("hi").build()]).provider_option(
            "anthropic",
            serde_json::json!({
                "container": {
                    "skills": [
                        { "type": "anthropic", "skillId": "pptx", "version": "latest" }
                    ]
                }
            }),
        );

        let base = ChatResponse::new(crate::types::MessageContent::Text("ok".to_string()));
        let out = mw.post_generate(&req, base).expect("post_generate");
        let warnings = out.warnings.expect("warnings");

        assert_eq!(
            warnings,
            vec![Warning::other(
                "code execution tool is required when using skills"
            )]
        );
    }

    #[test]
    fn beta_middleware_injects_pdfs_beta_for_pdf_document_parts() {
        let mw = AnthropicAutoBetaHeadersMiddleware;

        let msg = ChatMessage::user("")
            .with_content_parts(vec![crate::types::ContentPart::file_url(
                "https://example.com/a.pdf",
                "application/pdf",
            )])
            .build();

        let req = ChatRequest::new(vec![msg]);
        let out = mw.transform_params(req);

        let val = out
            .http_config
            .as_ref()
            .and_then(|hc| hc.headers.get("anthropic-beta"))
            .cloned()
            .unwrap_or_default();

        assert_eq!(val, "pdfs-2024-09-25");
    }
}
