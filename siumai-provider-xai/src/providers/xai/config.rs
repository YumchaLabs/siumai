//! `xAI` configuration.
//!
//! Provider-owned config-first surface for the xAI wrapper.

use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::http::transport::HttpTransport;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::provider_options::{
    XaiChatReasoningEffort, XaiImageOptions, XaiOptions, XaiReasoningSummary, XaiResponseInclude,
    XaiResponsesOptions, XaiSearchParameters, XaiVideoOptions,
};
use crate::types::{CommonParams, HttpConfig, ProviderOptionsMap};
use secrecy::{ExposeSecret, SecretString};
use std::collections::HashMap;
use std::sync::Arc;

fn rename_field(obj: &mut serde_json::Map<String, serde_json::Value>, from: &str, to: &str) {
    if let Some(value) = obj.remove(from) {
        obj.entry(to.to_string()).or_insert(value);
    }
}

fn take_field(
    obj: &mut serde_json::Map<String, serde_json::Value>,
    keys: &[&str],
) -> Option<serde_json::Value> {
    for key in keys {
        if let Some(value) = obj.remove(*key) {
            return Some(value);
        }
    }
    None
}

fn insert_xai_reasoning_effort(
    obj: &mut serde_json::Map<String, serde_json::Value>,
    effort: &'static str,
) {
    obj.insert(
        "reasoning_effort".to_string(),
        serde_json::Value::String(effort.to_string()),
    );
}

fn normalize_xai_search_parameters(value: &mut serde_json::Value) {
    let Some(obj) = value.as_object_mut() else {
        return;
    };

    rename_field(obj, "returnCitations", "return_citations");
    rename_field(obj, "maxSearchResults", "max_search_results");
    rename_field(obj, "fromDate", "from_date");
    rename_field(obj, "toDate", "to_date");

    if let Some(sources) = obj
        .get_mut("sources")
        .and_then(|value| value.as_array_mut())
    {
        for source in sources {
            let Some(source_obj) = source.as_object_mut() else {
                continue;
            };

            rename_field(source_obj, "allowedWebsites", "allowed_websites");
            rename_field(source_obj, "excludedWebsites", "excluded_websites");
            rename_field(source_obj, "safeSearch", "safe_search");
            rename_field(source_obj, "excludedXHandles", "excluded_x_handles");
            rename_field(source_obj, "includedXHandles", "included_x_handles");
            rename_field(source_obj, "postFavoriteCount", "post_favorite_count");
            rename_field(source_obj, "postViewCount", "post_view_count");
            rename_field(source_obj, "xHandles", "x_handles");

            if source_obj.get("included_x_handles").is_none() {
                if let Some(value) = source_obj.remove("x_handles") {
                    source_obj.insert("included_x_handles".to_string(), value);
                }
            } else {
                source_obj.remove("x_handles");
            }
        }
    }
}

fn normalize_xai_chat_defaults(
    mut params: HashMap<String, serde_json::Value>,
) -> HashMap<String, serde_json::Value> {
    let mut obj = serde_json::Map::from_iter(params.drain());

    rename_field(&mut obj, "reasoningEffort", "reasoning_effort");
    rename_field(&mut obj, "reasoningSummary", "reasoning_summary");
    rename_field(&mut obj, "searchParameters", "search_parameters");
    rename_field(&mut obj, "topLogprobs", "top_logprobs");
    rename_field(&mut obj, "previousResponseId", "previous_response_id");

    let legacy_reasoning_enabled = take_field(
        &mut obj,
        &[
            "enableReasoning",
            "enable_reasoning",
            "enableThinking",
            "enable_thinking",
        ],
    )
    .and_then(|value| value.as_bool());
    for key in [
        "enableReasoning",
        "enable_reasoning",
        "enableThinking",
        "enable_thinking",
    ] {
        obj.remove(key);
    }
    let legacy_reasoning_budget = take_field(
        &mut obj,
        &[
            "reasoningBudget",
            "reasoning_budget",
            "thinkingBudget",
            "thinking_budget",
        ],
    );
    for key in [
        "reasoningBudget",
        "reasoning_budget",
        "thinkingBudget",
        "thinking_budget",
    ] {
        obj.remove(key);
    }

    if obj.get("reasoning_effort").is_none() {
        if legacy_reasoning_budget.is_some() {
            insert_xai_reasoning_effort(&mut obj, "high");
        } else if matches!(legacy_reasoning_enabled, Some(true)) {
            insert_xai_reasoning_effort(&mut obj, "low");
        }
    }

    if let Some(value) = obj.get_mut("search_parameters") {
        normalize_xai_search_parameters(value);
    }

    if obj.get("top_logprobs").is_some() {
        obj.insert("logprobs".to_string(), serde_json::json!(true));
    }

    obj.remove("reasoning_summary");
    obj.remove("previous_response_id");
    obj.remove("include");
    obj.remove("store");

    HashMap::from_iter(obj)
}

/// `xAI` configuration (provider layer).
#[derive(Clone)]
pub struct XaiConfig {
    /// API key (securely stored).
    pub api_key: SecretString,
    /// Base URL for the xAI API.
    pub base_url: String,
    /// Common parameters shared across providers.
    pub common_params: CommonParams,
    /// HTTP configuration.
    pub http_config: HttpConfig,
    /// Optional custom HTTP transport.
    pub http_transport: Option<Arc<dyn HttpTransport>>,
    /// Optional HTTP interceptors applied to all requests built from this config.
    pub http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    /// Optional model-level middlewares applied before provider mapping (chat only).
    pub model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    /// Provider-specific request defaults merged into chat requests.
    pub provider_specific_config: HashMap<String, serde_json::Value>,
    /// Non-chat default provider options merged into request `providerOptions`.
    pub default_provider_options_map: ProviderOptionsMap,
}

impl std::fmt::Debug for XaiConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut ds = f.debug_struct("XaiConfig");
        ds.field("base_url", &self.base_url)
            .field("common_params", &self.common_params)
            .field("http_config", &self.http_config);

        if !self.api_key.expose_secret().is_empty() {
            ds.field("has_api_key", &true);
        }
        if self.http_transport.is_some() {
            ds.field("has_http_transport", &true);
        }
        if !self.provider_specific_config.is_empty() {
            ds.field("provider_specific_config", &self.provider_specific_config);
        }
        if !self.default_provider_options_map.is_empty() {
            ds.field(
                "default_provider_options_map",
                &self.default_provider_options_map,
            );
        }

        ds.finish()
    }
}

impl XaiConfig {
    /// Default xAI API base URL.
    pub const DEFAULT_BASE_URL: &'static str = "https://api.x.ai/v1";

    /// Create a new config with the given API key.
    pub fn new<S: Into<String>>(api_key: S) -> Self {
        Self {
            api_key: SecretString::from(api_key.into()),
            base_url: Self::DEFAULT_BASE_URL.to_string(),
            common_params: CommonParams::default(),
            http_config: HttpConfig::default(),
            http_transport: None,
            http_interceptors: Vec::new(),
            model_middlewares: Vec::new(),
            provider_specific_config: HashMap::new(),
            default_provider_options_map: ProviderOptionsMap::default(),
        }
    }

    /// Create config from `XAI_API_KEY`.
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("XAI_API_KEY")
            .map_err(|_| LlmError::MissingApiKey("xAI API key not provided".to_string()))?;
        Ok(Self::new(api_key))
    }

    pub fn with_base_url<S: Into<String>>(mut self, url: S) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.common_params.model = model.into();
        self
    }

    pub fn with_api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = SecretString::from(api_key.into());
        self
    }

    pub fn with_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.http_config.headers.extend(headers);
        self
    }

    pub fn with_header<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.http_config.headers.insert(key.into(), value.into());
        self
    }

    pub const fn with_temperature(mut self, temperature: f64) -> Self {
        self.common_params.temperature = Some(temperature);
        self
    }

    pub const fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.common_params.max_tokens = Some(max_tokens);
        self
    }

    pub const fn with_top_p(mut self, top_p: f64) -> Self {
        self.common_params.top_p = Some(top_p);
        self
    }

    pub fn with_stop_sequences(mut self, stop: Vec<String>) -> Self {
        self.common_params.stop_sequences = Some(stop);
        self
    }

    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.common_params.seed = Some(seed);
        self
    }

    pub fn with_http_config(mut self, http: HttpConfig) -> Self {
        self.http_config = http;
        self
    }

    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.http_config.timeout = Some(timeout);
        self
    }

    pub fn with_connect_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.http_config.connect_timeout = Some(timeout);
        self
    }

    pub fn with_http_stream_disable_compression(mut self, disable: bool) -> Self {
        self.http_config.stream_disable_compression = disable;
        self
    }

    pub fn with_http_transport(mut self, transport: Arc<dyn HttpTransport>) -> Self {
        self.http_transport = Some(transport);
        self
    }

    pub fn with_http_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.http_interceptors = interceptors;
        self
    }

    pub fn with_http_interceptor(mut self, interceptor: Arc<dyn HttpInterceptor>) -> Self {
        self.http_interceptors.push(interceptor);
        self
    }

    pub fn with_model_middlewares(
        mut self,
        middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    ) -> Self {
        self.model_middlewares = middlewares;
        self
    }

    pub fn with_provider_specific_config(
        mut self,
        params: HashMap<String, serde_json::Value>,
    ) -> Self {
        self.provider_specific_config.extend(params);
        self
    }

    pub fn with_provider_specific_param(
        mut self,
        key: impl Into<String>,
        value: serde_json::Value,
    ) -> Self {
        self.provider_specific_config.insert(key.into(), value);
        self
    }

    /// Replace the full non-chat default provider options map.
    pub fn with_provider_options_map(mut self, map: ProviderOptionsMap) -> Self {
        self.default_provider_options_map = map;
        self
    }

    fn merge_non_chat_xai_provider_options<T: serde::Serialize>(mut self, options: T) -> Self {
        if let Ok(serde_json::Value::Object(obj)) = serde_json::to_value(options) {
            let mut overrides = ProviderOptionsMap::default();
            overrides.insert("xai", serde_json::Value::Object(obj));
            self.default_provider_options_map.merge_overrides(overrides);
        }
        self
    }

    /// Merge typed xAI options into config defaults.
    fn merge_serialized_xai_provider_options<T: serde::Serialize>(mut self, options: T) -> Self {
        if let Ok(serde_json::Value::Object(obj)) = serde_json::to_value(options) {
            self.provider_specific_config
                .extend(obj.into_iter().filter(|(_, value)| !value.is_null()));
        }
        self
    }

    pub fn with_xai_options(self, options: XaiOptions) -> Self {
        self.merge_serialized_xai_provider_options(options)
    }

    /// Merge xAI image defaults into non-chat request `providerOptions`.
    pub fn with_xai_image_options(self, options: XaiImageOptions) -> Self {
        self.merge_non_chat_xai_provider_options(options)
    }

    /// Merge xAI video defaults into non-chat request `providerOptions`.
    pub fn with_xai_video_options(self, options: XaiVideoOptions) -> Self {
        self.merge_non_chat_xai_provider_options(options)
    }

    /// Set a coarse xAI reasoning default.
    ///
    /// xAI does not expose DeepSeek/OpenRouter-style `enable_reasoning`; AI SDK lowers
    /// reasoning to `reasoning_effort`. This compatibility helper maps `true` to the
    /// conservative `low` effort and removes any prior coarse default for `false`.
    pub fn with_reasoning(mut self, enable: bool) -> Self {
        self.provider_specific_config.remove("enableReasoning");
        self.provider_specific_config.remove("enable_reasoning");
        self.provider_specific_config.remove("enableThinking");
        self.provider_specific_config.remove("enable_thinking");
        self.provider_specific_config.remove("reasoningBudget");
        self.provider_specific_config.remove("reasoning_budget");
        self.provider_specific_config.remove("thinkingBudget");
        self.provider_specific_config.remove("thinking_budget");

        if enable {
            self.provider_specific_config
                .insert("reasoningEffort".to_string(), serde_json::json!("low"));
        } else {
            self.provider_specific_config.remove("reasoningEffort");
            self.provider_specific_config.remove("reasoning_effort");
        }
        self
    }

    /// Compatibility helper for legacy budget-shaped APIs.
    ///
    /// AI SDK's xAI provider has no token-budget option; the closest supported xAI
    /// control is `reasoning_effort`, so this maps to `high`.
    pub fn with_reasoning_budget(self, _budget: i32) -> Self {
        self.with_reasoning_effort(XaiChatReasoningEffort::High)
    }

    /// Set xAI-specific reasoning effort.
    pub fn with_reasoning_effort(self, effort: impl Into<XaiChatReasoningEffort>) -> Self {
        self.with_xai_options(XaiOptions::new().with_reasoning_effort(effort))
    }

    /// Set xAI-specific reasoning summary verbosity.
    pub fn with_reasoning_summary(self, summary: impl Into<XaiReasoningSummary>) -> Self {
        self.merge_serialized_xai_provider_options(
            XaiResponsesOptions::new().with_reasoning_summary(summary),
        )
    }

    /// Enable or disable logprobs by default.
    pub fn with_logprobs(self, enabled: bool) -> Self {
        self.with_xai_options(XaiOptions::new().with_logprobs(enabled))
    }

    /// Set top-logprobs by default.
    pub fn with_top_logprobs(self, count: u32) -> Self {
        self.with_xai_options(XaiOptions::new().with_top_logprobs(count))
    }

    /// Enable or disable parallel function calling by default.
    pub fn with_parallel_function_calling(self, enabled: bool) -> Self {
        self.with_xai_options(XaiOptions::new().with_parallel_function_calling(enabled))
    }

    /// Control response storage for Responses-style APIs.
    pub fn with_store(self, store: bool) -> Self {
        self.merge_serialized_xai_provider_options(XaiResponsesOptions::new().with_store(store))
    }

    /// Continue from a previous response id by default.
    pub fn with_previous_response(self, response_id: impl Into<String>) -> Self {
        self.merge_serialized_xai_provider_options(
            XaiResponsesOptions::new().with_previous_response(response_id),
        )
    }

    /// Include extra response sections by default.
    pub fn with_include<I, S>(self, include: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<XaiResponseInclude>,
    {
        self.merge_serialized_xai_provider_options(XaiResponsesOptions::new().with_include(include))
    }

    /// Set xAI-specific search parameters.
    pub fn with_search_parameters(self, params: XaiSearchParameters) -> Self {
        self.with_xai_options(XaiOptions::new().with_search(params))
    }

    /// Enable xAI search defaults.
    pub fn with_default_search(self) -> Self {
        self.with_search_parameters(XaiSearchParameters::default())
    }

    pub fn validate(&self) -> Result<(), LlmError> {
        if self.api_key.expose_secret().is_empty() {
            return Err(LlmError::ConfigurationError(
                "API key cannot be empty".to_string(),
            ));
        }
        if self.base_url.is_empty() {
            return Err(LlmError::ConfigurationError(
                "Base URL cannot be empty".to_string(),
            ));
        }
        if !self.base_url.starts_with("http://") && !self.base_url.starts_with("https://") {
            return Err(LlmError::ConfigurationError(
                "Base URL must start with http:// or https://".to_string(),
            ));
        }
        if self.common_params.model.is_empty() {
            return Err(LlmError::ConfigurationError(
                "Model is required for xAI config".to_string(),
            ));
        }
        Ok(())
    }

    /// Build a provider-owned config from an OpenAI-compatible config.
    pub fn from_compatible_config(
        config: siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleConfig,
    ) -> Result<Self, LlmError> {
        use siumai_provider_openai_compatible::providers::openai_compatible::RequestType;

        let mut provider_specific_config = HashMap::new();
        let mut request_defaults = serde_json::json!({});
        config.adapter.transform_request_params(
            &mut request_defaults,
            &config.model,
            RequestType::Chat,
        )?;
        if let Some(obj) = request_defaults.as_object() {
            provider_specific_config.extend(obj.iter().map(|(k, v)| (k.clone(), v.clone())));
        }

        Ok(Self {
            api_key: SecretString::from(config.api_key),
            base_url: config.base_url,
            common_params: config.common_params,
            http_config: config.http_config,
            http_transport: config.http_transport,
            http_interceptors: config.http_interceptors,
            model_middlewares: config.model_middlewares,
            provider_specific_config,
            default_provider_options_map: ProviderOptionsMap::default(),
        })
    }

    /// Convert to the OpenAI-compatible config used by the execution backend.
    pub fn into_compatible_config(
        self,
    ) -> Result<
        siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleConfig,
        LlmError,
    > {
        use siumai_provider_openai_compatible::providers::openai_compatible::{
            ConfigurableAdapter, OpenAiCompatibleConfig,
        };
        use siumai_provider_openai_compatible::providers::openai_compatible::{
            ProviderAdapter, get_provider_config,
        };

        self.validate()?;

        let provider = get_provider_config("xai").ok_or_else(|| {
            LlmError::ConfigurationError("OpenAI-compatible provider config not found: xai".into())
        })?;

        let mut adapter: Box<dyn ProviderAdapter> = Box::new(ConfigurableAdapter::new(provider));
        if !self.provider_specific_config.is_empty() {
            let normalized_defaults =
                normalize_xai_chat_defaults(self.provider_specific_config.clone());
            adapter = Box::new(
                siumai_provider_openai_compatible::providers::openai_compatible::adapter::ParamMergingAdapter::new(
                    adapter,
                    normalized_defaults,
                ),
            );
        }
        let adapter: Arc<dyn ProviderAdapter> = Arc::from(adapter);

        let mut config = OpenAiCompatibleConfig::new(
            "xai",
            self.api_key.expose_secret(),
            &self.base_url,
            adapter,
        )
        .with_supports_structured_outputs(true)
        .with_include_usage(true)
        .with_common_params(self.common_params.clone())
        .with_model(&self.common_params.model)
        .with_http_config(self.http_config.clone())
        .with_http_interceptors(self.http_interceptors.clone())
        .with_model_middlewares(self.model_middlewares.clone());

        if let Some(transport) = self.http_transport.clone() {
            config = config.with_http_transport(transport);
        }

        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[derive(Clone, Default)]
    struct NoopInterceptor;

    impl HttpInterceptor for NoopInterceptor {}

    #[test]
    fn xai_config_reasoning_and_search_defaults_roundtrip_into_compatible_config() {
        let http_config = HttpConfig {
            timeout: Some(Duration::from_secs(9)),
            ..Default::default()
        };

        let config = XaiConfig::new("test-key")
            .with_model("grok-4")
            .with_reasoning(true)
            .with_reasoning_budget(4096)
            .with_reasoning_effort("high")
            .with_reasoning_summary("detailed")
            .with_top_logprobs(2)
            .with_parallel_function_calling(false)
            .with_store(false)
            .with_previous_response("resp_prev_123")
            .with_include(["file_search_call.results"])
            .with_default_search()
            .with_http_config(http_config.clone());

        let compat = config
            .clone()
            .into_compatible_config()
            .expect("into_compatible_config");
        let mut params = serde_json::json!({});
        compat
            .adapter
            .transform_request_params(
                &mut params,
                &compat.model,
                siumai_provider_openai_compatible::providers::openai_compatible::RequestType::Chat,
            )
            .expect("transform request params");

        assert_eq!(compat.model, "grok-4");
        assert_eq!(compat.supports_structured_outputs, Some(true));
        assert_eq!(compat.include_usage, Some(true));
        assert_eq!(compat.http_config.timeout, Some(Duration::from_secs(9)));
        assert_eq!(params["reasoning_effort"], serde_json::json!("high"));
        assert!(params.get("enable_reasoning").is_none());
        assert!(params.get("reasoning_budget").is_none());
        assert!(params.get("enable_thinking").is_none());
        assert!(params.get("thinking_budget").is_none());
        assert_eq!(params["logprobs"], serde_json::json!(true));
        assert_eq!(params["top_logprobs"], serde_json::json!(2));
        assert_eq!(
            params["parallel_function_calling"],
            serde_json::json!(false)
        );
        assert!(params.get("reasoning_summary").is_none());
        assert!(params.get("store").is_none());
        assert!(params.get("previous_response_id").is_none());
        assert!(params.get("include").is_none());
        assert_eq!(
            params["search_parameters"]["return_citations"],
            serde_json::json!(true)
        );
        assert_eq!(
            params["search_parameters"]["max_search_results"],
            serde_json::json!(20)
        );
    }

    #[test]
    fn xai_config_http_convenience_helpers_update_http_state() {
        let config = XaiConfig::new("test-key")
            .with_model("grok-4")
            .with_timeout(Duration::from_secs(9))
            .with_connect_timeout(Duration::from_secs(3))
            .with_http_stream_disable_compression(true)
            .with_http_interceptor(Arc::new(NoopInterceptor));

        assert_eq!(config.http_config.timeout, Some(Duration::from_secs(9)));
        assert_eq!(
            config.http_config.connect_timeout,
            Some(Duration::from_secs(3))
        );
        assert!(config.http_config.stream_disable_compression);
        assert_eq!(config.http_interceptors.len(), 1);
    }
}
