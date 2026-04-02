//! `xAI` Builder Implementation
//!
//! Thin wrapper around the OpenAI-compatible vendor builder:
//! `LlmBuilder::new().openai().compatible("xai")`.

use crate::builder::BuilderBase;
use crate::error::LlmError;
use crate::execution::middleware::language_model::LanguageModelMiddleware;
use crate::provider_options::{
    XaiChatReasoningEffort, XaiImageOptions, XaiOptions, XaiReasoningSummary, XaiResponseInclude,
    XaiResponsesOptions, XaiSearchParameters, XaiVideoOptions,
};
use crate::retry_api::RetryOptions;
use siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleBuilder;
use std::collections::HashMap;
use std::sync::Arc;

/// `xAI` client builder.
///
/// This builder intentionally reuses the OpenAI-compatible vendor implementation.
#[derive(Clone)]
pub struct XaiBuilder {
    inner: OpenAiCompatibleBuilder,
    http_client_override: Option<reqwest::Client>,
    retry_options: Option<RetryOptions>,
    extra_model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    provider_specific_config: HashMap<String, serde_json::Value>,
    default_provider_options_map: crate::types::ProviderOptionsMap,
}

impl XaiBuilder {
    pub fn new(base: BuilderBase) -> Self {
        Self {
            inner: OpenAiCompatibleBuilder::new(base, "xai"),
            http_client_override: None,
            retry_options: None,
            extra_model_middlewares: Vec::new(),
            provider_specific_config: HashMap::new(),
            default_provider_options_map: crate::types::ProviderOptionsMap::default(),
        }
    }

    pub fn api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.inner = self.inner.api_key(api_key);
        self
    }

    pub fn base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.inner = self.inner.base_url(base_url);
        self
    }

    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.inner = self.inner.model(model);
        self
    }

    pub fn temperature(mut self, temperature: f64) -> Self {
        self.inner = self.inner.temperature(temperature);
        self
    }

    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.inner = self.inner.max_tokens(max_tokens);
        self
    }

    pub fn top_p(mut self, top_p: f64) -> Self {
        self.inner = self.inner.top_p(top_p);
        self
    }

    pub fn stop_sequences(mut self, stop: Vec<String>) -> Self {
        self.inner = self.inner.stop(stop);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.inner = self.inner.seed(seed);
        self
    }

    pub fn timeout(mut self, timeout: std::time::Duration) -> Self {
        self.inner = self.inner.timeout(timeout);
        self
    }

    pub fn connect_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.inner = self.inner.connect_timeout(timeout);
        self
    }

    pub fn http_stream_disable_compression(mut self, disable: bool) -> Self {
        self.inner = self.inner.http_stream_disable_compression(disable);
        self
    }

    pub fn with_http_config(mut self, config: crate::types::HttpConfig) -> Self {
        self.inner = self.inner.with_http_config(config);
        self
    }

    pub fn custom_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.inner = self.inner.custom_headers(headers);
        self
    }

    pub fn header<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.inner = self.inner.header(key, value);
        self
    }

    pub fn with_retry(mut self, options: RetryOptions) -> Self {
        self.retry_options = Some(options.clone());
        self.inner = self.inner.with_retry(options);
        self
    }

    pub fn with_http_interceptor(
        mut self,
        interceptor: Arc<dyn crate::execution::http::interceptor::HttpInterceptor>,
    ) -> Self {
        self.inner = self.inner.with_http_interceptor(interceptor);
        self
    }

    pub fn http_debug(mut self, enabled: bool) -> Self {
        self.inner = self.inner.http_debug(enabled);
        self
    }

    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.http_client_override = Some(client.clone());
        self.inner = self.inner.with_http_client(client);
        self
    }

    pub fn with_http_transport(
        mut self,
        transport: Arc<dyn crate::execution::http::transport::HttpTransport>,
    ) -> Self {
        self.inner = self.inner.with_http_transport(transport);
        self
    }

    /// Alias for `with_http_transport(...)` (Vercel-aligned: `fetch`).
    pub fn fetch(
        self,
        transport: Arc<dyn crate::execution::http::transport::HttpTransport>,
    ) -> Self {
        self.with_http_transport(transport)
    }

    pub fn with_thinking(mut self, enable: bool) -> Self {
        self.inner = self.inner.with_thinking(enable);
        self
    }

    pub fn with_thinking_budget(mut self, budget: u32) -> Self {
        self.inner = self.inner.with_thinking_budget(budget);
        self
    }

    pub fn reasoning(mut self, enable: bool) -> Self {
        self.inner = self.inner.reasoning(enable);
        self
    }

    pub fn reasoning_budget(mut self, budget: i32) -> Self {
        self.inner = self.inner.reasoning_budget(budget);
        self
    }

    pub fn with_model_middlewares(
        mut self,
        middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    ) -> Self {
        self.extra_model_middlewares = middlewares;
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
    pub fn with_provider_options_map(mut self, map: crate::types::ProviderOptionsMap) -> Self {
        self.default_provider_options_map = map;
        self
    }

    fn merge_serialized_xai_provider_options<T: serde::Serialize>(mut self, options: T) -> Self {
        if let Ok(serde_json::Value::Object(obj)) = serde_json::to_value(options) {
            self.provider_specific_config
                .extend(obj.into_iter().filter(|(_, value)| !value.is_null()));
        }
        self
    }

    fn merge_non_chat_xai_provider_options<T: serde::Serialize>(mut self, options: T) -> Self {
        if let Ok(serde_json::Value::Object(obj)) = serde_json::to_value(options) {
            let mut overrides = crate::types::ProviderOptionsMap::default();
            overrides.insert("xai", serde_json::Value::Object(obj));
            self.default_provider_options_map.merge_overrides(overrides);
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

    pub fn with_reasoning_effort(self, effort: impl Into<XaiChatReasoningEffort>) -> Self {
        self.with_xai_options(XaiOptions::new().with_reasoning_effort(effort))
    }

    /// Compatibility helper for xAI Responses-style provider options.
    pub fn with_reasoning_summary(self, summary: impl Into<XaiReasoningSummary>) -> Self {
        self.merge_serialized_xai_provider_options(
            XaiResponsesOptions::new().with_reasoning_summary(summary),
        )
    }

    pub fn with_logprobs(self, enabled: bool) -> Self {
        self.with_xai_options(XaiOptions::new().with_logprobs(enabled))
    }

    pub fn with_top_logprobs(self, count: u32) -> Self {
        self.with_xai_options(XaiOptions::new().with_top_logprobs(count))
    }

    pub fn with_parallel_function_calling(self, enabled: bool) -> Self {
        self.with_xai_options(XaiOptions::new().with_parallel_function_calling(enabled))
    }

    /// Compatibility helper for xAI Responses-style provider options.
    pub fn with_store(self, store: bool) -> Self {
        self.merge_serialized_xai_provider_options(XaiResponsesOptions::new().with_store(store))
    }

    /// Compatibility helper for xAI Responses-style provider options.
    pub fn with_previous_response(self, response_id: impl Into<String>) -> Self {
        self.merge_serialized_xai_provider_options(
            XaiResponsesOptions::new().with_previous_response(response_id),
        )
    }

    /// Compatibility helper for xAI Responses-style provider options.
    pub fn with_include<I, S>(self, include: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<XaiResponseInclude>,
    {
        self.merge_serialized_xai_provider_options(XaiResponsesOptions::new().with_include(include))
    }

    pub fn with_search_parameters(self, params: XaiSearchParameters) -> Self {
        self.with_xai_options(XaiOptions::new().with_search(params))
    }

    pub fn with_default_search(self) -> Self {
        self.with_search_parameters(XaiSearchParameters::default())
    }

    pub fn into_config(self) -> Result<super::XaiConfig, LlmError> {
        let compat = self.inner.into_config()?;
        let mut config = super::XaiConfig::from_compatible_config(compat)?;
        if !self.provider_specific_config.is_empty() {
            config = config.with_provider_specific_config(self.provider_specific_config);
        }
        if !self.default_provider_options_map.is_empty() {
            config = config.with_provider_options_map(self.default_provider_options_map);
        }
        if !self.extra_model_middlewares.is_empty() {
            let mut middlewares = config.model_middlewares.clone();
            middlewares.extend(self.extra_model_middlewares);
            config = config.with_model_middlewares(middlewares);
        }
        Ok(config)
    }

    pub async fn build(self) -> Result<super::XaiClient, LlmError> {
        let http_client_override = self.http_client_override.clone();
        let retry_options = self.retry_options.clone();
        let config = self.into_config()?;
        let mut client = if let Some(http_client) = http_client_override {
            super::XaiClient::with_http_client(config, http_client).await?
        } else {
            super::XaiClient::from_config(config).await?
        };
        client.set_retry_options(retry_options);
        Ok(client)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::middleware::language_model::LanguageModelMiddleware;
    use std::sync::Arc;
    use std::time::Duration;

    #[derive(Clone, Default)]
    struct NoopMiddleware;

    impl LanguageModelMiddleware for NoopMiddleware {}

    #[test]
    fn xai_builder_into_config_converges_on_compatible_config() {
        let config = XaiBuilder::new(BuilderBase::default())
            .api_key("test-key")
            .model("grok-2-1212")
            .temperature(0.6)
            .top_p(0.9)
            .stop_sequences(vec!["END".to_string()])
            .seed(7)
            .reasoning(true)
            .reasoning_budget(2048)
            .with_reasoning_effort("high")
            .with_reasoning_summary("detailed")
            .with_top_logprobs(2)
            .with_parallel_function_calling(false)
            .with_store(false)
            .with_previous_response("resp_prev_123")
            .with_include(["file_search_call.results"])
            .with_default_search()
            .timeout(Duration::from_secs(11))
            .connect_timeout(Duration::from_secs(4))
            .http_stream_disable_compression(true)
            .http_debug(true)
            .into_config()
            .expect("into_config ok");

        assert_eq!(
            config.base_url,
            crate::providers::xai::XaiConfig::DEFAULT_BASE_URL
        );
        assert_eq!(config.common_params.model, "grok-2-1212");
        assert_eq!(config.common_params.temperature, Some(0.6));
        assert_eq!(config.common_params.top_p, Some(0.9));
        assert_eq!(
            config.common_params.stop_sequences,
            Some(vec!["END".to_string()])
        );
        assert_eq!(config.common_params.seed, Some(7));
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
        assert_eq!(params["enable_reasoning"], serde_json::json!(true));
        assert_eq!(params["reasoning_budget"], serde_json::json!(2048));
        assert_eq!(params["reasoning_effort"], serde_json::json!("high"));
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
            params["search_parameters"]["mode"],
            serde_json::json!("auto")
        );
        assert_eq!(
            params["search_parameters"]["return_citations"],
            serde_json::json!(true)
        );
        assert_eq!(
            params["search_parameters"]["max_search_results"],
            serde_json::json!(20)
        );
        assert_eq!(config.http_config.timeout, Some(Duration::from_secs(11)));
        assert_eq!(
            config.http_config.connect_timeout,
            Some(Duration::from_secs(4))
        );
        assert!(config.http_config.stream_disable_compression);
        assert_eq!(config.http_interceptors.len(), 1);
    }

    #[test]
    fn xai_builder_into_config_matches_manual_provider_config() {
        let builder_config = XaiBuilder::new(BuilderBase::default())
            .api_key("test-key")
            .model("grok-2-1212")
            .temperature(0.6)
            .top_p(0.9)
            .stop_sequences(vec!["END".to_string()])
            .seed(7)
            .reasoning(true)
            .reasoning_budget(2048)
            .with_reasoning_effort("high")
            .with_reasoning_summary("detailed")
            .with_top_logprobs(2)
            .with_parallel_function_calling(false)
            .with_store(false)
            .with_previous_response("resp_prev_123")
            .with_include(["file_search_call.results"])
            .with_default_search()
            .timeout(Duration::from_secs(11))
            .connect_timeout(Duration::from_secs(4))
            .http_stream_disable_compression(true)
            .http_debug(true)
            .with_model_middlewares(vec![Arc::new(NoopMiddleware)])
            .into_config()
            .expect("builder config");

        let manual_config = crate::providers::xai::XaiConfig::new("test-key")
            .with_base_url(crate::providers::xai::XaiConfig::DEFAULT_BASE_URL)
            .with_model("grok-2-1212")
            .with_temperature(0.6)
            .with_top_p(0.9)
            .with_stop_sequences(vec!["END".to_string()])
            .with_seed(7)
            .with_reasoning(true)
            .with_reasoning_budget(2048)
            .with_reasoning_effort("high")
            .with_reasoning_summary("detailed")
            .with_top_logprobs(2)
            .with_parallel_function_calling(false)
            .with_store(false)
            .with_previous_response("resp_prev_123")
            .with_include(["file_search_call.results"])
            .with_default_search()
            .with_timeout(Duration::from_secs(11))
            .with_connect_timeout(Duration::from_secs(4))
            .with_http_stream_disable_compression(true)
            .with_http_interceptor(Arc::new(
                crate::execution::http::interceptor::LoggingInterceptor,
            ))
            .with_model_middlewares({
                let mut middlewares =
                    crate::execution::middleware::build_auto_middlewares_vec("xai", "grok-2-1212");
                middlewares.push(Arc::new(NoopMiddleware));
                middlewares
            });

        assert_eq!(builder_config.base_url, manual_config.base_url);
        assert_eq!(
            builder_config.common_params.model,
            manual_config.common_params.model
        );
        assert_eq!(
            builder_config.common_params.temperature,
            manual_config.common_params.temperature
        );
        assert_eq!(
            builder_config.common_params.top_p,
            manual_config.common_params.top_p
        );
        assert_eq!(
            builder_config.common_params.stop_sequences,
            manual_config.common_params.stop_sequences
        );
        assert_eq!(
            builder_config.common_params.seed,
            manual_config.common_params.seed
        );
        let builder_compat = builder_config
            .clone()
            .into_compatible_config()
            .expect("builder compat");
        let manual_compat = manual_config
            .clone()
            .into_compatible_config()
            .expect("manual compat");
        let mut builder_params = serde_json::json!({});
        builder_compat
            .adapter
            .transform_request_params(
                &mut builder_params,
                &builder_compat.model,
                siumai_provider_openai_compatible::providers::openai_compatible::RequestType::Chat,
            )
            .expect("builder transform request params");
        let mut manual_params = serde_json::json!({});
        manual_compat
            .adapter
            .transform_request_params(
                &mut manual_params,
                &manual_compat.model,
                siumai_provider_openai_compatible::providers::openai_compatible::RequestType::Chat,
            )
            .expect("manual transform request params");
        assert_eq!(builder_params, manual_params);
        assert_eq!(
            builder_config.http_config.timeout,
            manual_config.http_config.timeout
        );
        assert_eq!(
            builder_config.http_config.connect_timeout,
            manual_config.http_config.connect_timeout
        );
        assert_eq!(
            builder_config.http_config.stream_disable_compression,
            manual_config.http_config.stream_disable_compression
        );
        assert_eq!(
            builder_config.http_interceptors.len(),
            manual_config.http_interceptors.len()
        );
        assert_eq!(
            builder_config.model_middlewares.len(),
            manual_config.model_middlewares.len()
        );
    }

    #[tokio::test]
    async fn xai_builder_build_preserves_http_client_override_and_retry_options() {
        let client = XaiBuilder::new(BuilderBase::default())
            .api_key("test-key")
            .model("grok-2-1212")
            .with_http_config(crate::types::HttpConfig {
                proxy: Some("not-a-url".to_string()),
                ..Default::default()
            })
            .with_http_client(reqwest::Client::new())
            .with_retry(RetryOptions::default())
            .build()
            .await
            .expect("build client with explicit http client");

        assert!(client.retry_options().is_some());
    }
}
