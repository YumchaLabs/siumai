use crate::builder::{BuilderBase, ProviderCore};
use crate::error::LlmError;
use crate::retry_api::RetryOptions;
use crate::types::HttpConfig;
use std::sync::Arc;

use super::{VertexAnthropicClient, VertexAnthropicConfig};

/// Anthropic on Vertex AI builder。
///
/// 该 builder 采用 config-first 思路：
/// - `base_url` 必填（显式 Vertex publisher 前缀）
/// - 认证优先通过 `Authorization: Bearer ...` header 传入
/// - `fetch(...)` / `with_http_transport(...)` 与统一入口保持一致
#[derive(Clone)]
pub struct VertexAnthropicBuilder {
    pub(crate) core: ProviderCore,
    base_url: Option<String>,
    model: Option<String>,
    token_provider: Option<Arc<dyn crate::auth::TokenProvider>>,
}

impl VertexAnthropicBuilder {
    pub fn new(base: BuilderBase) -> Self {
        Self {
            core: ProviderCore::new(base),
            base_url: None,
            model: None,
            token_provider: None,
        }
    }

    /// 设置 Vertex publisher base URL。
    pub fn base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// 设置默认模型 ID。
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = Some(model.into());
        self
    }

    /// `Vercel AI SDK` 风格别名。
    pub fn language_model<S: Into<String>>(self, model: S) -> Self {
        self.model(model)
    }

    /// 直接写入 Bearer Token。
    pub fn bearer_token<S: Into<String>>(mut self, token: S) -> Self {
        let token = token.into();
        let trimmed = token.trim();
        if !trimmed.is_empty() {
            self.core
                .http_config
                .headers
                .insert("Authorization".to_string(), format!("Bearer {trimmed}"));
        }
        self
    }

    /// 直接写入 Authorization header。
    pub fn authorization<S: Into<String>>(mut self, value: S) -> Self {
        let value = value.into();
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            self.core
                .http_config
                .headers
                .insert("Authorization".to_string(), trimmed.to_string());
        }
        self
    }

    /// 设置 Bearer token provider（如 ADC）。
    pub fn token_provider(mut self, provider: Arc<dyn crate::auth::TokenProvider>) -> Self {
        self.token_provider = Some(provider);
        self
    }

    /// 自动启用 ADC token provider。
    #[cfg(feature = "gcp")]
    pub fn with_adc(mut self) -> Self {
        self.token_provider = Some(Arc::new(
            crate::auth::adc::AdcTokenProvider::default_client(),
        ));
        self
    }

    pub fn timeout(mut self, timeout: std::time::Duration) -> Self {
        self.core = self.core.timeout(timeout);
        self
    }

    pub fn connect_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.core = self.core.connect_timeout(timeout);
        self
    }

    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.core = self.core.with_http_client(client);
        self
    }

    pub fn http_stream_disable_compression(mut self, disable: bool) -> Self {
        self.core = self.core.http_stream_disable_compression(disable);
        self
    }

    pub fn with_retry(mut self, options: RetryOptions) -> Self {
        self.core = self.core.with_retry(options);
        self
    }

    pub fn with_http_interceptor(
        mut self,
        interceptor: Arc<dyn crate::execution::http::interceptor::HttpInterceptor>,
    ) -> Self {
        self.core = self.core.with_http_interceptor(interceptor);
        self
    }

    pub fn with_http_transport(
        mut self,
        transport: Arc<dyn crate::execution::http::transport::HttpTransport>,
    ) -> Self {
        self.core = self.core.with_http_transport(transport);
        self
    }

    /// `Vercel AI SDK` 风格别名。
    pub fn fetch(
        self,
        transport: Arc<dyn crate::execution::http::transport::HttpTransport>,
    ) -> Self {
        self.with_http_transport(transport)
    }

    pub fn http_debug(mut self, enabled: bool) -> Self {
        self.core = self.core.http_debug(enabled);
        self
    }

    pub fn tracing(mut self, config: crate::observability::tracing::TracingConfig) -> Self {
        self.core = self.core.tracing(config);
        self
    }

    pub fn debug_tracing(mut self) -> Self {
        self.core = self.core.debug_tracing();
        self
    }

    pub fn minimal_tracing(mut self) -> Self {
        self.core = self.core.minimal_tracing();
        self
    }

    pub fn json_tracing(mut self) -> Self {
        self.core = self.core.json_tracing();
        self
    }

    pub fn pretty_json(mut self, pretty: bool) -> Self {
        self.core = self.core.pretty_json(pretty);
        self
    }

    pub fn mask_sensitive_values(mut self, mask: bool) -> Self {
        self.core = self.core.mask_sensitive_values(mask);
        self
    }

    pub fn into_config(self) -> Result<VertexAnthropicConfig, LlmError> {
        let base_url = self.base_url.ok_or_else(|| {
            LlmError::ConfigurationError(
                "Anthropic on Vertex requires a non-empty base_url".to_string(),
            )
        })?;

        let model = self.model.ok_or_else(|| {
            LlmError::ConfigurationError(
                "Anthropic on Vertex requires a non-empty model id".to_string(),
            )
        })?;

        let token_provider = {
            #[cfg(feature = "gcp")]
            {
                fn has_auth_header(headers: &std::collections::HashMap<String, String>) -> bool {
                    headers
                        .keys()
                        .any(|key| key.eq_ignore_ascii_case("authorization"))
                }

                let mut token_provider = self.token_provider;
                if token_provider.is_none() && !has_auth_header(&self.core.http_config.headers) {
                    token_provider = Some(Arc::new(
                        crate::auth::adc::AdcTokenProvider::default_client(),
                    ));
                }
                token_provider
            }
            #[cfg(not(feature = "gcp"))]
            {
                self.token_provider
            }
        };

        let model_middlewares = self.core.get_auto_middlewares("anthropic", &model);
        let mut cfg = VertexAnthropicConfig::new(base_url, model)
            .with_http_config(HttpConfig {
                ..self.core.http_config.clone()
            })
            .with_http_interceptors(self.core.get_http_interceptors())
            .with_model_middlewares(model_middlewares);

        if let Some(transport) = self.core.http_transport.clone() {
            cfg = cfg.with_http_transport(transport);
        }
        if let Some(token_provider) = token_provider {
            cfg = cfg.with_token_provider(token_provider);
        }

        Ok(cfg)
    }

    pub fn build(self) -> Result<VertexAnthropicClient, LlmError> {
        let http_client = self.core.build_http_client()?;
        let retry_options = self.core.retry_options.clone();
        let mut client = VertexAnthropicClient::with_http_client(self.into_config()?, http_client)?;

        if let Some(opts) = retry_options {
            client.set_retry_options(Some(opts));
        }

        Ok(client)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::BuilderBase;
    use crate::client::LlmClient;

    #[derive(Clone, Default)]
    struct NoopInterceptor;

    impl crate::execution::http::interceptor::HttpInterceptor for NoopInterceptor {}

    #[test]
    fn into_config_preserves_authorization_and_interceptors() {
        let cfg = VertexAnthropicBuilder::new(BuilderBase::default())
            .base_url("https://example.com/custom")
            .model("claude-3-5-sonnet-20241022")
            .bearer_token("test-token")
            .with_http_interceptor(Arc::new(NoopInterceptor))
            .into_config()
            .expect("into_config");

        assert_eq!(cfg.base_url, "https://example.com/custom");
        assert_eq!(cfg.model, "claude-3-5-sonnet-20241022");
        assert_eq!(
            cfg.http_config
                .headers
                .get("Authorization")
                .map(String::as_str),
            Some("Bearer test-token")
        );
        assert_eq!(cfg.http_interceptors.len(), 1);
    }

    #[test]
    fn build_and_into_config_converge_on_config_first_client_construction() {
        let builder = VertexAnthropicBuilder::new(BuilderBase::default())
            .base_url("https://example.com/custom")
            .model("claude-3-5-sonnet-20241022")
            .bearer_token("test-token");

        let cfg = builder.clone().into_config().expect("into_config");
        let built = builder.build().expect("build client");
        let from_config = VertexAnthropicClient::from_config(cfg).expect("from_config client");

        assert_eq!(built.base_url(), from_config.base_url());
        assert_eq!(built.supported_models(), from_config.supported_models());
    }
}

#[cfg(all(test, feature = "gcp"))]
mod gcp_tests {
    use super::*;
    use crate::builder::BuilderBase;

    #[test]
    fn build_auto_enables_adc_token_provider_when_missing_auth() {
        let client = VertexAnthropicBuilder::new(BuilderBase::default())
            .base_url("https://example.com/custom")
            .model("claude-3-5-sonnet-20241022")
            .build()
            .expect("build");

        assert!(client._debug_has_token_provider());
    }

    #[test]
    fn build_does_not_override_user_authorization_header() {
        let mut base = BuilderBase::default();
        base.default_headers
            .insert("Authorization".to_string(), "Bearer user".to_string());

        let client = VertexAnthropicBuilder::new(base)
            .base_url("https://example.com/custom")
            .model("claude-3-5-sonnet-20241022")
            .build()
            .expect("build");

        assert!(
            !client._debug_has_token_provider(),
            "user Authorization should suppress auto ADC"
        );
    }
}
