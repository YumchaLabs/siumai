//! Anthropic token utilities (provider-specific).
//!
//! Implements `POST /v1/messages/count_tokens`.

use crate::error::LlmError;
use crate::execution::executors::common::{HttpBody, HttpExecutionConfig, execute_json_request};
use crate::execution::http::interceptor::HttpInterceptor;
use crate::retry_api::RetryOptions;
use crate::utils::url::join_url;
use crate::{core::ProviderSpec, types::ChatRequest};
use reqwest::Client as HttpClient;
use secrecy::{ExposeSecret, SecretString};
use std::collections::HashMap;
use std::sync::Arc;

use super::spec::AnthropicSpec;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct AnthropicCountTokensResponse {
    pub input_tokens: u32,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Clone)]
pub struct AnthropicTokens {
    pub(crate) api_key: SecretString,
    pub(crate) base_url: String,
    pub(crate) http_client: HttpClient,
    pub(crate) http_config: crate::types::HttpConfig,
    pub(crate) beta_features: Vec<String>,
    pub(crate) http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    pub(crate) http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    pub(crate) retry_options: Option<RetryOptions>,
}

impl AnthropicTokens {
    pub fn new(
        api_key: SecretString,
        base_url: String,
        http_client: HttpClient,
        http_config: crate::types::HttpConfig,
        beta_features: Vec<String>,
        http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
        http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
        retry_options: Option<RetryOptions>,
    ) -> Self {
        Self {
            api_key,
            base_url,
            http_client,
            http_config,
            beta_features,
            http_transport,
            http_interceptors,
            retry_options,
        }
    }

    /// Count tokens for a unified `ChatRequest`.
    ///
    /// This reuses the Anthropic Messages API request transformer (same as chat),
    /// then sends the resulting request body to `/v1/messages/count_tokens`.
    pub async fn count_tokens(
        &self,
        request: &ChatRequest,
    ) -> Result<AnthropicCountTokensResponse, LlmError> {
        let mut req = request.clone();
        req.stream = false;

        let ctx = self.build_context();
        let http_config = self.build_http_config(ctx.clone());
        let spec = AnthropicSpec::new();
        let transformers = spec.choose_chat_transformers(&req, &ctx);

        let mut body = transformers.request.transform_chat(&req)?;
        if let Some(hook) = spec.chat_before_send(&req, &ctx) {
            body = hook(&body)?;
        }

        let url = join_url(&self.base_url, "messages/count_tokens");
        let per_request_headers = spec.chat_request_headers(false, &req, &ctx);

        let call = || {
            let http_config = http_config.clone();
            let url = url.clone();
            let body = body.clone();
            let per_request_headers = per_request_headers.clone();
            async move {
                let result = execute_json_request(
                    &http_config,
                    &url,
                    HttpBody::Json(body),
                    Some(&per_request_headers),
                    false,
                )
                .await?;

                serde_json::from_value::<AnthropicCountTokensResponse>(result.json).map_err(|e| {
                    LlmError::ParseError(format!(
                        "Failed to parse Anthropic count_tokens response: {e}"
                    ))
                })
            }
        };

        crate::retry_api::maybe_retry(self.retry_options.clone(), call).await
    }

    /// Call `/v1/messages/count_tokens` with a pre-built Anthropic request body.
    ///
    /// This is useful for callers that already constructed an Anthropic Messages API request.
    pub async fn count_tokens_raw(
        &self,
        body: serde_json::Value,
        per_request_headers: Option<HashMap<String, String>>,
    ) -> Result<AnthropicCountTokensResponse, LlmError> {
        let ctx = self.build_context();
        let http_config = self.build_http_config(ctx);
        let url = join_url(&self.base_url, "messages/count_tokens");

        let call = || {
            let http_config = http_config.clone();
            let url = url.clone();
            let body = body.clone();
            let headers = per_request_headers.clone();
            async move {
                let result = execute_json_request(
                    &http_config,
                    &url,
                    HttpBody::Json(body),
                    headers.as_ref(),
                    false,
                )
                .await?;

                serde_json::from_value::<AnthropicCountTokensResponse>(result.json).map_err(|e| {
                    LlmError::ParseError(format!(
                        "Failed to parse Anthropic count_tokens response: {e}"
                    ))
                })
            }
        };

        crate::retry_api::maybe_retry(self.retry_options.clone(), call).await
    }

    fn build_context(&self) -> crate::core::ProviderContext {
        let mut headers = self.http_config.headers.clone();

        if !self.beta_features.is_empty() {
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
            merged.push_str(&self.beta_features.join(","));

            headers.insert("anthropic-beta".to_string(), merged);
        }

        crate::core::ProviderContext::new(
            "anthropic",
            self.base_url.clone(),
            Some(self.api_key.expose_secret().to_string()),
            headers,
        )
    }

    fn build_http_config(&self, ctx: crate::core::ProviderContext) -> HttpExecutionConfig {
        let mut wiring = crate::execution::wiring::HttpExecutionWiring::new(
            "anthropic",
            self.http_client.clone(),
            ctx,
        )
        .with_interceptors(self.http_interceptors.clone())
        .with_retry_options(self.retry_options.clone());

        if let Some(transport) = self.http_transport.clone() {
            wiring = wiring.with_transport(transport);
        }

        wiring.config(Arc::new(AnthropicSpec::new()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_tokens_uses_non_stream_messages_body() {
        let tokens = AnthropicTokens::new(
            SecretString::from("k".to_string()),
            "https://api.anthropic.com/v1".to_string(),
            reqwest::Client::new(),
            crate::types::HttpConfig::default(),
            vec![],
            None,
            vec![],
            None,
        );

        let mut req = ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()]);
        req.common_params.model = "claude-3-5-sonnet-20241022".to_string();
        req.stream = true;

        let mut non_stream = req.clone();
        non_stream.stream = false;

        let ctx = tokens.build_context();
        let spec = AnthropicSpec::new();
        let transformers = spec.choose_chat_transformers(&non_stream, &ctx);
        let body = transformers
            .request
            .transform_chat(&non_stream)
            .expect("body");

        assert!(
            body.get("stream").is_none(),
            "count_tokens request must not enable streaming"
        );
    }

    #[test]
    fn build_context_merges_beta_features_into_anthropic_beta_header() {
        let mut cfg = crate::types::HttpConfig::default();
        cfg.headers
            .insert("Anthropic-Beta".to_string(), "a,b".to_string());

        let tokens = AnthropicTokens::new(
            SecretString::from("k".to_string()),
            "https://api.anthropic.com/v1".to_string(),
            reqwest::Client::new(),
            cfg,
            vec!["b".to_string(), "c".to_string()],
            None,
            vec![],
            None,
        );

        let ctx = tokens.build_context();
        assert_eq!(
            ctx.http_extra_headers
                .get("anthropic-beta")
                .map(|s| s.as_str()),
            Some("a,b,b,c")
        );
    }
}
