use crate::error::LlmError;
use crate::execution::executors::common::{
    HttpBinaryResult, HttpBody, HttpExecutionConfig, execute_get_binary, execute_get_request,
    execute_json_request, execute_multipart_request, execute_patch_json_request,
};
use crate::execution::http::headers::HttpHeaderBuilder;
use crate::execution::wiring::HttpExecutionWiring;
use crate::retry_api::RetryOptions;
use crate::traits::ProviderCapabilities;
use crate::types::HttpConfig;
use secrecy::ExposeSecret;
use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;

use super::config::ElevenLabsConfig;

pub(crate) const PROVIDER_ID: &str = "elevenlabs";
pub(crate) const XI_API_KEY: &str = "xi-api-key";

#[derive(Clone)]
struct ElevenLabsResourcesSpec;

impl crate::core::ProviderSpec for ElevenLabsResourcesSpec {
    fn id(&self) -> &'static str {
        PROVIDER_ID
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
    }

    fn build_headers(
        &self,
        ctx: &crate::core::ProviderContext,
    ) -> Result<reqwest::header::HeaderMap, LlmError> {
        let api_key = ctx
            .api_key
            .as_deref()
            .ok_or_else(|| LlmError::MissingApiKey("ElevenLabs API key not provided".into()))?;

        Ok(HttpHeaderBuilder::new()
            .with_custom_auth(XI_API_KEY, api_key)?
            .with_custom_headers(&ctx.http_extra_headers)?
            .build())
    }
}

fn build_http_config(
    config: &ElevenLabsConfig,
    http_client: reqwest::Client,
    retry_options: Option<RetryOptions>,
) -> HttpExecutionConfig {
    let mut wiring = HttpExecutionWiring::new(
        PROVIDER_ID,
        http_client,
        crate::core::ProviderContext::new(
            PROVIDER_ID,
            config.base_url.trim_end_matches('/').to_string(),
            Some(config.api_key.expose_secret().to_string()),
            config.http_config.headers.clone(),
        ),
    )
    .with_interceptors(config.http_interceptors.clone())
    .with_retry_options(retry_options.clone());

    if let Some(transport) = config.http_transport.clone() {
        wiring = wiring.with_transport(transport);
    }

    wiring.config(Arc::new(ElevenLabsResourcesSpec))
}

pub(crate) async fn execute_get_json<T>(
    config: &ElevenLabsConfig,
    http_client: reqwest::Client,
    retry_options: Option<RetryOptions>,
    url: &str,
    http_config: Option<&HttpConfig>,
    operation: &str,
) -> Result<T, LlmError>
where
    T: for<'de> Deserialize<'de> + Send,
{
    let cfg = build_http_config(config, http_client, retry_options.clone());
    let call = || {
        let cfg = cfg.clone();
        let url = url.to_string();
        async move {
            let result = execute_get_request(&cfg, &url, http_config).await?;
            serde_json::from_value(result.json).map_err(|e| {
                LlmError::ParseError(format!(
                    "Failed to parse ElevenLabs {operation} response: {e}"
                ))
            })
        }
    };

    crate::retry_api::maybe_retry(retry_options, call).await
}

pub(crate) async fn execute_get_bytes(
    config: &ElevenLabsConfig,
    http_client: reqwest::Client,
    retry_options: Option<RetryOptions>,
    url: &str,
    http_config: Option<&HttpConfig>,
) -> Result<HttpBinaryResult, LlmError> {
    let cfg = build_http_config(config, http_client, retry_options.clone());
    let call = || {
        let cfg = cfg.clone();
        let url = url.to_string();
        async move { execute_get_binary(&cfg, &url, http_config).await }
    };

    crate::retry_api::maybe_retry(retry_options, call).await
}

pub(crate) async fn execute_post_json<T>(
    config: &ElevenLabsConfig,
    http_client: reqwest::Client,
    retry_options: Option<RetryOptions>,
    url: &str,
    body: Value,
    http_config: Option<&HttpConfig>,
    operation: &str,
) -> Result<T, LlmError>
where
    T: for<'de> Deserialize<'de> + Send,
{
    let cfg = build_http_config(config, http_client, retry_options.clone());
    let call = || {
        let cfg = cfg.clone();
        let url = url.to_string();
        let body = body.clone();
        async move {
            let result =
                execute_json_request(&cfg, &url, HttpBody::Json(body), http_config, false).await?;
            serde_json::from_value(result.json).map_err(|e| {
                LlmError::ParseError(format!(
                    "Failed to parse ElevenLabs {operation} response: {e}"
                ))
            })
        }
    };

    crate::retry_api::maybe_retry(retry_options, call).await
}

pub(crate) async fn execute_patch_json<T>(
    config: &ElevenLabsConfig,
    http_client: reqwest::Client,
    retry_options: Option<RetryOptions>,
    url: &str,
    body: Value,
    http_config: Option<&HttpConfig>,
    operation: &str,
) -> Result<T, LlmError>
where
    T: for<'de> Deserialize<'de> + Send,
{
    let cfg = build_http_config(config, http_client, retry_options.clone());
    let call = || {
        let cfg = cfg.clone();
        let url = url.to_string();
        let body = body.clone();
        async move {
            let result = execute_patch_json_request(&cfg, &url, body, http_config).await?;
            serde_json::from_value(result.json).map_err(|e| {
                LlmError::ParseError(format!(
                    "Failed to parse ElevenLabs {operation} response: {e}"
                ))
            })
        }
    };

    crate::retry_api::maybe_retry(retry_options, call).await
}

pub(crate) async fn execute_multipart_json<T, F>(
    config: &ElevenLabsConfig,
    http_client: reqwest::Client,
    retry_options: Option<RetryOptions>,
    url: &str,
    build_form: F,
    http_config: Option<&HttpConfig>,
    operation: &str,
) -> Result<T, LlmError>
where
    T: for<'de> Deserialize<'de> + Send,
    F: Fn() -> Result<reqwest::multipart::Form, LlmError> + Send + Sync,
{
    let cfg = build_http_config(config, http_client, retry_options.clone());
    let call = || {
        let cfg = cfg.clone();
        let url = url.to_string();
        let build_form = &build_form;
        async move {
            let result = execute_multipart_request(&cfg, &url, build_form, http_config).await?;
            serde_json::from_value(result.json).map_err(|e| {
                LlmError::ParseError(format!(
                    "Failed to parse ElevenLabs {operation} response: {e}"
                ))
            })
        }
    };

    crate::retry_api::maybe_retry(retry_options, call).await
}
