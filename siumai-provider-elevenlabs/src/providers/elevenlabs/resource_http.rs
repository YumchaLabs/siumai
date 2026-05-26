use crate::error::LlmError;
use crate::execution::executors::common::{HttpExecutionConfig, execute_get_request};
use crate::execution::http::headers::HttpHeaderBuilder;
use crate::execution::wiring::HttpExecutionWiring;
use crate::retry_api::RetryOptions;
use crate::traits::ProviderCapabilities;
use crate::types::HttpConfig;
use secrecy::ExposeSecret;
use serde::Deserialize;
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
