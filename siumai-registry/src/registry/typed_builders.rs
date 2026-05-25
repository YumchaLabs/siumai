//! Internal typed provider-client builders for built-in registry factories.
//!
//! These helpers are production wiring for provider-owned typed clients. Public generic-client
//! compatibility wrappers live in `registry::factory`.

#![allow(unused_imports)]

use crate::error::LlmError;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::middleware::LanguageModelMiddleware;
use crate::retry_api::RetryOptions;
use crate::types::{CommonParams, HttpConfig};
use std::sync::Arc;

#[cfg(feature = "openai")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAiChatApiMode {
    Responses,
    ChatCompletions,
}

#[cfg(any(
    feature = "openai",
    feature = "togetherai",
    feature = "deepinfra",
    feature = "google-vertex"
))]
#[allow(clippy::too_many_arguments)]
pub(crate) async fn build_openai_compatible_typed_client(
    provider_id: String,
    api_key: String,
    base_url: Option<String>,
    http_client: reqwest::Client,
    common_params: CommonParams,
    reasoning_enabled: Option<bool>,
    reasoning_budget: Option<i32>,
    http_config: HttpConfig,
    token_provider: Option<std::sync::Arc<dyn crate::auth::TokenProvider>>,
    _provider_params: Option<()>,
    tracing_config: Option<crate::observability::tracing::TracingConfig>,
    retry_options: Option<RetryOptions>,
    interceptors: Vec<Arc<dyn HttpInterceptor>>,
    middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
) -> Result<
    siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleClient,
    LlmError,
> {
    let registry = crate::registry::global_registry();
    let (resolved_id, adapter, resolved_base) = {
        let mut guard = registry
            .write()
            .map_err(|_| LlmError::InternalError("Registry lock poisoned".to_string()))?;
        let _ = guard.register_openai_compatible(&provider_id);
        let rec = guard.resolve(&provider_id).cloned().ok_or_else(|| {
            LlmError::ConfigurationError(format!(
                "Unknown OpenAI-compatible provider: {}",
                provider_id
            ))
        })?;
        let adapter = rec.adapter.ok_or_else(|| {
            LlmError::ConfigurationError(format!(
                "Adapter missing for OpenAI-compatible provider: {}",
                rec.id
            ))
        })?;
        let default_base = rec
            .base_url
            .unwrap_or_else(|| adapter.base_url().to_string());
        let base =
            crate::provider_utils::builder_helpers::resolve_base_url(base_url, &default_base);
        (rec.id, adapter, base)
    };

    let mut config =
        siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleConfig::new(
            &resolved_id,
            &api_key,
            &resolved_base,
            adapter,
        )
        .with_model(&{
            crate::provider::resolver::normalize_model_id(&resolved_id, &common_params.model)
        })
        .with_http_config(http_config.clone());
    if let Some(token_provider) = token_provider {
        config = config.with_token_provider(token_provider);
    }
    if let Some(enabled) = reasoning_enabled {
        config = config.with_reasoning(enabled);
    }
    if let Some(budget) = reasoning_budget {
        config = config.with_reasoning_budget(budget);
    }
    if let Some(transport) = http_transport {
        config = config.with_http_transport(transport);
    }
    if resolved_id == crate::provider::ids::GOOGLE_VERTEX_XAI {
        config = config
            .with_include_usage(true)
            .with_supports_structured_outputs(true)
            .with_request_body_transformer(
                siumai_provider_openai_compatible::providers::openai_compatible::settings::google_vertex_xai_request_body_transformer(),
            );
    }

    if let Some(temp) = common_params.temperature {
        config.common_params.temperature = Some(temp);
    }
    if let Some(max_tokens) = common_params.max_tokens {
        config.common_params.max_tokens = Some(max_tokens);
    }

    let mut client =
        siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleClient::with_http_client(
            config,
            http_client,
        )
        .await?;
    if let Some(opts) = retry_options {
        client.set_retry_options(Some(opts));
    }
    if !interceptors.is_empty() {
        client = client.with_http_interceptors(interceptors);
    }

    if let Some(tc) = tracing_config {
        let _ = tc;
    }

    let mut auto_mws = crate::execution::middleware::build_auto_middlewares_vec(
        &resolved_id,
        &common_params.model,
    );
    auto_mws.extend(middlewares);
    if !auto_mws.is_empty() {
        client = client.with_model_middlewares(auto_mws);
    }

    Ok(client)
}

#[cfg(feature = "google")]
#[allow(clippy::too_many_arguments)]
pub(crate) async fn build_gemini_typed_client(
    api_key: String,
    base_url: String,
    http_client: reqwest::Client,
    common_params: CommonParams,
    http_config: HttpConfig,
    _provider_params: Option<()>,
    #[allow(unused_variables)] google_token_provider: Option<
        std::sync::Arc<dyn crate::auth::TokenProvider>,
    >,
    tracing_config: Option<crate::observability::tracing::TracingConfig>,
    retry_options: Option<RetryOptions>,
    interceptors: Vec<Arc<dyn HttpInterceptor>>,
    middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
) -> Result<siumai_provider_gemini::providers::gemini::GeminiClient, LlmError> {
    use siumai_provider_gemini::providers::gemini::client::GeminiClient;
    use siumai_provider_gemini::providers::gemini::types::{GeminiConfig, GenerationConfig};

    let mut gcfg = GenerationConfig::new();
    if let Some(temp) = common_params.temperature {
        gcfg = gcfg.with_temperature(temp);
    }
    if let Some(max_tokens) = common_params.max_tokens {
        gcfg = gcfg.with_max_output_tokens(max_tokens as i32);
    }
    if let Some(top_p) = common_params.top_p {
        gcfg = gcfg.with_top_p(top_p);
    }
    if let Some(stop) = common_params.stop_sequences.clone() {
        gcfg = gcfg.with_stop_sequences(stop);
    }

    let mut config = GeminiConfig::new(api_key)
        .with_base_url(base_url)
        .with_model(common_params.model.clone())
        .with_generation_config(gcfg)
        .with_common_params(common_params.clone());
    config = config.with_http_config(http_config.clone());
    if let Some(transport) = http_transport {
        config = config.with_http_transport(transport);
    }

    if let Some(tp) = google_token_provider {
        config = config.with_token_provider(tp);
    }

    let mut client = GeminiClient::with_http_client(config, http_client)?;
    if let Some(opts) = retry_options {
        client.set_retry_options(Some(opts));
    }
    if !interceptors.is_empty() {
        client = client.with_http_interceptors(interceptors);
    }

    if let Some(tc) = tracing_config {
        client.set_tracing_config(Some(tc));
    }
    let mut auto_mws =
        crate::execution::middleware::build_auto_middlewares_vec("gemini", &common_params.model);
    auto_mws.push(std::sync::Arc::new(
        siumai_provider_gemini::providers::gemini::middleware::GeminiToolWarningsMiddleware::new(),
    ));
    auto_mws.extend(middlewares);
    if !auto_mws.is_empty() {
        client = client.with_model_middlewares(auto_mws);
    }

    Ok(client)
}

#[cfg(feature = "google-vertex")]
#[allow(clippy::too_many_arguments)]
pub(crate) async fn build_anthropic_vertex_typed_client(
    base_url: String,
    http_client: reqwest::Client,
    common_params: CommonParams,
    http_config: HttpConfig,
    #[allow(unused_variables)] google_token_provider: Option<
        std::sync::Arc<dyn crate::auth::TokenProvider>,
    >,
    _tracing_config: Option<crate::observability::tracing::TracingConfig>,
    retry_options: Option<RetryOptions>,
    interceptors: Vec<Arc<dyn HttpInterceptor>>,
    middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
) -> Result<
    siumai_provider_google_vertex::providers::anthropic_vertex::client::VertexAnthropicClient,
    LlmError,
> {
    let token_provider = {
        #[cfg(feature = "gcp")]
        {
            fn has_auth_header(headers: &std::collections::HashMap<String, String>) -> bool {
                headers
                    .keys()
                    .any(|key| key.eq_ignore_ascii_case("authorization"))
            }

            let mut token_provider = google_token_provider;
            if token_provider.is_none() && !has_auth_header(&http_config.headers) {
                token_provider = Some(Arc::new(
                    siumai_provider_google_vertex::auth::adc::AdcTokenProvider::default_client(),
                ));
            }
            token_provider
        }
        #[cfg(not(feature = "gcp"))]
        {
            google_token_provider
        }
    };

    let mut cfg =
        siumai_provider_google_vertex::providers::anthropic_vertex::client::VertexAnthropicConfig::new(
            base_url,
            common_params.model.clone(),
        )
        .with_http_config(http_config)
        .with_http_interceptors(interceptors)
        .with_model_middlewares(
            crate::execution::middleware::build_auto_middlewares_vec(
                "anthropic",
                &common_params.model,
            ),
        );

    if let Some(http_transport) = http_transport {
        cfg = cfg.with_http_transport(http_transport);
    }
    if let Some(token_provider) = token_provider {
        cfg = cfg.with_token_provider(token_provider);
    }
    if !middlewares.is_empty() {
        let mut all_middlewares = cfg.model_middlewares.clone();
        all_middlewares.extend(middlewares);
        cfg = cfg.with_model_middlewares(all_middlewares);
    }
    let mut client =
        siumai_provider_google_vertex::providers::anthropic_vertex::client::VertexAnthropicClient::with_http_client(
            cfg,
            http_client,
        )?;
    if let Some(opts) = retry_options {
        client.set_retry_options(Some(opts));
    }
    Ok(client)
}

#[cfg(feature = "google-vertex")]
#[allow(clippy::too_many_arguments)]
pub(crate) async fn build_google_vertex_typed_client(
    base_url: String,
    api_key: Option<String>,
    http_client: reqwest::Client,
    common_params: CommonParams,
    http_config: HttpConfig,
    token_provider: Option<std::sync::Arc<dyn crate::auth::TokenProvider>>,
    _tracing_config: Option<crate::observability::tracing::TracingConfig>,
    retry_options: Option<RetryOptions>,
    interceptors: Vec<Arc<dyn HttpInterceptor>>,
    middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
) -> Result<siumai_provider_google_vertex::providers::vertex::GoogleVertexClient, LlmError> {
    let mut cfg = siumai_provider_google_vertex::providers::vertex::GoogleVertexConfig::new(
        base_url,
        common_params.model.clone(),
    )
    .with_http_config(http_config)
    .with_http_interceptors(interceptors)
    .with_model_middlewares(middlewares);

    if let Some(api_key) = api_key {
        cfg = cfg.with_api_key(api_key);
    }
    if let Some(http_transport) = http_transport {
        cfg = cfg.with_http_transport(http_transport);
    }
    if let Some(token_provider) = token_provider {
        cfg = cfg.with_token_provider(token_provider);
    }

    let mut client =
        siumai_provider_google_vertex::providers::vertex::GoogleVertexClient::with_http_client(
            cfg,
            http_client,
        )?;
    client = client.with_common_params(common_params);
    if let Some(opts) = retry_options {
        client = client.with_retry_options(opts);
    }

    Ok(client)
}
