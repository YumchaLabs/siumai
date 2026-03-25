//! Provider construction helpers (registry-driven)
//!
//! These helpers encapsulate provider-specific client construction while
//! allowing SiumaiBuilder to resolve defaults (like base URLs) from the
//! ProviderRegistry v2. They keep the external API unchanged and reduce
//! duplication inside the builder.

#[allow(unused_imports)]
use crate::client::LlmClient;
#[allow(unused_imports)]
use crate::error::LlmError;
#[allow(unused_imports)]
use crate::execution::http::interceptor::HttpInterceptor;
#[allow(unused_imports)]
use crate::execution::middleware::LanguageModelMiddleware;
#[allow(unused_imports)]
use crate::retry_api::RetryOptions;
#[allow(unused_imports)]
use crate::types::{CommonParams, HttpConfig};
#[allow(unused_imports)]
use std::sync::Arc;

#[cfg(feature = "openai")]
#[allow(clippy::too_many_arguments)]
pub async fn build_openai_client(
    api_key: String,
    base_url: String,
    http_client: reqwest::Client,
    common_params: CommonParams,
    _http_config: HttpConfig,
    _provider_params: Option<()>, // Removed ProviderParams
    organization: Option<String>,
    project: Option<String>,
    _tracing_config: Option<crate::observability::tracing::TracingConfig>,
    retry_options: Option<RetryOptions>,
    interceptors: Vec<Arc<dyn HttpInterceptor>>,
    middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
) -> Result<Arc<dyn LlmClient>, LlmError> {
    build_openai_client_with_mode(
        api_key,
        base_url,
        http_client,
        common_params,
        _http_config,
        _provider_params,
        organization,
        project,
        _tracing_config,
        retry_options,
        interceptors,
        middlewares,
        http_transport,
        OpenAiChatApiMode::Responses,
    )
    .await
}

#[cfg(feature = "openai")]
#[allow(clippy::too_many_arguments)]
pub async fn build_openai_chat_completions_client(
    api_key: String,
    base_url: String,
    http_client: reqwest::Client,
    common_params: CommonParams,
    http_config: HttpConfig,
    provider_params: Option<()>, // Removed ProviderParams
    organization: Option<String>,
    project: Option<String>,
    tracing_config: Option<crate::observability::tracing::TracingConfig>,
    retry_options: Option<RetryOptions>,
    interceptors: Vec<Arc<dyn HttpInterceptor>>,
    middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
) -> Result<Arc<dyn LlmClient>, LlmError> {
    build_openai_client_with_mode(
        api_key,
        base_url,
        http_client,
        common_params,
        http_config,
        provider_params,
        organization,
        project,
        tracing_config,
        retry_options,
        interceptors,
        middlewares,
        http_transport,
        OpenAiChatApiMode::ChatCompletions,
    )
    .await
}

#[cfg(feature = "openai")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAiChatApiMode {
    Responses,
    ChatCompletions,
}

#[cfg(feature = "openai")]
#[allow(clippy::too_many_arguments)]
async fn build_openai_client_with_mode(
    api_key: String,
    base_url: String,
    http_client: reqwest::Client,
    common_params: CommonParams,
    _http_config: HttpConfig,
    _provider_params: Option<()>, // Removed ProviderParams
    organization: Option<String>,
    project: Option<String>,
    _tracing_config: Option<crate::observability::tracing::TracingConfig>,
    retry_options: Option<RetryOptions>,
    interceptors: Vec<Arc<dyn HttpInterceptor>>,
    middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    mode: OpenAiChatApiMode,
) -> Result<Arc<dyn LlmClient>, LlmError> {
    let mut config = siumai_provider_openai::providers::openai::OpenAiConfig::new(api_key)
        .with_base_url(base_url)
        .with_model(common_params.model.clone())
        .with_use_responses_api(mode == OpenAiChatApiMode::Responses);

    if let Some(temp) = common_params.temperature {
        config = config.with_temperature(temp);
    }
    if let Some(max_tokens) = common_params.max_tokens {
        config = config.with_max_tokens(max_tokens);
    }
    if let Some(org) = organization {
        config = config.with_organization(org);
    }
    if let Some(proj) = project {
        config = config.with_project(proj);
    }
    if let Some(transport) = http_transport {
        config = config.with_http_transport(transport);
    }

    let mut client =
        siumai_provider_openai::providers::openai::OpenAiClient::new(config, http_client);
    if let Some(opts) = retry_options {
        client.set_retry_options(Some(opts));
    }
    if !interceptors.is_empty() {
        client = client.with_http_interceptors(interceptors);
    }
    // Note: Tracing initialization has been moved to siumai-extras.
    // Users should initialize tracing manually using siumai_extras::telemetry
    // or tracing_subscriber directly before creating the client.
    // The tracing_config parameter is kept for backward compatibility but not used.
    // Install automatic + user-provided model middlewares
    let mut auto_mws =
        crate::execution::middleware::build_auto_middlewares_vec("openai", &common_params.model);
    auto_mws.extend(middlewares);
    if !auto_mws.is_empty() {
        client = client.with_model_middlewares(auto_mws);
    }

    Ok(Arc::new(client))
}

#[cfg(feature = "openai")]
#[allow(clippy::too_many_arguments)]
pub async fn build_openai_compatible_typed_client(
    provider_id: String,
    api_key: String,
    base_url: Option<String>,
    http_client: reqwest::Client,
    common_params: CommonParams,
    reasoning_enabled: Option<bool>,
    reasoning_budget: Option<i32>,
    http_config: HttpConfig,
    _provider_params: Option<()>, // Removed ProviderParams
    tracing_config: Option<crate::observability::tracing::TracingConfig>,
    retry_options: Option<RetryOptions>,
    interceptors: Vec<Arc<dyn HttpInterceptor>>,
    middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
) -> Result<
    siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleClient,
    LlmError,
> {
    // Resolve provider adapter and base URL via registry v2
    let registry = crate::registry::global_registry();
    let (resolved_id, adapter, resolved_base) = {
        let mut guard = registry
            .write()
            .map_err(|_| LlmError::InternalError("Registry lock poisoned".to_string()))?;
        // Ensure provider is registered
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
        // Prefer custom base_url when provided; treat it as the full API prefix (Vercel AI SDK style).
        let default_base = rec
            .base_url
            .unwrap_or_else(|| adapter.base_url().to_string());
        let base = crate::utils::builder_helpers::resolve_base_url(base_url.clone(), &default_base);
        (rec.id, adapter, base)
    };

    // Build config
    let mut config =
        siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleConfig::new(
            &resolved_id,
            &api_key,
            &resolved_base,
            adapter,
        )
        .with_model(&{
            // Normalize model id for provider-specific aliasing (e.g., OpenRouter, DeepSeek)
            crate::utils::model_alias::normalize_model_id(&resolved_id, &common_params.model)
        })
        .with_http_config(http_config.clone());
    if let Some(enabled) = reasoning_enabled {
        config = config.with_reasoning(enabled);
    }
    if let Some(budget) = reasoning_budget {
        config = config.with_reasoning_budget(budget);
    }
    if let Some(transport) = http_transport {
        config = config.with_http_transport(transport);
    }

    // Apply common params we support directly
    if let Some(temp) = common_params.temperature {
        config.common_params.temperature = Some(temp);
    }
    if let Some(max_tokens) = common_params.max_tokens {
        config.common_params.max_tokens = Some(max_tokens);
    }

    // Create client via provided HTTP client
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

    // Apply tracing if configured (no-op if client ignores)
    if let Some(tc) = tracing_config {
        // OpenAI-compatible client doesn?t currently expose tracing guard; keep placeholder for symmetry
        let _ = tc; // avoid unused warning
    }
    // Auto + user middlewares based on resolved provider id
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

#[cfg(feature = "openai")]
#[allow(clippy::too_many_arguments)]
pub async fn build_openai_compatible_client(
    provider_id: String,
    api_key: String,
    base_url: Option<String>,
    http_client: reqwest::Client,
    common_params: CommonParams,
    reasoning_enabled: Option<bool>,
    reasoning_budget: Option<i32>,
    http_config: HttpConfig,
    provider_params: Option<()>, // Removed ProviderParams
    tracing_config: Option<crate::observability::tracing::TracingConfig>,
    retry_options: Option<RetryOptions>,
    interceptors: Vec<Arc<dyn HttpInterceptor>>,
    middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
) -> Result<Arc<dyn LlmClient>, LlmError> {
    let client = build_openai_compatible_typed_client(
        provider_id,
        api_key,
        base_url,
        http_client,
        common_params,
        reasoning_enabled,
        reasoning_budget,
        http_config,
        provider_params,
        tracing_config,
        retry_options,
        interceptors,
        middlewares,
        http_transport,
    )
    .await?;

    Ok(Arc::new(client))
}

#[cfg(feature = "anthropic")]
#[allow(clippy::too_many_arguments)]
pub async fn build_anthropic_client(
    api_key: String,
    base_url: String,
    http_client: reqwest::Client,
    common_params: CommonParams,
    http_config: HttpConfig,
    _provider_params: Option<()>, // Removed ProviderParams
    tracing_config: Option<crate::observability::tracing::TracingConfig>,
    retry_options: Option<RetryOptions>,
    interceptors: Vec<Arc<dyn HttpInterceptor>>,
    middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
) -> Result<Arc<dyn LlmClient>, LlmError> {
    // Provider-specific parameters are now handled via provider_options in ChatRequest
    let anthropic_params =
        siumai_provider_anthropic::providers::anthropic::config::AnthropicParams::default();

    let model_id_for_mw = common_params.model.clone();
    let mut client = siumai_provider_anthropic::providers::anthropic::AnthropicClient::new(
        api_key,
        base_url,
        http_client,
        common_params,
        anthropic_params,
        http_config,
    );
    if let Some(transport) = http_transport {
        client = client.with_http_transport(transport);
    }
    if let Some(opts) = retry_options {
        client.set_retry_options(Some(opts));
    }
    if !interceptors.is_empty() {
        client = client.with_http_interceptors(interceptors);
    }
    // Note: Tracing initialization has been moved to siumai-extras.
    // Users should initialize tracing manually using siumai_extras::telemetry
    // or tracing_subscriber directly before creating the client.
    if let Some(tc) = tracing_config {
        client.set_tracing_config(Some(tc));
    }
    // Auto + user middlewares
    let mut auto_mws =
        crate::execution::middleware::build_auto_middlewares_vec("anthropic", &model_id_for_mw);
    auto_mws.extend(middlewares);
    if !auto_mws.is_empty() {
        client = client.with_model_middlewares(auto_mws);
    }
    Ok(Arc::new(client))
}

#[cfg(feature = "google")]
#[allow(clippy::too_many_arguments)]
pub async fn build_gemini_typed_client(
    api_key: String,
    base_url: String,
    http_client: reqwest::Client,
    common_params: CommonParams,
    http_config: HttpConfig,
    _provider_params: Option<()>, // Removed ProviderParams
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

    // Build base config
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

#[cfg(feature = "google")]
#[allow(clippy::too_many_arguments)]
pub async fn build_gemini_client(
    api_key: String,
    base_url: String,
    http_client: reqwest::Client,
    common_params: CommonParams,
    http_config: HttpConfig,
    provider_params: Option<()>, // Removed ProviderParams
    #[allow(unused_variables)] google_token_provider: Option<
        std::sync::Arc<dyn crate::auth::TokenProvider>,
    >,
    tracing_config: Option<crate::observability::tracing::TracingConfig>,
    retry_options: Option<RetryOptions>,
    interceptors: Vec<Arc<dyn HttpInterceptor>>,
    middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
) -> Result<Arc<dyn LlmClient>, LlmError> {
    let client = build_gemini_typed_client(
        api_key,
        base_url,
        http_client,
        common_params,
        http_config,
        provider_params,
        google_token_provider,
        tracing_config,
        retry_options,
        interceptors,
        middlewares,
        http_transport,
    )
    .await?;

    Ok(Arc::new(client))
}

/// Build Anthropic on Vertex AI client
#[cfg(feature = "google-vertex")]
#[allow(clippy::too_many_arguments)]
pub async fn build_anthropic_vertex_client(
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
) -> Result<Arc<dyn LlmClient>, LlmError> {
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
                    crate::auth::adc::AdcTokenProvider::default_client(),
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
    Ok(Arc::new(client))
}

#[cfg(feature = "google-vertex")]
#[allow(clippy::too_many_arguments)]
pub async fn build_google_vertex_typed_client(
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

/// Build Google Vertex client (Imagen via Vertex AI).
#[cfg(feature = "google-vertex")]
#[allow(clippy::too_many_arguments)]
pub async fn build_google_vertex_client(
    base_url: String,
    api_key: Option<String>,
    http_client: reqwest::Client,
    common_params: CommonParams,
    http_config: HttpConfig,
    token_provider: Option<std::sync::Arc<dyn crate::auth::TokenProvider>>,
    tracing_config: Option<crate::observability::tracing::TracingConfig>,
    retry_options: Option<RetryOptions>,
    interceptors: Vec<Arc<dyn HttpInterceptor>>,
    middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
) -> Result<Arc<dyn LlmClient>, LlmError> {
    let client = build_google_vertex_typed_client(
        base_url,
        api_key,
        http_client,
        common_params,
        http_config,
        token_provider,
        tracing_config,
        retry_options,
        interceptors,
        middlewares,
        http_transport,
    )
    .await?;

    Ok(Arc::new(client))
}

#[cfg(feature = "ollama")]
#[allow(clippy::too_many_arguments)]
pub async fn build_ollama_client(
    base_url: String,
    http_client: reqwest::Client,
    common_params: CommonParams,
    http_config: HttpConfig,
    _provider_params: Option<()>, // Removed ProviderParams
    tracing_config: Option<crate::observability::tracing::TracingConfig>,
    retry_options: Option<RetryOptions>,
    interceptors: Vec<Arc<dyn HttpInterceptor>>,
    middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
) -> Result<Arc<dyn LlmClient>, LlmError> {
    use siumai_provider_ollama::providers::ollama::OllamaClient;
    use siumai_provider_ollama::providers::ollama::config::{OllamaConfig, OllamaParams};

    // Provider-specific parameters are now handled via provider_options in ChatRequest
    let ollama_params = OllamaParams::default();

    let config = OllamaConfig {
        base_url,
        model: Some(common_params.model.clone()),
        common_params: common_params.clone(),
        ollama_params,
        http_config,
        http_transport,
        http_interceptors: interceptors.clone(),
        model_middlewares: Vec::new(),
    };

    let mut client = OllamaClient::new(config, http_client);
    if let Some(opts) = retry_options {
        client.set_retry_options(Some(opts));
    }
    if !interceptors.is_empty() {
        client = client.with_http_interceptors(interceptors);
    }
    // Note: Tracing initialization has been moved to siumai-extras.
    // Users should initialize tracing manually using siumai_extras::telemetry
    // or tracing_subscriber directly before creating the client.
    if let Some(tc) = tracing_config {
        client.set_tracing_config(Some(tc));
    }
    // Auto + user middlewares
    let mut auto_mws =
        crate::execution::middleware::build_auto_middlewares_vec("ollama", &common_params.model);
    auto_mws.extend(middlewares);
    if !auto_mws.is_empty() {
        client = client.with_model_middlewares(auto_mws);
    }
    Ok(Arc::new(client))
}

#[cfg(feature = "minimaxi")]
#[allow(clippy::too_many_arguments)]
pub async fn build_minimaxi_client(
    api_key: String,
    base_url: String,
    http_client: reqwest::Client,
    common_params: CommonParams,
    http_config: HttpConfig,
    tracing_config: Option<crate::observability::tracing::TracingConfig>,
    retry_options: Option<RetryOptions>,
    interceptors: Vec<Arc<dyn HttpInterceptor>>,
    middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
) -> Result<Arc<dyn LlmClient>, LlmError> {
    use siumai_provider_minimaxi::providers::minimaxi::client::MinimaxiClient;
    use siumai_provider_minimaxi::providers::minimaxi::config::MinimaxiConfig;

    let mut model_middlewares =
        crate::execution::middleware::build_auto_middlewares_vec("minimaxi", &common_params.model);
    model_middlewares.extend(middlewares);

    let mut config = MinimaxiConfig::new(api_key)
        .with_base_url(base_url)
        .with_http_config(http_config)
        .with_http_interceptors(interceptors)
        .with_model_middlewares(model_middlewares);
    if let Some(http_transport) = http_transport {
        config = config.with_http_transport(http_transport);
    }
    config.common_params = common_params;

    let mut client = MinimaxiClient::with_http_client(config, http_client)?;

    if let Some(tc) = tracing_config {
        client = client.with_tracing(tc);
    }
    if let Some(opts) = retry_options {
        client = client.with_retry(opts);
    }

    Ok(Arc::new(client))
}
