//! Legacy provider construction helpers (compatibility-only, registry-driven)
//!
//! These helpers predate the family-first `ProviderFactory` contract and return generic
//! `LlmClient` compatibility objects. New built-in provider construction should live in
//! `registry::factories::*` private typed builders plus `ProviderFactory::*_family_with_ctx(...)`
//! methods. Keep this module only as a migration surface for older direct imports.

#[allow(unused_imports)]
use crate::compat::client::LlmClient;
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
pub use crate::registry::typed_builders::OpenAiChatApiMode;

#[cfg(feature = "openai")]
#[deprecated(
    since = "0.11.0-beta.8",
    note = "compatibility-only; use registry ProviderFactory family methods or OpenAI provider config-first construction"
)]
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
#[deprecated(
    since = "0.11.0-beta.8",
    note = "compatibility-only; use registry ProviderFactory family methods or OpenAI provider config-first construction"
)]
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

#[cfg(any(
    feature = "openai",
    feature = "togetherai",
    feature = "deepinfra",
    feature = "google-vertex"
))]
#[deprecated(
    since = "0.11.0-beta.9",
    note = "compatibility wrapper; typed provider construction now lives in registry::typed_builders"
)]
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
    token_provider: Option<std::sync::Arc<dyn crate::auth::TokenProvider>>,
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
    crate::registry::typed_builders::build_openai_compatible_typed_client(
        provider_id,
        api_key,
        base_url,
        http_client,
        common_params,
        reasoning_enabled,
        reasoning_budget,
        http_config,
        token_provider,
        _provider_params,
        tracing_config,
        retry_options,
        interceptors,
        middlewares,
        http_transport,
    )
    .await
}

#[cfg(any(
    feature = "openai",
    feature = "togetherai",
    feature = "deepinfra",
    feature = "google-vertex"
))]
#[deprecated(
    since = "0.11.0-beta.8",
    note = "compatibility-only; use registry ProviderFactory family methods or OpenAI-compatible provider config-first construction"
)]
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
    token_provider: Option<std::sync::Arc<dyn crate::auth::TokenProvider>>,
    provider_params: Option<()>, // Removed ProviderParams
    tracing_config: Option<crate::observability::tracing::TracingConfig>,
    retry_options: Option<RetryOptions>,
    interceptors: Vec<Arc<dyn HttpInterceptor>>,
    middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
) -> Result<Arc<dyn LlmClient>, LlmError> {
    let client = crate::registry::typed_builders::build_openai_compatible_typed_client(
        provider_id,
        api_key,
        base_url,
        http_client,
        common_params,
        reasoning_enabled,
        reasoning_budget,
        http_config,
        token_provider,
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
#[deprecated(
    since = "0.11.0-beta.8",
    note = "compatibility-only; use registry ProviderFactory family methods or Anthropic provider config-first construction"
)]
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
#[deprecated(
    since = "0.11.0-beta.9",
    note = "compatibility wrapper; typed provider construction now lives in registry::typed_builders"
)]
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
    crate::registry::typed_builders::build_gemini_typed_client(
        api_key,
        base_url,
        http_client,
        common_params,
        http_config,
        _provider_params,
        google_token_provider,
        tracing_config,
        retry_options,
        interceptors,
        middlewares,
        http_transport,
    )
    .await
}

#[cfg(feature = "google")]
#[deprecated(
    since = "0.11.0-beta.8",
    note = "compatibility-only; use registry ProviderFactory family methods or Gemini provider config-first construction"
)]
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
    let client = crate::registry::typed_builders::build_gemini_typed_client(
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

/// Build Anthropic on Vertex AI typed client.
#[cfg(feature = "google-vertex")]
#[deprecated(
    since = "0.11.0-beta.9",
    note = "compatibility wrapper; typed provider construction now lives in registry::typed_builders"
)]
#[allow(clippy::too_many_arguments)]
pub async fn build_anthropic_vertex_typed_client(
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
    crate::registry::typed_builders::build_anthropic_vertex_typed_client(
        base_url,
        http_client,
        common_params,
        http_config,
        google_token_provider,
        _tracing_config,
        retry_options,
        interceptors,
        middlewares,
        http_transport,
    )
    .await
}

/// Build Anthropic on Vertex AI compatibility client.
#[cfg(feature = "google-vertex")]
#[deprecated(
    since = "0.11.0-beta.8",
    note = "compatibility-only; use registry ProviderFactory family methods or Google Vertex provider config-first construction"
)]
#[allow(clippy::too_many_arguments)]
pub async fn build_anthropic_vertex_client(
    base_url: String,
    http_client: reqwest::Client,
    common_params: CommonParams,
    http_config: HttpConfig,
    google_token_provider: Option<std::sync::Arc<dyn crate::auth::TokenProvider>>,
    tracing_config: Option<crate::observability::tracing::TracingConfig>,
    retry_options: Option<RetryOptions>,
    interceptors: Vec<Arc<dyn HttpInterceptor>>,
    middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
) -> Result<Arc<dyn LlmClient>, LlmError> {
    let client = crate::registry::typed_builders::build_anthropic_vertex_typed_client(
        base_url,
        http_client,
        common_params,
        http_config,
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

#[cfg(feature = "google-vertex")]
#[deprecated(
    since = "0.11.0-beta.9",
    note = "compatibility wrapper; typed provider construction now lives in registry::typed_builders"
)]
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
    crate::registry::typed_builders::build_google_vertex_typed_client(
        base_url,
        api_key,
        http_client,
        common_params,
        http_config,
        token_provider,
        _tracing_config,
        retry_options,
        interceptors,
        middlewares,
        http_transport,
    )
    .await
}

/// Build Google Vertex client (Imagen via Vertex AI).
#[cfg(feature = "google-vertex")]
#[deprecated(
    since = "0.11.0-beta.8",
    note = "compatibility-only; use registry ProviderFactory family methods or Google Vertex provider config-first construction"
)]
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
    let client = crate::registry::typed_builders::build_google_vertex_typed_client(
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
#[deprecated(
    since = "0.11.0-beta.8",
    note = "compatibility-only; use registry ProviderFactory family methods or Ollama provider config-first construction"
)]
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
#[deprecated(
    since = "0.11.0-beta.8",
    note = "compatibility-only; use registry ProviderFactory family methods or MiniMaxi provider config-first construction"
)]
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
