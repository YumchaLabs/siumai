//! Provider construction helpers (registry-driven)
//!
//! These helpers encapsulate provider-specific client construction while
//! allowing SiumaiBuilder to resolve defaults (like base URLs) from the
//! ProviderRegistry v2. They keep the external API unchanged and reduce
//! duplication inside the builder.

use crate::client::LlmClient;
use crate::error::LlmError;
use crate::types::{CommonParams, HttpConfig, ProviderParams};
use crate::utils::http_interceptor::HttpInterceptor;
use std::sync::Arc;

#[cfg(feature = "openai")]
#[allow(clippy::too_many_arguments)]
pub async fn build_openai_client(
    api_key: String,
    base_url: String,
    http_client: reqwest::Client,
    common_params: CommonParams,
    _http_config: HttpConfig,
    provider_params: Option<ProviderParams>,
    organization: Option<String>,
    project: Option<String>,
    tracing_config: Option<crate::tracing::TracingConfig>,
    interceptors: Vec<Arc<dyn HttpInterceptor>>,
) -> Result<Box<dyn LlmClient>, LlmError> {
    let mut config = crate::providers::openai::OpenAiConfig::new(api_key)
        .with_base_url(base_url)
        .with_model(common_params.model.clone());

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

    // Apply provider-specific toggles (e.g., Responses API)
    if let Some(ref params) = provider_params
        && let Some(use_responses) = params.get::<bool>("responses_api")
    {
        config = config.with_responses_api(use_responses);
    }

    let mut client = crate::providers::openai::OpenAiClient::new(config, http_client);
    if !interceptors.is_empty() {
        client = client.with_http_interceptors(interceptors);
    }
    if let Some(tc) = tracing_config {
        let guard = crate::tracing::init_tracing(tc)
            .map_err(|e| LlmError::ConfigurationError(format!("Failed to init tracing: {e}")))?;
        client.set_tracing_guard(guard);
    }

    Ok(Box::new(client))
}

#[cfg(feature = "openai")]
#[allow(clippy::too_many_arguments)]
pub async fn build_openai_compatible_client(
    provider_id: String,
    api_key: String,
    base_url: Option<String>,
    http_client: reqwest::Client,
    common_params: CommonParams,
    http_config: HttpConfig,
    _provider_params: Option<ProviderParams>,
    tracing_config: Option<crate::tracing::TracingConfig>,
    interceptors: Vec<Arc<dyn HttpInterceptor>>,
) -> Result<Box<dyn LlmClient>, LlmError> {
    // Resolve provider adapter and base URL via registry v2
    let registry = crate::registry::global_registry();
    let (resolved_id, adapter, resolved_base) = {
        let mut guard = registry
            .lock()
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
        let base = base_url
            .clone()
            .or(rec.base_url)
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
        (rec.id, adapter, base)
    };

    // Build config
    let mut config = crate::providers::openai_compatible::OpenAiCompatibleConfig::new(
        &resolved_id,
        &api_key,
        &resolved_base,
        adapter,
    )
    .with_model(&common_params.model)
    .with_http_config(http_config.clone());

    // Apply common params we support directly
    if let Some(temp) = common_params.temperature {
        config.common_params.temperature = Some(temp);
    }
    if let Some(max_tokens) = common_params.max_tokens {
        config.common_params.max_tokens = Some(max_tokens);
    }

    // Create client via provided HTTP client
    let mut client = crate::providers::openai_compatible::OpenAiCompatibleClient::with_http_client(
        config,
        http_client,
    )
    .await?;
    if !interceptors.is_empty() {
        client = client.with_http_interceptors(interceptors);
    }

    // Apply tracing if configured (no-op if client ignores)
    if let Some(tc) = tracing_config {
        // OpenAI-compatible client doesn’t currently expose tracing guard; keep placeholder for symmetry
        let _ = tc; // avoid unused warning
    }

    Ok(Box::new(client))
}

#[cfg(feature = "anthropic")]
#[allow(clippy::too_many_arguments)]
pub async fn build_anthropic_client(
    api_key: String,
    base_url: String,
    http_client: reqwest::Client,
    common_params: CommonParams,
    http_config: HttpConfig,
    provider_params: Option<ProviderParams>,
    tracing_config: Option<crate::tracing::TracingConfig>,
    interceptors: Vec<Arc<dyn HttpInterceptor>>,
) -> Result<Box<dyn LlmClient>, LlmError> {
    // Extract Anthropic-specific parameters from provider_params
    let mut anthropic_params = crate::params::AnthropicParams::default();
    if let Some(ref params) = provider_params
        && let Some(budget) = params.get::<u32>("thinking_budget")
    {
        anthropic_params.thinking_budget = Some(budget);
    }

    let mut client = crate::providers::anthropic::AnthropicClient::new(
        api_key,
        base_url,
        http_client,
        common_params,
        anthropic_params,
        http_config,
    );
    if !interceptors.is_empty() {
        client = client.with_http_interceptors(interceptors);
    }
    if let Some(tc) = tracing_config {
        let guard = crate::tracing::init_tracing(tc.clone())
            .map_err(|e| LlmError::ConfigurationError(format!("Failed to init tracing: {e}")))?;
        client.set_tracing_guard(guard);
        client.set_tracing_config(Some(tc));
    }
    Ok(Box::new(client))
}

#[cfg(feature = "google")]
#[allow(clippy::too_many_arguments)]
pub async fn build_gemini_client(
    api_key: String,
    base_url: String,
    http_client: reqwest::Client,
    common_params: CommonParams,
    http_config: HttpConfig,
    provider_params: Option<ProviderParams>,
    #[allow(unused_variables)] gemini_token_provider: Option<
        std::sync::Arc<dyn crate::auth::TokenProvider>,
    >,
    tracing_config: Option<crate::tracing::TracingConfig>,
    interceptors: Vec<Arc<dyn HttpInterceptor>>,
) -> Result<Box<dyn LlmClient>, LlmError> {
    use crate::providers::gemini::client::GeminiClient;
    use crate::providers::gemini::types::{GeminiConfig, GenerationConfig};

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
        .with_generation_config(gcfg);
    // Pass through HTTP config if present in builder path
    config = config.with_http_config(http_config.clone());

    // Attach token provider if present
    if let Some(tp) = gemini_token_provider {
        config = config.with_token_provider(tp);
    }

    // Create client with provided HTTP client
    let mut client = GeminiClient::with_http_client(config, http_client)?;
    if !interceptors.is_empty() {
        client = client.with_http_interceptors(interceptors);
    }

    // Apply thinking budget if provided
    if let Some(ref params) = provider_params
        && let Some(budget) = params.get::<i32>("thinking_budget")
    {
        client = client.with_thinking_budget(budget);
    }

    // Apply tracing if configured
    if let Some(tc) = tracing_config {
        let guard = crate::tracing::init_tracing(tc.clone())
            .map_err(|e| LlmError::ConfigurationError(format!("Failed to init tracing: {e}")))?;
        client.set_tracing_guard(guard);
        client.set_tracing_config(Some(tc));
    }

    Ok(Box::new(client))
}

/// Build Anthropic on Vertex AI client
#[cfg(feature = "anthropic")]
#[allow(clippy::too_many_arguments)]
pub async fn build_anthropic_vertex_client(
    base_url: String,
    http_client: reqwest::Client,
    common_params: CommonParams,
    http_config: HttpConfig,
    _tracing_config: Option<crate::tracing::TracingConfig>,
) -> Result<Box<dyn LlmClient>, LlmError> {
    let cfg = crate::providers::anthropic_vertex::client::VertexAnthropicConfig {
        base_url,
        model: common_params.model.clone(),
        http_config,
    };
    let client =
        crate::providers::anthropic_vertex::client::VertexAnthropicClient::new(cfg, http_client);
    // No tracing guard necessary; headers are injected via ProviderHeaders.
    Ok(Box::new(client))
}

#[cfg(feature = "ollama")]
pub async fn build_ollama_client(
    base_url: String,
    http_client: reqwest::Client,
    common_params: CommonParams,
    http_config: HttpConfig,
    provider_params: Option<ProviderParams>,
    tracing_config: Option<crate::tracing::TracingConfig>,
) -> Result<Box<dyn LlmClient>, LlmError> {
    use crate::providers::ollama::OllamaClient;
    use crate::providers::ollama::config::{OllamaConfig, OllamaParams};

    let mut ollama_params = OllamaParams::default();
    if let Some(ref params) = provider_params
        && let Some(think) = params.get::<bool>("think")
    {
        ollama_params.think = Some(think);
    }

    let config = OllamaConfig {
        base_url,
        model: Some(common_params.model.clone()),
        common_params: common_params.clone(),
        ollama_params,
        http_config,
    };

    let mut client = OllamaClient::new(config, http_client);
    if let Some(tc) = tracing_config {
        let guard = crate::tracing::init_tracing(tc.clone())
            .map_err(|e| LlmError::ConfigurationError(format!("Failed to init tracing: {e}")))?;
        client.set_tracing_guard(guard);
        client.set_tracing_config(Some(tc));
    }
    Ok(Box::new(client))
}
