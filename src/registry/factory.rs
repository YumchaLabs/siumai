//! Provider construction helpers (registry-driven)
//!
//! These helpers encapsulate provider-specific client construction while
//! allowing SiumaiBuilder to resolve defaults (like base URLs) from the
//! ProviderRegistry v2. They keep the external API unchanged and reduce
//! duplication inside the builder.

use crate::client::LlmClient;
use crate::error::LlmError;
use crate::types::{CommonParams, HttpConfig, ProviderParams};

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
    if let Some(tc) = tracing_config {
        let guard = crate::tracing::init_tracing(tc)
            .map_err(|e| LlmError::ConfigurationError(format!("Failed to init tracing: {e}")))?;
        client.set_tracing_guard(guard);
    }

    Ok(Box::new(client))
}

#[cfg(feature = "anthropic")]
pub async fn build_anthropic_client(
    api_key: String,
    base_url: String,
    http_client: reqwest::Client,
    common_params: CommonParams,
    http_config: HttpConfig,
    provider_params: Option<ProviderParams>,
    tracing_config: Option<crate::tracing::TracingConfig>,
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
    if let Some(tc) = tracing_config {
        let guard = crate::tracing::init_tracing(tc.clone())
            .map_err(|e| LlmError::ConfigurationError(format!("Failed to init tracing: {e}")))?;
        client.set_tracing_guard(guard);
        client.set_tracing_config(Some(tc));
    }
    Ok(Box::new(client))
}

#[cfg(feature = "google")]
pub async fn build_gemini_client(
    api_key: String,
    base_url: String,
    http_client: reqwest::Client,
    common_params: CommonParams,
    _http_config: HttpConfig,
    provider_params: Option<ProviderParams>,
    tracing_config: Option<crate::tracing::TracingConfig>,
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

    let config = GeminiConfig::new(api_key)
        .with_base_url(base_url)
        .with_model(common_params.model.clone())
        .with_generation_config(gcfg);

    // Create client with provided HTTP client
    let mut client = GeminiClient::with_http_client(config, http_client)?;

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
