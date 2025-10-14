use crate::client::LlmClient;
use crate::error::LlmError;
use crate::traits::ProviderCapabilities;
use crate::types::{HttpConfig, ProviderParams, ProviderType};

/// Build the unified Siumai provider from SiumaiBuilder
pub async fn build(builder: super::SiumaiBuilder) -> Result<super::Siumai, LlmError> {
    // Helper: build an HTTP client from HttpConfig
    fn build_http_client_from_config(cfg: &HttpConfig) -> Result<reqwest::Client, LlmError> {
        let mut builder = reqwest::Client::builder();

        if let Some(timeout) = cfg.timeout {
            builder = builder.timeout(timeout);
        }
        if let Some(connect_timeout) = cfg.connect_timeout {
            builder = builder.connect_timeout(connect_timeout);
        }
        if let Some(proxy_url) = &cfg.proxy {
            let proxy = reqwest::Proxy::all(proxy_url)
                .map_err(|e| LlmError::ConfigurationError(format!("Invalid proxy URL: {e}")))?;
            builder = builder.proxy(proxy);
        }
        if let Some(user_agent) = &cfg.user_agent {
            builder = builder.user_agent(user_agent);
        }

        // Default headers
        if !cfg.headers.is_empty() {
            let mut headers = reqwest::header::HeaderMap::new();
            for (k, v) in &cfg.headers {
                let name = reqwest::header::HeaderName::from_bytes(k.as_bytes()).map_err(|e| {
                    LlmError::ConfigurationError(format!("Invalid header name '{k}': {e}"))
                })?;
                let value = reqwest::header::HeaderValue::from_str(v).map_err(|e| {
                    LlmError::ConfigurationError(format!("Invalid header value for '{k}': {e}"))
                })?;
                headers.insert(name, value);
            }
            builder = builder.default_headers(headers);
        }

        builder
            .build()
            .map_err(|e| LlmError::ConfigurationError(format!("Failed to build HTTP client: {e}")))
    }

    // Extract all needed values first to avoid borrow checker issues
    let provider_type = builder
        .provider_type
        .clone()
        .ok_or_else(|| LlmError::ConfigurationError("Provider type not specified".to_string()))?;

    // Check if API key is required for this provider type
    let requires_api_key = match provider_type {
        ProviderType::Ollama => false, // Ollama doesn't require API key
        _ => true,                     // All other providers require API key
    };

    let api_key = if requires_api_key {
        builder
            .api_key
            .clone()
            .ok_or_else(|| LlmError::ConfigurationError("API key not specified".to_string()))?
    } else {
        // For providers that don't require API key, use empty string or None
        builder.api_key.clone().unwrap_or_default()
    };

    // Extract all needed values to avoid borrow checker issues
    let base_url = builder.base_url.clone();
    let organization = builder.organization.clone();
    let project = builder.project.clone();
    let reasoning_enabled = builder.reasoning_enabled;
    let reasoning_budget = builder.reasoning_budget;
    let http_config = builder.http_config.clone();
    // Build one HTTP client for this builder, reuse across providers when possible
    let built_http_client = build_http_client_from_config(&http_config)?;

    // Prepare common parameters with the correct model
    let mut common_params = builder.common_params.clone();

    // Set default model if none provided
    if common_params.model.is_empty() {
        // Set default model based on provider type
        #[cfg(any(feature = "openai", feature = "anthropic", feature = "google"))]
        use crate::types::models::model_constants as models;

        common_params.model = match provider_type {
            #[cfg(feature = "openai")]
            ProviderType::OpenAi => models::openai::GPT_4O.to_string(),
            #[cfg(feature = "anthropic")]
            ProviderType::Anthropic => models::anthropic::CLAUDE_SONNET_3_5.to_string(),
            #[cfg(feature = "google")]
            ProviderType::Gemini => models::gemini::GEMINI_2_5_FLASH.to_string(),
            #[cfg(feature = "ollama")]
            ProviderType::Ollama => "llama3.2".to_string(),
            #[cfg(feature = "xai")]
            ProviderType::XAI => "grok-beta".to_string(),
            #[cfg(feature = "groq")]
            ProviderType::Groq => "llama-3.1-70b-versatile".to_string(),
            ProviderType::Custom(ref name) => match name.as_str() {
                #[cfg(feature = "openai")]
                "siliconflow" => models::openai_compatible::siliconflow::DEEPSEEK_V3_1.to_string(),
                #[cfg(feature = "openai")]
                "deepseek" => models::openai_compatible::deepseek::CHAT.to_string(),
                #[cfg(feature = "openai")]
                "openrouter" => models::openai_compatible::openrouter::GPT_4O.to_string(),
                _ => "default-model".to_string(),
            },

            // For disabled features, return error
            #[cfg(not(feature = "openai"))]
            ProviderType::OpenAi => {
                return Err(LlmError::UnsupportedOperation(
                    "OpenAI feature not enabled".to_string(),
                ));
            }
            #[cfg(not(feature = "anthropic"))]
            ProviderType::Anthropic => {
                return Err(LlmError::UnsupportedOperation(
                    "Anthropic feature not enabled".to_string(),
                ));
            }
            #[cfg(not(feature = "google"))]
            ProviderType::Gemini => {
                return Err(LlmError::UnsupportedOperation(
                    "Google feature not enabled".to_string(),
                ));
            }
            #[cfg(not(feature = "ollama"))]
            ProviderType::Ollama => {
                return Err(LlmError::UnsupportedOperation(
                    "Ollama feature not enabled".to_string(),
                ));
            }
            #[cfg(not(feature = "xai"))]
            ProviderType::XAI => {
                return Err(LlmError::UnsupportedOperation(
                    "xAI feature not enabled".to_string(),
                ));
            }
            #[cfg(not(feature = "groq"))]
            ProviderType::Groq => {
                return Err(LlmError::UnsupportedOperation(
                    "Groq feature not enabled".to_string(),
                ));
            }
        };
    }

    // Build provider-specific parameters
    let mut provider_params = match provider_type {
        ProviderType::Anthropic => {
            let mut params = ProviderParams::anthropic();

            // Map unified reasoning parameters to Anthropic-specific parameters
            if let Some(budget) = reasoning_budget {
                params = params.with_param("thinking_budget", budget as u32);
            }

            Some(params)
        }
        ProviderType::Gemini => {
            let mut params = ProviderParams::gemini();

            // Map unified reasoning parameters to Gemini-specific parameters
            if let Some(budget) = reasoning_budget {
                params = params.with_param("thinking_budget", budget as u32);
            }

            Some(params)
        }
        ProviderType::Ollama => {
            let mut params = ProviderParams::new();

            // Map unified reasoning to Ollama thinking
            if reasoning_enabled.unwrap_or(false) {
                params = params.with_param("think", true);
            }

            Some(params)
        }
        _ => {
            // For other providers, no specific parameters for now
            None
        }
    };

    // Merge user-provided provider params (override defaults)
    if let Some(extra) = builder.user_provider_params.clone() {
        provider_params = Some(match provider_params {
            Some(p) => p.merge(extra),
            None => extra,
        });
    }

    // Validation moved to Transformers within Executors; skip pre-validation here

    // Now create the appropriate client based on provider type
    // Parameters have already been validated by RequestBuilder
    let client: Box<dyn LlmClient> = match provider_type {
        #[cfg(feature = "openai")]
        ProviderType::OpenAi => {
            // Resolve defaults via ProviderRegistry v2 (native provider)
            let resolved_base = {
                let registry = crate::registry::global_registry();
                let mut guard = registry
                    .lock()
                    .map_err(|_| LlmError::InternalError("Registry lock poisoned".to_string()))?;
                if guard.resolve("openai").is_none() {
                    guard.register_native(
                        "openai",
                        "OpenAI",
                        Some("https://api.openai.com/v1".to_string()),
                        ProviderCapabilities::new()
                            .with_chat()
                            .with_streaming()
                            .with_tools()
                            .with_embedding(),
                    );
                }
                guard.resolve("openai").and_then(|r| r.base_url.clone())
            };
            let resolved_base = base_url
                .or(resolved_base)
                .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
            crate::registry::factory::build_openai_client(
                api_key,
                resolved_base,
                built_http_client.clone(),
                common_params.clone(),
                http_config.clone(),
                provider_params.clone(),
                organization.clone(),
                project.clone(),
                builder.tracing_config.clone(),
            )
            .await?
        }
        #[cfg(feature = "anthropic")]
        ProviderType::Anthropic => {
            // Resolve defaults via ProviderRegistry v2 (native provider)
            let resolved_base = {
                let registry = crate::registry::global_registry();
                let mut guard = registry
                    .lock()
                    .map_err(|_| LlmError::InternalError("Registry lock poisoned".to_string()))?;
                if guard.resolve("anthropic").is_none() {
                    guard.register_native(
                        "anthropic",
                        "Anthropic",
                        Some("https://api.anthropic.com".to_string()),
                        ProviderCapabilities::new()
                            .with_chat()
                            .with_streaming()
                            .with_tools(),
                    );
                }
                guard.resolve("anthropic").and_then(|r| r.base_url.clone())
            };
            let anthropic_base_url = base_url
                .or(resolved_base)
                .unwrap_or_else(|| "https://api.anthropic.com".to_string());
            crate::registry::factory::build_anthropic_client(
                api_key,
                anthropic_base_url,
                built_http_client.clone(),
                common_params.clone(),
                http_config.clone(),
                provider_params.clone(),
                builder.tracing_config.clone(),
            )
            .await?
        }
        #[cfg(feature = "google")]
        ProviderType::Gemini => {
            // Resolve defaults via ProviderRegistry v2 (native provider)
            let resolved_base = {
                let registry = crate::registry::global_registry();
                let mut guard = registry
                    .lock()
                    .map_err(|_| LlmError::InternalError("Registry lock poisoned".to_string()))?;
                if guard.resolve("gemini").is_none() {
                    guard.register_native(
                        "gemini",
                        "Gemini",
                        Some("https://generativelanguage.googleapis.com/v1beta".to_string()),
                        ProviderCapabilities::new()
                            .with_chat()
                            .with_streaming()
                            .with_embedding()
                            .with_tools(),
                    );
                }
                guard.resolve("gemini").and_then(|r| r.base_url.clone())
            };
            let resolved_base = base_url
                .or(resolved_base)
                .unwrap_or_else(|| "https://generativelanguage.googleapis.com/v1beta".to_string());
            crate::registry::factory::build_gemini_client(
                api_key,
                resolved_base,
                built_http_client.clone(),
                common_params.clone(),
                http_config.clone(),
                provider_params.clone(),
                builder.tracing_config.clone(),
            )
            .await?
        }
        #[cfg(feature = "xai")]
        ProviderType::XAI => {
            crate::registry::factory::build_openai_compatible_client(
                "xai_openai_compatible".to_string(),
                api_key.clone(),
                base_url.clone(),
                built_http_client.clone(),
                common_params.clone(),
                http_config.clone(),
                provider_params.clone(),
                builder.tracing_config.clone(),
            )
            .await?
        }
        #[cfg(feature = "ollama")]
        ProviderType::Ollama => {
            let ollama_base_url = base_url.unwrap_or_else(|| "http://localhost:11434".to_string());
            crate::registry::factory::build_ollama_client(
                ollama_base_url,
                built_http_client.clone(),
                common_params.clone(),
                http_config.clone(),
                provider_params.clone(),
                builder.tracing_config.clone(),
            )
            .await?
        }
        #[cfg(feature = "groq")]
        ProviderType::Groq => {
            crate::registry::factory::build_openai_compatible_client(
                "groq".to_string(),
                api_key.clone(),
                base_url.clone(),
                built_http_client.clone(),
                common_params.clone(),
                http_config.clone(),
                provider_params.clone(),
                builder.tracing_config.clone(),
            )
            .await?
        }
        ProviderType::Custom(name) => {
            #[cfg(feature = "openai")]
            {
                // Build via registry/factory for any OpenAI-compatible provider id
                crate::registry::factory::build_openai_compatible_client(
                    name.clone(),
                    api_key.clone(),
                    base_url.clone(),
                    built_http_client.clone(),
                    common_params.clone(),
                    http_config.clone(),
                    provider_params.clone(),
                    builder.tracing_config.clone(),
                )
                .await?
            }
            #[cfg(not(feature = "openai"))]
            {
                return Err(LlmError::UnsupportedOperation(format!(
                    "Custom provider '{}' requires 'openai' feature",
                    name
                )));
            }
        }

        // Handle cases where required features are not enabled
        #[cfg(not(feature = "openai"))]
        ProviderType::OpenAi => {
            return Err(LlmError::UnsupportedOperation(
                "OpenAI provider requires the 'openai' feature to be enabled".to_string(),
            ));
        }
        #[cfg(not(feature = "anthropic"))]
        ProviderType::Anthropic => {
            return Err(LlmError::UnsupportedOperation(
                "Anthropic provider requires the 'anthropic' feature to be enabled".to_string(),
            ));
        }
        #[cfg(not(feature = "google"))]
        ProviderType::Gemini => {
            return Err(LlmError::UnsupportedOperation(
                "Gemini provider requires the 'google' feature to be enabled".to_string(),
            ));
        }
        #[cfg(not(feature = "ollama"))]
        ProviderType::Ollama => {
            return Err(LlmError::UnsupportedOperation(
                "Ollama provider requires the 'ollama' feature to be enabled".to_string(),
            ));
        }
        #[cfg(not(feature = "xai"))]
        ProviderType::XAI => {
            return Err(LlmError::UnsupportedOperation(
                "xAI provider requires the 'xai' feature to be enabled".to_string(),
            ));
        }
        #[cfg(not(feature = "groq"))]
        ProviderType::Groq => {
            return Err(LlmError::UnsupportedOperation(
                "Groq provider requires the 'groq' feature to be enabled".to_string(),
            ));
        }
    };

    let siumai = super::Siumai::new(client).with_retry_options(builder.retry_options.clone());
    Ok(siumai)
}
