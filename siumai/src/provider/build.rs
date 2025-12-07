use crate::client::LlmClient;
use crate::error::LlmError;
use crate::execution::http::interceptor::{HttpInterceptor, LoggingInterceptor};
use crate::execution::middleware::LanguageModelMiddleware;
use crate::registry::entry::{BuildContext, ProviderFactory};
#[allow(unused_imports)]
use crate::traits::ProviderCapabilities;
use crate::types::ProviderType;
use std::sync::Arc;

/// Build the unified Siumai provider from SiumaiBuilder
pub async fn build(mut builder: super::SiumaiBuilder) -> Result<super::Siumai, LlmError> {
    // Use unified HTTP client builder from utils
    use crate::execution::http::client::build_http_client_from_config;

    // Best-effort provider suggestion by model prefix (when provider is not set)
    if builder.provider_type.is_none() && !builder.common_params.model.is_empty() {
        let registry = crate::registry::global_registry();
        if let Ok(guard) = registry.read()
            && let Some(rec) = guard.resolve_for_model(&builder.common_params.model)
        {
            let mapped = match rec.id.as_str() {
                #[cfg(feature = "google")]
                "gemini" | "google" | "google-gemini" | "google-vertex" => {
                    Some(ProviderType::Gemini)
                }
                #[cfg(feature = "anthropic")]
                "anthropic" | "anthropic-vertex" | "google-vertex-anthropic" => {
                    Some(ProviderType::Anthropic)
                }
                _ => None,
            };
            if let Some(pt) = mapped {
                builder.provider_type = Some(pt);
                if rec.id == "anthropic-vertex" || rec.id == "google-vertex-anthropic" {
                    builder.provider_id = Some("anthropic-vertex".to_string());
                }
            }
        }
    }

    // Extract all needed values first to avoid borrow checker issues
    let provider_type = builder
        .provider_type
        .clone()
        .ok_or_else(|| LlmError::ConfigurationError("Provider type not specified".to_string()))?;

    // Check if API key is required for this provider type.
    // For Gemini: if Authorization (Bearer) is provided via default headers
    // or a TokenProvider is configured, do not enforce API Key (supports Vertex AI enterprise auth).
    let requires_api_key = match provider_type {
        ProviderType::Ollama => false,
        ProviderType::Gemini => {
            let has_auth_header = builder
                .http_config
                .headers
                .keys()
                .any(|k| k.eq_ignore_ascii_case("authorization"));
            let has_token_provider = {
                #[cfg(feature = "google")]
                {
                    builder.gemini_token_provider.is_some()
                }
                #[cfg(not(feature = "google"))]
                {
                    false
                }
            };
            !(has_auth_header || has_token_provider)
        }
        _ => true,
    };

    let api_key = if requires_api_key {
        // Try to get API key from builder first, then from environment variable
        if let Some(key) = builder.api_key.clone() {
            key
        } else {
            // For Custom providers (OpenAI-compatible), use shared helper function
            if let ProviderType::Custom(ref provider_id) = provider_type {
                crate::utils::builder_helpers::get_api_key_with_env(None, provider_id)?
            } else {
                // For native providers, check their specific environment variables
                match provider_type {
                    #[cfg(feature = "openai")]
                    ProviderType::OpenAi => std::env::var("OPENAI_API_KEY").ok().ok_or_else(|| {
                        LlmError::ConfigurationError(
                            "API key not specified (missing OPENAI_API_KEY or explicit .api_key())"
                                .to_string(),
                        )
                    })?,
                    #[cfg(feature = "anthropic")]
                    ProviderType::Anthropic => std::env::var("ANTHROPIC_API_KEY").ok().ok_or_else(|| {
                        LlmError::ConfigurationError(
                            "API key not specified (missing ANTHROPIC_API_KEY or explicit .api_key())"
                                .to_string(),
                        )
                    })?,
                    #[cfg(feature = "google")]
                    ProviderType::Gemini => std::env::var("GEMINI_API_KEY").ok().ok_or_else(|| {
                        LlmError::ConfigurationError(
                            "API key not specified (missing GEMINI_API_KEY or explicit .api_key(); or use Authorization/TokenProvider)"
                                .to_string(),
                        )
                    })?,
                    #[cfg(feature = "xai")]
                    ProviderType::XAI => std::env::var("XAI_API_KEY").ok().ok_or_else(|| {
                        LlmError::ConfigurationError(
                            "API key not specified (missing XAI_API_KEY or explicit .api_key())".to_string(),
                        )
                    })?,
                    #[cfg(feature = "groq")]
                    ProviderType::Groq => std::env::var("GROQ_API_KEY").ok().ok_or_else(|| {
                        LlmError::ConfigurationError(
                            "API key not specified (missing GROQ_API_KEY or explicit .api_key())".to_string(),
                        )
                    })?,
                    #[cfg(feature = "minimaxi")]
                    ProviderType::MiniMaxi => std::env::var("MINIMAXI_API_KEY").ok().ok_or_else(|| {
                        LlmError::ConfigurationError(
                            "API key not specified (missing MINIMAXI_API_KEY or explicit .api_key())"
                                .to_string(),
                        )
                    })?,
                    _ => {
                        return Err(LlmError::ConfigurationError(
                            "API key not specified".to_string(),
                        ));
                    }
                }
            }
        }
    } else {
        // For providers that don't require API key, use empty string or None
        builder.api_key.clone().unwrap_or_default()
    };

    // Extract all needed values to avoid borrow checker issues
    let base_url = builder.base_url.clone();
    #[cfg(feature = "openai")]
    let organization = builder.organization.clone();
    #[cfg(feature = "openai")]
    let project = builder.project.clone();
    let reasoning_enabled = builder.reasoning_enabled;
    let reasoning_budget = builder.reasoning_budget;
    let http_config = builder.http_config.clone();

    // Build one HTTP client for this builder, reuse across providers when possible
    let built_http_client = if let Some(client) = builder.http_client {
        // Use custom HTTP client if provided
        client
    } else {
        // Use unified HTTP client builder for all cases
        build_http_client_from_config(&http_config)?
    };

    // Prepare common parameters with the correct model
    let mut common_params = builder.common_params.clone();

    // Set default model if none provided
    if common_params.model.is_empty() {
        // Set default model based on provider type
        #[cfg(any(feature = "openai", feature = "anthropic", feature = "google"))]
        use crate::models;

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
            #[cfg(feature = "minimaxi")]
            ProviderType::MiniMaxi => "MiniMax-M2".to_string(),
            ProviderType::Custom(ref name) => {
                // Use shared helper function to get default model from registry
                #[cfg(feature = "openai")]
                {
                    crate::utils::builder_helpers::get_effective_model("", name)
                }
                #[cfg(not(feature = "openai"))]
                {
                    "default-model".to_string()
                }
            }

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
            #[cfg(not(feature = "minimaxi"))]
            ProviderType::MiniMaxi => {
                return Err(LlmError::UnsupportedOperation(
                    "MiniMaxi feature not enabled".to_string(),
                ));
            }
        };
    }

    // Normalize model ID for OpenAI-compatible providers (handle aliases)
    // This ensures that model aliases like "chat" -> "deepseek-chat" are properly resolved
    if let ProviderType::Custom(ref _provider_id) = provider_type {
        #[cfg(feature = "openai")]
        {
            let normalized_model = crate::utils::builder_helpers::normalize_model_id(
                _provider_id,
                &common_params.model,
            );
            if !normalized_model.is_empty() {
                common_params.model = normalized_model;
            }
        }
    }

    // Provider-specific parameters are now handled via provider_options in ChatRequest.
    // The old builder-level provider_params logic has been removed.
    let _reasoning_budget = reasoning_budget; // Suppress unused warning
    let _reasoning_enabled = reasoning_enabled; // Suppress unused warning

    // Validation moved to Transformers within Executors; skip pre-validation here

    // Now create the appropriate client based on provider type.
    // Parameters have already been validated by RequestBuilder.
    // Prepare interceptors (unified interface).
    let mut interceptors: Vec<Arc<dyn HttpInterceptor>> = builder.http_interceptors.clone();
    if builder.http_debug {
        interceptors.push(Arc::new(LoggingInterceptor));
    }

    // Model-level middlewares provided at unified builder level
    let user_model_middlewares: Vec<Arc<dyn LanguageModelMiddleware>> =
        builder.model_middlewares.clone();

    let client: Arc<dyn LlmClient> = match provider_type {
        #[cfg(feature = "openai")]
        ProviderType::OpenAi => {
            // Resolve defaults via ProviderRegistry v2 (native provider) to keep
            // metadata (aliases, prefixes) in sync; base URL still flows through
            // BuildContext into the OpenAI provider factory.
            let resolved_base_opt = {
                let registry = crate::registry::global_registry();
                let mut guard = registry
                    .write()
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
            let default_base =
                resolved_base_opt.unwrap_or_else(|| "https://api.openai.com/v1".to_string());
            let resolved_base =
                crate::utils::builder_helpers::resolve_base_url(base_url, &default_base);

            // Build unified context and delegate to OpenAIProviderFactory.
            let mut ctx = BuildContext::default();
            ctx.http_client = Some(built_http_client.clone());
            ctx.http_config = Some(http_config.clone());
            ctx.api_key = Some(api_key.clone());
            ctx.base_url = Some(resolved_base);
            ctx.organization = organization.clone();
            ctx.project = project.clone();
            ctx.tracing_config = builder.tracing_config.clone();
            ctx.http_interceptors = interceptors.clone();
            ctx.model_middlewares = user_model_middlewares.clone();
            ctx.retry_options = builder.retry_options.clone();
            ctx.common_params = Some(common_params.clone());

            let factory = crate::registry::factories::OpenAIProviderFactory;
            factory
                .language_model_with_ctx(&common_params.model, &ctx)
                .await?
        }
        #[cfg(feature = "anthropic")]
        ProviderType::Anthropic => {
            // Resolve defaults via ProviderRegistry v2 (native provider)
            let resolved_base = {
                let registry = crate::registry::global_registry();
                let mut guard = registry
                    .write()
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
                    if let Some(rec) = guard.resolve("anthropic").cloned() {
                        let rec = rec.with_model_prefix("claude");
                        guard.register(rec);
                    }
                }
                // Register an alias record for Anthropic on Vertex to support id/alias lookup
                if guard.resolve("anthropic-vertex").is_none() {
                    guard.register_native(
                        "anthropic-vertex",
                        "Anthropic on Vertex",
                        None,
                        ProviderCapabilities::new()
                            .with_chat()
                            .with_streaming()
                            .with_tools(),
                    );
                    if let Some(rec) = guard.resolve("anthropic-vertex").cloned() {
                        let rec = rec
                            .with_alias("google-vertex-anthropic")
                            .with_model_prefix("claude");
                        guard.register(rec);
                    }
                }
                guard.resolve("anthropic").and_then(|r| r.base_url.clone())
            };
            // Detect Anthropic on Vertex AI:
            // - If provider_id is explicitly set to "anthropic-vertex"
            // - Or base_url contains aiplatform.googleapis.com
            // - Or Authorization header is present (and base_url looks like Vertex)
            let is_vertex = builder
                .provider_id
                .as_deref()
                .map(|n| n == "anthropic-vertex")
                .unwrap_or(false)
                || base_url
                    .as_ref()
                    .map(|u| u.contains("aiplatform.googleapis.com"))
                    .unwrap_or(false);
            if is_vertex {
                let base = base_url
                    .or_else(|| Some(resolved_base.clone().unwrap_or_default()))
                    .unwrap_or_default();

                // Build unified context and delegate to AnthropicVertexProviderFactory.
                let mut ctx = BuildContext::default();
                ctx.http_client = Some(built_http_client.clone());
                ctx.http_config = Some(http_config.clone());
                ctx.base_url = Some(base);
                ctx.tracing_config = builder.tracing_config.clone();
                ctx.model_middlewares = user_model_middlewares.clone();
                ctx.retry_options = builder.retry_options.clone();
                ctx.common_params = Some(common_params.clone());

                let factory = crate::registry::factories::AnthropicVertexProviderFactory;
                factory
                    .language_model_with_ctx(&common_params.model, &ctx)
                    .await?
            } else {
                let anthropic_base_url = base_url
                    .or(resolved_base)
                    .unwrap_or_else(|| "https://api.anthropic.com".to_string());

                // Build unified context and delegate to AnthropicProviderFactory.
                let mut ctx = BuildContext::default();
                ctx.http_client = Some(built_http_client.clone());
                ctx.http_config = Some(http_config.clone());
                ctx.api_key = Some(api_key.clone());
                ctx.base_url = Some(anthropic_base_url);
                ctx.tracing_config = builder.tracing_config.clone();
                ctx.http_interceptors = interceptors.clone();
                ctx.model_middlewares = user_model_middlewares.clone();
                ctx.retry_options = builder.retry_options.clone();
                ctx.common_params = Some(common_params.clone());

                let factory = crate::registry::factories::AnthropicProviderFactory;
                factory
                    .language_model_with_ctx(&common_params.model, &ctx)
                    .await?
            }
        }
        #[cfg(feature = "google")]
        ProviderType::Gemini => {
            // Resolve defaults via ProviderRegistry v2 (native provider) to ensure
            // aliases and model prefixes are registered for routing.
            let resolved_base_opt = {
                let registry = crate::registry::global_registry();
                let mut guard = registry
                    .write()
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
                // Add common aliases and model prefixes to ease lookup/routing
                if let Some(rec) = guard.resolve("gemini").cloned() {
                    let rec = rec
                        .with_alias("google")
                        .with_alias("google-gemini")
                        .with_alias("google-vertex")
                        .with_model_prefix("gemini");
                    guard.register(rec);
                }
                guard.resolve("gemini").and_then(|r| r.base_url.clone())
            };
            let default_base = resolved_base_opt
                .unwrap_or_else(|| "https://generativelanguage.googleapis.com/v1beta".to_string());
            let resolved_base =
                crate::utils::builder_helpers::resolve_base_url(base_url, &default_base);

            // Build unified context and delegate to GeminiProviderFactory.
            let mut ctx = BuildContext::default();
            ctx.http_client = Some(built_http_client.clone());
            ctx.http_config = Some(http_config.clone());
            // Only override API key when explicitly set; otherwise allow factory
            // to fall back to GEMINI_API_KEY or token-based auth.
            if builder.api_key.is_some() {
                ctx.api_key = Some(api_key.clone());
            }
            ctx.base_url = Some(resolved_base);
            ctx.tracing_config = builder.tracing_config.clone();
            ctx.http_interceptors = interceptors.clone();
            ctx.model_middlewares = user_model_middlewares.clone();
            ctx.retry_options = builder.retry_options.clone();
            ctx.common_params = Some(common_params.clone());
            #[cfg(feature = "google")]
            {
                ctx.gemini_token_provider = builder.gemini_token_provider.clone();
            }

            let factory = crate::registry::factories::GeminiProviderFactory;
            factory
                .language_model_with_ctx(&common_params.model, &ctx)
                .await?
        }
        #[cfg(feature = "xai")]
        ProviderType::XAI => {
            // Build unified context and delegate to XAIProviderFactory.
            let default_base = "https://api.x.ai/v1".to_string();
            let resolved_base =
                crate::utils::builder_helpers::resolve_base_url(base_url, &default_base);

            let mut ctx = BuildContext::default();
            ctx.http_client = Some(built_http_client.clone());
            ctx.http_config = Some(http_config.clone());
            ctx.api_key = Some(api_key.clone());
            ctx.base_url = Some(resolved_base);
            ctx.tracing_config = builder.tracing_config.clone();
            ctx.http_interceptors = interceptors.clone();
            ctx.model_middlewares = user_model_middlewares.clone();
            ctx.retry_options = builder.retry_options.clone();
            ctx.common_params = Some(common_params.clone());

            let factory = crate::registry::factories::XAIProviderFactory;
            factory
                .language_model_with_ctx(&common_params.model, &ctx)
                .await?
        }
        #[cfg(feature = "ollama")]
        ProviderType::Ollama => {
            // Build unified context and delegate to OllamaProviderFactory.
            let ollama_base_url =
                crate::utils::builder_helpers::resolve_base_url(base_url, "http://localhost:11434");

            let mut ctx = BuildContext::default();
            ctx.http_client = Some(built_http_client.clone());
            ctx.http_config = Some(http_config.clone());
            ctx.base_url = Some(ollama_base_url);
            ctx.tracing_config = builder.tracing_config.clone();
            ctx.http_interceptors = interceptors.clone();
            ctx.model_middlewares = user_model_middlewares.clone();
            ctx.retry_options = builder.retry_options.clone();
            ctx.common_params = Some(common_params.clone());

            let factory = crate::registry::factories::OllamaProviderFactory;
            factory
                .language_model_with_ctx(&common_params.model, &ctx)
                .await?
        }
        #[cfg(feature = "groq")]
        ProviderType::Groq => {
            // Build unified context and delegate to GroqProviderFactory.
            let mut ctx = BuildContext::default();
            ctx.http_client = Some(built_http_client.clone());
            ctx.http_config = Some(http_config.clone());
            ctx.api_key = Some(api_key.clone());
            ctx.base_url = base_url.clone();
            ctx.tracing_config = builder.tracing_config.clone();
            ctx.http_interceptors = interceptors.clone();
            ctx.model_middlewares = user_model_middlewares.clone();
            ctx.retry_options = builder.retry_options.clone();
            ctx.common_params = Some(common_params.clone());

            let factory = crate::registry::factories::GroqProviderFactory;
            factory
                .language_model_with_ctx(&common_params.model, &ctx)
                .await?
        }
        ProviderType::Custom(name) => {
            #[cfg(feature = "openai")]
            {
                // Build unified context and delegate to a generic OpenAI-compatible
                // provider factory using the given provider id.
                let mut ctx = BuildContext::default();
                ctx.http_client = Some(built_http_client.clone());
                ctx.http_config = Some(http_config.clone());
                ctx.api_key = Some(api_key.clone());
                ctx.base_url = base_url.clone();
                ctx.tracing_config = builder.tracing_config.clone();
                ctx.http_interceptors = interceptors.clone();
                ctx.model_middlewares = user_model_middlewares.clone();
                ctx.retry_options = builder.retry_options.clone();
                ctx.common_params = Some(common_params.clone());

                let factory =
                    crate::registry::factories::OpenAICompatibleProviderFactory::new(name.clone());
                factory
                    .language_model_with_ctx(&common_params.model, &ctx)
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
        #[cfg(feature = "minimaxi")]
        ProviderType::MiniMaxi => {
            // Use Anthropic-compatible endpoint by default
            let default_base =
                crate::providers::minimaxi::config::MinimaxiConfig::DEFAULT_BASE_URL.to_string();
            let resolved_base =
                crate::utils::builder_helpers::resolve_base_url(base_url, &default_base);

            // Build unified context and delegate to MiniMaxiProviderFactory.
            let mut ctx = BuildContext::default();
            ctx.http_client = Some(built_http_client.clone());
            ctx.http_config = Some(http_config.clone());
            ctx.api_key = Some(api_key.clone());
            ctx.base_url = Some(resolved_base);
            ctx.tracing_config = builder.tracing_config.clone();
            ctx.http_interceptors = interceptors.clone();
            ctx.model_middlewares = user_model_middlewares.clone();
            ctx.retry_options = builder.retry_options.clone();
            ctx.common_params = Some(common_params.clone());

            let factory = crate::registry::factories::MiniMaxiProviderFactory;
            factory
                .language_model_with_ctx(&common_params.model, &ctx)
                .await?
        }
        #[cfg(not(feature = "minimaxi"))]
        ProviderType::MiniMaxi => {
            return Err(LlmError::UnsupportedOperation(
                "MiniMaxi provider requires the 'minimaxi' feature to be enabled".to_string(),
            ));
        }
    };

    // Retry options are now applied directly to underlying provider clients via
    // BuildContext and ProviderFactory. The outer Siumai wrapper keeps a
    // separate opt-in retry layer via `with_retry_options` for advanced use.
    let siumai = super::Siumai::new(client);
    Ok(siumai)
}
