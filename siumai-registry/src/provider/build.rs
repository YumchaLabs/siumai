use crate::error::LlmError;
#[cfg(any(
    test,
    feature = "openai",
    feature = "azure",
    feature = "anthropic",
    feature = "google",
    feature = "google-vertex",
    feature = "ollama",
    feature = "xai",
    feature = "groq",
    feature = "minimaxi"
))]
use crate::types::ProviderType;

#[cfg(any(
    test,
    feature = "openai",
    feature = "azure",
    feature = "anthropic",
    feature = "google",
    feature = "google-vertex",
    feature = "ollama",
    feature = "xai",
    feature = "groq",
    feature = "minimaxi"
))]
fn infer_provider_from_model(model: &str) -> Option<ProviderType> {
    let model = model.trim();
    if model.is_empty() {
        return None;
    }

    // Keep inference intentionally conservative:
    // - only infer when the model prefix is strongly associated with a provider
    // - do not infer OpenAI / OpenAI-compatible providers because prefixes overlap heavily
    #[cfg(feature = "anthropic")]
    {
        if model.starts_with("claude") {
            return Some(ProviderType::Anthropic);
        }
    }

    #[cfg(feature = "google")]
    {
        if model.starts_with("gemini")
            || model.contains("/models/gemini")
            || model.contains("models/gemini")
            || model.contains("publishers/google/models/gemini")
        {
            return Some(ProviderType::Gemini);
        }
    }

    #[cfg(feature = "google-vertex")]
    {
        if model.starts_with("imagen")
            || model.contains("/models/imagen")
            || model.contains("models/imagen")
            || model.contains("publishers/google/models/imagen")
        {
            return Some(ProviderType::Custom("vertex".to_string()));
        }
    }

    None
}

/// Build the unified Siumai provider from SiumaiBuilder
#[cfg(any(
    feature = "openai",
    feature = "azure",
    feature = "anthropic",
    feature = "google",
    feature = "google-vertex",
    feature = "ollama",
    feature = "xai",
    feature = "groq",
    feature = "minimaxi"
))]
pub async fn build(mut builder: super::SiumaiBuilder) -> Result<super::Siumai, LlmError> {
    use crate::client::LlmClient;
    use crate::execution::http::interceptor::{HttpInterceptor, LoggingInterceptor};
    use crate::execution::middleware::LanguageModelMiddleware;
    use crate::registry::entry::{BuildContext, ProviderFactory};
    use std::sync::Arc;

    // Use unified HTTP client builder from utils
    use crate::execution::http::client::build_http_client_from_config;

    // Best-effort provider suggestion by model prefix (when provider is not set)
    if builder.provider_type.is_none()
        && !builder.common_params.model.is_empty()
        && let Some(pt) = infer_provider_from_model(&builder.common_params.model)
    {
        builder.provider_type = Some(pt);
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
        ProviderType::Anthropic => {
            // Anthropic on Vertex AI uses Bearer auth (Authorization header) rather than an API key.
            // If we detect Vertex mode and Authorization is present, do not enforce API key.
            let is_vertex = builder
                .provider_id
                .as_deref()
                .map(|n| n == "anthropic-vertex")
                .unwrap_or(false)
                || builder
                    .base_url
                    .as_ref()
                    .map(|u| u.contains("aiplatform.googleapis.com"))
                    .unwrap_or(false);
            if is_vertex {
                let has_auth_header = builder
                    .http_config
                    .headers
                    .keys()
                    .any(|k| k.eq_ignore_ascii_case("authorization"));
                !has_auth_header
            } else {
                true
            }
        }
        #[cfg(feature = "google-vertex")]
        ProviderType::Custom(ref id) if id == "anthropic-vertex" => false,
        #[cfg(feature = "google-vertex")]
        ProviderType::Custom(ref id) if id == "vertex" => false,
        ProviderType::Gemini => {
            let has_auth_header = builder
                .http_config
                .headers
                .keys()
                .any(|k| k.eq_ignore_ascii_case("authorization"));
            let has_token_provider = {
                #[cfg(any(feature = "google", feature = "google-vertex"))]
                {
                    builder.gemini_token_provider.is_some()
                }
                #[cfg(not(any(feature = "google", feature = "google-vertex")))]
                {
                    false
                }
            };
            !(has_auth_header || has_token_provider)
        }
        _ => true,
    };

    // Vertex supports:
    // - Express mode: API key passed as query parameter (`?key=...`).
    // - Enterprise mode: Bearer auth via Authorization header or TokenProvider (recommended).
    //
    // We intentionally do not hard-require auth at build-time because users may supply auth
    // via external interceptors, proxies, or per-request headers.
    #[cfg(feature = "google-vertex")]
    let vertex_api_key = if matches!(provider_type, ProviderType::Custom(ref id) if id == "vertex")
    {
        let from_builder = builder.api_key.clone();
        let from_env = std::env::var("GOOGLE_VERTEX_API_KEY").ok();
        from_builder.or(from_env).and_then(|k| {
            let trimmed = k.trim().to_string();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed)
            }
        })
    } else {
        None
    };

    #[allow(unused_variables)]
    let api_key = if requires_api_key {
        // Try to get API key from builder first, then from environment variable
        if let Some(key) = builder.api_key.clone() {
            if key.trim().is_empty() {
                return Err(LlmError::ConfigurationError(
                    "API key cannot be empty".to_string(),
                ));
            }
            key
        } else {
            // For Custom providers (OpenAI-compatible), use shared helper function
            if let ProviderType::Custom(ref provider_id) = provider_type {
                #[cfg(all(feature = "builtins", feature = "openai"))]
                {
                    if let Some(cfg) =
                        siumai_provider_openai_compatible::providers::openai_compatible::get_provider_config(
                            provider_id,
                        )
                    {
                        crate::utils::builder_helpers::get_api_key_with_envs(
                            None,
                            provider_id,
                            cfg.api_key_env.as_deref(),
                            &cfg.api_key_env_aliases,
                        )?
                    } else {
                        crate::utils::builder_helpers::get_api_key_with_env(None, provider_id)?
                    }
                }

                #[cfg(not(all(feature = "builtins", feature = "openai")))]
                {
                    crate::utils::builder_helpers::get_api_key_with_env(None, provider_id)?
                }
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
        common_params.model = match provider_type {
            #[cfg(feature = "openai")]
            ProviderType::OpenAi => siumai_provider_openai::providers::openai::model_constants::gpt_4o::GPT_4O.to_string(),
            #[cfg(feature = "azure")]
            ProviderType::Custom(ref provider_id) if provider_id == "azure" => {
                return Err(LlmError::ConfigurationError(
                    "Azure OpenAI requires an explicit model (deployment id)".to_string(),
                ));
            }
            #[cfg(feature = "anthropic")]
            ProviderType::Anthropic => siumai_provider_anthropic::providers::anthropic::model_constants::claude_sonnet_3_5::CLAUDE_3_5_SONNET_20241022.to_string(),
            #[cfg(feature = "google")]
            ProviderType::Gemini => siumai_provider_gemini::providers::gemini::model_constants::gemini_2_5_flash::GEMINI_2_5_FLASH.to_string(),
            #[cfg(feature = "google-vertex")]
            ProviderType::Custom(ref provider_id) if provider_id == "anthropic-vertex" => {
                "claude-3-5-sonnet-20241022".to_string()
            }
            #[cfg(feature = "google-vertex")]
            ProviderType::Custom(ref provider_id) if provider_id == "vertex" => {
                "imagen-3.0-generate-002".to_string()
            }
            #[cfg(feature = "ollama")]
            ProviderType::Ollama => "llama3.2".to_string(),
            #[cfg(feature = "xai")]
            ProviderType::XAI => "grok-beta".to_string(),
            #[cfg(feature = "groq")]
            ProviderType::Groq => "llama-3.1-70b-versatile".to_string(),
            #[cfg(feature = "minimaxi")]
            ProviderType::MiniMaxi => "MiniMax-M2".to_string(),
            ProviderType::Custom(ref provider_id) => {
                // Use shared helper function to get default model from registry
                #[cfg(feature = "openai")]
                {
                    crate::utils::builder_helpers::get_effective_model("", provider_id)
                }
                #[cfg(not(feature = "openai"))]
                {
                    let _ = provider_id;
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
            let default_base = "https://api.openai.com/v1".to_string();
            let resolved_base =
                crate::utils::builder_helpers::resolve_base_url(base_url, &default_base);

            // Build unified context and delegate to OpenAIProviderFactory.
            let ctx = BuildContext {
                http_client: Some(built_http_client.clone()),
                http_transport: builder.http_transport.clone(),
                http_config: Some(http_config.clone()),
                api_key: Some(api_key.clone()),
                base_url: Some(resolved_base),
                organization: organization.clone(),
                project: project.clone(),
                tracing_config: builder.tracing_config.clone(),
                http_interceptors: interceptors.clone(),
                model_middlewares: user_model_middlewares.clone(),
                retry_options: builder.retry_options.clone(),
                common_params: Some(common_params.clone()),
                provider_id: builder.provider_id.clone(),
                ..Default::default()
            };

            let factory = crate::registry::factories::OpenAIProviderFactory;
            factory
                .language_model_with_ctx(&common_params.model, &ctx)
                .await?
        }
        #[cfg(feature = "anthropic")]
        ProviderType::Anthropic => {
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
                #[cfg(feature = "google-vertex")]
                {
                    let base = base_url.clone().ok_or_else(|| {
                        LlmError::ConfigurationError(
                            "Anthropic on Vertex requires an explicit base_url (aiplatform.googleapis.com)".to_string(),
                        )
                    })?;

                    // Build unified context and delegate to AnthropicVertexProviderFactory.
                    let ctx = BuildContext {
                        http_client: Some(built_http_client.clone()),
                        http_transport: builder.http_transport.clone(),
                        http_config: Some(http_config.clone()),
                        base_url: Some(base),
                        tracing_config: builder.tracing_config.clone(),
                        http_interceptors: interceptors.clone(),
                        model_middlewares: user_model_middlewares.clone(),
                        retry_options: builder.retry_options.clone(),
                        common_params: Some(common_params.clone()),
                        ..Default::default()
                    };

                    let factory = crate::registry::factories::AnthropicVertexProviderFactory;
                    factory
                        .language_model_with_ctx(&common_params.model, &ctx)
                        .await?
                }
                #[cfg(not(feature = "google-vertex"))]
                {
                    return Err(LlmError::UnsupportedOperation(
                        "Anthropic on Vertex requires the 'google-vertex' feature to be enabled"
                            .to_string(),
                    ));
                }
            } else {
                let default_base = "https://api.anthropic.com".to_string();
                let anthropic_base_url =
                    crate::utils::builder_helpers::resolve_base_url(base_url, &default_base);

                // Build unified context and delegate to AnthropicProviderFactory.
                let ctx = BuildContext {
                    http_client: Some(built_http_client.clone()),
                    http_transport: builder.http_transport.clone(),
                    http_config: Some(http_config.clone()),
                    api_key: Some(api_key.clone()),
                    base_url: Some(anthropic_base_url),
                    tracing_config: builder.tracing_config.clone(),
                    http_interceptors: interceptors.clone(),
                    model_middlewares: user_model_middlewares.clone(),
                    retry_options: builder.retry_options.clone(),
                    common_params: Some(common_params.clone()),
                    ..Default::default()
                };

                let factory = crate::registry::factories::AnthropicProviderFactory;
                factory
                    .language_model_with_ctx(&common_params.model, &ctx)
                    .await?
            }
        }
        #[cfg(feature = "google")]
        ProviderType::Gemini => {
            let default_base = "https://generativelanguage.googleapis.com/v1beta".to_string();
            let mut resolved_base =
                crate::utils::builder_helpers::resolve_base_url(base_url, &default_base);
            // Accept a more "root" style base URL (no version segment) for convenience.
            if resolved_base == "https://generativelanguage.googleapis.com" {
                resolved_base = "https://generativelanguage.googleapis.com/v1beta".to_string();
            }

            // Build unified context and delegate to GeminiProviderFactory.
            let ctx = BuildContext {
                http_client: Some(built_http_client.clone()),
                http_transport: builder.http_transport.clone(),
                http_config: Some(http_config.clone()),
                // Only override API key when explicitly set; otherwise allow factory
                // to fall back to GEMINI_API_KEY or token-based auth.
                api_key: builder.api_key.as_ref().map(|_| api_key.clone()),
                base_url: Some(resolved_base),
                tracing_config: builder.tracing_config.clone(),
                http_interceptors: interceptors.clone(),
                model_middlewares: user_model_middlewares.clone(),
                retry_options: builder.retry_options.clone(),
                common_params: Some(common_params.clone()),
                #[cfg(any(feature = "google", feature = "google-vertex"))]
                gemini_token_provider: builder.gemini_token_provider.clone(),
                ..Default::default()
            };

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

            let ctx = BuildContext {
                http_client: Some(built_http_client.clone()),
                http_transport: builder.http_transport.clone(),
                http_config: Some(http_config.clone()),
                api_key: Some(api_key.clone()),
                base_url: Some(resolved_base),
                tracing_config: builder.tracing_config.clone(),
                http_interceptors: interceptors.clone(),
                model_middlewares: user_model_middlewares.clone(),
                retry_options: builder.retry_options.clone(),
                common_params: Some(common_params.clone()),
                ..Default::default()
            };

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

            let ctx = BuildContext {
                http_client: Some(built_http_client.clone()),
                http_transport: builder.http_transport.clone(),
                http_config: Some(http_config.clone()),
                base_url: Some(ollama_base_url),
                tracing_config: builder.tracing_config.clone(),
                http_interceptors: interceptors.clone(),
                model_middlewares: user_model_middlewares.clone(),
                retry_options: builder.retry_options.clone(),
                common_params: Some(common_params.clone()),
                ..Default::default()
            };

            let factory = crate::registry::factories::OllamaProviderFactory;
            factory
                .language_model_with_ctx(&common_params.model, &ctx)
                .await?
        }
        #[cfg(feature = "groq")]
        ProviderType::Groq => {
            // Build unified context and delegate to GroqProviderFactory.
            let ctx = BuildContext {
                http_client: Some(built_http_client.clone()),
                http_transport: builder.http_transport.clone(),
                http_config: Some(http_config.clone()),
                api_key: Some(api_key.clone()),
                base_url: base_url.clone(),
                tracing_config: builder.tracing_config.clone(),
                http_interceptors: interceptors.clone(),
                model_middlewares: user_model_middlewares.clone(),
                retry_options: builder.retry_options.clone(),
                common_params: Some(common_params.clone()),
                ..Default::default()
            };

            let factory = crate::registry::factories::GroqProviderFactory;
            factory
                .language_model_with_ctx(&common_params.model, &ctx)
                .await?
        }
        #[cfg(feature = "azure")]
        ProviderType::Custom(name) if name == "azure" => {
            let ctx = BuildContext {
                http_client: Some(built_http_client.clone()),
                http_transport: builder.http_transport.clone(),
                http_config: Some(http_config.clone()),
                api_key: Some(api_key.clone()),
                base_url: base_url.clone(),
                tracing_config: builder.tracing_config.clone(),
                http_interceptors: interceptors.clone(),
                model_middlewares: user_model_middlewares.clone(),
                retry_options: builder.retry_options.clone(),
                common_params: Some(common_params.clone()),
                provider_id: builder.provider_id.clone(),
                ..Default::default()
            };

            let chat_mode = match builder.provider_id.as_deref() {
                Some("azure-chat") => {
                    siumai_provider_azure::providers::azure_openai::AzureChatMode::ChatCompletions
                }
                _ => siumai_provider_azure::providers::azure_openai::AzureChatMode::Responses,
            };

            let factory = crate::registry::factories::AzureOpenAiProviderFactory::new(chat_mode);
            factory
                .language_model_with_ctx(&common_params.model, &ctx)
                .await?
        }
        #[cfg(not(feature = "azure"))]
        ProviderType::Custom(name) if name == "azure" => {
            let _ = name;
            return Err(LlmError::UnsupportedOperation(
                "Azure OpenAI provider requires the 'azure' feature to be enabled".to_string(),
            ));
        }
        #[cfg(feature = "google-vertex")]
        ProviderType::Custom(name) if name == "anthropic-vertex" => {
            let ctx = BuildContext {
                http_client: Some(built_http_client.clone()),
                http_transport: builder.http_transport.clone(),
                http_config: Some(http_config.clone()),
                base_url: base_url.clone(),
                tracing_config: builder.tracing_config.clone(),
                http_interceptors: interceptors.clone(),
                model_middlewares: user_model_middlewares.clone(),
                retry_options: builder.retry_options.clone(),
                common_params: Some(common_params.clone()),
                ..Default::default()
            };

            let factory = crate::registry::factories::AnthropicVertexProviderFactory;
            factory
                .language_model_with_ctx(&common_params.model, &ctx)
                .await?
        }
        #[cfg(feature = "google-vertex")]
        ProviderType::Custom(name) if name == "vertex" => {
            let ctx = BuildContext {
                http_client: Some(built_http_client.clone()),
                http_transport: builder.http_transport.clone(),
                http_config: Some(http_config.clone()),
                api_key: vertex_api_key.clone(),
                base_url: base_url.clone(),
                tracing_config: builder.tracing_config.clone(),
                http_interceptors: interceptors.clone(),
                model_middlewares: user_model_middlewares.clone(),
                retry_options: builder.retry_options.clone(),
                common_params: Some(common_params.clone()),
                #[cfg(any(feature = "google", feature = "google-vertex"))]
                gemini_token_provider: builder.gemini_token_provider.clone(),
                ..Default::default()
            };

            let factory = crate::registry::factories::GoogleVertexProviderFactory;
            factory
                .language_model_with_ctx(&common_params.model, &ctx)
                .await?
        }
        ProviderType::Custom(name) => {
            #[cfg(feature = "openai")]
            {
                // Build unified context and delegate to a generic OpenAI-compatible
                // provider factory using the given provider id.
                let ctx = BuildContext {
                    http_client: Some(built_http_client.clone()),
                    http_transport: builder.http_transport.clone(),
                    http_config: Some(http_config.clone()),
                    api_key: Some(api_key.clone()),
                    base_url: base_url.clone(),
                    tracing_config: builder.tracing_config.clone(),
                    http_interceptors: interceptors.clone(),
                    model_middlewares: user_model_middlewares.clone(),
                    retry_options: builder.retry_options.clone(),
                    common_params: Some(common_params.clone()),
                    ..Default::default()
                };

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
                siumai_provider_minimaxi::providers::minimaxi::config::MinimaxiConfig::DEFAULT_BASE_URL
                    .to_string();
            let resolved_base =
                crate::utils::builder_helpers::resolve_base_url(base_url, &default_base);

            // Build unified context and delegate to MiniMaxiProviderFactory.
            let ctx = BuildContext {
                http_client: Some(built_http_client.clone()),
                http_transport: builder.http_transport.clone(),
                http_config: Some(http_config.clone()),
                api_key: Some(api_key.clone()),
                base_url: Some(resolved_base),
                tracing_config: builder.tracing_config.clone(),
                http_interceptors: interceptors.clone(),
                model_middlewares: user_model_middlewares.clone(),
                retry_options: builder.retry_options.clone(),
                common_params: Some(common_params.clone()),
                ..Default::default()
            };

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

/// Build stub when no provider features are enabled.
#[cfg(not(any(
    feature = "openai",
    feature = "azure",
    feature = "anthropic",
    feature = "google",
    feature = "google-vertex",
    feature = "ollama",
    feature = "xai",
    feature = "groq",
    feature = "minimaxi"
)))]
pub async fn build(_builder: super::SiumaiBuilder) -> Result<super::Siumai, LlmError> {
    Err(LlmError::UnsupportedOperation(
        "No provider features enabled (enable at least one of: openai, azure, anthropic, google, google-vertex, ollama, xai, groq, minimaxi)".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::infer_provider_from_model;

    #[test]
    fn infer_provider_empty_is_none() {
        assert!(infer_provider_from_model("").is_none());
        assert!(infer_provider_from_model("   ").is_none());
    }

    #[test]
    fn infer_provider_is_conservative_for_openai_like_models() {
        assert!(infer_provider_from_model("gpt-4o").is_none());
        assert!(infer_provider_from_model("gpt-4.1").is_none());
        assert!(infer_provider_from_model("text-embedding-3-large").is_none());
    }

    #[cfg(feature = "anthropic")]
    #[test]
    fn infer_provider_anthropic_claude() {
        use crate::types::ProviderType;
        assert_eq!(
            infer_provider_from_model("claude-3-5-sonnet-20241022"),
            Some(ProviderType::Anthropic)
        );
    }

    #[cfg(feature = "google")]
    #[test]
    fn infer_provider_gemini_prefixes() {
        use crate::types::ProviderType;
        assert_eq!(
            infer_provider_from_model("gemini-2.5-flash"),
            Some(ProviderType::Gemini)
        );
        assert_eq!(
            infer_provider_from_model("models/gemini-1.5-flash"),
            Some(ProviderType::Gemini)
        );
        assert_eq!(
            infer_provider_from_model("publishers/google/models/gemini-1.5-pro"),
            Some(ProviderType::Gemini)
        );
    }
}
