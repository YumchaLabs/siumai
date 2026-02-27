use crate::error::LlmError;

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
    use crate::types::ProviderType;
    use std::sync::Arc;

    // Use unified HTTP client builder from utils
    use crate::execution::http::client::build_http_client_from_config;

    // Resolve provider id aliases into canonical ids and keep provider_type consistent.
    if let Some(raw_id) = builder.provider_id.clone() {
        let (provider_id, inferred_type) = super::resolver::resolve_provider(&raw_id);
        builder.provider_id = Some(provider_id);

        if let Some(explicit_type) = builder.provider_type.clone()
            && explicit_type != inferred_type
        {
            return Err(LlmError::ConfigurationError(format!(
                "Conflicting provider configuration: provider_id '{}' resolves to '{}', but provider_type is '{}'",
                raw_id, inferred_type, explicit_type
            )));
        }

        builder.provider_type = Some(inferred_type);
    } else if let Some(pt) = builder.provider_type.clone() {
        // If only provider_type is set, derive a reasonable default provider_id for downstream routing.
        builder.provider_id = Some(pt.to_string());
    }

    // Best-effort provider suggestion by model prefix (when provider is not set)
    if builder.provider_type.is_none()
        && !builder.common_params.model.is_empty()
        && let Some(pt) =
            super::resolver::infer_provider_type_from_model(&builder.common_params.model)
    {
        builder.provider_type = Some(pt.clone());
        builder.provider_id.get_or_insert(pt.to_string());
    }

    // Extract all needed values first to avoid borrow checker issues
    let provider_type = builder
        .provider_type
        .clone()
        .ok_or_else(|| LlmError::ConfigurationError("Provider type not specified".to_string()))?;

    // Validate explicit API key.
    // Actual API key resolution (context override -> env vars) is handled inside provider factories.
    if let Some(key) = builder.api_key.as_ref()
        && key.trim().is_empty()
    {
        return Err(LlmError::ConfigurationError(
            "API key cannot be empty".to_string(),
        ));
    }

    // Extract all needed values to avoid borrow checker issues
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

    let base_ctx = BuildContext {
        http_client: Some(built_http_client.clone()),
        http_transport: builder.http_transport.clone(),
        http_config: Some(http_config.clone()),
        api_key: builder.api_key.clone(),
        base_url: builder.base_url.clone(),
        tracing_config: builder.tracing_config.clone(),
        http_interceptors: interceptors.clone(),
        model_middlewares: user_model_middlewares.clone(),
        retry_options: builder.retry_options.clone(),
        common_params: Some(common_params.clone()),
        provider_id: builder.provider_id.clone(),
        #[cfg(any(feature = "google", feature = "google-vertex"))]
        gemini_token_provider: builder.gemini_token_provider.clone(),
        ..Default::default()
    };

    let client: Arc<dyn LlmClient> = match provider_type {
        #[cfg(feature = "openai")]
        ProviderType::OpenAi => {
            let mut ctx = base_ctx.clone();
            ctx.organization = organization.clone();
            ctx.project = project.clone();

            let factory = crate::registry::factories::OpenAIProviderFactory;
            factory
                .language_model_with_ctx(&common_params.model, &ctx)
                .await?
        }
        #[cfg(feature = "anthropic")]
        ProviderType::Anthropic => {
            let base_url = builder.base_url.clone();
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
                    let mut ctx = base_ctx.clone();
                    ctx.base_url = Some(base);

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
                // Build unified context and delegate to AnthropicProviderFactory.
                let mut ctx = base_ctx.clone();

                let factory = crate::registry::factories::AnthropicProviderFactory;
                factory
                    .language_model_with_ctx(&common_params.model, &ctx)
                    .await?
            }
        }
        #[cfg(feature = "google")]
        ProviderType::Gemini => {
            // Build unified context and delegate to GeminiProviderFactory.
            let ctx = base_ctx.clone();

            let factory = crate::registry::factories::GeminiProviderFactory;
            factory
                .language_model_with_ctx(&common_params.model, &ctx)
                .await?
        }
        #[cfg(feature = "xai")]
        ProviderType::XAI => {
            // Build unified context and delegate to XAIProviderFactory.
            let ctx = base_ctx.clone();

            let factory = crate::registry::factories::XAIProviderFactory;
            factory
                .language_model_with_ctx(&common_params.model, &ctx)
                .await?
        }
        #[cfg(feature = "ollama")]
        ProviderType::Ollama => {
            // Build unified context and delegate to OllamaProviderFactory.
            let ctx = base_ctx.clone();

            let factory = crate::registry::factories::OllamaProviderFactory;
            factory
                .language_model_with_ctx(&common_params.model, &ctx)
                .await?
        }
        #[cfg(feature = "groq")]
        ProviderType::Groq => {
            // Build unified context and delegate to GroqProviderFactory.
            let ctx = base_ctx.clone();

            let factory = crate::registry::factories::GroqProviderFactory;
            factory
                .language_model_with_ctx(&common_params.model, &ctx)
                .await?
        }
        #[cfg(feature = "azure")]
        ProviderType::Custom(name) if name == "azure" => {
            let ctx = base_ctx.clone();

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
            let ctx = base_ctx.clone();

            let factory = crate::registry::factories::AnthropicVertexProviderFactory;
            factory
                .language_model_with_ctx(&common_params.model, &ctx)
                .await?
        }
        #[cfg(feature = "google-vertex")]
        ProviderType::Custom(name) if name == "vertex" => {
            let ctx = base_ctx.clone();

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
                let ctx = base_ctx.clone();

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
            // Build unified context and delegate to MiniMaxiProviderFactory.
            let ctx = base_ctx.clone();

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
    use super::super::resolver::infer_provider_type_from_model;

    #[test]
    fn infer_provider_empty_is_none() {
        assert!(infer_provider_type_from_model("").is_none());
        assert!(infer_provider_type_from_model("   ").is_none());
    }

    #[test]
    fn infer_provider_is_conservative_for_openai_like_models() {
        assert!(infer_provider_type_from_model("gpt-4o").is_none());
        assert!(infer_provider_type_from_model("gpt-4.1").is_none());
        assert!(infer_provider_type_from_model("text-embedding-3-large").is_none());
    }

    #[cfg(feature = "anthropic")]
    #[test]
    fn infer_provider_anthropic_claude() {
        use crate::types::ProviderType;
        assert_eq!(
            infer_provider_type_from_model("claude-3-5-sonnet-20241022"),
            Some(ProviderType::Anthropic)
        );
    }

    #[cfg(feature = "google")]
    #[test]
    fn infer_provider_gemini_prefixes() {
        use crate::types::ProviderType;
        assert_eq!(
            infer_provider_type_from_model("gemini-2.5-flash"),
            Some(ProviderType::Gemini)
        );
        assert_eq!(
            infer_provider_type_from_model("models/gemini-1.5-flash"),
            Some(ProviderType::Gemini)
        );
        assert_eq!(
            infer_provider_type_from_model("publishers/google/models/gemini-1.5-pro"),
            Some(ProviderType::Gemini)
        );
    }
}
