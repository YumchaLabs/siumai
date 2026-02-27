use crate::error::LlmError;
use crate::provider::ids;

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
fn select_factory(
    provider_id: &str,
) -> Result<std::sync::Arc<dyn crate::registry::entry::ProviderFactory>, LlmError> {
    use crate::registry::entry::ProviderFactory;
    use std::sync::Arc;

    match ids::BuiltinProviderId::parse(provider_id) {
        Some(
            ids::BuiltinProviderId::OpenAi
            | ids::BuiltinProviderId::OpenAiChat
            | ids::BuiltinProviderId::OpenAiResponses,
        ) => {
            #[cfg(feature = "openai")]
            {
                Ok(Arc::new(crate::registry::factories::OpenAIProviderFactory)
                    as Arc<dyn ProviderFactory>)
            }
            #[cfg(not(feature = "openai"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "OpenAI provider requires the 'openai' feature to be enabled".to_string(),
                ))
            }
        }
        Some(ids::BuiltinProviderId::Anthropic) => {
            #[cfg(feature = "anthropic")]
            {
                Ok(
                    Arc::new(crate::registry::factories::AnthropicProviderFactory)
                        as Arc<dyn ProviderFactory>,
                )
            }
            #[cfg(not(feature = "anthropic"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "Anthropic provider requires the 'anthropic' feature to be enabled".to_string(),
                ))
            }
        }
        Some(ids::BuiltinProviderId::AnthropicVertex) => {
            #[cfg(feature = "google-vertex")]
            {
                Ok(
                    Arc::new(crate::registry::factories::AnthropicVertexProviderFactory)
                        as Arc<dyn ProviderFactory>,
                )
            }
            #[cfg(not(feature = "google-vertex"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "Anthropic on Vertex requires the 'google-vertex' feature to be enabled"
                        .to_string(),
                ))
            }
        }
        Some(ids::BuiltinProviderId::Gemini) => {
            #[cfg(feature = "google")]
            {
                Ok(Arc::new(crate::registry::factories::GeminiProviderFactory)
                    as Arc<dyn ProviderFactory>)
            }
            #[cfg(not(feature = "google"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "Gemini provider requires the 'google' feature to be enabled".to_string(),
                ))
            }
        }
        Some(ids::BuiltinProviderId::Vertex) => {
            #[cfg(feature = "google-vertex")]
            {
                Ok(
                    Arc::new(crate::registry::factories::GoogleVertexProviderFactory)
                        as Arc<dyn ProviderFactory>,
                )
            }
            #[cfg(not(feature = "google-vertex"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "Google Vertex provider requires the 'google-vertex' feature to be enabled"
                        .to_string(),
                ))
            }
        }
        Some(ids::BuiltinProviderId::Ollama) => {
            #[cfg(feature = "ollama")]
            {
                Ok(Arc::new(crate::registry::factories::OllamaProviderFactory)
                    as Arc<dyn ProviderFactory>)
            }
            #[cfg(not(feature = "ollama"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "Ollama provider requires the 'ollama' feature to be enabled".to_string(),
                ))
            }
        }
        Some(ids::BuiltinProviderId::Xai) => {
            #[cfg(feature = "xai")]
            {
                Ok(Arc::new(crate::registry::factories::XAIProviderFactory)
                    as Arc<dyn ProviderFactory>)
            }
            #[cfg(not(feature = "xai"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "xAI provider requires the 'xai' feature to be enabled".to_string(),
                ))
            }
        }
        Some(ids::BuiltinProviderId::Groq) => {
            #[cfg(feature = "groq")]
            {
                Ok(Arc::new(crate::registry::factories::GroqProviderFactory)
                    as Arc<dyn ProviderFactory>)
            }
            #[cfg(not(feature = "groq"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "Groq provider requires the 'groq' feature to be enabled".to_string(),
                ))
            }
        }
        Some(ids::BuiltinProviderId::MiniMaxi) => {
            #[cfg(feature = "minimaxi")]
            {
                Ok(
                    Arc::new(crate::registry::factories::MiniMaxiProviderFactory)
                        as Arc<dyn ProviderFactory>,
                )
            }
            #[cfg(not(feature = "minimaxi"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "MiniMaxi provider requires the 'minimaxi' feature to be enabled".to_string(),
                ))
            }
        }
        Some(ids::BuiltinProviderId::Azure | ids::BuiltinProviderId::AzureChat) => {
            #[cfg(feature = "azure")]
            {
                let chat_mode = match provider_id {
                    ids::AZURE_CHAT => {
                        siumai_provider_azure::providers::azure_openai::AzureChatMode::ChatCompletions
                    }
                    _ => siumai_provider_azure::providers::azure_openai::AzureChatMode::Responses,
                };
                Ok(
                    Arc::new(crate::registry::factories::AzureOpenAiProviderFactory::new(
                        chat_mode,
                    )) as Arc<dyn ProviderFactory>,
                )
            }
            #[cfg(not(feature = "azure"))]
            {
                Err(LlmError::UnsupportedOperation(
                    "Azure OpenAI provider requires the 'azure' feature to be enabled".to_string(),
                ))
            }
        }
        None => {
            #[cfg(feature = "openai")]
            {
                Ok(Arc::new(
                    crate::registry::factories::OpenAICompatibleProviderFactory::new(
                        provider_id.to_string(),
                    ),
                ) as Arc<dyn ProviderFactory>)
            }
            #[cfg(not(feature = "openai"))]
            {
                Err(LlmError::UnsupportedOperation(format!(
                    "Custom provider '{}' requires 'openai' feature",
                    provider_id
                )))
            }
        }
    }
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
    use crate::registry::entry::BuildContext;
    use std::sync::Arc;

    // Use unified HTTP client builder from utils
    use crate::execution::http::client::build_http_client_from_config;

    // Normalize provider id aliases into canonical ids.
    if let Some(raw_id) = builder.provider_id.clone() {
        builder.provider_id = Some(super::resolver::normalize_provider_id(&raw_id));
    }

    // Best-effort provider suggestion by model prefix (when provider is not set).
    if builder.provider_id.is_none()
        && !builder.common_params.model.is_empty()
        && let Some(provider_id) =
            super::resolver::infer_provider_id_from_model(&builder.common_params.model)
    {
        builder.provider_id = Some(provider_id);
    }

    let provider_id = builder
        .provider_id
        .clone()
        .ok_or_else(|| LlmError::ConfigurationError("Provider id not specified".to_string()))?;

    // Some routing decisions depend on base_url (Anthropic on Vertex).
    let mut effective_provider_id = provider_id.clone();
    if effective_provider_id == ids::ANTHROPIC {
        let base_url = builder.base_url.clone();
        let is_vertex = base_url
            .as_ref()
            .map(|u| u.contains("aiplatform.googleapis.com"))
            .unwrap_or(false);
        if is_vertex {
            effective_provider_id = ids::ANTHROPIC_VERTEX.to_string();
        }
    }

    // Validate explicit API key.
    // Actual API key resolution (context override -> env vars) is handled inside provider factories.
    if let Some(key) = builder.api_key.as_ref()
        && key.trim().is_empty()
    {
        return Err(LlmError::ConfigurationError(
            "API key cannot be empty".to_string(),
        ));
    }

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
        // Set default model based on provider id.
        common_params.model = match effective_provider_id.as_str() {
            #[cfg(feature = "openai")]
            id if ids::is_openai_family(id) => {
                siumai_provider_openai::providers::openai::model_constants::gpt_4o::GPT_4O
                    .to_string()
            }
            #[cfg(not(feature = "openai"))]
            id if ids::is_openai_family(id) => {
                return Err(LlmError::UnsupportedOperation(
                    "OpenAI feature not enabled".to_string(),
                ));
            }
            #[cfg(feature = "azure")]
            id if ids::is_azure_family(id) => {
                return Err(LlmError::ConfigurationError(
                    "Azure OpenAI requires an explicit model (deployment id)".to_string(),
                ));
            }
            #[cfg(not(feature = "azure"))]
            id if ids::is_azure_family(id) => {
                return Err(LlmError::UnsupportedOperation(
                    "Azure OpenAI provider requires the 'azure' feature to be enabled".to_string(),
                ));
            }
            #[cfg(feature = "anthropic")]
            ids::ANTHROPIC => siumai_provider_anthropic::providers::anthropic::model_constants::claude_sonnet_3_5::CLAUDE_3_5_SONNET_20241022.to_string(),
            #[cfg(not(feature = "anthropic"))]
            ids::ANTHROPIC => {
                return Err(LlmError::UnsupportedOperation(
                    "Anthropic feature not enabled".to_string(),
                ));
            }
            #[cfg(feature = "google")]
            ids::GEMINI => siumai_provider_gemini::providers::gemini::model_constants::gemini_2_5_flash::GEMINI_2_5_FLASH.to_string(),
            #[cfg(not(feature = "google"))]
            ids::GEMINI => {
                return Err(LlmError::UnsupportedOperation(
                    "Google feature not enabled".to_string(),
                ));
            }
            #[cfg(feature = "google-vertex")]
            ids::ANTHROPIC_VERTEX => {
                "claude-3-5-sonnet-20241022".to_string()
            }
            #[cfg(feature = "google-vertex")]
            ids::VERTEX => {
                "imagen-3.0-generate-002".to_string()
            }
            #[cfg(not(feature = "google-vertex"))]
            ids::ANTHROPIC_VERTEX | ids::VERTEX => {
                return Err(LlmError::UnsupportedOperation(
                    "Google Vertex feature not enabled".to_string(),
                ));
            }
            #[cfg(feature = "ollama")]
            ids::OLLAMA => "llama3.2".to_string(),
            #[cfg(not(feature = "ollama"))]
            ids::OLLAMA => {
                return Err(LlmError::UnsupportedOperation(
                    "Ollama feature not enabled".to_string(),
                ));
            }
            #[cfg(feature = "xai")]
            ids::XAI => "grok-beta".to_string(),
            #[cfg(not(feature = "xai"))]
            ids::XAI => {
                return Err(LlmError::UnsupportedOperation(
                    "xAI feature not enabled".to_string(),
                ));
            }
            #[cfg(feature = "groq")]
            ids::GROQ => "llama-3.1-70b-versatile".to_string(),
            #[cfg(not(feature = "groq"))]
            ids::GROQ => {
                return Err(LlmError::UnsupportedOperation(
                    "Groq feature not enabled".to_string(),
                ));
            }
            #[cfg(feature = "minimaxi")]
            ids::MINIMAXI => "MiniMax-M2".to_string(),
            #[cfg(not(feature = "minimaxi"))]
            ids::MINIMAXI => {
                return Err(LlmError::UnsupportedOperation(
                    "MiniMaxi feature not enabled".to_string(),
                ));
            }
            other => {
                // Use shared helper function to get default model from registry (OpenAI-compatible).
                #[cfg(feature = "openai")]
                {
                    crate::utils::builder_helpers::get_effective_model("", other)
                }
                #[cfg(not(feature = "openai"))]
                {
                    let _ = other;
                    "default-model".to_string()
                }
            }
        };
    }

    // Normalize model ID for OpenAI-compatible providers (handle aliases)
    // This ensures that model aliases like "chat" -> "deepseek-chat" are properly resolved
    #[cfg(feature = "openai")]
    {
        if super::resolver::is_openai_compatible_provider_id(&effective_provider_id) {
            let normalized_model = crate::utils::builder_helpers::normalize_model_id(
                &effective_provider_id,
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
        organization: builder.organization.clone(),
        project: builder.project.clone(),
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

    let factory = select_factory(&effective_provider_id)?;
    let mut ctx = base_ctx.clone();
    ctx.provider_id = Some(effective_provider_id);
    let client: Arc<dyn LlmClient> = factory
        .language_model_with_ctx(&common_params.model, &ctx)
        .await?;

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
    use super::super::resolver::infer_provider_id_from_model;

    #[test]
    fn infer_provider_empty_is_none() {
        assert!(infer_provider_id_from_model("").is_none());
        assert!(infer_provider_id_from_model("   ").is_none());
    }

    #[test]
    fn infer_provider_is_conservative_for_openai_like_models() {
        assert!(infer_provider_id_from_model("gpt-4o").is_none());
        assert!(infer_provider_id_from_model("gpt-4.1").is_none());
        assert!(infer_provider_id_from_model("text-embedding-3-large").is_none());
    }

    #[cfg(feature = "anthropic")]
    #[test]
    fn infer_provider_anthropic_claude() {
        assert_eq!(
            infer_provider_id_from_model("claude-3-5-sonnet-20241022"),
            Some("anthropic".to_string())
        );
    }

    #[cfg(feature = "google")]
    #[test]
    fn infer_provider_gemini_prefixes() {
        assert_eq!(
            infer_provider_id_from_model("gemini-2.5-flash"),
            Some("gemini".to_string())
        );
        assert_eq!(
            infer_provider_id_from_model("models/gemini-1.5-flash"),
            Some("gemini".to_string())
        );
        assert_eq!(
            infer_provider_id_from_model("publishers/google/models/gemini-1.5-pro"),
            Some("gemini".to_string())
        );
    }
}
