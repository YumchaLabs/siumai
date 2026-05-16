use crate::error::LlmError;
#[cfg(feature = "builtins")]
use crate::provider::ids;
#[cfg(feature = "azure")]
use crate::registry::entry::ProviderFactory;

#[cfg(any(
    feature = "openai",
    feature = "azure",
    feature = "anthropic",
    feature = "google",
    feature = "google-vertex",
    feature = "cohere",
    feature = "togetherai",
    feature = "deepinfra",
    feature = "ollama",
    feature = "deepseek",
    feature = "xai",
    feature = "groq",
    feature = "minimaxi",
    feature = "bedrock"
))]
async fn build_default_client_with_capabilities(
    factory: &std::sync::Arc<dyn crate::registry::entry::ProviderFactory>,
    model_id: &str,
    ctx: &crate::registry::entry::BuildContext,
) -> Result<std::sync::Arc<dyn crate::client::LlmClient>, LlmError> {
    let caps = factory.capabilities();

    if caps.supports("chat") {
        return factory.compat_language_client_with_ctx(model_id, ctx).await;
    }
    if caps.supports("rerank") {
        return factory
            .compat_reranking_client_with_ctx(model_id, ctx)
            .await;
    }
    if caps.supports("embedding") {
        return factory
            .compat_embedding_client_with_ctx(model_id, ctx)
            .await;
    }
    if caps.supports("image_generation") {
        return factory.compat_image_client_with_ctx(model_id, ctx).await;
    }
    if caps.supports("speech") {
        return factory.compat_speech_client_with_ctx(model_id, ctx).await;
    }
    if caps.supports("transcription") {
        return factory
            .compat_transcription_client_with_ctx(model_id, ctx)
            .await;
    }

    Err(LlmError::UnsupportedOperation(format!(
        "Provider '{}' does not expose a default public family entry point",
        factory.provider_id()
    )))
}

/// Build the unified Siumai provider from SiumaiBuilder
#[cfg(any(
    feature = "openai",
    feature = "azure",
    feature = "anthropic",
    feature = "google",
    feature = "google-vertex",
    feature = "cohere",
    feature = "togetherai",
    feature = "deepinfra",
    feature = "ollama",
    feature = "deepseek",
    feature = "xai",
    feature = "groq",
    feature = "minimaxi",
    feature = "bedrock"
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

    let anthropic_base_url_override = match provider_id.as_str() {
        ids::ANTHROPIC | ids::ANTHROPIC_VERTEX => builder
            .base_url
            .clone()
            .or_else(|| {
                std::env::var("ANTHROPIC_BASE_URL")
                    .ok()
                    .filter(|value| !value.trim().is_empty())
            })
            .map(|base_url| {
                crate::utils::builder_helpers::resolve_base_url(
                    Some(base_url),
                    "https://api.anthropic.com",
                )
            }),
        _ => None,
    };

    // Some routing decisions depend on base_url (Anthropic on Vertex).
    let mut effective_provider_id = provider_id.clone();
    if effective_provider_id == ids::ANTHROPIC {
        let is_vertex = anthropic_base_url_override
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
        common_params.model =
            crate::registry::helpers::builtin_provider_default_model(&effective_provider_id)?;
    }

    // Normalize model ID for OpenAI-compatible providers (handle aliases)
    // This ensures that model aliases like "chat" -> "deepseek-chat" are properly resolved
    #[cfg(any(feature = "openai", feature = "deepseek", feature = "deepinfra"))]
    {
        if super::resolver::is_openai_compatible_provider_id(&effective_provider_id) {
            let normalized_model =
                super::resolver::normalize_model_id(&effective_provider_id, &common_params.model);
            if !normalized_model.is_empty() {
                common_params.model = normalized_model;
            }
        }
    }

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

    let effective_base_url = if effective_provider_id == ids::ANTHROPIC_VERTEX {
        anthropic_base_url_override.clone()
    } else {
        builder.base_url.clone()
    };

    let base_ctx = BuildContext {
        http_client: Some(built_http_client.clone()),
        http_transport: builder.http_transport.clone(),
        http_config: Some(http_config.clone()),
        api_key: builder.api_key.clone(),
        base_url: effective_base_url,
        organization: builder.organization.clone(),
        project: builder.project.clone(),
        location: builder.location.clone(),
        tracing_config: builder.tracing_config.clone(),
        http_interceptors: interceptors.clone(),
        model_middlewares: user_model_middlewares.clone(),
        retry_options: builder.retry_options.clone(),
        common_params: Some(common_params.clone()),
        reasoning_enabled,
        reasoning_budget,
        provider_id: builder.provider_id.clone(),
        #[cfg(any(feature = "google", feature = "google-vertex"))]
        google_token_provider: builder.google_token_provider.clone(),
        #[cfg(any(feature = "google", feature = "google-vertex"))]
        gemini_token_provider: builder.google_token_provider.clone(),
    };

    #[cfg(feature = "azure")]
    let factory: Arc<dyn ProviderFactory> = if ids::is_azure_family(&effective_provider_id) {
        crate::registry::helpers::azure_provider_factory_with_options(
            &effective_provider_id,
            builder.azure_url_config.clone(),
            builder.azure_provider_metadata_key,
        )?
    } else {
        crate::registry::helpers::builtin_provider_factory(&effective_provider_id)?
    };
    #[cfg(not(feature = "azure"))]
    let factory = crate::registry::helpers::builtin_provider_factory(&effective_provider_id)?;
    let mut ctx = base_ctx.clone();
    ctx.provider_id = Some(effective_provider_id);
    let client: Arc<dyn LlmClient> =
        build_default_client_with_capabilities(&factory, &common_params.model, &ctx).await?;

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
    feature = "cohere",
    feature = "togetherai",
    feature = "deepinfra",
    feature = "ollama",
    feature = "deepseek",
    feature = "xai",
    feature = "groq",
    feature = "minimaxi",
    feature = "bedrock"
)))]
pub async fn build(_builder: super::SiumaiBuilder) -> Result<super::Siumai, LlmError> {
    Err(LlmError::UnsupportedOperation(
        "No provider features enabled (enable at least one of: openai, azure, anthropic, google, google-vertex, cohere, togetherai, bedrock, ollama, deepseek, xai, groq, minimaxi)".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::super::resolver::infer_provider_id_from_model;

    #[test]
    fn default_client_builder_uses_explicit_compat_factory_methods() {
        let source = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("src")
                .join("provider")
                .join("build.rs"),
        )
        .unwrap();

        for family in [
            "language",
            "reranking",
            "embedding",
            "image",
            "speech",
            "transcription",
        ] {
            let compat_call = format!("compat_{family}_client_with_ctx");
            assert!(
                source.contains(&compat_call),
                "SiumaiBuilder compatibility construction should call {compat_call}"
            );

            let legacy_call = format!("factory.{family}_model_with_ctx");
            assert!(
                !source.contains(&legacy_call),
                "SiumaiBuilder compatibility construction must not call legacy {legacy_call}"
            );
        }
    }

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
