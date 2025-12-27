//! Provider factory implementations
//!
//! Each provider implements the ProviderFactory trait to create clients.

#[cfg(any(
    test,
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "ollama",
    feature = "xai",
    feature = "groq",
    feature = "minimaxi"
))]
use std::sync::Arc;

#[allow(unused_imports)]
use siumai_providers::builder::LlmBuilder;
#[cfg(any(
    test,
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "ollama",
    feature = "xai",
    feature = "groq",
    feature = "minimaxi"
))]
use crate::client::LlmClient;
#[cfg(any(
    test,
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "ollama",
    feature = "xai",
    feature = "groq",
    feature = "minimaxi"
))]
use crate::error::LlmError;
#[allow(unused_imports)]
use crate::execution::http::client::build_http_client_from_config;
#[cfg(any(
    test,
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "ollama",
    feature = "xai",
    feature = "groq",
    feature = "minimaxi"
))]
use crate::registry::entry::ProviderFactory;
#[cfg(any(
    test,
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "ollama",
    feature = "xai",
    feature = "groq",
    feature = "minimaxi"
))]
use crate::traits::ProviderCapabilities;

#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "ollama",
    feature = "xai",
    feature = "groq",
    feature = "minimaxi"
))]
use crate::registry::entry::BuildContext;

/// OpenAI provider factory
#[cfg(feature = "openai")]
pub struct OpenAIProviderFactory;

#[cfg(feature = "openai")]
#[async_trait::async_trait]
impl ProviderFactory for OpenAIProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        let meta = siumai_providers::providers::metadata::native_providers_metadata();
        meta.into_iter()
            .find(|m| m.id == "openai")
            .map(|m| m.capabilities)
            .unwrap_or_else(ProviderCapabilities::new)
    }

    async fn language_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Delegate to the context-aware implementation with default context.
        let ctx = BuildContext::default();
        self.language_model_with_ctx(model_id, &ctx).await
    }

    async fn language_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        use crate::execution::http::client::build_http_client_from_config;

        // Resolve HTTP configuration and client (prefer provided client).
        let http_config = ctx
            .http_config
            .clone()
            .unwrap_or_default();
        let http_client = if let Some(client) = &ctx.http_client {
            client.clone()
        } else {
            build_http_client_from_config(&http_config)?
        };

        // Resolve API key: context override → environment variable.
        let api_key = if let Some(key) = &ctx.api_key {
            key.clone()
        } else {
            std::env::var("OPENAI_API_KEY").map_err(|_| {
                LlmError::ConfigurationError(
                    "Missing OPENAI_API_KEY or explicit api_key in BuildContext".to_string(),
                )
            })?
        };

        // Resolve base URL (context override → default).
        let base_url = crate::utils::builder_helpers::resolve_base_url(
            ctx.base_url.clone(),
            "https://api.openai.com/v1",
        );

        // Resolve common parameters (model, temperature, max_tokens, etc.).
        let common_params = crate::utils::builder_helpers::resolve_common_params(
            ctx.common_params.clone(),
            model_id,
        );

        // Delegate to the shared OpenAI client builder used by SiumaiBuilder.
        crate::registry::factory::build_openai_client(
            api_key,
            base_url,
            http_client,
            common_params,
            http_config,
            None,
            ctx.organization.clone(),
            ctx.project.clone(),
            ctx.tracing_config.clone(),
            ctx.retry_options.clone(),
            ctx.http_interceptors.clone(),
            ctx.model_middlewares.clone(),
        )
        .await
    }

    async fn embedding_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        // For OpenAI, embeddings are served by the same client as chat.
        self.language_model_with_ctx(model_id, ctx).await
    }

    async fn image_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Image generation is also handled by the unified OpenAI client.
        self.language_model_with_ctx(model_id, ctx).await
    }

    async fn speech_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    async fn transcription_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("openai")
    }
}

/// Anthropic provider factory
#[cfg(feature = "anthropic")]
pub struct AnthropicProviderFactory;

#[cfg(feature = "anthropic")]
#[async_trait::async_trait]
impl ProviderFactory for AnthropicProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        let meta = siumai_providers::providers::metadata::native_providers_metadata();
        meta.into_iter()
            .find(|m| m.id == "anthropic")
            .map(|m| m.capabilities)
            .unwrap_or_else(ProviderCapabilities::new)
    }

    async fn language_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Delegate to the context-aware implementation with default context.
        let ctx = BuildContext::default();
        self.language_model_with_ctx(model_id, &ctx).await
    }

    async fn language_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Resolve HTTP configuration and client.
        let http_config = ctx
            .http_config
            .clone()
            .unwrap_or_default();
        let http_client = if let Some(client) = &ctx.http_client {
            client.clone()
        } else {
            build_http_client_from_config(&http_config)?
        };

        // Resolve API key: context override → environment variable.
        let api_key = if let Some(key) = &ctx.api_key {
            key.clone()
        } else {
            std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
                LlmError::ConfigurationError(
                    "Missing ANTHROPIC_API_KEY or explicit api_key in BuildContext".to_string(),
                )
            })?
        };

        // Resolve base URL (context override → default).
        let base_url = crate::utils::builder_helpers::resolve_base_url(
            ctx.base_url.clone(),
            "https://api.anthropic.com",
        );

        // Resolve common parameters (model, temperature, max_tokens, etc.).
        let common_params = crate::utils::builder_helpers::resolve_common_params(
            ctx.common_params.clone(),
            model_id,
        );

        crate::registry::factory::build_anthropic_client(
            api_key,
            base_url,
            http_client,
            common_params,
            http_config,
            None,
            ctx.tracing_config.clone(),
            ctx.retry_options.clone(),
            ctx.http_interceptors.clone(),
            ctx.model_middlewares.clone(),
        )
        .await
    }

    async fn embedding_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Anthropic uses a unified client for chat/embeddings/images.
        self.language_model_with_ctx(model_id, ctx).await
    }

    async fn image_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    async fn speech_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    async fn transcription_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("anthropic")
    }
}

/// Anthropic on Vertex AI provider factory
///
/// This factory builds `anthropic-vertex` clients that communicate with
/// Anthropic models hosted on Vertex AI. Authentication is handled via
/// `Authorization: Bearer` headers configured on the HTTP client.
#[cfg(feature = "anthropic")]
pub struct AnthropicVertexProviderFactory;

#[cfg(feature = "anthropic")]
#[async_trait::async_trait]
impl ProviderFactory for AnthropicVertexProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        let meta = siumai_providers::providers::metadata::native_providers_metadata();
        meta.into_iter()
            .find(|m| m.id == "anthropic-vertex")
            .map(|m| m.capabilities)
            .unwrap_or_else(ProviderCapabilities::new)
    }

    async fn language_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Delegate to the context-aware implementation with default context.
        let ctx = BuildContext::default();
        self.language_model_with_ctx(model_id, &ctx).await
    }

    async fn language_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Resolve HTTP configuration and client.
        let http_config = ctx
            .http_config
            .clone()
            .unwrap_or_default();
        let http_client = if let Some(client) = &ctx.http_client {
            client.clone()
        } else {
            build_http_client_from_config(&http_config)?
        };

        // Resolve common parameters (model id, etc.).
        let common_params = crate::utils::builder_helpers::resolve_common_params(
            ctx.common_params.clone(),
            model_id,
        );

        // For Vertex AI, base URL must point at the Vertex endpoint.
        // We do not synthesize a default here; callers should provide
        // a concrete base_url via BuildContext.
        let base_url = ctx.base_url.clone().unwrap_or_default();

        crate::registry::factory::build_anthropic_vertex_client(
            base_url,
            http_client,
            common_params,
            http_config,
            ctx.tracing_config.clone(),
            ctx.retry_options.clone(),
            ctx.http_interceptors.clone(),
            ctx.model_middlewares.clone(),
        )
        .await
    }

    async fn embedding_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Anthropic Vertex client is unified across capabilities.
        self.language_model_with_ctx(model_id, ctx).await
    }

    async fn image_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    async fn speech_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    async fn transcription_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("anthropic-vertex")
    }
}

/// Gemini provider factory
#[cfg(feature = "google")]
pub struct GeminiProviderFactory;

#[cfg(feature = "google")]
#[async_trait::async_trait]
impl ProviderFactory for GeminiProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        let meta = siumai_providers::providers::metadata::native_providers_metadata();
        meta.into_iter()
            .find(|m| m.id == "gemini")
            .map(|m| m.capabilities)
            .unwrap_or_else(ProviderCapabilities::new)
    }

    async fn language_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Delegate to the context-aware implementation with default context.
        let ctx = BuildContext::default();
        self.language_model_with_ctx(model_id, &ctx).await
    }

    async fn language_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Resolve HTTP configuration and client.
        let http_config = ctx
            .http_config
            .clone()
            .unwrap_or_default();
        let http_client = if let Some(client) = &ctx.http_client {
            client.clone()
        } else {
            build_http_client_from_config(&http_config)?
        };

        // Detect whether an explicit Authorization header or token provider is present.
        let has_auth_header = http_config
            .headers
            .keys()
            .any(|k| k.eq_ignore_ascii_case("authorization"));
        let has_token_provider = ctx
            .gemini_token_provider
            .as_ref()
            .map(|_| true)
            .unwrap_or(false);
        let requires_api_key = !(has_auth_header || has_token_provider);

        // Resolve API key: context override → GEMINI_API_KEY (when required) → empty for token-based auth.
        let api_key = if let Some(key) = &ctx.api_key {
            key.clone()
        } else if requires_api_key {
            std::env::var("GEMINI_API_KEY").map_err(|_| {
                LlmError::ConfigurationError(
                    "Missing GEMINI_API_KEY or explicit api_key in BuildContext (or provide Authorization header / token provider)"
                        .to_string(),
                )
            })?
        } else {
            String::new()
        };

        // Resolve base URL (context override → default).
        let base_url = crate::utils::builder_helpers::resolve_base_url(
            ctx.base_url.clone(),
            "https://generativelanguage.googleapis.com/v1beta",
        );

        // Resolve common parameters.
        let common_params = crate::utils::builder_helpers::resolve_common_params(
            ctx.common_params.clone(),
            model_id,
        );

        crate::registry::factory::build_gemini_client(
            api_key,
            base_url,
            http_client,
            common_params,
            http_config,
            None,
            ctx.gemini_token_provider.clone(),
            ctx.tracing_config.clone(),
            ctx.retry_options.clone(),
            ctx.http_interceptors.clone(),
            ctx.model_middlewares.clone(),
        )
        .await
    }

    async fn embedding_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Gemini uses a unified client for chat/embeddings/images.
        self.language_model_with_ctx(model_id, ctx).await
    }

    async fn image_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    async fn speech_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    async fn transcription_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("gemini")
    }
}

/// Groq provider factory
#[cfg(feature = "groq")]
pub struct GroqProviderFactory;

#[cfg(feature = "groq")]
#[async_trait::async_trait]
impl ProviderFactory for GroqProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        let meta = siumai_providers::providers::metadata::native_providers_metadata();
        meta.into_iter()
            .find(|m| m.id == "groq")
            .map(|m| m.capabilities)
            .unwrap_or_else(ProviderCapabilities::new)
    }

    async fn language_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Delegate to the context-aware implementation with default context.
        let ctx = BuildContext::default();
        self.language_model_with_ctx(model_id, &ctx).await
    }

    async fn language_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Resolve HTTP configuration and client.
        let http_config = ctx
            .http_config
            .clone()
            .unwrap_or_default();
        let http_client = if let Some(client) = &ctx.http_client {
            client.clone()
        } else {
            build_http_client_from_config(&http_config)?
        };

        // Resolve API key for Groq (OpenAI-compatible).
        let api_key = if let Some(key) = &ctx.api_key {
            key.clone()
        } else {
            std::env::var("GROQ_API_KEY").map_err(|_| {
                LlmError::ConfigurationError(
                    "Missing GROQ_API_KEY or explicit api_key in BuildContext".to_string(),
                )
            })?
        };

        // Resolve common parameters.
        let common_params = crate::utils::builder_helpers::resolve_common_params(
            ctx.common_params.clone(),
            model_id,
        );

        crate::registry::factory::build_openai_compatible_client(
            "groq".to_string(),
            api_key,
            ctx.base_url.clone(),
            http_client,
            common_params,
            http_config,
            None,
            ctx.tracing_config.clone(),
            ctx.retry_options.clone(),
            ctx.http_interceptors.clone(),
            ctx.model_middlewares.clone(),
        )
        .await
    }

    async fn embedding_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Groq client is OpenAI-compatible and unified across capabilities.
        self.language_model_with_ctx(model_id, ctx).await
    }

    async fn image_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    async fn speech_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    async fn transcription_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("groq")
    }
}

/// xAI provider factory
#[cfg(feature = "xai")]
pub struct XAIProviderFactory;

#[cfg(feature = "xai")]
#[async_trait::async_trait]
impl ProviderFactory for XAIProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        let meta = siumai_providers::providers::metadata::native_providers_metadata();
        meta.into_iter()
            .find(|m| m.id == "xai")
            .map(|m| m.capabilities)
            .unwrap_or_else(ProviderCapabilities::new)
    }

    async fn language_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Delegate to the context-aware implementation with default context.
        let ctx = BuildContext::default();
        self.language_model_with_ctx(model_id, &ctx).await
    }

    async fn language_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        use siumai_providers::providers::xai::{XaiClient, XaiConfig};

        // Resolve HTTP configuration and client.
        let http_config = ctx
            .http_config
            .clone()
            .unwrap_or_default();
        let http_client = if let Some(client) = &ctx.http_client {
            client.clone()
        } else {
            build_http_client_from_config(&http_config)?
        };

        // Resolve API key: context override → XAI_API_KEY.
        let api_key = if let Some(key) = &ctx.api_key {
            key.clone()
        } else {
            std::env::var("XAI_API_KEY").map_err(|_| {
                LlmError::ConfigurationError(
                    "Missing XAI_API_KEY or explicit api_key in BuildContext".to_string(),
                )
            })?
        };

        // Resolve base URL (context override → default).
        let base_url = crate::utils::builder_helpers::resolve_base_url(
            ctx.base_url.clone(),
            "https://api.x.ai/v1",
        );

        // Resolve common parameters for model selection.
        let common_params = crate::utils::builder_helpers::resolve_common_params(
            ctx.common_params.clone(),
            model_id,
        );

        // Build XAI config and client.
        let xai_cfg = XaiConfig::new(api_key)
            .with_base_url(base_url)
            .with_model(common_params.model.clone());

        let mut client = XaiClient::with_http_client(xai_cfg, http_client).await?;

        // Apply tracing configuration if provided.
        if let Some(tc) = ctx.tracing_config.clone() {
            client.set_tracing_config(Some(tc));
        }

        // Install HTTP interceptors.
        if !ctx.http_interceptors.is_empty() {
            client = client.with_http_interceptors(ctx.http_interceptors.clone());
        }

        // Apply retry options when present.
        if let Some(opts) = &ctx.retry_options {
            client.set_retry_options(Some(opts.clone()));
        }

        // Auto + user middlewares.
        let mut auto_mws =
            crate::execution::middleware::build_auto_middlewares_vec("xai", &common_params.model);
        auto_mws.extend(ctx.model_middlewares.clone());
        if !auto_mws.is_empty() {
            client = client.with_model_middlewares(auto_mws);
        }

        Ok(Arc::new(client))
    }

    async fn embedding_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        // xAI currently exposes chat/models; reuse chat client path.
        self.language_model_with_ctx(model_id, ctx).await
    }

    async fn image_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    async fn speech_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    async fn transcription_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("xai")
    }
}

/// Ollama provider factory
#[cfg(feature = "ollama")]
pub struct OllamaProviderFactory;

#[cfg(feature = "ollama")]
#[async_trait::async_trait]
impl ProviderFactory for OllamaProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        let meta = siumai_providers::providers::metadata::native_providers_metadata();
        meta.into_iter()
            .find(|m| m.id == "ollama")
            .map(|m| m.capabilities)
            .unwrap_or_else(ProviderCapabilities::new)
    }

    async fn language_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Delegate to the context-aware implementation with default context.
        let ctx = BuildContext::default();
        self.language_model_with_ctx(model_id, &ctx).await
    }

    async fn language_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Resolve HTTP configuration and client.
        let http_config = ctx
            .http_config
            .clone()
            .unwrap_or_default();
        let http_client = if let Some(client) = &ctx.http_client {
            client.clone()
        } else {
            build_http_client_from_config(&http_config)?
        };

        // Resolve base URL (context override → default).
        let base_url = crate::utils::builder_helpers::resolve_base_url(
            ctx.base_url.clone(),
            "http://localhost:11434",
        );

        // Resolve common parameters.
        let common_params = crate::utils::builder_helpers::resolve_common_params(
            ctx.common_params.clone(),
            model_id,
        );

        crate::registry::factory::build_ollama_client(
            base_url,
            http_client,
            common_params,
            http_config,
            None,
            ctx.tracing_config.clone(),
            ctx.retry_options.clone(),
            ctx.http_interceptors.clone(),
            ctx.model_middlewares.clone(),
        )
        .await
    }

    async fn embedding_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Ollama client is unified across capabilities.
        self.language_model_with_ctx(model_id, ctx).await
    }

    async fn image_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    async fn speech_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    async fn transcription_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("ollama")
    }
}

/// MiniMaxi provider factory
#[cfg(feature = "minimaxi")]
pub struct MiniMaxiProviderFactory;

#[cfg(feature = "minimaxi")]
#[async_trait::async_trait]
impl ProviderFactory for MiniMaxiProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        let meta = siumai_providers::providers::metadata::native_providers_metadata();
        meta.into_iter()
            .find(|m| m.id == "minimaxi")
            .map(|m| m.capabilities)
            .unwrap_or_else(ProviderCapabilities::new)
    }

    async fn language_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Delegate to the context-aware implementation with default context.
        let ctx = BuildContext::default();
        self.language_model_with_ctx(model_id, &ctx).await
    }

    async fn language_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Resolve HTTP configuration and client.
        let http_config = ctx
            .http_config
            .clone()
            .unwrap_or_default();
        let http_client = if let Some(client) = &ctx.http_client {
            client.clone()
        } else {
            build_http_client_from_config(&http_config)?
        };

        // Resolve API key: context override → MINIMAXI_API_KEY.
        let api_key = if let Some(key) = &ctx.api_key {
            key.clone()
        } else {
            std::env::var("MINIMAXI_API_KEY").map_err(|_| {
                LlmError::ConfigurationError(
                    "Missing MINIMAXI_API_KEY or explicit api_key in BuildContext".to_string(),
                )
            })?
        };

        // Resolve base URL (context override → config default).
        let base_url = crate::utils::builder_helpers::resolve_base_url(
            ctx.base_url.clone(),
            siumai_providers::providers::minimaxi::config::MinimaxiConfig::DEFAULT_BASE_URL,
        );

        // Resolve common parameters.
        let common_params = crate::utils::builder_helpers::resolve_common_params(
            ctx.common_params.clone(),
            model_id,
        );

        crate::registry::factory::build_minimaxi_client(
            api_key,
            base_url,
            http_client,
            common_params,
            http_config,
            ctx.tracing_config.clone(),
            ctx.retry_options.clone(),
            ctx.http_interceptors.clone(),
            ctx.model_middlewares.clone(),
        )
        .await
    }

    async fn embedding_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        // MiniMaxi client is unified across capabilities.
        self.language_model_with_ctx(model_id, ctx).await
    }

    async fn image_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    async fn speech_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    async fn transcription_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("minimaxi")
    }
}

/// OpenRouter provider factory (OpenAI-compatible)
#[cfg(feature = "openai")]
pub struct OpenRouterProviderFactory;

#[cfg(feature = "openai")]
#[async_trait::async_trait]
impl ProviderFactory for OpenRouterProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_vision()
            .with_embedding()
            .with_rerank()
            .with_image_generation()
    }

    async fn language_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Delegate to the context-aware implementation with default context.
        let ctx = BuildContext::default();
        self.language_model_with_ctx(model_id, &ctx).await
    }

    async fn language_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Resolve HTTP configuration and client.
        let http_config = ctx
            .http_config
            .clone()
            .unwrap_or_default();
        let http_client = if let Some(client) = &ctx.http_client {
            client.clone()
        } else {
            build_http_client_from_config(&http_config)?
        };

        // Resolve API key using shared helper (supports context override + env).
        let api_key =
            crate::utils::builder_helpers::get_api_key_with_env(ctx.api_key.clone(), "openrouter")?;

        // Resolve common parameters.
        let common_params = crate::utils::builder_helpers::resolve_common_params(
            ctx.common_params.clone(),
            model_id,
        );

        crate::registry::factory::build_openai_compatible_client(
            "openrouter".to_string(),
            api_key,
            ctx.base_url.clone(),
            http_client,
            common_params,
            http_config,
            None,
            ctx.tracing_config.clone(),
            ctx.retry_options.clone(),
            ctx.http_interceptors.clone(),
            ctx.model_middlewares.clone(),
        )
        .await
    }

    async fn embedding_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        // OpenRouter client is OpenAI-compatible and unified.
        self.language_model_with_ctx(model_id, ctx).await
    }

    async fn image_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    async fn speech_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    async fn transcription_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("openrouter")
    }
}

/// DeepSeek provider factory (OpenAI-compatible)
#[cfg(feature = "openai")]
pub struct DeepSeekProviderFactory;

#[cfg(feature = "openai")]
#[async_trait::async_trait]
impl ProviderFactory for DeepSeekProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
            .with_vision()
            .with_custom_feature("thinking", true)
    }

    async fn language_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Delegate to the context-aware implementation with default context.
        let ctx = BuildContext::default();
        self.language_model_with_ctx(model_id, &ctx).await
    }

    async fn language_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Resolve HTTP configuration and client.
        let http_config = ctx
            .http_config
            .clone()
            .unwrap_or_default();
        let http_client = if let Some(client) = &ctx.http_client {
            client.clone()
        } else {
            build_http_client_from_config(&http_config)?
        };

        // Resolve API key using shared helper.
        let api_key =
            crate::utils::builder_helpers::get_api_key_with_env(ctx.api_key.clone(), "deepseek")?;

        // Resolve common parameters.
        let common_params = crate::utils::builder_helpers::resolve_common_params(
            ctx.common_params.clone(),
            model_id,
        );

        crate::registry::factory::build_openai_compatible_client(
            "deepseek".to_string(),
            api_key,
            ctx.base_url.clone(),
            http_client,
            common_params,
            http_config,
            None,
            ctx.tracing_config.clone(),
            ctx.retry_options.clone(),
            ctx.http_interceptors.clone(),
            ctx.model_middlewares.clone(),
        )
        .await
    }

    async fn embedding_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        // DeepSeek client is OpenAI-compatible and unified.
        self.language_model_with_ctx(model_id, ctx).await
    }

    async fn image_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    async fn speech_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    async fn transcription_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("deepseek")
    }
}

/// Generic OpenAI-compatible provider factory
#[cfg(feature = "openai")]
pub struct OpenAICompatibleProviderFactory {
    provider_id: String,
}

#[cfg(feature = "openai")]
impl OpenAICompatibleProviderFactory {
    pub fn new(provider_id: String) -> Self {
        Self { provider_id }
    }
}

#[cfg(feature = "openai")]
#[async_trait::async_trait]
impl ProviderFactory for OpenAICompatibleProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        let mut caps = ProviderCapabilities::new().with_chat().with_streaming();
        let Some(cfg) =
            siumai_providers::providers::openai_compatible::config::get_provider_config(
                &self.provider_id,
            )
        else {
            return caps;
        };

        for c in cfg.capabilities {
            match c.as_str() {
                "tools" => {
                    caps = caps.with_tools();
                }
                "vision" => {
                    caps = caps.with_vision();
                }
                "embedding" => {
                    caps = caps.with_embedding();
                }
                "rerank" => {
                    caps = caps.with_rerank();
                }
                "reasoning" => {
                    caps = caps.with_custom_feature("thinking", true);
                }
                _ => {}
            }
        }
        caps
    }

    async fn language_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Delegate to the context-aware implementation with default context.
        let ctx = BuildContext::default();
        self.language_model_with_ctx(model_id, &ctx).await
    }

    async fn language_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Resolve HTTP configuration and client.
        let http_config = ctx
            .http_config
            .clone()
            .unwrap_or_default();
        let http_client = if let Some(client) = &ctx.http_client {
            client.clone()
        } else {
            build_http_client_from_config(&http_config)?
        };

        // Resolve API key using shared helper (context override + env).
        let api_key = crate::utils::builder_helpers::get_api_key_with_env(
            ctx.api_key.clone(),
            &self.provider_id,
        )?;

        // Resolve common parameters.
        let common_params = crate::utils::builder_helpers::resolve_common_params(
            ctx.common_params.clone(),
            model_id,
        );

        crate::registry::factory::build_openai_compatible_client(
            self.provider_id.clone(),
            api_key,
            ctx.base_url.clone(),
            http_client,
            common_params,
            http_config,
            None,
            ctx.tracing_config.clone(),
            ctx.retry_options.clone(),
            ctx.http_interceptors.clone(),
            ctx.model_middlewares.clone(),
        )
        .await
    }

    async fn embedding_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Generic OpenAI-compatible client is unified; reuse chat path.
        self.language_model_with_ctx(model_id, ctx).await
    }

    async fn image_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    async fn speech_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    async fn transcription_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        self.embedding_model_with_ctx(model_id, ctx).await
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("openai-compatible")
    }
}

/// Test provider factory (for testing)
#[cfg(test)]
pub struct TestProviderFactory;

#[cfg(test)]
#[async_trait::async_trait]
impl ProviderFactory for TestProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_chat()
    }

    async fn language_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        use crate::registry::entry::TEST_BUILD_COUNT;
        TEST_BUILD_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(Arc::new(crate::registry::entry::TestProvClient))
    }

    async fn embedding_model(&self, _model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        Ok(Arc::new(crate::registry::entry::TestProvEmbedClient))
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("testprov")
    }
}
