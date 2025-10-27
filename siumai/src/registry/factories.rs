//! Provider factory implementations
//!
//! Each provider implements the ProviderFactory trait to create clients.

use std::sync::Arc;

use crate::builder::LlmBuilder;
use crate::client::LlmClient;
use crate::error::LlmError;
use crate::registry::entry::BuildContext;
use crate::registry::entry::ProviderFactory;

/// OpenAI provider factory
#[cfg(feature = "openai")]
pub struct OpenAIProviderFactory;

#[cfg(feature = "openai")]
#[async_trait::async_trait]
impl ProviderFactory for OpenAIProviderFactory {
    async fn language_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        let client = crate::quick_openai_with_model(model_id).await?;
        Ok(Arc::new(client))
    }

    async fn language_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let mut client = crate::quick_openai_with_model(model_id).await?;
        if let Some(opts) = &ctx.retry_options {
            client.set_retry_options(Some(opts.clone()));
        }
        if !ctx.http_interceptors.is_empty() {
            client = client.with_http_interceptors(ctx.http_interceptors.clone());
        }
        Ok(Arc::new(client))
    }

    async fn embedding_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let mut client = crate::quick_openai_with_model(model_id).await?;
        if let Some(opts) = &ctx.retry_options {
            client.set_retry_options(Some(opts.clone()));
        }
        if !ctx.http_interceptors.is_empty() {
            client = client.with_http_interceptors(ctx.http_interceptors.clone());
        }
        Ok(Arc::new(client))
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
        std::borrow::Cow::Borrowed("openai")
    }
}

/// Anthropic provider factory
#[cfg(feature = "anthropic")]
pub struct AnthropicProviderFactory;

#[cfg(feature = "anthropic")]
#[async_trait::async_trait]
impl ProviderFactory for AnthropicProviderFactory {
    async fn language_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        let client = crate::quick_anthropic_with_model(model_id).await?;
        Ok(Arc::new(client))
    }

    async fn language_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let mut client = crate::quick_anthropic_with_model(model_id).await?;
        if let Some(opts) = &ctx.retry_options {
            client.set_retry_options(Some(opts.clone()));
        }
        if !ctx.http_interceptors.is_empty() {
            client = client.with_http_interceptors(ctx.http_interceptors.clone());
        }
        Ok(Arc::new(client))
    }

    async fn embedding_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let mut client = crate::quick_anthropic_with_model(model_id).await?;
        if let Some(opts) = &ctx.retry_options {
            client.set_retry_options(Some(opts.clone()));
        }
        if !ctx.http_interceptors.is_empty() {
            client = client.with_http_interceptors(ctx.http_interceptors.clone());
        }
        Ok(Arc::new(client))
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

/// Gemini provider factory
#[cfg(feature = "google")]
pub struct GeminiProviderFactory;

#[cfg(feature = "google")]
#[async_trait::async_trait]
impl ProviderFactory for GeminiProviderFactory {
    async fn language_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        let client = crate::quick_gemini_with_model(model_id).await?;
        Ok(Arc::new(client))
    }

    async fn language_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let mut client = crate::quick_gemini_with_model(model_id).await?;
        if let Some(opts) = &ctx.retry_options {
            client.set_retry_options(Some(opts.clone()));
        }
        if !ctx.http_interceptors.is_empty() {
            client = client.with_http_interceptors(ctx.http_interceptors.clone());
        }
        Ok(Arc::new(client))
    }

    async fn embedding_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let mut client = crate::quick_gemini_with_model(model_id).await?;
        if let Some(opts) = &ctx.retry_options {
            client.set_retry_options(Some(opts.clone()));
        }
        if !ctx.http_interceptors.is_empty() {
            client = client.with_http_interceptors(ctx.http_interceptors.clone());
        }
        Ok(Arc::new(client))
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
    async fn language_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        let client = crate::quick_groq_with_model(model_id).await?;
        Ok(Arc::new(client))
    }

    async fn language_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let mut client = crate::quick_groq_with_model(model_id).await?;
        if let Some(opts) = &ctx.retry_options {
            client.set_retry_options(Some(opts.clone()));
        }
        if !ctx.http_interceptors.is_empty() {
            client = client.with_http_interceptors(ctx.http_interceptors.clone());
        }
        Ok(Arc::new(client))
    }

    async fn embedding_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let mut client = crate::quick_groq_with_model(model_id).await?;
        if let Some(opts) = &ctx.retry_options {
            client.set_retry_options(Some(opts.clone()));
        }
        if !ctx.http_interceptors.is_empty() {
            client = client.with_http_interceptors(ctx.http_interceptors.clone());
        }
        Ok(Arc::new(client))
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
    async fn language_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        let client = crate::prelude::quick_xai_with_model(model_id).await?;
        Ok(Arc::new(client))
    }

    async fn language_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let mut client = crate::prelude::quick_xai_with_model(model_id).await?;
        if let Some(opts) = &ctx.retry_options {
            client.set_retry_options(Some(opts.clone()));
        }
        if !ctx.http_interceptors.is_empty() {
            client = client.with_http_interceptors(ctx.http_interceptors.clone());
        }
        Ok(Arc::new(client))
    }

    async fn embedding_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let mut client = crate::prelude::quick_xai_with_model(model_id).await?;
        if let Some(opts) = &ctx.retry_options {
            client.set_retry_options(Some(opts.clone()));
        }
        if !ctx.http_interceptors.is_empty() {
            client = client.with_http_interceptors(ctx.http_interceptors.clone());
        }
        Ok(Arc::new(client))
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
    async fn language_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        let client = LlmBuilder::new().ollama().model(model_id).build().await?;
        Ok(Arc::new(client))
    }

    async fn language_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let mut client = LlmBuilder::new().ollama().model(model_id).build().await?;
        if let Some(opts) = &ctx.retry_options {
            client.set_retry_options(Some(opts.clone()));
        }
        if !ctx.http_interceptors.is_empty() {
            client = client.with_http_interceptors(ctx.http_interceptors.clone());
        }
        Ok(Arc::new(client))
    }

    async fn embedding_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let mut client = LlmBuilder::new().ollama().model(model_id).build().await?;
        if let Some(opts) = &ctx.retry_options {
            client.set_retry_options(Some(opts.clone()));
        }
        if !ctx.http_interceptors.is_empty() {
            client = client.with_http_interceptors(ctx.http_interceptors.clone());
        }
        Ok(Arc::new(client))
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

/// OpenRouter provider factory (OpenAI-compatible)
#[cfg(feature = "openai")]
pub struct OpenRouterProviderFactory;

#[cfg(feature = "openai")]
#[async_trait::async_trait]
impl ProviderFactory for OpenRouterProviderFactory {
    async fn language_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        let client = LlmBuilder::new()
            .openrouter()
            .model(model_id)
            .build()
            .await?;
        Ok(Arc::new(client))
    }

    async fn language_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let mut client = LlmBuilder::new()
            .openrouter()
            .model(model_id)
            .build()
            .await?;
        if let Some(opts) = &ctx.retry_options {
            client.set_retry_options(Some(opts.clone()));
        }
        if !ctx.http_interceptors.is_empty() {
            client = client.with_http_interceptors(ctx.http_interceptors.clone());
        }
        Ok(Arc::new(client))
    }

    async fn embedding_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let mut client = LlmBuilder::new()
            .openrouter()
            .model(model_id)
            .build()
            .await?;
        if let Some(opts) = &ctx.retry_options {
            client.set_retry_options(Some(opts.clone()));
        }
        if !ctx.http_interceptors.is_empty() {
            client = client.with_http_interceptors(ctx.http_interceptors.clone());
        }
        Ok(Arc::new(client))
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
    async fn language_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        let client = LlmBuilder::new().deepseek().model(model_id).build().await?;
        Ok(Arc::new(client))
    }

    async fn language_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let mut client = LlmBuilder::new().deepseek().model(model_id).build().await?;
        if let Some(opts) = &ctx.retry_options {
            client.set_retry_options(Some(opts.clone()));
        }
        if !ctx.http_interceptors.is_empty() {
            client = client.with_http_interceptors(ctx.http_interceptors.clone());
        }
        Ok(Arc::new(client))
    }

    async fn embedding_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let mut client = LlmBuilder::new().deepseek().model(model_id).build().await?;
        if let Some(opts) = &ctx.retry_options {
            client.set_retry_options(Some(opts.clone()));
        }
        if !ctx.http_interceptors.is_empty() {
            client = client.with_http_interceptors(ctx.http_interceptors.clone());
        }
        Ok(Arc::new(client))
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
    async fn language_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        let mut builder = crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(
            LlmBuilder::new(),
            &self.provider_id,
        );

        let env_key = format!("{}_API_KEY", self.provider_id.to_uppercase());
        let api_key = std::env::var(&env_key).map_err(|_| {
            LlmError::ConfigurationError(format!(
                "Missing {} for OpenAI-compatible provider {}",
                env_key, self.provider_id
            ))
        })?;

        builder = builder.api_key(api_key).model(model_id);
        let client = builder.build().await?;
        Ok(Arc::new(client))
    }

    async fn language_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let mut builder = crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(
            LlmBuilder::new(),
            &self.provider_id,
        );

        let env_key = format!("{}_API_KEY", self.provider_id.to_uppercase());
        let api_key = std::env::var(&env_key).map_err(|_| {
            LlmError::ConfigurationError(format!(
                "Missing {} for OpenAI-compatible provider {}",
                env_key, self.provider_id
            ))
        })?;

        builder = builder.api_key(api_key).model(model_id);
        let mut client = builder.build().await?;
        if let Some(opts) = &ctx.retry_options {
            client.set_retry_options(Some(opts.clone()));
        }
        if !ctx.http_interceptors.is_empty() {
            client = client.with_http_interceptors(ctx.http_interceptors.clone());
        }
        Ok(Arc::new(client))
    }

    async fn embedding_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        let mut builder = crate::providers::openai_compatible::OpenAiCompatibleBuilder::new(
            LlmBuilder::new(),
            &self.provider_id,
        );

        let env_key = format!("{}_API_KEY", self.provider_id.to_uppercase());
        let api_key = std::env::var(&env_key).map_err(|_| {
            LlmError::ConfigurationError(format!(
                "Missing {} for OpenAI-compatible provider {}",
                env_key, self.provider_id
            ))
        })?;

        builder = builder.api_key(api_key).model(model_id);
        let mut client = builder.build().await?;
        if let Some(opts) = &ctx.retry_options {
            client.set_retry_options(Some(opts.clone()));
        }
        if !ctx.http_interceptors.is_empty() {
            client = client.with_http_interceptors(ctx.http_interceptors.clone());
        }
        Ok(Arc::new(client))
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
