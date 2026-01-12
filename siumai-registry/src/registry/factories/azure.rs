//! Provider factory implementations.

use super::*;

/// Azure OpenAI provider factory (Responses API by default).
///
/// Mirrors Vercel AI SDK's `@ai-sdk/azure` provider granularity.
#[cfg(feature = "azure")]
#[derive(Debug, Clone, Copy)]
pub struct AzureOpenAiProviderFactory {
    chat_mode: siumai_provider_azure::providers::azure_openai::AzureChatMode,
}

#[cfg(feature = "azure")]
impl Default for AzureOpenAiProviderFactory {
    fn default() -> Self {
        Self {
            chat_mode: siumai_provider_azure::providers::azure_openai::AzureChatMode::Responses,
        }
    }
}

#[cfg(feature = "azure")]
impl AzureOpenAiProviderFactory {
    pub const fn new(
        chat_mode: siumai_provider_azure::providers::azure_openai::AzureChatMode,
    ) -> Self {
        Self { chat_mode }
    }

    fn default_base_url_from_env() -> Result<String, LlmError> {
        let resource = std::env::var("AZURE_RESOURCE_NAME").map_err(|_| {
            LlmError::ConfigurationError(
                "Missing AZURE_RESOURCE_NAME or explicit base_url for Azure OpenAI".to_string(),
            )
        })?;
        Ok(format!(
            "https://{}.openai.azure.com/openai",
            resource.trim()
        ))
    }
}

#[cfg(feature = "azure")]
#[async_trait::async_trait]
impl ProviderFactory for AzureOpenAiProviderFactory {
    fn capabilities(&self) -> ProviderCapabilities {
        let meta = crate::native_provider_metadata::native_providers_metadata();
        meta.into_iter()
            .find(|m| m.id == "azure")
            .map(|m| m.capabilities)
            .unwrap_or_else(ProviderCapabilities::new)
    }

    async fn language_model(&self, model_id: &str) -> Result<Arc<dyn LlmClient>, LlmError> {
        let ctx = BuildContext::default();
        self.language_model_with_ctx(model_id, &ctx).await
    }

    async fn language_model_with_ctx(
        &self,
        model_id: &str,
        ctx: &BuildContext,
    ) -> Result<Arc<dyn LlmClient>, LlmError> {
        // Resolve HTTP configuration and client.
        let http_config = ctx.http_config.clone().unwrap_or_default();
        let http_client = if let Some(client) = &ctx.http_client {
            client.clone()
        } else {
            build_http_client_from_config(&http_config)?
        };

        // Resolve API key: context override 鈫?environment variable.
        let api_key = if let Some(key) = &ctx.api_key {
            key.clone()
        } else {
            crate::utils::builder_helpers::get_api_key_with_env(None, "azure")?
        };

        // Resolve base URL: context override 鈫?resourceName env.
        let base_url = if let Some(custom) = ctx.base_url.clone() {
            custom
        } else {
            Self::default_base_url_from_env()?
        };

        // Resolve common parameters (model, temperature, max_tokens, etc.).
        let common_params = crate::utils::builder_helpers::resolve_common_params(
            ctx.common_params.clone(),
            model_id,
        );
        if common_params.model.trim().is_empty() {
            return Err(LlmError::ConfigurationError(
                "Azure OpenAI requires a model (deployment id)".to_string(),
            ));
        }

        let mut cfg =
            siumai_provider_azure::providers::azure_openai::AzureOpenAiConfig::new(api_key)
                .with_base_url(base_url)
                .with_http_config(http_config.clone())
                .with_url_config(
                    siumai_provider_azure::providers::azure_openai::AzureUrlConfig::default(),
                )
                .with_chat_mode(self.chat_mode);
        cfg.common_params = common_params;
        if let Some(transport) = ctx.http_transport.clone() {
            cfg = cfg.with_http_transport(transport);
        }

        let client = siumai_provider_azure::providers::azure_openai::AzureOpenAiClient::new(
            cfg,
            http_client,
        )?
        .with_retry_options(ctx.retry_options.clone())
        .with_http_interceptors(ctx.http_interceptors.clone())
        .with_model_middlewares(ctx.model_middlewares.clone());

        Ok(Arc::new(client))
    }

    fn provider_id(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("azure")
    }
}
