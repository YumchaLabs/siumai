//! OpenAI Chat Model Implementation

use crate::executors::chat::HttpChatExecutor;
use crate::middleware::language_model::LanguageModelMiddleware;
use crate::provider_core::{ProviderContext, ProviderSpec};
use crate::provider_model::ChatModel;
use crate::retry_api::RetryOptions;
use crate::standards::openai::chat::OpenAiChatAdapter;
use crate::types::ChatRequest;
use crate::utils::http_interceptor::HttpInterceptor;
use secrecy::ExposeSecret;
use std::sync::Arc;

use super::super::config::OpenAiConfig;
use super::super::provider_impl::ModelConfig;
use super::super::spec::OpenAiSpec;

/// OpenAI Chat Model
///
/// Encapsulates configuration for OpenAI chat endpoint and creates HttpChatExecutor.
///
/// ## Example
///
/// ```rust,ignore
/// use siumai::providers::openai::models::OpenAiChatModel;
/// use siumai::provider_model::ChatModel;
///
/// let config = OpenAiConfig::new("your-api-key");
/// let model = OpenAiChatModel::new(config, "gpt-4".to_string(), None);
///
/// let executor = model.create_executor(
///     http_client,
///     vec![],  // interceptors
///     vec![],  // middlewares
///     None,    // retry_options
/// );
/// ```
pub struct OpenAiChatModel {
    config: ModelConfig,
    model: String,
    adapter: Option<Arc<dyn OpenAiChatAdapter>>,
    /// OpenAI-specific configuration (for Responses API, etc.)
    openai_config: OpenAiConfig,
}

impl OpenAiChatModel {
    /// Create a new OpenAI Chat Model
    ///
    /// # Arguments
    /// * `config` - OpenAI configuration
    /// * `model` - Model name (e.g., "gpt-4", "gpt-4-turbo")
    /// * `adapter` - Optional adapter for provider-specific customization
    pub fn new(
        config: OpenAiConfig,
        model: String,
        adapter: Option<Arc<dyn OpenAiChatAdapter>>,
    ) -> Self {
        let model_config = ModelConfig::from(&config);
        Self {
            config: model_config,
            model,
            adapter,
            openai_config: config,
        }
    }

    /// Create a new OpenAI Chat Model with adapter (for OpenAI-compatible providers)
    ///
    /// # Arguments
    /// * `config` - OpenAI configuration
    /// * `model` - Model name
    /// * `adapter` - Adapter for provider-specific customization
    pub fn new_with_adapter(
        config: OpenAiConfig,
        model: String,
        adapter: Arc<dyn OpenAiChatAdapter>,
    ) -> Self {
        Self::new(config, model, Some(adapter))
    }

    /// Get the model name
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Get the adapter
    pub fn adapter(&self) -> Option<&Arc<dyn OpenAiChatAdapter>> {
        self.adapter.as_ref()
    }
}

impl ChatModel for OpenAiChatModel {
    fn create_executor(
        &self,
        http_client: reqwest::Client,
        interceptors: Vec<Arc<dyn HttpInterceptor>>,
        middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
        retry_options: Option<RetryOptions>,
    ) -> HttpChatExecutor {
        // Create spec
        let spec = Arc::new(OpenAiSpec::new());

        // Create provider context
        let mut ctx = ProviderContext::new(
            &self.config.provider_id,
            self.config.base_url.clone(),
            Some(self.config.api_key.expose_secret().to_string()),
            self.config.headers.clone(),
        )
        .with_org_project(
            self.config.organization.clone(),
            self.config.project.clone(),
        );

        // Add extras for Responses API support
        let mut extras = std::collections::HashMap::new();
        if let Some(fmt) = &self.openai_config.openai_params.response_format {
            if let Ok(value) = serde_json::to_value(fmt) {
                extras.insert("openai.response_format".to_string(), value);
            }
        }
        ctx = ctx.with_extras(extras);

        // Get transformers from spec
        // Note: We use a dummy request here to get transformers
        // The actual request will be passed to the executor
        let dummy_request = ChatRequest {
            common_params: crate::types::CommonParams {
                model: self.model.clone(),
                ..Default::default()
            },
            ..Default::default()
        };
        let bundle = spec.choose_chat_transformers(&dummy_request, &ctx);
        let before_send = spec.chat_before_send(&dummy_request, &ctx);

        // Create executor
        HttpChatExecutor {
            provider_id: self.config.provider_id.clone(),
            http_client,
            request_transformer: bundle.request,
            response_transformer: bundle.response,
            stream_transformer: bundle.stream,
            json_stream_converter: bundle.json,
            stream_disable_compression: self.openai_config.http_config.stream_disable_compression,
            interceptors,
            middlewares,
            provider_spec: spec,
            provider_context: ctx,
            before_send,
            retry_options,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_openai_chat_model() {
        let config = OpenAiConfig::new("test-key").with_model("gpt-4");
        let model = OpenAiChatModel::new(config, "gpt-4".to_string(), None);

        assert_eq!(model.model(), "gpt-4");
        assert!(model.adapter().is_none());
    }

    #[test]
    fn test_create_executor() {
        let config = OpenAiConfig::new("test-key").with_model("gpt-4");
        let model = OpenAiChatModel::new(config, "gpt-4".to_string(), None);

        let http_client = reqwest::Client::new();
        let executor = model.create_executor(http_client, vec![], vec![], None);

        assert_eq!(executor.provider_id, "openai");
        assert!(executor.interceptors.is_empty());
        assert!(executor.middlewares.is_empty());
    }
}
