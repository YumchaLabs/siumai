//! OpenAI-Compatible Chat Model Implementation

use crate::executors::chat::HttpChatExecutor;
use crate::middleware::language_model::LanguageModelMiddleware;
use crate::provider_core::{ProviderContext, ProviderSpec};
use crate::provider_model::ChatModel;
use crate::providers::openai_compatible::openai_config::OpenAiCompatibleConfig;
use crate::providers::openai_compatible::spec::OpenAiCompatibleSpec;
use crate::retry_api::RetryOptions;
use crate::types::ChatRequest;
use crate::utils::http_headers::headermap_to_hashmap;
use crate::utils::http_interceptor::HttpInterceptor;
use std::sync::Arc;

/// OpenAI-Compatible Chat Model
///
/// Represents a chat model from an OpenAI-compatible provider.
/// Uses the OpenAI Standard Layer with provider-specific adapter.
pub struct OpenAiCompatibleChatModel {
    config: OpenAiCompatibleConfig,
    model: String,
}

impl OpenAiCompatibleChatModel {
    /// Create a new OpenAI-Compatible Chat Model
    ///
    /// # Arguments
    /// * `config` - Provider configuration
    /// * `model` - Model name
    pub fn new(config: OpenAiCompatibleConfig, model: String) -> Self {
        Self { config, model }
    }
}

impl ChatModel for OpenAiCompatibleChatModel {
    fn create_executor(
        &self,
        http_client: reqwest::Client,
        interceptors: Vec<Arc<dyn HttpInterceptor>>,
        middlewares: Vec<Arc<dyn LanguageModelMiddleware>>,
        retry_options: Option<RetryOptions>,
    ) -> HttpChatExecutor {
        // Create spec (OpenAiCompatibleSpec handles adapter resolution)
        let spec = Arc::new(OpenAiCompatibleSpec);

        // Create provider context
        let ctx = ProviderContext::new(
            &self.config.provider_id,
            self.config.base_url.clone(),
            Some(self.config.api_key.clone()),
            headermap_to_hashmap(&self.config.custom_headers),
        );

        // Get transformers from spec
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
            stream_disable_compression: self.config.http_config.stream_disable_compression,
            interceptors,
            middlewares,
            provider_spec: spec,
            provider_context: ctx,
            before_send,
            retry_options,
        }
    }
}
