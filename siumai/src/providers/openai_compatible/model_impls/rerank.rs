//! OpenAI-Compatible Rerank Model Implementation

use crate::error::LlmError;
use crate::executors::rerank::HttpRerankExecutor;
use crate::provider_model::RerankModel;
use crate::providers::openai_compatible::openai_config::OpenAiCompatibleConfig;
use crate::retry_api::RetryOptions;
use crate::standards::openai::rerank::{OpenAiRerankAdapter, OpenAiRerankStandard};
use crate::types::RerankRequest;
use crate::utils::http_interceptor::HttpInterceptor;
use std::sync::Arc;

/// Wrapper adapter to convert ProviderAdapter to OpenAiRerankAdapter
struct ProviderAdapterWrapper {
    provider_id: String,
}

impl OpenAiRerankAdapter for ProviderAdapterWrapper {
    fn transform_request(
        &self,
        _req: &RerankRequest,
        _body: &mut serde_json::Value,
    ) -> Result<(), LlmError> {
        // Default implementation - no transformation needed
        // Provider-specific adapters can override this
        Ok(())
    }

    fn transform_response(&self, _resp: &mut serde_json::Value) -> Result<(), LlmError> {
        // Default implementation - no transformation needed
        Ok(())
    }
}

/// OpenAI-Compatible Rerank Model
///
/// Represents a rerank model from an OpenAI-compatible provider.
/// Uses the OpenAI Rerank Standard Layer with provider-specific adapter.
///
/// # Note
/// Not all OpenAI-compatible providers support rerank.
/// Currently known to be supported by:
/// - SiliconFlow
pub struct OpenAiCompatibleRerankModel {
    config: OpenAiCompatibleConfig,
    model: String,
}

impl OpenAiCompatibleRerankModel {
    /// Create a new OpenAI-Compatible Rerank Model
    ///
    /// # Arguments
    /// * `config` - Provider configuration
    /// * `model` - Model name
    pub fn new(config: OpenAiCompatibleConfig, model: String) -> Self {
        Self { config, model }
    }
}

impl RerankModel for OpenAiCompatibleRerankModel {
    fn create_executor(
        &self,
        http_client: reqwest::Client,
        interceptors: Vec<Arc<dyn HttpInterceptor>>,
        retry_options: Option<RetryOptions>,
    ) -> HttpRerankExecutor {
        // Create wrapper adapter
        let wrapper_adapter = Arc::new(ProviderAdapterWrapper {
            provider_id: self.config.provider_id.clone(),
        });

        // Create OpenAI Rerank Standard with wrapper adapter
        let standard = OpenAiRerankStandard::with_adapter(wrapper_adapter);

        // Get transformers from standard
        let transformers = standard.create_transformers(&self.config.provider_id);

        // Build rerank URL
        let url = format!("{}/rerank", self.config.base_url.trim_end_matches('/'));

        // Build headers
        let mut headers = self.config.custom_headers.clone();
        headers.insert(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_str(&format!("Bearer {}", self.config.api_key))
                .unwrap_or_else(|_| reqwest::header::HeaderValue::from_static("")),
        );
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            reqwest::header::HeaderValue::from_static("application/json"),
        );

        // Create executor
        HttpRerankExecutor {
            provider_id: self.config.provider_id.clone(),
            http_client,
            request_transformer: transformers.request,
            response_transformer: transformers.response,
            interceptors,
            retry_options,
            url,
            headers,
            before_send: None,
        }
    }
}
