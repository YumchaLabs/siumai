//! Groq API Models Capability Implementation
//!
//! Implements model listing and information capabilities for Groq.

use crate::error::LlmError;
use crate::traits::ModelListingCapability;
use crate::types::{HttpConfig, ModelInfo};
use crate::{core::ProviderContext, execution::executors::common::HttpExecutionConfig};
use async_trait::async_trait;
use secrecy::SecretString;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub enum ModelType {
    Chat,
    Audio,
    Embedding,
    Image,
}

use super::types::*;
use super::utils::*;

/// Groq Models API Implementation
#[derive(Clone)]
pub struct GroqModels {
    pub api_key: SecretString,
    pub base_url: String,
    pub http_client: reqwest::Client,
    pub http_config: HttpConfig,
}

impl GroqModels {
    /// Create a new Groq models API instance
    pub fn new(
        api_key: SecretString,
        base_url: String,
        http_client: reqwest::Client,
        http_config: HttpConfig,
    ) -> Self {
        Self {
            api_key,
            base_url,
            http_client,
            http_config,
        }
    }

    /// Convert Groq model to our ModelInfo
    fn convert_groq_model(&self, groq_model: GroqModel) -> ModelInfo {
        convert_groq_model_to_model_info(&groq_model)
    }
}

impl GroqModels {
    async fn list_models_internal(&self) -> Result<Vec<ModelInfo>, LlmError> {
        use crate::execution::executors::common::execute_get_request;
        use secrecy::ExposeSecret;

        let ctx = ProviderContext::new(
            "groq",
            self.base_url.clone(),
            Some(self.api_key.expose_secret().to_string()),
            self.http_config.headers.clone(),
        );
        let spec: Arc<dyn crate::core::ProviderSpec> = Arc::new(crate::providers::groq::spec::GroqSpec);
        let url = spec.models_url(&ctx);
        let config = HttpExecutionConfig {
            provider_id: "groq".to_string(),
            http_client: self.http_client.clone(),
            provider_spec: spec,
            provider_context: ctx,
            interceptors: Vec::new(),
            retry_options: None,
        };

        let res = execute_get_request(&config, &url, None)
            .await
            .map_err(|e| match e {
                LlmError::ApiError {
                    code,
                    message: _,
                    details,
                } => {
                    let error_message = details
                        .as_ref()
                        .map(|d| extract_error_message(&d.to_string()))
                        .unwrap_or_else(|| "Unknown error".to_string());
                    LlmError::ApiError {
                        code,
                        message: format!("Groq list models error: {error_message}"),
                        details,
                    }
                }
                other => other,
            })?;

        let groq_response: GroqModelsResponse = serde_json::from_value(res.json)?;
        let models = groq_response
            .data
            .into_iter()
            .map(|m| self.convert_groq_model(m))
            .collect();

        Ok(models)
    }

    async fn get_model_internal(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        use crate::execution::executors::common::execute_get_request;
        use secrecy::ExposeSecret;

        let ctx = ProviderContext::new(
            "groq",
            self.base_url.clone(),
            Some(self.api_key.expose_secret().to_string()),
            self.http_config.headers.clone(),
        );
        let spec: Arc<dyn crate::core::ProviderSpec> = Arc::new(crate::providers::groq::spec::GroqSpec);
        let url = spec.model_url(&model_id, &ctx);
        let config = HttpExecutionConfig {
            provider_id: "groq".to_string(),
            http_client: self.http_client.clone(),
            provider_spec: spec,
            provider_context: ctx,
            interceptors: Vec::new(),
            retry_options: None,
        };

        let res = execute_get_request(&config, &url, None)
            .await
            .map_err(|e| match e {
                LlmError::ApiError {
                    code,
                    message: _,
                    details,
                } => {
                    let error_message = details
                        .as_ref()
                        .map(|d| extract_error_message(&d.to_string()))
                        .unwrap_or_else(|| "Unknown error".to_string());
                    LlmError::ApiError {
                        code,
                        message: format!("Groq get model error: {error_message}"),
                        details,
                    }
                }
                other => other,
            })?;

        let groq_model: GroqModel = serde_json::from_value(res.json)?;
        Ok(self.convert_groq_model(groq_model))
    }

    #[cfg(test)]
    fn supports_model_listing(&self) -> bool {
        true
    }
}

pub(crate) fn groq_model_capabilities(model: &GroqModel) -> Vec<String> {
    let mut capabilities = Vec::new();

    if model.id.contains("whisper") {
        capabilities.push("transcription".to_string());
        if !model.id.contains("-en") {
            capabilities.push("translation".to_string());
        }
    } else if model.id.contains("tts") || model.id.contains("playai") {
        capabilities.push("speech_synthesis".to_string());
    } else {
        capabilities.push("chat".to_string());
        capabilities.push("streaming".to_string());
        if !model.id.contains("guard") {
            capabilities.push("function_calling".to_string());
        }
        if model.id.contains("qwen") {
            capabilities.push("reasoning".to_string());
        }
    }

    capabilities
}

pub(crate) fn convert_groq_model_to_model_info(model: &GroqModel) -> ModelInfo {
    let model_id = model.id.clone();
    ModelInfo {
        id: model_id.clone(),
        name: Some(model_id),
        description: None,
        owned_by: model.owned_by.clone(),
        created: Some(model.created),
        capabilities: groq_model_capabilities(model),
        context_window: Some(model.context_window),
        max_output_tokens: model.max_completion_tokens,
        input_cost_per_token: None,
        output_cost_per_token: None,
    }
}

#[async_trait]
impl ModelListingCapability for GroqModels {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.list_models_internal().await
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        self.get_model_internal(model_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::HttpConfig;

    fn create_test_models() -> GroqModels {
        use secrecy::SecretString;
        GroqModels::new(
            SecretString::from("test-api-key".to_string()),
            "https://api.groq.com/openai/v1".to_string(),
            reqwest::Client::new(),
            HttpConfig::default(),
        )
    }

    #[test]
    fn test_convert_groq_model_chat() {
        let models = create_test_models();
        let groq_model = GroqModel {
            id: "llama-3.3-70b-versatile".to_string(),
            object: "model".to_string(),
            created: 1640995200,
            owned_by: "Meta".to_string(),
            active: true,
            context_window: 32768,
            public_apps: None,
            max_completion_tokens: Some(8192),
        };

        let model_info = models.convert_groq_model(groq_model);

        assert_eq!(model_info.id, "llama-3.3-70b-versatile");
        assert_eq!(model_info.context_window, Some(32768));
        assert_eq!(model_info.max_output_tokens, Some(8192));
        assert!(model_info.capabilities.contains(&"chat".to_string()));
        assert!(model_info.capabilities.contains(&"streaming".to_string()));
        assert!(
            model_info
                .capabilities
                .contains(&"function_calling".to_string())
        );
    }

    #[test]
    fn test_convert_groq_model_whisper() {
        let models = create_test_models();
        let groq_model = GroqModel {
            id: "whisper-large-v3".to_string(),
            object: "model".to_string(),
            created: 1640995200,
            owned_by: "OpenAI".to_string(),
            active: true,
            context_window: 448,
            public_apps: None,
            max_completion_tokens: None,
        };

        let model_info = models.convert_groq_model(groq_model);

        assert_eq!(model_info.id, "whisper-large-v3");
        assert!(
            model_info
                .capabilities
                .contains(&"transcription".to_string())
        );
        assert!(model_info.capabilities.contains(&"translation".to_string()));
    }

    #[test]
    fn test_convert_groq_model_qwen() {
        let models = create_test_models();
        let groq_model = GroqModel {
            id: "qwen3-8b-instruct".to_string(),
            object: "model".to_string(),
            created: 1640995200,
            owned_by: "Alibaba".to_string(),
            active: true,
            context_window: 8192,
            public_apps: None,
            max_completion_tokens: Some(4096),
        };

        let model_info = models.convert_groq_model(groq_model);

        assert_eq!(model_info.id, "qwen3-8b-instruct");
        assert!(model_info.capabilities.contains(&"reasoning".to_string()));
    }

    #[test]
    fn test_capability_support() {
        let models = create_test_models();
        assert!(models.supports_model_listing());
    }
}
