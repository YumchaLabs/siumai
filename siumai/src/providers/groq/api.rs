//! Groq API Models Capability Implementation
//!
//! Implements model listing and information capabilities for Groq.

use crate::error::LlmError;
use crate::traits::ModelListingCapability;
use crate::types::{HttpConfig, ModelInfo};
use async_trait::async_trait;
use secrecy::SecretString;

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
        let model_id = groq_model.id.clone();

        ModelInfo {
            id: model_id.clone(),
            name: Some(model_id),
            description: None, // Groq doesn't provide descriptions
            owned_by: groq_model.owned_by.clone(),
            created: Some(groq_model.created),
            capabilities: self.get_model_capabilities(&groq_model),
            context_window: Some(groq_model.context_window),
            max_output_tokens: groq_model.max_completion_tokens,
            input_cost_per_token: None, // Groq doesn't provide pricing info via API
            output_cost_per_token: None,
        }
    }

    /// Get capabilities for a specific model
    fn get_model_capabilities(&self, model: &GroqModel) -> Vec<String> {
        let mut capabilities = Vec::new();

        if model.id.contains("whisper") {
            capabilities.push("transcription".to_string());
            if !model.id.contains("-en") {
                capabilities.push("translation".to_string());
            }
        } else if model.id.contains("tts") || model.id.contains("playai") {
            capabilities.push("speech_synthesis".to_string());
        } else {
            // Chat models
            capabilities.push("chat".to_string());
            capabilities.push("streaming".to_string());

            // Most Groq chat models support function calling
            if !model.id.contains("guard") {
                capabilities.push("function_calling".to_string());
            }

            // Some models support reasoning
            if model.id.contains("qwen") {
                capabilities.push("reasoning".to_string());
            }
        }

        capabilities
    }
}

impl GroqModels {
    async fn list_models_internal(&self) -> Result<Vec<ModelInfo>, LlmError> {
        use secrecy::ExposeSecret;
        let url = format!("{}/models", self.base_url);
        let headers = build_headers(self.api_key.expose_secret(), &self.http_config.headers)?;

        let response = self.http_client.get(&url).headers(headers).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            let error_message = extract_error_message(&error_text);

            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!("Groq list models error: {error_message}"),
                details: serde_json::from_str(&error_text).ok(),
            });
        }

        let groq_response: GroqModelsResponse = response.json().await?;
        let models = groq_response
            .data
            .into_iter()
            .map(|m| self.convert_groq_model(m))
            .collect();

        Ok(models)
    }

    async fn get_model_internal(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        use secrecy::ExposeSecret;
        let url = format!("{}/models/{}", self.base_url, model_id);
        let headers = build_headers(self.api_key.expose_secret(), &self.http_config.headers)?;

        let response = self.http_client.get(&url).headers(headers).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            let error_message = extract_error_message(&error_text);

            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!("Groq get model error: {error_message}"),
                details: serde_json::from_str(&error_text).ok(),
            });
        }

        let groq_model: GroqModel = response.json().await?;
        Ok(self.convert_groq_model(groq_model))
    }

    #[cfg(test)]
    fn supports_model_listing(&self) -> bool {
        true
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
