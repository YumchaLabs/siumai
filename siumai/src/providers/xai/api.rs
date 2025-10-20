//! xAI API Models Capability Implementation
//!
//! Implements model listing and information capabilities for xAI.

use crate::error::LlmError;
use crate::traits::ModelListingCapability;
use crate::types::{HttpConfig, ModelInfo};
use async_trait::async_trait;

use super::types::*;
use super::utils::build_headers;

/// xAI Models API implementation
#[derive(Debug, Clone)]
pub struct XaiModels {
    pub api_key: String,
    pub base_url: String,
    pub http_client: reqwest::Client,
    pub http_config: HttpConfig,
}

impl XaiModels {
    /// Create a new xAI models API instance
    pub const fn new(
        api_key: String,
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

    /// Convert xAI model to our ModelInfo
    fn convert_xai_model(&self, xai_model: XaiModel) -> ModelInfo {
        let model_id = xai_model.id.clone();

        ModelInfo {
            id: model_id.clone(),
            name: Some(model_id),
            description: None, // xAI doesn't provide descriptions via API
            owned_by: xai_model.owned_by.clone(),
            created: Some(xai_model.created),
            capabilities: self.get_model_capabilities(&xai_model),
            context_window: None,       // Not provided by xAI API
            max_output_tokens: None,    // Not provided by xAI API
            input_cost_per_token: None, // xAI doesn't provide pricing info via API
            output_cost_per_token: None,
        }
    }

    /// Get capabilities for a specific model
    fn get_model_capabilities(&self, model: &XaiModel) -> Vec<String> {
        let mut capabilities = Vec::new();

        // All xAI models support chat
        capabilities.push("chat".to_string());
        capabilities.push("streaming".to_string());

        // Most xAI models support function calling
        if !model.id.contains("beta") {
            capabilities.push("function_calling".to_string());
        }

        // Grok 4 models support vision
        if model.id.contains("grok-4") {
            capabilities.push("vision".to_string());
        }

        // Grok 4 models support reasoning
        if model.id.contains("grok-4") {
            capabilities.push("reasoning".to_string());
        }

        // Image generation models
        if model.id.contains("image") {
            capabilities.push("image_generation".to_string());
        }

        capabilities
    }

    /// List models from xAI API
    async fn list_models_from_api(&self) -> Result<Vec<ModelInfo>, LlmError> {
        let url = format!("{}/models", self.base_url);
        let headers = build_headers(&self.api_key, &self.http_config.headers)?;

        let response = self.http_client.get(&url).headers(headers).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            let error_message = error_text.clone();

            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!("xAI list models error: {error_message}"),
                details: serde_json::from_str(&error_text).ok(),
            });
        }

        let xai_response: XaiModelsResponse = response.json().await?;
        let models = xai_response
            .data
            .into_iter()
            .map(|m| self.convert_xai_model(m))
            .collect();

        Ok(models)
    }

    /// Get specific model from xAI API
    async fn get_model_from_api(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        let url = format!("{}/models/{}", self.base_url, model_id);
        let headers = build_headers(&self.api_key, &self.http_config.headers)?;

        let response = self.http_client.get(&url).headers(headers).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            let error_message = error_text.clone();

            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: format!("xAI get model error: {error_message}"),
                details: serde_json::from_str(&error_text).ok(),
            });
        }

        let xai_model: XaiModel = response.json().await?;
        Ok(self.convert_xai_model(xai_model))
    }

    /// Get hardcoded model list (fallback when API is not available)
    fn get_hardcoded_models(&self) -> Vec<ModelInfo> {
        crate::providers::xai::models::all_models()
            .into_iter()
            .map(|model_id| ModelInfo {
                id: model_id.to_string(),
                name: Some(model_id.to_string()),
                description: Some(format!("xAI {} model", model_id)),
                owned_by: "xai".to_string(),
                created: None,
                capabilities: self.get_hardcoded_capabilities(model_id),
                context_window: None,
                max_output_tokens: None,
                input_cost_per_token: None,
                output_cost_per_token: None,
            })
            .collect()
    }

    /// Get hardcoded capabilities for a model
    fn get_hardcoded_capabilities(&self, model_id: &str) -> Vec<String> {
        let mut capabilities = vec!["chat".to_string(), "streaming".to_string()];

        if model_id.contains("grok-4") {
            capabilities.extend_from_slice(&[
                "vision".to_string(),
                "reasoning".to_string(),
                "function_calling".to_string(),
            ]);
        } else if !model_id.contains("beta") {
            capabilities.push("function_calling".to_string());
        }

        if model_id.contains("image") {
            capabilities.push("image_generation".to_string());
        }

        capabilities
    }
}

#[async_trait]
impl ModelListingCapability for XaiModels {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        // Try API first, fallback to hardcoded list
        match self.list_models_from_api().await {
            Ok(models) => Ok(models),
            Err(_) => {
                // If API fails, return hardcoded model list
                Ok(self.get_hardcoded_models())
            }
        }
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        // Try API first, fallback to hardcoded info
        match self.get_model_from_api(model_id.clone()).await {
            Ok(model) => Ok(model),
            Err(_) => {
                // If API fails, return hardcoded model info
                let hardcoded_models = self.get_hardcoded_models();
                hardcoded_models
                    .into_iter()
                    .find(|m| m.id == model_id)
                    .ok_or_else(|| {
                        LlmError::InvalidInput(format!("Model '{}' not found", model_id))
                    })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::HttpConfig;

    fn create_test_models() -> XaiModels {
        XaiModels::new(
            "test-api-key".to_string(),
            "https://api.x.ai/v1".to_string(),
            reqwest::Client::new(),
            HttpConfig::default(),
        )
    }

    #[test]
    fn test_get_hardcoded_models() {
        let models = create_test_models();
        let hardcoded = models.get_hardcoded_models();

        assert!(!hardcoded.is_empty());
        assert!(hardcoded.iter().any(|m| m.id.contains("grok")));
    }

    #[test]
    fn test_get_hardcoded_capabilities() {
        let models = create_test_models();

        let grok4_caps = models.get_hardcoded_capabilities("grok-4");
        assert!(grok4_caps.contains(&"vision".to_string()));
        assert!(grok4_caps.contains(&"reasoning".to_string()));

        let grok3_caps = models.get_hardcoded_capabilities("grok-3");
        assert!(grok3_caps.contains(&"chat".to_string()));
        assert!(grok3_caps.contains(&"function_calling".to_string()));
    }
}
