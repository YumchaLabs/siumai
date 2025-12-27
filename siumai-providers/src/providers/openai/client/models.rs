use super::OpenAiClient;
use crate::error::LlmError;
use crate::execution::executors::common::{HttpExecutionConfig, execute_get_request};
use crate::traits::ModelListingCapability;
use crate::types::ModelInfo;
use async_trait::async_trait;
use secrecy::ExposeSecret;
use std::sync::Arc;

#[async_trait]
impl ModelListingCapability for OpenAiClient {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        let spec = Arc::new(crate::providers::openai::spec::OpenAiSpec::new());
        let ctx = crate::core::ProviderContext::new(
            "openai",
            self.base_url.clone(),
            Some(self.api_key.expose_secret().to_string()),
            self.http_config.headers.clone(),
        )
        .with_org_project(self.organization.clone(), self.project.clone());

        let config = HttpExecutionConfig {
            provider_id: "openai".to_string(),
            http_client: self.http_client.clone(),
            provider_spec: spec,
            provider_context: ctx,
            interceptors: self.http_interceptors.clone(),
            retry_options: self.retry_options.clone(),
        };
        let url = config.provider_spec.models_url(&config.provider_context);

        let res = execute_get_request(&config, &url, None).await?;
        let models_response: crate::providers::openai::types::OpenAiModelsResponse =
            serde_json::from_value(res.json).map_err(|e| {
                LlmError::ParseError(format!("Failed to parse OpenAI models response: {e}"))
            })?;

        Ok(models_response
            .data
            .into_iter()
            .map(crate::providers::openai::models::convert_openai_model_to_model_info)
            .collect())
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        let spec = Arc::new(crate::providers::openai::spec::OpenAiSpec::new());
        let ctx = crate::core::ProviderContext::new(
            "openai",
            self.base_url.clone(),
            Some(self.api_key.expose_secret().to_string()),
            self.http_config.headers.clone(),
        )
        .with_org_project(self.organization.clone(), self.project.clone());

        let config = HttpExecutionConfig {
            provider_id: "openai".to_string(),
            http_client: self.http_client.clone(),
            provider_spec: spec,
            provider_context: ctx,
            interceptors: self.http_interceptors.clone(),
            retry_options: self.retry_options.clone(),
        };
        let url = config.provider_spec.model_url(&model_id, &config.provider_context);

        let res = execute_get_request(&config, &url, None).await?;
        let model: crate::providers::openai::types::OpenAiModel = serde_json::from_value(res.json)
            .map_err(|e| {
                LlmError::ParseError(format!("Failed to parse OpenAI model response: {e}"))
            })?;
        Ok(crate::providers::openai::models::convert_openai_model_to_model_info(model))
    }
}
