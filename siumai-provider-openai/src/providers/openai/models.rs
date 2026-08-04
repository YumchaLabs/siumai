//! Legacy OpenAI Models resource client.
//!
//! The Models API only returns identity and ownership metadata. This adapter
//! deliberately does not infer capabilities, limits, recommendations, or
//! prices from model-name patterns. Current policy belongs to the configured
//! provider profile and unsupported metadata remains unknown.

use std::sync::Arc;

use async_trait::async_trait;
use secrecy::{ExposeSecret, SecretString};

use crate::error::LlmError;
use crate::execution::http::transport::HttpTransport;
use crate::traits::ModelListingCapability;
use crate::types::{HttpConfig, ModelInfo};

use super::types::{OpenAiModel, OpenAiModelsResponse};

/// OpenAI Models resource client.
#[derive(Clone)]
pub struct OpenAiModels {
    pub api_key: SecretString,
    pub base_url: String,
    pub http_client: reqwest::Client,
    pub organization: Option<String>,
    pub project: Option<String>,
    pub http_config: HttpConfig,
    pub http_transport: Option<Arc<dyn HttpTransport>>,
}

impl OpenAiModels {
    pub fn new(
        api_key: SecretString,
        base_url: String,
        http_client: reqwest::Client,
        organization: Option<String>,
        project: Option<String>,
        http_config: HttpConfig,
        http_transport: Option<Arc<dyn HttpTransport>>,
    ) -> Self {
        Self {
            api_key,
            base_url,
            http_client,
            organization,
            project,
            http_config,
            http_transport,
        }
    }

    fn build_http_config(&self) -> crate::execution::executors::common::HttpExecutionConfig {
        let spec = Arc::new(super::spec::OpenAiSpec::new());
        let context = crate::core::ProviderContext::new(
            "openai",
            self.base_url.clone(),
            Some(self.api_key.expose_secret().to_string()),
            self.http_config.headers.clone(),
        )
        .with_org_project(self.organization.clone(), self.project.clone());

        let mut wiring = crate::execution::wiring::HttpExecutionWiring::new(
            "openai",
            self.http_client.clone(),
            context,
        );
        if let Some(transport) = self.http_transport.clone() {
            wiring = wiring.with_transport(transport);
        }
        wiring.config(spec)
    }
}

#[async_trait]
impl ModelListingCapability for OpenAiModels {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        let config = self.build_http_config();
        let url = config
            .provider_spec
            .try_models_url(&config.provider_context)?;
        let result =
            crate::execution::executors::common::execute_get_request(&config, &url, None).await?;
        let response: OpenAiModelsResponse =
            serde_json::from_value(result.json).map_err(|error| {
                LlmError::ParseError(format!("Failed to parse OpenAI models response: {error}"))
            })?;

        Ok(response
            .data
            .into_iter()
            .map(convert_openai_model_to_model_info)
            .collect())
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        let config = self.build_http_config();
        let url = config
            .provider_spec
            .try_model_url(&model_id, &config.provider_context)?;
        let result =
            crate::execution::executors::common::execute_get_request(&config, &url, None).await?;
        let model: OpenAiModel = serde_json::from_value(result.json).map_err(|error| {
            LlmError::ParseError(format!("Failed to parse OpenAI model response: {error}"))
        })?;
        Ok(convert_openai_model_to_model_info(model))
    }
}

pub(crate) fn convert_openai_model_to_model_info(model: OpenAiModel) -> ModelInfo {
    ModelInfo {
        id: model.id.clone(),
        name: Some(model.id),
        description: None,
        owned_by: model.owned_by,
        created: model.created,
        capabilities: Vec::new(),
        context_window: None,
        max_output_tokens: None,
        input_cost_per_token: None,
        output_cost_per_token: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::openai::OpenAiConfig;

    #[test]
    fn models_endpoint_uses_provider_resource_paths() {
        let config = OpenAiConfig::new("test-key");
        let models = OpenAiModels::new(
            config.api_key.clone(),
            config.base_url.clone(),
            reqwest::Client::new(),
            config.organization.clone(),
            config.project.clone(),
            config.http_config.clone(),
            None,
        );
        let execution = models.build_http_config();
        assert_eq!(
            execution
                .provider_spec
                .try_models_url(&execution.provider_context)
                .unwrap(),
            "https://api.openai.com/v1/models"
        );
        assert_eq!(
            execution
                .provider_spec
                .try_model_url("gpt-5.6-sol", &execution.provider_context)
                .unwrap(),
            "https://api.openai.com/v1/models/gpt-5.6-sol"
        );
    }

    #[test]
    fn models_endpoint_does_not_invent_capabilities_limits_or_prices() {
        let model = convert_openai_model_to_model_info(OpenAiModel {
            id: "future-model".to_string(),
            object: "model".to_string(),
            created: Some(1),
            owned_by: "openai".to_string(),
            permission: None,
            root: None,
            parent: None,
        });

        assert!(model.capabilities.is_empty());
        assert_eq!(model.description, None);
        assert_eq!(model.context_window, None);
        assert_eq!(model.max_output_tokens, None);
        assert_eq!(model.input_cost_per_token, None);
        assert_eq!(model.output_cost_per_token, None);
    }
}
