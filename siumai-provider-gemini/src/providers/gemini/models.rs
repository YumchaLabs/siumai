//! Gemini Models Capability Implementation
//!
//! This module implements model listing functionality for Google Gemini API.

use async_trait::async_trait;
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::core::ProviderSpec;
use crate::error::LlmError;
use crate::traits::ModelListingCapability;
use crate::types::ModelInfo;

use super::types::GeminiConfig;
use super::{
    model_constants::{gemini_2_5_flash, gemini_2_5_flash_lite, gemini_2_5_pro, gemini_3},
    model_policy::{CapabilitySupport, model_capability_support, model_policy},
};

/// Gemini model information from API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiModel {
    /// The resource name of the Model.
    pub name: String,
    /// The human-readable name of the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    /// A short description of the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// For Tuned Models, this is the version of the base model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Maximum number of input tokens allowed for this model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_token_limit: Option<i32>,
    /// Maximum number of output tokens allowed for this model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_token_limit: Option<i32>,
    /// The model's supported generation methods.
    #[serde(default, rename = "supportedGenerationMethods")]
    pub supported_generation_methods: Vec<String>,
    /// Controls the randomness of the output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// For Nucleus sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// For Top-k sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
}

/// Response from the list models API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListModelsResponse {
    /// The returned Models.
    #[serde(default)]
    pub models: Vec<GeminiModel>,
    /// A token, which can be sent as `page_token` to retrieve the next page.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_page_token: Option<String>,
}

/// Gemini models capability implementation
#[derive(Debug, Clone)]
pub struct GeminiModels {
    config: GeminiConfig,
    http_client: HttpClient,
}

impl GeminiModels {
    /// Create a new Gemini models capability
    pub const fn new(config: GeminiConfig, http_client: HttpClient) -> Self {
        Self {
            config,
            http_client,
        }
    }

    fn build_http_config(
        &self,
        ctx: crate::core::ProviderContext,
    ) -> crate::execution::executors::common::HttpExecutionConfig {
        let mut wiring = crate::execution::wiring::HttpExecutionWiring::new(
            "gemini",
            self.http_client.clone(),
            ctx,
        );
        if let Some(transport) = self.config.http_transport.clone() {
            wiring = wiring.with_transport(transport);
        }
        wiring.config(Arc::new(crate::providers::gemini::spec::GeminiSpec))
    }

    /// Convert `GeminiModel` to `ModelInfo`
    fn convert_model(&self, model: GeminiModel) -> ModelInfo {
        // Extract model ID from the full name (e.g., "models/gemini-1.5-flash" -> "gemini-1.5-flash")
        let id = model
            .name
            .strip_prefix("models/")
            .unwrap_or(&model.name)
            .to_string();

        // Derive transport capabilities from API metadata and provider capabilities from the
        // exact audited model policy. Unknown ids remain callable without gaining guessed claims.
        let mut capabilities = Vec::new();

        if model
            .supported_generation_methods
            .contains(&"generateContent".to_string())
        {
            capabilities.push("chat".to_string());
        }

        if model
            .supported_generation_methods
            .contains(&"streamGenerateContent".to_string())
        {
            capabilities.push("streaming".to_string());
        }

        let policy = model_policy(&id);
        if let Some(policy) = policy {
            for capability in policy.capabilities {
                if !capabilities
                    .iter()
                    .any(|known| known.as_str() == *capability)
                {
                    capabilities.push((*capability).to_string());
                }
            }
        }

        // Prefer API-provided limits and fall back only to exact curated policies.
        let context_window = model
            .input_token_limit
            .and_then(|tokens| u32::try_from(tokens).ok())
            .or_else(|| policy.and_then(|policy| policy.context_window));
        let max_output_tokens = model
            .output_token_limit
            .and_then(|tokens| u32::try_from(tokens).ok())
            .or_else(|| policy.and_then(|policy| policy.max_output_tokens));

        ModelInfo {
            id,
            name: Some(model.display_name.unwrap_or(model.name)),
            description: model.description,
            context_window,
            max_output_tokens,
            capabilities,
            input_cost_per_token: None,
            output_cost_per_token: None,
            created: None,
            owned_by: "Google".to_string(),
        }
    }

    /// Get all available models with pagination
    async fn fetch_all_models(&self) -> Result<Vec<GeminiModel>, LlmError> {
        let mut all_models = Vec::new();
        let mut page_token: Option<String> = None;

        loop {
            let ctx = super::context::build_context(&self.config).await;
            let spec = crate::providers::gemini::spec::GeminiSpec;
            let mut url = spec.try_models_url(&ctx)?;

            // Add pagination parameters
            let mut params = Vec::new();
            if let Some(token) = &page_token {
                params.push(format!("pageToken={token}"));
            }
            params.push("pageSize=50".to_string()); // Request up to 50 models per page

            if !params.is_empty() {
                url.push('?');
                url.push_str(&params.join("&"));
            }

            let config = self.build_http_config(ctx);
            let result =
                crate::execution::executors::common::execute_get_request(&config, &url, None)
                    .await?;

            let list_response: ListModelsResponse =
                serde_json::from_value(result.json).map_err(|e| {
                    LlmError::ParseError(format!("Failed to parse models response: {e}"))
                })?;

            all_models.extend(list_response.models);

            // Check if there are more pages
            if let Some(next_token) = list_response.next_page_token {
                page_token = Some(next_token);
            } else {
                break;
            }
        }

        Ok(all_models)
    }
}

#[async_trait]
impl ModelListingCapability for GeminiModels {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        let models = self.fetch_all_models().await?;

        // Filter to only include generative models (exclude embedding models, etc.)
        let generative_models: Vec<ModelInfo> = models
            .into_iter()
            .filter(|model| {
                // Only include models that support generateContent
                model
                    .supported_generation_methods
                    .contains(&"generateContent".to_string())
            })
            .map(|model| self.convert_model(model))
            .collect();

        Ok(generative_models)
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        let ctx = super::context::build_context(&self.config).await;
        let spec = crate::providers::gemini::spec::GeminiSpec;
        let url = spec.try_model_url(&model_id, &ctx)?;
        let config = self.build_http_config(ctx);
        let result =
            crate::execution::executors::common::execute_get_request(&config, &url, None).await?;

        let model: GeminiModel = serde_json::from_value(result.json)
            .map_err(|e| LlmError::ParseError(format!("Failed to parse model response: {e}")))?;

        Ok(self.convert_model(model))
    }
}

/// Get default Gemini models
pub fn get_default_models() -> Vec<String> {
    vec![
        gemini_3::GEMINI_3_6_FLASH.to_string(),
        gemini_3::GEMINI_3_5_FLASH.to_string(),
        gemini_3::GEMINI_3_5_FLASH_LITE.to_string(),
        gemini_3::GEMINI_3_1_PRO_PREVIEW.to_string(),
        gemini_3::GEMINI_3_1_FLASH_LITE.to_string(),
        gemini_3::GEMINI_3_1_FLASH_IMAGE.to_string(),
        gemini_3::GEMINI_3_1_FLASH_LITE_IMAGE.to_string(),
        gemini_3::GEMINI_3_1_FLASH_LIVE_PREVIEW.to_string(),
        gemini_3::GEMINI_3_1_FLASH_TTS_PREVIEW.to_string(),
        gemini_2_5_pro::GEMINI_2_5_PRO.to_string(),
        gemini_2_5_flash::GEMINI_2_5_FLASH.to_string(),
        gemini_2_5_flash_lite::GEMINI_2_5_FLASH_LITE.to_string(),
    ]
}

/// Check if a model explicitly supports a capability.
///
/// Unknown models and unknown capability names conservatively return `false`. Use
/// [`model_capability_support`] when the caller needs to distinguish unknown from unsupported.
pub fn model_supports_capability(model_id: &str, capability: &str) -> bool {
    model_capability_support(model_id, capability) == CapabilitySupport::Supported
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;

    fn api_model(name: &str) -> GeminiModel {
        GeminiModel {
            name: name.to_string(),
            display_name: None,
            description: None,
            version: None,
            input_token_limit: None,
            output_token_limit: None,
            supported_generation_methods: vec!["generateContent".to_string()],
            temperature: None,
            top_p: None,
            top_k: None,
        }
    }

    #[test]
    fn default_models_track_current_google_model_ids() {
        let models = get_default_models();
        assert_eq!(models.first().map(String::as_str), Some("gemini-3.6-flash"));
        assert!(models.iter().any(|model| model == "gemini-3.5-flash-lite"));
        assert!(models.iter().any(|model| model == "gemini-3.1-flash-lite"));
        assert!(models.iter().any(|model| model == "gemini-3.1-flash-image"));
        assert!(!models.iter().any(|model| model == "gemini-3-flash-preview"));
        assert!(
            !models
                .iter()
                .any(|model| model == "gemini-3.1-flash-lite-preview")
        );
    }

    #[test]
    fn explicit_model_limits_do_not_invent_unknown_fallbacks() {
        assert_eq!(
            get_model_context_window(gemini_3::GEMINI_3_6_FLASH),
            Some(1_048_576)
        );
        assert_eq!(
            get_model_max_output_tokens(gemini_3::GEMINI_3_6_FLASH),
            Some(65_536)
        );
        assert_eq!(get_model_context_window("gemini-4-future"), None);
        assert_eq!(get_model_max_output_tokens("gemini-4-future"), None);
    }

    #[test]
    fn capability_queries_are_explicit_and_conservative() {
        assert!(model_supports_capability(
            gemini_3::GEMINI_3_6_FLASH,
            "chat"
        ));
        assert!(model_supports_capability(
            gemini_3::GEMINI_3_6_FLASH,
            "computer_use"
        ));
        assert!(!model_supports_capability(
            gemini_3::GEMINI_3_5_FLASH_LITE,
            "computer_use"
        ));
        assert_eq!(
            model_capability_support("gemini-4-future", "vision"),
            CapabilitySupport::Unknown
        );
    }

    #[test]
    fn api_conversion_uses_policy_only_for_exact_known_models() {
        let models = GeminiModels::new(GeminiConfig::new("test-key"), reqwest::Client::new());

        let known = models.convert_model(api_model("models/gemini-3.6-flash"));
        assert_eq!(known.context_window, Some(1_048_576));
        assert_eq!(known.max_output_tokens, Some(65_536));
        assert!(known.capabilities.iter().any(|value| value == "vision"));

        let unknown = models.convert_model(api_model("models/gemini-4-future"));
        assert_eq!(unknown.context_window, None);
        assert_eq!(unknown.max_output_tokens, None);
        assert_eq!(unknown.capabilities, vec!["chat"]);
    }
}

/// Get the documented context window for an exact audited model id.
pub fn get_model_context_window(model_id: &str) -> Option<u32> {
    model_policy(model_id).and_then(|policy| policy.context_window)
}

/// Get the documented maximum output size for an exact audited model id.
pub fn get_model_max_output_tokens(model_id: &str) -> Option<u32> {
    model_policy(model_id).and_then(|policy| policy.max_output_tokens)
}
