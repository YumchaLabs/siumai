//! Anthropic Models API Implementation
//!
//! Implements model listing functionality according to the official Anthropic API documentation:
//! <https://platform.claude.com/docs/en/api/models/list>

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use secrecy::SecretString;
use std::sync::Arc;

use crate::error::LlmError;
use crate::execution::http::transport::HttpTransport;
use crate::traits::ModelListingCapability;
use crate::types::ModelInfo;

use super::model_constants::{
    CapabilitySupport, ThinkingDisablePolicy, ThinkingSupport, claude_fable_5, claude_mythos_5,
    claude_opus_3, claude_opus_4, claude_opus_4_1, claude_opus_4_5, claude_opus_4_6,
    claude_opus_4_7, claude_opus_5, claude_sonnet_3, claude_sonnet_3_5, claude_sonnet_3_7,
    claude_sonnet_4, claude_sonnet_4_5, claude_sonnet_4_6, claude_sonnet_5, model_profile,
};
use super::types::{AnthropicModelInfo, AnthropicModelsResponse};
use super::utils::map_anthropic_error;

/// Anthropic Models API implementation
#[derive(Clone)]
pub struct AnthropicModels {
    pub api_key: SecretString,
    pub base_url: String,
    pub http_client: reqwest::Client,
    pub http_config: crate::types::HttpConfig,
    pub http_transport: Option<Arc<dyn HttpTransport>>,
}

impl AnthropicModels {
    /// Create a new Anthropic models instance
    pub fn new(
        api_key: SecretString,
        base_url: String,
        http_client: reqwest::Client,
        http_config: crate::types::HttpConfig,
    ) -> Self {
        Self {
            api_key,
            base_url,
            http_client,
            http_config,
            http_transport: None,
        }
    }

    fn build_context(&self) -> crate::core::ProviderContext {
        use secrecy::ExposeSecret;
        crate::core::ProviderContext::new(
            "anthropic",
            self.base_url.clone(),
            Some(self.api_key.expose_secret().to_string()),
            self.http_config.headers.clone(),
        )
    }

    fn build_http_config(
        &self,
        ctx: crate::core::ProviderContext,
    ) -> crate::execution::executors::common::HttpExecutionConfig {
        let mut wiring = crate::execution::wiring::HttpExecutionWiring::new(
            "anthropic",
            self.http_client.clone(),
            ctx,
        );
        if let Some(transport) = self.http_transport.clone() {
            wiring = wiring.with_transport(transport);
        }
        wiring.config(Arc::new(super::spec::AnthropicSpec::new()))
    }

    fn remap_anthropic_api_error(&self, err: LlmError) -> LlmError {
        let (code, details) = match &err {
            LlmError::ApiError { code, details, .. } => (*code, details.as_ref()),
            _ => return err,
        };
        let details = match details {
            Some(d) => d,
            None => return err,
        };
        let response = match details.get("response") {
            Some(v) => v,
            None => return err,
        };
        let error_obj = match response.get("error") {
            Some(v) => v,
            None => return err,
        };
        let error_type = error_obj
            .get("type")
            .and_then(|t| t.as_str())
            .unwrap_or("unknown");
        let error_message = error_obj
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("Unknown error");

        map_anthropic_error(code, error_type, error_message, response.clone())
    }

    /// List models with pagination support
    pub async fn list_models_paginated(
        &self,
        before_id: Option<String>,
        after_id: Option<String>,
        limit: Option<u32>,
    ) -> Result<AnthropicModelsResponse, LlmError> {
        let ctx = self.build_context();
        let config = self.build_http_config(ctx);
        let mut url = config
            .provider_spec
            .try_models_url(&config.provider_context)?;

        // Build query parameters
        let mut query_params = Vec::new();
        if let Some(before) = before_id {
            query_params.push(format!("before_id={before}"));
        }
        if let Some(after) = after_id {
            query_params.push(format!("after_id={after}"));
        }
        if let Some(limit_val) = limit {
            query_params.push(format!("limit={limit_val}"));
        }

        if !query_params.is_empty() {
            url.push('?');
            url.push_str(&query_params.join("&"));
        }

        let result = crate::execution::executors::common::execute_get_request(&config, &url, None)
            .await
            .map_err(|e| self.remap_anthropic_api_error(e))?;
        let anthropic_response: AnthropicModelsResponse = serde_json::from_value(result.json)
            .map_err(|e| {
                LlmError::ParseError(format!("Failed to parse Anthropic models response: {e}"))
            })?;
        Ok(anthropic_response)
    }

    /// Get information about a specific model
    pub async fn get_model_info(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        let ctx = self.build_context();
        let config = self.build_http_config(ctx);
        let url = config
            .provider_spec
            .try_model_url(&model_id, &config.provider_context)?;

        let result = crate::execution::executors::common::execute_get_request(&config, &url, None)
            .await
            .map_err(|e| self.remap_anthropic_api_error(e))?;

        let anthropic_model: AnthropicModelInfo =
            serde_json::from_value(result.json).map_err(|e| {
                LlmError::ParseError(format!("Failed to parse Anthropic model response: {e}"))
            })?;
        Ok(convert_anthropic_model_to_model_info(anthropic_model))
    }
}

#[async_trait]
impl ModelListingCapability for AnthropicModels {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        let mut all_models = Vec::new();
        let mut after_id: Option<String> = None;

        // Fetch all models with pagination
        loop {
            let response = self
                .list_models_paginated(None, after_id, Some(100))
                .await?;

            for model in response.data {
                all_models.push(convert_anthropic_model_to_model_info(model));
            }

            if !response.has_more {
                break;
            }

            after_id = response.last_id;
        }

        Ok(all_models)
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        self.get_model_info(model_id).await
    }
}

/// Convert Anthropic model info to our `ModelInfo` structure
fn convert_anthropic_model_to_model_info(anthropic_model: AnthropicModelInfo) -> ModelInfo {
    // Parse creation date
    let created = anthropic_model
        .created_at
        .parse::<DateTime<Utc>>()
        .map(|dt| dt.timestamp() as u64)
        .ok();

    // Determine capabilities based on model ID
    let capabilities = determine_model_capabilities(&anthropic_model.id);

    // Estimate context window and costs based on model type
    let (context_window, max_output_tokens, input_cost, output_cost) =
        estimate_model_specs(&anthropic_model.id);

    ModelInfo {
        id: anthropic_model.id.clone(),
        name: Some(anthropic_model.display_name),
        description: Some(format!("Anthropic {} model", anthropic_model.id)),
        owned_by: "anthropic".to_string(),
        created,
        capabilities,
        context_window,
        max_output_tokens,
        input_cost_per_token: input_cost,
        output_cost_per_token: output_cost,
    }
}

/// Determine model capabilities based on model ID
fn determine_model_capabilities(model_id: &str) -> Vec<String> {
    let mut capabilities = vec!["chat".to_string(), "text".to_string()];
    let Some(profile) = model_profile(model_id) else {
        return capabilities;
    };

    if !matches!(profile.thinking, ThinkingSupport::Unsupported) {
        capabilities.push("thinking".to_string());
        match profile.thinking {
            ThinkingSupport::AdaptiveOnly | ThinkingSupport::AdaptiveAndManual => {
                capabilities.push("adaptive_thinking".to_string());
            }
            ThinkingSupport::Manual | ThinkingSupport::Unsupported => {}
        }
        match profile.thinking {
            ThinkingSupport::Manual | ThinkingSupport::AdaptiveAndManual => {
                capabilities.push("extended_thinking".to_string());
            }
            ThinkingSupport::AdaptiveOnly | ThinkingSupport::Unsupported => {}
        }
        if profile.thinking_disable == ThinkingDisablePolicy::Forbidden {
            capabilities.push("always_thinking".to_string());
        }
    }

    if profile.supports_vision {
        capabilities.push("vision".to_string());
        capabilities.push("multimodal".to_string());
    }

    if profile.supports_tools {
        capabilities.push("tools".to_string());
        capabilities.push("function_calling".to_string());
    }

    if profile.supports_streaming {
        capabilities.push("streaming".to_string());
    }

    if profile.supports_prompt_caching {
        capabilities.push("prompt_caching".to_string());
    }

    if profile.supports_prefill == CapabilitySupport::Supported {
        capabilities.push("prefill".to_string());
    }

    if profile.supports_priority_tier == CapabilitySupport::Supported {
        capabilities.push("priority_tier".to_string());
    }

    capabilities
}

/// Estimate model specifications based on model ID
fn estimate_model_specs(model_id: &str) -> (Option<u32>, Option<u32>, Option<f64>, Option<f64>) {
    let Some(profile) = model_profile(model_id) else {
        return (None, None, None, None);
    };
    let (input_cost, output_cost) = estimate_model_costs(model_id);

    (
        Some(profile.context_window),
        Some(profile.max_output_tokens),
        input_cost,
        output_cost,
    )
}

fn in_family(model_id: &str, family: &[&str]) -> bool {
    family.contains(&model_id)
}

fn estimate_model_costs(model_id: &str) -> (Option<f64>, Option<f64>) {
    if in_family(model_id, claude_opus_5::ALL) {
        return (Some(0.000_005), Some(0.000_025));
    }
    if in_family(model_id, claude_sonnet_5::ALL) {
        return (Some(0.000_003), Some(0.000_015));
    }
    if in_family(model_id, claude_fable_5::ALL) || in_family(model_id, claude_mythos_5::ALL) {
        return (Some(0.000_010), Some(0.000_050));
    }
    if in_family(model_id, claude_opus_4_7::ALL)
        || in_family(model_id, claude_opus_4_6::ALL)
        || in_family(model_id, claude_opus_4_5::ALL)
        || in_family(model_id, claude_opus_4_1::ALL)
        || in_family(model_id, claude_opus_4::ALL)
        || in_family(model_id, claude_opus_3::ALL)
    {
        return (Some(0.000_015), Some(0.000_075));
    }
    if in_family(model_id, claude_sonnet_4_6::ALL)
        || in_family(model_id, claude_sonnet_4_5::ALL)
        || in_family(model_id, claude_sonnet_4::ALL)
        || in_family(model_id, claude_sonnet_3_7::ALL)
        || in_family(model_id, claude_sonnet_3_5::ALL)
        || in_family(model_id, claude_sonnet_3::ALL)
    {
        return (Some(0.000_003), Some(0.000_015));
    }

    (None, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_capabilities() {
        let caps = determine_model_capabilities(claude_fable_5::CLAUDE_FABLE_5);
        assert!(caps.contains(&"thinking".to_string()));
        assert!(caps.contains(&"adaptive_thinking".to_string()));
        assert!(caps.contains(&"always_thinking".to_string()));
        assert!(caps.contains(&"vision".to_string()));
        assert!(caps.contains(&"tools".to_string()));
        assert!(!caps.contains(&"priority_tier".to_string()));
    }

    #[test]
    fn test_model_specs() {
        let (context, max_output, input_cost, output_cost) =
            estimate_model_specs("claude-3-5-sonnet-20241022");
        assert_eq!(context, Some(200_000));
        assert_eq!(max_output, Some(8192));
        assert!(input_cost.is_some());
        assert!(output_cost.is_some());
    }

    #[test]
    fn test_convert_anthropic_model() {
        let anthropic_model = AnthropicModelInfo {
            id: "claude-3-5-sonnet-20241022".to_string(),
            display_name: "Claude 3.5 Sonnet".to_string(),
            created_at: "2024-10-22T00:00:00Z".to_string(),
            r#type: "model".to_string(),
        };

        let model_info = convert_anthropic_model_to_model_info(anthropic_model);
        assert_eq!(model_info.id, "claude-3-5-sonnet-20241022");
        assert_eq!(model_info.name, Some("Claude 3.5 Sonnet".to_string()));
        assert_eq!(model_info.owned_by, "anthropic");
        assert!(model_info.capabilities.contains(&"vision".to_string()));
    }

    #[test]
    fn current_model_specs_use_verified_limits() {
        let (context, max_output, input_cost, output_cost) =
            estimate_model_specs(claude_opus_5::CLAUDE_OPUS_5);
        assert_eq!(context, Some(1_000_000));
        assert_eq!(max_output, Some(128_000));
        assert_eq!(input_cost, Some(0.000_005));
        assert_eq!(output_cost, Some(0.000_025));

        let (_, _, input_cost, output_cost) = estimate_model_specs(claude_fable_5::CLAUDE_FABLE_5);
        assert_eq!(input_cost, Some(0.000_010));
        assert_eq!(output_cost, Some(0.000_050));
    }

    #[test]
    fn unknown_model_specs_and_optional_capabilities_stay_unset() {
        let model_id = "claude-future-custom-model";
        assert_eq!(estimate_model_specs(model_id), (None, None, None, None));
        assert_eq!(
            determine_model_capabilities(model_id),
            vec!["chat".to_string(), "text".to_string()]
        );
    }
}
