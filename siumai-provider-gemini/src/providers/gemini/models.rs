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
        crate::execution::wiring::HttpExecutionWiring::new("gemini", self.http_client.clone(), ctx)
            .config(Arc::new(crate::providers::gemini::spec::GeminiSpec))
    }

    /// Convert `GeminiModel` to `ModelInfo`
    fn convert_model(&self, model: GeminiModel) -> ModelInfo {
        // Extract model ID from the full name (e.g., "models/gemini-1.5-flash" -> "gemini-1.5-flash")
        let id = model
            .name
            .strip_prefix("models/")
            .unwrap_or(&model.name)
            .to_string();

        // Determine capabilities based on model name and supported generation methods
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

        // Most Gemini models support these features
        if id.contains("gemini") {
            capabilities.extend_from_slice(&[
                "vision".to_string(),
                "function_calling".to_string(),
                "code_execution".to_string(),
            ]);
        }

        // Determine context window (prefer API); and max output tokens
        let context_window: u32 = model
            .input_token_limit
            .map(|t| t as u32)
            .unwrap_or_else(|| get_model_context_window(&id));
        let max_output_tokens: u32 = model
            .output_token_limit
            .map(|t| t as u32)
            .unwrap_or_else(|| get_model_max_output_tokens(&id));

        ModelInfo {
            id,
            name: Some(model.display_name.unwrap_or(model.name)),
            description: model.description,
            context_window: Some(context_window),
            // Prefer API-provided output limit; otherwise use curated mapping
            max_output_tokens: Some(max_output_tokens),
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
            let mut url = spec.models_url(&ctx);

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
        let url = spec.model_url(&model_id, &ctx);
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
        // Latest Gemini 2.5 models
        "gemini-2.5-pro".to_string(),
        "gemini-2.5-pro-exp-03-25".to_string(),
        "gemini-2.5-flash".to_string(),
        "gemini-2.5-flash-lite".to_string(),
        // Gemini 2.0 models
        "gemini-2.0-pro-exp-02-05".to_string(),
        "gemini-2.0-flash".to_string(),
        "gemini-2.0-flash-001".to_string(),
        "gemini-2.0-flash-exp".to_string(),
        "gemini-2.0-flash-thinking-exp-01-21".to_string(),
        "gemini-2.0-flash-lite".to_string(),
        // Legacy models (deprecated but still available)
        "gemini-1.5-flash".to_string(),
        "gemini-1.5-flash-001".to_string(),
        "gemini-1.5-flash-002".to_string(),
        "gemini-1.5-flash-8b".to_string(),
        "gemini-1.5-pro".to_string(),
        "gemini-1.5-pro-001".to_string(),
        "gemini-1.5-pro-002".to_string(),
        // LearnLM
        "learnlm-1.5-pro-experimental".to_string(),
    ]
}

/// Check if a model supports a specific capability
pub fn model_supports_capability(model_id: &str, capability: &str) -> bool {
    match capability {
        "chat" => true,                                    // All Gemini models support chat
        "streaming" => true,                               // All Gemini models support streaming
        "vision" => model_id.contains("gemini"),           // Most Gemini models support vision
        "function_calling" => model_id.contains("gemini"), // Most Gemini models support function calling
        "code_execution" => model_id.contains("gemini"), // Most Gemini models support code execution
        "thinking" => {
            model_id.contains("gemini-2.5")
                || model_id.contains("gemini-2.0")
                || model_id.contains("exp")
        } // 2.5+ models support thinking
        "audio_generation" => {
            model_id.contains("tts")
                || model_id.contains("live")
                || model_id.contains("native-audio")
        } // Audio models
        "image_generation" => model_id.contains("image-generation"), // Image generation models
        "live_api" => model_id.contains("live"),         // Live API models
        _ => false,
    }
}

/// Get the context window size for a model
pub fn get_model_context_window(model_id: &str) -> u32 {
    if model_id.contains("2.5-pro")
        || model_id.contains("2.5-flash")
        || model_id.contains("2.0-flash")
    {
        1_048_576 // 1M tokens for Gemini 2.5 Pro, 2.5 Flash and 2.0 Flash
    } else if model_id.contains("2.0-pro") || model_id.contains("1.5-pro") {
        2_097_152 // 2M tokens for Gemini 2.0 Pro / 1.5 Pro
    } else if model_id.contains("1.5-flash") {
        1_048_576 // 1M tokens for Gemini 1.5 Flash
    } else {
        128_000 // Default fallback
    }
}

/// Get the maximum output tokens for a model
pub fn get_model_max_output_tokens(model_id: &str) -> u32 {
    if model_id.contains("2.5-pro") || model_id.contains("2.5-flash") {
        65_536 // Gemini 2.5 Pro and Flash max output
    } else if model_id.contains("2.0-flash")
        || model_id.contains("1.5-pro")
        || model_id.contains("1.5-flash")
        || model_id.contains("2.0-pro")
    {
        8192 // Gemini 2.0 Flash/Pro, 1.5 Pro and Flash max output
    } else if model_id.contains("tts") {
        16_000 // TTS models have different output limits
    } else {
        8192 // Default fallback
    }
}
