//! Gemini Models Capability Implementation
//!
//! This module implements model listing functionality for Google Gemini API.

use async_trait::async_trait;
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};

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

        // Determine context window
        let context_window = model.input_token_limit.unwrap_or_else(|| {
            // Default context windows based on model name
            if id.contains("1.5-pro") {
                2_000_000 // 2M tokens for Gemini 1.5 Pro
            } else if id.contains("1.5-flash") || id.contains("2.0") {
                1_000_000 // 1M tokens for Gemini 1.5 Flash and 2.0
            } else {
                32_000 // Default fallback
            }
        });

        ModelInfo {
            id,
            name: Some(model.display_name.unwrap_or(model.name)),
            description: model.description,
            context_window: Some(context_window as u32),
            max_output_tokens: model.output_token_limit.map(|t| t as u32),
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
            let mut url = format!("{}/models", self.config.base_url);

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

            // Allow Authorization-based (Bearer) Vertex AI auth via custom headers; otherwise use x-goog-api-key
            let mut extra = self.config.http_config.headers.clone();
            if let Some(ref tp) = self.config.token_provider {
                if let Ok(tok) = tp.token().await {
                    extra.insert("Authorization".to_string(), format!("Bearer {tok}"));
                }
            }
            use secrecy::ExposeSecret;
            let headers = crate::utils::http_headers::ProviderHeaders::gemini(
                self.config.api_key.expose_secret(),
                &extra,
            )?;

            let response = self
                .http_client
                .get(&url)
                .headers(headers)
                .send()
                .await
                .map_err(|e| LlmError::HttpError(e.to_string()))?;

            if !response.status().is_success() {
                let status_code = response.status().as_u16();
                let error_text = response.text().await.unwrap_or_default();
                return Err(LlmError::api_error(
                    status_code,
                    format!("Gemini API error: {status_code} - {error_text}"),
                ));
            }

            let list_response: ListModelsResponse = response.json().await.map_err(|e| {
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
        // Ensure the model ID has the proper prefix
        let full_model_name = if model_id.starts_with("models/") {
            model_id
        } else {
            format!("models/{model_id}")
        };

        let url = crate::utils::url::join_url(&self.config.base_url, &full_model_name);

        // Allow Authorization-based (Bearer) Vertex AI auth via custom headers; otherwise use x-goog-api-key
        let mut extra = self.config.http_config.headers.clone();
        if let Some(ref tp) = self.config.token_provider {
            if let Ok(tok) = tp.token().await {
                extra.insert("Authorization".to_string(), format!("Bearer {tok}"));
            }
        }
        use secrecy::ExposeSecret;
        let headers = crate::utils::http_headers::ProviderHeaders::gemini(
            self.config.api_key.expose_secret(),
            &extra,
        )?;

        let response = self
            .http_client
            .get(&url)
            .headers(headers)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::api_error(
                status_code,
                format!("Gemini API error: {status_code} - {error_text}"),
            ));
        }

        let model: GeminiModel = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(format!("Failed to parse model response: {e}")))?;

        Ok(self.convert_model(model))
    }
}

/// Get default Gemini models
pub fn get_default_models() -> Vec<String> {
    vec![
        // Latest Gemini 2.5 models
        "gemini-2.5-pro".to_string(),
        "gemini-2.5-flash".to_string(),
        "gemini-2.5-flash-lite".to_string(),
        // Gemini 2.0 models
        "gemini-2.0-flash".to_string(),
        "gemini-2.0-flash-lite".to_string(),
        // Legacy models (deprecated but still available)
        "gemini-1.5-flash".to_string(),
        "gemini-1.5-flash-8b".to_string(),
        "gemini-1.5-pro".to_string(),
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
    } else if model_id.contains("1.5-pro") {
        2_097_152 // 2M tokens for Gemini 1.5 Pro
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
    {
        8192 // Gemini 2.0 Flash, 1.5 Pro and Flash max output
    } else if model_id.contains("tts") {
        16_000 // TTS models have different output limits
    } else {
        8192 // Default fallback
    }
}
