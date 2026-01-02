//! `OpenAI` Models API Implementation
//!
//! This module implements the `OpenAI` Models API for listing and retrieving
//! information about available models.
//!
//! API Reference: <https://platform.openai.com/docs/api-reference/models>

use async_trait::async_trait;
use secrecy::{ExposeSecret, SecretString};
use std::sync::Arc;

use crate::error::LlmError;
use crate::traits::ModelListingCapability;
use crate::types::{HttpConfig, ModelInfo};

use super::types::*;

/// `OpenAI` Models API client
#[derive(Clone)]
pub struct OpenAiModels {
    /// API key for authentication (securely stored)
    pub api_key: SecretString,
    /// Base URL for `OpenAI` API
    pub base_url: String,
    /// HTTP client
    pub http_client: reqwest::Client,
    /// Organization ID (optional)
    pub organization: Option<String>,
    /// Project ID (optional)
    pub project: Option<String>,
    /// HTTP configuration
    pub http_config: HttpConfig,
}

impl OpenAiModels {
    /// Create a new `OpenAI` models client
    pub fn new(
        api_key: SecretString,
        base_url: String,
        http_client: reqwest::Client,
        organization: Option<String>,
        project: Option<String>,
        http_config: HttpConfig,
    ) -> Self {
        Self {
            api_key,
            base_url,
            http_client,
            organization,
            project,
            http_config,
        }
    }

    fn build_http_config(&self) -> crate::execution::executors::common::HttpExecutionConfig {
        let spec = Arc::new(super::spec::OpenAiSpec::new());
        let ctx = crate::core::ProviderContext::new(
            "openai",
            self.base_url.clone(),
            Some(self.api_key.expose_secret().to_string()),
            self.http_config.headers.clone(),
        )
        .with_org_project(self.organization.clone(), self.project.clone());

        crate::execution::wiring::HttpExecutionWiring::new("openai", self.http_client.clone(), ctx)
            .config(spec)
    }

    /// Convert `OpenAI` model response to `ModelInfo`
    fn convert_openai_model_to_model_info(&self, openai_model: OpenAiModel) -> ModelInfo {
        convert_openai_model_to_model_info(openai_model)
    }

    /// Get models by capability
    pub async fn get_models_by_capability(
        &self,
        capability: &str,
    ) -> Result<Vec<ModelInfo>, LlmError> {
        let all_models = self.list_models().await?;
        Ok(all_models
            .into_iter()
            .filter(|model| model.capabilities.contains(&capability.to_string()))
            .collect())
    }

    /// Get chat models only
    pub async fn get_chat_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.get_models_by_capability("chat").await
    }

    /// Get image generation models only
    pub async fn get_image_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.get_models_by_capability("image_generation").await
    }

    /// Get audio models (TTS/STT)
    pub async fn get_audio_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.get_models_by_capability("audio").await
    }

    /// Get embedding models only
    pub async fn get_embedding_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.get_models_by_capability("embeddings").await
    }

    /// Get moderation models only
    pub async fn get_moderation_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.get_models_by_capability("moderation").await
    }

    /// Check if a model supports a specific capability
    pub async fn model_supports_capability(
        &self,
        model_id: &str,
        capability: &str,
    ) -> Result<bool, LlmError> {
        let model = self.get_model(model_id.to_string()).await?;
        Ok(model.capabilities.contains(&capability.to_string()))
    }

    /// Get recommended model for a specific use case
    pub fn get_recommended_model(&self, use_case: &str) -> String {
        match use_case {
            "chat" | "conversation" => "gpt-4o".to_string(),
            "chat_fast" | "quick_response" => "gpt-4o-mini".to_string(),
            "reasoning" | "complex_analysis" => "o1-preview".to_string(),
            "reasoning_fast" => "o1-mini".to_string(),
            "vision" | "image_analysis" => "gpt-4o".to_string(),
            "tts" | "text_to_speech" => "tts-1".to_string(),
            "tts_hd" | "high_quality_tts" => "tts-1-hd".to_string(),
            "tts_custom" | "custom_voice" => "gpt-4o-mini-tts".to_string(),
            "stt" | "speech_to_text" => "whisper-1".to_string(),
            "image_generation" => "dall-e-3".to_string(),
            "image_generation_fast" => "dall-e-2".to_string(),
            "image_generation_hd" => "gpt-image-1".to_string(),
            "embeddings" => "text-embedding-3-large".to_string(),
            "embeddings_fast" => "text-embedding-3-small".to_string(),
            "moderation" => "text-moderation-latest".to_string(),
            _ => "gpt-4o".to_string(), // Default fallback
        }
    }
}

#[async_trait]
impl ModelListingCapability for OpenAiModels {
    /// List all available models
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        let config = self.build_http_config();
        let url = config.provider_spec.models_url(&config.provider_context);
        let result =
            crate::execution::executors::common::execute_get_request(&config, &url, None).await?;

        let models_response: OpenAiModelsResponse =
            serde_json::from_value(result.json).map_err(|e| {
                LlmError::ParseError(format!("Failed to parse OpenAI models response: {e}"))
            })?;

        let models = models_response
            .data
            .into_iter()
            .map(|model| self.convert_openai_model_to_model_info(model))
            .collect();

        Ok(models)
    }

    /// Get information about a specific model
    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        let config = self.build_http_config();
        let url = config
            .provider_spec
            .model_url(&model_id, &config.provider_context);
        let result =
            crate::execution::executors::common::execute_get_request(&config, &url, None).await?;

        let openai_model: OpenAiModel = serde_json::from_value(result.json).map_err(|e| {
            LlmError::ParseError(format!("Failed to parse OpenAI model response: {e}"))
        })?;
        Ok(self.convert_openai_model_to_model_info(openai_model))
    }
}

pub(crate) fn convert_openai_model_to_model_info(openai_model: OpenAiModel) -> ModelInfo {
    let capabilities = determine_model_capabilities(&openai_model.id);
    let (context_window, max_output_tokens, input_cost, output_cost) =
        estimate_model_specs(&openai_model.id);

    ModelInfo {
        id: openai_model.id.clone(),
        name: Some(openai_model.id.clone()),
        description: Some(format!("OpenAI {} model", openai_model.id)),
        owned_by: openai_model.owned_by,
        created: openai_model.created,
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

    // GPT-4 models have vision capability
    if model_id.contains("gpt-4") && !model_id.contains("gpt-4-turbo-preview") {
        capabilities.push("vision".to_string());
    }

    // GPT-4.1 models have enhanced vision and reasoning
    if model_id.contains("gpt-4.1") {
        capabilities.push("vision".to_string());
        capabilities.push("advanced_reasoning".to_string());
    }

    // GPT-4.5 models have advanced capabilities
    if model_id.contains("gpt-4.5") {
        capabilities.push("vision".to_string());
        capabilities.push("advanced_reasoning".to_string());
        capabilities.push("multimodal".to_string());
    }

    // GPT-5 models have next-generation capabilities
    if model_id.contains("gpt-5") {
        capabilities.push("vision".to_string());
        capabilities.push("advanced_reasoning".to_string());
        capabilities.push("multimodal".to_string());
        capabilities.push("long_context".to_string());
    }

    // o1, o3, o4 models have reasoning capability
    if model_id.contains("o1") || model_id.contains("o3") || model_id.contains("o4") {
        capabilities.push("reasoning".to_string());
        capabilities.push("advanced_reasoning".to_string());
    }

    // Audio capabilities for specific models
    if model_id.contains("gpt-4o")
        || model_id.contains("gpt-4o-mini")
        || model_id.contains("audio-preview")
    {
        capabilities.push("audio".to_string());
    }

    // TTS models
    if model_id.contains("tts") || model_id == "gpt-4o-mini-tts" {
        capabilities.clear(); // TTS models don't have chat/text capabilities
        capabilities.push("tts".to_string());
        capabilities.push("audio".to_string());
    }

    // Whisper models
    if model_id.contains("whisper") {
        capabilities.clear(); // Whisper models don't have chat/text capabilities
        capabilities.push("stt".to_string());
        capabilities.push("audio".to_string());
        capabilities.push("transcription".to_string());
        capabilities.push("translation".to_string());
    }

    // DALL-E models
    if model_id.contains("dall-e") || model_id == "gpt-image-1" {
        capabilities.clear(); // Image models don't have chat/text capabilities
        capabilities.push("image_generation".to_string());
    }

    // Embedding models
    if model_id.contains("embedding") || model_id.contains("ada") {
        capabilities.clear(); // Embedding models don't have chat/text capabilities
        capabilities.push("embeddings".to_string());
        capabilities.push("text".to_string());
    }

    // Moderation models
    if model_id.contains("moderation") {
        capabilities.clear(); // Moderation models don't have chat/text capabilities
        capabilities.push("moderation".to_string());
        capabilities.push("text".to_string());
    }

    // All modern chat models support tools (except reasoning models which have limited tool support)
    if capabilities.contains(&"chat".to_string())
        && !model_id.contains("o1")
        && !model_id.contains("o3")
        && !model_id.contains("o4")
        && (!model_id.contains("gpt-3.5") || model_id.contains("gpt-3.5-turbo"))
    {
        capabilities.push("tools".to_string());
    }

    // Streaming support for chat models (reasoning models don't support streaming)
    if capabilities.contains(&"chat".to_string())
        && !model_id.contains("o1")
        && !model_id.contains("o3")
        && !model_id.contains("o4")
    {
        capabilities.push("streaming".to_string());
    }

    capabilities
}

/// Estimate model specifications based on model ID
fn estimate_model_specs(model_id: &str) -> (Option<u32>, Option<u32>, Option<f64>, Option<f64>) {
    match model_id {
        // GPT-4o models
        "gpt-4o" => (
            Some(128_000),
            Some(16_384),
            Some(0.000_002_5),
            Some(0.000_01),
        ),
        "gpt-4o-mini" => (
            Some(128_000),
            Some(16_384),
            Some(0.000_000_15),
            Some(0.000_000_6),
        ),
        "gpt-4o-audio-preview"
        | "gpt-4o-audio-preview-2024-12-17"
        | "gpt-4o-audio-preview-2024-10-01" => (
            Some(128_000),
            Some(16_384),
            Some(0.000_002_5),
            Some(0.000_01),
        ),
        "gpt-4o-mini-audio-preview" | "gpt-4o-mini-audio-preview-2024-12-17" => (
            Some(128_000),
            Some(16_384),
            Some(0.000_000_15),
            Some(0.000_000_6),
        ),
        "gpt-4o-mini-tts" => (None, None, Some(0.000_015), None), // TTS pricing per character

        // GPT-4.1 models (new)
        "gpt-4.1" => (
            Some(200_000),
            Some(32_768),
            Some(0.000_003),
            Some(0.000_015),
        ),
        "gpt-4.1-mini" => (
            Some(200_000),
            Some(32_768),
            Some(0.000_000_2),
            Some(0.000_000_8),
        ),
        "gpt-4.1-nano" => (
            Some(128_000),
            Some(16_384),
            Some(0.000_000_1),
            Some(0.000_000_4),
        ),

        // GPT-4.5 models (new)
        "gpt-4.5-preview-2025-02-27" | "gpt-4.5-preview" | "gpt-4.5" => (
            Some(200_000),
            Some(32_768),
            Some(0.000_005),
            Some(0.000_025),
        ),

        // GPT-4 Turbo models
        id if id.contains("gpt-4-turbo") => {
            (Some(128_000), Some(4096), Some(0.000_01), Some(0.000_03))
        }

        // GPT-4 models
        "gpt-4" => (Some(8192), Some(4096), Some(0.000_03), Some(0.000_06)),
        "gpt-4-32k" => (Some(32_768), Some(4096), Some(0.000_06), Some(0.000_12)),

        // o1 models (reasoning models)
        "o1" | "o1-2024-12-17" => (
            Some(200_000),
            Some(100_000),
            Some(0.000_015),
            Some(0.000_06),
        ),
        "o1-preview" => (Some(128_000), Some(32_768), Some(0.000_015), Some(0.000_06)),
        "o1-mini" => (
            Some(128_000),
            Some(65_536),
            Some(0.000_003),
            Some(0.000_012),
        ),

        // o3 models (new reasoning models)
        "o3-mini" => (
            Some(200_000),
            Some(65_536),
            Some(0.000_004),
            Some(0.000_016),
        ),
        "o3" => (Some(200_000), Some(100_000), Some(0.000_02), Some(0.000_08)),

        // o4 models (new)
        "o4-mini" => (Some(200_000), Some(65_536), Some(0.000_005), Some(0.000_02)),

        // GPT-5 models (new)
        "gpt-5" | "gpt-5-2025-08-07" => (
            Some(1_000_000),
            Some(100_000),
            Some(0.000_01),
            Some(0.000_05),
        ),
        "gpt-5-mini" | "gpt-5-mini-2025-08-07" => {
            (Some(500_000), Some(50_000), Some(0.000_002), Some(0.000_01))
        }
        "gpt-5-nano" | "gpt-5-nano-2025-08-07" => (
            Some(200_000),
            Some(32_768),
            Some(0.000_001),
            Some(0.000_005),
        ),

        // GPT-3.5 Turbo models
        "gpt-3.5-turbo" => (
            Some(16_385),
            Some(4096),
            Some(0.000_000_5),
            Some(0.000_001_5),
        ),
        "gpt-3.5-turbo-16k" => (Some(16_385), Some(4096), Some(0.000_003), Some(0.000_004)),

        // TTS models
        "tts-1" => (None, None, Some(0.000_015), None), // Per character
        "tts-1-hd" => (None, None, Some(0.00003), None), // Per character

        // Whisper models
        "whisper-1" => (None, None, Some(0.006), None), // Per minute

        // DALL-E models
        "dall-e-2" => (None, None, Some(0.02), None), // Per image (1024x1024)
        "dall-e-3" => (None, None, Some(0.04), None), // Per image (1024x1024)
        "gpt-image-1" => (None, None, Some(0.03), None), // Per image (estimated)

        // Embedding models
        "text-embedding-3-small" => (Some(8191), None, Some(0.000_000_02), None),
        "text-embedding-3-large" => (Some(8191), None, Some(0.000_000_13), None),
        "text-embedding-ada-002" => (Some(8191), None, Some(0.000_000_1), None),

        // Moderation models
        id if id.contains("text-moderation") => (Some(32_768), None, Some(0.0), None), // Free

        // Default fallback for unknown models
        _ => (Some(4096), Some(2048), Some(0.00001), Some(0.00003)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::openai::OpenAiConfig;

    #[test]
    fn test_models_endpoint() {
        let config = OpenAiConfig::new("test-key");
        let models = OpenAiModels::new(
            config.api_key.clone(),
            config.base_url.clone(),
            reqwest::Client::new(),
            config.organization.clone(),
            config.project.clone(),
            config.http_config.clone(),
        );
        let cfg = models.build_http_config();
        assert_eq!(
            cfg.provider_spec.models_url(&cfg.provider_context),
            "https://api.openai.com/v1/models"
        );
        assert_eq!(
            cfg.provider_spec.model_url("gpt-4", &cfg.provider_context),
            "https://api.openai.com/v1/models/gpt-4"
        );
    }

    #[test]
    fn test_determine_model_capabilities() {
        let gpt4_caps = determine_model_capabilities("gpt-4");
        assert!(gpt4_caps.contains(&"vision".to_string()));
        assert!(gpt4_caps.contains(&"tools".to_string()));

        let o1_caps = determine_model_capabilities("o1-preview");
        assert!(o1_caps.contains(&"reasoning".to_string()));

        let gpt35_caps = determine_model_capabilities("gpt-3.5-turbo");
        assert!(gpt35_caps.contains(&"tools".to_string()));
    }

    #[test]
    fn test_estimate_model_specs() {
        let (context, max_output, input_cost, output_cost) = estimate_model_specs("gpt-4o");
        assert_eq!(context, Some(128_000));
        assert_eq!(max_output, Some(16_384));
        assert!(input_cost.is_some());
        assert!(output_cost.is_some());
    }
}
