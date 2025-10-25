//! `OpenAI` Moderation API Implementation
//!
//! This module provides the `OpenAI` implementation of the `ModerationCapability` trait,
//! including content moderation for text and image content.

use async_trait::async_trait;
use serde::Deserialize;
use std::collections::HashMap;

use crate::error::LlmError;
use crate::execution::transformers::request::RequestTransformer;
use crate::traits::ModerationCapability;
use crate::types::{ModerationRequest, ModerationResponse, ModerationResult};

use super::config::OpenAiConfig;
use secrecy::ExposeSecret;

/// `OpenAI` moderation API response structure
#[derive(Debug, Clone, Deserialize)]
struct OpenAiModerationResponse {
    /// Unique identifier for the moderation request
    #[allow(dead_code)]
    id: String,
    /// Model used for moderation
    model: String,
    /// List of moderation results
    results: Vec<OpenAiModerationResult>,
}

/// Individual moderation result from `OpenAI`
#[derive(Debug, Clone, Deserialize)]
struct OpenAiModerationResult {
    /// Whether the content was flagged
    flagged: bool,
    /// Category flags
    categories: OpenAiModerationCategories,
    /// Category confidence scores
    category_scores: OpenAiModerationCategoryScores,
}

/// `OpenAI` moderation categories (boolean flags)
#[derive(Debug, Clone, Deserialize)]
struct OpenAiModerationCategories {
    /// Hate speech
    hate: bool,
    /// Hate speech with threatening content
    #[serde(rename = "hate/threatening")]
    hate_threatening: bool,
    /// Harassment
    harassment: bool,
    /// Harassment with threatening content
    #[serde(rename = "harassment/threatening")]
    harassment_threatening: bool,
    /// Self-harm content
    #[serde(rename = "self-harm")]
    self_harm: bool,
    /// Self-harm intent
    #[serde(rename = "self-harm/intent")]
    self_harm_intent: bool,
    /// Self-harm instructions
    #[serde(rename = "self-harm/instructions")]
    self_harm_instructions: bool,
    /// Sexual content
    sexual: bool,
    /// Sexual content involving minors
    #[serde(rename = "sexual/minors")]
    sexual_minors: bool,
    /// Violence
    violence: bool,
    /// Graphic violence
    #[serde(rename = "violence/graphic")]
    violence_graphic: bool,
}

/// `OpenAI` moderation category scores (confidence values)
#[derive(Debug, Clone, Deserialize)]
struct OpenAiModerationCategoryScores {
    /// Hate speech score
    hate: f32,
    /// Hate speech with threatening content score
    #[serde(rename = "hate/threatening")]
    hate_threatening: f32,
    /// Harassment score
    harassment: f32,
    /// Harassment with threatening content score
    #[serde(rename = "harassment/threatening")]
    harassment_threatening: f32,
    /// Self-harm content score
    #[serde(rename = "self-harm")]
    self_harm: f32,
    /// Self-harm intent score
    #[serde(rename = "self-harm/intent")]
    self_harm_intent: f32,
    /// Self-harm instructions score
    #[serde(rename = "self-harm/instructions")]
    self_harm_instructions: f32,
    /// Sexual content score
    sexual: f32,
    /// Sexual content involving minors score
    #[serde(rename = "sexual/minors")]
    sexual_minors: f32,
    /// Violence score
    violence: f32,
    /// Graphic violence score
    #[serde(rename = "violence/graphic")]
    violence_graphic: f32,
}

/// `OpenAI` moderation capability implementation.
///
/// This struct provides the OpenAI-specific implementation of content moderation
/// using the `OpenAI` Moderation API.
///
/// # Supported Features
/// - Text content moderation
/// - Multiple moderation models (text-moderation-stable, text-moderation-latest)
/// - Comprehensive category detection (hate, harassment, self-harm, sexual, violence)
/// - Confidence scores for each category
/// - Batch processing support
///
/// # API Reference
/// <https://platform.openai.com/docs/api-reference/moderations>
#[derive(Debug, Clone)]
pub struct OpenAiModeration {
    /// `OpenAI` configuration
    config: OpenAiConfig,
    /// HTTP client
    http_client: reqwest::Client,
}

impl OpenAiModeration {
    /// Create a new `OpenAI` moderation instance.
    ///
    /// # Arguments
    /// * `config` - `OpenAI` configuration
    /// * `http_client` - HTTP client for making requests
    pub const fn new(config: OpenAiConfig, http_client: reqwest::Client) -> Self {
        Self {
            config,
            http_client,
        }
    }

    /// Get supported moderation models.
    pub fn get_supported_models(&self) -> Vec<String> {
        vec![
            "text-moderation-stable".to_string(),
            "text-moderation-latest".to_string(),
        ]
    }

    /// Get the default moderation model.
    pub fn default_model(&self) -> String {
        "text-moderation-latest".to_string()
    }

    // Removed legacy validation; stable checks are performed in `moderate()`

    /// Convert `OpenAI` categories to our standard format.
    fn convert_categories(&self, categories: &OpenAiModerationCategories) -> HashMap<String, bool> {
        let mut result = HashMap::new();
        result.insert("hate".to_string(), categories.hate);
        result.insert("hate/threatening".to_string(), categories.hate_threatening);
        result.insert("harassment".to_string(), categories.harassment);
        result.insert(
            "harassment/threatening".to_string(),
            categories.harassment_threatening,
        );
        result.insert("self-harm".to_string(), categories.self_harm);
        result.insert("self-harm/intent".to_string(), categories.self_harm_intent);
        result.insert(
            "self-harm/instructions".to_string(),
            categories.self_harm_instructions,
        );
        result.insert("sexual".to_string(), categories.sexual);
        result.insert("sexual/minors".to_string(), categories.sexual_minors);
        result.insert("violence".to_string(), categories.violence);
        result.insert("violence/graphic".to_string(), categories.violence_graphic);
        result
    }

    /// Convert `OpenAI` category scores to our standard format.
    fn convert_category_scores(
        &self,
        scores: &OpenAiModerationCategoryScores,
    ) -> HashMap<String, f32> {
        let mut result = HashMap::new();
        result.insert("hate".to_string(), scores.hate);
        result.insert("hate/threatening".to_string(), scores.hate_threatening);
        result.insert("harassment".to_string(), scores.harassment);
        result.insert(
            "harassment/threatening".to_string(),
            scores.harassment_threatening,
        );
        result.insert("self-harm".to_string(), scores.self_harm);
        result.insert("self-harm/intent".to_string(), scores.self_harm_intent);
        result.insert(
            "self-harm/instructions".to_string(),
            scores.self_harm_instructions,
        );
        result.insert("sexual".to_string(), scores.sexual);
        result.insert("sexual/minors".to_string(), scores.sexual_minors);
        result.insert("violence".to_string(), scores.violence);
        result.insert("violence/graphic".to_string(), scores.violence_graphic);
        result
    }

    /// Convert `OpenAI` moderation result to our standard format.
    fn convert_result(&self, openai_result: OpenAiModerationResult) -> ModerationResult {
        ModerationResult {
            flagged: openai_result.flagged,
            categories: self.convert_categories(&openai_result.categories),
            category_scores: self.convert_category_scores(&openai_result.category_scores),
        }
    }

    /// Make HTTP request with proper headers.
    async fn make_request(&self) -> Result<reqwest::RequestBuilder, LlmError> {
        let url = format!("{}/moderations", self.config.base_url);
        // Build headers via ProviderHeaders to ensure consistency and support custom headers
        let headers = crate::utils::http_headers::ProviderHeaders::openai(
            self.config.api_key.expose_secret(),
            self.config.organization.as_deref(),
            self.config.project.as_deref(),
            &self.config.http_config.headers,
        )?;
        Ok(self.http_client.post(&url).headers(headers))
    }

    /// Handle API response errors.
    async fn handle_response_error(&self, response: reqwest::Response) -> LlmError {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());

        match status.as_u16() {
            400 => LlmError::InvalidInput(format!("Bad request: {error_text}")),
            401 => LlmError::AuthenticationError("Invalid API key".to_string()),
            429 => LlmError::RateLimitError("Rate limit exceeded".to_string()),
            _ => LlmError::ApiError {
                code: status.as_u16(),
                message: format!("OpenAI Moderation API error {status}: {error_text}"),
                details: None,
            },
        }
    }
}

#[async_trait]
impl ModerationCapability for OpenAiModeration {
    /// Moderate content for policy violations.
    async fn moderate(&self, request: ModerationRequest) -> Result<ModerationResponse, LlmError> {
        // Minimal stable validation only: allow either single input or non-empty array
        let has_single = !request.input.trim().is_empty();
        let has_array = request
            .inputs
            .as_ref()
            .map(|v| !v.is_empty() && v.iter().any(|s| !s.trim().is_empty()))
            .unwrap_or(false);
        if !has_single && !has_array {
            return Err(LlmError::InvalidInput(
                "Input text cannot be empty".to_string(),
            ));
        }

        // Build request payload via centralized transformer
        let req_tx = crate::providers::openai::transformers::OpenAiRequestTransformer;
        let openai_request = req_tx.transform_moderation(&request)?;

        // Make API request
        let request_builder = self.make_request().await?;
        let response = request_builder
            .json(&openai_request)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("Request failed: {e}")))?;

        if !response.status().is_success() {
            return Err(self.handle_response_error(response).await);
        }

        let openai_response: OpenAiModerationResponse = response
            .json()
            .await
            .map_err(|e| LlmError::ParseError(format!("Failed to parse response: {e}")))?;

        // Convert to our standard format
        let results: Vec<ModerationResult> = openai_response
            .results
            .into_iter()
            .map(|r| self.convert_result(r))
            .collect();

        Ok(ModerationResponse {
            results,
            model: openai_response.model,
        })
    }

    /// Get supported moderation categories.
    fn supported_categories(&self) -> Vec<String> {
        vec![
            "hate".to_string(),
            "hate/threatening".to_string(),
            "harassment".to_string(),
            "harassment/threatening".to_string(),
            "self-harm".to_string(),
            "self-harm/intent".to_string(),
            "self-harm/instructions".to_string(),
            "sexual".to_string(),
            "sexual/minors".to_string(),
            "violence".to_string(),
            "violence/graphic".to_string(),
        ]
    }
}
