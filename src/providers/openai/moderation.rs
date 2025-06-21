//! `OpenAI` Moderation API Implementation
//!
//! This module provides the `OpenAI` implementation of the `ModerationCapability` trait,
//! including content moderation for text and image content.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::LlmError;
use crate::traits::ModerationCapability;
use crate::types::{ModerationRequest, ModerationResponse, ModerationResult};

use super::config::OpenAiConfig;

/// `OpenAI` moderation API request structure
#[derive(Debug, Clone, Serialize)]
struct OpenAiModerationRequest {
    /// Input text or array of texts to moderate
    input: serde_json::Value,
    /// Model to use for moderation
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
}

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

    /// Validate moderation request.
    fn validate_request(&self, request: &ModerationRequest) -> Result<(), LlmError> {
        // Validate input text
        if request.input.trim().is_empty() {
            return Err(LlmError::InvalidInput(
                "Input text cannot be empty".to_string(),
            ));
        }

        // Validate input length (OpenAI has a limit)
        if request.input.len() > 32768 {
            return Err(LlmError::InvalidInput(
                "Input text exceeds maximum length of 32,768 characters".to_string(),
            ));
        }

        // Validate model if specified
        if let Some(ref model) = request.model {
            if !self.get_supported_models().contains(model) {
                return Err(LlmError::InvalidInput(format!(
                    "Unsupported moderation model: {}. Supported models: {:?}",
                    model,
                    self.get_supported_models()
                )));
            }
        }

        Ok(())
    }

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

        let mut headers = reqwest::header::HeaderMap::new();
        for (key, value) in self.config.get_headers() {
            let header_name = reqwest::header::HeaderName::from_bytes(key.as_bytes())
                .map_err(|e| LlmError::HttpError(format!("Invalid header name: {e}")))?;
            let header_value = reqwest::header::HeaderValue::from_str(&value)
                .map_err(|e| LlmError::HttpError(format!("Invalid header value: {e}")))?;
            headers.insert(header_name, header_value);
        }

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
        // Validate request
        self.validate_request(&request)?;

        // Prepare OpenAI request
        let openai_request = OpenAiModerationRequest {
            input: serde_json::Value::String(request.input),
            model: request.model.or_else(|| Some(self.default_model())),
        };

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
