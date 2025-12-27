//! `OpenAI` Moderation API Implementation
//!
//! This module provides the `OpenAI` implementation of the `ModerationCapability` trait,
//! including content moderation for text and image content.

use async_trait::async_trait;

use crate::error::LlmError;
use crate::execution::transformers::request::RequestTransformer;
use crate::execution::transformers::response::ResponseTransformer;
use crate::traits::ModerationCapability;
use crate::types::{ModerationRequest, ModerationResponse};

use super::config::OpenAiConfig;
use secrecy::ExposeSecret;

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

    fn build_context(&self) -> crate::core::ProviderContext {
        crate::core::ProviderContext::new(
            "openai",
            self.config.base_url.clone(),
            Some(self.config.api_key.expose_secret().to_string()),
            self.config.http_config.headers.clone(),
        )
        .with_org_project(
            self.config.organization.clone(),
            self.config.project.clone(),
        )
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
        let body = req_tx.transform_moderation(&request)?;

        let spec = std::sync::Arc::new(super::spec::OpenAiSpec::new());
        let ctx = self.build_context();
        let config = crate::execution::executors::common::HttpExecutionConfig {
            provider_id: "openai".to_string(),
            http_client: self.http_client.clone(),
            provider_spec: spec,
            provider_context: ctx,
            interceptors: Vec::new(),
            retry_options: None,
        };

        let url = format!("{}/moderations", self.config.base_url.trim_end_matches('/'));
        let result = crate::execution::executors::common::execute_json_request(
            &config,
            &url,
            crate::execution::executors::common::HttpBody::Json(body),
            None,
            false,
        )
        .await?;

        let resp_tx = crate::providers::openai::transformers::OpenAiResponseTransformer;
        resp_tx.transform_moderation_response(&result.json)
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
