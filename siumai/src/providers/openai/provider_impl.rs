//! OpenAI Provider Implementation
//!
//! Implements the Provider trait for OpenAI, creating Model instances for different endpoints.

use crate::error::LlmError;
use crate::provider_model::Provider;
use crate::provider_model::model::{ChatModel, EmbeddingModel, ImageModel};
use std::collections::HashMap;

use super::config::OpenAiConfig;
use super::model_impls::{OpenAiChatModel, OpenAiEmbeddingModel, OpenAiImageModel};

/// OpenAI Provider
///
/// A lightweight factory for creating OpenAI Model instances.
///
/// ## Example
///
/// ```rust,ignore
/// use siumai::providers::openai::OpenAiProvider;
/// use siumai::provider_model::Provider;
///
/// let config = OpenAiConfig::new("your-api-key").with_model("gpt-4");
/// let provider = OpenAiProvider::new(config);
///
/// // Create chat model
/// let chat_model = provider.chat("gpt-4")?;
///
/// // Create embedding model
/// let embedding_model = provider.embedding("text-embedding-3-small")?;
///
/// // Create image model
/// let image_model = provider.image("dall-e-3")?;
/// ```
pub struct OpenAiProvider {
    config: OpenAiConfig,
}

impl OpenAiProvider {
    /// Create a new OpenAI Provider
    ///
    /// # Arguments
    /// * `config` - OpenAI configuration (API key, base URL, etc.)
    pub fn new(config: OpenAiConfig) -> Self {
        Self { config }
    }

    /// Get the configuration
    pub fn config(&self) -> &OpenAiConfig {
        &self.config
    }
}

impl Provider for OpenAiProvider {
    fn id(&self) -> &str {
        "openai"
    }

    fn chat(&self, model: &str) -> Result<Box<dyn ChatModel>, LlmError> {
        Ok(Box::new(OpenAiChatModel::new(
            self.config.clone(),
            model.to_string(),
            None, // No adapter for native OpenAI
        )))
    }

    fn embedding(&self, model: &str) -> Result<Box<dyn EmbeddingModel>, LlmError> {
        Ok(Box::new(OpenAiEmbeddingModel::new(
            self.config.clone(),
            model.to_string(),
            None, // No adapter for native OpenAI
        )))
    }

    fn image(&self, model: &str) -> Result<Box<dyn ImageModel>, LlmError> {
        Ok(Box::new(OpenAiImageModel::new(
            self.config.clone(),
            model.to_string(),
            None, // No adapter for native OpenAI
        )))
    }
}

/// Model configuration extracted from OpenAiConfig
///
/// This is used by Model implementations to create Executors.
#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub provider_id: String,
    pub base_url: String,
    pub api_key: secrecy::SecretString,
    pub headers: HashMap<String, String>,
    pub organization: Option<String>,
    pub project: Option<String>,
}

impl From<&OpenAiConfig> for ModelConfig {
    fn from(config: &OpenAiConfig) -> Self {
        Self {
            provider_id: "openai".to_string(),
            base_url: config.base_url.clone(),
            api_key: config.api_key.clone(),
            headers: config.http_config.headers.clone(),
            organization: config.organization.clone(),
            project: config.project.clone(),
        }
    }
}
