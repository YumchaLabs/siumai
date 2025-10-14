//! Request transformation traits
//!
//! Converts unified request structs into provider-specific JSON bodies or structs.
//! This matches Cherry Studio's RequestTransformer concept. For now this is an
//! interface definition used by upcoming executors.

use crate::error::LlmError;
use crate::types::{
    ChatRequest, EmbeddingRequest, ImageEditRequest, ImageGenerationRequest, ImageVariationRequest,
};

/// Body for image HTTP requests
pub enum ImageHttpBody {
    Json(serde_json::Value),
    Multipart(reqwest::multipart::Form),
}

/// Transform unified chat request into provider-specific payload
pub trait RequestTransformer: Send + Sync {
    /// Provider identifier (e.g., "openai", "anthropic", "gemini", or compat id)
    fn provider_id(&self) -> &str;

    /// Transform a unified ChatRequest into a provider-specific JSON body
    fn transform_chat(&self, req: &ChatRequest) -> Result<serde_json::Value, LlmError>;

    /// Transform an EmbeddingRequest into a provider-specific JSON body
    fn transform_embedding(&self, _req: &EmbeddingRequest) -> Result<serde_json::Value, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "{} does not implement embedding transformer",
            self.provider_id()
        )))
    }

    /// Transform an ImageGenerationRequest into a provider-specific JSON body
    fn transform_image(
        &self,
        _req: &ImageGenerationRequest,
    ) -> Result<serde_json::Value, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "{} does not implement image transformer",
            self.provider_id()
        )))
    }

    /// Transform an ImageEditRequest into a provider-specific body (JSON or multipart)
    fn transform_image_edit(&self, _req: &ImageEditRequest) -> Result<ImageHttpBody, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "{} does not implement image edit transformer",
            self.provider_id()
        )))
    }

    /// Transform an ImageVariationRequest into a provider-specific body (JSON or multipart)
    fn transform_image_variation(
        &self,
        _req: &ImageVariationRequest,
    ) -> Result<ImageHttpBody, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "{} does not implement image variation transformer",
            self.provider_id()
        )))
    }
}
