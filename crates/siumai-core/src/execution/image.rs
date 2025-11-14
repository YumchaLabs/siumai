//! Minimal image transformers for standards

use crate::error::LlmError;
use crate::types::image::{
    ImageEditRequest, ImageGenerationRequest, ImageGenerationResponse, ImageVariationRequest,
};

/// Body for image HTTP requests
pub enum ImageHttpBody {
    Json(serde_json::Value),
    Multipart(reqwest::multipart::Form),
}

/// Request transformer focused on image operations
pub trait ImageRequestTransformer: Send + Sync {
    /// Provider identifier (e.g., "openai", "anthropic", compat id)
    fn provider_id(&self) -> &str;

    /// Transform an ImageGenerationRequest into a provider-specific JSON body
    fn transform_image(&self, req: &ImageGenerationRequest) -> Result<serde_json::Value, LlmError>;

    /// Transform an ImageEditRequest into a provider-specific body (JSON or multipart)
    fn transform_image_edit(&self, _req: &ImageEditRequest) -> Result<ImageHttpBody, LlmError>;

    /// Transform an ImageVariationRequest into a provider-specific body (JSON or multipart)
    fn transform_image_variation(
        &self,
        _req: &ImageVariationRequest,
    ) -> Result<ImageHttpBody, LlmError>;
}

/// Response transformer focused on image operations
pub trait ImageResponseTransformer: Send + Sync {
    /// Provider identifier
    fn provider_id(&self) -> &str;

    fn transform_image_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<ImageGenerationResponse, LlmError>;
}
