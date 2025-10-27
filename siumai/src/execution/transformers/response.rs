//! Response transformation traits
//!
//! Converts provider responses into unified response types.
//! This mirrors Cherry Studio's response transformer idea.

use crate::error::LlmError;
use crate::types::{
    ChatResponse, EmbeddingResponse, ImageGenerationResponse, ModerationResponse, RerankResponse,
};

/// Transform provider-specific responses into unified responses
pub trait ResponseTransformer: Send + Sync {
    /// Provider identifier
    fn provider_id(&self) -> &str;

    /// Transform provider-specific chat response JSON to unified ChatResponse
    fn transform_chat_response(&self, _raw: &serde_json::Value) -> Result<ChatResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "{} does not implement chat response transformer",
            self.provider_id()
        )))
    }

    /// Transform provider-specific embedding response JSON to unified EmbeddingResponse
    fn transform_embedding_response(
        &self,
        _raw: &serde_json::Value,
    ) -> Result<EmbeddingResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "{} does not implement embedding response transformer",
            self.provider_id()
        )))
    }

    /// Transform provider-specific image response JSON to unified ImageGenerationResponse
    fn transform_image_response(
        &self,
        _raw: &serde_json::Value,
    ) -> Result<ImageGenerationResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "{} does not implement image response transformer",
            self.provider_id()
        )))
    }

    /// Transform provider-specific rerank response JSON to unified RerankResponse
    fn transform_rerank_response(
        &self,
        _raw: &serde_json::Value,
    ) -> Result<RerankResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "{} does not implement rerank response transformer",
            self.provider_id()
        )))
    }

    /// Transform provider-specific moderation response JSON to unified ModerationResponse
    fn transform_moderation_response(
        &self,
        _raw: &serde_json::Value,
    ) -> Result<ModerationResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(format!(
            "{} does not implement moderation response transformer",
            self.provider_id()
        )))
    }
}
