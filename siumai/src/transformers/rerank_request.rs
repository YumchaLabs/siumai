//! Rerank Request Transformer Trait
//!
//! Specialized transformer for rerank requests.

use crate::error::LlmError;
use crate::types::RerankRequest;

/// Transform unified rerank request into provider-specific payload
pub trait RerankRequestTransformer: Send + Sync {
    /// Transform a unified RerankRequest into a provider-specific JSON body
    fn transform(&self, req: &RerankRequest) -> Result<serde_json::Value, LlmError>;
}
