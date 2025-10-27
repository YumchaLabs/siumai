//! Rerank Response Transformer Trait
//!
//! Specialized transformer for rerank responses.

use crate::error::LlmError;
use crate::types::RerankResponse;

/// Transform provider-specific rerank response into unified response
pub trait RerankResponseTransformer: Send + Sync {
    /// Transform provider-specific rerank response JSON to unified RerankResponse
    fn transform(&self, raw: serde_json::Value) -> Result<RerankResponse, LlmError>;
}
