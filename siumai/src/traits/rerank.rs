//! Rerank capability trait

use crate::error::LlmError;
use crate::types::{RerankRequest, RerankResponse};
use async_trait::async_trait;

#[async_trait]
pub trait RerankCapability: Send + Sync {
    async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, LlmError>;
    fn max_documents(&self) -> Option<u32> {
        None
    }
    fn supported_models(&self) -> Vec<String> {
        vec![]
    }
}
