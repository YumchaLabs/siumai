use super::Siumai;
use crate::error::LlmError;
use crate::traits::RerankCapability;
use crate::types::{RerankRequest, RerankResponse};

#[async_trait::async_trait]
impl RerankCapability for Siumai {
    async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, LlmError> {
        if let Some(rerank_cap) = self.client.as_rerank_capability() {
            rerank_cap.rerank(request).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support rerank.",
                self.client.provider_id()
            )))
        }
    }
}
