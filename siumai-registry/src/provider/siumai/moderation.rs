use super::Siumai;
use crate::error::LlmError;
use crate::traits::ModerationCapability;
use crate::types::*;

#[async_trait::async_trait]
impl ModerationCapability for Siumai {
    async fn moderate(&self, request: ModerationRequest) -> Result<ModerationResponse, LlmError> {
        if let Some(m) = self.client.as_moderation_capability() {
            m.moderate(request).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support moderation.",
                self.client.provider_id()
            )))
        }
    }
}
