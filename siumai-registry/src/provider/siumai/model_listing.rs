use super::Siumai;
use crate::error::LlmError;
use crate::traits::ModelListingCapability;
use crate::types::*;

#[async_trait::async_trait]
impl ModelListingCapability for Siumai {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        if let Some(m) = self.client.as_model_listing_capability() {
            m.list_models().await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support model listing.",
                self.client.provider_id()
            )))
        }
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        if let Some(m) = self.client.as_model_listing_capability() {
            m.get_model(model_id).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support model listing.",
                self.client.provider_id()
            )))
        }
    }
}
