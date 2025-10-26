use super::GeminiClient;
use crate::error::LlmError;
use crate::traits::ModelListingCapability;
use crate::types::ModelInfo;
use async_trait::async_trait;

#[async_trait]
impl ModelListingCapability for GeminiClient {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        self.models_capability.list_models().await
    }

    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError> {
        self.models_capability.get_model(model_id).await
    }
}
