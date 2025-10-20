//! Model listing capability trait

use crate::error::LlmError;
use crate::types::ModelInfo;
use async_trait::async_trait;

#[async_trait]
pub trait ModelListingCapability: Send + Sync {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError>;
    async fn get_model(&self, model_id: String) -> Result<ModelInfo, LlmError>;

    async fn is_model_available(&self, model_id: String) -> Result<bool, LlmError> {
        match self.get_model(model_id).await {
            Ok(_) => Ok(true),
            Err(LlmError::NotFound(_)) => Ok(false),
            Err(e) => Err(e),
        }
    }
}
