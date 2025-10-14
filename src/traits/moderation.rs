//! Moderation capability trait

use crate::error::LlmError;
use crate::types::{ModerationRequest, ModerationResponse};
use async_trait::async_trait;

#[async_trait]
pub trait ModerationCapability: Send + Sync {
    async fn moderate(&self, request: ModerationRequest) -> Result<ModerationResponse, LlmError>;

    fn supported_categories(&self) -> Vec<String> {
        vec![
            "hate".to_string(),
            "hate/threatening".to_string(),
            "harassment".to_string(),
            "harassment/threatening".to_string(),
            "self-harm".to_string(),
            "self-harm/intent".to_string(),
            "self-harm/instructions".to_string(),
            "sexual".to_string(),
            "sexual/minors".to_string(),
            "violence".to_string(),
            "violence/graphic".to_string(),
        ]
    }
}
