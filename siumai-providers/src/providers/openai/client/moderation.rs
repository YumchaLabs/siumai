//! OpenAI moderation capability implementation (delegating to `OpenAiModeration`).

use async_trait::async_trait;

use crate::error::LlmError;
use crate::traits::ModerationCapability;
use crate::types::{ModerationRequest, ModerationResponse};

use super::OpenAiClient;

#[async_trait]
impl ModerationCapability for OpenAiClient {
    async fn moderate(&self, request: ModerationRequest) -> Result<ModerationResponse, LlmError> {
        let cfg = super::super::OpenAiConfig {
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            organization: self.organization.clone(),
            project: self.project.clone(),
            common_params: self.common_params.clone(),
            openai_params: self.openai_params.clone(),
            http_config: self.http_config.clone(),
        };

        let moderation =
            super::super::moderation::OpenAiModeration::new(cfg, self.http_client.clone());
        moderation.moderate(request).await
    }
}
