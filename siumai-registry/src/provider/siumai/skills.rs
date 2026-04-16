use super::Siumai;
use crate::error::LlmError;
use crate::traits::SkillsCapability;
use crate::types::{SkillUploadRequest, SkillUploadResult};

#[async_trait::async_trait]
impl SkillsCapability for Siumai {
    async fn upload_skill(
        &self,
        request: SkillUploadRequest,
    ) -> Result<SkillUploadResult, LlmError> {
        if let Some(skills) = self.client.as_skills_capability() {
            skills.upload_skill(request).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support skills.",
                self.client.provider_id()
            )))
        }
    }
}
