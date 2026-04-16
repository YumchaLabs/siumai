//! Skill upload capability trait

use crate::error::LlmError;
use crate::types::{SkillUploadRequest, SkillUploadResult};
use async_trait::async_trait;

#[async_trait]
pub trait SkillsCapability: Send + Sync {
    async fn upload_skill(
        &self,
        request: SkillUploadRequest,
    ) -> Result<SkillUploadResult, LlmError>;
}
