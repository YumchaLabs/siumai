use std::sync::Arc;

use crate::client::LlmClient;
use crate::error::LlmError;
use crate::traits::{FileManagementCapability, MusicGenerationCapability, SkillsCapability};
use crate::types::{
    FileDeleteResponse, FileListQuery, FileListResponse, FileObject, FileUploadRequest,
    MusicGenerationRequest, MusicGenerationResponse, MusicStyle, SkillUploadRequest,
    SkillUploadResult,
};

fn unsupported(provider_id: &str, extension: &str) -> LlmError {
    LlmError::UnsupportedOperation(format!(
        "Provider {provider_id} does not support {extension}."
    ))
}

pub(in crate::registry::entry) struct ClientBackedFileManagementCapability {
    client: Arc<dyn LlmClient>,
    provider_id: String,
}

impl ClientBackedFileManagementCapability {
    pub(in crate::registry::entry) fn new(client: Arc<dyn LlmClient>, provider_id: String) -> Self {
        Self {
            client,
            provider_id,
        }
    }

    fn capability(&self) -> Result<&dyn FileManagementCapability, LlmError> {
        self.client
            .as_file_management_capability()
            .ok_or_else(|| unsupported(&self.provider_id, "file management"))
    }
}

#[async_trait::async_trait]
impl FileManagementCapability for ClientBackedFileManagementCapability {
    async fn upload_file(&self, request: FileUploadRequest) -> Result<FileObject, LlmError> {
        self.capability()?.upload_file(request).await
    }

    async fn list_files(&self, query: Option<FileListQuery>) -> Result<FileListResponse, LlmError> {
        self.capability()?.list_files(query).await
    }

    async fn retrieve_file(&self, file_id: String) -> Result<FileObject, LlmError> {
        self.capability()?.retrieve_file(file_id).await
    }

    async fn delete_file(&self, file_id: String) -> Result<FileDeleteResponse, LlmError> {
        self.capability()?.delete_file(file_id).await
    }

    async fn get_file_content(&self, file_id: String) -> Result<Vec<u8>, LlmError> {
        self.capability()?.get_file_content(file_id).await
    }
}

pub(in crate::registry::entry) struct ClientBackedSkillsCapability {
    client: Arc<dyn LlmClient>,
    provider_id: String,
}

impl ClientBackedSkillsCapability {
    pub(in crate::registry::entry) fn new(client: Arc<dyn LlmClient>, provider_id: String) -> Self {
        Self {
            client,
            provider_id,
        }
    }

    fn capability(&self) -> Result<&dyn SkillsCapability, LlmError> {
        self.client
            .as_skills_capability()
            .ok_or_else(|| unsupported(&self.provider_id, "skills"))
    }
}

#[async_trait::async_trait]
impl SkillsCapability for ClientBackedSkillsCapability {
    async fn upload_skill(
        &self,
        request: SkillUploadRequest,
    ) -> Result<SkillUploadResult, LlmError> {
        self.capability()?.upload_skill(request).await
    }
}

pub(in crate::registry::entry) struct ClientBackedMusicGenerationCapability {
    client: Arc<dyn LlmClient>,
    provider_id: String,
}

impl ClientBackedMusicGenerationCapability {
    pub(in crate::registry::entry) fn new(client: Arc<dyn LlmClient>, provider_id: String) -> Self {
        Self {
            client,
            provider_id,
        }
    }

    fn capability(&self) -> Result<&dyn MusicGenerationCapability, LlmError> {
        self.client
            .as_music_generation_capability()
            .ok_or_else(|| unsupported(&self.provider_id, "music generation"))
    }
}

#[async_trait::async_trait]
impl MusicGenerationCapability for ClientBackedMusicGenerationCapability {
    async fn generate_music(
        &self,
        request: MusicGenerationRequest,
    ) -> Result<MusicGenerationResponse, LlmError> {
        self.capability()?.generate_music(request).await
    }

    async fn continue_music(
        &self,
        request: MusicGenerationRequest,
    ) -> Result<MusicGenerationResponse, LlmError> {
        self.capability()?.continue_music(request).await
    }

    fn get_supported_music_models(&self) -> Vec<String> {
        self.client
            .as_music_generation_capability()
            .map(|music| music.get_supported_music_models())
            .unwrap_or_default()
    }

    fn get_supported_styles(&self) -> Vec<MusicStyle> {
        self.client
            .as_music_generation_capability()
            .map(|music| music.get_supported_styles())
            .unwrap_or_default()
    }

    fn get_supported_audio_formats(&self) -> Vec<String> {
        self.client
            .as_music_generation_capability()
            .map(|music| music.get_supported_audio_formats())
            .unwrap_or_else(|| vec!["mp3".to_string(), "wav".to_string()])
    }

    fn supports_lyrics(&self) -> bool {
        self.client
            .as_music_generation_capability()
            .map(|music| music.supports_lyrics())
            .unwrap_or(false)
    }

    fn supports_continuation(&self) -> bool {
        self.client
            .as_music_generation_capability()
            .map(|music| music.supports_continuation())
            .unwrap_or(false)
    }

    fn supports_instrumental(&self) -> bool {
        self.client
            .as_music_generation_capability()
            .map(|music| music.supports_instrumental())
            .unwrap_or(true)
    }
}
