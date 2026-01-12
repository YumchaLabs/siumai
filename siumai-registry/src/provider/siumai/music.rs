use super::Siumai;
use crate::error::LlmError;
use crate::traits::MusicGenerationCapability;
use crate::types::*;

#[async_trait::async_trait]
impl MusicGenerationCapability for Siumai {
    async fn generate_music(
        &self,
        request: MusicGenerationRequest,
    ) -> Result<MusicGenerationResponse, LlmError> {
        if let Some(music) = self.client.as_music_generation_capability() {
            music.generate_music(request).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support music generation.",
                self.client.provider_id()
            )))
        }
    }

    fn get_supported_music_models(&self) -> Vec<String> {
        self.client
            .as_music_generation_capability()
            .map(|m| m.get_supported_music_models())
            .unwrap_or_default()
    }

    fn get_supported_audio_formats(&self) -> Vec<String> {
        self.client
            .as_music_generation_capability()
            .map(|m| m.get_supported_audio_formats())
            .unwrap_or_default()
    }

    fn supports_lyrics(&self) -> bool {
        self.client
            .as_music_generation_capability()
            .map(|m| m.supports_lyrics())
            .unwrap_or(false)
    }
}
