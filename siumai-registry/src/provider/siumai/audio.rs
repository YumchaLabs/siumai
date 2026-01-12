use super::Siumai;
use crate::error::LlmError;
use crate::traits::AudioCapability;
use crate::types::*;

#[async_trait::async_trait]
impl AudioCapability for Siumai {
    fn supported_features(&self) -> &[AudioFeature] {
        self.client
            .as_audio_capability()
            .map(|a| a.supported_features())
            .unwrap_or(&[])
    }

    async fn text_to_speech(&self, request: TtsRequest) -> Result<TtsResponse, LlmError> {
        if let Some(audio) = self.client.as_audio_capability() {
            audio.text_to_speech(request).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support audio.",
                self.client.provider_id()
            )))
        }
    }

    async fn text_to_speech_stream(&self, request: TtsRequest) -> Result<AudioStream, LlmError> {
        if let Some(audio) = self.client.as_audio_capability() {
            audio.text_to_speech_stream(request).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support audio streaming.",
                self.client.provider_id()
            )))
        }
    }

    async fn speech_to_text(&self, request: SttRequest) -> Result<SttResponse, LlmError> {
        if let Some(audio) = self.client.as_audio_capability() {
            audio.speech_to_text(request).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support speech-to-text.",
                self.client.provider_id()
            )))
        }
    }

    async fn speech_to_text_stream(&self, request: SttRequest) -> Result<AudioStream, LlmError> {
        if let Some(audio) = self.client.as_audio_capability() {
            audio.speech_to_text_stream(request).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support streaming speech-to-text.",
                self.client.provider_id()
            )))
        }
    }

    async fn translate_audio(
        &self,
        request: AudioTranslationRequest,
    ) -> Result<SttResponse, LlmError> {
        if let Some(audio) = self.client.as_audio_capability() {
            audio.translate_audio(request).await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support audio translation.",
                self.client.provider_id()
            )))
        }
    }

    async fn get_voices(&self) -> Result<Vec<VoiceInfo>, LlmError> {
        if let Some(audio) = self.client.as_audio_capability() {
            audio.get_voices().await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support voice listing.",
                self.client.provider_id()
            )))
        }
    }

    async fn get_supported_languages(&self) -> Result<Vec<LanguageInfo>, LlmError> {
        if let Some(audio) = self.client.as_audio_capability() {
            audio.get_supported_languages().await
        } else {
            Err(LlmError::UnsupportedOperation(format!(
                "Provider {} does not support language listing.",
                self.client.provider_id()
            )))
        }
    }

    fn get_supported_audio_formats(&self) -> Vec<String> {
        self.client
            .as_audio_capability()
            .map(|a| a.get_supported_audio_formats())
            .unwrap_or_else(|| vec!["mp3".to_string(), "wav".to_string(), "ogg".to_string()])
    }
}
