//! Audio capability trait

use crate::error::LlmError;
use crate::types::{
    AudioFeature, AudioStream, AudioTranslationRequest, LanguageInfo, SttRequest, SttResponse,
    TtsRequest, TtsResponse, VoiceInfo,
};
use async_trait::async_trait;

#[async_trait]
pub trait AudioCapability: Send + Sync {
    fn supported_features(&self) -> &[AudioFeature];

    async fn text_to_speech(&self, _request: TtsRequest) -> Result<TtsResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Text-to-speech not supported by this provider".to_string(),
        ))
    }

    async fn text_to_speech_stream(&self, _request: TtsRequest) -> Result<AudioStream, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Streaming text-to-speech not supported by this provider".to_string(),
        ))
    }

    async fn speech_to_text(&self, _request: SttRequest) -> Result<SttResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Speech-to-text not supported by this provider".to_string(),
        ))
    }

    async fn speech_to_text_stream(&self, _request: SttRequest) -> Result<AudioStream, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Streaming speech-to-text not supported by this provider".to_string(),
        ))
    }

    async fn translate_audio(
        &self,
        _request: AudioTranslationRequest,
    ) -> Result<SttResponse, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Audio translation not supported by this provider".to_string(),
        ))
    }

    async fn get_voices(&self) -> Result<Vec<VoiceInfo>, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Voice listing not supported by this provider".to_string(),
        ))
    }

    async fn get_supported_languages(&self) -> Result<Vec<LanguageInfo>, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Language listing not supported by this provider".to_string(),
        ))
    }

    fn get_supported_audio_formats(&self) -> Vec<String> {
        vec!["mp3".to_string(), "wav".to_string(), "ogg".to_string()]
    }

    async fn speech(&self, text: String) -> Result<Vec<u8>, LlmError> {
        let request = TtsRequest::new(text);
        let response = self.text_to_speech(request).await?;
        Ok(response.audio_data)
    }

    async fn transcribe(&self, audio: Vec<u8>) -> Result<String, LlmError> {
        let request = SttRequest::from_audio(audio);
        let response = self.speech_to_text(request).await?;
        Ok(response.text)
    }

    async fn transcribe_file(&self, file_path: String) -> Result<String, LlmError> {
        let request = SttRequest::from_file(file_path);
        let response = self.speech_to_text(request).await?;
        Ok(response.text)
    }

    async fn translate(&self, audio: Vec<u8>) -> Result<String, LlmError> {
        let request = AudioTranslationRequest::from_audio(audio);
        let response = self.translate_audio(request).await?;
        Ok(response.text)
    }

    async fn translate_file(&self, file_path: String) -> Result<String, LlmError> {
        let request = AudioTranslationRequest::from_file(file_path);
        let response = self.translate_audio(request).await?;
        Ok(response.text)
    }
}
