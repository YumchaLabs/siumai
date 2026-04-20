//! Audio capability trait

use crate::error::LlmError;
use crate::types::{
    AudioFeature, AudioStream, AudioTranslationRequest, LanguageInfo, SttRequest, SttResponse,
    TtsRequest, TtsResponse, VoiceInfo,
};
use async_trait::async_trait;

async fn load_audio_file_request(path: &str) -> Result<(Vec<u8>, String), LlmError> {
    let bytes = tokio::fs::read(path)
        .await
        .map_err(|e| LlmError::IoError(format!("Failed to read audio file '{path}': {e}")))?;
    let media_type = crate::utils::guess_mime_from_path_or_url(path).ok_or_else(|| {
        LlmError::InvalidInput(format!(
            "Could not infer audio media type from file path '{path}'; pass a request with an explicit media_type instead."
        ))
    })?;
    Ok((bytes, media_type))
}

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

    async fn transcribe(&self, audio: Vec<u8>, media_type: String) -> Result<String, LlmError> {
        let request = SttRequest::from_audio(audio, media_type);
        let response = self.speech_to_text(request).await?;
        Ok(response.text)
    }

    async fn transcribe_file(&self, file_path: String) -> Result<String, LlmError> {
        let (audio, media_type) = load_audio_file_request(&file_path).await?;
        let request = SttRequest::from_audio(audio, media_type);
        let response = self.speech_to_text(request).await?;
        Ok(response.text)
    }

    async fn translate(&self, audio: Vec<u8>, media_type: String) -> Result<String, LlmError> {
        let request = AudioTranslationRequest::from_audio(audio, media_type);
        let response = self.translate_audio(request).await?;
        Ok(response.text)
    }

    async fn translate_file(&self, file_path: String) -> Result<String, LlmError> {
        let (audio, media_type) = load_audio_file_request(&file_path).await?;
        let request = AudioTranslationRequest::from_audio(audio, media_type);
        let response = self.translate_audio(request).await?;
        Ok(response.text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Default)]
    struct RecordingAudioCapability {
        stt_request: Mutex<Option<SttRequest>>,
        translation_request: Mutex<Option<AudioTranslationRequest>>,
    }

    #[async_trait]
    impl AudioCapability for RecordingAudioCapability {
        fn supported_features(&self) -> &[AudioFeature] {
            &[]
        }

        async fn speech_to_text(&self, request: SttRequest) -> Result<SttResponse, LlmError> {
            *self.stt_request.lock().expect("stt lock") = Some(request);
            Ok(SttResponse {
                text: "ok".to_string(),
                language: None,
                confidence: None,
                words: None,
                duration: None,
                metadata: Default::default(),
                response: None,
            })
        }

        async fn translate_audio(
            &self,
            request: AudioTranslationRequest,
        ) -> Result<SttResponse, LlmError> {
            *self.translation_request.lock().expect("translation lock") = Some(request);
            Ok(SttResponse {
                text: "translated".to_string(),
                language: None,
                confidence: None,
                words: None,
                duration: None,
                metadata: Default::default(),
                response: None,
            })
        }
    }

    #[tokio::test]
    async fn transcribe_file_loads_bytes_into_canonical_audio_input() {
        let tmp = tempfile::Builder::new()
            .suffix(".wav")
            .tempfile()
            .expect("temp file");
        std::fs::write(tmp.path(), b"abc").expect("write");

        let capability = RecordingAudioCapability::default();
        let text = capability
            .transcribe_file(tmp.path().to_string_lossy().to_string())
            .await
            .expect("transcribe file");

        assert_eq!(text, "ok");

        let request = capability
            .stt_request
            .lock()
            .expect("stt lock")
            .clone()
            .expect("recorded request");
        assert_eq!(request.audio_bytes().expect("audio bytes"), b"abc");
        assert_eq!(request.media_type, "audio/wav");
    }

    #[tokio::test]
    async fn translate_file_loads_bytes_into_canonical_audio_input() {
        let tmp = tempfile::Builder::new()
            .suffix(".wav")
            .tempfile()
            .expect("temp file");
        std::fs::write(tmp.path(), b"xyz").expect("write");

        let capability = RecordingAudioCapability::default();
        let text = capability
            .translate_file(tmp.path().to_string_lossy().to_string())
            .await
            .expect("translate file");

        assert_eq!(text, "translated");

        let request = capability
            .translation_request
            .lock()
            .expect("translation lock")
            .clone()
            .expect("recorded request");
        assert_eq!(request.audio_bytes().expect("audio bytes"), b"xyz");
        assert_eq!(request.media_type, "audio/wav");
    }
}
