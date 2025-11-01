use super::OpenAiClient;
use crate::error::LlmError;
use crate::traits::AudioCapability;
use async_trait::async_trait;

#[async_trait]
impl AudioCapability for OpenAiClient {
    fn supported_features(&self) -> &[crate::types::AudioFeature] {
        use crate::types::AudioFeature::*;
        const FEATURES: &[crate::types::AudioFeature] = &[TextToSpeech, SpeechToText];
        FEATURES
    }

    async fn text_to_speech(
        &self,
        request: crate::types::TtsRequest,
    ) -> Result<crate::types::TtsResponse, LlmError> {
        use crate::execution::executors::audio::AudioExecutor;

        let exec = self.build_audio_executor();
        let result = AudioExecutor::tts(&exec, request.clone()).await?;

        Ok(crate::types::TtsResponse {
            audio_data: result.audio_data,
            format: request.format.unwrap_or_else(|| "mp3".to_string()),
            duration: result.duration,
            sample_rate: result.sample_rate,
            metadata: std::collections::HashMap::new(),
        })
    }

    async fn speech_to_text(
        &self,
        request: crate::types::SttRequest,
    ) -> Result<crate::types::SttResponse, LlmError> {
        use crate::execution::executors::audio::AudioExecutor;

        let exec = self.build_audio_executor();
        let text = AudioExecutor::stt(&exec, request).await?;

        Ok(crate::types::SttResponse {
            text,
            language: None,
            confidence: None,
            words: None,
            duration: None,
            metadata: std::collections::HashMap::new(),
        })
    }
}
