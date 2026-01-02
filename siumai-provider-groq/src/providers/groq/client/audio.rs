use super::GroqClient;
use crate::error::LlmError;
use crate::providers::groq::spec;
use crate::traits::AudioCapability;
use async_trait::async_trait;

#[async_trait]
impl AudioCapability for GroqClient {
    fn supported_features(&self) -> &[crate::types::AudioFeature] {
        use crate::types::AudioFeature::*;
        const FEATURES: &[crate::types::AudioFeature] = &[TextToSpeech, SpeechToText];
        FEATURES
    }

    async fn text_to_speech(
        &self,
        request: crate::types::TtsRequest,
    ) -> Result<crate::types::TtsResponse, LlmError> {
        use crate::execution::executors::audio::{AudioExecutor, AudioExecutorBuilder};

        let spec = std::sync::Arc::new(spec::GroqSpec);
        let ctx = self.provider_context();

        let mut builder = AudioExecutorBuilder::new("groq", self.http_client())
            .with_spec(spec)
            .with_context(ctx)
            .with_interceptors(self.http_interceptors());

        if let Some(retry) = self.retry_options() {
            builder = builder.with_retry_options(retry);
        }

        let exec = builder.build();
        let result = AudioExecutor::tts(&*exec, request).await?;
        Ok(crate::types::TtsResponse {
            audio_data: result.audio_data,
            format: "wav".to_string(),
            duration: result.duration,
            sample_rate: result.sample_rate,
            metadata: std::collections::HashMap::new(),
        })
    }

    async fn speech_to_text(
        &self,
        request: crate::types::SttRequest,
    ) -> Result<crate::types::SttResponse, LlmError> {
        use crate::execution::executors::audio::{AudioExecutor, AudioExecutorBuilder};

        let spec = std::sync::Arc::new(spec::GroqSpec);
        let ctx = self.provider_context();

        let mut builder = AudioExecutorBuilder::new("groq", self.http_client())
            .with_spec(spec)
            .with_context(ctx)
            .with_interceptors(self.http_interceptors());

        if let Some(retry) = self.retry_options() {
            builder = builder.with_retry_options(retry);
        }

        let exec = builder.build();
        let result = AudioExecutor::stt(&*exec, request).await?;
        Ok(crate::types::SttResponse {
            text: result.text,
            language: None,
            confidence: None,
            words: None,
            duration: None,
            metadata: std::collections::HashMap::new(),
        })
    }
}
