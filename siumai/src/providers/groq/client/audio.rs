use super::GroqClient;
use crate::error::LlmError;
use crate::providers::groq::{spec, transformers};
use crate::traits::AudioCapability;
use async_trait::async_trait;
use secrecy::ExposeSecret;

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
        use crate::execution::executors::audio::{AudioExecutor, HttpAudioExecutor};

        let spec = std::sync::Arc::new(spec::GroqSpec);
        let ctx = crate::core::ProviderContext::new(
            "groq",
            self.config.base_url.clone(),
            Some(self.config.api_key.expose_secret().to_string()),
            self.config.http_config.headers.clone(),
        );

        let exec = HttpAudioExecutor {
            provider_id: "groq".to_string(),
            http_client: self.http_client.clone(),
            transformer: std::sync::Arc::new(transformers::GroqAudioTransformer),
            provider_spec: spec,
            provider_context: ctx,
            policy: crate::execution::ExecutionPolicy::new()
                .with_interceptors(self.http_interceptors.clone())
                .with_retry_options(self.retry_options.clone()),
        };
        let result = exec.tts(request).await?;
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
        use crate::execution::executors::audio::{AudioExecutor, HttpAudioExecutor};

        let spec = std::sync::Arc::new(spec::GroqSpec);
        let ctx = crate::core::ProviderContext::new(
            "groq",
            self.config.base_url.clone(),
            Some(self.config.api_key.expose_secret().to_string()),
            self.config.http_config.headers.clone(),
        );

        let exec = HttpAudioExecutor {
            provider_id: "groq".to_string(),
            http_client: self.http_client.clone(),
            transformer: std::sync::Arc::new(transformers::GroqAudioTransformer),
            provider_spec: spec,
            provider_context: ctx,
            policy: crate::execution::ExecutionPolicy::new()
                .with_interceptors(self.http_interceptors.clone())
                .with_retry_options(self.retry_options.clone()),
        };
        let text = exec.stt(request).await?;
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
