use super::GroqClient;
use crate::error::LlmError;
use crate::providers::groq::spec;
use crate::traits::AudioCapability;
use async_trait::async_trait;
use std::collections::HashMap;

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
        let request = request.with_model_if_missing(self.inner().model().to_string());

        let spec = std::sync::Arc::new(spec::GroqSpec);
        let ctx = self.provider_context();

        let mut builder = AudioExecutorBuilder::new("groq", self.http_client())
            .with_spec(spec)
            .with_context(ctx)
            .with_interceptors(self.http_interceptors());

        if let Some(transport) = self.http_transport() {
            builder = builder.with_transport(transport);
        }

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
            response: result.response,
        })
    }

    async fn speech_to_text(
        &self,
        request: crate::types::SttRequest,
    ) -> Result<crate::types::SttResponse, LlmError> {
        use crate::execution::executors::audio::{AudioExecutor, AudioExecutorBuilder};
        let request = request.with_model_if_missing(self.inner().model().to_string());

        let spec = std::sync::Arc::new(spec::GroqSpec);
        let ctx = self.provider_context();

        let mut builder = AudioExecutorBuilder::new("groq", self.http_client())
            .with_spec(spec)
            .with_context(ctx)
            .with_interceptors(self.http_interceptors());

        if let Some(transport) = self.http_transport() {
            builder = builder.with_transport(transport);
        }

        if let Some(retry) = self.retry_options() {
            builder = builder.with_retry_options(retry);
        }

        let exec = builder.build();
        let result = AudioExecutor::stt(&*exec, request).await?;
        let text = result.text;
        let response = result.response;
        let raw = result.raw;
        let language = raw
            .get("language")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let duration = raw
            .get("duration")
            .and_then(|v| v.as_f64())
            .map(|d| d as f32);
        let words = raw.get("words").and_then(|v| v.as_array()).map(|arr| {
            arr.iter()
                .filter_map(|item| {
                    let obj = item.as_object()?;
                    let word = obj.get("word")?.as_str()?.to_string();
                    let start = obj.get("start")?.as_f64()? as f32;
                    let end = obj.get("end")?.as_f64()? as f32;
                    Some(crate::types::WordTimestamp {
                        word,
                        start,
                        end,
                        confidence: None,
                    })
                })
                .collect::<Vec<_>>()
        });
        let mut metadata = HashMap::new();
        for key in ["segments", "usage", "logprobs", "x_groq", "task"] {
            if let Some(value) = raw.get(key) {
                metadata.insert(key.to_string(), value.clone());
            }
        }

        Ok(crate::types::SttResponse {
            text,
            language,
            confidence: None,
            words,
            duration,
            metadata,
            response,
        })
    }
}
