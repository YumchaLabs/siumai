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
        let result = AudioExecutor::tts(&*exec, request.clone()).await?;

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
        let result = AudioExecutor::stt(&*exec, request).await?;
        let raw = result.raw;

        let language = raw.get("language").and_then(|v| v.as_str()).map(|s| s.to_string());
        let duration = raw
            .get("duration")
            .and_then(|v| v.as_f64())
            .map(|s| s as f32);
        let words = raw
            .get("words")
            .and_then(|v| v.as_array())
            .map(|arr| {
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

        let mut metadata: std::collections::HashMap<String, serde_json::Value> =
            std::collections::HashMap::new();
        if let Some(usage) = raw.get("usage") {
            metadata.insert("usage".to_string(), usage.clone());
        }
        if let Some(segments) = raw.get("segments") {
            metadata.insert("segments".to_string(), segments.clone());
        }
        if let Some(logprobs) = raw.get("logprobs") {
            metadata.insert("logprobs".to_string(), logprobs.clone());
        }

        Ok(crate::types::SttResponse {
            text: result.text,
            language,
            confidence: None,
            words,
            duration,
            metadata,
        })
    }
}
