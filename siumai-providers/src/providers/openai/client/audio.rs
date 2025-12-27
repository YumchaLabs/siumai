use super::OpenAiClient;
use crate::error::LlmError;
use crate::traits::AudioCapability;
use crate::types::AudioTranslationRequest;
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

    async fn translate_audio(
        &self,
        request: AudioTranslationRequest,
    ) -> Result<crate::types::SttResponse, LlmError> {
        use crate::execution::executors::common::{execute_multipart_bytes_request, HttpExecutionConfig};
        use reqwest::multipart::{Form, Part};
        use secrecy::ExposeSecret;
        use std::sync::Arc;

        let spec = Arc::new(crate::providers::openai::spec::OpenAiSpec::new());
        let ctx = crate::core::ProviderContext::new(
            "openai",
            self.base_url.clone(),
            Some(self.api_key.expose_secret().to_string()),
            self.http_config.headers.clone(),
        )
        .with_org_project(self.organization.clone(), self.project.clone());

        let config = HttpExecutionConfig {
            provider_id: "openai".to_string(),
            http_client: self.http_client.clone(),
            provider_spec: spec,
            provider_context: ctx,
            interceptors: self.http_interceptors.clone(),
            retry_options: self.retry_options.clone(),
        };

        // Allow users to pass either raw bytes or a file path (mirrors the STT executor behavior).
        let mut req = request;
        if req.audio_data.is_none()
            && let Some(path) = req.file_path.as_deref()
        {
            let bytes = tokio::fs::read(path)
                .await
                .map_err(|e| LlmError::IoError(format!("Failed to read audio file '{path}': {e}")))?;
            req.audio_data = Some(bytes);
        }

        let audio = req.audio_data.clone().ok_or_else(|| {
            LlmError::InvalidInput("audio_data or file_path is required for audio translation".into())
        })?;

        let model = req.model.clone().unwrap_or_else(|| "whisper-1".to_string());
        let ext = req.format.clone().unwrap_or_else(|| "mp3".to_string());

        // Provider-specific option lookup (OpenAI accepts prompt/response_format/temperature here).
        let lookup = |key: &str| -> Option<&serde_json::Value> {
            match &req.provider_options {
                crate::types::ProviderOptions::Custom { provider_id, options }
                    if provider_id == "openai" =>
                {
                    options.get(key).or_else(|| req.extra_params.get(key))
                }
                _ => req.extra_params.get(key),
            }
        };

        let response_format = lookup("response_format")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "json".to_string());
        let prompt = lookup("prompt").and_then(|v| v.as_str()).map(|s| s.to_string());
        let temperature = lookup("temperature").and_then(|v| v.as_f64());
        let media_type = req.media_type.clone();

        let build_form = || -> Result<Form, LlmError> {
            let file_name = format!("audio.{ext}");
            let mut part = Part::bytes(audio.clone()).file_name(file_name);
            if let Some(mt) = media_type.as_deref() {
                part = part
                    .mime_str(mt)
                    .map_err(|e| LlmError::InvalidParameter(format!("Invalid media_type '{mt}': {e}")))?;
            }

            let mut form = Form::new().part("file", part).text("model", model.clone());

            if let Some(p) = prompt.as_deref() {
                form = form.text("prompt", p.to_string());
            }

            form = form.text("response_format", response_format.clone());

            if let Some(temp) = temperature {
                form = form.text("temperature", temp.to_string());
            }

            Ok(form)
        };

        let base = config.provider_context.base_url.trim_end_matches('/');
        let url = format!("{base}/audio/translations");

        let per_request_headers = req.http_config.as_ref().map(|hc| &hc.headers);
        let res = execute_multipart_bytes_request(&config, &url, build_form, per_request_headers).await?;

        let mut metadata = std::collections::HashMap::new();

        if response_format.ends_with("json") {
            let json: serde_json::Value = serde_json::from_slice(&res.bytes).map_err(|e| {
                LlmError::ParseError(format!("Invalid OpenAI audio translation JSON response: {e}"))
            })?;

            if let Some(usage) = json.get("usage") {
                metadata.insert("usage".to_string(), usage.clone());
            }
            if let Some(segments) = json.get("segments") {
                metadata.insert("segments".to_string(), segments.clone());
            }

            let text = json
                .get("text")
                .and_then(|v| v.as_str())
                .ok_or_else(|| LlmError::ParseError("Missing 'text' field in translation response".into()))?
                .to_string();

            return Ok(crate::types::SttResponse {
                text,
                language: json.get("language").and_then(|v| v.as_str()).map(|s| s.to_string()),
                confidence: None,
                words: None,
                duration: json.get("duration").and_then(|v| v.as_f64()).map(|d| d as f32),
                metadata,
            });
        }

        let text = String::from_utf8(res.bytes)
            .map_err(|e| LlmError::ParseError(format!("Invalid UTF-8 translation response: {e}")))?;

        Ok(crate::types::SttResponse {
            text,
            language: None,
            confidence: None,
            words: None,
            duration: None,
            metadata,
        })
    }
}
