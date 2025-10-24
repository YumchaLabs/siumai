//! Audio executor traits

use crate::error::LlmError;
use crate::transformers::audio::{AudioHttpBody, AudioTransformer};
use crate::types::{SttRequest, TtsRequest};
use std::sync::Arc;

#[async_trait::async_trait]
pub trait AudioExecutor: Send + Sync {
    async fn tts(&self, req: TtsRequest) -> Result<Vec<u8>, LlmError>;
    async fn stt(&self, req: SttRequest) -> Result<String, LlmError>;
}

pub struct HttpAudioExecutor {
    pub provider_id: String,
    pub http_client: reqwest::Client,
    pub transformer: Arc<dyn AudioTransformer>,
    pub provider_spec: Arc<dyn crate::provider_core::ProviderSpec>,
    pub provider_context: crate::provider_core::ProviderContext,
}

#[async_trait::async_trait]
impl AudioExecutor for HttpAudioExecutor {
    async fn tts(&self, req: TtsRequest) -> Result<Vec<u8>, LlmError> {
        // Capability guard
        let caps = self.provider_spec.capabilities();
        if !caps.supports("audio") {
            return Err(LlmError::UnsupportedOperation(
                "Text-to-speech is not supported by this provider".to_string(),
            ));
        }
        let body = self.transformer.build_tts_body(&req)?;
        let base_url = self.provider_spec.audio_base_url(&self.provider_context);
        let url = format!("{}{}", base_url, self.transformer.tts_endpoint());

        let mut headers = self.provider_spec.build_headers(&self.provider_context)?;
        crate::utils::http_headers::inject_tracing_headers(&mut headers);

        let builder = self.http_client.post(url).headers(headers);
        let resp = match body {
            AudioHttpBody::Json(json) => builder.json(&json).send().await,
            AudioHttpBody::Multipart(form) => builder.multipart(form).send().await,
        }
        .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: text,
                details: None,
            });
        }

        let bytes = resp
            .bytes()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;
        Ok(bytes.to_vec())
    }

    async fn stt(&self, req: SttRequest) -> Result<String, LlmError> {
        let caps = self.provider_spec.capabilities();
        if !caps.supports("audio") {
            return Err(LlmError::UnsupportedOperation(
                "Speech-to-text is not supported by this provider".to_string(),
            ));
        }
        let body = self.transformer.build_stt_body(&req)?;
        let base_url = self.provider_spec.audio_base_url(&self.provider_context);
        let url = format!("{}{}", base_url, self.transformer.stt_endpoint());

        let mut headers = self.provider_spec.build_headers(&self.provider_context)?;
        crate::utils::http_headers::inject_tracing_headers(&mut headers);

        let builder = self.http_client.post(url).headers(headers);
        let resp = match body {
            AudioHttpBody::Json(json) => builder.json(&json).send().await,
            AudioHttpBody::Multipart(form) => builder.multipart(form).send().await,
        }
        .map_err(|e| LlmError::HttpError(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(LlmError::ApiError {
                code: status.as_u16(),
                message: text,
                details: None,
            });
        }
        let text = resp
            .text()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        // Use parse_json_with_repair for automatic JSON repair when enabled
        let json: serde_json::Value = crate::streaming::parse_json_with_repair(&text)
            .map_err(|e| LlmError::ParseError(e.to_string()))?;
        self.transformer.parse_stt_response(&json)
    }
}
