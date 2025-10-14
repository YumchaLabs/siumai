//! Audio executor traits (Phase 0 scaffolding)

use crate::error::LlmError;
use crate::transformers::audio::{AudioHttpBody, AudioTransformer};
use crate::types::{SttRequest, TtsRequest};
use reqwest::header::HeaderMap;
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
    pub build_base_url: Box<dyn Fn() -> String + Send + Sync>,
    pub build_headers: Box<dyn Fn() -> Result<HeaderMap, LlmError> + Send + Sync>,
}

#[async_trait::async_trait]
impl AudioExecutor for HttpAudioExecutor {
    async fn tts(&self, req: TtsRequest) -> Result<Vec<u8>, LlmError> {
        let body = self.transformer.build_tts_body(&req)?;
        let url = format!(
            "{}{}",
            (self.build_base_url)(),
            self.transformer.tts_endpoint()
        );
        let headers = (self.build_headers)()?;

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
        let body = self.transformer.build_stt_body(&req)?;
        let url = format!(
            "{}{}",
            (self.build_base_url)(),
            self.transformer.stt_endpoint()
        );
        let headers = (self.build_headers)()?;

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
        let json: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| LlmError::ParseError(e.to_string()))?;
        self.transformer.parse_stt_response(&json)
    }
}
