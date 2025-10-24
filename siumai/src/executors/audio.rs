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

        // Use common bytes execution (JSON only)
        let config = crate::executors::common::HttpExecutionConfig {
            provider_id: self.provider_id.clone(),
            http_client: self.http_client.clone(),
            provider_spec: self.provider_spec.clone(),
            provider_context: self.provider_context.clone(),
            interceptors: vec![],
            retry_options: None,
        };

        match body {
            AudioHttpBody::Json(json) => {
                let result = crate::executors::common::execute_bytes_request(
                    &config,
                    &url,
                    crate::executors::common::HttpBody::Json(json),
                    None,
                )
                .await?;
                Ok(result.bytes)
            }
            AudioHttpBody::Multipart(form) => {
                // Fallback to direct send for multipart
                let mut headers = self.provider_spec.build_headers(&self.provider_context)?;
                crate::utils::http_headers::inject_tracing_headers(&mut headers);
                let resp = self
                    .http_client
                    .post(url)
                    .headers(headers)
                    .multipart(form)
                    .send()
                    .await
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
        }
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

        let config = crate::executors::common::HttpExecutionConfig {
            provider_id: self.provider_id.clone(),
            http_client: self.http_client.clone(),
            provider_spec: self.provider_spec.clone(),
            provider_context: self.provider_context.clone(),
            interceptors: vec![],
            retry_options: None,
        };

        let per_request_headers = None;
        let result = match body {
            AudioHttpBody::Json(json) => {
                crate::executors::common::execute_json_request(
                    &config,
                    &url,
                    crate::executors::common::HttpBody::Json(json),
                    per_request_headers,
                    false,
                )
                .await?
            }
            AudioHttpBody::Multipart(_) => {
                crate::executors::common::execute_multipart_request(
                    &config,
                    &url,
                    || match self.transformer.build_stt_body(&req)? {
                        AudioHttpBody::Multipart(form) => Ok(form),
                        _ => Err(LlmError::InvalidParameter(
                            "Expected multipart form for STT".into(),
                        )),
                    },
                    per_request_headers,
                )
                .await?
            }
        };

        self.transformer.parse_stt_response(&result.json)
    }
}
