//! Audio executor traits

use crate::error::LlmError;
use crate::execution::ExecutionPolicy;
use crate::execution::transformers::audio::{AudioHttpBody, AudioTransformer};
use crate::observability::tracing::ProviderTracer;
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
    pub provider_spec: Arc<dyn crate::core::ProviderSpec>,
    pub provider_context: crate::core::ProviderContext,
    /// Execution policy
    pub policy: ExecutionPolicy,
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

        // Use common bytes/multipart execution with interceptors/retry
        let config = crate::execution::executors::common::HttpExecutionConfig {
            provider_id: self.provider_id.clone(),
            http_client: self.http_client.clone(),
            provider_spec: self.provider_spec.clone(),
            provider_context: self.provider_context.clone(),
            interceptors: self.policy.interceptors.clone(),
            retry_options: self.policy.retry_options.clone(),
        };

        let tracer = ProviderTracer::new(&self.provider_id);
        tracer.trace_request_start("POST", &url);
        let start = std::time::Instant::now();
        match body {
            AudioHttpBody::Json(mut json) => {
                // Apply before_send if present
                if let Some(cb) = &self.policy.before_send {
                    json = cb(&json)?;
                }
                let result = crate::execution::executors::common::execute_bytes_request(
                    &config,
                    &url,
                    crate::execution::executors::common::HttpBody::Json(json),
                    None,
                )
                .await?;
                tracer.trace_request_complete(start, result.bytes.len());
                Ok(result.bytes)
            }
            AudioHttpBody::Multipart(_form) => {
                // Use unified multipart->bytes helper with retry/interceptors
                let transformer = self.transformer.clone();
                let req_cloned = req.clone();
                let build_form = move || match transformer.build_tts_body(&req_cloned)? {
                    AudioHttpBody::Multipart(form) => Ok(form),
                    _ => Err(LlmError::InvalidParameter(
                        "Expected multipart form for TTS".into(),
                    )),
                };

                let result = crate::execution::executors::common::execute_multipart_bytes_request(
                    &config, &url, build_form, None,
                )
                .await?;
                tracer.trace_request_complete(start, result.bytes.len());
                Ok(result.bytes)
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

        let config = crate::execution::executors::common::HttpExecutionConfig {
            provider_id: self.provider_id.clone(),
            http_client: self.http_client.clone(),
            provider_spec: self.provider_spec.clone(),
            provider_context: self.provider_context.clone(),
            interceptors: self.policy.interceptors.clone(),
            retry_options: self.policy.retry_options.clone(),
        };

        let per_request_headers = None;
        let tracer = ProviderTracer::new(&self.provider_id);
        tracer.trace_request_start("POST", &url);
        let start = std::time::Instant::now();
        let result = match body {
            AudioHttpBody::Json(mut json) => {
                // Apply before_send if present
                if let Some(cb) = &self.policy.before_send {
                    json = cb(&json)?;
                }
                crate::execution::executors::common::execute_json_request(
                    &config,
                    &url,
                    crate::execution::executors::common::HttpBody::Json(json),
                    per_request_headers,
                    false,
                )
                .await?
            }
            AudioHttpBody::Multipart(_) => {
                crate::execution::executors::common::execute_multipart_request(
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
        // We don't have raw status/headers here (common path). Mark completion with approximate size.
        let resp_len = serde_json::to_string(&result.json)
            .map(|s| s.len())
            .unwrap_or(0);
        tracer.trace_request_complete(start, resp_len);
        self.transformer.parse_stt_response(&result.json)
    }
}
