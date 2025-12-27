//! Audio executor traits

use crate::error::LlmError;
use crate::execution::ExecutionPolicy;
use crate::execution::transformers::audio::{AudioHttpBody, AudioTransformer};
use crate::types::{SttRequest, TtsRequest};
use std::sync::Arc;

/// TTS execution result with audio data and metadata
#[derive(Debug, Clone)]
pub struct TtsExecutionResult {
    /// Audio bytes
    pub audio_data: Vec<u8>,
    /// Duration in seconds (if available)
    pub duration: Option<f32>,
    /// Sample rate in Hz (if available)
    pub sample_rate: Option<u32>,
}

/// STT execution result with transcript and raw provider payload
#[derive(Debug, Clone)]
pub struct SttExecutionResult {
    /// Transcribed text
    pub text: String,
    /// Raw JSON payload returned by the provider (best-effort)
    pub raw: serde_json::Value,
}

#[async_trait::async_trait]
pub trait AudioExecutor: Send + Sync {
    async fn tts(&self, req: TtsRequest) -> Result<TtsExecutionResult, LlmError>;
    async fn stt(&self, req: SttRequest) -> Result<SttExecutionResult, LlmError>;
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

/// Builder for creating HttpAudioExecutor instances
pub struct AudioExecutorBuilder {
    provider_id: String,
    http_client: reqwest::Client,
    spec: Option<Arc<dyn crate::core::ProviderSpec>>,
    context: Option<crate::core::ProviderContext>,
    transformer: Option<Arc<dyn AudioTransformer>>,
    policy: crate::execution::ExecutionPolicy,
}

impl AudioExecutorBuilder {
    pub fn new(provider_id: impl Into<String>, http_client: reqwest::Client) -> Self {
        Self {
            provider_id: provider_id.into(),
            http_client,
            spec: None,
            context: None,
            transformer: None,
            policy: crate::execution::ExecutionPolicy::new(),
        }
    }

    pub fn with_spec(mut self, spec: Arc<dyn crate::core::ProviderSpec>) -> Self {
        self.spec = Some(spec);
        self
    }

    pub fn with_context(mut self, context: crate::core::ProviderContext) -> Self {
        self.context = Some(context);
        self
    }

    pub fn with_transformer(mut self, transformer: Arc<dyn AudioTransformer>) -> Self {
        self.transformer = Some(transformer);
        self
    }

    pub fn with_before_send(mut self, hook: crate::execution::executors::BeforeSendHook) -> Self {
        self.policy.before_send = Some(hook);
        self
    }

    pub fn with_interceptors(
        mut self,
        interceptors: Vec<Arc<dyn crate::execution::http::interceptor::HttpInterceptor>>,
    ) -> Self {
        self.policy.interceptors = interceptors;
        self
    }

    pub fn with_retry_options(mut self, retry_options: crate::retry_api::RetryOptions) -> Self {
        self.policy.retry_options = Some(retry_options);
        self
    }

    pub fn build(self) -> Arc<HttpAudioExecutor> {
        let spec = self.spec.expect("provider_spec is required");
        let context = self.context.expect("provider_context is required");
        let transformer = match self.transformer {
            Some(t) => t,
            None => spec.choose_audio_transformer(&context).transformer,
        };
        Arc::new(HttpAudioExecutor {
            provider_id: self.provider_id,
            http_client: self.http_client,
            transformer,
            provider_spec: spec,
            provider_context: context,
            policy: self.policy,
        })
    }
}

#[async_trait::async_trait]
impl AudioExecutor for HttpAudioExecutor {
    async fn tts(&self, req: TtsRequest) -> Result<TtsExecutionResult, LlmError> {
        // Capability guard
        let caps = self.provider_spec.capabilities();
        if !caps.supports("speech") {
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

        let raw_bytes = match body {
            AudioHttpBody::Json(mut json) => {
                // Apply before_send if present
                if let Some(cb) = &self.policy.before_send {
                    json = cb(&json)?;
                }
                let result = crate::execution::executors::common::execute_bytes_request(
                    &config,
                    &url,
                    crate::execution::executors::common::HttpBody::Json(json),
                    req.http_config.as_ref().map(|hc| &hc.headers),
                )
                .await?;
                result.bytes
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
                    &config,
                    &url,
                    build_form,
                    req.http_config.as_ref().map(|hc| &hc.headers),
                )
                .await?;
                result.bytes
            }
        };

        // Parse response using transformer
        // This allows providers to handle different response formats
        // (e.g., binary audio vs JSON with encoded audio)
        let audio_data = self.transformer.parse_tts_response(raw_bytes.clone())?;

        // Extract metadata if response is JSON
        let (duration, sample_rate) = if self.transformer.tts_response_is_json() {
            // Try to parse as JSON to extract metadata
            if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&raw_bytes) {
                self.transformer.parse_tts_metadata(&json)?
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        Ok(TtsExecutionResult {
            audio_data,
            duration,
            sample_rate,
        })
    }

    async fn stt(&self, req: SttRequest) -> Result<SttExecutionResult, LlmError> {
        let caps = self.provider_spec.capabilities();
        if !caps.supports("transcription") {
            return Err(LlmError::UnsupportedOperation(
                "Speech-to-text is not supported by this provider".to_string(),
            ));
        }

        // Allow users to pass either raw bytes or a file path (common in provider docs).
        // We load the file here (async) so transformer implementations can stay sync.
        let mut req = req;
        if req.audio_data.is_none()
            && let Some(path) = req.file_path.as_deref()
        {
            let bytes = tokio::fs::read(path)
                .await
                .map_err(|e| LlmError::IoError(format!("Failed to read audio file '{path}': {e}")))?;
            req.audio_data = Some(bytes);
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

        let per_request_headers = req.http_config.as_ref().map(|hc| &hc.headers);
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
        let text = self.transformer.parse_stt_response(&result.json)?;
        Ok(SttExecutionResult {
            text,
            raw: result.json,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::transformers::audio::AudioHttpBody;
    use crate::traits::ProviderCapabilities;
    use reqwest::header::HeaderMap;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[derive(Clone)]
    struct SupportsTranscriptionSpec;

    impl crate::core::ProviderSpec for SupportsTranscriptionSpec {
        fn id(&self) -> &'static str {
            "test"
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new().with_transcription()
        }

        fn build_headers(&self, _ctx: &crate::core::ProviderContext) -> Result<HeaderMap, LlmError> {
            Ok(HeaderMap::new())
        }

        fn chat_url(&self, _stream: bool, _req: &crate::types::ChatRequest, _ctx: &crate::core::ProviderContext) -> String {
            unreachable!()
        }

        fn choose_chat_transformers(
            &self,
            _req: &crate::types::ChatRequest,
            _ctx: &crate::core::ProviderContext,
        ) -> crate::core::ChatTransformers {
            unreachable!()
        }
    }

    #[derive(Clone)]
    struct ExpectFileLoadedTransformer {
        expected: Vec<u8>,
    }

    impl AudioTransformer for ExpectFileLoadedTransformer {
        fn provider_id(&self) -> &str {
            "test"
        }

        fn build_tts_body(&self, _req: &TtsRequest) -> Result<AudioHttpBody, LlmError> {
            unreachable!()
        }

        fn build_stt_body(&self, req: &SttRequest) -> Result<AudioHttpBody, LlmError> {
            let got = req
                .audio_data
                .as_ref()
                .ok_or_else(|| LlmError::InvalidInput("audio_data should be loaded".into()))?;
            assert_eq!(got, &self.expected);
            Err(LlmError::InvalidInput("stop after assertion".into()))
        }

        fn tts_endpoint(&self) -> &str {
            unreachable!()
        }

        fn stt_endpoint(&self) -> &str {
            "/audio/transcriptions"
        }

        fn parse_stt_response(&self, _json: &serde_json::Value) -> Result<String, LlmError> {
            unreachable!()
        }
    }

    #[tokio::test]
    async fn stt_loads_file_path_into_audio_data_before_transformer() {
        let tmp = tempfile::NamedTempFile::new().expect("temp file");
        std::fs::write(tmp.path(), b"abc").expect("write");

        let exec = HttpAudioExecutor {
            provider_id: "test".to_string(),
            http_client: reqwest::Client::new(),
            transformer: Arc::new(ExpectFileLoadedTransformer {
                expected: b"abc".to_vec(),
            }),
            provider_spec: Arc::new(SupportsTranscriptionSpec),
            provider_context: crate::core::ProviderContext::new(
                "test",
                "https://example.invalid",
                None,
                HashMap::new(),
            ),
            policy: ExecutionPolicy::new(),
        };

        let req = SttRequest::from_file(tmp.path().to_string_lossy().to_string());
        let err = exec.stt(req).await.expect_err("should short-circuit");
        assert!(matches!(err, LlmError::InvalidInput(_)));
    }
}
