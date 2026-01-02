//! OpenAI transcription streaming (SSE) extension implementation.
//!
//! This is intentionally not part of the Vercel-aligned unified `TranscriptionCapability`.
//! Users should access this via `siumai::provider_ext::openai::transcription_streaming::*`.

use super::OpenAiClient;
use crate::error::LlmError;
use crate::execution::http::interceptor::{HttpRequestContext, generate_request_id};
use crate::types::SttRequest;
use futures_util::StreamExt;
use std::pin::Pin;
use std::sync::Arc;

/// OpenAI transcription streaming event (SSE).
///
/// This mirrors OpenAI's `CreateTranscriptionResponseStreamEvent` schema at a pragmatic level:
/// - `transcript.text.delta`
/// - `transcript.text.segment` (only when `response_format=diarized_json`)
/// - `transcript.text.done`
#[derive(Debug, Clone)]
pub enum OpenAiTranscriptionStreamEvent {
    TextDelta {
        delta: String,
        logprobs: Option<serde_json::Value>,
    },
    Segment {
        id: String,
        start: f32,
        end: f32,
        text: String,
        speaker: Option<String>,
    },
    Done {
        text: Option<String>,
        usage: Option<serde_json::Value>,
        logprobs: Option<serde_json::Value>,
    },
    /// Forward-compatible custom event (raw JSON).
    Custom {
        event_type: String,
        data: serde_json::Value,
    },
}

pub type OpenAiTranscriptionStream = Pin<
    Box<
        dyn futures_util::Stream<Item = Result<OpenAiTranscriptionStreamEvent, LlmError>>
            + Send
            + Sync,
    >,
>;

impl OpenAiClient {
    /// Stream OpenAI STT transcript using SSE (`stream=true`).
    ///
    /// Notes:
    /// - This is a provider extension (not part of the unified surface).
    /// - The request is sent to `POST /audio/transcriptions` with `multipart/form-data`.
    pub(crate) async fn stt_sse_stream(
        &self,
        request: SttRequest,
    ) -> Result<OpenAiTranscriptionStream, LlmError> {
        use crate::core::ProviderSpec;
        use crate::execution::transformers::audio::AudioHttpBody;

        let wiring = self.http_wiring();
        let ctx = wiring.provider_context.clone();
        let spec: Arc<dyn ProviderSpec> =
            Arc::new(crate::providers::openai::spec::OpenAiSpec::new());

        // Allow users to pass either raw bytes or a file path.
        let mut request = request;
        if request.audio_data.is_none()
            && let Some(path) = request.file_path.as_deref()
        {
            let bytes = tokio::fs::read(path).await.map_err(|e| {
                LlmError::IoError(format!("Failed to read audio file '{path}': {e}"))
            })?;
            request.audio_data = Some(bytes);
        }

        // Build multipart form using the existing transformer, but enforce streaming at the HTTP layer.
        // (The unified transformer explicitly rejects `extra_params["stream"]=true`.)
        request.extra_params.remove("stream");
        let audio_tx = spec.choose_audio_transformer(&ctx).transformer;

        let base = spec.audio_base_url(&ctx);
        let url = format!("{}/audio/transcriptions", base.trim_end_matches('/'));

        let request_id = generate_request_id();
        let http_ctx = HttpRequestContext {
            request_id,
            provider_id: "openai".to_string(),
            url: url.clone(),
            stream: true,
        };

        // Per-request headers (SSE + user headers).
        let mut per_request_headers = request
            .http_config
            .as_ref()
            .map(|hc| hc.headers.clone())
            .unwrap_or_default();
        per_request_headers.insert("accept".to_string(), "text/event-stream".to_string());
        per_request_headers.insert("cache-control".to_string(), "no-cache".to_string());
        per_request_headers.insert("connection".to_string(), "keep-alive".to_string());
        if self.http_config.stream_disable_compression {
            per_request_headers.insert("accept-encoding".to_string(), "identity".to_string());
        }
        let config = wiring.config(spec.clone());

        let request_for_form = request.clone();
        let audio_tx_for_form = audio_tx.clone();
        let build_form = move || -> Result<reqwest::multipart::Form, LlmError> {
            let body = audio_tx_for_form.build_stt_body(&request_for_form)?;
            let AudioHttpBody::Multipart(form) = body else {
                return Err(LlmError::InvalidParameter(
                    "OpenAI STT streaming expects multipart/form-data request body".to_string(),
                ));
            };
            Ok(super::sse_helpers::openai_stt_force_stream_true(form))
        };

        let resp = crate::execution::executors::common::execute_multipart_request_streaming_response_with_ctx(
            &config,
            &url,
            build_form,
            Some(&per_request_headers),
            http_ctx.clone(),
        )
        .await?;

        // Validate SSE content-type (avoid silently treating JSON as SSE).
        super::sse_helpers::ensure_openai_sse_content_type("transcription", resp.headers())?;

        let bytes_stream = resp
            .bytes_stream()
            .map(|chunk| chunk.map_err(|e| LlmError::HttpError(format!("Stream error: {e}"))));
        let mut json_stream = crate::streaming::stream_sse_json_values(
            bytes_stream,
            self.http_interceptors.clone(),
            http_ctx.clone(),
            super::sse_helpers::openai_sse_json_config("transcription"),
        );

        let stream = async_stream::stream! {
            while let Some(item) = json_stream.next().await {
                let payload = match item {
                    Ok(v) => v,
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                };
                if let Some(err) =
                    super::sse_helpers::openai_sse_error_event("transcription", &payload)
                {
                    yield Err(err);
                    return;
                }

                let Some(kind) = super::sse_helpers::openai_sse_event_type(&payload) else {
                    continue;
                };
                if super::sse_helpers::openai_sse_should_ignore_event_type(kind, &["transcript."]) {
                    continue;
                }

                match kind {
                    "transcript.text.delta" => {
                        match super::sse_helpers::openai_transcript_text_delta(&payload) {
                            Ok(ev) => yield Ok(ev),
                            Err(e) => {
                                yield Err(e);
                                return;
                            }
                        };
                    }
                    "transcript.text.segment" => {
                        match super::sse_helpers::openai_transcript_text_segment(&payload) {
                            Ok(ev) => yield Ok(ev),
                            Err(e) => {
                                yield Err(e);
                                return;
                            }
                        }
                    }
                    "transcript.text.done" => {
                        match super::sse_helpers::openai_transcript_text_done(&payload) {
                            Ok(ev) => yield Ok(ev),
                            Err(e) => {
                                yield Err(e);
                                return;
                            }
                        }
                        return;
                    }
                    _ => {
                        yield Ok(OpenAiTranscriptionStreamEvent::Custom {
                            event_type: kind.to_string(),
                            data: payload,
                        });
                    }
                }
            }

            // If the upstream closes without sending transcript.text.done, emit a best-effort Done.
            yield Ok(OpenAiTranscriptionStreamEvent::Done {
                text: None,
                usage: None,
                logprobs: None,
            });
        };

        Ok(Box::pin(stream))
    }
}
