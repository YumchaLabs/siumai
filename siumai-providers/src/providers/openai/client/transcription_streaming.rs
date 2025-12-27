//! OpenAI transcription streaming (SSE) extension implementation.
//!
//! This is intentionally not part of the Vercel-aligned unified `TranscriptionCapability`.
//! Users should access this via `siumai::provider_ext::openai::transcription_streaming::*`.

use super::OpenAiClient;
use crate::error::LlmError;
use crate::execution::http::interceptor::{generate_request_id, HttpRequestContext};
use crate::streaming::SseStreamExt;
use crate::types::SttRequest;
use futures_util::StreamExt;
use std::collections::HashMap;
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
        use crate::execution::executors::errors as exec_errors;
        use crate::execution::http::headers::merge_headers;
        use crate::execution::transformers::audio::AudioHttpBody;

        let ctx = self.build_context();
        let spec: Arc<dyn ProviderSpec> = Arc::new(crate::providers::openai::spec::OpenAiSpec::new());

        // Allow users to pass either raw bytes or a file path.
        let mut request = request;
        if request.audio_data.is_none()
            && let Some(path) = request.file_path.as_deref()
        {
            let bytes = tokio::fs::read(path)
                .await
                .map_err(|e| LlmError::IoError(format!("Failed to read audio file '{path}': {e}")))?;
            request.audio_data = Some(bytes);
        }

        // Build multipart form using the existing transformer, but enforce streaming at the HTTP layer.
        // (The unified transformer explicitly rejects `extra_params["stream"]=true`.)
        request.extra_params.remove("stream");
        let audio_tx = spec.choose_audio_transformer(&ctx).transformer;
        let body = audio_tx.build_stt_body(&request)?;
        let AudioHttpBody::Multipart(form) = body else {
            return Err(LlmError::InvalidParameter(
                "OpenAI STT streaming expects multipart/form-data request body".to_string(),
            ));
        };
        let form = form.text("stream", "true");

        let base = spec.audio_base_url(&ctx);
        let url = format!("{}/audio/transcriptions", base.trim_end_matches('/'));

        let request_id = generate_request_id();
        let http_ctx = HttpRequestContext {
            request_id,
            provider_id: "openai".to_string(),
            url: url.clone(),
            stream: true,
        };

        // Build headers (spec + per-request SSE headers).
        let base_headers = spec.build_headers(&ctx)?;
        let mut per_request_headers = HashMap::new();
        per_request_headers.insert("accept".to_string(), "text/event-stream".to_string());
        per_request_headers.insert("cache-control".to_string(), "no-cache".to_string());
        per_request_headers.insert("connection".to_string(), "keep-alive".to_string());
        if self.http_config.stream_disable_compression {
            per_request_headers.insert("accept-encoding".to_string(), "identity".to_string());
        }
        let mut effective_headers = merge_headers(base_headers, &per_request_headers);
        // Multipart must own its boundary-based Content-Type.
        effective_headers.remove(reqwest::header::CONTENT_TYPE);

        let mut rb = self
            .http_client
            .post(url.clone())
            .headers(effective_headers.clone())
            .multipart(form);

        // Interceptors (before send)
        {
            let empty_json = serde_json::json!({});
            let cloned_headers = rb
                .try_clone()
                .and_then(|req| req.build().ok().map(|r| r.headers().clone()))
                .unwrap_or_else(|| effective_headers.clone());
            for it in &self.http_interceptors {
                rb = it.on_before_send(&http_ctx, rb, &empty_json, &cloned_headers)?;
            }
        }

        let mut resp = rb
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        // 401 retry once (rebuild headers and multipart form)
        if !resp.status().is_success() {
            let should_retry_401 = self
                .retry_options
                .as_ref()
                .map(|opts| opts.retry_401)
                .unwrap_or(true);
            if resp.status().as_u16() == 401 && should_retry_401 {
                for interceptor in &self.http_interceptors {
                    interceptor.on_retry(
                        &http_ctx,
                        &LlmError::HttpError("401 Unauthorized".into()),
                        1,
                    );
                }

                let retry_headers = spec.build_headers(&ctx)?;
                let mut retry_effective_headers = merge_headers(retry_headers, &per_request_headers);
                retry_effective_headers.remove(reqwest::header::CONTENT_TYPE);

                // Rebuild multipart form (consumed by reqwest).
                let body_retry = audio_tx.build_stt_body(&request)?;
                let AudioHttpBody::Multipart(form_retry) = body_retry else {
                    return Err(LlmError::InvalidParameter(
                        "OpenAI STT streaming expects multipart/form-data request body".to_string(),
                    ));
                };
                let form_retry = form_retry.text("stream", "true");

                let mut rb_retry = self
                    .http_client
                    .post(url.clone())
                    .headers(retry_effective_headers.clone())
                    .multipart(form_retry);

                let empty_json = serde_json::json!({});
                let cloned_headers = rb_retry
                    .try_clone()
                    .and_then(|req| req.build().ok().map(|r| r.headers().clone()))
                    .unwrap_or_else(|| retry_effective_headers.clone());
                for it in &self.http_interceptors {
                    rb_retry = it.on_before_send(&http_ctx, rb_retry, &empty_json, &cloned_headers)?;
                }

                #[cfg(test)]
                {
                    rb_retry = rb_retry.header("x-retry-attempt", "1");
                }

                resp = rb_retry
                    .send()
                    .await
                    .map_err(|e| LlmError::HttpError(e.to_string()))?;
            }
        }

        // Error classification (read text on non-success)
        if !resp.status().is_success() {
            let err = exec_errors::classify_error_with_text(
                "openai",
                resp,
                &http_ctx,
                &self.http_interceptors,
            )
            .await;
            return Err(err);
        }

        for it in &self.http_interceptors {
            it.on_response(&http_ctx, &resp)?;
        }

        // Validate SSE content-type (avoid silently treating JSON as SSE).
        let is_sse = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|ct| ct.to_ascii_lowercase().contains("text/event-stream"))
            .unwrap_or(false);
        if !is_sse {
            let ct = resp
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("<missing>");
            return Err(LlmError::ParseError(format!(
                "Expected 'text/event-stream' for OpenAI transcription SSE streaming, got '{ct}'."
            )));
        }

        let interceptors = self.http_interceptors.clone();
        let http_ctx_for_stream = http_ctx.clone();

        let bytes_stream = resp
            .bytes_stream()
            .map(|chunk| chunk.map_err(|e| LlmError::HttpError(format!("Stream error: {e}"))));
        let mut sse_stream = bytes_stream.into_sse_stream();

        let stream = async_stream::stream! {
            while let Some(item) = sse_stream.next().await {
                let event = match item {
                    Ok(ev) => ev,
                    Err(e) => {
                        yield Err(LlmError::StreamError(format!("SSE stream error: {e}")));
                        return;
                    }
                };

                for it in &interceptors {
                    if let Err(e) = it.on_sse_event(&http_ctx_for_stream, &event) {
                        yield Err(e);
                        return;
                    }
                }

                let data = event.data.trim();
                if data.is_empty() || data == "[DONE]" {
                    continue;
                }

                let payload: serde_json::Value = match crate::streaming::parse_json_with_repair(data) {
                    Ok(v) => v,
                    Err(e) => {
                        yield Err(LlmError::ParseError(format!("Failed to parse OpenAI transcription SSE JSON: {e}")));
                        return;
                    }
                };

                if let Some(err) = payload.get("error") {
                    yield Err(LlmError::ApiError {
                        code: 200,
                        message: "OpenAI transcription SSE error event".to_string(),
                        details: Some(err.clone()),
                    });
                    return;
                }

                let Some(kind) = payload.get("type").and_then(|v| v.as_str()) else {
                    continue;
                };

                match kind {
                    "transcript.text.delta" => {
                        let Some(delta) = payload.get("delta").and_then(|v| v.as_str()) else {
                            yield Err(LlmError::ParseError("Missing 'delta' field in transcript.text.delta event".to_string()));
                            return;
                        };
                        let logprobs = payload.get("logprobs").cloned();
                        yield Ok(OpenAiTranscriptionStreamEvent::TextDelta { delta: delta.to_string(), logprobs });
                    }
                    "transcript.text.segment" => {
                        let Some(id) = payload.get("id").and_then(|v| v.as_str()) else {
                            yield Err(LlmError::ParseError("Missing 'id' field in transcript.text.segment event".to_string()));
                            return;
                        };
                        let Some(start) = payload.get("start").and_then(|v| v.as_f64()) else {
                            yield Err(LlmError::ParseError("Missing 'start' field in transcript.text.segment event".to_string()));
                            return;
                        };
                        let Some(end) = payload.get("end").and_then(|v| v.as_f64()) else {
                            yield Err(LlmError::ParseError("Missing 'end' field in transcript.text.segment event".to_string()));
                            return;
                        };
                        let Some(text) = payload.get("text").and_then(|v| v.as_str()) else {
                            yield Err(LlmError::ParseError("Missing 'text' field in transcript.text.segment event".to_string()));
                            return;
                        };
                        let speaker = payload.get("speaker").and_then(|v| v.as_str()).map(|s| s.to_string());
                        yield Ok(OpenAiTranscriptionStreamEvent::Segment {
                            id: id.to_string(),
                            start: start as f32,
                            end: end as f32,
                            text: text.to_string(),
                            speaker,
                        });
                    }
                    "transcript.text.done" => {
                        let text = payload.get("text").and_then(|v| v.as_str()).map(|s| s.to_string());
                        let usage = payload.get("usage").cloned();
                        let logprobs = payload.get("logprobs").cloned();
                        yield Ok(OpenAiTranscriptionStreamEvent::Done { text, usage, logprobs });
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
