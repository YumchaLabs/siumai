//! OpenAI speech streaming (SSE) extension implementation.
//!
//! This is intentionally not part of the Vercel-aligned unified `SpeechCapability`.
//! Users should access this via `siumai::provider_ext::openai::speech_streaming::*`.

use super::OpenAiClient;
use crate::error::LlmError;
use crate::execution::http::interceptor::{generate_request_id, HttpRequestContext};
use crate::streaming::SseStreamExt;
use crate::types::{AudioStream, AudioStreamEvent, TtsRequest};
use base64::Engine;
use futures_util::StreamExt;
use std::collections::HashMap;
use std::sync::Arc;

impl OpenAiClient {
    /// Stream OpenAI TTS audio using SSE (`stream_format: "sse"`).
    ///
    /// Notes:
    /// - This is a provider extension (not part of the unified surface).
    /// - Requires models that support SSE streaming (OpenAI docs: not supported for `tts-1` / `tts-1-hd`).
    pub(crate) async fn tts_sse_stream(&self, request: TtsRequest) -> Result<AudioStream, LlmError> {
        use crate::core::ProviderSpec;
        use crate::execution::executors::errors as exec_errors;
        use crate::execution::http::headers::merge_headers;

        let ctx = self.build_context();
        let spec: Arc<dyn ProviderSpec> = Arc::new(crate::providers::openai::spec::OpenAiSpec::new());

        let audio_tx = spec.choose_audio_transformer(&ctx).transformer;
        let body = audio_tx.build_tts_body(&request)?;
        let mut json = match body {
            crate::execution::transformers::audio::AudioHttpBody::Json(v) => v,
            crate::execution::transformers::audio::AudioHttpBody::Multipart(_) => {
                return Err(LlmError::InvalidParameter(
                    "OpenAI TTS streaming expects JSON request body".to_string(),
                ));
            }
        };

        // Ensure the model supports SSE streaming.
        let model = json
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        if model.is_empty() || model == "tts-1" || model == "tts-1-hd" {
            return Err(LlmError::UnsupportedOperation(format!(
                "OpenAI speech SSE streaming requires a model that supports stream_format=sse (got '{model}'). Try 'gpt-4o-mini-tts'."
            )));
        }

        json["stream_format"] = serde_json::Value::String("sse".to_string());

        let base = spec.audio_base_url(&ctx);
        let url = format!("{}/audio/speech", base.trim_end_matches('/'));

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
        let effective_headers = merge_headers(base_headers, &per_request_headers);

        let mut rb = self
            .http_client
            .post(url.clone())
            .headers(effective_headers.clone())
            .json(&json);

        // Interceptors (before send)
        {
            let cloned_headers = rb
                .try_clone()
                .and_then(|req| req.build().ok().map(|r| r.headers().clone()))
                .unwrap_or_else(|| effective_headers.clone());
            for it in &self.http_interceptors {
                rb = it.on_before_send(&http_ctx, rb, &json, &cloned_headers)?;
            }
        }

        let mut resp = rb
            .send()
            .await
            .map_err(|e| LlmError::HttpError(e.to_string()))?;

        // 401 retry once (rebuild headers)
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
                let retry_effective_headers = merge_headers(retry_headers, &per_request_headers);

                let mut rb_retry = self
                    .http_client
                    .post(url.clone())
                    .headers(retry_effective_headers.clone())
                    .json(&json);

                let cloned_headers = rb_retry
                    .try_clone()
                    .and_then(|req| req.build().ok().map(|r| r.headers().clone()))
                    .unwrap_or_else(|| retry_effective_headers.clone());
                for it in &self.http_interceptors {
                    rb_retry = it.on_before_send(&http_ctx, rb_retry, &json, &cloned_headers)?;
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

        // Validate SSE content-type (avoid silently treating binary as SSE).
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
                "Expected 'text/event-stream' for OpenAI speech SSE streaming, got '{ct}'."
            )));
        }

        let format = request
            .format
            .clone()
            .unwrap_or_else(|| "mp3".to_string());
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
                        yield Err(LlmError::ParseError(format!("Failed to parse OpenAI speech SSE JSON: {e}")));
                        return;
                    }
                };

                if let Some(err) = payload.get("error") {
                    yield Err(LlmError::ApiError {
                        code: 200,
                        message: "OpenAI speech SSE error event".to_string(),
                        details: Some(err.clone()),
                    });
                    return;
                }

                let Some(kind) = payload.get("type").and_then(|v| v.as_str()) else {
                    continue;
                };

                match kind {
                    "speech.audio.delta" => {
                        let Some(b64) = payload.get("audio").and_then(|v| v.as_str()) else {
                            yield Err(LlmError::ParseError("Missing 'audio' field in speech.audio.delta event".to_string()));
                            return;
                        };
                        let decoded = match base64::engine::general_purpose::STANDARD.decode(b64) {
                            Ok(b) => b,
                            Err(e) => {
                                yield Err(LlmError::ParseError(format!("Invalid base64 audio chunk: {e}")));
                                return;
                            }
                        };
                        yield Ok(AudioStreamEvent::AudioDelta { data: decoded, format: format.clone() });
                    }
                    "speech.audio.done" => {
                        let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
                        if let Some(usage) = payload.get("usage") {
                            metadata.insert("usage".to_string(), usage.clone());
                        }
                        metadata.insert("provider".to_string(), serde_json::json!("openai"));
                        metadata.insert("event".to_string(), payload.clone());
                        yield Ok(AudioStreamEvent::Done { duration: None, metadata });
                        return;
                    }
                    _ => {
                        // Ignore unknown/forward-compatible events.
                    }
                }
            }

            let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
            metadata.insert("provider".to_string(), serde_json::json!("openai"));
            metadata.insert("reason".to_string(), serde_json::json!("eof"));
            yield Ok(AudioStreamEvent::Done { duration: None, metadata });
        };

        Ok(Box::pin(stream))
    }
}
