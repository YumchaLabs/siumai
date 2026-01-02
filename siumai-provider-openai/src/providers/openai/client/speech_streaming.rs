//! OpenAI speech streaming (SSE) extension implementation.
//!
//! This is intentionally not part of the Vercel-aligned unified `SpeechCapability`.
//! Users should access this via `siumai::provider_ext::openai::speech_streaming::*`.

use super::OpenAiClient;
use crate::error::LlmError;
use crate::execution::http::interceptor::{HttpRequestContext, generate_request_id};
use crate::types::{AudioStream, AudioStreamEvent, TtsRequest};
use futures_util::StreamExt;
use std::collections::HashMap;
use std::sync::Arc;

impl OpenAiClient {
    /// Stream OpenAI TTS audio using SSE (`stream_format: "sse"`).
    ///
    /// Notes:
    /// - This is a provider extension (not part of the unified surface).
    /// - Requires models that support SSE streaming (OpenAI docs: not supported for `tts-1` / `tts-1-hd`).
    pub(crate) async fn tts_sse_stream(
        &self,
        request: TtsRequest,
    ) -> Result<AudioStream, LlmError> {
        use crate::core::ProviderSpec;

        let wiring = self.http_wiring();
        let ctx = wiring.provider_context.clone();
        let spec: Arc<dyn ProviderSpec> =
            Arc::new(crate::providers::openai::spec::OpenAiSpec::new());

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

        super::sse_helpers::openai_tts_force_stream_format_sse(&mut json);

        let base = spec.audio_base_url(&ctx);
        let url = format!("{}/audio/speech", base.trim_end_matches('/'));

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

        let resp =
            crate::execution::executors::common::execute_json_request_streaming_response_with_ctx(
                &config,
                &url,
                json,
                Some(&per_request_headers),
                http_ctx.clone(),
            )
            .await?;

        // Validate SSE content-type (avoid silently treating binary as SSE).
        super::sse_helpers::ensure_openai_sse_content_type("speech", resp.headers())?;

        let format = request.format.clone().unwrap_or_else(|| "mp3".to_string());

        let bytes_stream = resp
            .bytes_stream()
            .map(|chunk| chunk.map_err(|e| LlmError::HttpError(format!("Stream error: {e}"))));
        let mut json_stream = crate::streaming::stream_sse_json_values(
            bytes_stream,
            self.http_interceptors.clone(),
            http_ctx.clone(),
            super::sse_helpers::openai_sse_json_config("speech"),
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
                if let Some(err) = super::sse_helpers::openai_sse_error_event("speech", &payload) {
                    yield Err(err);
                    return;
                }

                let Some(kind) = super::sse_helpers::openai_sse_event_type(&payload) else {
                    continue;
                };
                if super::sse_helpers::openai_sse_should_ignore_event_type(kind, &["speech."]) {
                    continue;
                }

                match kind {
                    "speech.audio.delta" => {
                        match super::sse_helpers::openai_speech_audio_delta(&payload, &format) {
                            Ok(ev) => yield Ok(ev),
                            Err(e) => {
                                yield Err(e);
                                return;
                            }
                        }
                    }
                    "speech.audio.done" => {
                        match super::sse_helpers::openai_speech_audio_done(&payload) {
                            Ok(ev) => yield Ok(ev),
                            Err(e) => {
                                yield Err(e);
                                return;
                            }
                        }
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
