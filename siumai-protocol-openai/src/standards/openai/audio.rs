//! OpenAI Audio API Standard
//!
//! This module provides a reusable `AudioTransformer` implementation for providers
//! that follow OpenAI's TTS/STT endpoints:
//! - `POST /audio/speech` (TTS)
//! - `POST /audio/transcriptions` (STT)

use crate::error::LlmError;
use crate::execution::transformers::audio::{AudioHttpBody, AudioTransformer};
use crate::types::{AudioStreamEvent, ProviderOptionsMap, SttRequest, TtsRequest};
use std::borrow::Cow;
use std::pin::Pin;

#[derive(Debug, Clone)]
pub struct OpenAiAudioDefaults {
    pub tts_model: Cow<'static, str>,
    pub tts_voice: Cow<'static, str>,
    pub tts_format: Cow<'static, str>,
    pub tts_speed: Option<f32>,
    pub tts_include_language: bool,

    pub stt_model: Cow<'static, str>,
    pub stt_file_name: Cow<'static, str>,
    pub stt_response_format: Option<Cow<'static, str>>,
    pub stt_include_language: bool,
    pub stt_include_timestamp_granularities: bool,
}

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

impl Default for OpenAiAudioDefaults {
    fn default() -> Self {
        Self {
            tts_model: Cow::Borrowed("tts-1"),
            tts_voice: Cow::Borrowed("alloy"),
            tts_format: Cow::Borrowed("mp3"),
            tts_speed: None,
            tts_include_language: false,
            stt_model: Cow::Borrowed("whisper-1"),
            stt_file_name: Cow::Borrowed("audio.mp3"),
            stt_response_format: Some(Cow::Borrowed("json")),
            stt_include_language: true,
            stt_include_timestamp_granularities: true,
        }
    }
}

fn lookup_extra<'a>(
    provider_id: &str,
    provider_options_map: &'a ProviderOptionsMap,
    extra_params: &'a std::collections::HashMap<String, serde_json::Value>,
    key: &str,
) -> Option<&'a serde_json::Value> {
    lookup_extra_any(provider_id, provider_options_map, extra_params, &[key])
}

pub fn openai_sse_error_event(label: &str, payload: &serde_json::Value) -> Option<LlmError> {
    let err = payload.get("error")?;
    Some(LlmError::ApiError {
        code: 200,
        message: format!("OpenAI {label} SSE error event"),
        details: Some(err.clone()),
    })
}

pub fn openai_sse_json_config(label: &str) -> crate::streaming::SseJsonStreamConfig {
    openai_sse_json_config_with_done_markers(label, &["[DONE]"])
}

pub fn openai_sse_json_config_with_done_markers(
    label: &str,
    done_markers: &[&str],
) -> crate::streaming::SseJsonStreamConfig {
    crate::streaming::SseJsonStreamConfig::new(format!("openai {label}"))
        .with_done_markers(done_markers.iter().copied())
}

pub fn openai_sse_event_type(payload: &serde_json::Value) -> Option<&str> {
    payload.get("type").and_then(|v| v.as_str())
}

pub fn openai_sse_should_ignore_event_type(kind: &str, allowed_prefixes: &[&str]) -> bool {
    if kind.is_empty() {
        return true;
    }
    !allowed_prefixes.iter().any(|p| kind.starts_with(p))
}

pub fn openai_speech_audio_delta(
    payload: &serde_json::Value,
    format: &str,
) -> Result<AudioStreamEvent, LlmError> {
    use base64::Engine;

    let Some(b64) = payload.get("audio").and_then(|v| v.as_str()) else {
        return Err(LlmError::ParseError(
            "Missing 'audio' field in speech.audio.delta event".to_string(),
        ));
    };
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|e| LlmError::ParseError(format!("Invalid base64 audio chunk: {e}")))?;

    Ok(AudioStreamEvent::AudioDelta {
        data: decoded,
        format: format.to_string(),
    })
}

pub fn openai_speech_audio_done(payload: &serde_json::Value) -> Result<AudioStreamEvent, LlmError> {
    let mut metadata: std::collections::HashMap<String, serde_json::Value> =
        std::collections::HashMap::new();
    if let Some(usage) = payload.get("usage") {
        metadata.insert("usage".to_string(), usage.clone());
    }
    metadata.insert("provider".to_string(), serde_json::json!("openai"));
    metadata.insert("event".to_string(), payload.clone());

    Ok(AudioStreamEvent::Done {
        duration: None,
        metadata,
    })
}

pub fn openai_transcript_text_delta(
    payload: &serde_json::Value,
) -> Result<OpenAiTranscriptionStreamEvent, LlmError> {
    let Some(delta) = payload.get("delta").and_then(|v| v.as_str()) else {
        return Err(LlmError::ParseError(
            "Missing 'delta' field in transcript.text.delta event".to_string(),
        ));
    };
    let logprobs = payload.get("logprobs").cloned();
    Ok(OpenAiTranscriptionStreamEvent::TextDelta {
        delta: delta.to_string(),
        logprobs,
    })
}

pub fn openai_transcript_text_segment(
    payload: &serde_json::Value,
) -> Result<OpenAiTranscriptionStreamEvent, LlmError> {
    let Some(id) = payload.get("id").and_then(|v| v.as_str()) else {
        return Err(LlmError::ParseError(
            "Missing 'id' field in transcript.text.segment event".to_string(),
        ));
    };
    let Some(start) = payload.get("start").and_then(|v| v.as_f64()) else {
        return Err(LlmError::ParseError(
            "Missing 'start' field in transcript.text.segment event".to_string(),
        ));
    };
    let Some(end) = payload.get("end").and_then(|v| v.as_f64()) else {
        return Err(LlmError::ParseError(
            "Missing 'end' field in transcript.text.segment event".to_string(),
        ));
    };
    let Some(text) = payload.get("text").and_then(|v| v.as_str()) else {
        return Err(LlmError::ParseError(
            "Missing 'text' field in transcript.text.segment event".to_string(),
        ));
    };
    let speaker = payload
        .get("speaker")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    Ok(OpenAiTranscriptionStreamEvent::Segment {
        id: id.to_string(),
        start: start as f32,
        end: end as f32,
        text: text.to_string(),
        speaker,
    })
}

pub fn openai_transcript_text_done(
    payload: &serde_json::Value,
) -> Result<OpenAiTranscriptionStreamEvent, LlmError> {
    let text = payload
        .get("text")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let usage = payload.get("usage").cloned();
    let logprobs = payload.get("logprobs").cloned();
    Ok(OpenAiTranscriptionStreamEvent::Done {
        text,
        usage,
        logprobs,
    })
}

pub fn openai_tts_force_stream_format_sse(body: &mut serde_json::Value) {
    body["stream_format"] = serde_json::Value::String("sse".to_string());
}

pub fn openai_stt_force_stream_true(form: reqwest::multipart::Form) -> reqwest::multipart::Form {
    form.text("stream", "true")
}

pub fn ensure_openai_sse_content_type(
    label: &str,
    headers: &reqwest::header::HeaderMap,
) -> Result<(), LlmError> {
    let is_sse = headers
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|ct| ct.to_ascii_lowercase().contains("text/event-stream"))
        .unwrap_or(false);
    if is_sse {
        return Ok(());
    }

    let ct = headers
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("<missing>");
    Err(LlmError::ParseError(format!(
        "Expected 'text/event-stream' for OpenAI {label} SSE streaming, got '{ct}'."
    )))
}

fn lookup_extra_any<'a>(
    provider_id: &str,
    provider_options_map: &'a ProviderOptionsMap,
    extra_params: &'a std::collections::HashMap<String, serde_json::Value>,
    keys: &[&str],
) -> Option<&'a serde_json::Value> {
    if let Some(obj) = provider_options_map.get_object(provider_id) {
        for key in keys {
            if let Some(v) = obj.get(*key) {
                return Some(v);
            }
        }
    }
    for key in keys {
        if let Some(v) = extra_params.get(*key) {
            return Some(v);
        }
    }
    None
}

fn build_tts_body_impl(
    req: &TtsRequest,
    defaults: &OpenAiAudioDefaults,
    provider_id: &str,
) -> Result<AudioHttpBody, LlmError> {
    let model = req
        .model
        .clone()
        .unwrap_or_else(|| defaults.tts_model.to_string());
    let voice = req
        .voice
        .clone()
        .unwrap_or_else(|| defaults.tts_voice.to_string());
    let format = req
        .format
        .clone()
        .unwrap_or_else(|| defaults.tts_format.to_string());
    let speed = req
        .speed
        .or_else(|| {
            lookup_extra_any(
                provider_id,
                &req.provider_options_map,
                &req.extra_params,
                &["speed"],
            )
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
        })
        .or(defaults.tts_speed);
    let instructions = req.instructions.clone().or_else(|| {
        lookup_extra(
            provider_id,
            &req.provider_options_map,
            &req.extra_params,
            "instructions",
        )
        .and_then(|v| v.as_str())
        .map(ToOwned::to_owned)
    });
    let language = req.language.clone().or_else(|| {
        lookup_extra_any(
            provider_id,
            &req.provider_options_map,
            &req.extra_params,
            &["language"],
        )
        .and_then(|v| v.as_str())
        .map(ToOwned::to_owned)
    });
    let mut json = serde_json::json!({
        "model": model,
        "input": req.text,
        "voice": voice,
        "response_format": format,
    });
    if let Some(s) = speed {
        json["speed"] = serde_json::json!(s);
    }
    if let Some(instr) = instructions {
        json["instructions"] = serde_json::json!(instr);
    }
    if defaults.tts_include_language
        && let Some(language) = language
    {
        json["language"] = serde_json::json!(language);
    }
    Ok(AudioHttpBody::Json(json))
}

fn build_stt_body_impl(
    req: &SttRequest,
    defaults: &OpenAiAudioDefaults,
    provider_id: &str,
) -> Result<AudioHttpBody, LlmError> {
    let model = req
        .model
        .clone()
        .unwrap_or_else(|| defaults.stt_model.to_string());
    let audio = req
        .audio_bytes()
        .map_err(|err| LlmError::InvalidParameter(format!("Invalid STT audio input: {err}")))?;

    if lookup_extra(
        provider_id,
        &req.provider_options_map,
        &req.extra_params,
        "stream",
    )
    .and_then(|v| v.as_bool())
    .unwrap_or(false)
    {
        return Err(LlmError::UnsupportedOperation(
            "OpenAI STT streaming is not supported in the unified Transcription surface; use provider-specific APIs instead.".to_string(),
        ));
    }

    let mut part =
        reqwest::multipart::Part::bytes(audio).file_name(defaults.stt_file_name.to_string());
    part = part.mime_str(&req.media_type).map_err(|e| {
        LlmError::InvalidParameter(format!("Invalid STT media_type '{}': {e}", req.media_type))
    })?;
    let mut form = reqwest::multipart::Form::new().part("file", part);
    form = form.text("model", model);
    if let Some(fmt) = lookup_extra_any(
        provider_id,
        &req.provider_options_map,
        &req.extra_params,
        &["response_format", "responseFormat"],
    )
    .and_then(|v| v.as_str())
    {
        form = form.text("response_format", fmt.to_string());
    } else if let Some(fmt) = defaults.stt_response_format.as_ref() {
        form = form.text("response_format", fmt.to_string());
    }
    let language = lookup_extra_any(
        provider_id,
        &req.provider_options_map,
        &req.extra_params,
        &["language"],
    )
    .and_then(|v| v.as_str())
    .map(ToOwned::to_owned);
    if defaults.stt_include_language
        && let Some(lang) = language
    {
        form = form.text("language", lang.clone());
    }
    let timestamp_granularities = lookup_extra_any(
        provider_id,
        &req.provider_options_map,
        &req.extra_params,
        &["timestamp_granularities", "timestampGranularities"],
    )
    .and_then(|v| v.as_array())
    .map(|items| {
        items
            .iter()
            .filter_map(|item| item.as_str().map(ToOwned::to_owned))
            .collect::<Vec<_>>()
    })
    .filter(|items| !items.is_empty());
    if defaults.stt_include_timestamp_granularities
        && let Some(grans) = timestamp_granularities
    {
        for g in grans {
            form = form.text("timestamp_granularities[]", g.clone());
        }
    }

    // OpenAI-specific optional fields (escape hatches).
    // Keep these in `extra_params` to avoid expanding the unified surface.
    if let Some(prompt) = lookup_extra(
        provider_id,
        &req.provider_options_map,
        &req.extra_params,
        "prompt",
    )
    .and_then(|v| v.as_str())
    {
        form = form.text("prompt", prompt.to_string());
    }
    if let Some(temp) = lookup_extra(
        provider_id,
        &req.provider_options_map,
        &req.extra_params,
        "temperature",
    )
    .and_then(|v| v.as_f64())
    {
        form = form.text("temperature", temp.to_string());
    }
    if let Some(strategy) = lookup_extra_any(
        provider_id,
        &req.provider_options_map,
        &req.extra_params,
        &["chunking_strategy", "chunkingStrategy"],
    )
    .and_then(|v| v.as_str())
    {
        form = form.text("chunking_strategy", strategy.to_string());
    }
    if let Some(include) = lookup_extra(
        provider_id,
        &req.provider_options_map,
        &req.extra_params,
        "include",
    )
    .and_then(|v| v.as_array())
    {
        for item in include {
            if let Some(s) = item.as_str() {
                form = form.text("include[]", s.to_string());
            }
        }
    }
    if let Some(names) = lookup_extra_any(
        provider_id,
        &req.provider_options_map,
        &req.extra_params,
        &["known_speaker_names", "knownSpeakerNames"],
    )
    .and_then(|v| v.as_array())
    {
        for item in names {
            if let Some(s) = item.as_str() {
                form = form.text("known_speaker_names[]", s.to_string());
            }
        }
    }
    if let Some(refs) = lookup_extra_any(
        provider_id,
        &req.provider_options_map,
        &req.extra_params,
        &["known_speaker_references", "knownSpeakerReferences"],
    )
    .and_then(|v| v.as_array())
    {
        for item in refs {
            if let Some(s) = item.as_str() {
                form = form.text("known_speaker_references[]", s.to_string());
            }
        }
    }

    Ok(AudioHttpBody::Multipart(form))
}

/// OpenAI Audio Transformer for TTS and STT.
#[derive(Clone)]
pub struct OpenAiAudioTransformer;

impl AudioTransformer for OpenAiAudioTransformer {
    fn provider_id(&self) -> &str {
        "openai"
    }

    fn build_tts_body(&self, req: &TtsRequest) -> Result<AudioHttpBody, LlmError> {
        build_tts_body_impl(req, &OpenAiAudioDefaults::default(), "openai")
    }

    fn build_stt_body(&self, req: &SttRequest) -> Result<AudioHttpBody, LlmError> {
        build_stt_body_impl(req, &OpenAiAudioDefaults::default(), "openai")
    }

    fn tts_endpoint(&self) -> &str {
        "/audio/speech"
    }

    fn stt_endpoint(&self) -> &str {
        "/audio/transcriptions"
    }

    fn parse_stt_response(&self, json: &serde_json::Value) -> Result<String, LlmError> {
        json.get("text")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| LlmError::ParseError("Missing 'text' field in STT response".to_string()))
    }
}

/// OpenAI audio transformer with configurable provider id and defaults.
#[derive(Debug, Clone)]
pub struct OpenAiAudioTransformerWithProviderId {
    provider_id: Cow<'static, str>,
    defaults: OpenAiAudioDefaults,
}

impl OpenAiAudioTransformerWithProviderId {
    pub fn new(provider_id: impl Into<Cow<'static, str>>) -> Self {
        Self {
            provider_id: provider_id.into(),
            defaults: OpenAiAudioDefaults::default(),
        }
    }

    pub fn with_defaults(
        provider_id: impl Into<Cow<'static, str>>,
        defaults: OpenAiAudioDefaults,
    ) -> Self {
        Self {
            provider_id: provider_id.into(),
            defaults,
        }
    }
}

impl AudioTransformer for OpenAiAudioTransformerWithProviderId {
    fn provider_id(&self) -> &str {
        self.provider_id.as_ref()
    }

    fn build_tts_body(&self, req: &TtsRequest) -> Result<AudioHttpBody, LlmError> {
        build_tts_body_impl(req, &self.defaults, self.provider_id.as_ref())
    }

    fn build_stt_body(&self, req: &SttRequest) -> Result<AudioHttpBody, LlmError> {
        build_stt_body_impl(req, &self.defaults, self.provider_id.as_ref())
    }

    fn tts_endpoint(&self) -> &str {
        "/audio/speech"
    }

    fn stt_endpoint(&self) -> &str {
        "/audio/transcriptions"
    }

    fn parse_stt_response(&self, json: &serde_json::Value) -> Result<String, LlmError> {
        OpenAiAudioTransformer.parse_stt_response(json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use http_body_util::BodyExt as _;

    fn multipart_body_text(form: reqwest::multipart::Form) -> String {
        let client = reqwest::Client::new();
        let mut request = client
            .post("http://example.invalid")
            .multipart(form)
            .build()
            .expect("build multipart request");
        let body = request.body_mut().take().expect("multipart body");
        let bytes = futures::executor::block_on(async move {
            body.collect()
                .await
                .expect("collect multipart body")
                .to_bytes()
        });
        String::from_utf8(bytes.to_vec()).expect("multipart body utf8")
    }

    #[test]
    fn build_tts_body_defaults() {
        let tx = OpenAiAudioTransformer;
        let req = TtsRequest::new("hello".to_string());
        let body = tx.build_tts_body(&req).unwrap();
        match body {
            AudioHttpBody::Json(j) => {
                assert_eq!(j["model"], "tts-1");
                assert_eq!(j["voice"], "alloy");
                assert_eq!(j["response_format"], "mp3");
                assert_eq!(j["input"], "hello");
            }
            _ => panic!("expected JSON body"),
        }
    }

    #[test]
    fn parse_stt_response_reads_text() {
        let tx = OpenAiAudioTransformer;
        let json = serde_json::json!({ "text": "hi" });
        assert_eq!(tx.parse_stt_response(&json).unwrap(), "hi");
    }

    #[test]
    fn openai_speech_sse_helpers_parse_audio_delta_and_done() {
        let delta = openai_speech_audio_delta(
            &serde_json::json!({
                "type": "speech.audio.delta",
                "audio": "YWJj"
            }),
            "mp3",
        )
        .expect("audio delta");
        match delta {
            AudioStreamEvent::AudioDelta { data, format } => {
                assert_eq!(data, b"abc");
                assert_eq!(format, "mp3");
            }
            other => panic!("expected audio delta, got {other:?}"),
        }

        let done = openai_speech_audio_done(&serde_json::json!({
            "type": "speech.audio.done",
            "usage": { "total_tokens": 3 }
        }))
        .expect("audio done");
        match done {
            AudioStreamEvent::Done { metadata, .. } => {
                assert_eq!(metadata.get("provider"), Some(&serde_json::json!("openai")));
                assert_eq!(metadata["usage"]["total_tokens"], serde_json::json!(3));
            }
            other => panic!("expected done, got {other:?}"),
        }
    }

    #[test]
    fn openai_transcription_sse_helpers_parse_text_events() {
        let delta = openai_transcript_text_delta(&serde_json::json!({
            "type": "transcript.text.delta",
            "delta": "hel"
        }))
        .expect("text delta");
        assert!(matches!(
            delta,
            OpenAiTranscriptionStreamEvent::TextDelta { ref delta, .. } if delta == "hel"
        ));

        let segment = openai_transcript_text_segment(&serde_json::json!({
            "type": "transcript.text.segment",
            "id": "seg_1",
            "start": 0.5,
            "end": 1.25,
            "text": "hello",
            "speaker": "A"
        }))
        .expect("text segment");
        assert!(matches!(
            segment,
            OpenAiTranscriptionStreamEvent::Segment {
                ref id,
                start,
                end,
                ref text,
                ref speaker,
            } if id == "seg_1"
                && (start - 0.5).abs() < f32::EPSILON
                && (end - 1.25).abs() < f32::EPSILON
                && text == "hello"
                && speaker.as_deref() == Some("A")
        ));

        let done = openai_transcript_text_done(&serde_json::json!({
            "type": "transcript.text.done",
            "text": "hello",
            "usage": { "total_tokens": 3 }
        }))
        .expect("text done");
        assert!(matches!(
            done,
            OpenAiTranscriptionStreamEvent::Done {
                ref text,
                ref usage,
                ..
            } if text.as_deref() == Some("hello")
                && usage.as_ref().and_then(|u| u.get("total_tokens")) == Some(&serde_json::json!(3))
        ));
    }

    #[test]
    fn configurable_provider_id_reports_custom_id() {
        let tx = OpenAiAudioTransformerWithProviderId::new("openai-compatible");
        assert_eq!(tx.provider_id(), "openai-compatible");
    }

    #[test]
    fn configurable_defaults_apply_speed_and_stt_options() {
        let defaults = OpenAiAudioDefaults {
            tts_speed: Some(1.2),
            tts_include_language: false,
            stt_response_format: None,
            stt_include_language: false,
            stt_include_timestamp_granularities: false,
            ..Default::default()
        };
        let tx = OpenAiAudioTransformerWithProviderId::with_defaults("x", defaults);

        let tts = TtsRequest::new("hello".to_string());
        match tx.build_tts_body(&tts).unwrap() {
            AudioHttpBody::Json(j) => {
                let speed = j["speed"].as_f64().expect("speed should be a number");
                assert!((speed - 1.2).abs() < 1e-6);
            }
            _ => panic!("expected JSON body"),
        }
    }

    #[test]
    fn tts_provider_options_can_supply_speed() {
        let tx = OpenAiAudioTransformer;
        let req = TtsRequest::new("hello".to_string())
            .with_provider_option("openai", serde_json::json!({ "speed": 1.5 }));

        match tx.build_tts_body(&req).unwrap() {
            AudioHttpBody::Json(j) => {
                let speed = j["speed"].as_f64().expect("speed should be a number");
                assert!((speed - 1.5).abs() < 1e-6);
            }
            _ => panic!("expected JSON body"),
        }
    }

    #[test]
    fn tts_unified_instructions_override_provider_options() {
        let tx = OpenAiAudioTransformer;
        let req = TtsRequest::new("hello".to_string())
            .with_instructions("speak calmly")
            .with_provider_option("openai", serde_json::json!({ "instructions": "ignored" }));

        match tx.build_tts_body(&req).unwrap() {
            AudioHttpBody::Json(j) => {
                assert_eq!(j["instructions"], "speak calmly");
            }
            _ => panic!("expected JSON body"),
        }
    }

    #[test]
    fn tts_language_serializes_only_when_provider_defaults_enable_it() {
        let defaults = OpenAiAudioDefaults {
            tts_include_language: true,
            ..Default::default()
        };
        let tx = OpenAiAudioTransformerWithProviderId::with_defaults("compat", defaults);
        let req = TtsRequest::new("hello".to_string()).with_language("en");

        match tx.build_tts_body(&req).unwrap() {
            AudioHttpBody::Json(j) => {
                assert_eq!(j["language"], "en");
            }
            _ => panic!("expected JSON body"),
        }
    }

    #[test]
    fn stt_provider_options_accept_camel_case_language_and_timestamps() {
        let tx = OpenAiAudioTransformer;
        let req = SttRequest::from_audio(b"abc".to_vec(), "audio/mpeg").with_provider_option(
            "openai",
            serde_json::json!({
                "language": "en",
                "timestampGranularities": ["word", "segment"]
            }),
        );

        match tx.build_stt_body(&req).unwrap() {
            AudioHttpBody::Multipart(form) => {
                let body_text = multipart_body_text(form);
                assert!(body_text.contains("name=\"language\""));
                assert!(body_text.contains("en"));
                assert!(body_text.contains("name=\"timestamp_granularities[]\""));
                assert!(body_text.contains("word"));
                assert!(body_text.contains("segment"));
            }
            _ => panic!("expected multipart body"),
        }
    }
}
