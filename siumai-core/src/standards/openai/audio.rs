//! OpenAI Audio API Standard
//!
//! This module provides a reusable `AudioTransformer` implementation for providers
//! that follow OpenAI's TTS/STT endpoints:
//! - `POST /audio/speech` (TTS)
//! - `POST /audio/transcriptions` (STT)

use crate::error::LlmError;
use crate::execution::transformers::audio::{AudioHttpBody, AudioTransformer};
use crate::types::{SttRequest, TtsRequest};
use std::borrow::Cow;

#[derive(Debug, Clone)]
pub struct OpenAiAudioDefaults {
    pub tts_model: Cow<'static, str>,
    pub tts_voice: Cow<'static, str>,
    pub tts_format: Cow<'static, str>,
    pub tts_speed: Option<f32>,

    pub stt_model: Cow<'static, str>,
    pub stt_file_name: Cow<'static, str>,
    pub stt_response_format: Option<Cow<'static, str>>,
    pub stt_include_language: bool,
    pub stt_include_timestamp_granularities: bool,
}

impl Default for OpenAiAudioDefaults {
    fn default() -> Self {
        Self {
            tts_model: Cow::Borrowed("tts-1"),
            tts_voice: Cow::Borrowed("alloy"),
            tts_format: Cow::Borrowed("mp3"),
            tts_speed: None,
            stt_model: Cow::Borrowed("whisper-1"),
            stt_file_name: Cow::Borrowed("audio.mp3"),
            stt_response_format: Some(Cow::Borrowed("json")),
            stt_include_language: true,
            stt_include_timestamp_granularities: true,
        }
    }
}

fn build_tts_body_impl(
    req: &TtsRequest,
    defaults: &OpenAiAudioDefaults,
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
    let speed = req.speed.or(defaults.tts_speed);
    let mut json = serde_json::json!({
        "model": model,
        "input": req.text,
        "voice": voice,
        "response_format": format,
    });
    if let Some(s) = speed {
        json["speed"] = serde_json::json!(s);
    }
    if let Some(instr) = req
        .extra_params
        .get("instructions")
        .and_then(|v| v.as_str())
    {
        json["instructions"] = serde_json::json!(instr);
    }
    Ok(AudioHttpBody::Json(json))
}

fn build_stt_body_impl(
    req: &SttRequest,
    defaults: &OpenAiAudioDefaults,
) -> Result<AudioHttpBody, LlmError> {
    let model = req
        .model
        .clone()
        .unwrap_or_else(|| defaults.stt_model.to_string());
    let audio = req
        .audio_data
        .clone()
        .ok_or_else(|| LlmError::InvalidInput("audio_data required for STT".to_string()))?;

    if req
        .extra_params
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        return Err(LlmError::UnsupportedOperation(
            "OpenAI STT streaming is not supported in the unified Transcription surface; use provider-specific APIs instead.".to_string(),
        ));
    }

    let mut form = reqwest::multipart::Form::new().part(
        "file",
        reqwest::multipart::Part::bytes(audio).file_name(defaults.stt_file_name.to_string()),
    );
    form = form.text("model", model);
    if let Some(fmt) = req
        .extra_params
        .get("response_format")
        .and_then(|v| v.as_str())
    {
        form = form.text("response_format", fmt.to_string());
    } else if let Some(fmt) = defaults.stt_response_format.as_ref() {
        form = form.text("response_format", fmt.to_string());
    }
    if defaults.stt_include_language
        && let Some(lang) = &req.language
    {
        form = form.text("language", lang.clone());
    }
    if defaults.stt_include_timestamp_granularities
        && let Some(grans) = &req.timestamp_granularities
    {
        for g in grans {
            form = form.text("timestamp_granularities[]", g.clone());
        }
    }

    // OpenAI-specific optional fields (escape hatches).
    // Keep these in `extra_params` to avoid expanding the unified surface.
    if let Some(prompt) = req.extra_params.get("prompt").and_then(|v| v.as_str()) {
        form = form.text("prompt", prompt.to_string());
    }
    if let Some(temp) = req.extra_params.get("temperature").and_then(|v| v.as_f64()) {
        form = form.text("temperature", temp.to_string());
    }
    if let Some(strategy) = req
        .extra_params
        .get("chunking_strategy")
        .and_then(|v| v.as_str())
    {
        form = form.text("chunking_strategy", strategy.to_string());
    }
    if let Some(include) = req.extra_params.get("include").and_then(|v| v.as_array()) {
        for item in include {
            if let Some(s) = item.as_str() {
                form = form.text("include[]", s.to_string());
            }
        }
    }
    if let Some(names) = req
        .extra_params
        .get("known_speaker_names")
        .and_then(|v| v.as_array())
    {
        for item in names {
            if let Some(s) = item.as_str() {
                form = form.text("known_speaker_names[]", s.to_string());
            }
        }
    }
    if let Some(refs) = req
        .extra_params
        .get("known_speaker_references")
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
        build_tts_body_impl(req, &OpenAiAudioDefaults::default())
    }

    fn build_stt_body(&self, req: &SttRequest) -> Result<AudioHttpBody, LlmError> {
        build_stt_body_impl(req, &OpenAiAudioDefaults::default())
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
        build_tts_body_impl(req, &self.defaults)
    }

    fn build_stt_body(&self, req: &SttRequest) -> Result<AudioHttpBody, LlmError> {
        build_stt_body_impl(req, &self.defaults)
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
    fn configurable_provider_id_reports_custom_id() {
        let tx = OpenAiAudioTransformerWithProviderId::new("openai-compatible");
        assert_eq!(tx.provider_id(), "openai-compatible");
    }

    #[test]
    fn configurable_defaults_apply_speed_and_stt_options() {
        let defaults = OpenAiAudioDefaults {
            tts_speed: Some(1.2),
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
}
