//! OpenAI Audio Transformer

use crate::error::LlmError;
use crate::execution::transformers::audio::{AudioHttpBody, AudioTransformer};
use crate::types::{SttRequest, TtsRequest};

/// OpenAI Audio Transformer for TTS and STT
#[derive(Clone)]
pub struct OpenAiAudioTransformer;

impl AudioTransformer for OpenAiAudioTransformer {
    fn provider_id(&self) -> &str {
        "openai"
    }

    fn build_tts_body(&self, req: &TtsRequest) -> Result<AudioHttpBody, LlmError> {
        let model = req.model.clone().unwrap_or_else(|| "tts-1".to_string());
        let voice = req.voice.clone().unwrap_or_else(|| "alloy".to_string());
        let format = req.format.clone().unwrap_or_else(|| "mp3".to_string());
        let mut json = serde_json::json!({
            "model": model,
            "input": req.text,
            "voice": voice,
            "response_format": format,
        });
        if let Some(s) = req.speed {
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

    fn build_stt_body(&self, req: &SttRequest) -> Result<AudioHttpBody, LlmError> {
        let model = req.model.clone().unwrap_or_else(|| "whisper-1".to_string());
        let audio = req
            .audio_data
            .clone()
            .ok_or_else(|| LlmError::InvalidInput("audio_data required for STT".to_string()))?;
        let mut form = reqwest::multipart::Form::new()
            .part(
                "file",
                reqwest::multipart::Part::bytes(audio).file_name("audio.mp3"),
            )
            .text("model", model)
            .text("response_format", "json");
        if let Some(lang) = &req.language {
            form = form.text("language", lang.clone());
        }
        if let Some(grans) = &req.timestamp_granularities {
            for g in grans {
                form = form.text("timestamp_granularities[]", g.clone());
            }
        }
        Ok(AudioHttpBody::Multipart(form))
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
