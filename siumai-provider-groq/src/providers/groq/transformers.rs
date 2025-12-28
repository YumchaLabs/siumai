//! Audio transformers for Groq (TTS/STT)
use crate::error::LlmError;
use crate::execution::transformers::audio::{AudioHttpBody, AudioTransformer};

#[derive(Clone)]
pub struct GroqAudioTransformer;

impl AudioTransformer for GroqAudioTransformer {
    fn provider_id(&self) -> &str {
        "groq"
    }

    fn build_tts_body(&self, req: &crate::types::TtsRequest) -> Result<AudioHttpBody, LlmError> {
        let defaults = crate::standards::openai::audio::OpenAiAudioDefaults {
            tts_model: std::borrow::Cow::Borrowed("playai-tts"),
            tts_voice: std::borrow::Cow::Borrowed("Fritz-PlayAI"),
            tts_format: std::borrow::Cow::Borrowed("wav"),
            tts_speed: Some(1.0),
            stt_model: std::borrow::Cow::Borrowed("whisper-large-v3"),
            stt_file_name: std::borrow::Cow::Borrowed("audio.wav"),
            stt_response_format: None,
            stt_include_language: false,
            stt_include_timestamp_granularities: false,
        };
        let tx =
            crate::standards::openai::audio::OpenAiAudioTransformerWithProviderId::with_defaults(
                std::borrow::Cow::Borrowed("groq"),
                defaults,
            );
        tx.build_tts_body(req)
    }

    fn build_stt_body(&self, req: &crate::types::SttRequest) -> Result<AudioHttpBody, LlmError> {
        let defaults = crate::standards::openai::audio::OpenAiAudioDefaults {
            tts_model: std::borrow::Cow::Borrowed("playai-tts"),
            tts_voice: std::borrow::Cow::Borrowed("Fritz-PlayAI"),
            tts_format: std::borrow::Cow::Borrowed("wav"),
            tts_speed: Some(1.0),
            stt_model: std::borrow::Cow::Borrowed("whisper-large-v3"),
            stt_file_name: std::borrow::Cow::Borrowed("audio.wav"),
            stt_response_format: None,
            stt_include_language: false,
            stt_include_timestamp_granularities: false,
        };
        let tx =
            crate::standards::openai::audio::OpenAiAudioTransformerWithProviderId::with_defaults(
                std::borrow::Cow::Borrowed("groq"),
                defaults,
            );
        tx.build_stt_body(req)
    }

    fn tts_endpoint(&self) -> &str {
        "/audio/speech"
    }
    fn stt_endpoint(&self) -> &str {
        "/audio/transcriptions"
    }

    fn parse_stt_response(&self, json: &serde_json::Value) -> Result<String, LlmError> {
        let text = json
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| LlmError::ParseError("missing 'text' field".to_string()))?;
        Ok(text.to_string())
    }
}
