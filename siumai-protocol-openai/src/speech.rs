//! OpenAI buffered text-to-speech request and response codecs.

use bytes::Bytes;
use serde::Serialize;
use serde_json::Value;
use siumai_core::{Error, ErrorKind, ModelId, ResponseMetadata, SpeechResponse, Usage};

/// OpenAI Speech API mode identifier.
pub const API_MODE_ID: &str = "audio-speech";
/// OpenAI Audio protocol identifier used by buffered speech synthesis.
pub const PROTOCOL_ID: &str = "openai.audio";
/// Relative OpenAI speech-synthesis endpoint.
pub const TARGET: &str = "audio/speech";

/// Buffered audio formats accepted by the OpenAI Speech endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeechFormat {
    Mp3,
    Opus,
    Aac,
    Flac,
    Wav,
    Pcm,
}

impl SpeechFormat {
    pub const fn as_wire(self) -> &'static str {
        match self {
            Self::Mp3 => "mp3",
            Self::Opus => "opus",
            Self::Aac => "aac",
            Self::Flac => "flac",
            Self::Wav => "wav",
            Self::Pcm => "pcm",
        }
    }

    pub const fn media_type(self) -> &'static str {
        match self {
            Self::Mp3 => "audio/mpeg",
            Self::Opus => "audio/ogg",
            Self::Aac => "audio/aac",
            Self::Flac => "audio/flac",
            Self::Wav => "audio/wav",
            Self::Pcm => "audio/pcm",
        }
    }
}

/// Fully resolved provider-owned speech request fields.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeechConfig {
    pub voice: String,
    pub format: SpeechFormat,
    pub speed: Option<f32>,
    pub instructions: Option<String>,
}

/// Encode one non-streaming OpenAI speech request.
pub fn encode_speech_request(
    model: &ModelId,
    input: &str,
    config: &SpeechConfig,
) -> Result<Value, Error> {
    serde_json::to_value(SpeechRequestWire {
        model: model.as_str(),
        input,
        voice: &config.voice,
        response_format: config.format.as_wire(),
        speed: config.speed,
        instructions: config.instructions.as_deref(),
    })
    .map_err(|source| {
        Error::new(
            ErrorKind::Internal,
            "OpenAI speech request could not be serialized",
        )
        .with_source(source)
    })
}

/// Decode one buffered binary speech response.
pub fn decode_speech_response(
    body: Bytes,
    requested_model: &ModelId,
    media_type: impl Into<String>,
) -> Result<SpeechResponse, Error> {
    let response = SpeechResponse {
        media_type: media_type.into(),
        audio: body,
        duration_seconds: None,
        sample_rate_hz: None,
        metadata: ResponseMetadata {
            response_id: None,
            request_id: None,
            model: Some(requested_model.clone()),
        },
        usage: Usage::default(),
        warnings: Vec::new(),
        provider: Default::default(),
    };
    response.validate()?;
    Ok(response)
}

#[derive(Debug, Serialize)]
struct SpeechRequestWire<'a> {
    model: &'a str,
    input: &'a str,
    voice: &'a str,
    response_format: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    speed: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<&'a str>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_preserves_resolved_voice_format_speed_and_instructions() {
        let config = SpeechConfig {
            voice: "alloy".to_string(),
            format: SpeechFormat::Wav,
            speed: Some(1.25),
            instructions: Some("Speak warmly".to_string()),
        };
        let body =
            encode_speech_request(&ModelId::new("gpt-4o-mini-tts").unwrap(), "Hello", &config)
                .unwrap();
        assert_eq!(body["voice"], "alloy");
        assert_eq!(body["response_format"], "wav");
        assert_eq!(body["speed"], 1.25);
        assert_eq!(body["instructions"], "Speak warmly");
    }

    #[test]
    fn decoder_rejects_empty_audio() {
        let error = decode_speech_response(
            Bytes::new(),
            &ModelId::new("future-speech-model").unwrap(),
            "audio/mpeg",
        )
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ProtocolViolation);
    }
}
