use serde_json::{Map, Value};
use siumai_core::{
    ContentPart, Error, ErrorKind, MediaData, ModelId, ProviderScope, SpeechResponse,
};

use super::decode_language_response;

/// Current Gemini Interactions TTS target.
pub const V1BETA_SPEECH_TARGET: &str = "v1beta/interactions";

const DEFAULT_SAMPLE_RATE_HZ: u32 = 24_000;

/// Checked single-speaker configuration for the portable buffered speech adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InteractionSpeechConfig {
    voice: String,
}

impl InteractionSpeechConfig {
    pub fn new(voice: impl Into<String>) -> Result<Self, Error> {
        let voice = voice.into();
        validate_text(&voice, 256, "Gemini speech voice is invalid")?;
        Ok(Self { voice })
    }

    pub fn voice(&self) -> &str {
        &self.voice
    }
}

/// Encode one current v1beta Interactions buffered TTS request.
pub fn encode_speech_request(
    model: &ModelId,
    text: &str,
    config: &InteractionSpeechConfig,
) -> Result<Value, Error> {
    validate_text(text, 1024 * 1024, "Gemini speech text is invalid")?;

    let mut speech = Map::new();
    speech.insert(
        "voice".to_string(),
        Value::String(config.voice().to_string()),
    );

    Ok(serde_json::json!({
        "model": model.as_str(),
        "input": text,
        "response_format": {
            "type": "audio"
        },
        "generation_config": {
            "speech_config": [Value::Object(speech)]
        }
    }))
}

/// Decode one terminal Interactions TTS resource into the portable buffered response.
pub fn decode_speech_response(
    body: &[u8],
    scope: &ProviderScope,
    requested_model: &ModelId,
) -> Result<SpeechResponse, Error> {
    let decoded = decode_language_response(body, scope, requested_model)?;
    let canonical = decoded.canonical();
    let mut audio = None;

    for part in canonical.content() {
        let ContentPart::Media(media) = part else {
            continue;
        };
        if !media.media_type.to_ascii_lowercase().starts_with("audio/") {
            continue;
        }
        let MediaData::Bytes(bytes) = &media.data else {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "Gemini returned URI audio that cannot satisfy buffered SpeechModel",
            ));
        };
        if audio.is_some() {
            return Err(Error::protocol_violation(
                "Gemini returned more than one buffered speech artifact",
            ));
        }
        audio = Some((media.media_type.clone(), bytes.clone()));
    }

    let Some((media_type, audio)) = audio else {
        return Err(Error::protocol_violation(
            "Gemini Interactions TTS response omitted buffered audio",
        ));
    };
    let mut response = SpeechResponse {
        media_type,
        audio,
        duration_seconds: None,
        sample_rate_hz: Some(DEFAULT_SAMPLE_RATE_HZ),
        metadata: siumai_core::ResponseMetadata {
            response_id: canonical.id().map(ToOwned::to_owned),
            request_id: None,
            model: canonical.model().cloned(),
        },
        usage: canonical.usage().clone(),
        warnings: canonical.warnings().to_vec(),
        provider: canonical.provider_metadata().clone(),
    };
    response.validate()?;
    if response.media_type.eq_ignore_ascii_case("audio/l16") {
        response.media_type = "audio/pcm".to_string();
    }
    Ok(response)
}

fn validate_text(value: &str, maximum: usize, message: &'static str) -> Result<(), Error> {
    if value.trim().is_empty()
        || value.len() > maximum
        || value.chars().any(|character| character == '\0')
    {
        return Err(Error::new(ErrorKind::InvalidInput, message));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use siumai_core::{
        ApiModeId, PlatformId, ProtocolId, ProviderId, ReplayDomain, ReplayDomainId,
    };

    use super::*;

    fn scope() -> ProviderScope {
        ProviderScope::new(ProviderId::new("google").unwrap())
            .with_platform(PlatformId::new("gemini-api").unwrap())
            .with_protocol(ProtocolId::new("gemini-interactions").unwrap())
            .with_api_mode(ApiModeId::new("interactions-speech").unwrap())
            .with_replay_domain(ReplayDomain::official(
                ReplayDomainId::new("google-gemini-api").unwrap(),
            ))
    }

    #[test]
    fn request_uses_current_interactions_audio_shape() {
        let config = InteractionSpeechConfig::new("Kore").unwrap();
        let value = encode_speech_request(
            &ModelId::new("gemini-3.1-flash-tts-preview").unwrap(),
            "Say hello",
            &config,
        )
        .unwrap();

        assert_eq!(value["response_format"], json!({"type": "audio"}));
        assert_eq!(
            value["generation_config"]["speech_config"],
            json!([{"voice": "Kore"}])
        );
        assert!(value.get("speechConfig").is_none());
    }

    #[test]
    fn response_requires_exactly_one_inline_audio_artifact() {
        let bytes = [0_u8, 1, 2, 3];
        let body = serde_json::to_vec(&json!({
            "id": "interaction-tts",
            "status": "completed",
            "model": "gemini-3.1-flash-tts-preview",
            "steps": [{
                "type": "model_output",
                "content": [{
                    "type": "audio",
                    "mime_type": "audio/L16",
                    "data": base64::Engine::encode(
                        &base64::engine::general_purpose::STANDARD,
                        bytes,
                    )
                }]
            }],
            "usage": {
                "total_input_tokens": 2,
                "total_output_tokens": 1,
                "total_tokens": 3
            }
        }))
        .unwrap();

        let response = decode_speech_response(
            &body,
            &scope(),
            &ModelId::new("gemini-3.1-flash-tts-preview").unwrap(),
        )
        .unwrap();
        assert_eq!(response.media_type, "audio/pcm");
        assert_eq!(response.audio.as_ref(), bytes);
        assert_eq!(response.sample_rate_hz, Some(24_000));
    }
}
