//! MiniMaxi Audio Transformer for TTS
//!
//! MiniMaxi uses a custom API format for text-to-speech that differs from OpenAI.
//! API Endpoint: POST /v1/t2a_v2
//!
//! Request format:
//! ```json
//! {
//!   "model": "speech-2.6-hd",
//!   "text": "今天是不是很开心呀，当然了！",
//!   "stream": false,
//!   "voice_setting": {
//!     "voice_id": "male-qn-qingse",
//!     "speed": 1,
//!     "vol": 1,
//!     "pitch": 0,
//!     "emotion": "happy"
//!   },
//!   "audio_setting": {
//!     "sample_rate": 32000,
//!     "bitrate": 128000,
//!     "format": "mp3",
//!     "channel": 1
//!   }
//! }
//! ```
//!
//! Response format:
//! ```json
//! {
//!   "data": {
//!     "audio": "<hex-encoded audio data>",
//!     "status": 2
//!   },
//!   "extra_info": {
//!     "audio_length": 9900,
//!     "audio_sample_rate": 32000,
//!     "audio_size": 160323,
//!     "bitrate": 128000,
//!     "word_count": 52,
//!     "usage_characters": 101,
//!     "audio_format": "mp3",
//!     "audio_channel": 1
//!   }
//! }
//! ```

use crate::error::LlmError;
use crate::execution::transformers::audio::{AudioHttpBody, AudioTransformer};
use crate::types::{SttRequest, TtsRequest};
use serde_json::json;

/// MiniMaxi Audio Transformer for TTS and STT
#[derive(Clone)]
pub struct MinimaxiAudioTransformer;

impl AudioTransformer for MinimaxiAudioTransformer {
    fn provider_id(&self) -> &str {
        "minimaxi"
    }

    fn tts_response_is_json(&self) -> bool {
        true // MiniMaxi returns JSON with hex-encoded audio
    }

    fn parse_tts_response(&self, response_bytes: Vec<u8>) -> Result<Vec<u8>, LlmError> {
        // Parse JSON response
        let json: serde_json::Value = serde_json::from_slice(&response_bytes).map_err(|e| {
            LlmError::ParseError(format!(
                "Failed to parse MiniMaxi TTS response as JSON: {}",
                e
            ))
        })?;

        // Extract and decode hex-encoded audio
        parse_minimaxi_tts_response(&json)
    }

    fn parse_tts_metadata(
        &self,
        json: &serde_json::Value,
    ) -> Result<(Option<f32>, Option<u32>), LlmError> {
        let sample_rate = json
            .get("extra_info")
            .and_then(|info| info.get("audio_sample_rate"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);

        let duration = json
            .get("extra_info")
            .and_then(|info| info.get("audio_length"))
            .and_then(|v| v.as_u64())
            .map(|v| v as f32 / 1000.0); // Convert ms to seconds

        Ok((duration, sample_rate))
    }

    fn build_tts_body(&self, req: &TtsRequest) -> Result<AudioHttpBody, LlmError> {
        // Model: default to speech-2.6-hd
        let model = req
            .model
            .clone()
            .unwrap_or_else(|| "speech-2.6-hd".to_string());

        // Voice: default to male-qn-qingse
        let voice_id = req
            .voice
            .clone()
            .unwrap_or_else(|| "male-qn-qingse".to_string());

        // Format: default to mp3
        let format = req.format.clone().unwrap_or_else(|| "mp3".to_string());

        // Speed: default to 1.0
        let speed = req.speed.unwrap_or(1.0);

        // Extract additional parameters from extra_params
        let vol = req
            .extra_params
            .get("vol")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        let pitch = req
            .extra_params
            .get("pitch")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);

        let emotion = req
            .extra_params
            .get("emotion")
            .and_then(|v| v.as_str())
            .unwrap_or("neutral");

        let sample_rate = req
            .extra_params
            .get("sample_rate")
            .and_then(|v| v.as_u64())
            .unwrap_or(32000);

        let bitrate = req
            .extra_params
            .get("bitrate")
            .and_then(|v| v.as_u64())
            .unwrap_or(128000);

        let channel = req
            .extra_params
            .get("channel")
            .and_then(|v| v.as_u64())
            .unwrap_or(1);

        // Build request body
        let mut body = json!({
            "model": model,
            "text": req.text,
            "stream": false,
            "voice_setting": {
                "voice_id": voice_id,
                "speed": speed,
                "vol": vol,
                "pitch": pitch,
                "emotion": emotion
            },
            "audio_setting": {
                "sample_rate": sample_rate,
                "bitrate": bitrate,
                "format": format,
                "channel": channel
            }
        });

        // Add optional pronunciation_dict if provided
        if let Some(pronunciation_dict) = req.extra_params.get("pronunciation_dict") {
            body["pronunciation_dict"] = pronunciation_dict.clone();
        }

        // Add optional voice_modify if provided
        if let Some(voice_modify) = req.extra_params.get("voice_modify") {
            body["voice_modify"] = voice_modify.clone();
        }

        // Add optional subtitle_enable if provided
        if let Some(subtitle_enable) = req.extra_params.get("subtitle_enable") {
            body["subtitle_enable"] = subtitle_enable.clone();
        }

        Ok(AudioHttpBody::Json(body))
    }

    fn build_stt_body(&self, _req: &SttRequest) -> Result<AudioHttpBody, LlmError> {
        // MiniMaxi STT is not implemented yet
        Err(LlmError::UnsupportedOperation(
            "Speech-to-text is not yet supported for MiniMaxi".to_string(),
        ))
    }

    fn tts_endpoint(&self) -> &str {
        "/v1/t2a_v2"
    }

    fn stt_endpoint(&self) -> &str {
        // Not implemented yet
        "/v1/stt"
    }

    fn parse_stt_response(&self, _json: &serde_json::Value) -> Result<String, LlmError> {
        // MiniMaxi STT is not implemented yet
        Err(LlmError::UnsupportedOperation(
            "Speech-to-text is not yet supported for MiniMaxi".to_string(),
        ))
    }
}

/// Parse MiniMaxi TTS response and extract audio data
///
/// MiniMaxi returns audio as hex-encoded string in the response
pub(crate) fn parse_minimaxi_tts_response(json: &serde_json::Value) -> Result<Vec<u8>, LlmError> {
    // Extract audio hex string from response
    let audio_hex = json
        .get("data")
        .and_then(|d| d.get("audio"))
        .and_then(|a| a.as_str())
        .ok_or_else(|| {
            LlmError::ParseError("Missing 'data.audio' field in MiniMaxi TTS response".to_string())
        })?;

    // Decode hex string to bytes
    hex::decode(audio_hex)
        .map_err(|e| LlmError::ParseError(format!("Failed to decode hex audio data: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_build_tts_body_minimal() {
        let transformer = MinimaxiAudioTransformer;
        let req = TtsRequest::new("Hello world".to_string());

        let body = transformer.build_tts_body(&req).unwrap();
        match body {
            AudioHttpBody::Json(json) => {
                assert_eq!(json["model"], "speech-2.6-hd");
                assert_eq!(json["text"], "Hello world");
                assert_eq!(json["stream"], false);
                assert_eq!(json["voice_setting"]["voice_id"], "male-qn-qingse");
                assert_eq!(json["voice_setting"]["speed"], 1.0);
                assert_eq!(json["audio_setting"]["format"], "mp3");
            }
            _ => panic!("Expected JSON body"),
        }
    }

    #[test]
    fn test_build_tts_body_with_options() {
        let transformer = MinimaxiAudioTransformer;
        let mut req = TtsRequest::new("你好世界".to_string())
            .with_model("speech-2.6-turbo".to_string())
            .with_voice("female-shaonv".to_string())
            .with_format("wav".to_string())
            .with_speed(1.2);

        req.extra_params
            .insert("emotion".to_string(), json!("happy"));
        req.extra_params.insert("pitch".to_string(), json!(5));

        let body = transformer.build_tts_body(&req).unwrap();
        match body {
            AudioHttpBody::Json(json) => {
                assert_eq!(json["model"], "speech-2.6-turbo");
                assert_eq!(json["text"], "你好世界");
                assert_eq!(json["voice_setting"]["voice_id"], "female-shaonv");
                // Use approximate comparison for floating point
                let speed = json["voice_setting"]["speed"].as_f64().unwrap();
                assert!(
                    (speed - 1.2).abs() < 0.001,
                    "Speed should be approximately 1.2, got {}",
                    speed
                );
                assert_eq!(json["voice_setting"]["emotion"], "happy");
                assert_eq!(json["voice_setting"]["pitch"], 5);
                assert_eq!(json["audio_setting"]["format"], "wav");
            }
            _ => panic!("Expected JSON body"),
        }
    }

    #[test]
    fn test_parse_tts_response() {
        let response = json!({
            "data": {
                "audio": "48656c6c6f", // "Hello" in hex
                "status": 2
            },
            "extra_info": {
                "audio_length": 1000,
                "audio_format": "mp3"
            }
        });

        let audio_data = parse_minimaxi_tts_response(&response).unwrap();
        assert_eq!(audio_data, b"Hello");
    }

    #[test]
    fn test_parse_tts_response_missing_field() {
        let response = json!({
            "extra_info": {
                "audio_length": 1000
            }
        });

        let result = parse_minimaxi_tts_response(&response);
        assert!(result.is_err());
    }
}
