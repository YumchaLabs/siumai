//! Audio transformers (TTS/STT) for providers
//!
//! This module defines generic audio transformation traits to unify how
//! providers build requests for text-to-speech and speech-to-text.

use crate::error::LlmError;

/// Output body for audio HTTP requests
pub enum AudioHttpBody {
    Json(serde_json::Value),
    Multipart(reqwest::multipart::Form),
}

pub trait AudioTransformer: Send + Sync {
    fn provider_id(&self) -> &str;

    /// Build TTS request body
    fn build_tts_body(&self, req: &crate::types::TtsRequest) -> Result<AudioHttpBody, LlmError>;

    /// Build STT request body
    fn build_stt_body(&self, req: &crate::types::SttRequest) -> Result<AudioHttpBody, LlmError>;

    /// Endpoint paths (relative)
    fn tts_endpoint(&self) -> &str;
    fn stt_endpoint(&self) -> &str;

    /// Parse TTS response bytes to audio data
    ///
    /// This method allows providers to handle different response formats:
    /// - For providers that return binary audio directly (like OpenAI),
    ///   the default implementation returns the bytes as-is.
    /// - For providers that return JSON with encoded audio (like MiniMaxi),
    ///   override this method to decode the audio from the response.
    ///
    /// # Arguments
    /// * `response_bytes` - Raw response bytes from the HTTP request
    ///
    /// # Returns
    /// Decoded audio bytes ready to be used
    fn parse_tts_response(&self, response_bytes: Vec<u8>) -> Result<Vec<u8>, LlmError> {
        // Default implementation: assume response is binary audio
        Ok(response_bytes)
    }

    /// Parse TTS response metadata from JSON
    ///
    /// For providers that return JSON responses with metadata (like MiniMaxi),
    /// this method extracts additional information like duration, sample rate, etc.
    ///
    /// # Arguments
    /// * `json` - JSON response from the provider
    ///
    /// # Returns
    /// Tuple of (duration_seconds, sample_rate_hz)
    fn parse_tts_metadata(
        &self,
        _json: &serde_json::Value,
    ) -> Result<(Option<f32>, Option<u32>), LlmError> {
        // Default: no metadata
        Ok((None, None))
    }

    /// Indicates whether TTS response is JSON (true) or binary (false)
    ///
    /// This helps the executor decide how to parse the HTTP response.
    fn tts_response_is_json(&self) -> bool {
        false // Default: binary audio
    }

    /// Parse STT JSON payload to text
    fn parse_stt_response(&self, json: &serde_json::Value) -> Result<String, LlmError>;
}
