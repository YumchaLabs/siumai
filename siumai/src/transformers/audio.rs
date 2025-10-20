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

    /// Parse STT JSON payload to text
    fn parse_stt_response(&self, json: &serde_json::Value) -> Result<String, LlmError>;
}
