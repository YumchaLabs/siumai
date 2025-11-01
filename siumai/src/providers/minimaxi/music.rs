//! MiniMaxi Music Generation Helper Functions
//!
//! Internal helper functions for music generation capability implementation.

use crate::error::LlmError;
use crate::types::music::{MusicGenerationRequest, MusicGenerationResponse, MusicMetadata};
use serde::{Deserialize, Serialize};

/// Get music generation endpoint URL
pub(super) fn music_generation_url(base_url: &str) -> String {
    format!("{}/v1/music_generation", base_url.trim_end_matches('/'))
}

/// MiniMaxi-specific music generation request
///
/// This is the actual request format sent to MiniMaxi API.
/// It's converted from the generic `MusicGenerationRequest`.
#[derive(Debug, Serialize)]
pub(super) struct MinimaxiMusicRequest {
    pub model: String,
    pub prompt: String,
    pub lyrics: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_setting: Option<crate::types::music::MusicAudioSetting>,
}

/// MiniMaxi music generation API response
#[derive(Debug, Deserialize, Serialize)]
pub(super) struct MinimaxiMusicResponse {
    pub data: MinimaxiMusicData,
    pub extra_info: MinimaxiMusicExtraInfo,
}

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct MinimaxiMusicData {
    pub audio: String, // hex-encoded audio data
    pub status: u32,
}

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct MinimaxiMusicExtraInfo {
    pub music_duration: Option<u32>,
    pub music_sample_rate: Option<u32>,
    pub music_channel: Option<u32>,
    pub bitrate: Option<u32>,
    pub music_size: Option<u32>,
}

/// Generate music
pub(super) async fn generate_music(
    api_key: &str,
    base_url: &str,
    http_client: &reqwest::Client,
    request: MusicGenerationRequest,
) -> Result<MusicGenerationResponse, LlmError> {
    // MiniMaxi requires lyrics field, so provide default if not specified
    let lyrics = request.lyrics.unwrap_or_else(|| {
        // Generate default instrumental lyrics structure
        "[Intro]\n[Main]\n[Outro]".to_string()
    });

    // Convert generic request to MiniMaxi-specific format
    let minimaxi_request = MinimaxiMusicRequest {
        model: request.model,
        prompt: request.prompt,
        lyrics,
        audio_setting: request.audio_setting,
    };

    let url = music_generation_url(base_url);

    // Build HTTP request
    let response = http_client
        .post(&url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&minimaxi_request)
        .send()
        .await
        .map_err(|e| {
            LlmError::provider_error(
                "minimaxi",
                format!("Music generation request failed: {}", e),
            )
        })?;

    // Check HTTP status
    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(LlmError::provider_error(
            "minimaxi",
            format!(
                "Music generation failed with status {}: {}",
                status, error_text
            ),
        ));
    }

    // Parse response
    let response_bytes = response.bytes().await.map_err(|e| {
        LlmError::provider_error("minimaxi", format!("Failed to read response: {}", e))
    })?;

    let music_response: MinimaxiMusicResponse =
        serde_json::from_slice(&response_bytes).map_err(|e| {
            LlmError::provider_error("minimaxi", format!("Failed to parse music response: {}", e))
        })?;

    // Decode hex-encoded audio
    let audio_data = hex::decode(&music_response.data.audio).map_err(|e| {
        LlmError::provider_error("minimaxi", format!("Failed to decode audio hex: {}", e))
    })?;

    // Extract metadata
    let metadata = MusicMetadata {
        music_duration: music_response.extra_info.music_duration,
        music_sample_rate: music_response.extra_info.music_sample_rate,
        music_channel: music_response.extra_info.music_channel,
        bitrate: music_response.extra_info.bitrate,
        music_size: music_response.extra_info.music_size,
    };

    Ok(MusicGenerationResponse {
        audio_data,
        metadata,
    })
}

/// Get supported music models
pub(super) fn get_supported_music_models() -> Vec<String> {
    vec!["music-2.0".to_string()]
}

/// Get supported audio formats
pub(super) fn get_supported_audio_formats() -> Vec<String> {
    vec!["mp3".to_string(), "wav".to_string()]
}
