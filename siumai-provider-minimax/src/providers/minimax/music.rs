//! MiniMax Music Generation Helper Functions
//!
//! Internal helper functions for music generation capability implementation.

use crate::error::LlmError;
use crate::execution::executors::common::{HttpBody, execute_json_request};
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::wiring::HttpExecutionWiring;
use crate::retry_api::RetryOptions;
use crate::types::HttpConfig;
use crate::types::music::{MusicGenerationRequest, MusicGenerationResponse, MusicMetadata};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::spec::MinimaxMusicSpec;

/// MiniMax-specific music generation request
///
/// This is the actual request format sent to MiniMax API.
/// It's converted from the generic `MusicGenerationRequest`.
#[derive(Debug, Serialize)]
pub(super) struct MinimaxMusicRequest {
    pub model: String,
    pub prompt: String,
    pub lyrics: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_setting: Option<crate::types::music::MusicAudioSetting>,
}

/// MiniMax music generation API response
#[derive(Debug, Deserialize, Serialize)]
pub(super) struct MinimaxMusicResponse {
    pub data: MinimaxMusicData,
    pub extra_info: MinimaxMusicExtraInfo,
}

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct MinimaxMusicData {
    pub audio: String, // hex-encoded audio data
    pub status: u32,
}

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct MinimaxMusicExtraInfo {
    pub music_duration: Option<u32>,
    pub music_sample_rate: Option<u32>,
    pub music_channel: Option<u32>,
    pub bitrate: Option<u32>,
    pub music_size: Option<u32>,
}

/// Generate music
#[allow(clippy::too_many_arguments)]
pub(super) async fn generate_music(
    api_key: &str,
    base_url: &str,
    http_config: &HttpConfig,
    http_client: &reqwest::Client,
    retry_options: Option<&RetryOptions>,
    interceptors: &[Arc<dyn HttpInterceptor>],
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    request: MusicGenerationRequest,
) -> Result<MusicGenerationResponse, LlmError> {
    let mut wiring = HttpExecutionWiring::new(
        "minimax",
        http_client.clone(),
        super::utils::build_context(api_key, base_url, http_config),
    )
    .with_interceptors(interceptors.to_vec())
    .with_retry_options(retry_options.cloned());

    if let Some(transport) = http_transport {
        wiring = wiring.with_transport(transport);
    }

    let config = wiring.config(Arc::new(MinimaxMusicSpec::new()));
    let url = MinimaxMusicSpec::new().music_generation_url(&config.provider_context);

    // MiniMax requires lyrics field, so provide default if not specified
    let lyrics = request.lyrics.unwrap_or_else(|| {
        // Generate default instrumental lyrics structure
        "[Intro]\n[Main]\n[Outro]".to_string()
    });

    // Convert generic request to MiniMax-specific format
    let minimax_request = MinimaxMusicRequest {
        model: request.model,
        prompt: request.prompt,
        lyrics,
        audio_setting: request.audio_setting,
    };

    let body = serde_json::to_value(minimax_request)
        .map_err(|e| LlmError::ParseError(format!("Failed to serialize music request: {}", e)))?;
    let res = execute_json_request(&config, &url, HttpBody::Json(body), None, false).await?;
    let music_response: MinimaxMusicResponse = serde_json::from_value(res.json).map_err(|e| {
        LlmError::provider_error("minimax", format!("Failed to parse music response: {}", e))
    })?;

    // Decode hex-encoded audio
    let audio_data = hex::decode(&music_response.data.audio).map_err(|e| {
        LlmError::provider_error("minimax", format!("Failed to decode audio hex: {}", e))
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
    super::models::ALL_MUSIC
        .iter()
        .map(|model| (*model).to_string())
        .collect()
}

/// Get supported audio formats
pub(super) fn get_supported_audio_formats() -> Vec<String> {
    vec!["mp3".to_string(), "wav".to_string()]
}
