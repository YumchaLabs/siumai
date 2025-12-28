//! MiniMaxi Music Generation Helper Functions
//!
//! Internal helper functions for music generation capability implementation.

use crate::error::LlmError;
use crate::execution::executors::common::{HttpBody, HttpExecutionConfig, execute_json_request};
use crate::execution::http::interceptor::HttpInterceptor;
use crate::retry_api::RetryOptions;
use crate::types::HttpConfig;
use crate::types::music::{MusicGenerationRequest, MusicGenerationResponse, MusicMetadata};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::spec::MinimaxiMusicSpec;

fn build_http_execution_config(
    api_key: &str,
    base_url: &str,
    http_config: &HttpConfig,
    http_client: &reqwest::Client,
    retry_options: Option<&RetryOptions>,
    interceptors: &[Arc<dyn HttpInterceptor>],
) -> HttpExecutionConfig {
    HttpExecutionConfig {
        provider_id: "minimaxi".to_string(),
        http_client: http_client.clone(),
        provider_spec: Arc::new(MinimaxiMusicSpec::new()),
        provider_context: super::utils::build_context(api_key, base_url, http_config),
        interceptors: interceptors.to_vec(),
        retry_options: retry_options.cloned(),
    }
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
    http_config: &HttpConfig,
    http_client: &reqwest::Client,
    retry_options: Option<&RetryOptions>,
    interceptors: &[Arc<dyn HttpInterceptor>],
    request: MusicGenerationRequest,
) -> Result<MusicGenerationResponse, LlmError> {
    let config = build_http_execution_config(
        api_key,
        base_url,
        http_config,
        http_client,
        retry_options,
        interceptors,
    );
    let url = MinimaxiMusicSpec::new().music_generation_url(&config.provider_context);

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

    let body = serde_json::to_value(minimaxi_request)
        .map_err(|e| LlmError::ParseError(format!("Failed to serialize music request: {}", e)))?;
    let res = execute_json_request(&config, &url, HttpBody::Json(body), None, false).await?;
    let music_response: MinimaxiMusicResponse = serde_json::from_value(res.json).map_err(|e| {
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
