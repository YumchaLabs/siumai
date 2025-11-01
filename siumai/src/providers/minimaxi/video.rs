//! MiniMaxi Video Generation Helper Functions
//!
//! Internal helper functions for video generation capability implementation.

use crate::error::LlmError;
use crate::types::video::{
    VideoGenerationRequest, VideoGenerationResponse, VideoTaskStatusResponse,
};

/// Get video generation endpoint URL
pub(super) fn video_generation_url(base_url: &str) -> String {
    format!("{}/v1/video_generation", base_url.trim_end_matches('/'))
}

/// Get video query endpoint URL
pub(super) fn video_query_url(base_url: &str, task_id: &str) -> String {
    format!(
        "{}/v1/query/video_generation?task_id={}",
        base_url.trim_end_matches('/'),
        task_id
    )
}

/// Create video task
pub(super) async fn create_video_task(
    api_key: &str,
    base_url: &str,
    http_client: &reqwest::Client,
    request: VideoGenerationRequest,
) -> Result<VideoGenerationResponse, LlmError> {
    let url = video_generation_url(base_url);

    // Build HTTP request
    let req_builder = http_client
        .post(&url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request);

    // Execute request
    let response = req_builder.send().await.map_err(|e| {
        LlmError::provider_error(
            "minimaxi",
            format!("Failed to send video generation request: {}", e),
        )
    })?;

    // Check status
    let status = response.status();
    if !status.is_success() {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(LlmError::provider_error(
            "minimaxi",
            format!(
                "Video generation failed with status {}: {}",
                status, error_text
            ),
        ));
    }

    // Parse response
    let response_json: VideoGenerationResponse = response.json().await.map_err(|e| {
        LlmError::ParseError(format!("Failed to parse video generation response: {}", e))
    })?;

    Ok(response_json)
}

/// Query video task status
pub(super) async fn query_video_task(
    api_key: &str,
    base_url: &str,
    http_client: &reqwest::Client,
    task_id: &str,
) -> Result<VideoTaskStatusResponse, LlmError> {
    let url = video_query_url(base_url, task_id);

    // Build HTTP request
    let req_builder = http_client
        .get(&url)
        .header("Authorization", format!("Bearer {}", api_key));

    // Execute request
    let response = req_builder.send().await.map_err(|e| {
        LlmError::provider_error(
            "minimaxi",
            format!("Failed to query video task status: {}", e),
        )
    })?;

    // Check status
    let status = response.status();
    if !status.is_success() {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(LlmError::provider_error(
            "minimaxi",
            format!("Video query failed with status {}: {}", status, error_text),
        ));
    }

    // Parse response
    let response_json: VideoTaskStatusResponse = response.json().await.map_err(|e| {
        LlmError::ParseError(format!("Failed to parse video task status response: {}", e))
    })?;

    Ok(response_json)
}

/// Get supported video models
pub(super) fn get_supported_video_models() -> Vec<String> {
    vec![
        "MiniMax-Hailuo-2.3".to_string(),
        "MiniMax-Hailuo-02".to_string(),
        "T2V-01-Director".to_string(),
        "T2V-01".to_string(),
    ]
}

/// Get supported resolutions for a model
pub(super) fn get_supported_resolutions(model: &str) -> Vec<String> {
    match model {
        "MiniMax-Hailuo-2.3" | "MiniMax-Hailuo-02" => {
            vec!["768P".to_string(), "1080P".to_string()]
        }
        "T2V-01-Director" | "T2V-01" => {
            vec!["720P".to_string()]
        }
        _ => vec![],
    }
}

/// Get supported durations for a model
pub(super) fn get_supported_durations(model: &str) -> Vec<u32> {
    match model {
        "MiniMax-Hailuo-2.3" | "MiniMax-Hailuo-02" => {
            vec![6, 10]
        }
        "T2V-01-Director" | "T2V-01" => {
            vec![6]
        }
        _ => vec![],
    }
}
