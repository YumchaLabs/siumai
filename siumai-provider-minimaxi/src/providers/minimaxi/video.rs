//! MiniMaxi Video Generation Helper Functions
//!
//! Internal helper functions for video generation capability implementation.

use crate::error::LlmError;
use crate::execution::executors::common::{HttpBody, execute_get_request, execute_json_request};
use crate::execution::http::interceptor::HttpInterceptor;
use crate::retry_api::RetryOptions;
use crate::types::HttpConfig;
use crate::types::video::{
    VideoGenerationRequest, VideoGenerationResponse, VideoTaskStatusResponse,
};
use std::sync::Arc;

use super::spec::MinimaxiVideoSpec;

fn build_http_execution_config(
    api_key: &str,
    base_url: &str,
    http_config: &HttpConfig,
    http_client: &reqwest::Client,
    retry_options: Option<&RetryOptions>,
    interceptors: &[Arc<dyn HttpInterceptor>],
) -> crate::execution::executors::common::HttpExecutionConfig {
    crate::execution::executors::common::HttpExecutionConfig {
        provider_id: "minimaxi".to_string(),
        http_client: http_client.clone(),
        provider_spec: Arc::new(MinimaxiVideoSpec::new()),
        provider_context: super::utils::build_context(api_key, base_url, http_config),
        interceptors: interceptors.to_vec(),
        retry_options: retry_options.cloned(),
    }
}

/// Create video task
pub(super) async fn create_video_task(
    api_key: &str,
    base_url: &str,
    http_config: &HttpConfig,
    http_client: &reqwest::Client,
    retry_options: Option<&RetryOptions>,
    interceptors: &[Arc<dyn HttpInterceptor>],
    request: VideoGenerationRequest,
) -> Result<VideoGenerationResponse, LlmError> {
    let config = build_http_execution_config(
        api_key,
        base_url,
        http_config,
        http_client,
        retry_options,
        interceptors,
    );
    let url = MinimaxiVideoSpec::new().video_generation_url(&config.provider_context);

    let body = serde_json::to_value(request)
        .map_err(|e| LlmError::ParseError(format!("Failed to serialize video request: {}", e)))?;
    let res = execute_json_request(&config, &url, HttpBody::Json(body), None, false).await?;
    serde_json::from_value(res.json).map_err(|e| {
        LlmError::ParseError(format!("Failed to parse video generation response: {}", e))
    })
}

/// Query video task status
pub(super) async fn query_video_task(
    api_key: &str,
    base_url: &str,
    http_config: &HttpConfig,
    http_client: &reqwest::Client,
    retry_options: Option<&RetryOptions>,
    interceptors: &[Arc<dyn HttpInterceptor>],
    task_id: &str,
) -> Result<VideoTaskStatusResponse, LlmError> {
    let config = build_http_execution_config(
        api_key,
        base_url,
        http_config,
        http_client,
        retry_options,
        interceptors,
    );
    let url = MinimaxiVideoSpec::new().video_query_url(&config.provider_context, task_id);

    let res = execute_get_request(&config, &url, None).await?;
    serde_json::from_value(res.json).map_err(|e| {
        LlmError::ParseError(format!("Failed to parse video task status response: {}", e))
    })
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
