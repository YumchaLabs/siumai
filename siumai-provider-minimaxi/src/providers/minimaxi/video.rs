//! MiniMaxi Video Generation Helper Functions
//!
//! Internal helper functions for video generation capability implementation.

use crate::error::LlmError;
use crate::execution::executors::common::{HttpBody, execute_get_request, execute_json_request};
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::wiring::HttpExecutionWiring;
use crate::retry_api::RetryOptions;
use crate::types::HttpConfig;
use crate::types::video::{
    VideoGenerationRequest, VideoGenerationResponse, VideoTaskStatusResponse,
};
use std::sync::Arc;

use super::spec::MinimaxiVideoSpec;

fn build_request_body(request: VideoGenerationRequest) -> serde_json::Value {
    let mut body = serde_json::Map::new();
    body.insert("model".to_string(), serde_json::json!(request.model));
    body.insert("prompt".to_string(), serde_json::json!(request.prompt));

    if let Some(duration) = request.duration {
        body.insert("duration".to_string(), serde_json::json!(duration));
    }
    if let Some(resolution) = request.resolution {
        body.insert("resolution".to_string(), serde_json::json!(resolution));
    }
    if let Some(prompt_optimizer) = request.prompt_optimizer {
        body.insert(
            "prompt_optimizer".to_string(),
            serde_json::json!(prompt_optimizer),
        );
    }
    if let Some(fast_pretreatment) = request.fast_pretreatment {
        body.insert(
            "fast_pretreatment".to_string(),
            serde_json::json!(fast_pretreatment),
        );
    }
    if let Some(callback_url) = request.callback_url {
        body.insert("callback_url".to_string(), serde_json::json!(callback_url));
    }
    if let Some(aigc_watermark) = request.aigc_watermark {
        body.insert(
            "aigc_watermark".to_string(),
            serde_json::json!(aigc_watermark),
        );
    }

    if let Some(extra_params) = request.extra_params {
        for (key, value) in extra_params {
            body.entry(key).or_insert(value);
        }
    }

    serde_json::Value::Object(body)
}

fn build_wiring(
    api_key: &str,
    base_url: &str,
    http_config: &HttpConfig,
    http_client: &reqwest::Client,
    retry_options: Option<&RetryOptions>,
    interceptors: &[Arc<dyn HttpInterceptor>],
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
) -> HttpExecutionWiring {
    let mut wiring = HttpExecutionWiring::new(
        "minimaxi",
        http_client.clone(),
        super::utils::build_context(api_key, base_url, http_config),
    )
    .with_interceptors(interceptors.to_vec())
    .with_retry_options(retry_options.cloned());

    if let Some(transport) = http_transport {
        wiring = wiring.with_transport(transport);
    }

    wiring
}

/// Create video task
#[allow(clippy::too_many_arguments)]
pub(super) async fn create_video_task(
    api_key: &str,
    base_url: &str,
    http_config: &HttpConfig,
    http_client: &reqwest::Client,
    retry_options: Option<&RetryOptions>,
    interceptors: &[Arc<dyn HttpInterceptor>],
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    request: VideoGenerationRequest,
) -> Result<VideoGenerationResponse, LlmError> {
    let wiring = build_wiring(
        api_key,
        base_url,
        http_config,
        http_client,
        retry_options,
        interceptors,
        http_transport,
    );
    let config = wiring.config(Arc::new(MinimaxiVideoSpec::new()));
    let url = MinimaxiVideoSpec::new().video_generation_url(&config.provider_context);

    let body = build_request_body(request);
    let res = execute_json_request(&config, &url, HttpBody::Json(body), None, false).await?;
    serde_json::from_value(res.json).map_err(|e| {
        LlmError::ParseError(format!("Failed to parse video generation response: {}", e))
    })
}

/// Query video task status
#[allow(clippy::too_many_arguments)]
pub(super) async fn query_video_task(
    api_key: &str,
    base_url: &str,
    http_config: &HttpConfig,
    http_client: &reqwest::Client,
    retry_options: Option<&RetryOptions>,
    interceptors: &[Arc<dyn HttpInterceptor>],
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    task_id: &str,
) -> Result<VideoTaskStatusResponse, LlmError> {
    let wiring = build_wiring(
        api_key,
        base_url,
        http_config,
        http_client,
        retry_options,
        interceptors,
        http_transport,
    );
    let config = wiring.config(Arc::new(MinimaxiVideoSpec::new()));
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
