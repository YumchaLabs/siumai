use crate::request_options::{EffectiveRequestOptions, retry_or_call_with_abort};
use crate::video::{CreateTaskOptions, GenerateOptions, QueryTaskOptions};
use siumai_core::error::LlmError;
use siumai_core::types::{
    HttpConfig, VideoGenerationRequest, VideoGenerationResponse, VideoTaskStatusResponse,
};
use siumai_core::video::VideoModel;
use std::collections::HashMap;
use std::time::Duration;

pub(super) fn apply_video_call_options(
    mut request: VideoGenerationRequest,
    timeout: Option<Duration>,
    headers: HashMap<String, String>,
) -> VideoGenerationRequest {
    if timeout.is_some() || !headers.is_empty() {
        let mut http = request.http_config.take().unwrap_or_else(HttpConfig::empty);
        if let Some(timeout) = timeout {
            http.timeout = Some(timeout);
        }
        if !headers.is_empty() {
            http.headers.extend(headers);
        }
        request.http_config = Some(http);
    }

    request
}

/// Submit a video-generation task.
pub async fn create_task<M: VideoModel + ?Sized>(
    model: &M,
    request: VideoGenerationRequest,
    options: CreateTaskOptions,
) -> Result<VideoGenerationResponse, LlmError> {
    let effective = EffectiveRequestOptions::from_parts(
        options.request_options,
        options.retry,
        options.timeout,
        options.headers,
    );
    let request = apply_video_call_options(request, effective.timeout(), effective.headers());
    retry_or_call_with_abort(effective.retry(), effective.abort_signal(), || {
        let request = request.clone();
        async move { model.create_task(request).await }
    })
    .await
}

/// Query a video-generation task.
pub async fn query_task<M: VideoModel + ?Sized>(
    model: &M,
    task_id: &str,
    options: QueryTaskOptions,
) -> Result<VideoTaskStatusResponse, LlmError> {
    let effective = EffectiveRequestOptions::from_parts(
        options.request_options,
        options.retry,
        None,
        HashMap::new(),
    );
    retry_or_call_with_abort(effective.retry(), effective.abort_signal(), || {
        let task_id = task_id.to_string();
        async move { model.query_task(&task_id).await }
    })
    .await
}

pub(super) fn validate_poll_interval(poll_interval: Duration) -> Result<(), LlmError> {
    if poll_interval.is_zero() {
        return Err(LlmError::InvalidParameter(
            "video polling interval must be greater than 0".to_string(),
        ));
    }

    Ok(())
}

pub(super) fn resolve_generate_polling_options<M: VideoModel + ?Sized>(
    model: &M,
    request: &VideoGenerationRequest,
    options: &GenerateOptions,
) -> Result<(Duration, Option<Duration>), LlmError> {
    let provider_options = model.polling_options(request)?;
    let poll_interval = provider_options
        .poll_interval
        .unwrap_or(options.poll_interval);
    validate_poll_interval(poll_interval)?;

    Ok((
        poll_interval,
        provider_options.poll_timeout.or(options.poll_timeout),
    ))
}

fn resolve_requested_video_count(request: &VideoGenerationRequest) -> Result<u32, LlmError> {
    let requested_count = request.count.unwrap_or(1);
    if requested_count == 0 {
        return Err(LlmError::InvalidParameter(
            "VideoGenerationRequest.count must be greater than 0".to_string(),
        ));
    }

    Ok(requested_count)
}

pub(super) fn resolve_effective_max_videos_per_call(
    explicit: Option<u32>,
    model_default: Option<u32>,
) -> Result<u32, LlmError> {
    let limit = explicit.or(model_default).unwrap_or(1);
    if limit == 0 {
        return Err(LlmError::InvalidParameter(
            "GenerateOptions.max_videos_per_call must be greater than 0".to_string(),
        ));
    }

    Ok(limit)
}

fn split_call_video_counts(total_videos: u32, max_videos_per_call: u32) -> Vec<u32> {
    let mut remaining = total_videos;
    let mut counts = Vec::new();
    while remaining > 0 {
        let current = remaining.min(max_videos_per_call);
        counts.push(current);
        remaining -= current;
    }
    counts
}

pub(super) fn split_generate_requests(
    request: VideoGenerationRequest,
    max_videos_per_call: u32,
) -> Result<Vec<VideoGenerationRequest>, LlmError> {
    let requested_count = resolve_requested_video_count(&request)?;
    let call_counts = split_call_video_counts(requested_count, max_videos_per_call);

    let mut requests = Vec::with_capacity(call_counts.len());
    for count in call_counts {
        let mut split = request.clone();
        split.count = Some(count);
        requests.push(split);
    }

    Ok(requests)
}

pub(super) fn build_failed_task_error(
    task_id: &str,
    response: &VideoTaskStatusResponse,
) -> LlmError {
    let message = response
        .base_resp
        .as_ref()
        .map(|base| base.status_msg.clone())
        .filter(|message| !message.trim().is_empty())
        .unwrap_or_else(|| format!("Video task '{task_id}' failed"));

    LlmError::ProcessingError(format!("Video task '{task_id}' failed: {message}"))
}
