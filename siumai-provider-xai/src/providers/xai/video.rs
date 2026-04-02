use super::XaiClient;
use super::http::{build_http_execution_config, headers_to_map};
use crate::error::LlmError;
use crate::provider_options::{XaiVideoOptions, XaiVideoResolution};
use crate::types::video::{
    VideoGenerationInput, VideoGenerationRequest, VideoGenerationResponse, VideoTaskStatus,
    VideoTaskStatusResponse,
};
use crate::types::{BaseResponse, HttpResponseInfo, Warning};
use crate::utils::mime::guess_mime_from_bytes;
use std::collections::HashMap;

fn parse_xai_video_options(
    map: &crate::types::ProviderOptionsMap,
) -> Result<Option<XaiVideoOptions>, LlmError> {
    let Some(value) = map.get("xai") else {
        return Ok(None);
    };

    serde_json::from_value(value.clone())
        .map(Some)
        .map_err(|err| {
            LlmError::InvalidParameter(format!(
                "Invalid xAI video options in providerOptions.xai: {err}"
            ))
        })
}

fn string_from_extra(
    extra_params: Option<&HashMap<String, serde_json::Value>>,
    primary: &str,
    alias: &str,
) -> Result<Option<String>, LlmError> {
    let value = extra_params.and_then(|params| params.get(primary).or_else(|| params.get(alias)));
    let Some(value) = value else {
        return Ok(None);
    };
    value.as_str().map(|v| Some(v.to_string())).ok_or_else(|| {
        LlmError::InvalidParameter(format!(
            "xAI video extra param '{primary}' must be a string when provided"
        ))
    })
}

fn u64_from_extra(
    extra_params: Option<&HashMap<String, serde_json::Value>>,
    primary: &str,
    alias: &str,
) -> Result<Option<u64>, LlmError> {
    let value = extra_params.and_then(|params| params.get(primary).or_else(|| params.get(alias)));
    let Some(value) = value else {
        return Ok(None);
    };
    value.as_u64().map(Some).ok_or_else(|| {
        LlmError::InvalidParameter(format!(
            "xAI video extra param '{primary}' must be an unsigned integer when provided"
        ))
    })
}

fn input_to_url(input: &VideoGenerationInput, default_mime: &str) -> Result<String, LlmError> {
    match input {
        VideoGenerationInput::Url { url, .. } => Ok(url.clone()),
        VideoGenerationInput::File {
            data, media_type, ..
        } => {
            let mime = if let Some(media_type) = media_type {
                media_type.clone()
            } else {
                let bytes = data.as_bytes().map_err(|err| {
                    LlmError::InvalidParameter(format!(
                        "Invalid base64 video input payload for xAI video request: {err}"
                    ))
                })?;
                guess_mime_from_bytes(&bytes).unwrap_or_else(|| default_mime.to_string())
            };
            Ok(format!("data:{mime};base64,{}", data.as_base64()))
        }
    }
}

fn push_warning(warnings: &mut Vec<Warning>, feature: &str, details: &'static str) {
    warnings.push(Warning::unsupported(feature, Some(details)));
}

fn resolve_xai_video_options(
    request: &VideoGenerationRequest,
) -> Result<XaiVideoOptions, LlmError> {
    let mut options = parse_xai_video_options(&request.provider_options_map)?.unwrap_or_default();
    let extra_params = request.extra_params.as_ref();

    if options.poll_interval_ms.is_none() {
        options.poll_interval_ms =
            u64_from_extra(extra_params, "poll_interval_ms", "pollIntervalMs")?;
    }
    if options.poll_timeout_ms.is_none() {
        options.poll_timeout_ms = u64_from_extra(extra_params, "poll_timeout_ms", "pollTimeoutMs")?;
    }
    if options.resolution.is_none()
        && let Some(value) = string_from_extra(extra_params, "resolution", "resolution")?
    {
        options.resolution = Some(XaiVideoResolution::from(value));
    }
    if options.video_url.is_none() {
        options.video_url = string_from_extra(extra_params, "video_url", "videoUrl")?;
    }

    Ok(options)
}

fn map_resolution(value: &str) -> Option<&'static str> {
    match value.trim() {
        "480p" | "480P" | "854x480" | "640x480" => Some("480p"),
        "720p" | "720P" | "1280x720" => Some("720p"),
        _ => None,
    }
}

fn status_from_wire(status: Option<&str>, has_video_url: bool) -> VideoTaskStatus {
    match status.map(|value| value.to_ascii_lowercase()) {
        Some(value) if value == "done" => VideoTaskStatus::Success,
        Some(value) if value == "expired" || value == "failed" || value == "error" => {
            VideoTaskStatus::Fail
        }
        Some(value) if value == "queueing" || value == "queued" => VideoTaskStatus::Queueing,
        Some(value) if value == "preparing" => VideoTaskStatus::Preparing,
        Some(value) if value == "processing" || value == "running" || value == "pending" => {
            VideoTaskStatus::Processing
        }
        None if has_video_url => VideoTaskStatus::Success,
        _ => VideoTaskStatus::Processing,
    }
}

fn build_create_body(
    request: &VideoGenerationRequest,
) -> Result<(serde_json::Value, bool, Vec<Warning>), LlmError> {
    let xai_options = resolve_xai_video_options(request)?;
    let mut warnings = Vec::new();
    let mut body = serde_json::Map::new();
    body.insert("model".to_string(), serde_json::json!(request.model));
    body.insert("prompt".to_string(), serde_json::json!(request.prompt));

    let provider_video_url = xai_options.video_url.clone();
    let is_edit = provider_video_url.is_some() || request.video.is_some();

    if request.count.unwrap_or(1) > 1 {
        push_warning(
            &mut warnings,
            "n",
            "xAI video models do not support generating multiple videos per call.",
        );
    }
    if request.fps.is_some() {
        push_warning(
            &mut warnings,
            "fps",
            "xAI video models do not support custom FPS.",
        );
    }
    if request.seed.is_some() {
        push_warning(
            &mut warnings,
            "seed",
            "xAI video models do not support deterministic seeds.",
        );
    }

    if is_edit {
        if request.duration.is_some() {
            push_warning(
                &mut warnings,
                "duration",
                "xAI video editing does not support custom duration.",
            );
        }
        if request.aspect_ratio.is_some() {
            push_warning(
                &mut warnings,
                "aspect_ratio",
                "xAI video editing does not support custom aspect ratios.",
            );
        }
        if request.resolution.is_some() || xai_options.resolution.is_some() {
            push_warning(
                &mut warnings,
                "resolution",
                "xAI video editing does not support custom resolutions.",
            );
        }
    } else {
        if let Some(duration) = request.duration {
            body.insert("duration".to_string(), serde_json::json!(duration));
        }
        if let Some(aspect_ratio) = request.aspect_ratio.as_ref().cloned().or_else(|| {
            string_from_extra(request.extra_params.as_ref(), "aspect_ratio", "aspectRatio")
                .ok()
                .flatten()
        }) {
            body.insert("aspect_ratio".to_string(), serde_json::json!(aspect_ratio));
        }

        if let Some(resolution) = xai_options.resolution {
            body.insert(
                "resolution".to_string(),
                serde_json::json!(resolution.as_str()),
            );
        } else if let Some(resolution) = request.resolution.as_deref() {
            if let Some(mapped) = map_resolution(resolution) {
                body.insert("resolution".to_string(), serde_json::json!(mapped));
            } else {
                push_warning(
                    &mut warnings,
                    "resolution",
                    "Unrecognized xAI video resolution. Use `480p`, `720p`, or providerOptions.xai.resolution.",
                );
            }
        }
    }

    if let Some(video_url) = provider_video_url {
        if request.video.is_some() {
            warnings.push(Warning::compatibility(
                "video",
                Some(
                    "providerOptions.xai.videoUrl takes precedence over `request.video` on the xAI provider-owned path.",
                ),
            ));
        }
        body.insert("video".to_string(), serde_json::json!({ "url": video_url }));
    } else if let Some(video) = request.video.as_ref() {
        body.insert(
            "video".to_string(),
            serde_json::json!({ "url": input_to_url(video, "video/mp4")? }),
        );
    }

    if let Some(image) = request.image.as_ref() {
        body.insert(
            "image".to_string(),
            serde_json::json!({ "url": input_to_url(image, "image/png")? }),
        );
    }

    for (key, value) in xai_options.extra_fields {
        body.entry(key).or_insert(value);
    }

    if let Some(extra_params) = request.extra_params.as_ref() {
        for (key, value) in extra_params {
            if matches!(
                key.as_str(),
                "poll_interval_ms"
                    | "pollIntervalMs"
                    | "poll_timeout_ms"
                    | "pollTimeoutMs"
                    | "resolution"
                    | "video_url"
                    | "videoUrl"
                    | "aspect_ratio"
                    | "aspectRatio"
            ) {
                continue;
            }
            body.entry(key.clone()).or_insert_with(|| value.clone());
        }
    }

    Ok((serde_json::Value::Object(body), is_edit, warnings))
}

#[derive(Debug, serde::Deserialize)]
struct XaiCreateVideoResponse {
    #[serde(default)]
    request_id: Option<String>,
    #[serde(flatten)]
    extra_fields: HashMap<String, serde_json::Value>,
}

#[derive(Debug, serde::Deserialize)]
struct XaiVideoStatusResponse {
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    video: Option<XaiVideoAsset>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    usage: Option<XaiVideoUsage>,
    #[serde(flatten)]
    extra_fields: HashMap<String, serde_json::Value>,
}

#[derive(Debug, serde::Deserialize)]
struct XaiVideoAsset {
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    duration: Option<f32>,
    #[serde(default)]
    respect_moderation: Option<bool>,
    #[serde(flatten)]
    extra_fields: HashMap<String, serde_json::Value>,
}

#[derive(Debug, serde::Deserialize)]
struct XaiVideoUsage {
    #[serde(default)]
    cost_in_usd_ticks: Option<serde_json::Value>,
    #[serde(flatten)]
    extra_fields: HashMap<String, serde_json::Value>,
}

pub(super) async fn create_video_task(
    client: &XaiClient,
    mut request: VideoGenerationRequest,
) -> Result<VideoGenerationResponse, LlmError> {
    client.merge_default_provider_options_map_non_chat(&mut request.provider_options_map);
    let (body, is_edit, warnings) = build_create_body(&request)?;
    let config = build_http_execution_config(client);
    let url = format!(
        "{}/videos/{}",
        client.base_url().trim_end_matches('/'),
        if is_edit { "edits" } else { "generations" }
    );
    let result = crate::execution::executors::common::execute_json_request(
        &config,
        &url,
        crate::execution::executors::common::HttpBody::Json(body),
        request.http_config.as_ref(),
        false,
    )
    .await?;

    let parsed: XaiCreateVideoResponse = serde_json::from_value(result.json).map_err(|err| {
        LlmError::ParseError(format!("Failed to parse xAI video create response: {err}"))
    })?;
    let task_id = parsed.request_id.ok_or_else(|| {
        LlmError::ParseError("xAI video create response did not include `request_id`".to_string())
    })?;

    Ok(VideoGenerationResponse {
        task_id,
        base_resp: Some(BaseResponse {
            status_code: 0,
            status_msg: "ok".to_string(),
        }),
        metadata: parsed.extra_fields,
        warnings: (!warnings.is_empty()).then_some(warnings),
        response: Some(HttpResponseInfo {
            timestamp: chrono::Utc::now(),
            model_id: Some(request.model),
            headers: headers_to_map(&result.headers),
        }),
    })
}

pub(super) async fn query_video_task(
    client: &XaiClient,
    task_id: &str,
) -> Result<VideoTaskStatusResponse, LlmError> {
    let config = build_http_execution_config(client);
    let url = format!(
        "{}/videos/{task_id}",
        client.base_url().trim_end_matches('/')
    );
    let result =
        crate::execution::executors::common::execute_get_request(&config, &url, None).await?;

    let parsed: XaiVideoStatusResponse = serde_json::from_value(result.json).map_err(|err| {
        LlmError::ParseError(format!("Failed to parse xAI video status response: {err}"))
    })?;

    let mut metadata = parsed.extra_fields;
    if let Some(model) = parsed.model.as_ref() {
        metadata.insert("model".to_string(), serde_json::json!(model));
    }
    if let Some(usage) = parsed.usage {
        let mut usage_meta = usage.extra_fields;
        if let Some(cost) = usage.cost_in_usd_ticks {
            usage_meta.insert("cost_in_usd_ticks".to_string(), cost);
        }
        if !usage_meta.is_empty() {
            metadata.insert(
                "usage".to_string(),
                serde_json::Value::Object(usage_meta.into_iter().collect()),
            );
        }
    }

    let (video_url, duration, respect_moderation) = if let Some(video) = parsed.video {
        if !video.extra_fields.is_empty() {
            metadata.insert(
                "video".to_string(),
                serde_json::Value::Object(video.extra_fields.into_iter().collect()),
            );
        }
        (video.url, video.duration, video.respect_moderation)
    } else {
        (None, None, None)
    };

    if let Some(respect_moderation) = respect_moderation {
        metadata.insert(
            "respect_moderation".to_string(),
            serde_json::json!(respect_moderation),
        );
    }

    Ok(VideoTaskStatusResponse {
        task_id: task_id.to_string(),
        status: status_from_wire(parsed.status.as_deref(), video_url.is_some()),
        file_id: None,
        video_url,
        duration,
        video_width: None,
        video_height: None,
        base_resp: Some(BaseResponse {
            status_code: 0,
            status_msg: "ok".to_string(),
        }),
        metadata,
        response: Some(HttpResponseInfo {
            timestamp: chrono::Utc::now(),
            model_id: parsed.model,
            headers: headers_to_map(&result.headers),
        }),
    })
}

pub(super) fn supported_models() -> Vec<String> {
    vec![super::models::video::GROK_IMAGINE_VIDEO.to_string()]
}

pub(super) fn supported_resolutions(_model: &str) -> Vec<String> {
    vec!["480p".to_string(), "720p".to_string()]
}

pub(super) fn supported_durations(_model: &str) -> Vec<u32> {
    Vec::new()
}
