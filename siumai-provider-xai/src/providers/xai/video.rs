use super::XaiClient;
use super::http::{build_http_execution_config, headers_to_map};
use crate::error::LlmError;
use crate::provider_options::{XaiVideoMode, XaiVideoOptions, XaiVideoResolution};
use crate::types::video::{
    VideoGenerationInput, VideoGenerationRequest, VideoGenerationResponse, VideoTaskStatus,
    VideoTaskStatusResponse,
};
use crate::types::{BaseResponse, HttpResponseInfo, Warning};
use crate::utils::mime::guess_mime_from_bytes;
use siumai_core::video::VideoPollingOptions;
use std::collections::HashMap;
use std::time::Duration;

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

fn string_list_from_extra(
    extra_params: Option<&HashMap<String, serde_json::Value>>,
    primary: &str,
    alias: &str,
) -> Result<Option<Vec<String>>, LlmError> {
    let value = extra_params.and_then(|params| params.get(primary).or_else(|| params.get(alias)));
    let Some(value) = value else {
        return Ok(None);
    };
    let Some(values) = value.as_array() else {
        return Err(LlmError::InvalidParameter(format!(
            "xAI video extra param '{primary}' must be an array of strings when provided"
        )));
    };

    let mut parsed = Vec::with_capacity(values.len());
    for entry in values {
        let Some(entry) = entry.as_str() else {
            return Err(LlmError::InvalidParameter(format!(
                "xAI video extra param '{primary}' must contain only strings"
            )));
        };
        parsed.push(entry.to_string());
    }

    Ok(Some(parsed))
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
    if options.mode.is_none()
        && let Some(value) = string_from_extra(extra_params, "mode", "mode")?
    {
        options.mode = Some(XaiVideoMode::from(value));
    }
    if options.video_url.is_none() {
        options.video_url = string_from_extra(extra_params, "video_url", "videoUrl")?;
    }
    if options.reference_image_urls.is_none() {
        options.reference_image_urls =
            string_list_from_extra(extra_params, "reference_image_urls", "referenceImageUrls")?;
    }

    Ok(options)
}

pub(super) fn polling_options(
    client: &XaiClient,
    request: &VideoGenerationRequest,
) -> Result<VideoPollingOptions, LlmError> {
    let mut request = request.clone();
    client.merge_default_provider_options_map_non_chat(&mut request.provider_options_map);
    let options = resolve_xai_video_options(&request)?;

    Ok(VideoPollingOptions {
        poll_interval: options.poll_interval_ms.map(Duration::from_millis),
        poll_timeout: options.poll_timeout_ms.map(Duration::from_millis),
    })
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum XaiVideoCreateRoute {
    Generation,
    Edit,
    Extension,
}

fn resolve_video_mode(
    request: &VideoGenerationRequest,
    options: &XaiVideoOptions,
) -> Option<XaiVideoMode> {
    options.mode.clone().or_else(|| {
        if options.video_url.is_some() || request.video.is_some() {
            Some(XaiVideoMode::EditVideo)
        } else if options
            .reference_image_urls
            .as_ref()
            .is_some_and(|urls| !urls.is_empty())
        {
            Some(XaiVideoMode::ReferenceToVideo)
        } else {
            None
        }
    })
}

fn build_create_body(
    request: &VideoGenerationRequest,
) -> Result<(serde_json::Value, XaiVideoCreateRoute, Vec<Warning>), LlmError> {
    let xai_options = resolve_xai_video_options(request)?;
    let mut warnings = Vec::new();
    let mut body = serde_json::Map::new();
    let prompt = request
        .prompt
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            LlmError::InvalidParameter("xAI video requests require a non-empty prompt".to_string())
        })?;
    body.insert("model".to_string(), serde_json::json!(request.model));
    body.insert("prompt".to_string(), serde_json::json!(prompt));

    let provider_video_url = xai_options.video_url.clone();
    let effective_mode = resolve_video_mode(request, &xai_options);
    let is_edit = matches!(effective_mode.as_ref(), Some(XaiVideoMode::EditVideo));
    let is_extension = matches!(effective_mode.as_ref(), Some(XaiVideoMode::ExtendVideo));
    let has_reference_images = matches!(
        effective_mode.as_ref(),
        Some(XaiVideoMode::ReferenceToVideo)
    );

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
    } else if is_extension {
        if request.aspect_ratio.is_some() {
            push_warning(
                &mut warnings,
                "aspect_ratio",
                "xAI video extension does not support custom aspect ratios.",
            );
        }
        if request.resolution.is_some() || xai_options.resolution.is_some() {
            push_warning(
                &mut warnings,
                "resolution",
                "xAI video extension does not support custom resolutions.",
            );
        }
    }

    let allow_duration = !is_edit;
    let allow_aspect_ratio = !is_edit && !is_extension;
    let allow_resolution = !is_edit && !is_extension;

    if allow_duration && let Some(duration) = request.duration {
        body.insert("duration".to_string(), serde_json::json!(duration));
    }
    if allow_aspect_ratio {
        if let Some(aspect_ratio) = request.aspect_ratio.as_ref().cloned().or_else(|| {
            string_from_extra(request.extra_params.as_ref(), "aspect_ratio", "aspectRatio")
                .ok()
                .flatten()
        }) {
            body.insert("aspect_ratio".to_string(), serde_json::json!(aspect_ratio));
        }
    }

    if allow_resolution {
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

    if is_edit || is_extension {
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
    }

    if let Some(image) = request.image.as_ref() {
        body.insert(
            "image".to_string(),
            serde_json::json!({ "url": input_to_url(image, "image/png")? }),
        );
    }

    if has_reference_images
        && let Some(reference_image_urls) = xai_options.reference_image_urls.as_ref()
    {
        body.insert(
            "reference_images".to_string(),
            serde_json::Value::Array(
                reference_image_urls
                    .iter()
                    .cloned()
                    .map(|url| serde_json::json!({ "url": url }))
                    .collect(),
            ),
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
                    | "mode"
                    | "video_url"
                    | "videoUrl"
                    | "reference_image_urls"
                    | "referenceImageUrls"
                    | "aspect_ratio"
                    | "aspectRatio"
            ) {
                continue;
            }
            body.entry(key.clone()).or_insert_with(|| value.clone());
        }
    }

    let route = if is_extension {
        XaiVideoCreateRoute::Extension
    } else if is_edit {
        XaiVideoCreateRoute::Edit
    } else {
        XaiVideoCreateRoute::Generation
    };

    Ok((serde_json::Value::Object(body), route, warnings))
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
    let (body, route, warnings) = build_create_body(&request)?;
    let config = build_http_execution_config(client);
    let url = format!(
        "{}/videos/{}",
        client.base_url().trim_end_matches('/'),
        match route {
            XaiVideoCreateRoute::Generation => "generations",
            XaiVideoCreateRoute::Edit => "edits",
            XaiVideoCreateRoute::Extension => "extensions",
        }
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
        provider_reference: None,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider_options::XaiVideoMode;
    use crate::providers::xai::ext::video_options::XaiVideoRequestExt;
    use crate::types::ProviderOptionsMap;

    #[test]
    fn build_create_body_routes_extend_video_requests_to_extensions_endpoint() {
        let request = VideoGenerationRequest::new("grok-imagine-video", "extend the clip")
            .with_duration(6)
            .with_aspect_ratio("16:9")
            .with_xai_video_options(
                XaiVideoOptions::new()
                    .with_mode(XaiVideoMode::ExtendVideo)
                    .with_video_url("https://example.com/input.mp4")
                    .with_resolution("720p"),
            );

        let (body, route, warnings) = build_create_body(&request).expect("build body");

        assert_eq!(route, XaiVideoCreateRoute::Extension);
        assert_eq!(body["duration"], serde_json::json!(6));
        assert_eq!(
            body["video"]["url"],
            serde_json::json!("https://example.com/input.mp4")
        );
        assert!(body.get("aspect_ratio").is_none());
        assert!(body.get("resolution").is_none());
        assert_eq!(
            warnings,
            vec![
                Warning::unsupported(
                    "aspect_ratio",
                    Some("xAI video extension does not support custom aspect ratios."),
                ),
                Warning::unsupported(
                    "resolution",
                    Some("xAI video extension does not support custom resolutions."),
                ),
            ]
        );
    }

    #[test]
    fn build_create_body_routes_reference_to_video_requests_with_reference_images() {
        let request = VideoGenerationRequest::new("grok-imagine-video", "animate this style")
            .with_duration(4)
            .with_aspect_ratio("16:9")
            .with_xai_video_options(
                XaiVideoOptions::new()
                    .with_mode(XaiVideoMode::ReferenceToVideo)
                    .with_resolution("720p")
                    .with_reference_image_urls([
                        "https://example.com/ref-1.png",
                        "https://example.com/ref-2.png",
                    ]),
            );

        let (body, route, warnings) = build_create_body(&request).expect("build body");

        assert_eq!(route, XaiVideoCreateRoute::Generation);
        assert_eq!(body["duration"], serde_json::json!(4));
        assert_eq!(body["aspect_ratio"], serde_json::json!("16:9"));
        assert_eq!(body["resolution"], serde_json::json!("720p"));
        assert_eq!(
            body["reference_images"],
            serde_json::json!([
                { "url": "https://example.com/ref-1.png" },
                { "url": "https://example.com/ref-2.png" }
            ])
        );
        assert!(warnings.is_empty());
    }

    #[test]
    fn build_create_body_rejects_promptless_requests() {
        let request = VideoGenerationRequest::new_without_prompt("grok-imagine-video").with_image(
            VideoGenerationInput::file_with_media_type(vec![1, 2, 3], "image/png"),
        );

        let err = build_create_body(&request).unwrap_err();
        assert!(
            matches!(err, LlmError::InvalidParameter(message) if message.contains("require a non-empty prompt"))
        );
    }

    #[tokio::test]
    async fn polling_options_merges_client_defaults_and_provider_option_overrides() {
        let mut defaults = ProviderOptionsMap::default();
        defaults.insert(
            "xai",
            serde_json::json!({
                "pollIntervalMs": 500,
                "pollTimeoutMs": 30_000
            }),
        );
        let client = super::super::XaiClient::from_config(
            super::super::XaiConfig::new("test-key")
                .with_model("grok-imagine-video")
                .with_provider_options_map(defaults),
        )
        .await
        .expect("xai client");
        let request = VideoGenerationRequest::new("grok-imagine-video", "animate")
            .with_xai_video_options(XaiVideoOptions::new().with_poll_interval_ms(250));

        let options = polling_options(&client, &request).expect("xai polling options");

        assert_eq!(options.poll_interval, Some(Duration::from_millis(250)));
        assert_eq!(options.poll_timeout, Some(Duration::from_millis(30_000)));
    }
}
