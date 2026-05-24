use crate::video::{GeneratedVideo, GeneratedVideoData, GeneratedVideoMetadata};
use base64::{Engine, engine::general_purpose::STANDARD};
use reqwest::header::CONTENT_TYPE;
use siumai_core::error::LlmError;
use siumai_core::execution::http::build_http_client_from_config;
use siumai_core::types::{
    HttpConfig, MaterializedVideoAsset, ProviderReference, VideoTaskStatusResponse, Warning,
};
use siumai_core::video::VideoModelV4;
use siumai_provider_utils::mime::{guess_mime_from_bytes, guess_mime_from_path_or_url};
use std::collections::HashMap;

fn generated_video_media_type(
    metadata: &GeneratedVideoMetadata,
    fallback_path_or_url: Option<&str>,
) -> String {
    metadata
        .get("mediaType")
        .or_else(|| metadata.get("mimeType"))
        .and_then(|value| value.as_str())
        .filter(|value| !value.trim().is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| fallback_path_or_url.and_then(guess_mime_from_path_or_url))
        .unwrap_or_else(|| "video/mp4".to_string())
}

fn provider_declared_video_media_type(metadata: &GeneratedVideoMetadata) -> Option<String> {
    metadata
        .get("mediaType")
        .or_else(|| metadata.get("mimeType"))
        .and_then(|value| value.as_str())
        .filter(|value| !value.trim().is_empty())
        .map(ToOwned::to_owned)
}

fn usable_video_media_type(value: Option<&str>) -> Option<String> {
    value
        .filter(|value| !value.trim().is_empty() && *value != "application/octet-stream")
        .map(ToOwned::to_owned)
}

fn parse_video_data_url(url: &str) -> Result<(Vec<u8>, Option<String>), LlmError> {
    let Some(payload) = url.strip_prefix("data:") else {
        return Err(LlmError::InvalidParameter(
            "Expected a data URL for generated video materialization".to_string(),
        ));
    };
    let Some((meta, data)) = payload.split_once(',') else {
        return Err(LlmError::InvalidParameter(
            "Invalid generated video data URL".to_string(),
        ));
    };

    let Some(meta) = meta.strip_suffix(";base64") else {
        return Err(LlmError::InvalidParameter(
            "Generated video data URLs must use base64 encoding".to_string(),
        ));
    };

    let bytes = STANDARD.decode(data).map_err(|error| {
        LlmError::InvalidInput(format!(
            "Invalid base64 payload in generated video data URL: {error}"
        ))
    })?;
    let media_type = (!meta.is_empty()).then_some(meta.to_string());
    Ok((bytes, media_type))
}

fn can_materialize_generated_video_url(url: &str) -> bool {
    url.starts_with("data:") || url.starts_with("http://") || url.starts_with("https://")
}

fn generated_video_url_scheme(url: &str) -> &str {
    url.split(':').next().unwrap_or("unknown")
}

pub(super) async fn download_generated_video_url(
    url: &str,
    http_config: Option<&HttpConfig>,
) -> Result<(Vec<u8>, Option<String>), LlmError> {
    if url.starts_with("data:") {
        return parse_video_data_url(url);
    }

    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Err(LlmError::UnsupportedOperation(format!(
            "Unsupported generated video URL scheme '{}' for materialization. Only data:, http:, and https: URLs can be materialized on this path.",
            generated_video_url_scheme(url)
        )));
    }

    let client = if let Some(http_config) = http_config {
        build_http_client_from_config(http_config)?
    } else {
        reqwest::Client::new()
    };

    let response = client.get(url).send().await.map_err(|error| {
        LlmError::HttpError(format!("Failed to download generated video: {error}"))
    })?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(LlmError::ApiError {
            code: status.as_u16(),
            message: format!("Failed to download generated video from {url}"),
            details: Some(serde_json::json!({
                "url": url,
                "body": body,
            })),
        });
    }

    let downloaded_media_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let bytes = response.bytes().await.map_err(|error| {
        LlmError::HttpError(format!(
            "Failed to read generated video download bytes: {error}"
        ))
    })?;

    Ok((bytes.to_vec(), downloaded_media_type))
}

pub(super) fn resolve_materialized_video_media_type(
    video: &GeneratedVideo,
    bytes: Option<&[u8]>,
    downloaded_media_type: Option<String>,
) -> String {
    provider_declared_video_media_type(&video.metadata)
        .or_else(|| usable_video_media_type(downloaded_media_type.as_deref()))
        .or_else(|| bytes.and_then(guess_mime_from_bytes))
        .or_else(|| usable_video_media_type(Some(video.media_type.as_str())))
        .or_else(|| video.url().and_then(guess_mime_from_path_or_url))
        .unwrap_or_else(|| "video/mp4".to_string())
}

pub(super) fn video_url_from_metadata(metadata: &GeneratedVideoMetadata) -> Option<&str> {
    metadata
        .get("url")
        .or_else(|| metadata.get("videoUrl"))
        .and_then(serde_json::Value::as_str)
}

fn provider_reference_from_metadata(
    metadata: &GeneratedVideoMetadata,
) -> Option<ProviderReference> {
    metadata
        .get("providerReference")
        .and_then(|value| serde_json::from_value::<ProviderReference>(value.clone()).ok())
}

fn provider_metadata_root_candidates(provider_id: &str) -> Vec<&str> {
    match provider_id {
        "vertex" => vec!["vertex", "google-vertex"],
        "gemini" => vec!["gemini", "google"],
        other => vec![other],
    }
}

pub(super) fn provider_metadata_root_object<'a>(
    metadata: &'a HashMap<String, serde_json::Value>,
    provider_id: &str,
) -> Option<&'a serde_json::Map<String, serde_json::Value>> {
    provider_metadata_root_candidates(provider_id)
        .into_iter()
        .find_map(|candidate| metadata.get(candidate).and_then(|value| value.as_object()))
}

fn provider_metadata_root_value<'a>(
    metadata: &'a HashMap<String, serde_json::Value>,
    provider_id: &str,
    key: &str,
) -> Option<&'a serde_json::Value> {
    provider_metadata_root_object(metadata, provider_id)?.get(key)
}

fn internal_generated_video_items(
    metadata: &HashMap<String, serde_json::Value>,
) -> Option<&serde_json::Value> {
    metadata
        .get("_siumai")
        .and_then(|value| value.get("generatedVideos"))
}

pub(super) fn public_task_metadata(
    metadata: &HashMap<String, serde_json::Value>,
) -> HashMap<String, serde_json::Value> {
    metadata
        .iter()
        .filter(|(key, _)| key.as_str() != "_siumai")
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

fn metadata_from_value(value: &serde_json::Value) -> GeneratedVideoMetadata {
    value
        .as_object()
        .map(|object| {
            object
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect()
        })
        .unwrap_or_default()
}

fn metadata_value(metadata: &GeneratedVideoMetadata) -> serde_json::Value {
    serde_json::Value::Object(metadata.clone().into_iter().collect())
}

pub(super) fn provider_metadata_video_value(video: &GeneratedVideo) -> serde_json::Value {
    let mut metadata = video.metadata.clone();

    match &video.data {
        GeneratedVideoData::Base64 { .. } => {
            metadata.remove("bytesBase64Encoded");
            metadata.remove("base64");
            if metadata.get("type").and_then(|value| value.as_str()) == Some("base64") {
                metadata.remove("data");
            }
        }
        GeneratedVideoData::Bytes { .. } => {
            metadata.remove("bytes");
            if metadata.get("type").and_then(|value| value.as_str()) == Some("bytes") {
                metadata.remove("data");
            }
        }
        GeneratedVideoData::Url { .. } | GeneratedVideoData::ProviderReference { .. } => {}
    }

    metadata_value(&metadata)
}

fn generated_video_from_metadata_item(
    task_id: &str,
    item: &serde_json::Value,
) -> Option<GeneratedVideo> {
    if let Some(url) = item.as_str() {
        let metadata = GeneratedVideoMetadata::from([("url".to_string(), serde_json::json!(url))]);
        return Some(GeneratedVideo {
            task_id: task_id.to_string(),
            media_type: guess_mime_from_path_or_url(url).unwrap_or_else(|| "video/mp4".to_string()),
            data: GeneratedVideoData::Url {
                url: url.to_string(),
            },
            metadata,
        });
    }

    let metadata = metadata_from_value(item);
    if let Some(data) = item
        .get("bytesBase64Encoded")
        .or_else(|| item.get("base64"))
        .or_else(|| {
            (item.get("type").and_then(|value| value.as_str()) == Some("base64"))
                .then(|| item.get("data"))
                .flatten()
        })
        .and_then(|value| value.as_str())
    {
        return Some(GeneratedVideo {
            task_id: task_id.to_string(),
            media_type: generated_video_media_type(&metadata, None),
            data: GeneratedVideoData::Base64 {
                data: data.to_string(),
            },
            metadata,
        });
    }

    if let Some(url) = item
        .get("url")
        .or_else(|| item.get("uri"))
        .or_else(|| item.get("gcsUri"))
        .and_then(|value| value.as_str())
    {
        return Some(GeneratedVideo {
            task_id: task_id.to_string(),
            media_type: generated_video_media_type(&metadata, Some(url)),
            data: GeneratedVideoData::Url {
                url: url.to_string(),
            },
            metadata,
        });
    }

    let bytes = item
        .get("bytes")
        .and_then(|value| value.as_array())
        .and_then(|values| {
            values
                .iter()
                .map(|value| value.as_u64().map(|value| value as u8))
                .collect::<Option<Vec<u8>>>()
        });
    if let Some(bytes) = bytes {
        return Some(GeneratedVideo {
            task_id: task_id.to_string(),
            media_type: generated_video_media_type(&metadata, None),
            data: GeneratedVideoData::Bytes { data: bytes },
            metadata,
        });
    }

    item.get("providerReference")
        .and_then(|value| serde_json::from_value::<ProviderReference>(value.clone()).ok())
        .map(|provider_reference| GeneratedVideo {
            task_id: task_id.to_string(),
            media_type: generated_video_media_type(&metadata, None),
            data: GeneratedVideoData::ProviderReference { provider_reference },
            metadata,
        })
}

fn generated_video_fallback(
    provider_id: &str,
    response: &VideoTaskStatusResponse,
) -> Option<GeneratedVideo> {
    let mut metadata = GeneratedVideoMetadata::new();
    let provider_reference = response.effective_provider_reference(provider_id);
    if let Some(file_id) = response.file_id.as_ref() {
        metadata.insert("fileId".to_string(), serde_json::json!(file_id));
    }
    if let Some(provider_reference) = provider_reference.as_ref() {
        metadata.insert(
            "providerReference".to_string(),
            serde_json::json!(provider_reference),
        );
    }
    if let Some(video_url) = response.video_url.as_ref() {
        metadata.insert("videoUrl".to_string(), serde_json::json!(video_url));
    }
    if let Some(duration) = response.duration {
        metadata.insert("duration".to_string(), serde_json::json!(duration));
    }
    if let Some(width) = response.video_width {
        metadata.insert("width".to_string(), serde_json::json!(width));
    }
    if let Some(height) = response.video_height {
        metadata.insert("height".to_string(), serde_json::json!(height));
    }

    if let Some(video_url) = response.video_url.as_ref() {
        return Some(GeneratedVideo {
            task_id: response.task_id.clone(),
            media_type: generated_video_media_type(&metadata, Some(video_url)),
            data: GeneratedVideoData::Url {
                url: video_url.clone(),
            },
            metadata,
        });
    }

    provider_reference.map(|provider_reference| GeneratedVideo {
        task_id: response.task_id.clone(),
        media_type: generated_video_media_type(
            &metadata,
            response
                .file_id
                .as_deref()
                .or_else(|| provider_reference.preferred_value(&[provider_id])),
        ),
        data: GeneratedVideoData::ProviderReference { provider_reference },
        metadata,
    })
}

pub(super) fn extract_generated_videos(
    provider_id: &str,
    response: &VideoTaskStatusResponse,
) -> Result<Vec<GeneratedVideo>, LlmError> {
    let metadata_videos = internal_generated_video_items(&response.metadata)
        .or_else(|| provider_metadata_root_value(&response.metadata, provider_id, "videos"))
        .or_else(|| response.metadata.get("videos"))
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| generated_video_from_metadata_item(&response.task_id, item))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    if !metadata_videos.is_empty() {
        return Ok(metadata_videos);
    }

    Ok(generated_video_fallback(provider_id, response)
        .map(|video| vec![video])
        .unwrap_or_default())
}

pub(super) async fn materialize_url_backed_generated_videos(
    videos: Vec<GeneratedVideo>,
    http_config: Option<&HttpConfig>,
) -> Result<(Vec<GeneratedVideo>, Vec<Warning>), LlmError> {
    let mut materialized_videos = Vec::with_capacity(videos.len());
    let mut warnings = Vec::new();

    for mut video in videos {
        if let GeneratedVideoData::Url { url } = &video.data {
            if !can_materialize_generated_video_url(url) {
                warnings.push(Warning::unsupported(
                    "generatedVideoUrlMaterialization",
                    Some(format!(
                        "Skipping automatic generated-video URL materialization for scheme '{}'. Only data:, http:, and https: URLs are supported on this helper path.",
                        generated_video_url_scheme(url)
                    )),
                ));
                materialized_videos.push(video);
                continue;
            }

            let (bytes, downloaded_media_type) =
                download_generated_video_url(url, http_config).await?;
            if video_url_from_metadata(&video.metadata).is_none() {
                video
                    .metadata
                    .insert("url".to_string(), serde_json::json!(url));
            }
            video.media_type =
                resolve_materialized_video_media_type(&video, Some(&bytes), downloaded_media_type);
            video.data = GeneratedVideoData::Bytes { data: bytes };
        }

        materialized_videos.push(video);
    }

    Ok((materialized_videos, warnings))
}

pub(super) async fn materialize_provider_reference_backed_generated_videos<
    M: VideoModelV4 + ?Sized,
>(
    model: &M,
    videos: Vec<GeneratedVideo>,
) -> Result<Vec<GeneratedVideo>, LlmError> {
    let mut materialized_videos = Vec::with_capacity(videos.len());

    for mut video in videos {
        let Some(provider_reference) = (match &video.data {
            GeneratedVideoData::ProviderReference { provider_reference } => {
                Some(provider_reference.clone())
            }
            _ => None,
        }) else {
            materialized_videos.push(video);
            continue;
        };

        match model.materialize_video_reference(&provider_reference).await {
            Ok(MaterializedVideoAsset { bytes, media_type }) => {
                if provider_reference_from_metadata(&video.metadata).is_none() {
                    video.metadata.insert(
                        "providerReference".to_string(),
                        serde_json::json!(provider_reference),
                    );
                }
                video.media_type =
                    resolve_materialized_video_media_type(&video, Some(&bytes), media_type);
                video.data = GeneratedVideoData::Bytes { data: bytes };
            }
            Err(LlmError::UnsupportedOperation(_)) => {}
            Err(error) => return Err(error),
        }

        materialized_videos.push(video);
    }

    Ok(materialized_videos)
}
