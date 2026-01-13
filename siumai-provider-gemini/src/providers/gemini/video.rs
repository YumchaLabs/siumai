//! Gemini Video Generation (Veo) helpers.
//!
//! Official docs:
//! - https://ai.google.dev/gemini-api/docs/video
//!
//! The REST API uses `models/{model}:predictLongRunning` and returns a long-running
//! operation resource name (e.g. `operations/...`) that must be polled via `GET /{name}`.

use crate::error::LlmError;
use crate::execution::executors::common::{HttpBody, execute_get_request, execute_json_request};
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::wiring::HttpExecutionWiring;
use crate::retry_api::RetryOptions;
use crate::types::HttpConfig;
use crate::types::video::{
    BaseResponse, VideoGenerationRequest, VideoGenerationResponse, VideoTaskStatus,
    VideoTaskStatusResponse,
};
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize)]
struct VeoImage {
    #[serde(rename = "imageBytes")]
    image_bytes: String,
    #[serde(rename = "mimeType")]
    mime_type: String,
}

#[derive(Debug, Clone, Serialize)]
struct VeoVideo {
    #[serde(rename = "videoBytes")]
    video_bytes: String,
    #[serde(rename = "mimeType")]
    mime_type: String,
}

fn normalize_gemini_model_id(model: &str) -> String {
    let trimmed = model.trim().trim_matches('/');
    if trimmed.is_empty() {
        return String::new();
    }

    if let Some(pos) = trimmed.rfind("/models/") {
        return trimmed[(pos + "/models/".len())..].to_string();
    }
    if let Some(rest) = trimmed.strip_prefix("models/") {
        return rest.to_string();
    }

    trimmed.to_string()
}

fn take_string_opt(map: &mut HashMap<String, serde_json::Value>, key: &str) -> Option<String> {
    map.remove(key)
        .and_then(|v| v.as_str().map(|s| s.to_string()))
}

fn take_u64_opt(map: &mut HashMap<String, serde_json::Value>, key: &str) -> Option<u64> {
    map.remove(key).and_then(|v| v.as_u64())
}

fn build_veo_image_from_bytes(bytes: Vec<u8>) -> VeoImage {
    let mime_type = crate::utils::mime::guess_mime_from_bytes(&bytes)
        .unwrap_or_else(|| "image/png".to_string());
    VeoImage {
        image_bytes: base64::engine::general_purpose::STANDARD.encode(bytes),
        mime_type,
    }
}

fn build_veo_video_from_bytes(bytes: Vec<u8>) -> VeoVideo {
    let mime_type = crate::utils::mime::guess_mime_from_bytes(&bytes)
        .unwrap_or_else(|| "video/mp4".to_string());
    VeoVideo {
        video_bytes: base64::engine::general_purpose::STANDARD.encode(bytes),
        mime_type,
    }
}

#[derive(Debug, Clone, Deserialize)]
struct OperationError {
    code: Option<i32>,
    message: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct LongRunningOperation {
    name: String,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    response: Option<serde_json::Value>,
    #[serde(default)]
    error: Option<OperationError>,
}

fn extract_video_uri(operation_response: &serde_json::Value) -> Option<String> {
    let paths = [
        "/generateVideoResponse/generatedSamples/0/video/uri",
        "/generateVideoResponse/generatedSamples/0/videoUri",
        "/generateVideoResponse/generatedSamples/0/video/uri",
    ];
    for path in paths {
        if let Some(uri) = operation_response.pointer(path).and_then(|v| v.as_str()) {
            return Some(uri.to_string());
        }
    }
    None
}

fn build_wiring(
    ctx: crate::core::ProviderContext,
    http_client: &reqwest::Client,
    retry_options: Option<&RetryOptions>,
    interceptors: &[Arc<dyn HttpInterceptor>],
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
) -> HttpExecutionWiring {
    let mut wiring = HttpExecutionWiring::new("gemini", http_client.clone(), ctx)
        .with_interceptors(interceptors.to_vec())
        .with_retry_options(retry_options.cloned());

    if let Some(transport) = http_transport {
        wiring = wiring.with_transport(transport);
    }

    wiring
}

fn map_operation_error_to_base(error: &OperationError) -> BaseResponse {
    BaseResponse {
        status_code: error.code.unwrap_or(-1),
        status_msg: error
            .message
            .clone()
            .unwrap_or_else(|| "Video generation failed".to_string()),
    }
}

fn build_predict_long_running_url(base_url: &str, model: &str) -> String {
    let base = base_url.trim_end_matches('/');
    let model = normalize_gemini_model_id(model);
    crate::utils::url::join_url(base, &format!("models/{model}:predictLongRunning"))
}

fn build_operation_get_url(base_url: &str, op_name: &str) -> String {
    let trimmed = op_name.trim().trim_start_matches('/');
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        return trimmed.to_string();
    }
    crate::utils::url::join_url(base_url.trim_end_matches('/'), trimmed)
}

fn build_video_request_body(
    request: VideoGenerationRequest,
) -> Result<serde_json::Value, LlmError> {
    let mut extra_params: HashMap<String, serde_json::Value> =
        request.extra_params.unwrap_or_default();

    let negative_prompt = take_string_opt(&mut extra_params, "negativePrompt")
        .or_else(|| take_string_opt(&mut extra_params, "negative_prompt"));
    let aspect_ratio = take_string_opt(&mut extra_params, "aspectRatio")
        .or_else(|| take_string_opt(&mut extra_params, "aspect_ratio"));
    let person_generation = take_string_opt(&mut extra_params, "personGeneration")
        .or_else(|| take_string_opt(&mut extra_params, "person_generation"));
    let seed = take_u64_opt(&mut extra_params, "seed");

    let mut instance = serde_json::Map::new();
    instance.insert("prompt".to_string(), serde_json::json!(request.prompt));

    // Prefer explicit instance objects if caller supplies them.
    if let Some(v) = extra_params.remove("image") {
        instance.insert("image".to_string(), v);
    }
    if let Some(v) = extra_params.remove("video") {
        instance.insert("video".to_string(), v);
    }
    if let Some(v) = extra_params
        .remove("lastFrame")
        .or_else(|| extra_params.remove("last_frame"))
    {
        instance.insert("lastFrame".to_string(), v);
    }
    if let Some(v) = extra_params.remove("referenceImages")
        && v.is_array()
    {
        instance.insert("referenceImages".to_string(), v);
    }

    if !instance.contains_key("image")
        && let Some(bytes) = request.seed_image
    {
        let image = build_veo_image_from_bytes(bytes);
        instance.insert(
            "image".to_string(),
            serde_json::to_value(image)
                .map_err(|e| LlmError::ParseError(format!("Failed to serialize Veo image: {e}")))?,
        );
    }

    if !instance.contains_key("video")
        && let Some(bytes) = request.seed_video
    {
        let video = build_veo_video_from_bytes(bytes);
        instance.insert(
            "video".to_string(),
            serde_json::to_value(video)
                .map_err(|e| LlmError::ParseError(format!("Failed to serialize Veo video: {e}")))?,
        );
    }

    let mut parameters = serde_json::Map::new();
    if let Some(negative_prompt) = negative_prompt {
        parameters.insert(
            "negativePrompt".to_string(),
            serde_json::json!(negative_prompt),
        );
    }
    if let Some(aspect_ratio) = aspect_ratio {
        parameters.insert("aspectRatio".to_string(), serde_json::json!(aspect_ratio));
    }
    if let Some(duration) = request.duration {
        parameters.insert("durationSeconds".to_string(), serde_json::json!(duration));
    }
    if let Some(resolution) = request.resolution {
        parameters.insert("resolution".to_string(), serde_json::json!(resolution));
    }
    if let Some(person_generation) = person_generation {
        parameters.insert(
            "personGeneration".to_string(),
            serde_json::json!(person_generation),
        );
    }
    if let Some(seed) = seed {
        parameters.insert("seed".to_string(), serde_json::json!(seed));
    }

    // Preserve any remaining provider-specific parameters.
    for (k, v) in extra_params {
        if parameters.contains_key(&k) {
            continue;
        }
        parameters.insert(k, v);
    }

    let mut root = serde_json::Map::new();
    root.insert("instances".to_string(), serde_json::json!([instance]));
    if !parameters.is_empty() {
        root.insert(
            "parameters".to_string(),
            serde_json::Value::Object(parameters),
        );
    }
    Ok(serde_json::Value::Object(root))
}

/// Provider-specific helper for Veo video generation.
#[derive(Clone)]
pub struct GeminiVideo {
    config: super::types::GeminiConfig,
    http_client: reqwest::Client,
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    retry_options: Option<RetryOptions>,
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    http_config: HttpConfig,
}

impl GeminiVideo {
    pub fn new(
        config: super::types::GeminiConfig,
        http_client: reqwest::Client,
        http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
        retry_options: Option<RetryOptions>,
        http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
        http_config: HttpConfig,
    ) -> Self {
        Self {
            config,
            http_client,
            http_interceptors,
            retry_options,
            http_transport,
            http_config,
        }
    }

    pub async fn create_video_task(
        &self,
        request: VideoGenerationRequest,
    ) -> Result<VideoGenerationResponse, LlmError> {
        let ctx = super::context::build_context(&self.config).await;
        let wiring = build_wiring(
            ctx,
            &self.http_client,
            self.retry_options.as_ref(),
            &self.http_interceptors,
            self.http_transport.clone(),
        );
        let cfg = wiring.config(Arc::new(super::spec::GeminiSpec));
        let url = build_predict_long_running_url(&cfg.provider_context.base_url, &request.model);
        let body = build_video_request_body(request)?;
        let res = execute_json_request(&cfg, &url, HttpBody::Json(body), None, false).await?;
        let op: LongRunningOperation = serde_json::from_value(res.json).map_err(|e| {
            LlmError::ParseError(format!(
                "Failed to parse Gemini video operation response: {e}"
            ))
        })?;

        Ok(VideoGenerationResponse {
            task_id: op.name,
            base_resp: Some(BaseResponse {
                status_code: 0,
                status_msg: "OK".to_string(),
            }),
        })
    }

    pub async fn query_video_task(
        &self,
        task_id: &str,
    ) -> Result<VideoTaskStatusResponse, LlmError> {
        let ctx = super::context::build_context(&self.config).await;
        let wiring = build_wiring(
            ctx,
            &self.http_client,
            self.retry_options.as_ref(),
            &self.http_interceptors,
            self.http_transport.clone(),
        );
        let cfg = wiring.config(Arc::new(super::spec::GeminiSpec));
        let url = build_operation_get_url(&cfg.provider_context.base_url, task_id);
        let res = execute_get_request(&cfg, &url, None).await?;
        let op: LongRunningOperation = serde_json::from_value(res.json).map_err(|e| {
            LlmError::ParseError(format!("Failed to parse Gemini operation status: {e}"))
        })?;

        if !op.done {
            return Ok(VideoTaskStatusResponse {
                task_id: op.name,
                status: VideoTaskStatus::Processing,
                file_id: None,
                video_width: None,
                video_height: None,
                base_resp: None,
            });
        }

        if let Some(err) = op.error {
            return Ok(VideoTaskStatusResponse {
                task_id: op.name,
                status: VideoTaskStatus::Fail,
                file_id: None,
                video_width: None,
                video_height: None,
                base_resp: Some(map_operation_error_to_base(&err)),
            });
        }

        let file_id = op
            .response
            .as_ref()
            .and_then(extract_video_uri)
            .or_else(|| {
                op.response
                    .as_ref()
                    .and_then(|r| r.pointer("/uri"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            });

        Ok(VideoTaskStatusResponse {
            task_id: op.name,
            status: VideoTaskStatus::Success,
            file_id,
            video_width: None,
            video_height: None,
            base_resp: Some(BaseResponse {
                status_code: 0,
                status_msg: "OK".to_string(),
            }),
        })
    }
}

pub fn get_supported_veo_models() -> Vec<String> {
    vec![
        "veo-3.1-generate-preview".to_string(),
        "veo-3.1-fast-generate-preview".to_string(),
        "veo-3.0-generate-001".to_string(),
        "veo-3.0-fast-generate-001".to_string(),
        "veo-2.0-generate-001".to_string(),
    ]
}

pub fn get_supported_veo_durations(model: &str) -> Vec<u32> {
    let m = normalize_gemini_model_id(model);
    if m.starts_with("veo-2.") {
        vec![5, 6, 8]
    } else {
        vec![4, 6, 8]
    }
}

pub fn get_supported_veo_resolutions(model: &str) -> Vec<String> {
    let m = normalize_gemini_model_id(model);
    if m.starts_with("veo-2.") {
        Vec::new()
    } else {
        vec!["720p".to_string(), "1080p".to_string()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_model_accepts_resource_style() {
        assert_eq!(
            normalize_gemini_model_id("models/veo-3.1-generate-preview"),
            "veo-3.1-generate-preview"
        );
        assert_eq!(
            normalize_gemini_model_id("publishers/google/models/veo-2.0-generate-001"),
            "veo-2.0-generate-001"
        );
    }

    #[test]
    fn build_request_body_includes_prompt_and_parameters() {
        let req = VideoGenerationRequest::new("veo-3.1-generate-preview", "hi")
            .with_duration(6)
            .with_resolution("720p")
            .with_extra_param("negativePrompt", serde_json::json!("no cats"));

        let body = build_video_request_body(req).unwrap();
        assert_eq!(body["instances"][0]["prompt"], serde_json::json!("hi"));
        assert_eq!(body["parameters"]["durationSeconds"], serde_json::json!(6));
        assert_eq!(body["parameters"]["resolution"], serde_json::json!("720p"));
        assert_eq!(
            body["parameters"]["negativePrompt"],
            serde_json::json!("no cats")
        );
    }

    #[test]
    fn extract_video_uri_from_operation_response() {
        let v = serde_json::json!({
            "generateVideoResponse": {
                "generatedSamples": [
                    { "video": { "uri": "https://example/video.mp4" } }
                ]
            }
        });
        assert_eq!(
            extract_video_uri(&v),
            Some("https://example/video.mp4".to_string())
        );
    }
}
