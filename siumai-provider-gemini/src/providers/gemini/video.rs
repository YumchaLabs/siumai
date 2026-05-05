//! Gemini Video Generation (Veo) helpers.
//!
//! Official docs:
//! - <https://ai.google.dev/gemini-api/docs/video>
//!
//! The REST API uses `models/{model}:predictLongRunning` and returns a long-running
//! operation resource name (e.g. `operations/...`) that must be polled via `GET /{name}`.

use crate::error::LlmError;
use crate::execution::executors::common::{HttpBody, execute_get_request, execute_json_request};
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::wiring::HttpExecutionWiring;
use crate::retry_api::RetryOptions;
use crate::types::video::{
    BaseResponse, VideoGenerationInput, VideoGenerationRequest, VideoGenerationResponse,
    VideoTaskStatus, VideoTaskStatusResponse,
};
use crate::types::{ProviderReference, Warning};
use serde::{Deserialize, Serialize};
use siumai_core::video::VideoPollingOptions;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

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

fn video_provider_options(
    request: &VideoGenerationRequest,
) -> Option<&serde_json::Map<String, serde_json::Value>> {
    request
        .provider_options_map
        .get_object("google")
        .or_else(|| request.provider_options_map.get_object("gemini"))
}

pub(super) fn polling_options(
    request: &VideoGenerationRequest,
) -> Result<VideoPollingOptions, LlmError> {
    let mut extra_params: HashMap<String, serde_json::Value> = video_provider_options(request)
        .cloned()
        .map(|map| map.into_iter().collect())
        .unwrap_or_default();
    extra_params.extend(request.extra_params.clone().unwrap_or_default());

    let poll_interval_ms = take_u64_opt(&mut extra_params, "pollIntervalMs")
        .or_else(|| take_u64_opt(&mut extra_params, "poll_interval_ms"));
    let poll_timeout_ms = take_u64_opt(&mut extra_params, "pollTimeoutMs")
        .or_else(|| take_u64_opt(&mut extra_params, "poll_timeout_ms"));

    Ok(VideoPollingOptions {
        poll_interval: poll_interval_ms.map(Duration::from_millis),
        poll_timeout: poll_timeout_ms.map(Duration::from_millis),
    })
}

fn build_veo_image_from_input(input: &VideoGenerationInput) -> Result<Option<VeoImage>, Warning> {
    match input {
        VideoGenerationInput::Url { .. } => Err(Warning::unsupported(
            "URL-based image input",
            Some(
                "Gemini video models currently require file-backed image inputs on the provider-owned path.",
            ),
        )),
        VideoGenerationInput::File {
            data, media_type, ..
        } => {
            let mime_type = if let Some(media_type) = media_type {
                media_type.clone()
            } else {
                let bytes = data.as_bytes().map_err(|err| {
                    Warning::compatibility(
                        "image",
                        Some(format!(
                            "Failed to decode base64 image input for Gemini video request: {err}"
                        )),
                    )
                })?;
                crate::utils::mime::guess_mime_from_bytes(&bytes)
                    .unwrap_or_else(|| "image/png".to_string())
            };
            Ok(Some(VeoImage {
                image_bytes: data.as_base64(),
                mime_type,
            }))
        }
    }
}

fn build_veo_video_from_input(input: &VideoGenerationInput) -> Result<Option<VeoVideo>, Warning> {
    match input {
        VideoGenerationInput::Url { .. } => Err(Warning::unsupported(
            "URL-based video input",
            Some(
                "Gemini video models currently require file-backed video inputs on the provider-owned path.",
            ),
        )),
        VideoGenerationInput::File {
            data, media_type, ..
        } => {
            let mime_type = if let Some(media_type) = media_type {
                media_type.clone()
            } else {
                let bytes = data.as_bytes().map_err(|err| {
                    Warning::compatibility(
                        "video",
                        Some(format!(
                            "Failed to decode base64 video input for Gemini video request: {err}"
                        )),
                    )
                })?;
                crate::utils::mime::guess_mime_from_bytes(&bytes)
                    .unwrap_or_else(|| "video/mp4".to_string())
            };
            Ok(Some(VeoVideo {
                video_bytes: data.as_base64(),
                mime_type,
            }))
        }
    }
}

fn map_google_resolution(value: &str) -> String {
    match value.trim() {
        "1280x720" => "720p".to_string(),
        "1920x1080" => "1080p".to_string(),
        "3840x2160" => "4k".to_string(),
        other => other.to_string(),
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

fn has_auth_header(headers: &HashMap<String, String>) -> bool {
    headers
        .keys()
        .any(|key| key.eq_ignore_ascii_case("authorization"))
}

fn append_api_key_query(url: String, api_key: &str) -> String {
    let key = urlencoding::encode(api_key);
    if url.contains('?') {
        format!("{url}&key={key}")
    } else {
        format!("{url}?key={key}")
    }
}

fn normalize_download_uri(uri: &str, provider_context: &crate::core::ProviderContext) -> String {
    if !uri.starts_with("http://") && !uri.starts_with("https://") {
        return uri.to_string();
    }

    if let Some(api_key) = provider_context.api_key.as_deref()
        && !api_key.is_empty()
        && !has_auth_header(&provider_context.http_extra_headers)
    {
        append_api_key_query(uri.to_string(), api_key)
    } else {
        uri.to_string()
    }
}

fn extract_generated_video_metadata(
    operation_response: &serde_json::Value,
    provider_context: &crate::core::ProviderContext,
) -> Vec<serde_json::Value> {
    operation_response
        .pointer("/generateVideoResponse/generatedSamples")
        .and_then(|value| value.as_array())
        .map(|samples| {
            samples
                .iter()
                .filter_map(|sample| {
                    let uri = sample
                        .pointer("/video/uri")
                        .or_else(|| sample.get("videoUri"))
                        .and_then(|value| value.as_str())?;
                    let mime_type = sample
                        .pointer("/video/mimeType")
                        .or_else(|| sample.get("mimeType"))
                        .and_then(|value| value.as_str())
                        .unwrap_or("video/mp4");

                    Some(serde_json::json!({
                        "uri": normalize_download_uri(uri, provider_context),
                        "mediaType": mime_type,
                    }))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn build_query_metadata(
    operation_response: &serde_json::Value,
    provider_context: &crate::core::ProviderContext,
) -> HashMap<String, serde_json::Value> {
    let videos = extract_generated_video_metadata(operation_response, provider_context);
    if videos.is_empty() {
        HashMap::new()
    } else {
        let google_metadata = serde_json::json!({
            "videos": videos,
        });
        HashMap::from([
            ("google".to_string(), google_metadata.clone()),
            ("gemini".to_string(), google_metadata),
        ])
    }
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
) -> Result<(serde_json::Value, Vec<Warning>), LlmError> {
    let mut extra_params: HashMap<String, serde_json::Value> = video_provider_options(&request)
        .cloned()
        .map(|map| map.into_iter().collect())
        .unwrap_or_default();
    extra_params.extend(request.extra_params.clone().unwrap_or_default());
    let mut warnings = Vec::new();

    let _ = take_u64_opt(&mut extra_params, "pollIntervalMs")
        .or_else(|| take_u64_opt(&mut extra_params, "poll_interval_ms"));
    let _ = take_u64_opt(&mut extra_params, "pollTimeoutMs")
        .or_else(|| take_u64_opt(&mut extra_params, "poll_timeout_ms"));

    let negative_prompt = take_string_opt(&mut extra_params, "negativePrompt")
        .or_else(|| take_string_opt(&mut extra_params, "negative_prompt"));
    let aspect_ratio = request.aspect_ratio.clone().or_else(|| {
        take_string_opt(&mut extra_params, "aspectRatio")
            .or_else(|| take_string_opt(&mut extra_params, "aspect_ratio"))
    });
    let person_generation = take_string_opt(&mut extra_params, "personGeneration")
        .or_else(|| take_string_opt(&mut extra_params, "person_generation"));
    let seed = request
        .seed
        .or_else(|| take_u64_opt(&mut extra_params, "seed"));

    let mut instance = serde_json::Map::new();
    if let Some(prompt) = request.prompt {
        instance.insert("prompt".to_string(), serde_json::json!(prompt));
    }

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
        && let Some(image_input) = request.image.as_ref()
    {
        match build_veo_image_from_input(image_input) {
            Ok(Some(image)) => {
                instance.insert(
                    "image".to_string(),
                    serde_json::to_value(image).map_err(|e| {
                        LlmError::ParseError(format!("Failed to serialize Veo image: {e}"))
                    })?,
                );
            }
            Ok(None) => {}
            Err(warning) => warnings.push(warning),
        }
    }

    if !instance.contains_key("video")
        && let Some(video_input) = request.video.as_ref()
    {
        match build_veo_video_from_input(video_input) {
            Ok(Some(video)) => {
                instance.insert(
                    "video".to_string(),
                    serde_json::to_value(video).map_err(|e| {
                        LlmError::ParseError(format!("Failed to serialize Veo video: {e}"))
                    })?,
                );
            }
            Ok(None) => {}
            Err(warning) => warnings.push(warning),
        }
    }

    let mut parameters = serde_json::Map::new();
    if let Some(count) = request.count {
        parameters.insert("sampleCount".to_string(), serde_json::json!(count));
    }
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
        parameters.insert(
            "resolution".to_string(),
            serde_json::json!(map_google_resolution(&resolution)),
        );
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
    if request.fps.is_some() {
        warnings.push(Warning::unsupported(
            "fps",
            Some("Gemini video models do not expose custom FPS on the provider-owned path."),
        ));
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
    Ok((serde_json::Value::Object(root), warnings))
}

/// Provider-specific helper for Veo video generation.
#[derive(Clone)]
pub struct GeminiVideo {
    config: super::types::GeminiConfig,
    http_client: reqwest::Client,
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
    retry_options: Option<RetryOptions>,
    http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
}

impl GeminiVideo {
    pub fn new(
        config: super::types::GeminiConfig,
        http_client: reqwest::Client,
        http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
        retry_options: Option<RetryOptions>,
        http_transport: Option<Arc<dyn crate::execution::http::transport::HttpTransport>>,
    ) -> Self {
        let effective_http_transport = http_transport.or_else(|| config.http_transport.clone());
        Self {
            config,
            http_client,
            http_interceptors,
            retry_options,
            http_transport: effective_http_transport,
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
        let (body, warnings) = build_video_request_body(request)?;
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
            metadata: HashMap::new(),
            warnings: (!warnings.is_empty()).then_some(warnings),
            response: None,
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
                video_url: None,
                provider_reference: None,
                duration: None,
                video_width: None,
                video_height: None,
                base_resp: None,
                metadata: HashMap::new(),
                response: None,
            });
        }

        if let Some(err) = op.error {
            return Ok(VideoTaskStatusResponse {
                task_id: op.name,
                status: VideoTaskStatus::Fail,
                file_id: None,
                video_url: None,
                provider_reference: None,
                duration: None,
                video_width: None,
                video_height: None,
                base_resp: Some(map_operation_error_to_base(&err)),
                metadata: HashMap::new(),
                response: None,
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
        let metadata = op
            .response
            .as_ref()
            .map(|response| build_query_metadata(response, &cfg.provider_context))
            .unwrap_or_default();

        Ok(VideoTaskStatusResponse {
            task_id: op.name,
            status: VideoTaskStatus::Success,
            file_id: file_id.clone(),
            video_url: None,
            provider_reference: file_id.as_ref().map(|file_id| {
                ProviderReference::from([
                    ("gemini", file_id.as_str()),
                    ("google", file_id.as_str()),
                ])
            }),
            duration: None,
            video_width: None,
            video_height: None,
            base_resp: Some(BaseResponse {
                status_code: 0,
                status_msg: "OK".to_string(),
            }),
            metadata,
            response: None,
        })
    }
}

pub fn get_supported_veo_models() -> Vec<String> {
    vec![
        "veo-3.1-generate".to_string(),
        "veo-3.1-generate-preview".to_string(),
        "veo-3.1-fast-generate-preview".to_string(),
        "veo-3.1-lite-generate-preview".to_string(),
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
    use crate::execution::http::transport::{
        HttpTransport, HttpTransportGetRequest, HttpTransportRequest, HttpTransportResponse,
        HttpTransportStreamBody, HttpTransportStreamResponse,
    };
    use async_trait::async_trait;
    use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct JsonCaptureTransport {
        response_body: Arc<Vec<u8>>,
        last: Arc<Mutex<Option<HttpTransportRequest>>>,
        last_get: Arc<Mutex<Option<HttpTransportGetRequest>>>,
    }

    impl JsonCaptureTransport {
        fn new(response: serde_json::Value) -> Self {
            Self {
                response_body: Arc::new(serde_json::to_vec(&response).expect("response json")),
                last: Arc::new(Mutex::new(None)),
                last_get: Arc::new(Mutex::new(None)),
            }
        }

        fn take_get(&self) -> Option<HttpTransportGetRequest> {
            self.last_get.lock().expect("lock get request").take()
        }
    }

    #[async_trait]
    impl HttpTransport for JsonCaptureTransport {
        async fn execute_json(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            *self.last.lock().expect("lock request") = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportResponse {
                status: 200,
                headers,
                body: self.response_body.as_ref().clone(),
            })
        }

        async fn execute_get(
            &self,
            request: HttpTransportGetRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            *self.last_get.lock().expect("lock get request") = Some(request);

            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

            Ok(HttpTransportResponse {
                status: 200,
                headers,
                body: self.response_body.as_ref().clone(),
            })
        }

        async fn execute_stream(
            &self,
            _request: HttpTransportRequest,
        ) -> Result<HttpTransportStreamResponse, LlmError> {
            let mut headers = HeaderMap::new();
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));

            Ok(HttpTransportStreamResponse {
                status: 200,
                headers,
                body: HttpTransportStreamBody::from_bytes(b"data: [DONE]\n\n".to_vec()),
            })
        }
    }

    fn warning_feature(warning: Warning) -> String {
        match warning {
            Warning::Unsupported { feature, .. } => feature,
            Warning::UnsupportedSetting { setting, .. } => setting,
            Warning::UnsupportedTool { tool_name, .. } => tool_name,
            Warning::Compatibility { feature, .. } => feature,
            Warning::Deprecated { setting, .. } => setting,
            Warning::Other { message } => message,
        }
    }

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
    fn supported_veo_models_track_current_google_ids() {
        let models = get_supported_veo_models();
        assert!(models.iter().any(|model| model == "veo-3.1-generate"));
        assert!(
            models
                .iter()
                .any(|model| model == "veo-3.1-generate-preview")
        );
        assert!(
            models
                .iter()
                .any(|model| model == "veo-3.1-fast-generate-preview")
        );
        assert!(
            models
                .iter()
                .any(|model| model == "veo-3.1-lite-generate-preview")
        );
    }

    #[test]
    fn build_request_body_includes_prompt_and_parameters() {
        let req = VideoGenerationRequest::new("veo-3.1-generate-preview", "hi")
            .with_count(2)
            .with_aspect_ratio("16:9")
            .with_duration(6)
            .with_resolution("1280x720")
            .with_seed(9)
            .with_image(VideoGenerationInput::file_with_media_type(
                vec![1, 2, 3],
                "image/png",
            ))
            .with_extra_param("negativePrompt", serde_json::json!("no cats"));

        let (body, warnings) = build_video_request_body(req).unwrap();
        assert!(warnings.is_empty());
        assert_eq!(body["instances"][0]["prompt"], serde_json::json!("hi"));
        assert_eq!(
            body["instances"][0]["image"]["imageBytes"],
            serde_json::json!("AQID")
        );
        assert_eq!(
            body["instances"][0]["image"]["mimeType"],
            serde_json::json!("image/png")
        );
        assert_eq!(body["parameters"]["sampleCount"], serde_json::json!(2));
        assert_eq!(body["parameters"]["aspectRatio"], serde_json::json!("16:9"));
        assert_eq!(body["parameters"]["durationSeconds"], serde_json::json!(6));
        assert_eq!(body["parameters"]["resolution"], serde_json::json!("720p"));
        assert_eq!(body["parameters"]["seed"], serde_json::json!(9));
        assert_eq!(
            body["parameters"]["negativePrompt"],
            serde_json::json!("no cats")
        );
    }

    #[test]
    fn build_request_body_surfaces_url_and_fps_warnings() {
        let req = VideoGenerationRequest::new("veo-3.1-generate-preview", "hi")
            .with_fps(24)
            .with_image(VideoGenerationInput::url("https://example.com/start.png"));

        let (body, warnings) = build_video_request_body(req).unwrap();
        assert!(body["instances"][0].get("image").is_none());

        let warning_features = warnings
            .into_iter()
            .map(warning_feature)
            .collect::<Vec<_>>();
        assert_eq!(warning_features, vec!["URL-based image input", "fps"]);
    }

    #[test]
    fn build_request_body_reads_google_provider_options_and_strips_polling_knobs() {
        let req = VideoGenerationRequest::new("veo-3.1-generate-preview", "hi")
            .with_provider_option(
                "google",
                serde_json::json!({
                    "negativePrompt": "no cats",
                    "personGeneration": "allow_all",
                    "pollIntervalMs": 500,
                    "pollTimeoutMs": 30000,
                    "referenceImages": [
                        {
                            "bytesBase64Encoded": "Zm9v"
                        }
                    ]
                }),
            );

        let (body, warnings) = build_video_request_body(req).unwrap();
        assert_eq!(
            body["parameters"]["negativePrompt"],
            serde_json::json!("no cats")
        );
        assert_eq!(
            body["parameters"]["personGeneration"],
            serde_json::json!("allow_all")
        );
        assert_eq!(
            body["instances"][0]["referenceImages"][0]["bytesBase64Encoded"],
            serde_json::json!("Zm9v")
        );
        assert!(body["parameters"].get("pollIntervalMs").is_none());
        assert!(body["parameters"].get("pollTimeoutMs").is_none());
        assert!(warnings.is_empty());
    }

    #[test]
    fn polling_options_prefers_request_extra_params_over_google_provider_options() {
        let req = VideoGenerationRequest::new("veo-3.1-generate-preview", "hi")
            .with_provider_option(
                "google",
                serde_json::json!({
                    "pollIntervalMs": 500,
                    "pollTimeoutMs": 30000
                }),
            )
            .with_extra_param("pollIntervalMs", serde_json::json!(250))
            .with_extra_param("pollTimeoutMs", serde_json::json!(15_000));

        let options = polling_options(&req).expect("gemini polling options");

        assert_eq!(options.poll_interval, Some(Duration::from_millis(250)));
        assert_eq!(options.poll_timeout, Some(Duration::from_millis(15_000)));
    }

    #[test]
    fn build_request_body_allows_promptless_image_to_video_requests() {
        let req =
            VideoGenerationRequest::new_without_prompt("veo-3.1-generate-preview").with_image(
                VideoGenerationInput::file_with_media_type(vec![1, 2, 3], "image/png"),
            );

        let (body, warnings) = build_video_request_body(req).unwrap();
        assert!(warnings.is_empty());
        assert!(body["instances"][0].get("prompt").is_none());
        assert_eq!(
            body["instances"][0]["image"]["imageBytes"],
            serde_json::json!("AQID")
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

    #[test]
    fn build_query_metadata_keeps_generated_video_entries() {
        let response = serde_json::json!({
            "generateVideoResponse": {
                "generatedSamples": [
                    {
                        "video": {
                            "uri": "https://example/video-1.mp4",
                            "mimeType": "video/mp4"
                        }
                    },
                    {
                        "video": {
                            "uri": "https://example/video-2.webm",
                            "mimeType": "video/webm"
                        }
                    }
                ]
            }
        });
        let provider_context = crate::core::ProviderContext::new(
            "gemini",
            "https://generativelanguage.googleapis.com",
            Some("test-key".to_string()),
            HashMap::new(),
        );

        let metadata = build_query_metadata(&response, &provider_context);
        let videos = metadata
            .get("google")
            .and_then(|value| value.get("videos"))
            .and_then(|value| value.as_array())
            .expect("google videos metadata");

        assert_eq!(videos.len(), 2);
        assert_eq!(
            videos[0].get("uri").and_then(|value| value.as_str()),
            Some("https://example/video-1.mp4?key=test-key")
        );
        assert_eq!(
            videos[1].get("mediaType").and_then(|value| value.as_str()),
            Some("video/webm")
        );
        assert_eq!(metadata.get("gemini"), metadata.get("google"));
    }

    #[tokio::test]
    async fn query_video_task_returns_provider_reference_aliases_for_generated_file() {
        let transport = JsonCaptureTransport::new(serde_json::json!({
            "name": "operations/test-video-123",
            "done": true,
            "response": {
                "generateVideoResponse": {
                    "generatedSamples": [
                        {
                            "video": {
                                "uri": "https://example.com/generated/video.mp4",
                                "mimeType": "video/mp4"
                            }
                        }
                    ]
                }
            }
        }));
        let config = crate::providers::gemini::GeminiConfig::new("test-key")
            .with_base_url("https://example.com/v1beta".to_string())
            .with_model("veo-3.1-generate-preview".to_string())
            .with_http_transport(Arc::new(transport.clone()));
        let video = GeminiVideo::new(config, reqwest::Client::new(), Vec::new(), None, None);

        let response = video
            .query_video_task("operations/test-video-123")
            .await
            .expect("query video task");

        assert_eq!(response.status, VideoTaskStatus::Success);
        assert_eq!(
            response.file_id.as_deref(),
            Some("https://example.com/generated/video.mp4")
        );
        let provider_reference = response
            .provider_reference()
            .expect("provider reference on gemini video task");
        assert_eq!(
            provider_reference.get("gemini"),
            Some("https://example.com/generated/video.mp4")
        );
        assert_eq!(
            provider_reference.get("google"),
            Some("https://example.com/generated/video.mp4")
        );
        assert_eq!(response.video_url, None);
        assert_eq!(
            response
                .metadata
                .get("google")
                .and_then(|value| value.get("videos"))
                .and_then(|value| value.as_array())
                .map(Vec::len),
            Some(1)
        );
        assert_eq!(
            response.metadata.get("gemini"),
            response.metadata.get("google")
        );

        let req = transport.take_get().expect("captured get request");
        assert_eq!(
            req.url,
            "https://example.com/v1beta/operations/test-video-123"
        );
        assert_eq!(
            req.headers
                .get("x-goog-api-key")
                .and_then(|value| value.to_str().ok()),
            Some("test-key")
        );
    }
}
