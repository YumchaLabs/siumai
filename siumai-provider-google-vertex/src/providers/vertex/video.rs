use crate::core::{ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::executors::common::{HttpBody, execute_json_request};
use crate::execution::http::headers::HttpHeaderBuilder;
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::wiring::HttpExecutionWiring;
use crate::provider_options::vertex::{GoogleVertexReferenceImage, GoogleVertexVideoModelOptions};
use crate::retry_api::RetryOptions;
use crate::types::video::{
    VideoGenerationInput, VideoGenerationRequest, VideoGenerationResponse, VideoTaskStatus,
    VideoTaskStatusResponse,
};
use crate::types::{BaseResponse, HttpResponseInfo, Warning};
use reqwest::header::HeaderMap;
use serde::{Deserialize, Serialize};
use siumai_core::video::VideoPollingOptions;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use super::client::GoogleVertexConfig;

const VERTEX_USER_AGENT: &str = concat!("siumai/google-vertex/", env!("CARGO_PKG_VERSION"));
const VIDEO_INPUT_WARNING: &str = "Google Vertex video models on the provider-owned path support prompt and optional image-to-video input, but not direct video-to-video inputs.";
const URL_IMAGE_WARNING: &str = "Google Vertex video models require base64-backed image input or GCS reference images. URL-backed seed images are ignored on this task-based path.";

#[derive(Clone, Default)]
struct VertexVideoSpec;

impl ProviderSpec for VertexVideoSpec {
    fn id(&self) -> &'static str {
        "vertex"
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new().with_custom_feature("video", true)
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        build_vertex_headers(&ctx.http_extra_headers)
    }
}

#[derive(Debug, Clone, Deserialize)]
struct VertexOperation {
    name: String,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    error: Option<VertexOperationError>,
    #[serde(default)]
    response: Option<VertexOperationResponse>,
}

#[derive(Debug, Clone, Deserialize)]
struct VertexOperationError {
    #[serde(default)]
    code: Option<i32>,
    #[serde(default)]
    message: String,
    #[serde(default)]
    status: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct VertexOperationResponse {
    #[serde(default)]
    videos: Vec<VertexGeneratedVideo>,
    #[serde(rename = "raiMediaFilteredCount", default)]
    rai_media_filtered_count: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VertexGeneratedVideo {
    #[serde(rename = "bytesBase64Encoded", default)]
    bytes_base64_encoded: Option<String>,
    #[serde(rename = "gcsUri", default)]
    gcs_uri: Option<String>,
    #[serde(rename = "mimeType", default)]
    mime_type: Option<String>,
}

fn vertex_provider_metadata_video(video: &VertexGeneratedVideo) -> serde_json::Value {
    let mut value = serde_json::Map::new();

    if let Some(gcs_uri) = video.gcs_uri.as_ref() {
        value.insert("gcsUri".to_string(), serde_json::json!(gcs_uri));
    }
    if let Some(mime_type) = video.mime_type.as_ref() {
        value.insert("mimeType".to_string(), serde_json::json!(mime_type));
    }

    serde_json::Value::Object(value)
}

fn build_vertex_headers(custom_headers: &HashMap<String, String>) -> Result<HeaderMap, LlmError> {
    let builder = HttpHeaderBuilder::new()
        .with_json_content_type()
        .with_user_agent(VERTEX_USER_AGENT)?
        .with_custom_headers(custom_headers)?;
    Ok(builder.build())
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

fn headers_to_map(headers: &HeaderMap) -> HashMap<String, String> {
    headers
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.as_str().to_ascii_lowercase(), value.to_string()))
        })
        .collect()
}

fn normalize_vertex_model_id(model: &str) -> String {
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

fn map_vertex_resolution(value: &str) -> String {
    match value.trim() {
        "1280x720" => "720p".to_string(),
        "1920x1080" => "1080p".to_string(),
        "3840x2160" => "4k".to_string(),
        other => other.to_string(),
    }
}

fn build_predict_long_running_url(model: &str, ctx: &ProviderContext) -> String {
    let base = ctx.base_url.trim_end_matches('/');
    let model = normalize_vertex_model_id(model);
    let url = format!("{base}/models/{model}:predictLongRunning");
    if let Some(key) = ctx.api_key.as_deref()
        && !key.is_empty()
        && !has_auth_header(&ctx.http_extra_headers)
    {
        append_api_key_query(url, key)
    } else {
        url
    }
}

fn build_fetch_predict_operation_url(model: &str, ctx: &ProviderContext) -> String {
    let base = ctx.base_url.trim_end_matches('/');
    let model = normalize_vertex_model_id(model);
    let url = format!("{base}/models/{model}:fetchPredictOperation");
    if let Some(key) = ctx.api_key.as_deref()
        && !key.is_empty()
        && !has_auth_header(&ctx.http_extra_headers)
    {
        append_api_key_query(url, key)
    } else {
        url
    }
}

fn parse_vertex_video_options(
    map: &crate::types::ProviderOptionsMap,
) -> Result<GoogleVertexVideoModelOptions, LlmError> {
    let Some(value) = map.get("vertex") else {
        return Ok(GoogleVertexVideoModelOptions::default());
    };

    serde_json::from_value(value.clone()).map_err(|err| {
        LlmError::InvalidParameter(format!(
            "Invalid Google Vertex video options in providerOptions.vertex: {err}"
        ))
    })
}

fn take_string_opt(
    map: &mut HashMap<String, serde_json::Value>,
    camel: &str,
    snake: &str,
) -> Option<String> {
    map.remove(camel)
        .or_else(|| map.remove(snake))
        .and_then(|value| value.as_str().map(|value| value.to_string()))
}

fn take_bool_opt(
    map: &mut HashMap<String, serde_json::Value>,
    camel: &str,
    snake: &str,
) -> Option<bool> {
    map.remove(camel)
        .or_else(|| map.remove(snake))
        .and_then(|value| value.as_bool())
}

fn take_u64_opt(
    map: &mut HashMap<String, serde_json::Value>,
    camel: &str,
    snake: &str,
) -> Option<u64> {
    map.remove(camel)
        .or_else(|| map.remove(snake))
        .and_then(|value| value.as_u64())
}

fn take_reference_images_opt(
    map: &mut HashMap<String, serde_json::Value>,
    camel: &str,
    snake: &str,
) -> Result<Option<Vec<GoogleVertexReferenceImage>>, LlmError> {
    let Some(value) = map.remove(camel).or_else(|| map.remove(snake)) else {
        return Ok(None);
    };

    serde_json::from_value(value).map(Some).map_err(|err| {
        LlmError::InvalidParameter(format!(
            "Invalid Google Vertex video `{camel}` payload: {err}"
        ))
    })
}

pub(super) fn polling_options(
    request: &VideoGenerationRequest,
) -> Result<VideoPollingOptions, LlmError> {
    let mut options = parse_vertex_video_options(&request.provider_options_map)?;
    let mut extra_params = request.extra_params.clone().unwrap_or_default();

    if options.poll_interval_ms.is_none() {
        options.poll_interval_ms =
            take_u64_opt(&mut extra_params, "pollIntervalMs", "poll_interval_ms");
    }
    if options.poll_timeout_ms.is_none() {
        options.poll_timeout_ms =
            take_u64_opt(&mut extra_params, "pollTimeoutMs", "poll_timeout_ms");
    }

    Ok(VideoPollingOptions {
        poll_interval: options.poll_interval_ms.map(Duration::from_millis),
        poll_timeout: options.poll_timeout_ms.map(Duration::from_millis),
    })
}

fn image_input_to_vertex_payload(
    input: &VideoGenerationInput,
) -> Result<Option<serde_json::Value>, Warning> {
    match input {
        VideoGenerationInput::Url { .. } => Err(Warning::unsupported(
            "URL-based image input",
            Some(URL_IMAGE_WARNING),
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
                            "Failed to decode base64 image input for Google Vertex video request: {err}"
                        )),
                    )
                })?;
                crate::utils::mime::guess_mime_from_bytes(&bytes)
                    .unwrap_or_else(|| "image/png".to_string())
            };

            Ok(Some(serde_json::json!({
                "bytesBase64Encoded": data.as_base64(),
                "mimeType": mime_type
            })))
        }
    }
}

fn build_create_request_body(
    request: VideoGenerationRequest,
) -> Result<(serde_json::Value, Vec<Warning>), LlmError> {
    let mut options = parse_vertex_video_options(&request.provider_options_map)?;
    let mut extra_params = request.extra_params.unwrap_or_default();
    let mut warnings = Vec::new();

    if options.poll_interval_ms.is_none() {
        options.poll_interval_ms =
            take_u64_opt(&mut extra_params, "pollIntervalMs", "poll_interval_ms");
    }
    if options.poll_timeout_ms.is_none() {
        options.poll_timeout_ms =
            take_u64_opt(&mut extra_params, "pollTimeoutMs", "poll_timeout_ms");
    }
    if options.person_generation.is_none() {
        options.person_generation =
            take_string_opt(&mut extra_params, "personGeneration", "person_generation");
    }
    if options.negative_prompt.is_none() {
        options.negative_prompt =
            take_string_opt(&mut extra_params, "negativePrompt", "negative_prompt");
    }
    if options.generate_audio.is_none() {
        options.generate_audio =
            take_bool_opt(&mut extra_params, "generateAudio", "generate_audio");
    }
    if options.gcs_output_directory.is_none() {
        options.gcs_output_directory = take_string_opt(
            &mut extra_params,
            "gcsOutputDirectory",
            "gcs_output_directory",
        );
    }
    if options.reference_images.is_none() {
        options.reference_images =
            take_reference_images_opt(&mut extra_params, "referenceImages", "reference_images")?;
    }

    if request.video.is_some() {
        warnings.push(Warning::unsupported("video", Some(VIDEO_INPUT_WARNING)));
    }

    let mut instance = serde_json::Map::new();
    if let Some(prompt) = request.prompt {
        instance.insert("prompt".to_string(), serde_json::json!(prompt));
    }

    if let Some(image) = request.image.as_ref() {
        match image_input_to_vertex_payload(image) {
            Ok(Some(image)) => {
                instance.insert("image".to_string(), image);
            }
            Ok(None) => {}
            Err(warning) => warnings.push(warning),
        }
    }

    if let Some(reference_images) = options.reference_images {
        instance.insert(
            "referenceImages".to_string(),
            serde_json::to_value(reference_images).map_err(|err| {
                LlmError::ParseError(format!(
                    "Serialize Google Vertex video reference images failed: {err}"
                ))
            })?,
        );
    }

    let mut parameters = serde_json::Map::new();
    parameters.insert(
        "sampleCount".to_string(),
        serde_json::json!(request.count.unwrap_or(1)),
    );

    if let Some(aspect_ratio) = request.aspect_ratio {
        parameters.insert("aspectRatio".to_string(), serde_json::json!(aspect_ratio));
    }
    if let Some(resolution) = request.resolution {
        parameters.insert(
            "resolution".to_string(),
            serde_json::json!(map_vertex_resolution(&resolution)),
        );
    }
    if let Some(duration) = request.duration {
        parameters.insert("durationSeconds".to_string(), serde_json::json!(duration));
    }
    if let Some(seed) = request.seed {
        parameters.insert("seed".to_string(), serde_json::json!(seed));
    }
    if let Some(person_generation) = options.person_generation {
        parameters.insert(
            "personGeneration".to_string(),
            serde_json::json!(person_generation),
        );
    }
    if let Some(negative_prompt) = options.negative_prompt {
        parameters.insert(
            "negativePrompt".to_string(),
            serde_json::json!(negative_prompt),
        );
    }
    if let Some(generate_audio) = options.generate_audio {
        parameters.insert(
            "generateAudio".to_string(),
            serde_json::json!(generate_audio),
        );
    }
    if let Some(gcs_output_directory) = options.gcs_output_directory {
        parameters.insert(
            "gcsOutputDirectory".to_string(),
            serde_json::json!(gcs_output_directory),
        );
    }

    for (key, value) in options.extra_fields {
        parameters.entry(key).or_insert(value);
    }
    for (key, value) in extra_params {
        if key == "referenceImages" || key == "reference_images" {
            continue;
        }
        parameters.entry(key).or_insert(value);
    }

    Ok((
        serde_json::json!({
            "instances": [instance],
            "parameters": parameters,
        }),
        warnings,
    ))
}

fn build_status_metadata(
    response: &VertexOperationResponse,
) -> Result<HashMap<String, serde_json::Value>, LlmError> {
    let provider_videos = response
        .videos
        .iter()
        .map(vertex_provider_metadata_video)
        .collect::<Vec<_>>();
    let mut metadata = HashMap::new();
    metadata.insert(
        "vertex".to_string(),
        serde_json::json!({
            "videos": provider_videos,
            "raiMediaFilteredCount": response.rai_media_filtered_count
        }),
    );
    if !response.videos.is_empty() {
        metadata.insert(
            "_siumai".to_string(),
            serde_json::json!({
                "generatedVideos": response.videos
            }),
        );
    }
    Ok(metadata)
}

fn response_info(status: u16, headers: &HeaderMap, model: &str) -> Option<HttpResponseInfo> {
    Some(HttpResponseInfo {
        timestamp: chrono::Utc::now(),
        model_id: Some(model.to_string()),
        headers: headers_to_map(headers),
        body: None,
    })
    .filter(|_| status > 0)
}

async fn build_wiring(
    config: &GoogleVertexConfig,
    http_client: &reqwest::Client,
    retry_options: Option<&RetryOptions>,
    interceptors: &[Arc<dyn HttpInterceptor>],
) -> HttpExecutionWiring {
    let mut wiring = HttpExecutionWiring::new(
        "vertex",
        http_client.clone(),
        super::context::build_context(config).await,
    )
    .with_interceptors(interceptors.to_vec())
    .with_retry_options(retry_options.cloned());

    if let Some(transport) = config.http_transport.clone() {
        wiring = wiring.with_transport(transport);
    }

    wiring
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn create_video_task(
    config: &GoogleVertexConfig,
    default_model: &str,
    http_client: &reqwest::Client,
    retry_options: Option<&RetryOptions>,
    interceptors: &[Arc<dyn HttpInterceptor>],
    request: VideoGenerationRequest,
) -> Result<VideoGenerationResponse, LlmError> {
    let mut request = request;
    if request.model.trim().is_empty() {
        request.model = default_model.to_string();
    }
    if request.model.trim().is_empty() {
        return Err(LlmError::ConfigurationError(
            "Google Vertex video request requires a non-empty model id".to_string(),
        ));
    }
    if request.http_config.is_none() {
        request.http_config = Some(config.http_config.clone());
    }

    let model = request.model.clone();
    let per_request_http_config = request.http_config.clone();
    let (body, warnings) = build_create_request_body(request)?;
    let wiring = build_wiring(config, http_client, retry_options, interceptors).await;
    let exec = wiring.config(Arc::new(VertexVideoSpec));
    let url = build_predict_long_running_url(&model, &exec.provider_context);
    let result = execute_json_request(
        &exec,
        &url,
        HttpBody::Json(body),
        per_request_http_config.as_ref(),
        false,
    )
    .await?;

    let operation: VertexOperation = serde_json::from_value(result.json).map_err(|err| {
        LlmError::ParseError(format!(
            "Failed to parse Google Vertex video operation response: {err}"
        ))
    })?;

    Ok(VideoGenerationResponse {
        task_id: operation.name,
        base_resp: Some(BaseResponse {
            status_code: 0,
            status_msg: "OK".to_string(),
        }),
        metadata: HashMap::new(),
        warnings: (!warnings.is_empty()).then_some(warnings),
        response: response_info(result.status, &result.headers, &model),
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn query_video_task(
    config: &GoogleVertexConfig,
    default_model: &str,
    http_client: &reqwest::Client,
    retry_options: Option<&RetryOptions>,
    interceptors: &[Arc<dyn HttpInterceptor>],
    task_id: &str,
) -> Result<VideoTaskStatusResponse, LlmError> {
    if default_model.trim().is_empty() {
        return Err(LlmError::ConfigurationError(
            "Google Vertex video status queries require a configured model id".to_string(),
        ));
    }

    let wiring = build_wiring(config, http_client, retry_options, interceptors).await;
    let exec = wiring.config(Arc::new(VertexVideoSpec));
    let url = build_fetch_predict_operation_url(default_model, &exec.provider_context);
    let result = execute_json_request(
        &exec,
        &url,
        HttpBody::Json(serde_json::json!({ "operationName": task_id })),
        Some(&config.http_config),
        false,
    )
    .await?;

    let operation: VertexOperation = serde_json::from_value(result.json).map_err(|err| {
        LlmError::ParseError(format!(
            "Failed to parse Google Vertex video operation status: {err}"
        ))
    })?;

    let response = if !operation.done {
        VideoTaskStatusResponse {
            task_id: operation.name,
            status: VideoTaskStatus::Processing,
            file_id: None,
            video_url: None,
            provider_reference: None,
            duration: None,
            video_width: None,
            video_height: None,
            base_resp: None,
            metadata: HashMap::new(),
            response: response_info(result.status, &result.headers, default_model),
        }
    } else if let Some(error) = operation.error {
        VideoTaskStatusResponse {
            task_id: operation.name,
            status: VideoTaskStatus::Fail,
            file_id: None,
            video_url: None,
            provider_reference: None,
            duration: None,
            video_width: None,
            video_height: None,
            base_resp: Some(BaseResponse {
                status_code: error.code.unwrap_or(-1),
                status_msg: if error.message.trim().is_empty() {
                    error
                        .status
                        .unwrap_or_else(|| "Google Vertex video task failed".to_string())
                } else {
                    error.message
                },
            }),
            metadata: HashMap::new(),
            response: response_info(result.status, &result.headers, default_model),
        }
    } else if let Some(operation_response) = operation.response {
        let first_url = operation_response
            .videos
            .iter()
            .find_map(|video| video.gcs_uri.clone());

        VideoTaskStatusResponse {
            task_id: operation.name,
            status: VideoTaskStatus::Success,
            file_id: None,
            video_url: first_url,
            provider_reference: None,
            duration: None,
            video_width: None,
            video_height: None,
            base_resp: Some(BaseResponse {
                status_code: 0,
                status_msg: "OK".to_string(),
            }),
            metadata: build_status_metadata(&operation_response)?,
            response: response_info(result.status, &result.headers, default_model),
        }
    } else {
        VideoTaskStatusResponse {
            task_id: operation.name,
            status: VideoTaskStatus::Fail,
            file_id: None,
            video_url: None,
            provider_reference: None,
            duration: None,
            video_width: None,
            video_height: None,
            base_resp: Some(BaseResponse {
                status_code: -1,
                status_msg: "Google Vertex video task completed without a response payload"
                    .to_string(),
            }),
            metadata: HashMap::new(),
            response: response_info(result.status, &result.headers, default_model),
        }
    };
    Ok(response)
}

pub(super) fn get_supported_video_models() -> Vec<String> {
    super::models::ALL_VIDEO
        .iter()
        .map(|model| (*model).to_string())
        .collect()
}

pub(super) fn get_supported_resolutions(model: &str) -> Vec<String> {
    let model = normalize_vertex_model_id(model);
    if model.starts_with("veo-2.") {
        Vec::new()
    } else {
        vec!["720p".to_string(), "1080p".to_string()]
    }
}

pub(super) fn get_supported_durations(model: &str) -> Vec<u32> {
    let model = normalize_vertex_model_id(model);
    if model.starts_with("veo-2.") {
        vec![5, 6, 8]
    } else {
        vec![4, 6, 8]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::http::transport::{
        HttpTransport, HttpTransportRequest, HttpTransportResponse, HttpTransportStreamBody,
        HttpTransportStreamResponse,
    };
    use crate::providers::vertex::VertexVideoRequestExt;
    use async_trait::async_trait;
    use reqwest::header::{CONTENT_TYPE, HeaderValue};
    use std::sync::{Arc, Mutex};

    fn source_section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
        let start_index = source.find(start).expect("section start marker");
        let end_index = source[start_index..]
            .find(end)
            .map(|offset| start_index + offset)
            .expect("section end marker");
        &source[start_index..end_index]
    }

    #[test]
    fn video_request_and_response_paths_keep_provider_maps_directional() {
        let source = include_str!("video.rs");

        let request_section = source_section(
            source,
            "fn parse_vertex_video_options(",
            "fn build_status_metadata(",
        );
        assert!(request_section.contains("provider_options_map"));
        assert!(
            !request_section.contains("provider_metadata"),
            "Google Vertex video request helpers must not read legacy provider_metadata"
        );
        assert!(
            !request_section.contains("providerMetadata"),
            "Google Vertex video request helpers must not read legacy providerMetadata"
        );

        let response_section = source_section(source, "fn build_status_metadata(", "#[cfg(test)]");
        assert!(
            !response_section.contains("provider_options"),
            "Google Vertex video response helpers must not read request provider_options"
        );
        assert!(
            !response_section.contains("providerOptions"),
            "Google Vertex video response helpers must not read request providerOptions"
        );
    }

    #[derive(Clone)]
    struct JsonCaptureTransport {
        response_body: Arc<Vec<u8>>,
        last: Arc<Mutex<Option<HttpTransportRequest>>>,
    }

    impl JsonCaptureTransport {
        fn new(response: serde_json::Value) -> Self {
            Self {
                response_body: Arc::new(serde_json::to_vec(&response).expect("response json")),
                last: Arc::new(Mutex::new(None)),
            }
        }

        fn take(&self) -> Option<HttpTransportRequest> {
            self.last.lock().expect("lock request").take()
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

    #[tokio::test]
    async fn create_video_task_builds_ai_sdk_aligned_vertex_body() {
        let transport = JsonCaptureTransport::new(serde_json::json!({
            "name": "operations/test-video-123",
            "done": false
        }));
        let config =
            GoogleVertexConfig::new("https://example.com/custom", "veo-3.1-generate-preview")
                .with_api_key("test-key")
                .with_http_transport(Arc::new(transport.clone()));
        let http_client = reqwest::Client::new();

        let response = create_video_task(
            &config,
            "veo-3.1-generate-preview",
            &http_client,
            None,
            &[],
            VideoGenerationRequest::new("veo-3.1-generate-preview", "draw a robot")
                .with_n(2)
                .with_aspect_ratio("16:9")
                .with_duration(6)
                .with_resolution("1920x1080")
                .with_seed(9)
                .with_image(VideoGenerationInput::file_with_media_type(
                    vec![1, 2, 3],
                    "image/png",
                ))
                .with_vertex_video_options(
                    GoogleVertexVideoModelOptions::new()
                        .with_poll_interval_ms(250)
                        .with_poll_timeout_ms(30_000)
                        .with_person_generation("allow_adult")
                        .with_negative_prompt("blurry")
                        .with_generate_audio(true)
                        .with_gcs_output_directory("gs://bucket/output/")
                        .with_reference_images(vec![
                            GoogleVertexReferenceImage::new()
                                .with_gcs_uri("gs://bucket/reference.png"),
                        ]),
                ),
        )
        .await
        .expect("create video task");

        assert_eq!(response.task_id, "operations/test-video-123");
        assert_eq!(response.warnings, None);

        let req = transport.take().expect("captured request");
        assert_eq!(
            req.url,
            "https://example.com/custom/models/veo-3.1-generate-preview:predictLongRunning?key=test-key"
        );
        assert_eq!(
            req.body["instances"][0]["prompt"],
            serde_json::json!("draw a robot")
        );
        assert_eq!(
            req.body["instances"][0]["image"],
            serde_json::json!({
                "bytesBase64Encoded": "AQID",
                "mimeType": "image/png"
            })
        );
        assert_eq!(
            req.body["instances"][0]["referenceImages"][0]["gcsUri"],
            serde_json::json!("gs://bucket/reference.png")
        );
        assert_eq!(req.body["parameters"]["sampleCount"], serde_json::json!(2));
        assert_eq!(
            req.body["parameters"]["aspectRatio"],
            serde_json::json!("16:9")
        );
        assert_eq!(
            req.body["parameters"]["durationSeconds"],
            serde_json::json!(6)
        );
        assert_eq!(
            req.body["parameters"]["resolution"],
            serde_json::json!("1080p")
        );
        assert_eq!(req.body["parameters"]["seed"], serde_json::json!(9));
        assert_eq!(
            req.body["parameters"]["personGeneration"],
            serde_json::json!("allow_adult")
        );
        assert_eq!(
            req.body["parameters"]["negativePrompt"],
            serde_json::json!("blurry")
        );
        assert_eq!(
            req.body["parameters"]["generateAudio"],
            serde_json::json!(true)
        );
        assert_eq!(
            req.body["parameters"]["gcsOutputDirectory"],
            serde_json::json!("gs://bucket/output/")
        );
        assert!(req.body["parameters"].get("pollIntervalMs").is_none());
        assert!(req.body["parameters"].get("pollTimeoutMs").is_none());
    }

    #[test]
    fn polling_options_reads_provider_options_before_extra_params() {
        let request = VideoGenerationRequest::new("veo-3.1-generate-preview", "draw a robot")
            .with_extra_param("pollIntervalMs", serde_json::json!(999))
            .with_extra_param("pollTimeoutMs", serde_json::json!(9999))
            .with_vertex_video_options(
                GoogleVertexVideoModelOptions::new()
                    .with_poll_interval_ms(250)
                    .with_poll_timeout_ms(30_000),
            );

        let options = polling_options(&request).expect("vertex polling options");

        assert_eq!(options.poll_interval, Some(Duration::from_millis(250)));
        assert_eq!(options.poll_timeout, Some(Duration::from_millis(30_000)));
    }

    #[tokio::test]
    async fn create_video_task_warns_when_seed_image_is_url_backed() {
        let transport = JsonCaptureTransport::new(serde_json::json!({
            "name": "operations/test-video-123",
            "done": false
        }));
        let config =
            GoogleVertexConfig::new("https://example.com/custom", "veo-3.1-generate-preview")
                .with_api_key("test-key")
                .with_http_transport(Arc::new(transport.clone()));
        let http_client = reqwest::Client::new();

        let response = create_video_task(
            &config,
            "veo-3.1-generate-preview",
            &http_client,
            None,
            &[],
            VideoGenerationRequest::new("veo-3.1-generate-preview", "draw a robot")
                .with_image(VideoGenerationInput::url("https://example.com/start.png")),
        )
        .await
        .expect("create video task");

        assert_eq!(
            response.warnings,
            Some(vec![Warning::unsupported(
                "URL-based image input",
                Some(URL_IMAGE_WARNING),
            )])
        );

        let req = transport.take().expect("captured request");
        assert!(req.body["instances"][0].get("image").is_none());
    }

    #[test]
    fn build_create_request_body_allows_promptless_image_to_video_requests() {
        let (body, warnings) = build_create_request_body(
            VideoGenerationRequest::new_without_prompt("veo-3.1-generate-preview").with_image(
                VideoGenerationInput::file_with_media_type(vec![1, 2, 3], "image/png"),
            ),
        )
        .expect("build vertex video body");

        assert!(warnings.is_empty());
        assert!(body["instances"][0].get("prompt").is_none());
        assert_eq!(
            body["instances"][0]["image"],
            serde_json::json!({
                "bytesBase64Encoded": "AQID",
                "mimeType": "image/png"
            })
        );
    }

    #[tokio::test]
    async fn query_video_task_maps_vertex_operation_status_into_task_response() {
        let transport = JsonCaptureTransport::new(serde_json::json!({
            "name": "operations/test-video-123",
            "done": true,
            "response": {
                "videos": [
                    {
                        "gcsUri": "gs://bucket/output/video.mp4",
                        "mimeType": "video/mp4"
                    },
                    {
                        "bytesBase64Encoded": "Zm9v",
                        "mimeType": "video/mp4"
                    }
                ],
                "raiMediaFilteredCount": 1
            }
        }));
        let config =
            GoogleVertexConfig::new("https://example.com/custom", "veo-3.1-generate-preview")
                .with_api_key("test-key")
                .with_http_transport(Arc::new(transport.clone()));
        let http_client = reqwest::Client::new();

        let response = query_video_task(
            &config,
            "veo-3.1-generate-preview",
            &http_client,
            None,
            &[],
            "operations/test-video-123",
        )
        .await
        .expect("query video task");

        assert_eq!(response.status, VideoTaskStatus::Success);
        assert_eq!(response.file_id, None);
        assert_eq!(
            response.video_url.as_deref(),
            Some("gs://bucket/output/video.mp4")
        );
        assert!(response.provider_reference().is_none());
        assert_eq!(
            response
                .metadata
                .get("vertex")
                .and_then(|value| value.get("videos"))
                .and_then(|value| value.as_array())
                .map(Vec::len),
            Some(2)
        );
        assert_eq!(
            response
                .metadata
                .get("vertex")
                .and_then(|value| value.get("videos"))
                .and_then(|value| value.as_array())
                .and_then(|videos| videos.get(1))
                .and_then(|video| video.get("bytesBase64Encoded")),
            None
        );
        assert_eq!(
            response
                .metadata
                .get("vertex")
                .and_then(|value| value.get("raiMediaFilteredCount"))
                .and_then(|value| value.as_u64()),
            Some(1)
        );
        assert_eq!(
            response
                .metadata
                .get("_siumai")
                .and_then(|value| value.get("generatedVideos"))
                .and_then(|value| value.as_array())
                .and_then(|videos| videos.get(1))
                .and_then(|video| video.get("bytesBase64Encoded"))
                .and_then(|value| value.as_str()),
            Some("Zm9v")
        );

        let req = transport.take().expect("captured request");
        assert_eq!(
            req.url,
            "https://example.com/custom/models/veo-3.1-generate-preview:fetchPredictOperation?key=test-key"
        );
        assert_eq!(
            req.body,
            serde_json::json!({
                "operationName": "operations/test-video-123"
            })
        );
    }
}
