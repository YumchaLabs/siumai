//! Alibaba DashScope native video model.
//!
//! The AI SDK Alibaba package exposes video generation through DashScope's native task API rather
//! than the OpenAI-compatible chat endpoint. Rust keeps video generation task-oriented, so this
//! model implements `VideoGenerationCapability` with explicit create/query operations.

use super::providers::models::alibaba;
use crate::LlmError;
use crate::core::{ProviderContext, ProviderSpec};
use crate::execution::executors::common::{
    HttpBody, HttpExecutionConfig, execute_get_binary, execute_get_request, execute_json_request,
};
use crate::execution::http::interceptor::HttpInterceptor;
use crate::execution::http::transport::HttpTransport;
use crate::provider_options::AlibabaVideoModelOptions;
use crate::traits::{ProviderCapabilities, VideoGenerationCapability};
use crate::types::{
    BaseResponse, HttpConfig, HttpResponseInfo, MaterializedVideoAsset, ProviderOptionsMap,
    ProviderReference, VideoGenerationInput, VideoGenerationRequest, VideoGenerationResponse,
    VideoTaskStatus, VideoTaskStatusResponse, Warning,
};
use async_trait::async_trait;
use reqwest::header::HeaderMap;
use serde::Deserialize;
use serde_json::{Map, Value};
use siumai_core::traits::ModelMetadata;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

/// AI SDK Alibaba package default DashScope native video endpoint.
pub const ALIBABA_VIDEO_DEFAULT_BASE_URL: &str = "https://dashscope-intl.aliyuncs.com";
const ALIBABA_VIDEO_CREATE_PATH: &str = "/api/v1/services/aigc/video-generation/video-synthesis";
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AlibabaVideoMode {
    TextToVideo,
    ImageToVideo,
    ReferenceToVideo,
}

#[derive(Clone)]
pub struct AlibabaVideoModel {
    model_id: String,
    base_url: String,
    api_key: Option<String>,
    headers: HashMap<String, String>,
    fetch: Option<Arc<dyn HttpTransport>>,
    http_client: reqwest::Client,
    http_interceptors: Vec<Arc<dyn HttpInterceptor>>,
}

impl AlibabaVideoModel {
    /// Create an Alibaba video model with the default DashScope native endpoint.
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            base_url: ALIBABA_VIDEO_DEFAULT_BASE_URL.to_string(),
            api_key: None,
            headers: HashMap::new(),
            fetch: None,
            http_client: reqwest::Client::new(),
            http_interceptors: Vec::new(),
        }
    }

    /// Set the API key used as `Authorization: Bearer ...`.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the DashScope native video base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = normalize_base_url(base_url.into());
        self
    }

    /// Extend request headers.
    pub fn with_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.headers.extend(headers);
        self
    }

    /// Add one request header.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }

    /// Set a custom HTTP transport, mirroring AI SDK `fetch`.
    pub fn with_fetch(mut self, fetch: Arc<dyn HttpTransport>) -> Self {
        self.fetch = Some(fetch);
        self
    }

    /// Set a custom `reqwest` client for real HTTP execution.
    pub fn with_http_client(mut self, http_client: reqwest::Client) -> Self {
        self.http_client = http_client;
        self
    }

    /// Install HTTP interceptors.
    pub fn with_http_interceptors(mut self, interceptors: Vec<Arc<dyn HttpInterceptor>>) -> Self {
        self.http_interceptors = interceptors;
        self
    }

    /// Return the configured DashScope native video base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    fn mode_for_model(model_id: &str) -> AlibabaVideoMode {
        if model_id.contains("-i2v") {
            AlibabaVideoMode::ImageToVideo
        } else if model_id.contains("-r2v") {
            AlibabaVideoMode::ReferenceToVideo
        } else {
            AlibabaVideoMode::TextToVideo
        }
    }

    fn effective_model_id<'a>(&'a self, request_model: &'a str) -> &'a str {
        if request_model.trim().is_empty() {
            &self.model_id
        } else {
            request_model
        }
    }

    fn execution_config(&self) -> HttpExecutionConfig {
        let ctx = ProviderContext::new(
            "alibaba.video",
            self.base_url.clone(),
            self.api_key
                .as_ref()
                .filter(|api_key| !api_key.trim().is_empty())
                .cloned(),
            self.headers.clone(),
        );

        HttpExecutionConfig {
            provider_id: "alibaba.video".to_string(),
            http_client: self.http_client.clone(),
            transport: self.fetch.clone(),
            provider_spec: Arc::new(AlibabaVideoSpec),
            provider_context: ctx,
            interceptors: self.http_interceptors.clone(),
            retry_options: None,
        }
    }

    fn download_execution_config(&self) -> HttpExecutionConfig {
        let ctx =
            ProviderContext::new("alibaba.video", self.base_url.clone(), None, HashMap::new());

        HttpExecutionConfig {
            provider_id: "alibaba.video".to_string(),
            http_client: self.http_client.clone(),
            transport: self.fetch.clone(),
            provider_spec: Arc::new(AlibabaVideoDownloadSpec),
            provider_context: ctx,
            interceptors: self.http_interceptors.clone(),
            retry_options: None,
        }
    }
}

impl ModelMetadata for AlibabaVideoModel {
    fn provider_id(&self) -> &str {
        "alibaba.video"
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[derive(Clone, Copy)]
struct AlibabaVideoSpec;

impl ProviderSpec for AlibabaVideoSpec {
    fn id(&self) -> &'static str {
        "alibaba.video"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::new().with_custom_feature("video_generation", true)
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        crate::standards::openai::headers::build_openai_compatible_json_headers(ctx)
    }

    fn classify_http_error(
        &self,
        status: u16,
        body_text: &str,
        _headers: &HeaderMap,
    ) -> Option<LlmError> {
        #[derive(Deserialize)]
        struct AlibabaVideoError {
            code: Option<String>,
            message: String,
            request_id: Option<String>,
        }

        let error: AlibabaVideoError = serde_json::from_str(body_text).ok()?;
        Some(LlmError::ApiError {
            code: status,
            message: error.message,
            details: Some(serde_json::json!({
                "code": error.code,
                "requestId": error.request_id
            })),
        })
    }
}

#[derive(Clone, Copy)]
struct AlibabaVideoDownloadSpec;

impl ProviderSpec for AlibabaVideoDownloadSpec {
    fn id(&self) -> &'static str {
        "alibaba.video.download"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::default()
    }

    fn build_headers(&self, _ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        Ok(HeaderMap::new())
    }
}

#[derive(Debug, Deserialize)]
struct AlibabaCreateTaskOutput {
    task_status: Option<String>,
    task_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AlibabaCreateTaskResponse {
    output: Option<AlibabaCreateTaskOutput>,
    request_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AlibabaTaskStatusOutput {
    task_id: Option<String>,
    task_status: Option<String>,
    video_url: Option<String>,
    submit_time: Option<String>,
    scheduled_time: Option<String>,
    end_time: Option<String>,
    orig_prompt: Option<String>,
    actual_prompt: Option<String>,
    code: Option<String>,
    message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AlibabaTaskUsage {
    duration: Option<f32>,
    output_video_duration: Option<f32>,
    #[serde(rename = "SR")]
    sr: Option<f32>,
    size: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AlibabaTaskStatusResponse {
    output: Option<AlibabaTaskStatusOutput>,
    usage: Option<AlibabaTaskUsage>,
    request_id: Option<String>,
}

fn normalize_base_url(base_url: String) -> String {
    base_url.trim_end_matches('/').to_string()
}

fn headers_to_map(headers: &HeaderMap) -> HashMap<String, String> {
    headers
        .iter()
        .filter_map(|(key, value)| {
            Some((key.as_str().to_string(), value.to_str().ok()?.to_string()))
        })
        .collect()
}

fn response_info(headers: &HeaderMap, model_id: &str) -> HttpResponseInfo {
    HttpResponseInfo {
        timestamp: chrono::Utc::now(),
        model_id: Some(model_id.to_string()),
        headers: headers_to_map(headers),
        body: None,
    }
}

fn value_to_options(value: &Value) -> Result<AlibabaVideoModelOptions, LlmError> {
    serde_json::from_value(value.clone()).map_err(|err| {
        LlmError::InvalidParameter(format!("Invalid Alibaba video provider options: {err}"))
    })
}

fn merge_options(target: &mut AlibabaVideoModelOptions, source: AlibabaVideoModelOptions) {
    if source.negative_prompt.is_some() {
        target.negative_prompt = source.negative_prompt;
    }
    if source.audio_url.is_some() {
        target.audio_url = source.audio_url;
    }
    if source.prompt_extend.is_some() {
        target.prompt_extend = source.prompt_extend;
    }
    if source.shot_type.is_some() {
        target.shot_type = source.shot_type;
    }
    if source.watermark.is_some() {
        target.watermark = source.watermark;
    }
    if source.audio.is_some() {
        target.audio = source.audio;
    }
    if source.reference_urls.is_some() {
        target.reference_urls = source.reference_urls;
    }
    if source.poll_interval_ms.is_some() {
        target.poll_interval_ms = source.poll_interval_ms;
    }
    if source.poll_timeout_ms.is_some() {
        target.poll_timeout_ms = source.poll_timeout_ms;
    }
}

fn request_options(map: &ProviderOptionsMap) -> Result<AlibabaVideoModelOptions, LlmError> {
    let mut options = AlibabaVideoModelOptions::default();
    for provider_id in ["alibaba", "qwen"] {
        if let Some(value) = map.get(provider_id) {
            merge_options(&mut options, value_to_options(value)?);
        }
    }

    if let Some(shot_type) = options.shot_type.as_deref()
        && shot_type != "single"
        && shot_type != "multi"
    {
        return Err(LlmError::InvalidParameter(
            "Alibaba video shotType must be either `single` or `multi`".to_string(),
        ));
    }
    if matches!(options.poll_interval_ms, Some(0)) {
        return Err(LlmError::InvalidParameter(
            "Alibaba video pollIntervalMs must be positive".to_string(),
        ));
    }
    if matches!(options.poll_timeout_ms, Some(0)) {
        return Err(LlmError::InvalidParameter(
            "Alibaba video pollTimeoutMs must be positive".to_string(),
        ));
    }

    Ok(options)
}

fn insert_if_some<T>(target: &mut Map<String, Value>, key: &str, value: Option<T>)
where
    T: serde::Serialize,
{
    if let Some(value) = value
        && let Ok(value) = serde_json::to_value(value)
    {
        target.insert(key.to_string(), value);
    }
}

fn add_unsupported_video_warnings(request: &VideoGenerationRequest, warnings: &mut Vec<Warning>) {
    if request.aspect_ratio.is_some() {
        warnings.push(Warning::unsupported(
            "aspectRatio",
            Some(
                "Alibaba video models use explicit size/resolution dimensions. Use the resolution option or providerOptions.alibaba for size control.",
            ),
        ));
    }
    if request.fps.is_some() {
        warnings.push(Warning::unsupported(
            "fps",
            Some("Alibaba video models do not support custom FPS."),
        ));
    }
    if request.count.is_some_and(|count| count > 1) {
        warnings.push(Warning::unsupported(
            "n",
            Some("Alibaba video models only support generating 1 video per call."),
        ));
    }
}

fn map_i2v_resolution(resolution: &str) -> String {
    match resolution {
        "1280x720" | "720x1280" | "960x960" | "1088x832" | "832x1088" => "720P".to_string(),
        "1920x1080" | "1080x1920" | "1440x1440" | "1632x1248" | "1248x1632" => "1080P".to_string(),
        "832x480" | "480x832" | "624x624" => "480P".to_string(),
        _ => resolution.to_string(),
    }
}

fn image_input_to_img_url(input: &VideoGenerationInput) -> Result<String, LlmError> {
    match input {
        VideoGenerationInput::Url { url, .. } => Ok(url.clone()),
        VideoGenerationInput::File { data, .. } => Ok(data.as_base64()),
    }
}

fn build_create_body(
    model_id: &str,
    request: &VideoGenerationRequest,
    options: &AlibabaVideoModelOptions,
) -> Result<Value, LlmError> {
    let mode = AlibabaVideoModel::mode_for_model(model_id);
    let mut input = Map::new();
    let mut parameters = Map::new();

    insert_if_some(&mut input, "prompt", request.prompt.clone());
    insert_if_some(
        &mut input,
        "negative_prompt",
        options.negative_prompt.clone(),
    );
    insert_if_some(&mut input, "audio_url", options.audio_url.clone());

    if mode == AlibabaVideoMode::ImageToVideo
        && let Some(image) = request.image.as_ref()
    {
        input.insert(
            "img_url".to_string(),
            Value::String(image_input_to_img_url(image)?),
        );
    }

    if mode == AlibabaVideoMode::ReferenceToVideo {
        insert_if_some(&mut input, "reference_urls", options.reference_urls.clone());
    }

    if let Some(extra) = request.extra_params.as_ref() {
        for (key, value) in extra {
            parameters.insert(key.clone(), value.clone());
        }
    }

    insert_if_some(&mut parameters, "duration", request.duration);
    insert_if_some(&mut parameters, "seed", request.seed);

    if let Some(resolution) = request.resolution.as_deref() {
        if mode == AlibabaVideoMode::ImageToVideo {
            parameters.insert(
                "resolution".to_string(),
                Value::String(map_i2v_resolution(resolution)),
            );
        } else {
            parameters.insert(
                "size".to_string(),
                Value::String(resolution.replace('x', "*")),
            );
        }
    }

    insert_if_some(&mut parameters, "prompt_extend", options.prompt_extend);
    insert_if_some(&mut parameters, "shot_type", options.shot_type.clone());
    insert_if_some(&mut parameters, "watermark", options.watermark);
    insert_if_some(&mut parameters, "audio", options.audio);

    Ok(serde_json::json!({
        "model": model_id,
        "input": input,
        "parameters": parameters
    }))
}

fn create_http_config(base: Option<&HttpConfig>) -> HttpConfig {
    let mut config = base.cloned().unwrap_or_default();
    config
        .headers
        .insert("X-DashScope-Async".to_string(), "enable".to_string());
    config
}

fn create_metadata(raw: &AlibabaCreateTaskResponse) -> HashMap<String, Value> {
    let mut metadata = HashMap::new();
    let mut alibaba = Map::new();
    insert_if_some(&mut alibaba, "requestId", raw.request_id.clone());
    if let Some(output) = raw.output.as_ref() {
        insert_if_some(&mut alibaba, "taskStatus", output.task_status.clone());
    }
    if !alibaba.is_empty() {
        metadata.insert("alibaba".to_string(), Value::Object(alibaba));
    }
    metadata
}

fn task_status(value: Option<&str>) -> VideoTaskStatus {
    match value {
        Some("SUCCEEDED") => VideoTaskStatus::Success,
        Some("FAILED") | Some("CANCELED") => VideoTaskStatus::Fail,
        Some("PENDING") => VideoTaskStatus::Queueing,
        Some("RUNNING") => VideoTaskStatus::Processing,
        _ => VideoTaskStatus::Processing,
    }
}

fn parse_size(size: Option<&str>) -> (Option<u32>, Option<u32>) {
    let Some(size) = size else {
        return (None, None);
    };
    let split = size
        .split_once('*')
        .or_else(|| size.split_once('x'))
        .or_else(|| size.split_once('X'));
    let Some((width, height)) = split else {
        return (None, None);
    };
    (width.parse().ok(), height.parse().ok())
}

fn task_metadata(raw: &AlibabaTaskStatusResponse) -> HashMap<String, Value> {
    let mut metadata = HashMap::new();
    let mut alibaba = Map::new();

    insert_if_some(&mut alibaba, "requestId", raw.request_id.clone());
    if let Some(output) = raw.output.as_ref() {
        insert_if_some(&mut alibaba, "submitTime", output.submit_time.clone());
        insert_if_some(&mut alibaba, "scheduledTime", output.scheduled_time.clone());
        insert_if_some(&mut alibaba, "endTime", output.end_time.clone());
        insert_if_some(&mut alibaba, "origPrompt", output.orig_prompt.clone());
        insert_if_some(&mut alibaba, "actualPrompt", output.actual_prompt.clone());
        insert_if_some(&mut alibaba, "code", output.code.clone());
        insert_if_some(&mut alibaba, "message", output.message.clone());
    }

    if let Some(usage) = raw.usage.as_ref() {
        let mut usage_value = Map::new();
        insert_if_some(&mut usage_value, "duration", usage.duration);
        insert_if_some(
            &mut usage_value,
            "outputVideoDuration",
            usage.output_video_duration,
        );
        insert_if_some(&mut usage_value, "resolution", usage.sr);
        insert_if_some(&mut usage_value, "size", usage.size.clone());
        if !usage_value.is_empty() {
            alibaba.insert("usage".to_string(), Value::Object(usage_value));
        }
    }

    if !alibaba.is_empty() {
        metadata.insert("alibaba".to_string(), Value::Object(alibaba));
    }
    metadata
}

#[async_trait]
impl VideoGenerationCapability for AlibabaVideoModel {
    async fn create_video_task(
        &self,
        request: VideoGenerationRequest,
    ) -> Result<VideoGenerationResponse, LlmError> {
        let model_id = self.effective_model_id(&request.model).to_string();
        let options = request_options(&request.provider_options_map)?;
        let body = build_create_body(&model_id, &request, &options)?;
        let url = format!(
            "{}{}",
            self.base_url.trim_end_matches('/'),
            ALIBABA_VIDEO_CREATE_PATH
        );
        let http_config = create_http_config(request.http_config.as_ref());
        let result = execute_json_request(
            &self.execution_config(),
            &url,
            HttpBody::Json(body),
            Some(&http_config),
            false,
        )
        .await?;

        let raw: AlibabaCreateTaskResponse =
            serde_json::from_value(result.json.clone()).map_err(|err| {
                LlmError::ParseError(format!(
                    "Failed to parse Alibaba video task response: {err}"
                ))
            })?;
        let task_id = raw
            .output
            .as_ref()
            .and_then(|output| output.task_id.clone())
            .ok_or_else(|| LlmError::ApiError {
                code: result.status,
                message: "No task_id returned from Alibaba video API".to_string(),
                details: Some(result.json.clone()),
            })?;

        let mut warnings = Vec::new();
        add_unsupported_video_warnings(&request, &mut warnings);

        Ok(VideoGenerationResponse {
            task_id,
            base_resp: None,
            metadata: create_metadata(&raw),
            warnings: (!warnings.is_empty()).then_some(warnings),
            response: Some(response_info(&result.headers, &model_id)),
        })
    }

    async fn query_video_task(&self, task_id: &str) -> Result<VideoTaskStatusResponse, LlmError> {
        let url = format!(
            "{}/api/v1/tasks/{task_id}",
            self.base_url.trim_end_matches('/')
        );
        let result = execute_get_request(&self.execution_config(), &url, None).await?;
        let raw: AlibabaTaskStatusResponse =
            serde_json::from_value(result.json.clone()).map_err(|err| {
                LlmError::ParseError(format!(
                    "Failed to parse Alibaba video status response: {err}"
                ))
            })?;

        let output = raw.output.as_ref();
        let video_url = output.and_then(|output| output.video_url.clone());
        let task_id = output
            .and_then(|output| output.task_id.as_deref())
            .unwrap_or(task_id)
            .to_string();
        let (video_width, video_height) =
            parse_size(raw.usage.as_ref().and_then(|usage| usage.size.as_deref()));
        let status = task_status(output.and_then(|output| output.task_status.as_deref()));
        let failed_message = output
            .and_then(|output| output.message.clone())
            .unwrap_or_else(|| "Alibaba video task failed".to_string());

        Ok(VideoTaskStatusResponse {
            task_id,
            status: status.clone(),
            file_id: None,
            video_url: video_url.clone(),
            provider_reference: video_url.map(|url| ProviderReference::single("alibaba", url)),
            duration: raw
                .usage
                .as_ref()
                .and_then(|usage| usage.output_video_duration.or(usage.duration)),
            video_width,
            video_height,
            base_resp: (status == VideoTaskStatus::Fail).then_some(BaseResponse {
                status_code: -1,
                status_msg: failed_message,
            }),
            metadata: task_metadata(&raw),
            response: Some(response_info(&result.headers, &self.model_id)),
        })
    }

    async fn materialize_video_reference(
        &self,
        provider_reference: &ProviderReference,
    ) -> Result<MaterializedVideoAsset, LlmError> {
        let url = provider_reference
            .get("alibaba")
            .or_else(|| provider_reference.get("alibaba.video"))
            .ok_or_else(|| {
                LlmError::InvalidInput(
                    "Alibaba video provider reference must contain an `alibaba` URL".to_string(),
                )
            })?;
        let result = execute_get_binary(&self.download_execution_config(), url, None).await?;
        let media_type = result
            .headers
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.split(';').next())
            .filter(|value| !value.is_empty())
            .unwrap_or("video/mp4")
            .to_string();

        Ok(MaterializedVideoAsset::new(result.bytes).with_media_type(media_type))
    }

    fn polling_options(
        &self,
        request: &VideoGenerationRequest,
    ) -> Result<siumai_core::video::VideoPollingOptions, LlmError> {
        let options = request_options(&request.provider_options_map)?;
        let mut polling = siumai_core::video::VideoPollingOptions::default();
        if let Some(ms) = options.poll_interval_ms {
            polling = polling.with_poll_interval(Duration::from_millis(ms));
        }
        if let Some(ms) = options.poll_timeout_ms {
            polling = polling.with_poll_timeout(Duration::from_millis(ms));
        }
        Ok(polling)
    }

    fn max_videos_per_call(&self) -> Option<u32> {
        Some(1)
    }

    fn get_supported_models(&self) -> Vec<String> {
        alibaba::ALL_VIDEO
            .iter()
            .map(|model| (*model).to_string())
            .collect()
    }

    fn get_supported_resolutions(&self, model: &str) -> Vec<String> {
        if Self::mode_for_model(model) == AlibabaVideoMode::ImageToVideo {
            vec!["480P".to_string(), "720P".to_string(), "1080P".to_string()]
        } else {
            vec![
                "1280x720".to_string(),
                "720x1280".to_string(),
                "1920x1080".to_string(),
                "1080x1920".to_string(),
            ]
        }
    }

    fn get_supported_durations(&self, _model: &str) -> Vec<u32> {
        vec![5, 10]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::http::interceptor::HttpRequestContext;
    use crate::execution::http::transport::{
        HttpTransportGetRequest, HttpTransportRequest, HttpTransportResponse,
    };
    use std::collections::VecDeque;
    use std::sync::Mutex;

    #[derive(Clone)]
    struct CapturedRequest {
        url: String,
        headers: HeaderMap,
        body: Option<Value>,
    }

    #[derive(Default)]
    struct QueueTransport {
        responses: Mutex<VecDeque<Value>>,
        captured: Mutex<Vec<CapturedRequest>>,
    }

    impl QueueTransport {
        fn with_responses(responses: impl IntoIterator<Item = Value>) -> Self {
            Self {
                responses: Mutex::new(responses.into_iter().collect()),
                captured: Mutex::new(Vec::new()),
            }
        }

        fn captured(&self) -> Vec<CapturedRequest> {
            self.captured.lock().expect("captured").clone()
        }
    }

    #[async_trait]
    impl HttpTransport for QueueTransport {
        async fn execute_json(
            &self,
            request: HttpTransportRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            self.captured
                .lock()
                .expect("captured")
                .push(CapturedRequest {
                    url: request.url,
                    headers: request.headers,
                    body: Some(request.body),
                });
            let body = self
                .responses
                .lock()
                .expect("responses")
                .pop_front()
                .expect("queued response");
            Ok(HttpTransportResponse {
                status: 200,
                headers: HeaderMap::new(),
                body: serde_json::to_vec(&body).expect("json bytes"),
            })
        }

        async fn execute_get(
            &self,
            request: HttpTransportGetRequest,
        ) -> Result<HttpTransportResponse, LlmError> {
            self.captured
                .lock()
                .expect("captured")
                .push(CapturedRequest {
                    url: request.url,
                    headers: request.headers,
                    body: None,
                });
            let body = self
                .responses
                .lock()
                .expect("responses")
                .pop_front()
                .expect("queued response");
            Ok(HttpTransportResponse {
                status: 200,
                headers: HeaderMap::new(),
                body: serde_json::to_vec(&body).expect("json bytes"),
            })
        }
    }

    fn image_request() -> VideoGenerationRequest {
        VideoGenerationRequest::new("wan2.6-i2v", "animate")
            .with_image(VideoGenerationInput::base64_with_media_type(
                "AQID",
                "image/png",
            ))
            .with_duration(5)
            .with_seed(7)
            .with_resolution("1280x720")
            .with_aspect_ratio("16:9")
            .with_fps(24)
            .with_count(2)
            .with_provider_option(
                "alibaba",
                serde_json::json!({
                    "negativePrompt": "no blur",
                    "audioUrl": "https://example.com/audio.mp3",
                    "promptExtend": true,
                    "shotType": "single",
                    "watermark": false,
                    "audio": true,
                    "pollIntervalMs": 1000,
                    "pollTimeoutMs": 2000
                }),
            )
    }

    #[tokio::test]
    async fn create_video_task_maps_ai_sdk_alibaba_shape() {
        let transport = Arc::new(QueueTransport::with_responses([serde_json::json!({
            "output": {
                "task_status": "PENDING",
                "task_id": "task-123"
            },
            "request_id": "req-123"
        })]));
        let model = AlibabaVideoModel::new("wan2.6-i2v")
            .with_api_key("test-key")
            .with_base_url("https://dashscope.test")
            .with_fetch(transport.clone());

        let response = model
            .create_video_task(image_request())
            .await
            .expect("create task");

        assert_eq!(response.task_id, "task-123");
        assert_eq!(
            response
                .metadata
                .get("alibaba")
                .and_then(|value| value.get("requestId"))
                .and_then(Value::as_str),
            Some("req-123")
        );
        let warnings = response.warnings.expect("warnings");
        assert_eq!(warnings.len(), 3);

        let captured = transport.captured();
        assert_eq!(
            captured[0].url,
            "https://dashscope.test/api/v1/services/aigc/video-generation/video-synthesis"
        );
        assert_eq!(
            captured[0]
                .headers
                .get("Authorization")
                .and_then(|value| value.to_str().ok()),
            Some("Bearer test-key")
        );
        assert_eq!(
            captured[0]
                .headers
                .get("X-DashScope-Async")
                .and_then(|value| value.to_str().ok()),
            Some("enable")
        );

        let body = captured[0].body.as_ref().expect("body");
        assert_eq!(body["model"], serde_json::json!("wan2.6-i2v"));
        assert_eq!(body["input"]["prompt"], serde_json::json!("animate"));
        assert_eq!(
            body["input"]["negative_prompt"],
            serde_json::json!("no blur")
        );
        assert_eq!(
            body["input"]["audio_url"],
            serde_json::json!("https://example.com/audio.mp3")
        );
        assert_eq!(body["input"]["img_url"], serde_json::json!("AQID"));
        assert_eq!(body["parameters"]["duration"], serde_json::json!(5));
        assert_eq!(body["parameters"]["seed"], serde_json::json!(7));
        assert_eq!(body["parameters"]["resolution"], serde_json::json!("720P"));
        assert_eq!(body["parameters"]["prompt_extend"], serde_json::json!(true));
        assert_eq!(body["parameters"]["shot_type"], serde_json::json!("single"));
        assert_eq!(body["parameters"]["watermark"], serde_json::json!(false));
        assert_eq!(body["parameters"]["audio"], serde_json::json!(true));
        assert!(body["parameters"].get("pollIntervalMs").is_none());
    }

    #[tokio::test]
    async fn query_video_task_maps_status_metadata_and_dimensions() {
        let transport = Arc::new(QueueTransport::with_responses([serde_json::json!({
            "output": {
                "task_id": "task-123",
                "task_status": "SUCCEEDED",
                "video_url": "https://example.com/out.mp4",
                "actual_prompt": "expanded prompt"
            },
            "usage": {
                "duration": 5,
                "output_video_duration": 5,
                "SR": 720,
                "size": "1280*720"
            },
            "request_id": "req-456"
        })]));
        let model = AlibabaVideoModel::new("wan2.6-t2v")
            .with_api_key("test-key")
            .with_base_url("https://dashscope.test")
            .with_fetch(transport.clone());

        let response = model
            .query_video_task("task-123")
            .await
            .expect("query task");

        assert_eq!(response.status, VideoTaskStatus::Success);
        assert_eq!(
            response.video_url.as_deref(),
            Some("https://example.com/out.mp4")
        );
        assert_eq!(response.video_width, Some(1280));
        assert_eq!(response.video_height, Some(720));
        assert_eq!(
            response
                .provider_reference
                .as_ref()
                .and_then(|reference| reference.get("alibaba")),
            Some("https://example.com/out.mp4")
        );
        assert_eq!(
            response
                .metadata
                .get("alibaba")
                .and_then(|value| value.get("actualPrompt"))
                .and_then(Value::as_str),
            Some("expanded prompt")
        );
        assert_eq!(
            response
                .metadata
                .get("alibaba")
                .and_then(|value| value.get("usage"))
                .and_then(|value| value.get("outputVideoDuration"))
                .and_then(Value::as_f64),
            Some(5.0)
        );

        let captured = transport.captured();
        assert_eq!(
            captured[0].url,
            "https://dashscope.test/api/v1/tasks/task-123"
        );
    }

    #[tokio::test]
    async fn materialize_video_reference_downloads_without_provider_auth_header() {
        let transport = Arc::new(QueueTransport::with_responses([serde_json::json!("video")]));
        let model = AlibabaVideoModel::new("wan2.6-t2v")
            .with_api_key("test-key")
            .with_fetch(transport.clone());

        let asset = model
            .materialize_video_reference(&ProviderReference::single(
                "alibaba",
                "https://objects.example.com/out.mp4",
            ))
            .await
            .expect("materialize reference");

        assert_eq!(asset.media_type.as_deref(), Some("video/mp4"));
        let captured = transport.captured();
        assert_eq!(captured[0].url, "https://objects.example.com/out.mp4");
        assert!(
            captured[0].headers.get("Authorization").is_none(),
            "asset download should not leak provider Authorization to a generated asset URL"
        );
    }

    #[test]
    fn polling_options_reads_provider_options() {
        let model = AlibabaVideoModel::new("wan2.6-t2v");
        let request = VideoGenerationRequest::new("wan2.6-t2v", "prompt").with_provider_option(
            "alibaba",
            serde_json::json!({
                "pollIntervalMs": 1234,
                "pollTimeoutMs": 5678
            }),
        );

        let polling = model.polling_options(&request).expect("polling options");
        assert_eq!(polling.poll_interval, Some(Duration::from_millis(1234)));
        assert_eq!(polling.poll_timeout, Some(Duration::from_millis(5678)));
    }

    #[allow(dead_code)]
    fn _assert_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<HttpRequestContext>();
    }
}
