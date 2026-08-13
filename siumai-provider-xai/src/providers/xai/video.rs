//! Provider-owned typed xAI video-generation jobs.

use std::fmt;

use http::{Method, StatusCode};
use serde::Deserialize;
use serde_json::{Map, Value};
use siumai_core::{
    CallOptions, Error, ErrorKind, ModelId, PublicDiagnosticText, ResponseDiagnostics,
    SensitiveResponse,
};
use siumai_transport::{
    ProviderTransport, ReplaySafety, RequestBody, RequestBuildError, RequestPlan, RequestTarget,
    ResponseHeaders, TransportResponse,
};

use super::files::XaiFileId;

pub const VIDEO_SOURCE: &str = "https://docs.x.ai/developers/model-capabilities/video/generation";
pub const VIDEO_VERIFIED_ON: &str = "2026-08-09";

const CREATE_TARGET: &str = "videos/generations";
const MAX_PROMPT_BYTES: usize = 16 * 1024;

/// xAI video output resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XaiVideoResolution {
    P480,
    P720,
    P1080,
}

/// xAI video aspect ratio for text/image-to-video generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XaiVideoAspectRatio {
    Landscape16By9,
    Portrait9By16,
    Square,
    Standard4By3,
    Portrait3By4,
    Landscape3By2,
    Portrait2By3,
}

/// Optional first-frame input for video generation.
#[derive(Clone, PartialEq, Eq)]
pub enum XaiVideoImage {
    Url(String),
    FileId(XaiFileId),
}

impl fmt::Debug for XaiVideoImage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Url(_) => formatter.debug_tuple("Url").field(&"<redacted>").finish(),
            Self::FileId(id) => formatter.debug_tuple("FileId").field(id).finish(),
        }
    }
}

/// Typed creation request for one xAI video job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XaiVideoCreateRequest {
    model: ModelId,
    prompt: String,
    duration_seconds: Option<u8>,
    aspect_ratio: Option<XaiVideoAspectRatio>,
    resolution: Option<XaiVideoResolution>,
    image: Option<XaiVideoImage>,
}

impl XaiVideoCreateRequest {
    pub fn new(model: impl Into<String>, prompt: impl Into<String>) -> Result<Self, Error> {
        let model = ModelId::new(model.into()).map_err(|source| {
            Error::new(ErrorKind::InvalidInput, "xAI video model ID is invalid").with_source(source)
        })?;
        let prompt = prompt.into();
        validate_text(&prompt, MAX_PROMPT_BYTES, "xAI video prompt is invalid")?;
        Ok(Self {
            model,
            prompt,
            duration_seconds: None,
            aspect_ratio: None,
            resolution: None,
            image: None,
        })
    }

    pub fn with_duration_seconds(mut self, seconds: u8) -> Result<Self, Error> {
        if seconds == 0 || seconds > 15 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "xAI video duration must be between one and 15 seconds",
            ));
        }
        self.duration_seconds = Some(seconds);
        Ok(self)
    }

    pub const fn with_aspect_ratio(mut self, aspect_ratio: XaiVideoAspectRatio) -> Self {
        self.aspect_ratio = Some(aspect_ratio);
        self
    }

    pub const fn with_resolution(mut self, resolution: XaiVideoResolution) -> Self {
        self.resolution = Some(resolution);
        self
    }

    pub fn with_image_url(mut self, url: impl Into<String>) -> Result<Self, Error> {
        let url = url.into();
        validate_remote_url(&url)?;
        self.image = Some(XaiVideoImage::Url(url));
        Ok(self)
    }

    pub fn with_image_file(mut self, file_id: XaiFileId) -> Self {
        self.image = Some(XaiVideoImage::FileId(file_id));
        self
    }
}

/// Validated video request identity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct XaiVideoJobId(String);

impl XaiVideoJobId {
    pub fn new(value: impl Into<String>) -> Result<Self, Error> {
        let value = value.into();
        if value.is_empty()
            || value.len() > 256
            || value.chars().any(|character| {
                !character.is_ascii_alphanumeric() && !matches!(character, '-' | '_')
            })
        {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "xAI video request ID is invalid",
            ));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Created xAI video job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XaiVideoJob {
    pub id: XaiVideoJobId,
}

/// Completed video artifact. The generated URL is redacted from `Debug`.
#[derive(Clone, PartialEq)]
pub struct XaiVideoArtifact {
    pub url: String,
    pub duration_seconds: Option<f64>,
    pub respect_moderation: Option<bool>,
}

impl fmt::Debug for XaiVideoArtifact {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("XaiVideoArtifact")
            .field("url", &"<redacted>")
            .field("duration_seconds", &self.duration_seconds)
            .field("respect_moderation", &self.respect_moderation)
            .finish()
    }
}

/// Current provider-owned xAI video job state.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum XaiVideoJobState {
    Pending {
        progress: Option<f64>,
    },
    Completed {
        video: XaiVideoArtifact,
        progress: Option<f64>,
        cost_in_usd_ticks: Option<u64>,
    },
    Failed {
        code: Option<String>,
    },
    Expired,
}

/// Provider-owned xAI Videos job resource.
#[derive(Clone)]
pub struct XaiVideoJobs {
    transport: ProviderTransport,
}

impl XaiVideoJobs {
    pub(crate) fn new(transport: ProviderTransport) -> Self {
        Self { transport }
    }

    pub async fn create(
        &self,
        request: XaiVideoCreateRequest,
        call: CallOptions,
    ) -> Result<XaiVideoJob, Error> {
        let plan = create_plan(request)?;
        let response = self.transport.execute(plan, call).await?;
        if !response.status().is_success() {
            return Err(response_error("xAI video creation failed", response));
        }
        let (_, _, body) = response.into_parts();
        let wire: CreateResponseWire = serde_json::from_slice(&body).map_err(|source| {
            Error::new(
                ErrorKind::Protocol,
                "xAI returned malformed video creation JSON",
            )
            .with_source(source)
        })?;
        Ok(XaiVideoJob {
            id: XaiVideoJobId::new(wire.request_id)?,
        })
    }

    pub async fn get(
        &self,
        id: &XaiVideoJobId,
        call: CallOptions,
    ) -> Result<XaiVideoJobState, Error> {
        let plan = RequestPlan::new(
            Method::GET,
            RequestTarget::new(format!("videos/{}", id.as_str())).map_err(request_build_error)?,
        )
        .with_replay_safety(ReplaySafety::SemanticallyIdempotent)
        .map_err(request_build_error)?;
        let response = self.transport.execute(plan, call).await?;
        if !response.status().is_success() {
            return Err(response_error("xAI video status request failed", response));
        }
        decode_state(response)
    }
}

impl fmt::Debug for XaiVideoJobs {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("XaiVideoJobs")
            .field("transport", &"shared")
            .finish()
    }
}

fn create_plan(request: XaiVideoCreateRequest) -> Result<RequestPlan, Error> {
    let mut body = Map::from_iter([
        (
            "model".to_string(),
            Value::String(request.model.to_string()),
        ),
        ("prompt".to_string(), Value::String(request.prompt)),
    ]);
    if let Some(duration) = request.duration_seconds {
        body.insert("duration".to_string(), Value::from(duration));
    }
    if let Some(aspect_ratio) = request.aspect_ratio {
        body.insert(
            "aspect_ratio".to_string(),
            Value::String(
                match aspect_ratio {
                    XaiVideoAspectRatio::Landscape16By9 => "16:9",
                    XaiVideoAspectRatio::Portrait9By16 => "9:16",
                    XaiVideoAspectRatio::Square => "1:1",
                    XaiVideoAspectRatio::Standard4By3 => "4:3",
                    XaiVideoAspectRatio::Portrait3By4 => "3:4",
                    XaiVideoAspectRatio::Landscape3By2 => "3:2",
                    XaiVideoAspectRatio::Portrait2By3 => "2:3",
                }
                .to_string(),
            ),
        );
    }
    if let Some(resolution) = request.resolution {
        body.insert(
            "resolution".to_string(),
            Value::String(
                match resolution {
                    XaiVideoResolution::P480 => "480p",
                    XaiVideoResolution::P720 => "720p",
                    XaiVideoResolution::P1080 => "1080p",
                }
                .to_string(),
            ),
        );
    }
    if let Some(image) = request.image {
        body.insert(
            "image".to_string(),
            match image {
                XaiVideoImage::Url(url) => serde_json::json!({"url":url}),
                XaiVideoImage::FileId(id) => serde_json::json!({"file_id":id.as_str()}),
            },
        );
    }
    RequestPlan::new(
        Method::POST,
        RequestTarget::new(CREATE_TARGET).map_err(request_build_error)?,
    )
    .with_body(RequestBody::json(&Value::Object(body)).map_err(request_build_error)?)
    .with_replay_safety(ReplaySafety::Never)
    .map_err(request_build_error)
}

#[derive(Debug, Deserialize)]
struct CreateResponseWire {
    request_id: String,
}

#[derive(Debug, Deserialize)]
struct StatusResponseWire {
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    video: Option<VideoWire>,
    #[serde(default)]
    usage: Option<UsageWire>,
    #[serde(default)]
    progress: Option<f64>,
    #[serde(default)]
    error: Option<ErrorWire>,
}

#[derive(Debug, Deserialize)]
struct VideoWire {
    url: String,
    #[serde(default)]
    duration: Option<f64>,
    #[serde(default)]
    respect_moderation: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct UsageWire {
    #[serde(default)]
    cost_in_usd_ticks: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct ErrorWire {
    #[serde(default)]
    code: Option<String>,
}

fn decode_state(response: TransportResponse) -> Result<XaiVideoJobState, Error> {
    let status_code = response.status();
    let (_, _, body) = response.into_parts();
    if status_code == StatusCode::ACCEPTED && body.iter().all(u8::is_ascii_whitespace) {
        return Ok(XaiVideoJobState::Pending { progress: None });
    }
    let wire: StatusResponseWire = serde_json::from_slice(&body).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "xAI returned malformed video status JSON",
        )
        .with_source(source)
    })?;
    if wire
        .progress
        .is_some_and(|progress| !progress.is_finite() || progress < 0.0)
    {
        return Err(Error::protocol_violation(
            "xAI video status contains invalid progress",
        ));
    }
    match wire.status.as_deref() {
        Some("pending") | None if wire.video.is_none() => Ok(XaiVideoJobState::Pending {
            progress: wire.progress,
        }),
        Some("done") | None if wire.video.is_some() => {
            let video = wire.video.expect("matched present video");
            validate_remote_url(&video.url)?;
            if video
                .duration
                .is_some_and(|duration| !duration.is_finite() || duration < 0.0)
            {
                return Err(Error::protocol_violation(
                    "xAI video status contains invalid duration",
                ));
            }
            Ok(XaiVideoJobState::Completed {
                video: XaiVideoArtifact {
                    url: video.url,
                    duration_seconds: video.duration,
                    respect_moderation: video.respect_moderation,
                },
                progress: wire.progress,
                cost_in_usd_ticks: wire.usage.and_then(|usage| usage.cost_in_usd_ticks),
            })
        }
        Some("failed") => Ok(XaiVideoJobState::Failed {
            code: wire
                .error
                .and_then(|error| error.code)
                .and_then(public_code),
        }),
        Some("expired") => Ok(XaiVideoJobState::Expired),
        _ => Err(Error::protocol_violation(
            "xAI video status contains an unknown terminal state",
        )),
    }
}

fn public_code(value: String) -> Option<String> {
    PublicDiagnosticText::new(value.clone()).ok().map(|_| value)
}

fn validate_text(value: &str, maximum: usize, message: &'static str) -> Result<(), Error> {
    if value.trim().is_empty() || value.len() > maximum || value.chars().any(char::is_control) {
        return Err(Error::new(ErrorKind::InvalidInput, message));
    }
    Ok(())
}

fn validate_remote_url(value: &str) -> Result<(), Error> {
    if value.len() > 4_096 || value.chars().any(char::is_control) {
        return Err(Error::new(
            ErrorKind::Protocol,
            "xAI video URL is invalid or too long",
        ));
    }
    let url = url::Url::parse(value).map_err(|source| {
        Error::new(ErrorKind::Protocol, "xAI video URL is invalid").with_source(source)
    })?;
    if url.scheme() != "https"
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
    {
        return Err(Error::protocol_violation(
            "xAI video URL must be HTTPS without embedded credentials",
        ));
    }
    Ok(())
}

fn response_error(message: &'static str, response: TransportResponse) -> Error {
    let (status, headers, body) = response.into_parts();
    let kind = match status {
        StatusCode::UNAUTHORIZED => ErrorKind::Authentication,
        StatusCode::FORBIDDEN => ErrorKind::Authorization,
        StatusCode::TOO_MANY_REQUESTS => ErrorKind::RateLimited,
        StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY => ErrorKind::InvalidInput,
        _ => ErrorKind::Provider,
    };
    let mut diagnostics = ResponseDiagnostics::default().with_status(status.as_u16());
    if let Some(request_id) = response_request_id(&headers)
        && let Ok(request_id) = PublicDiagnosticText::new(request_id)
    {
        diagnostics = diagnostics.with_request_id(request_id);
    }
    let raw_headers = headers
        .expose()
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.as_str().to_string(), value.to_string()))
        })
        .collect();
    Error::new(kind, message)
        .with_diagnostics(diagnostics)
        .with_sensitive_response(SensitiveResponse::new(raw_headers, body.to_vec()))
}

fn response_request_id(headers: &ResponseHeaders) -> Option<String> {
    ["x-request-id", "request-id"].into_iter().find_map(|name| {
        headers
            .get(&http::header::HeaderName::from_static(name))
            .and_then(|value| value.to_str().ok())
            .and_then(|value| PublicDiagnosticText::new(value.to_owned()).ok())
            .map(|value| value.as_str().to_owned())
    })
}

fn request_build_error(source: RequestBuildError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "xAI video request violates the transport contract",
    )
    .with_source(source)
}

#[cfg(test)]
mod tests {
    use mockito::Matcher;
    use siumai_core::{ReplayDomain, ReplayDomainId};
    use siumai_transport::EndpointConfig;

    use super::*;
    use crate::{XaiCredential, XaiProvider};

    #[tokio::test]
    async fn create_and_retrieve_typed_video_job() {
        let mut server = mockito::Server::new_async().await;
        let create = server
            .mock("POST", "/v1/videos/generations")
            .match_body(Matcher::Json(serde_json::json!({
                "model":"grok-imagine-video-1.5",
                "prompt":"a crab walking through tokyo",
                "duration":5,
                "aspect_ratio":"4:3",
                "resolution":"1080p"
            })))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(serde_json::json!({"request_id":"video_job_1"}).to_string())
            .create_async()
            .await;
        let status = server
            .mock("GET", "/v1/videos/video_job_1")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "status":"done",
                    "progress":1.0,
                    "video":{
                        "url":"https://vidgen.example.com/video.mp4",
                        "duration":5.0,
                        "respect_moderation":true
                    },
                    "usage":{"cost_in_usd_ticks":25}
                })
                .to_string(),
            )
            .create_async()
            .await;
        let provider = XaiProvider::builder(XaiCredential::unauthenticated())
            .with_endpoint(EndpointConfig::local_explicit(format!("{}/v1", server.url())).unwrap())
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("test-xai-video").unwrap(),
            ))
            .build()
            .unwrap();
        let request =
            XaiVideoCreateRequest::new("grok-imagine-video-1.5", "a crab walking through tokyo")
                .unwrap()
                .with_duration_seconds(5)
                .unwrap()
                .with_aspect_ratio(XaiVideoAspectRatio::Standard4By3)
                .with_resolution(XaiVideoResolution::P1080);
        let job = provider
            .video_jobs()
            .create(request, CallOptions::default())
            .await
            .unwrap();
        let state = provider
            .video_jobs()
            .get(&job.id, CallOptions::default())
            .await
            .unwrap();

        create.assert_async().await;
        status.assert_async().await;
        let XaiVideoJobState::Completed { video, .. } = state else {
            panic!("expected completed video");
        };
        assert_eq!(video.duration_seconds, Some(5.0));
        assert!(!format!("{video:?}").contains("vidgen.example.com"));
    }
}
