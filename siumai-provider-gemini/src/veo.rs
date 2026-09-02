use std::fmt;
use std::sync::Arc;

use base64::Engine as _;
use bytes::Bytes;
use http::Method;
use http::header::{ACCEPT, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use siumai_core::{
    CallOptions, Error, ErrorContext, ErrorKind, ModelId, ProviderProvenance, ProviderScope,
};
use siumai_transport::{
    ReplaySafety, RequestBody, RequestBuildError, RequestHeaders, RequestPlan, RequestTarget,
};

use crate::http::response_error;
use crate::provider::ProviderRuntime;

/// Current full-quality Veo 3.1 preview model hint.
pub const VEO_3_1_GENERATE_PREVIEW: &str = "veo-3.1-generate-preview";
/// Current low-latency Veo 3.1 preview model hint.
pub const VEO_3_1_FAST_GENERATE_PREVIEW: &str = "veo-3.1-fast-generate-preview";
/// Current cost-efficient Veo 3.1 preview model hint.
pub const VEO_3_1_LITE_GENERATE_PREVIEW: &str = "veo-3.1-lite-generate-preview";

pub(crate) const VEO_PROTOCOL_ID: &str = "gemini-veo";
pub(crate) const VEO_API_MODE_ID: &str = "predict-long-running-v1beta";

const MAX_PROMPT_BYTES: usize = 1024 * 1024;
const MAX_OPERATION_NAME_BYTES: usize = 4 * 1024;
const MAX_URI_BYTES: usize = 16 * 1024;
const MAX_METADATA_BYTES: usize = 64 * 1024;
const MAX_FILTER_REASONS: usize = 32;
const MAX_FILTER_REASON_BYTES: usize = 4 * 1024;

/// Inline image input accepted by the bounded Veo text/image slice.
#[derive(Clone, PartialEq, Eq)]
pub struct GeminiVeoImage {
    mime_type: String,
    bytes: Bytes,
}

impl GeminiVeoImage {
    pub fn new(mime_type: impl Into<String>, bytes: impl Into<Bytes>) -> Result<Self, Error> {
        let mime_type = mime_type.into();
        let bytes = bytes.into();
        if !matches!(
            mime_type.to_ascii_lowercase().as_str(),
            "image/png" | "image/jpeg" | "image/webp"
        ) || bytes.is_empty()
        {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Gemini Veo image must contain non-empty PNG, JPEG, or WebP bytes",
            ));
        }
        Ok(Self { mime_type, bytes })
    }

    pub fn mime_type(&self) -> &str {
        &self.mime_type
    }

    pub fn bytes(&self) -> &Bytes {
        &self.bytes
    }
}

impl fmt::Debug for GeminiVeoImage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GeminiVeoImage")
            .field("mime_type", &self.mime_type)
            .field("bytes", &self.bytes.len())
            .finish()
    }
}

/// Mutually exclusive input modes supported by the initial Veo job API.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GeminiVeoSource {
    Text {
        prompt: String,
    },
    Image {
        prompt: Option<String>,
        first_frame: GeminiVeoImage,
        last_frame: Option<GeminiVeoImage>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum GeminiVeoAspectRatio {
    #[serde(rename = "16:9")]
    Landscape,
    #[serde(rename = "9:16")]
    Portrait,
}

impl GeminiVeoAspectRatio {
    const fn as_wire(self) -> &'static str {
        match self {
            Self::Landscape => "16:9",
            Self::Portrait => "9:16",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum GeminiVeoResolution {
    #[serde(rename = "720p")]
    Hd720,
    #[serde(rename = "1080p")]
    Hd1080,
    #[serde(rename = "4k")]
    FourK,
}

impl GeminiVeoResolution {
    const fn as_wire(self) -> &'static str {
        match self {
            Self::Hd720 => "720p",
            Self::Hd1080 => "1080p",
            Self::FourK => "4k",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum GeminiVeoPersonGeneration {
    AllowAll,
    AllowAdult,
    DontAllow,
}

impl GeminiVeoPersonGeneration {
    const fn as_wire(self) -> &'static str {
        match self {
            Self::AllowAll => "allow_all",
            Self::AllowAdult => "allow_adult",
            Self::DontAllow => "dont_allow",
        }
    }
}

/// Typed Veo request configuration for the documented text/image subset.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GeminiVeoConfig {
    aspect_ratio: Option<GeminiVeoAspectRatio>,
    resolution: Option<GeminiVeoResolution>,
    duration_seconds: Option<u8>,
    person_generation: Option<GeminiVeoPersonGeneration>,
    negative_prompt: Option<String>,
    enhance_prompt: Option<bool>,
}

impl GeminiVeoConfig {
    pub const fn new() -> Self {
        Self {
            aspect_ratio: None,
            resolution: None,
            duration_seconds: None,
            person_generation: None,
            negative_prompt: None,
            enhance_prompt: None,
        }
    }

    pub const fn with_aspect_ratio(mut self, aspect_ratio: GeminiVeoAspectRatio) -> Self {
        self.aspect_ratio = Some(aspect_ratio);
        self
    }

    pub const fn with_resolution(mut self, resolution: GeminiVeoResolution) -> Self {
        self.resolution = Some(resolution);
        self
    }

    pub fn with_duration_seconds(mut self, duration_seconds: u8) -> Result<Self, Error> {
        if !matches!(duration_seconds, 4 | 6 | 8) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Gemini Veo duration must be 4, 6, or 8 seconds",
            ));
        }
        self.duration_seconds = Some(duration_seconds);
        Ok(self)
    }

    pub const fn with_person_generation(
        mut self,
        person_generation: GeminiVeoPersonGeneration,
    ) -> Self {
        self.person_generation = Some(person_generation);
        self
    }

    pub fn with_negative_prompt(mut self, prompt: impl Into<String>) -> Result<Self, Error> {
        let prompt = prompt.into();
        validate_optional_prompt(&prompt)?;
        self.negative_prompt = Some(prompt);
        Ok(self)
    }

    pub const fn with_enhance_prompt(mut self, enhance_prompt: bool) -> Self {
        self.enhance_prompt = Some(enhance_prompt);
        self
    }
}

/// One typed Veo long-running request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeminiVeoRequest {
    model: ModelId,
    source: GeminiVeoSource,
    config: GeminiVeoConfig,
}

impl GeminiVeoRequest {
    pub fn text(model: ModelId, prompt: impl Into<String>) -> Result<Self, Error> {
        let prompt = prompt.into();
        validate_prompt(&prompt)?;
        Ok(Self {
            model,
            source: GeminiVeoSource::Text { prompt },
            config: GeminiVeoConfig::default(),
        })
    }

    pub fn image(
        model: ModelId,
        prompt: Option<String>,
        first_frame: GeminiVeoImage,
        last_frame: Option<GeminiVeoImage>,
    ) -> Result<Self, Error> {
        if let Some(prompt) = &prompt {
            validate_optional_prompt(prompt)?;
        }
        Ok(Self {
            model,
            source: GeminiVeoSource::Image {
                prompt,
                first_frame,
                last_frame,
            },
            config: GeminiVeoConfig::default(),
        })
    }

    pub fn with_config(mut self, config: GeminiVeoConfig) -> Result<Self, Error> {
        validate_config(&self.source, &config)?;
        self.config = config;
        Ok(self)
    }

    pub fn model(&self) -> &ModelId {
        &self.model
    }

    pub fn source(&self) -> &GeminiVeoSource {
        &self.source
    }

    pub fn config(&self) -> &GeminiVeoConfig {
        &self.config
    }
}

/// Replay-bound reference to one Veo long-running operation.
#[derive(Clone, PartialEq, Eq)]
pub struct GeminiVeoOperationRef {
    name: String,
    provenance: ProviderProvenance,
}

impl GeminiVeoOperationRef {
    pub fn from_parts(
        scope: &ProviderScope,
        model: ModelId,
        name: impl Into<String>,
    ) -> Result<Self, Error> {
        let name = name.into();
        validate_operation_name(&name, &model)?;
        let provenance = ProviderProvenance::from_scope(scope, model).map_err(|source| {
            Error::new(
                ErrorKind::Configuration,
                "Gemini Veo operation provenance is invalid",
            )
            .with_source(source)
        })?;
        Ok(Self { name, provenance })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn provenance(&self) -> &ProviderProvenance {
        &self.provenance
    }
}

impl fmt::Debug for GeminiVeoOperationRef {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GeminiVeoOperationRef")
            .field("name", &"[REDACTED]")
            .field("provenance", &self.provenance)
            .finish()
    }
}

/// One potentially signed generated-video URI.
#[derive(Clone, PartialEq, Eq)]
pub struct GeminiGeneratedVideoUri(String);

impl GeminiGeneratedVideoUri {
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for GeminiGeneratedVideoUri {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("GeminiGeneratedVideoUri([REDACTED])")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeminiGeneratedVideo {
    pub uri: GeminiGeneratedVideoUri,
    pub media_type: String,
}

#[derive(Clone, PartialEq, Eq)]
pub struct GeminiVeoResult {
    pub videos: Vec<GeminiGeneratedVideo>,
    pub filtered_count: Option<u32>,
    pub filtered_reasons: Vec<String>,
}

impl fmt::Debug for GeminiVeoResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GeminiVeoResult")
            .field("videos", &self.videos)
            .field("filtered_count", &self.filtered_count)
            .field("filtered_reason_count", &self.filtered_reasons.len())
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeminiVeoFailure {
    pub code: Option<i32>,
    pub details_present: bool,
}

#[derive(Clone, PartialEq)]
pub struct GeminiVeoOperationMetadata {
    value: Value,
}

impl GeminiVeoOperationMetadata {
    pub fn value(&self) -> &Value {
        &self.value
    }
}

impl fmt::Debug for GeminiVeoOperationMetadata {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("GeminiVeoOperationMetadata([REDACTED])")
    }
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum GeminiVeoOperationState {
    Pending,
    Succeeded(GeminiVeoResult),
    Failed(GeminiVeoFailure),
}

#[derive(Debug, Clone, PartialEq)]
pub struct GeminiVeoOperation {
    pub reference: GeminiVeoOperationRef,
    pub state: GeminiVeoOperationState,
    pub metadata: Option<GeminiVeoOperationMetadata>,
}

/// Shared Veo 3.1 submit/status handle.
#[derive(Clone)]
pub struct GeminiVeo {
    runtime: Arc<ProviderRuntime>,
}

impl GeminiVeo {
    pub(crate) fn new(runtime: Arc<ProviderRuntime>) -> Self {
        Self { runtime }
    }

    pub async fn submit(&self, request: GeminiVeoRequest) -> Result<GeminiVeoOperation, Error> {
        self.submit_with_options(request, CallOptions::default())
            .await
    }

    pub async fn submit_with_options(
        &self,
        request: GeminiVeoRequest,
        options: CallOptions,
    ) -> Result<GeminiVeoOperation, Error> {
        reject_provider_options(&options)?;
        validate_config(request.source(), request.config())?;
        let body = encode_request(&request)?;
        let target = RequestTarget::new(format!(
            "v1beta/models/{}:predictLongRunning",
            urlencoding::encode(request.model().as_str())
        ))
        .map_err(request_error)?;
        let wire: OperationWire = self
            .execute_json(
                Method::POST,
                target,
                RequestBody::json(&body).map_err(request_error)?,
                ReplaySafety::Never,
                options,
            )
            .await?;
        decode_operation(wire, &self.runtime.veo_scope, request.model(), None)
    }

    pub async fn get(
        &self,
        operation: &GeminiVeoOperationRef,
    ) -> Result<GeminiVeoOperation, Error> {
        self.get_with_options(operation, CallOptions::default())
            .await
    }

    pub async fn get_with_options(
        &self,
        operation: &GeminiVeoOperationRef,
        options: CallOptions,
    ) -> Result<GeminiVeoOperation, Error> {
        reject_provider_options(&options)?;
        if !operation
            .provenance()
            .matches_replay_target(&self.runtime.veo_scope)
        {
            return Err(self.contextualize(Error::new(
                ErrorKind::InvalidInput,
                "Gemini Veo operation belongs to another replay domain",
            )));
        }
        let target =
            RequestTarget::new(format!("v1beta/{}", operation.name())).map_err(request_error)?;
        let wire: OperationWire = self
            .execute_json(
                Method::GET,
                target,
                RequestBody::Empty,
                ReplaySafety::SemanticallyIdempotent,
                options,
            )
            .await?;
        decode_operation(
            wire,
            &self.runtime.veo_scope,
            operation.provenance().model(),
            Some(operation.name()),
        )
    }

    async fn execute_json<T>(
        &self,
        method: Method,
        target: RequestTarget,
        body: RequestBody,
        replay_safety: ReplaySafety,
        options: CallOptions,
    ) -> Result<T, Error>
    where
        T: for<'de> Deserialize<'de>,
    {
        let headers = RequestHeaders::new()
            .try_insert(ACCEPT, HeaderValue::from_static("application/json"))
            .map_err(request_error)?;
        let plan = RequestPlan::new(method, target)
            .with_headers(headers)
            .with_body(body)
            .with_replay_safety(replay_safety)
            .map_err(request_error)?;
        let response = self
            .runtime
            .transport
            .execute(plan, options)
            .await
            .map_err(|error| self.contextualize(error))?;
        if !response.status().is_success() {
            return Err(self.contextualize(response_error(
                response,
                "Gemini rejected the Veo job request",
            )));
        }
        let (_, _, body) = response.into_parts();
        serde_json::from_slice(&body).map_err(|source| {
            self.contextualize(
                Error::new(
                    ErrorKind::Protocol,
                    "Gemini returned malformed Veo job JSON",
                )
                .with_source(source),
            )
        })
    }

    fn contextualize(&self, error: Error) -> Error {
        error.with_context(ErrorContext {
            operation: None,
            provider: Some(self.runtime.veo_scope.provider_id().clone()),
            route: None,
            model: None,
        })
    }
}

impl fmt::Debug for GeminiVeo {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GeminiVeo")
            .field("runtime", &"shared")
            .finish()
    }
}

fn encode_request(request: &GeminiVeoRequest) -> Result<Value, Error> {
    let mut instance = serde_json::Map::new();
    match request.source() {
        GeminiVeoSource::Text { prompt } => {
            instance.insert("prompt".to_string(), Value::String(prompt.clone()));
        }
        GeminiVeoSource::Image {
            prompt,
            first_frame,
            last_frame,
        } => {
            if let Some(prompt) = prompt {
                instance.insert("prompt".to_string(), Value::String(prompt.clone()));
            }
            instance.insert("image".to_string(), encode_image(first_frame));
            if let Some(last_frame) = last_frame {
                instance.insert("lastFrame".to_string(), encode_image(last_frame));
            }
        }
    }

    let config = request.config();
    let mut parameters = serde_json::Map::new();
    if let Some(value) = config.aspect_ratio {
        parameters.insert(
            "aspectRatio".to_string(),
            Value::String(value.as_wire().to_string()),
        );
    }
    if let Some(value) = config.resolution {
        parameters.insert(
            "resolution".to_string(),
            Value::String(value.as_wire().to_string()),
        );
    }
    if let Some(value) = config.duration_seconds {
        parameters.insert("durationSeconds".to_string(), Value::from(value));
    }
    if let Some(value) = config.person_generation {
        parameters.insert(
            "personGeneration".to_string(),
            Value::String(value.as_wire().to_string()),
        );
    }
    if let Some(value) = &config.negative_prompt {
        parameters.insert("negativePrompt".to_string(), Value::String(value.clone()));
    }
    if let Some(value) = config.enhance_prompt {
        parameters.insert("enhancePrompt".to_string(), Value::Bool(value));
    }

    Ok(serde_json::json!({
        "instances": [Value::Object(instance)],
        "parameters": Value::Object(parameters)
    }))
}

fn encode_image(image: &GeminiVeoImage) -> Value {
    serde_json::json!({
        "inlineData": {
            "mimeType": image.mime_type(),
            "data": base64::engine::general_purpose::STANDARD.encode(image.bytes())
        }
    })
}

fn decode_operation(
    wire: OperationWire,
    scope: &ProviderScope,
    model: &ModelId,
    expected_name: Option<&str>,
) -> Result<GeminiVeoOperation, Error> {
    if expected_name.is_some_and(|expected| expected != wire.name) {
        return Err(Error::protocol_violation(
            "Gemini Veo returned a different operation name",
        ));
    }
    let reference = GeminiVeoOperationRef::from_parts(scope, model.clone(), wire.name)?;
    let metadata = wire
        .metadata
        .map(|value| {
            let size = serde_json::to_vec(&value)
                .map_err(|source| {
                    Error::protocol_violation("Gemini Veo metadata could not be measured")
                        .with_source(source)
                })?
                .len();
            if size > MAX_METADATA_BYTES {
                return Err(Error::new(
                    ErrorKind::ResponseLimit,
                    "Gemini Veo operation metadata exceeded the response limit",
                ));
            }
            Ok(GeminiVeoOperationMetadata { value })
        })
        .transpose()?;

    let state = match (wire.done.unwrap_or(false), wire.response, wire.error) {
        (false, None, None) => GeminiVeoOperationState::Pending,
        (false, _, _) => {
            return Err(Error::protocol_violation(
                "pending Gemini Veo operation contained a terminal payload",
            ));
        }
        (true, Some(response), None) => {
            GeminiVeoOperationState::Succeeded(decode_result(response)?)
        }
        (true, None, Some(error)) => GeminiVeoOperationState::Failed(GeminiVeoFailure {
            code: error.code,
            details_present: error.details.is_some_and(|details| !details.is_empty()),
        }),
        (true, None, None) => {
            return Err(Error::protocol_violation(
                "completed Gemini Veo operation omitted its terminal payload",
            ));
        }
        (true, Some(_), Some(_)) => {
            return Err(Error::protocol_violation(
                "completed Gemini Veo operation contained both response and error",
            ));
        }
    };
    Ok(GeminiVeoOperation {
        reference,
        state,
        metadata,
    })
}

fn decode_result(response: OperationResponseWire) -> Result<GeminiVeoResult, Error> {
    let response = response.generate_video_response.ok_or_else(|| {
        Error::protocol_violation("Gemini Veo operation omitted generateVideoResponse")
    })?;
    let videos = response
        .generated_samples
        .into_iter()
        .map(|sample| {
            let video = sample.video.ok_or_else(|| {
                Error::protocol_violation("Gemini Veo generated sample omitted video")
            })?;
            let uri = video.uri.ok_or_else(|| {
                Error::protocol_violation("Gemini Veo generated video omitted URI")
            })?;
            if uri.is_empty() || uri.len() > MAX_URI_BYTES || uri.chars().any(char::is_control) {
                return Err(Error::protocol_violation(
                    "Gemini Veo generated video URI is invalid",
                ));
            }
            let media_type = video.encoding.unwrap_or_else(|| "video/mp4".to_string());
            if !media_type.to_ascii_lowercase().starts_with("video/")
                || media_type.len() > 256
                || media_type.chars().any(char::is_control)
            {
                return Err(Error::protocol_violation(
                    "Gemini Veo generated video encoding is invalid",
                ));
            }
            Ok(GeminiGeneratedVideo {
                uri: GeminiGeneratedVideoUri(uri),
                media_type,
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    if response.rai_media_filtered_reasons.len() > MAX_FILTER_REASONS
        || response.rai_media_filtered_reasons.iter().any(|reason| {
            reason.is_empty()
                || reason.len() > MAX_FILTER_REASON_BYTES
                || reason.chars().any(char::is_control)
        })
    {
        return Err(Error::new(
            ErrorKind::ResponseLimit,
            "Gemini Veo filter diagnostics exceeded the response limit",
        ));
    }
    if videos.is_empty() && response.rai_media_filtered_count.unwrap_or(0) == 0 {
        return Err(Error::protocol_violation(
            "Gemini Veo succeeded without video or filtering outcome",
        ));
    }
    Ok(GeminiVeoResult {
        videos,
        filtered_count: response.rai_media_filtered_count,
        filtered_reasons: response.rai_media_filtered_reasons,
    })
}

fn validate_config(source: &GeminiVeoSource, config: &GeminiVeoConfig) -> Result<(), Error> {
    if matches!(
        config.resolution,
        Some(GeminiVeoResolution::Hd1080 | GeminiVeoResolution::FourK)
    ) && config
        .duration_seconds
        .is_some_and(|duration| duration != 8)
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Gemini Veo 1080p and 4K output requires an 8-second duration",
        ));
    }
    if matches!(
        source,
        GeminiVeoSource::Image {
            last_frame: Some(_),
            ..
        }
    ) && config
        .duration_seconds
        .is_some_and(|duration| duration != 8)
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Gemini Veo frame interpolation requires an 8-second duration",
        ));
    }
    Ok(())
}

fn validate_prompt(prompt: &str) -> Result<(), Error> {
    if prompt.trim().is_empty()
        || prompt.len() > MAX_PROMPT_BYTES
        || prompt.chars().any(|character| character == '\0')
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Gemini Veo prompt is invalid",
        ));
    }
    Ok(())
}

fn validate_optional_prompt(prompt: &str) -> Result<(), Error> {
    validate_prompt(prompt)
}

fn validate_operation_name(name: &str, model: &ModelId) -> Result<(), Error> {
    if name.is_empty()
        || name.len() > MAX_OPERATION_NAME_BYTES
        || name.chars().any(char::is_control)
        || name.contains(['?', '#', '\\'])
    {
        return Err(Error::protocol_violation(
            "Gemini Veo returned an invalid operation name",
        ));
    }
    let expected_prefix = format!("models/{}/operations/", model.as_str());
    let Some(operation_id) = name.strip_prefix(&expected_prefix) else {
        return Err(Error::protocol_violation(
            "Gemini Veo operation does not match the requested model",
        ));
    };
    if operation_id.is_empty() || operation_id.contains('/') {
        return Err(Error::protocol_violation(
            "Gemini Veo returned an invalid operation identifier",
        ));
    }
    Ok(())
}

fn reject_provider_options(options: &CallOptions) -> Result<(), Error> {
    if options.has_provider_options() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Gemini Veo options must be expressed by GeminiVeoRequest",
        ));
    }
    Ok(())
}

fn request_error(source: RequestBuildError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "Gemini Veo request violates the transport contract",
    )
    .with_source(source)
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct OperationWire {
    name: String,
    #[serde(default)]
    metadata: Option<Value>,
    #[serde(default)]
    done: Option<bool>,
    #[serde(default)]
    error: Option<OperationErrorWire>,
    #[serde(default)]
    response: Option<OperationResponseWire>,
}

#[derive(Deserialize)]
struct OperationErrorWire {
    #[serde(default)]
    code: Option<i32>,
    #[serde(default)]
    details: Option<Vec<Value>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct OperationResponseWire {
    #[serde(default)]
    generate_video_response: Option<GenerateVideoResponseWire>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GenerateVideoResponseWire {
    #[serde(default)]
    generated_samples: Vec<GeneratedSampleWire>,
    #[serde(default)]
    rai_media_filtered_count: Option<u32>,
    #[serde(default)]
    rai_media_filtered_reasons: Vec<String>,
}

#[derive(Deserialize)]
struct GeneratedSampleWire {
    #[serde(default)]
    video: Option<GeneratedVideoWire>,
}

#[derive(Deserialize)]
struct GeneratedVideoWire {
    #[serde(default)]
    uri: Option<String>,
    #[serde(default)]
    encoding: Option<String>,
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use siumai_core::{
        ApiModeId, PlatformId, ProtocolId, ProviderId, ReplayDomain, ReplayDomainId,
    };
    use siumai_transport::EndpointConfig;

    use super::*;
    use crate::{GeminiCredential, GeminiProvider};

    fn provider(base_url: String) -> GeminiProvider {
        GeminiProvider::builder(GeminiCredential::api_key("test-key"))
            .with_endpoint(EndpointConfig::local_explicit(base_url).unwrap())
            .with_replay_domain(ReplayDomain::custom(
                ReplayDomainId::new("gemini-veo-test").unwrap(),
            ))
            .build()
            .unwrap()
    }

    fn scope() -> ProviderScope {
        ProviderScope::new(ProviderId::new("google").unwrap())
            .with_platform(PlatformId::new("gemini-api").unwrap())
            .with_protocol(ProtocolId::new(VEO_PROTOCOL_ID).unwrap())
            .with_api_mode(ApiModeId::new(VEO_API_MODE_ID).unwrap())
            .with_replay_domain(ReplayDomain::official(
                ReplayDomainId::new("google-gemini-api").unwrap(),
            ))
    }

    #[test]
    fn text_request_has_one_typed_instance() {
        let request = GeminiVeoRequest::text(
            ModelId::new(VEO_3_1_GENERATE_PREVIEW).unwrap(),
            "A paper boat on a river",
        )
        .unwrap()
        .with_config(
            GeminiVeoConfig::new()
                .with_aspect_ratio(GeminiVeoAspectRatio::Landscape)
                .with_resolution(GeminiVeoResolution::Hd1080)
                .with_duration_seconds(8)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            encode_request(&request).unwrap(),
            json!({
                "instances": [{"prompt": "A paper boat on a river"}],
                "parameters": {
                    "aspectRatio": "16:9",
                    "resolution": "1080p",
                    "durationSeconds": 8
                }
            })
        );
    }

    #[test]
    fn structural_duration_and_frame_relationships_fail_closed() {
        assert!(GeminiVeoConfig::new().with_duration_seconds(5).is_err());

        let duration_error = GeminiVeoRequest::text(
            ModelId::new("private-veo-next").expect("future model ID"),
            "A paper boat",
        )
        .expect("Veo request")
        .with_config(
            GeminiVeoConfig::new()
                .with_resolution(GeminiVeoResolution::FourK)
                .with_duration_seconds(6)
                .expect("supported duration value"),
        )
        .unwrap_err();
        assert_eq!(duration_error.kind(), ErrorKind::InvalidInput);

        let first_frame = GeminiVeoImage::new("image/png", vec![1_u8]).expect("first frame");
        let last_frame = GeminiVeoImage::new("image/png", vec![2_u8]).expect("last frame");
        let interpolation_error = GeminiVeoRequest::image(
            ModelId::new("private-veo-next").expect("future model ID"),
            None,
            first_frame,
            Some(last_frame),
        )
        .expect("Veo image request")
        .with_config(
            GeminiVeoConfig::new()
                .with_duration_seconds(6)
                .expect("supported duration value"),
        )
        .unwrap_err();
        assert_eq!(interpolation_error.kind(), ErrorKind::InvalidInput);
    }

    #[tokio::test]
    async fn known_and_future_model_configs_reach_the_exact_veo_wire() {
        const FUTURE_MODEL: &str = "private-veo-next";

        let mut server = mockito::Server::new_async().await;
        let known_operation = format!("models/{VEO_3_1_LITE_GENERATE_PREVIEW}/operations/known-op");
        let known = server
            .mock(
                "POST",
                "/v1beta/models/veo-3.1-lite-generate-preview:predictLongRunning",
            )
            .match_body(mockito::Matcher::Json(json!({
                "instances": [{"prompt": "A paper boat"}],
                "parameters": {
                    "aspectRatio": "9:16",
                    "resolution": "4k",
                    "durationSeconds": 8
                }
            })))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(json!({"name": known_operation, "done": false}).to_string())
            .expect(1)
            .create_async()
            .await;
        let future_operation = format!("models/{FUTURE_MODEL}/operations/future-op");
        let future = server
            .mock("POST", "/v1beta/models/private-veo-next:predictLongRunning")
            .match_body(mockito::Matcher::Json(json!({
                "instances": [{"prompt": "A paper boat"}],
                "parameters": {
                    "resolution": "1080p",
                    "durationSeconds": 8,
                    "enhancePrompt": true
                }
            })))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(json!({"name": future_operation, "done": false}).to_string())
            .expect(1)
            .create_async()
            .await;
        let veo = provider(server.url()).veo();

        let known_request = GeminiVeoRequest::text(
            ModelId::new(VEO_3_1_LITE_GENERATE_PREVIEW).expect("known model ID"),
            "A paper boat",
        )
        .expect("known-model request")
        .with_config(
            GeminiVeoConfig::new()
                .with_aspect_ratio(GeminiVeoAspectRatio::Portrait)
                .with_resolution(GeminiVeoResolution::FourK)
                .with_duration_seconds(8)
                .expect("supported duration"),
        )
        .expect("known-model config");
        let known_result = veo.submit(known_request).await.expect("known-model submit");
        assert!(matches!(
            known_result.state,
            GeminiVeoOperationState::Pending
        ));

        let future_request = GeminiVeoRequest::text(
            ModelId::new(FUTURE_MODEL).expect("future model ID"),
            "A paper boat",
        )
        .expect("future-model request")
        .with_config(
            GeminiVeoConfig::new()
                .with_resolution(GeminiVeoResolution::Hd1080)
                .with_duration_seconds(8)
                .expect("supported duration")
                .with_enhance_prompt(true),
        )
        .expect("future-model config");
        let future_result = veo
            .submit(future_request)
            .await
            .expect("future-model submit");
        assert!(matches!(
            future_result.state,
            GeminiVeoOperationState::Pending
        ));

        known.assert_async().await;
        future.assert_async().await;
    }

    #[test]
    fn operation_states_preserve_success_filtering_and_failure() {
        let model = ModelId::new(VEO_3_1_GENERATE_PREVIEW).unwrap();
        let name = format!("models/{}/operations/op-1", model.as_str());
        let success = serde_json::from_value::<OperationWire>(json!({
            "name": name.clone(),
            "done": true,
            "response": {
                "generateVideoResponse": {
                    "generatedSamples": [{
                        "video": {"uri": "https://example.invalid/video?secret=sentinel", "encoding": "video/mp4"}
                    }]
                }
            }
        }))
        .unwrap();
        let success = decode_operation(success, &scope(), &model, None).unwrap();
        let GeminiVeoOperationState::Succeeded(result) = success.state else {
            panic!("expected success");
        };
        assert_eq!(result.videos.len(), 1);
        assert!(!format!("{result:?}").contains("secret=sentinel"));

        let filtered = serde_json::from_value::<OperationWire>(json!({
            "name": name.clone(),
            "done": true,
            "response": {
                "generateVideoResponse": {
                    "generatedSamples": [],
                    "raiMediaFilteredCount": 1,
                    "raiMediaFilteredReasons": ["safety"]
                }
            }
        }))
        .unwrap();
        let filtered = decode_operation(filtered, &scope(), &model, None).unwrap();
        assert!(matches!(
            filtered.state,
            GeminiVeoOperationState::Succeeded(_)
        ));

        let failed = serde_json::from_value::<OperationWire>(json!({
            "name": name,
            "done": true,
            "error": {"code": 8, "message": "sentinel", "details": [{"secret": "sentinel"}]}
        }))
        .unwrap();
        let failed = decode_operation(failed, &scope(), &model, None).unwrap();
        assert!(!format!("{failed:?}").contains("sentinel"));
    }

    #[test]
    fn operation_terminal_invariants_fail_closed() {
        let model = ModelId::new(VEO_3_1_GENERATE_PREVIEW).unwrap();
        let name = format!("models/{}/operations/op-1", model.as_str());
        for value in [
            json!({"name": name.clone(), "done": true}),
            json!({
                "name": name.clone(),
                "done": true,
                "response": {},
                "error": {"code": 13}
            }),
            json!({"name": name, "done": false, "response": {}}),
        ] {
            let wire = serde_json::from_value::<OperationWire>(value).unwrap();
            assert!(decode_operation(wire, &scope(), &model, None).is_err());
        }
    }

    #[tokio::test]
    async fn submit_and_get_preserve_the_typed_operation_lifecycle() {
        let mut server = mockito::Server::new_async().await;
        let operation_name = format!("models/{VEO_3_1_GENERATE_PREVIEW}/operations/op-1");
        let submit = server
            .mock(
                "POST",
                "/v1beta/models/veo-3.1-generate-preview:predictLongRunning",
            )
            .match_header("x-goog-api-key", "test-key")
            .match_body(mockito::Matcher::Regex(
                "\\\"prompt\\\":\\\"A paper boat\\\"".to_string(),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(json!({"name": operation_name, "done": false}).to_string())
            .create_async()
            .await;
        let get = server
            .mock(
                "GET",
                "/v1beta/models/veo-3.1-generate-preview/operations/op-1",
            )
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                json!({
                    "name": operation_name,
                    "done": true,
                    "response": {
                        "generateVideoResponse": {
                            "generatedSamples": [{
                                "video": {
                                    "uri": "https://example.invalid/video",
                                    "encoding": "video/mp4"
                                }
                            }]
                        }
                    }
                })
                .to_string(),
            )
            .create_async()
            .await;
        let veo = provider(server.url()).veo();
        let request = GeminiVeoRequest::text(
            ModelId::new(VEO_3_1_GENERATE_PREVIEW).unwrap(),
            "A paper boat",
        )
        .unwrap();

        let pending = veo.submit(request).await.unwrap();
        assert!(matches!(pending.state, GeminiVeoOperationState::Pending));
        let completed = veo.get(&pending.reference).await.unwrap();
        assert!(matches!(
            completed.state,
            GeminiVeoOperationState::Succeeded(_)
        ));

        submit.assert_async().await;
        get.assert_async().await;
    }
}
