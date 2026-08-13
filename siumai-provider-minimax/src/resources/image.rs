use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use http::Method;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use siumai_core::{CallOptions, Error, ErrorKind};
use siumai_transport::{ReplaySafety, RequestBody};

use crate::models::image::{IMAGE_01, IMAGE_01_LIVE};

use super::common::{BaseResponse, NativeResponseEnvelope, NativeRuntime, execute_json, target};

const IMAGE_GENERATION_TARGET: &str = "v1/image_generation";
const MAX_PROMPT_CHARACTERS: usize = 1_500;

/// A MiniMax image aspect ratio with its documented output dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MinimaxImageAspectRatio {
    Square,
    Landscape16By9,
    Landscape4By3,
    Landscape3By2,
    Portrait2By3,
    Portrait3By4,
    Portrait9By16,
    Ultrawide21By9,
}

impl MinimaxImageAspectRatio {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Square => "1:1",
            Self::Landscape16By9 => "16:9",
            Self::Landscape4By3 => "4:3",
            Self::Landscape3By2 => "3:2",
            Self::Portrait2By3 => "2:3",
            Self::Portrait3By4 => "3:4",
            Self::Portrait9By16 => "9:16",
            Self::Ultrawide21By9 => "21:9",
        }
    }
}

/// Validated explicit dimensions for the `image-01` model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MinimaxImageDimensions {
    width: u16,
    height: u16,
}

impl MinimaxImageDimensions {
    /// Create dimensions in the documented inclusive `512..=2048` range.
    ///
    /// Both dimensions must be divisible by eight.
    pub fn new(width: u16, height: u16) -> Result<Self, Error> {
        validate_dimension(width)?;
        validate_dimension(height)?;
        Ok(Self { width, height })
    }

    pub const fn width(self) -> u16 {
        self.width
    }

    pub const fn height(self) -> u16 {
        self.height
    }
}

/// Image sizing mode. The enum prevents sending dimensions that MiniMax would
/// silently ignore in favor of an aspect ratio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MinimaxImageSize {
    AspectRatio(MinimaxImageAspectRatio),
    Dimensions(MinimaxImageDimensions),
}

/// Requested image representation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MinimaxImageResponseFormat {
    #[default]
    Url,
    Base64,
}

impl MinimaxImageResponseFormat {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Url => "url",
            Self::Base64 => "base64",
        }
    }
}

/// A character reference for MiniMax image-to-image generation.
///
/// The reference may be a public image URL or a base64 data URL. Its value is
/// intentionally omitted from `Debug` because it may contain signed URLs or
/// large private image data.
#[derive(Clone, Serialize)]
pub struct MinimaxImageSubjectReference {
    #[serde(rename = "type")]
    subject_type: MinimaxImageSubjectType,
    image_file: String,
}

impl MinimaxImageSubjectReference {
    pub fn character(image_file: impl Into<String>) -> Result<Self, Error> {
        let image_file = image_file.into();
        validate_non_empty(
            &image_file,
            "MiniMax image subject reference must not be empty",
        )?;
        Ok(Self {
            subject_type: MinimaxImageSubjectType::Character,
            image_file,
        })
    }

    /// Explicitly expose the reference URL or data URL.
    pub fn image_file(&self) -> &str {
        &self.image_file
    }
}

impl fmt::Debug for MinimaxImageSubjectReference {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxImageSubjectReference")
            .field("subject_type", &self.subject_type)
            .field("image_file", &"redacted")
            .field("image_file_bytes", &self.image_file.len())
            .finish()
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum MinimaxImageSubjectType {
    Character,
}

/// Typed request for MiniMax's provider-native Image Generation API.
#[derive(Clone)]
pub struct MinimaxImageRequest {
    model: String,
    prompt: String,
    size: Option<MinimaxImageSize>,
    response_format: MinimaxImageResponseFormat,
    seed: Option<i64>,
    count: u8,
    prompt_optimizer: bool,
    subject_references: Vec<MinimaxImageSubjectReference>,
}

impl MinimaxImageRequest {
    pub fn new(model: impl Into<String>, prompt: impl Into<String>) -> Result<Self, Error> {
        let request = Self {
            model: model.into(),
            prompt: prompt.into(),
            size: None,
            response_format: MinimaxImageResponseFormat::default(),
            seed: None,
            count: 1,
            prompt_optimizer: false,
            subject_references: Vec::new(),
        };
        request.validate_basics()?;
        Ok(request)
    }

    pub fn with_size(mut self, size: MinimaxImageSize) -> Result<Self, Error> {
        self.size = Some(size);
        self.validate_size_policy()?;
        Ok(self)
    }

    pub fn with_aspect_ratio(mut self, aspect_ratio: MinimaxImageAspectRatio) -> Self {
        self.size = Some(MinimaxImageSize::AspectRatio(aspect_ratio));
        self
    }

    pub fn with_dimensions(self, dimensions: MinimaxImageDimensions) -> Result<Self, Error> {
        self.with_size(MinimaxImageSize::Dimensions(dimensions))
    }

    pub fn with_response_format(mut self, response_format: MinimaxImageResponseFormat) -> Self {
        self.response_format = response_format;
        self
    }

    pub fn with_seed(mut self, seed: i64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn with_count(mut self, count: u8) -> Result<Self, Error> {
        validate_count(count)?;
        self.count = count;
        Ok(self)
    }

    pub fn with_prompt_optimizer(mut self, enabled: bool) -> Self {
        self.prompt_optimizer = enabled;
        self
    }

    pub fn with_subject_reference(mut self, reference: MinimaxImageSubjectReference) -> Self {
        self.subject_references.push(reference);
        self
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    pub const fn size(&self) -> Option<MinimaxImageSize> {
        self.size
    }

    pub const fn response_format(&self) -> MinimaxImageResponseFormat {
        self.response_format
    }

    pub const fn seed(&self) -> Option<i64> {
        self.seed
    }

    pub const fn count(&self) -> u8 {
        self.count
    }

    pub const fn prompt_optimizer(&self) -> bool {
        self.prompt_optimizer
    }

    pub fn subject_references(&self) -> &[MinimaxImageSubjectReference] {
        &self.subject_references
    }

    fn validate(&self) -> Result<(), Error> {
        self.validate_basics()?;
        self.validate_size_policy()?;
        if self.model == IMAGE_01_LIVE && self.subject_references.is_empty() {
            return Err(invalid_input(
                "MiniMax image-01-live requires a subject reference",
            ));
        }
        Ok(())
    }

    fn validate_basics(&self) -> Result<(), Error> {
        validate_non_empty(&self.model, "MiniMax image model must not be empty")?;
        validate_character_count(
            &self.prompt,
            1,
            MAX_PROMPT_CHARACTERS,
            "MiniMax image prompt must contain between 1 and 1500 characters",
        )?;
        validate_count(self.count)
    }

    fn validate_size_policy(&self) -> Result<(), Error> {
        if self.model != IMAGE_01 && matches!(self.size, Some(MinimaxImageSize::Dimensions(_))) {
            return Err(invalid_input(
                "MiniMax explicit dimensions are supported only by image-01",
            ));
        }
        Ok(())
    }

    fn wire(&self) -> ImageRequestWire<'_> {
        let (aspect_ratio, width, height) = match self.size {
            Some(MinimaxImageSize::AspectRatio(ratio)) => (Some(ratio.as_str()), None, None),
            Some(MinimaxImageSize::Dimensions(dimensions)) => {
                (None, Some(dimensions.width()), Some(dimensions.height()))
            }
            None => (None, None, None),
        };
        ImageRequestWire {
            model: &self.model,
            prompt: &self.prompt,
            subject_reference: &self.subject_references,
            aspect_ratio,
            width,
            height,
            response_format: self.response_format.as_str(),
            seed: self.seed,
            count: self.count,
            prompt_optimizer: self.prompt_optimizer,
        }
    }
}

impl fmt::Debug for MinimaxImageRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxImageRequest")
            .field("model", &self.model)
            .field("prompt_characters", &self.prompt.chars().count())
            .field("size", &self.size)
            .field("response_format", &self.response_format)
            .field("seed", &self.seed)
            .field("count", &self.count)
            .field("prompt_optimizer", &self.prompt_optimizer)
            .field("subject_reference_count", &self.subject_references.len())
            .finish()
    }
}

#[derive(Serialize)]
struct ImageRequestWire<'a> {
    model: &'a str,
    prompt: &'a str,
    #[serde(skip_serializing_if = "subject_references_are_empty")]
    subject_reference: &'a [MinimaxImageSubjectReference],
    #[serde(skip_serializing_if = "Option::is_none")]
    aspect_ratio: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    width: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    height: Option<u16>,
    response_format: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
    #[serde(rename = "n")]
    count: u8,
    prompt_optimizer: bool,
}

/// Counts reported for a MiniMax image-generation response.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MinimaxImageMetadata {
    success_count: Option<u32>,
    failed_count: Option<u32>,
}

impl MinimaxImageMetadata {
    pub const fn success_count(self) -> Option<u32> {
        self.success_count
    }

    pub const fn failed_count(self) -> Option<u32> {
        self.failed_count
    }
}

/// A completed MiniMax image-generation response.
#[derive(Clone)]
pub struct MinimaxImageGeneration {
    id: Option<String>,
    urls: Vec<String>,
    base64_images: Vec<String>,
    metadata: MinimaxImageMetadata,
    extra: BTreeMap<String, Value>,
    data_extra: BTreeMap<String, Value>,
}

impl MinimaxImageGeneration {
    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    /// Explicitly expose generated URLs. MiniMax URLs may expire after 24
    /// hours and may contain signed query parameters.
    pub fn urls(&self) -> &[String] {
        &self.urls
    }

    /// Explicitly expose base64-encoded images.
    pub fn base64_images(&self) -> &[String] {
        &self.base64_images
    }

    pub const fn metadata(&self) -> MinimaxImageMetadata {
        self.metadata
    }

    pub fn extra(&self) -> &BTreeMap<String, Value> {
        &self.extra
    }

    pub fn data_extra(&self) -> &BTreeMap<String, Value> {
        &self.data_extra
    }

    pub fn into_urls(self) -> Vec<String> {
        self.urls
    }

    pub fn into_base64_images(self) -> Vec<String> {
        self.base64_images
    }
}

impl fmt::Debug for MinimaxImageGeneration {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxImageGeneration")
            .field("has_id", &self.id.is_some())
            .field("url_count", &self.urls.len())
            .field("base64_image_count", &self.base64_images.len())
            .field("metadata", &self.metadata)
            .field("extra_field_count", &self.extra.len())
            .field("data_extra_field_count", &self.data_extra.len())
            .finish()
    }
}

/// Shared, lightweight handle for MiniMax's provider-native Image API.
#[derive(Clone)]
pub struct MinimaxImages {
    runtime: Arc<NativeRuntime>,
}

impl MinimaxImages {
    pub(crate) fn new(runtime: Arc<NativeRuntime>) -> Self {
        Self { runtime }
    }

    pub async fn generate(
        &self,
        request: MinimaxImageRequest,
    ) -> Result<MinimaxImageGeneration, Error> {
        self.generate_with_options(request, CallOptions::default())
            .await
    }

    pub async fn generate_with_options(
        &self,
        request: MinimaxImageRequest,
        options: CallOptions,
    ) -> Result<MinimaxImageGeneration, Error> {
        request.validate()?;
        let expected_format = request.response_format;
        let body = RequestBody::json(&request.wire()).map_err(|source| {
            Error::new(ErrorKind::InvalidInput, "MiniMax image request is invalid")
                .with_source(source)
        })?;
        let response: ImageResponseEnvelope = execute_json(
            &self.runtime,
            Method::POST,
            target(IMAGE_GENERATION_TARGET)?,
            body,
            ReplaySafety::Never,
            options,
        )
        .await?;
        response.into_generation(expected_format)
    }
}

impl fmt::Debug for MinimaxImages {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimaxImages")
            .field("runtime", &"shared")
            .finish()
    }
}

#[derive(Deserialize)]
struct ImageResponseEnvelope {
    #[serde(default)]
    data: Option<ImageDataWire>,
    #[serde(default)]
    metadata: Option<ImageMetadataWire>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    base_resp: Option<BaseResponse>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

impl ImageResponseEnvelope {
    fn into_generation(
        self,
        expected_format: MinimaxImageResponseFormat,
    ) -> Result<MinimaxImageGeneration, Error> {
        let data = self.data.ok_or_else(|| {
            Error::new(
                ErrorKind::ProtocolViolation,
                "MiniMax image response omitted generation data",
            )
        })?;
        let metadata = self
            .metadata
            .map_or_else(MinimaxImageMetadata::default, |metadata| {
                MinimaxImageMetadata {
                    success_count: metadata.success_count,
                    failed_count: metadata.failed_count,
                }
            });
        let actual_count = match expected_format {
            MinimaxImageResponseFormat::Url => {
                if !data.image_base64.is_empty() {
                    return Err(Error::new(
                        ErrorKind::ProtocolViolation,
                        "MiniMax image response returned an unexpected representation",
                    ));
                }
                data.image_urls.len()
            }
            MinimaxImageResponseFormat::Base64 => {
                if !data.image_urls.is_empty() {
                    return Err(Error::new(
                        ErrorKind::ProtocolViolation,
                        "MiniMax image response returned an unexpected representation",
                    ));
                }
                data.image_base64.len()
            }
        };
        if metadata
            .success_count
            .is_some_and(|reported| reported as usize != actual_count)
        {
            return Err(Error::new(
                ErrorKind::ProtocolViolation,
                "MiniMax image response success count did not match its output",
            ));
        }
        Ok(MinimaxImageGeneration {
            id: self.id,
            urls: data.image_urls,
            base64_images: data.image_base64,
            metadata,
            extra: self.extra,
            data_extra: data.extra,
        })
    }
}

impl NativeResponseEnvelope for ImageResponseEnvelope {
    fn base_response(&self) -> Option<&BaseResponse> {
        self.base_resp.as_ref()
    }
}

#[derive(Deserialize)]
struct ImageDataWire {
    #[serde(default)]
    image_urls: Vec<String>,
    #[serde(default)]
    image_base64: Vec<String>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

#[derive(Deserialize)]
struct ImageMetadataWire {
    #[serde(default, deserialize_with = "deserialize_optional_u32")]
    success_count: Option<u32>,
    #[serde(default, deserialize_with = "deserialize_optional_u32")]
    failed_count: Option<u32>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum WireU32 {
    Number(u32),
    Decimal(String),
}

fn deserialize_optional_u32<'de, D>(deserializer: D) -> Result<Option<u32>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<WireU32>::deserialize(deserializer)?;
    value
        .map(|value| match value {
            WireU32::Number(value) => Ok(value),
            WireU32::Decimal(value) => value.parse::<u32>().map_err(serde::de::Error::custom),
        })
        .transpose()
}

fn validate_dimension(value: u16) -> Result<(), Error> {
    if !(512..=2048).contains(&value) || !value.is_multiple_of(8) {
        return Err(invalid_input(
            "MiniMax image dimensions must be between 512 and 2048 and divisible by 8",
        ));
    }
    Ok(())
}

fn validate_count(count: u8) -> Result<(), Error> {
    if !(1..=9).contains(&count) {
        return Err(invalid_input("MiniMax image count must be between 1 and 9"));
    }
    Ok(())
}

fn subject_references_are_empty(value: &&[MinimaxImageSubjectReference]) -> bool {
    value.is_empty()
}

fn validate_character_count(
    value: &str,
    minimum: usize,
    maximum: usize,
    message: &'static str,
) -> Result<(), Error> {
    let count = value.chars().count();
    if !(minimum..=maximum).contains(&count) {
        return Err(invalid_input(message));
    }
    Ok(())
}

fn validate_non_empty(value: &str, message: &'static str) -> Result<(), Error> {
    if value.trim().is_empty() {
        return Err(invalid_input(message));
    }
    Ok(())
}

fn invalid_input(message: &'static str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::common::validate_base_response;

    #[test]
    fn image_request_encodes_the_native_wire_contract() {
        let request = MinimaxImageRequest::new(IMAGE_01, "private image prompt")
            .expect("request should be valid")
            .with_dimensions(
                MinimaxImageDimensions::new(1_024, 768).expect("dimensions should be valid"),
            )
            .expect("image-01 should support dimensions")
            .with_response_format(MinimaxImageResponseFormat::Base64)
            .with_seed(-42)
            .with_count(2)
            .expect("count should be valid")
            .with_prompt_optimizer(true)
            .with_subject_reference(
                MinimaxImageSubjectReference::character(
                    "https://example.invalid/private.png?token=secret",
                )
                .expect("reference should be valid"),
            );

        assert_eq!(
            serde_json::to_value(request.wire()).expect("request should serialize"),
            serde_json::json!({
                "model": "image-01",
                "prompt": "private image prompt",
                "subject_reference": [{
                    "type": "character",
                    "image_file": "https://example.invalid/private.png?token=secret"
                }],
                "width": 1024,
                "height": 768,
                "response_format": "base64",
                "seed": -42,
                "n": 2,
                "prompt_optimizer": true
            })
        );
    }

    #[test]
    fn image_validation_is_operation_and_model_specific() {
        let live = MinimaxImageRequest::new(IMAGE_01_LIVE, "animate this portrait")
            .expect("a staged request should be constructible");
        assert_eq!(
            live.validate()
                .expect_err("live model must require a subject")
                .kind(),
            ErrorKind::InvalidInput
        );

        let live = live.with_subject_reference(
            MinimaxImageSubjectReference::character("data:image/png;base64,cHJpdmF0ZQ==")
                .expect("reference should be valid"),
        );
        live.validate()
            .expect("live request with a subject should be valid");

        let future = MinimaxImageRequest::new("image-future", "future baseline request")
            .expect("future model ids should remain open");
        assert_eq!(
            future
                .with_dimensions(
                    MinimaxImageDimensions::new(512, 512).expect("dimensions should be valid"),
                )
                .expect_err("unverified models must not inherit image-01 dimensions")
                .kind(),
            ErrorKind::InvalidInput
        );

        assert_eq!(
            MinimaxImageRequest::new(IMAGE_01, "valid prompt")
                .expect("request should be valid")
                .with_count(0)
                .expect_err("zero images must fail")
                .kind(),
            ErrorKind::InvalidInput
        );
    }

    #[test]
    fn image_response_decodes_counts_and_preserves_unknown_fields() {
        let response: ImageResponseEnvelope = serde_json::from_value(serde_json::json!({
            "id": "private-trace-id",
            "data": {
                "image_urls": ["https://example.invalid/generated.png?token=secret"],
                "future_data": {"nested": true}
            },
            "metadata": {
                "success_count": "1",
                "failed_count": 0
            },
            "future_response": [1, 2, 3],
            "base_resp": {"status_code": 0, "status_msg": "success"}
        }))
        .expect("response should decode");
        validate_base_response(response.base_response()).expect("base response should succeed");

        let generation = response
            .into_generation(MinimaxImageResponseFormat::Url)
            .expect("generation should decode");
        assert_eq!(generation.id(), Some("private-trace-id"));
        assert_eq!(generation.urls().len(), 1);
        assert_eq!(generation.metadata().success_count(), Some(1));
        assert_eq!(generation.metadata().failed_count(), Some(0));
        assert!(generation.extra().contains_key("future_response"));
        assert!(generation.data_extra().contains_key("future_data"));
    }

    #[test]
    fn image_response_rejects_the_wrong_representation() {
        let response: ImageResponseEnvelope = serde_json::from_value(serde_json::json!({
            "data": {"image_base64": ["cHJpdmF0ZQ=="]},
            "metadata": {"success_count": 1},
            "base_resp": {"status_code": 0, "status_msg": "success"}
        }))
        .expect("response should decode");

        assert_eq!(
            response
                .into_generation(MinimaxImageResponseFormat::Url)
                .expect_err("unexpected representation must fail")
                .kind(),
            ErrorKind::ProtocolViolation
        );
    }

    #[test]
    fn image_debug_output_redacts_prompts_references_and_generated_content() {
        let request = MinimaxImageRequest::new(IMAGE_01, "SUPER_SECRET_IMAGE_PROMPT")
            .expect("request should be valid")
            .with_subject_reference(
                MinimaxImageSubjectReference::character(
                    "https://example.invalid/private.png?token=SUPER_SECRET_TOKEN",
                )
                .expect("reference should be valid"),
            );
        let request_debug = format!("{request:?}");
        assert!(!request_debug.contains("SUPER_SECRET_IMAGE_PROMPT"));
        assert!(!request_debug.contains("SUPER_SECRET_TOKEN"));

        let response: ImageResponseEnvelope = serde_json::from_value(serde_json::json!({
            "id": "SUPER_SECRET_TRACE",
            "data": {
                "image_urls": [
                    "https://example.invalid/generated.png?token=SUPER_SECRET_OUTPUT"
                ]
            },
            "metadata": {"success_count": 1},
            "future_secret": "SUPER_SECRET_EXTRA",
            "base_resp": {"status_code": 0, "status_msg": "success"}
        }))
        .expect("response should decode");
        let generation = response
            .into_generation(MinimaxImageResponseFormat::Url)
            .expect("generation should decode");
        let response_debug = format!("{generation:?}");
        assert!(!response_debug.contains("SUPER_SECRET_TRACE"));
        assert!(!response_debug.contains("SUPER_SECRET_OUTPUT"));
        assert!(!response_debug.contains("SUPER_SECRET_EXTRA"));
    }
}
