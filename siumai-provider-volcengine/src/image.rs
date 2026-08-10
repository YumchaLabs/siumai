//! ARK image generation and the provider-neutral image adapter.

use std::collections::BTreeMap;
use std::fmt;

use async_trait::async_trait;
use base64::Engine as _;
use bytes::Bytes;
use http::Method;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use siumai_core::{
    CallOptions, Error, ErrorContext, ErrorKind, ImageArtifact, ImageLimits, ImageModel,
    ImageRequest, ImageResponse, MediaData, Model, ModelDescriptor, ModelFamily, ModelId,
    ModelOperation, ProviderOptionContext, ProviderOptionError, ProviderOptionLayers,
    ProviderOptionMerger, ProviderOptionOrigin, ProviderOptions, ResponseMetadata,
    TypedProviderOptions, Usage,
};
use siumai_transport::{ReplaySafety, RequestBody};

use crate::native::{SharedArkNativeRuntime, execute_json, target};

pub const ARK_IMAGE_API_MODE: &str = "images-generations";
const MAX_PROMPT_BYTES: usize = 64 * 1024;
const MAX_IMAGE_INPUTS: usize = 32;
const MAX_IMAGE_SOURCE_BYTES: usize = 16 * 1024 * 1024;
const MAX_IMAGES_PER_PORTABLE_CALL: u32 = 1;

/// Output file format accepted by current Seedream image APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum ArkImageOutputFormat {
    Png,
    Jpeg,
}

impl ArkImageOutputFormat {
    const fn media_type(self) -> &'static str {
        match self {
            Self::Png => "image/png",
            Self::Jpeg => "image/jpeg",
        }
    }
}

/// Provider response representation for ARK image generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ArkImageResponseFormat {
    Url,
    B64Json,
}

/// Whether Seedream should generate a related sequence rather than one independent image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum ArkSequentialImageGeneration {
    Auto,
    Disabled,
}

/// Provider-owned prompt optimization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum ArkOptimizePromptMode {
    Standard,
    Fast,
}

/// One image reference accepted by the native Seedream request.
#[derive(Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ArkImageInput {
    Url(String),
    DataUri(String),
}

impl ArkImageInput {
    pub fn url(value: impl Into<String>) -> Result<Self, Error> {
        let value = value.into();
        validate_image_source(&value, true)?;
        Ok(Self::Url(value))
    }

    pub fn data_uri(value: impl Into<String>) -> Result<Self, Error> {
        let value = value.into();
        validate_image_source(&value, false)?;
        Ok(Self::DataUri(value))
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Url(value) | Self::DataUri(value) => value,
        }
    }
}

impl fmt::Debug for ArkImageInput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArkImageInput")
            .field(
                "kind",
                &match self {
                    Self::Url(_) => "url",
                    Self::DataUri(_) => "data_uri",
                },
            )
            .field("encoded_bytes", &self.as_str().len())
            .finish()
    }
}

/// Full provider-owned ARK image request.
#[derive(Clone)]
pub struct ArkImageRequest {
    model: ModelId,
    prompt: String,
    inputs: Vec<ArkImageInput>,
    size: Option<String>,
    watermark: Option<bool>,
    output_format: Option<ArkImageOutputFormat>,
    response_format: Option<ArkImageResponseFormat>,
    sequential_image_generation: Option<ArkSequentialImageGeneration>,
    max_images: Option<u32>,
    optimize_prompt_mode: Option<ArkOptimizePromptMode>,
}

impl ArkImageRequest {
    pub fn new(model: impl Into<String>, prompt: impl Into<String>) -> Result<Self, Error> {
        let request = Self {
            model: ModelId::new(model.into()).map_err(invalid_model)?,
            prompt: prompt.into(),
            inputs: Vec::new(),
            size: None,
            watermark: None,
            output_format: None,
            response_format: None,
            sequential_image_generation: None,
            max_images: None,
            optimize_prompt_mode: None,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn with_input(mut self, input: ArkImageInput) -> Result<Self, Error> {
        self.inputs.push(input);
        self.validate()?;
        Ok(self)
    }

    pub fn with_size(mut self, size: impl Into<String>) -> Result<Self, Error> {
        self.size = Some(size.into());
        self.validate()?;
        Ok(self)
    }

    pub const fn with_watermark(mut self, watermark: bool) -> Self {
        self.watermark = Some(watermark);
        self
    }

    pub const fn with_output_format(mut self, output_format: ArkImageOutputFormat) -> Self {
        self.output_format = Some(output_format);
        self
    }

    pub const fn with_response_format(mut self, response_format: ArkImageResponseFormat) -> Self {
        self.response_format = Some(response_format);
        self
    }

    pub fn with_sequential_generation(
        mut self,
        mode: ArkSequentialImageGeneration,
        max_images: Option<u32>,
    ) -> Result<Self, Error> {
        self.sequential_image_generation = Some(mode);
        self.max_images = max_images;
        self.validate()?;
        Ok(self)
    }

    pub const fn with_optimize_prompt_mode(mut self, mode: ArkOptimizePromptMode) -> Self {
        self.optimize_prompt_mode = Some(mode);
        self
    }

    pub fn model(&self) -> &ModelId {
        &self.model
    }

    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    pub fn inputs(&self) -> &[ArkImageInput] {
        &self.inputs
    }

    fn validate(&self) -> Result<(), Error> {
        if self.prompt.trim().is_empty()
            || self.prompt != self.prompt.trim()
            || self.prompt.len() > MAX_PROMPT_BYTES
            || self.prompt.chars().any(char::is_control)
        {
            return Err(invalid("ARK image prompt is invalid"));
        }
        if self.inputs.len() > MAX_IMAGE_INPUTS {
            return Err(Error::new(
                ErrorKind::LimitExceeded,
                "ARK image request contains too many references",
            ));
        }
        for input in &self.inputs {
            validate_image_source(input.as_str(), matches!(input, ArkImageInput::Url(_)))?;
        }
        if let Some(size) = self.size.as_deref()
            && (size.trim().is_empty()
                || size != size.trim()
                || size.len() > 64
                || size.chars().any(char::is_control))
        {
            return Err(invalid("ARK image size is invalid"));
        }
        if self
            .max_images
            .is_some_and(|value| value == 0 || value > 64)
        {
            return Err(invalid("ARK sequential image maximum is invalid"));
        }
        if self.max_images.is_some()
            && self.sequential_image_generation != Some(ArkSequentialImageGeneration::Auto)
        {
            return Err(invalid(
                "ARK max_images requires sequential image generation in auto mode",
            ));
        }
        Ok(())
    }

    fn wire(&self) -> Value {
        let mut body = serde_json::Map::new();
        body.insert("model".to_string(), Value::String(self.model.to_string()));
        body.insert("prompt".to_string(), Value::String(self.prompt.clone()));
        if self.inputs.len() == 1 {
            body.insert(
                "image".to_string(),
                Value::String(self.inputs[0].as_str().to_string()),
            );
        } else if !self.inputs.is_empty() {
            body.insert(
                "image".to_string(),
                Value::Array(
                    self.inputs
                        .iter()
                        .map(|input| Value::String(input.as_str().to_string()))
                        .collect(),
                ),
            );
        }
        if let Some(size) = &self.size {
            body.insert("size".to_string(), Value::String(size.clone()));
        }
        insert_value(&mut body, "watermark", self.watermark);
        insert_value(&mut body, "output_format", self.output_format);
        insert_value(&mut body, "response_format", self.response_format);
        insert_value(
            &mut body,
            "sequential_image_generation",
            self.sequential_image_generation,
        );
        if let Some(max_images) = self.max_images {
            body.insert(
                "sequential_image_generation_options".to_string(),
                serde_json::json!({"max_images": max_images}),
            );
        }
        if let Some(mode) = self.optimize_prompt_mode {
            body.insert(
                "optimize_prompt_options".to_string(),
                serde_json::json!({"mode": mode}),
            );
        }
        Value::Object(body)
    }
}

impl fmt::Debug for ArkImageRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArkImageRequest")
            .field("model", &self.model)
            .field("prompt_bytes", &self.prompt.len())
            .field("inputs", &self.inputs)
            .field("size", &self.size)
            .field("watermark", &self.watermark)
            .field("output_format", &self.output_format)
            .field("response_format", &self.response_format)
            .field(
                "sequential_image_generation",
                &self.sequential_image_generation,
            )
            .field("max_images", &self.max_images)
            .field("optimize_prompt_mode", &self.optimize_prompt_mode)
            .finish()
    }
}

/// One image returned by ARK. URL and base64 payloads are redacted from `Debug`.
#[derive(Clone, PartialEq, Deserialize)]
pub struct ArkGeneratedImage {
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    b64_json: Option<String>,
    #[serde(default)]
    size: Option<String>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

impl ArkGeneratedImage {
    pub fn url(&self) -> Option<&str> {
        self.url.as_deref()
    }

    pub fn base64_json(&self) -> Option<&str> {
        self.b64_json.as_deref()
    }

    pub fn size(&self) -> Option<&str> {
        self.size.as_deref()
    }

    pub fn extra(&self) -> &BTreeMap<String, Value> {
        &self.extra
    }
}

impl fmt::Debug for ArkGeneratedImage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArkGeneratedImage")
            .field("has_url", &self.url.is_some())
            .field("base64_bytes", &self.b64_json.as_ref().map(String::len))
            .field("size", &self.size)
            .field("extra_fields", &self.extra.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// Native ARK image response.
#[derive(Clone, PartialEq, Deserialize)]
pub struct ArkImageResponse {
    #[serde(default)]
    pub created: Option<u64>,
    #[serde(default)]
    pub data: Vec<ArkGeneratedImage>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl fmt::Debug for ArkImageResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArkImageResponse")
            .field("created", &self.created)
            .field("data", &self.data)
            .field("extra_fields", &self.extra.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// Shared provider-native ARK Images client.
#[derive(Clone)]
pub struct ArkImages {
    runtime: SharedArkNativeRuntime,
}

impl ArkImages {
    pub(crate) fn new(runtime: SharedArkNativeRuntime) -> Self {
        Self { runtime }
    }

    pub async fn generate(&self, request: ArkImageRequest) -> Result<ArkImageResponse, Error> {
        self.generate_with_options(request, CallOptions::default())
            .await
    }

    pub async fn generate_with_options(
        &self,
        request: ArkImageRequest,
        options: CallOptions,
    ) -> Result<ArkImageResponse, Error> {
        request.validate()?;
        execute_json(
            &self.runtime,
            Method::POST,
            target("images/generations")?,
            RequestBody::json(&request.wire()).map_err(request_error)?,
            ReplaySafety::Never,
            options,
        )
        .await
    }
}

impl fmt::Debug for ArkImages {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArkImages")
            .field("runtime", &"shared")
            .finish()
    }
}

/// Provider-owned options for the portable ARK image adapter.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArkImageOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub watermark: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optimize_prompt_mode: Option<ArkOptimizePromptMode>,
}

impl ArkImageOptions {
    pub const fn new() -> Self {
        Self {
            watermark: None,
            optimize_prompt_mode: None,
        }
    }

    pub const fn with_watermark(mut self, watermark: bool) -> Self {
        self.watermark = Some(watermark);
        self
    }

    pub const fn with_optimize_prompt_mode(mut self, mode: ArkOptimizePromptMode) -> Self {
        self.optimize_prompt_mode = Some(mode);
        self
    }
}

impl TypedProviderOptions for ArkImageOptions {
    const NAMESPACE: &'static str = "volcengine";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Image;
    const API_MODE: Option<&'static str> = Some(ARK_IMAGE_API_MODE);
}

/// Portable image handle backed by one configured ARK runtime.
#[derive(Clone)]
pub struct ArkImageModel {
    images: ArkImages,
    descriptor: ModelDescriptor,
    defaults: ArkImageOptions,
}

impl ArkImageModel {
    pub(crate) fn new(
        runtime: SharedArkNativeRuntime,
        descriptor: ModelDescriptor,
        defaults: ArkImageOptions,
    ) -> Self {
        Self {
            images: ArkImages::new(runtime),
            descriptor,
            defaults,
        }
    }

    fn options(&self, call: &CallOptions) -> Result<ArkImageOptions, Error> {
        let layers = call
            .apply_provider_options(self.provider_id(), ProviderOptionLayers::default())
            .map_err(option_error)?;
        layers
            .merge_for(
                ProviderOptionContext::new(
                    self.provider_id(),
                    ModelFamily::Image,
                    self.descriptor.scope().api_mode(),
                ),
                &ArkImageOptionMerger {
                    defaults: self.defaults.clone(),
                },
            )
            .map_err(option_error)
    }

    fn contextualize(&self, error: Error) -> Error {
        error.with_context(ErrorContext {
            operation: Some(ModelOperation::GenerateImage),
            provider: Some(self.provider_id().clone()),
            route: None,
            model: Some(self.model_id().clone()),
        })
    }
}

impl Model for ArkImageModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl ImageModel for ArkImageModel {
    fn limits(&self) -> ImageLimits {
        ImageLimits {
            max_outputs_per_call: Some(MAX_IMAGES_PER_PORTABLE_CALL),
        }
    }

    async fn generate_image(
        &self,
        request: ImageRequest,
        call: CallOptions,
    ) -> Result<ImageResponse, Error> {
        self.limits()
            .validate(&request)
            .map_err(|error| self.contextualize(error))?;
        let options = self
            .options(&call)
            .map_err(|error| self.contextualize(error))?;
        let output_format = request
            .format()
            .map(parse_portable_format)
            .transpose()
            .map_err(|error| self.contextualize(error))?
            .unwrap_or(ArkImageOutputFormat::Jpeg);
        let mut native = ArkImageRequest::new(self.model_id().to_string(), request.prompt())
            .map_err(|error| self.contextualize(error))?
            .with_output_format(output_format)
            .with_response_format(ArkImageResponseFormat::B64Json);
        if let Some(size) = request.size() {
            native = native
                .with_size(format!("{}x{}", size.width(), size.height()))
                .map_err(|error| self.contextualize(error))?;
        }
        if let Some(watermark) = options.watermark {
            native = native.with_watermark(watermark);
        }
        if let Some(mode) = options.optimize_prompt_mode {
            native = native.with_optimize_prompt_mode(mode);
        }
        let response = self
            .images
            .generate_with_options(native, call)
            .await
            .map_err(|error| self.contextualize(error))?;
        let images = response
            .data
            .into_iter()
            .map(|image| decode_portable_image(image, output_format))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| self.contextualize(error))?;
        let output = ImageResponse {
            images,
            metadata: ResponseMetadata {
                response_id: None,
                request_id: None,
                model: Some(self.model_id().clone()),
            },
            usage: Usage::default(),
            warnings: Vec::new(),
            provider: response.extra,
        };
        output
            .validate(&request)
            .map_err(|error| self.contextualize(error))?;
        Ok(output)
    }
}

impl fmt::Debug for ArkImageModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArkImageModel")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

struct ArkImageOptionMerger {
    defaults: ArkImageOptions,
}

impl ProviderOptionMerger for ArkImageOptionMerger {
    type Output = ArkImageOptions;

    fn validate_layer(
        &self,
        _origin: ProviderOptionOrigin,
        options: &ProviderOptions,
    ) -> Result<(), ProviderOptionError> {
        decode_options(options).map(|_| ())
    }

    fn merge(&self, layers: &ProviderOptionLayers) -> Result<Self::Output, ProviderOptionError> {
        let mut merged = self.defaults.clone();
        for (_, options) in layers.in_precedence_order() {
            let layer = decode_options(options)?;
            if layer.watermark.is_some() {
                merged.watermark = layer.watermark;
            }
            if layer.optimize_prompt_mode.is_some() {
                merged.optimize_prompt_mode = layer.optimize_prompt_mode;
            }
        }
        Ok(merged)
    }
}

fn decode_options(options: &ProviderOptions) -> Result<ArkImageOptions, ProviderOptionError> {
    serde_json::from_value(Value::Object(options.value().clone())).map_err(|_| {
        ProviderOptionError::Rejected {
            path: "volcengine".to_string(),
            reason: "options do not match the ARK image schema".to_string(),
        }
    })
}

fn decode_portable_image(
    image: ArkGeneratedImage,
    format: ArkImageOutputFormat,
) -> Result<ImageArtifact, Error> {
    let encoded = image.b64_json.ok_or_else(|| {
        Error::new(
            ErrorKind::Protocol,
            "ARK portable image response omitted b64_json",
        )
    })?;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|source| {
            Error::new(
                ErrorKind::Protocol,
                "ARK portable image response contained invalid base64",
            )
            .with_source(source)
        })?;
    if bytes.is_empty() {
        return Err(Error::new(
            ErrorKind::Protocol,
            "ARK portable image response contained an empty artifact",
        ));
    }
    Ok(ImageArtifact {
        media_type: format.media_type().to_string(),
        data: MediaData::Bytes(Bytes::from(bytes)),
        revised_prompt: None,
    })
}

fn parse_portable_format(format: &str) -> Result<ArkImageOutputFormat, Error> {
    match format.to_ascii_lowercase().as_str() {
        "png" | "image/png" => Ok(ArkImageOutputFormat::Png),
        "jpeg" | "jpg" | "image/jpeg" => Ok(ArkImageOutputFormat::Jpeg),
        _ => Err(Error::new(
            ErrorKind::Unsupported,
            "ARK portable image generation supports png or jpeg output",
        )),
    }
}

fn validate_image_source(value: &str, remote: bool) -> Result<(), Error> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_IMAGE_SOURCE_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(invalid("ARK image source is invalid"));
    }
    let valid_prefix = if remote {
        value.starts_with("https://")
    } else {
        value.starts_with("data:image/") && value.contains(";base64,")
    };
    if !valid_prefix {
        return Err(invalid("ARK image source has an unsupported scheme"));
    }
    Ok(())
}

fn insert_value<T: Serialize>(
    body: &mut serde_json::Map<String, Value>,
    key: &'static str,
    value: Option<T>,
) {
    if let Some(value) = value {
        body.insert(
            key.to_string(),
            serde_json::to_value(value).expect("ARK image option serialization is infallible"),
        );
    }
}

fn invalid_model(source: siumai_core::InvalidId) -> Error {
    invalid("ARK image model identifier is invalid").with_source(source)
}

fn option_error(source: ProviderOptionError) -> Error {
    invalid("provider options are invalid for ARK image generation").with_source(source)
}

fn request_error(source: siumai_transport::RequestBuildError) -> Error {
    Error::new(
        ErrorKind::Configuration,
        "ARK image request violates the transport contract",
    )
    .with_source(source)
}

fn invalid(message: &'static str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}
