//! OpenAI portable non-streaming image generation model.

use std::sync::Arc;

use async_trait::async_trait;
use http::Method;
use http::header::{ACCEPT, HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use siumai_core::{
    CallOptions, Error, ErrorContext, ErrorKind, ImageLimits, ImageModel, ImageRequest,
    ImageResponse, Model, ModelDescriptor, ModelFamily, ModelId, ModelOperation,
    ProviderOptionContext, ProviderOptionError, ProviderOptionLayers, ProviderOptionMerger,
    ProviderOptionOrigin, ProviderOptions, ProviderScope, TypedProviderOptions, Warning,
    WarningKind,
};
use siumai_protocol_openai::image::{
    API_MODE_ID, ImageBackground, ImageGenerationConfig, ImageModeration, ImageOutputFormat,
    ImageQuality, ImageResponseFormat, ImageStyle, TARGET, decode_image_response,
    encode_image_request,
};
use siumai_transport::{ReplaySafety, RequestBody, RequestHeaders, RequestPlan, RequestTarget};

use super::http_error::{request_build_error, response_error};
use super::provider::OpenAiRuntime;
use super::tools::{OpenAiImageBackground, OpenAiImageModeration, OpenAiImageOutputFormat};

pub const GPT_IMAGE_1: &str = "gpt-image-1";
pub const GPT_IMAGE_1_MINI: &str = "gpt-image-1-mini";
pub const GPT_IMAGE_1_5: &str = "gpt-image-1.5";
pub const GPT_IMAGE_2: &str = "gpt-image-2";
pub const CHATGPT_IMAGE_LATEST: &str = "chatgpt-image-latest";
pub const DALL_E_2: &str = "dall-e-2";
pub const DALL_E_3: &str = "dall-e-3";

const MAX_IMAGES_PER_CALL: u32 = 10;
const MAX_USER_BYTES: usize = 2_048;

/// Quality values spanning GPT Image and legacy DALL-E generation APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum OpenAiImageGenerationQuality {
    Auto,
    Low,
    Medium,
    High,
    Standard,
    Hd,
}

/// Legacy Images API response representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum OpenAiImageResponseFormat {
    B64Json,
    Url,
}

/// DALL-E 3 image style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum OpenAiImageStyle {
    Natural,
    Vivid,
}

/// Provider-owned controls for non-streaming OpenAI image generation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiImageOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<OpenAiImageGenerationQuality>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<OpenAiImageBackground>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub moderation: Option<OpenAiImageModeration>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_format: Option<OpenAiImageOutputFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_compression: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<OpenAiImageResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<OpenAiImageStyle>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

impl OpenAiImageOptions {
    pub const fn new() -> Self {
        Self {
            quality: None,
            background: None,
            moderation: None,
            output_format: None,
            output_compression: None,
            response_format: None,
            style: None,
            user: None,
        }
    }

    pub const fn with_quality(mut self, quality: OpenAiImageGenerationQuality) -> Self {
        self.quality = Some(quality);
        self
    }

    pub const fn with_background(mut self, background: OpenAiImageBackground) -> Self {
        self.background = Some(background);
        self
    }

    pub const fn with_moderation(mut self, moderation: OpenAiImageModeration) -> Self {
        self.moderation = Some(moderation);
        self
    }

    pub const fn with_output_format(mut self, format: OpenAiImageOutputFormat) -> Self {
        self.output_format = Some(format);
        self
    }

    pub fn with_output_compression(mut self, compression: u8) -> Result<Self, ProviderOptionError> {
        self.output_compression = Some(compression);
        self.validate()?;
        Ok(self)
    }

    pub const fn with_response_format(mut self, format: OpenAiImageResponseFormat) -> Self {
        self.response_format = Some(format);
        self
    }

    pub const fn with_style(mut self, style: OpenAiImageStyle) -> Self {
        self.style = Some(style);
        self
    }

    pub fn with_user(mut self, user: impl Into<String>) -> Result<Self, ProviderOptionError> {
        self.user = Some(user.into());
        self.validate()?;
        Ok(self)
    }
}

impl TypedProviderOptions for OpenAiImageOptions {
    const NAMESPACE: &'static str = "openai";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Image;
    const API_MODE: Option<&'static str> = Some(API_MODE_ID);

    fn validate(&self) -> Result<(), ProviderOptionError> {
        if self.output_compression.is_some_and(|value| value > 100) {
            return Err(rejected("output_compression", "must be between 0 and 100"));
        }
        if self.response_format.is_some() && self.output_format.is_some() {
            return Err(rejected(
                "response_format",
                "legacy response_format and GPT Image output_format cannot be combined",
            ));
        }
        if self.background == Some(OpenAiImageBackground::Transparent)
            && self.output_format == Some(OpenAiImageOutputFormat::Jpeg)
        {
            return Err(rejected(
                "background",
                "transparent output requires png or webp",
            ));
        }
        if self.output_compression.is_some()
            && matches!(self.output_format, Some(OpenAiImageOutputFormat::Png))
        {
            return Err(rejected(
                "output_compression",
                "compression is supported only for jpeg or webp output",
            ));
        }
        if let Some(user) = self.user.as_deref()
            && (user.trim().is_empty()
                || user.len() > MAX_USER_BYTES
                || user.chars().any(char::is_control))
        {
            return Err(rejected(
                "user",
                "must be non-empty, contain no control characters, and be at most 2048 bytes",
            ));
        }
        Ok(())
    }
}

/// Lightweight OpenAI image generation handle.
#[derive(Clone)]
pub struct OpenAiImageModel {
    runtime: Arc<OpenAiRuntime>,
    descriptor: ModelDescriptor,
    defaults: OpenAiImageOptions,
}

impl OpenAiImageModel {
    pub(crate) fn new(
        runtime: Arc<OpenAiRuntime>,
        scope: Arc<ProviderScope>,
        model: ModelId,
        defaults: OpenAiImageOptions,
    ) -> Self {
        Self {
            runtime,
            descriptor: ModelDescriptor::from_scope(scope, model, ModelFamily::Image),
            defaults,
        }
    }

    fn options(&self, call: &CallOptions) -> Result<OpenAiImageOptions, Error> {
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
                &OpenAiImageOptionMerger {
                    defaults: self.defaults.clone(),
                },
            )
            .map_err(option_error)
    }

    fn plan(
        &self,
        request: &ImageRequest,
        options: &OpenAiImageOptions,
    ) -> Result<(RequestPlan, Option<ImageOutputFormat>), Error> {
        let portable_format = request.format().map(parse_portable_format).transpose()?;
        let typed_format = options.output_format.map(protocol_output_format);
        if portable_format.is_some() && typed_format.is_some() && portable_format != typed_format {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "portable image format conflicts with OpenAI output_format",
            ));
        }
        let output_format = portable_format.or(typed_format);
        if request.format().is_some() && options.response_format.is_some() {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "portable image format cannot be combined with legacy OpenAI response_format",
            ));
        }
        if options.background == Some(OpenAiImageBackground::Transparent)
            && output_format == Some(ImageOutputFormat::Jpeg)
        {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "transparent OpenAI images require png or webp output",
            ));
        }
        if options.output_compression.is_some() && output_format == Some(ImageOutputFormat::Png) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "OpenAI image compression is supported only for jpeg or webp output",
            ));
        }
        validate_known_request(
            self.descriptor.scope(),
            self.model_id(),
            request,
            options,
            output_format,
        )?;
        let config = ImageGenerationConfig {
            quality: options.quality.map(protocol_quality),
            background: options.background.map(protocol_background),
            moderation: options.moderation.map(protocol_moderation),
            output_format,
            output_compression: options.output_compression,
            response_format: options.response_format.map(protocol_response_format),
            style: options.style.map(protocol_style),
            user: options.user.clone(),
        };
        let body = encode_image_request(request, self.model_id(), &config)?;
        let headers = RequestHeaders::new()
            .try_insert(ACCEPT, HeaderValue::from_static("application/json"))
            .map_err(request_plan_error)?;
        let plan = RequestPlan::new(
            Method::POST,
            RequestTarget::new(TARGET).map_err(request_plan_error)?,
        )
        .with_headers(headers)
        .with_body(RequestBody::json(&body).map_err(request_plan_error)?)
        .with_replay_safety(ReplaySafety::Never)
        .map_err(request_plan_error)?;
        Ok((plan, output_format))
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

impl std::fmt::Debug for OpenAiImageModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenAiImageModel")
            .field("descriptor", &self.descriptor)
            .finish()
    }
}

impl Model for OpenAiImageModel {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
}

#[async_trait]
impl ImageModel for OpenAiImageModel {
    fn limits(&self) -> ImageLimits {
        ImageLimits {
            max_outputs_per_call: Some(
                if known_image_model(self.descriptor.scope(), self.model_id())
                    == Some(KnownImageModel::DallE3)
                {
                    1
                } else {
                    MAX_IMAGES_PER_CALL
                },
            ),
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
        let (plan, output_format) = self
            .plan(&request, &options)
            .map_err(|error| self.contextualize(error))?;
        let response = self
            .runtime
            .transport
            .execute(plan, call)
            .await
            .map_err(|error| self.contextualize(error))?;
        if !response.status().is_success() {
            return Err(self.contextualize(response_error(
                "OpenAI rejected the image generation request",
                response,
            )));
        }
        let (_, headers, body) = response.into_parts();
        let mut decoded = decode_image_response(&body, &request, self.model_id(), output_format)
            .map_err(|error| self.contextualize(error))?;
        decoded.metadata.request_id = response_request_id(&headers);
        if !is_verified_model(self.descriptor.scope(), self.model_id()) {
            decoded.warnings.push(Warning::new(
                WarningKind::UnknownModel,
                "model support is not verified for OpenAI image generation",
            ));
        }
        Ok(decoded)
    }
}

struct OpenAiImageOptionMerger {
    defaults: OpenAiImageOptions,
}

impl ProviderOptionMerger for OpenAiImageOptionMerger {
    type Output = OpenAiImageOptions;

    fn validate_layer(
        &self,
        _origin: ProviderOptionOrigin,
        options: &ProviderOptions,
    ) -> Result<(), ProviderOptionError> {
        decode_options(options).and_then(|options| options.validate())
    }

    fn merge(&self, layers: &ProviderOptionLayers) -> Result<Self::Output, ProviderOptionError> {
        let mut merged = serde_json::to_value(&self.defaults)
            .map_err(|error| ProviderOptionError::Serialization(error.to_string()))?
            .as_object()
            .cloned()
            .unwrap_or_default();
        for (_, options) in layers.in_precedence_order() {
            merged.extend(options.value().clone());
        }
        let output = serde_json::from_value::<OpenAiImageOptions>(Value::Object(merged))
            .map_err(|error| ProviderOptionError::Serialization(error.to_string()))?;
        output.validate()?;
        Ok(output)
    }
}

fn decode_options(options: &ProviderOptions) -> Result<OpenAiImageOptions, ProviderOptionError> {
    serde_json::from_value(Value::Object(options.value().clone())).map_err(|_| {
        ProviderOptionError::Rejected {
            path: "openai".to_string(),
            reason: "options do not match the OpenAI image-generation schema".to_string(),
        }
    })
}

fn parse_portable_format(format: &str) -> Result<ImageOutputFormat, Error> {
    match format.to_ascii_lowercase().as_str() {
        "png" | "image/png" => Ok(ImageOutputFormat::Png),
        "jpeg" | "jpg" | "image/jpeg" => Ok(ImageOutputFormat::Jpeg),
        "webp" | "image/webp" => Ok(ImageOutputFormat::Webp),
        _ => Err(Error::new(
            ErrorKind::Unsupported,
            "OpenAI portable image generation supports png, jpeg, or webp output",
        )),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KnownImageModel {
    GptImage,
    GptImage2,
    DallE2,
    DallE3,
}

fn known_image_model(scope: &ProviderScope, model: &ModelId) -> Option<KnownImageModel> {
    if scope.platform().map(|value| value.as_str()) != Some("openai-api") {
        return None;
    }
    match model.as_str() {
        GPT_IMAGE_1 | GPT_IMAGE_1_MINI | GPT_IMAGE_1_5 | CHATGPT_IMAGE_LATEST => {
            Some(KnownImageModel::GptImage)
        }
        GPT_IMAGE_2 => Some(KnownImageModel::GptImage2),
        DALL_E_2 => Some(KnownImageModel::DallE2),
        DALL_E_3 => Some(KnownImageModel::DallE3),
        _ => None,
    }
}

fn validate_known_request(
    scope: &ProviderScope,
    model: &ModelId,
    request: &ImageRequest,
    options: &OpenAiImageOptions,
    output_format: Option<ImageOutputFormat>,
) -> Result<(), Error> {
    let Some(model_kind) = known_image_model(scope, model) else {
        return Ok(());
    };
    let prompt_chars = request.prompt().chars().count();
    let maximum = match model_kind {
        KnownImageModel::DallE2 => 1_000,
        KnownImageModel::DallE3 => 4_000,
        KnownImageModel::GptImage | KnownImageModel::GptImage2 => 32_000,
    };
    if prompt_chars > maximum {
        return Err(invalid_known_request(
            "OpenAI image prompt exceeds the verified model limit",
        ));
    }
    if let Some(size) = request.size() {
        let valid = match model_kind {
            KnownImageModel::GptImage => matches!(
                (size.width(), size.height()),
                (1024, 1024) | (1536, 1024) | (1024, 1536)
            ),
            KnownImageModel::GptImage2 => valid_gpt_image_2_size(size.width(), size.height()),
            KnownImageModel::DallE2 => matches!(
                (size.width(), size.height()),
                (256, 256) | (512, 512) | (1024, 1024)
            ),
            KnownImageModel::DallE3 => matches!(
                (size.width(), size.height()),
                (1024, 1024) | (1792, 1024) | (1024, 1792)
            ),
        };
        if !valid {
            return Err(invalid_known_request(
                "OpenAI image size is not supported by the selected model",
            ));
        }
    }
    if let Some(quality) = options.quality {
        let valid = match model_kind {
            KnownImageModel::GptImage | KnownImageModel::GptImage2 => matches!(
                quality,
                OpenAiImageGenerationQuality::Auto
                    | OpenAiImageGenerationQuality::Low
                    | OpenAiImageGenerationQuality::Medium
                    | OpenAiImageGenerationQuality::High
            ),
            KnownImageModel::DallE2 => matches!(
                quality,
                OpenAiImageGenerationQuality::Auto | OpenAiImageGenerationQuality::Standard
            ),
            KnownImageModel::DallE3 => matches!(
                quality,
                OpenAiImageGenerationQuality::Auto
                    | OpenAiImageGenerationQuality::Standard
                    | OpenAiImageGenerationQuality::Hd
            ),
        };
        if !valid {
            return Err(invalid_known_request(
                "OpenAI image quality is not supported by the selected model",
            ));
        }
    }
    match model_kind {
        KnownImageModel::GptImage | KnownImageModel::GptImage2 => {
            if options.response_format.is_some() {
                return Err(invalid_known_request(
                    "OpenAI GPT Image models do not support response_format",
                ));
            }
            if options.style.is_some() {
                return Err(invalid_known_request(
                    "OpenAI GPT Image models do not support DALL-E style",
                ));
            }
            if model_kind == KnownImageModel::GptImage2
                && options.background == Some(OpenAiImageBackground::Transparent)
            {
                return Err(invalid_known_request(
                    "OpenAI gpt-image-2 does not support transparent backgrounds",
                ));
            }
        }
        KnownImageModel::DallE2 | KnownImageModel::DallE3 => {
            if options.background.is_some()
                || options.moderation.is_some()
                || options.output_compression.is_some()
                || output_format.is_some()
            {
                return Err(invalid_known_request(
                    "OpenAI DALL-E models do not support GPT Image output controls",
                ));
            }
            if model_kind == KnownImageModel::DallE2 && options.style.is_some() {
                return Err(invalid_known_request(
                    "OpenAI dall-e-2 does not support style",
                ));
            }
        }
    }
    Ok(())
}

fn valid_gpt_image_2_size(width: u32, height: u32) -> bool {
    let shorter = u64::from(width.min(height));
    let longer = u64::from(width.max(height));
    let pixels = u64::from(width) * u64::from(height);

    width.is_multiple_of(16)
        && height.is_multiple_of(16)
        && shorter >= 512
        && longer <= 3_840
        && longer <= shorter * 3
        && pixels <= 3_840 * 2_160
}

fn invalid_known_request(message: &'static str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

fn protocol_quality(value: OpenAiImageGenerationQuality) -> ImageQuality {
    match value {
        OpenAiImageGenerationQuality::Auto => ImageQuality::Auto,
        OpenAiImageGenerationQuality::Low => ImageQuality::Low,
        OpenAiImageGenerationQuality::Medium => ImageQuality::Medium,
        OpenAiImageGenerationQuality::High => ImageQuality::High,
        OpenAiImageGenerationQuality::Standard => ImageQuality::Standard,
        OpenAiImageGenerationQuality::Hd => ImageQuality::Hd,
    }
}

fn protocol_background(value: OpenAiImageBackground) -> ImageBackground {
    match value {
        OpenAiImageBackground::Auto => ImageBackground::Auto,
        OpenAiImageBackground::Opaque => ImageBackground::Opaque,
        OpenAiImageBackground::Transparent => ImageBackground::Transparent,
    }
}

fn protocol_moderation(value: OpenAiImageModeration) -> ImageModeration {
    match value {
        OpenAiImageModeration::Auto => ImageModeration::Auto,
        OpenAiImageModeration::Low => ImageModeration::Low,
    }
}

fn protocol_output_format(value: OpenAiImageOutputFormat) -> ImageOutputFormat {
    match value {
        OpenAiImageOutputFormat::Png => ImageOutputFormat::Png,
        OpenAiImageOutputFormat::Jpeg => ImageOutputFormat::Jpeg,
        OpenAiImageOutputFormat::Webp => ImageOutputFormat::Webp,
    }
}

fn protocol_response_format(value: OpenAiImageResponseFormat) -> ImageResponseFormat {
    match value {
        OpenAiImageResponseFormat::B64Json => ImageResponseFormat::Base64Json,
        OpenAiImageResponseFormat::Url => ImageResponseFormat::Url,
    }
}

fn protocol_style(value: OpenAiImageStyle) -> ImageStyle {
    match value {
        OpenAiImageStyle::Natural => ImageStyle::Natural,
        OpenAiImageStyle::Vivid => ImageStyle::Vivid,
    }
}

fn is_verified_model(scope: &ProviderScope, model: &ModelId) -> bool {
    known_image_model(scope, model).is_some()
}

fn rejected(path: &str, reason: &str) -> ProviderOptionError {
    ProviderOptionError::Rejected {
        path: path.to_string(),
        reason: reason.to_string(),
    }
}

fn option_error(source: ProviderOptionError) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        "provider options are invalid for OpenAI image generation",
    )
    .with_source(source)
}

fn request_plan_error(source: siumai_transport::RequestBuildError) -> Error {
    request_build_error(
        "OpenAI image request violates the transport contract",
        source,
    )
}

fn response_request_id(headers: &siumai_transport::ResponseHeaders) -> Option<String> {
    ["x-request-id", "request-id"].into_iter().find_map(|name| {
        headers
            .get(&HeaderName::from_static(name))
            .and_then(|value| value.to_str().ok())
            .filter(|value| {
                !value.is_empty()
                    && value.len() <= 256
                    && value.bytes().all(|byte| {
                        byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':')
                    })
            })
            .map(str::to_owned)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::configured::profile::OpenAiProfile;

    #[test]
    fn options_reject_incompatible_output_controls() {
        let options = OpenAiImageOptions::new()
            .with_background(OpenAiImageBackground::Transparent)
            .with_output_format(OpenAiImageOutputFormat::Jpeg);
        assert!(options.validate().is_err());

        let options = OpenAiImageOptions::new()
            .with_output_format(OpenAiImageOutputFormat::Png)
            .with_response_format(OpenAiImageResponseFormat::Url);
        assert!(options.validate().is_err());
    }

    #[test]
    fn official_known_models_reject_invalid_size_and_option_combinations() {
        let profile = OpenAiProfile::current().unwrap();
        let scope = profile.family_provider_scope(ModelFamily::Image).unwrap();

        let tiny = ImageRequest::new("image")
            .unwrap()
            .with_size(16, 16)
            .unwrap();
        assert!(
            validate_known_request(
                scope,
                &ModelId::new(GPT_IMAGE_2).unwrap(),
                &tiny,
                &OpenAiImageOptions::default(),
                None,
            )
            .is_err()
        );

        let standard = ImageRequest::new("image").unwrap();
        assert!(
            validate_known_request(
                scope,
                &ModelId::new(GPT_IMAGE_1).unwrap(),
                &standard,
                &OpenAiImageOptions::new().with_response_format(OpenAiImageResponseFormat::Url),
                None,
            )
            .is_err()
        );
        assert!(
            validate_known_request(
                scope,
                &ModelId::new(DALL_E_2).unwrap(),
                &standard,
                &OpenAiImageOptions::new().with_output_format(OpenAiImageOutputFormat::Png),
                Some(ImageOutputFormat::Png),
            )
            .is_err()
        );
        assert!(
            validate_known_request(
                scope,
                &ModelId::new(GPT_IMAGE_2).unwrap(),
                &standard,
                &OpenAiImageOptions::new().with_background(OpenAiImageBackground::Transparent),
                None,
            )
            .is_err()
        );
    }

    #[test]
    fn custom_endpoints_keep_open_model_baseline_behavior() {
        let profile = OpenAiProfile::custom(siumai_core::ReplayDomain::custom(
            siumai_core::ReplayDomainId::new("custom-image-fixture").unwrap(),
        ))
        .unwrap();
        let scope = profile.family_provider_scope(ModelFamily::Image).unwrap();
        let request = ImageRequest::new("image")
            .unwrap()
            .with_size(16, 16)
            .unwrap();
        let options =
            OpenAiImageOptions::new().with_response_format(OpenAiImageResponseFormat::Url);

        assert!(
            validate_known_request(
                scope,
                &ModelId::new(GPT_IMAGE_2).unwrap(),
                &request,
                &options,
                None,
            )
            .is_ok()
        );
    }
}
