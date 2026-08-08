use std::collections::BTreeMap;

use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use siumai_core::{
    Error, ErrorKind, ImageArtifact, ImageResponse, MediaData, ModelId, ResourceKind,
    ResponseMetadata, Usage, Warning,
};

/// Stable Gemini Interactions create target relative to the official API origin.
pub const STABLE_V1_CREATE_TARGET: &str = "v1/interactions";

const DEFAULT_IMAGE_MEDIA_TYPE: &str = "image/jpeg";
const MAX_RESPONSE_BODY_BYTES: usize = 64 * 1024 * 1024;
const MAX_IMAGE_BYTES: usize = 32 * 1024 * 1024;
const MAX_IMAGE_ENCODED_BYTES: usize = (MAX_IMAGE_BYTES / 3 + 1) * 4;
const MAX_IMAGE_BLOCKS: usize = 16;
const MAX_STEPS: usize = 128;
const MAX_CONTENT_BLOCKS_PER_STEP: usize = 256;
const MAX_RESPONSE_ID_BYTES: usize = 4 * 1024;
const MAX_REQUEST_ID_BYTES: usize = 4 * 1024;
const MAX_SERVICE_TIER_BYTES: usize = 128;
const MAX_MEDIA_TYPE_BYTES: usize = 128;
const MAX_URI_BYTES: usize = 16 * 1024;
const MAX_MODALITY_ENTRIES: usize = 32;
const MAX_MODALITY_BYTES: usize = 64;

/// MIME types explicitly configurable for stable v1 image output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ImageMimeType {
    Jpeg,
}

impl ImageMimeType {
    const fn as_wire(self) -> &'static str {
        match self {
            Self::Jpeg => "image/jpeg",
        }
    }
}

/// Aspect ratios accepted by the stable v1 image response format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ImageAspectRatio {
    Square,
    PortraitOneFour,
    PortraitOneEight,
    PortraitTwoThree,
    LandscapeThreeTwo,
    PortraitThreeFour,
    LandscapeFourOne,
    LandscapeFourThree,
    PortraitFourFive,
    LandscapeFiveFour,
    LandscapeEightOne,
    PortraitNineSixteen,
    LandscapeSixteenNine,
    Ultrawide,
}

impl ImageAspectRatio {
    const fn as_wire(self) -> &'static str {
        match self {
            Self::Square => "1:1",
            Self::PortraitOneFour => "1:4",
            Self::PortraitOneEight => "1:8",
            Self::PortraitTwoThree => "2:3",
            Self::LandscapeThreeTwo => "3:2",
            Self::PortraitThreeFour => "3:4",
            Self::LandscapeFourOne => "4:1",
            Self::LandscapeFourThree => "4:3",
            Self::PortraitFourFive => "4:5",
            Self::LandscapeFiveFour => "5:4",
            Self::LandscapeEightOne => "8:1",
            Self::PortraitNineSixteen => "9:16",
            Self::LandscapeSixteenNine => "16:9",
            Self::Ultrawide => "21:9",
        }
    }
}

/// Resolution tiers accepted by the stable v1 image response format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ImageSize {
    Pixels512,
    OneK,
    TwoK,
    FourK,
}

impl ImageSize {
    const fn as_wire(self) -> &'static str {
        match self {
            Self::Pixels512 => "512",
            Self::OneK => "1K",
            Self::TwoK => "2K",
            Self::FourK => "4K",
        }
    }
}

/// Delivery modes accepted by the stable v1 image response format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ImageDelivery {
    Inline,
    Uri,
}

impl ImageDelivery {
    const fn as_wire(self) -> &'static str {
        match self {
            Self::Inline => "inline",
            Self::Uri => "uri",
        }
    }
}

/// Checked stable v1 image response-format configuration.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ImageResponseFormat {
    mime_type: Option<ImageMimeType>,
    aspect_ratio: Option<ImageAspectRatio>,
    image_size: Option<ImageSize>,
    delivery: Option<ImageDelivery>,
}

impl ImageResponseFormat {
    pub const fn new() -> Self {
        Self {
            mime_type: None,
            aspect_ratio: None,
            image_size: None,
            delivery: None,
        }
    }

    pub const fn with_mime_type(mut self, mime_type: ImageMimeType) -> Self {
        self.mime_type = Some(mime_type);
        self
    }

    pub const fn with_aspect_ratio(mut self, aspect_ratio: ImageAspectRatio) -> Self {
        self.aspect_ratio = Some(aspect_ratio);
        self
    }

    pub const fn with_image_size(mut self, image_size: ImageSize) -> Self {
        self.image_size = Some(image_size);
        self
    }

    pub const fn with_delivery(mut self, delivery: ImageDelivery) -> Self {
        self.delivery = Some(delivery);
        self
    }

    pub const fn mime_type(&self) -> Option<ImageMimeType> {
        self.mime_type
    }

    pub const fn aspect_ratio(&self) -> Option<ImageAspectRatio> {
        self.aspect_ratio
    }

    pub const fn image_size(&self) -> Option<ImageSize> {
        self.image_size
    }

    pub const fn delivery(&self) -> Option<ImageDelivery> {
        self.delivery
    }
}

/// Encode one stable v1 image interaction request.
pub fn encode_image_request(
    model: &ModelId,
    prompt: &str,
    response_format: &ImageResponseFormat,
) -> Result<Value, Error> {
    if prompt.trim().is_empty() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Gemini Interactions image prompt must not be empty",
        ));
    }

    serde_json::to_value(ImageInteractionRequestWire {
        model: model.as_str(),
        input: prompt,
        response_format: ImageResponseFormatWire {
            kind: "image",
            mime_type: response_format.mime_type.map(ImageMimeType::as_wire),
            aspect_ratio: response_format.aspect_ratio.map(ImageAspectRatio::as_wire),
            image_size: response_format.image_size.map(ImageSize::as_wire),
            delivery: response_format.delivery.map(ImageDelivery::as_wire),
        },
    })
    .map_err(|source| {
        Error::new(
            ErrorKind::Internal,
            "Gemini Interactions image request could not be encoded",
        )
        .with_source(source)
    })
}

/// Decode one terminal stable v1 image interaction response.
pub fn decode_image_response(
    body: &[u8],
    requested_model: &ModelId,
    request_id: Option<&str>,
    mut warnings: Vec<Warning>,
) -> Result<ImageResponse, Error> {
    if body.len() > MAX_RESPONSE_BODY_BYTES {
        return Err(Error::new(
            ErrorKind::ResponseLimit,
            "Gemini Interactions response exceeded the protocol body limit",
        ));
    }

    let wire = serde_json::from_slice::<InteractionResponseWire>(body).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "Gemini returned malformed Interactions JSON",
        )
        .with_source(source)
    })?;

    if !matches!(
        wire.status,
        InteractionStatus::Completed | InteractionStatus::Incomplete
    ) {
        validate_terminal_status(wire.status, 0)?;
    }

    if wire.steps.len() > MAX_STEPS {
        return Err(Error::new(
            ErrorKind::ResponseLimit,
            "Gemini Interactions response exceeded the step limit",
        ));
    }

    let mut images = Vec::new();
    for step in wire.steps {
        let InteractionStepWire::ModelOutput { content } = step else {
            continue;
        };
        if content.len() > MAX_CONTENT_BLOCKS_PER_STEP {
            return Err(Error::new(
                ErrorKind::ResponseLimit,
                "Gemini Interactions response exceeded the content-block limit",
            ));
        }
        for block in content {
            if let InteractionContentWire::Image {
                data,
                mime_type,
                uri,
            } = block
            {
                if images.len() == MAX_IMAGE_BLOCKS {
                    return Err(Error::new(
                        ErrorKind::ResponseLimit,
                        "Gemini Interactions response exceeded the image-block limit",
                    ));
                }
                images.push(decode_image(data, mime_type, uri)?);
            }
        }
    }

    validate_terminal_status(wire.status, images.len())?;

    let response_id = wire.id.map(checked_response_id).transpose()?;
    let request_id = request_id.map(checked_request_id).transpose()?;
    let model = wire
        .model
        .map(|model| {
            ModelId::new(model).map_err(|source| {
                Error::protocol_violation("Gemini returned an invalid model identifier")
                    .with_source(source)
            })
        })
        .transpose()?
        .unwrap_or_else(|| requested_model.clone());

    let image_count = images.len();
    let final_image = images
        .pop()
        .ok_or_else(|| Error::partial_result(ResourceKind::ImageOutputs, 1, 0))?;

    let mut provider = BTreeMap::from([(
        "google.status".to_string(),
        Value::String(wire.status.as_str().to_string()),
    )]);
    if image_count > 1 {
        provider.insert(
            "google.image_block_count".to_string(),
            Value::from(as_u64(image_count)),
        );
        warnings.push(Warning::provider(
            "multiple_image_outputs",
            "Gemini returned multiple image blocks; the final block was selected",
        ));
    }
    if let Some(service_tier) = wire.service_tier {
        provider.insert(
            "google.service_tier".to_string(),
            Value::String(checked_service_tier(service_tier)?),
        );
    }

    Ok(ImageResponse {
        images: vec![final_image],
        metadata: ResponseMetadata {
            response_id,
            request_id,
            model: Some(model),
        },
        usage: decode_usage(wire.usage)?,
        warnings,
        provider,
    })
}

#[derive(Serialize)]
struct ImageInteractionRequestWire<'a> {
    model: &'a str,
    input: &'a str,
    response_format: ImageResponseFormatWire<'a>,
}

#[derive(Serialize)]
struct ImageResponseFormatWire<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    mime_type: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    aspect_ratio: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_size: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    delivery: Option<&'a str>,
}

#[derive(Deserialize)]
struct InteractionResponseWire {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    model: Option<String>,
    status: InteractionStatus,
    #[serde(default)]
    steps: Vec<InteractionStepWire>,
    #[serde(default)]
    usage: Option<InteractionUsageWire>,
    #[serde(default)]
    service_tier: Option<String>,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum InteractionStepWire {
    #[serde(rename = "model_output")]
    ModelOutput {
        #[serde(default)]
        content: Vec<InteractionContentWire>,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum InteractionContentWire {
    #[serde(rename = "image")]
    Image {
        #[serde(default)]
        data: Option<String>,
        #[serde(default)]
        mime_type: Option<String>,
        #[serde(default)]
        uri: Option<String>,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum InteractionStatus {
    InProgress,
    RequiresAction,
    Completed,
    Failed,
    Cancelled,
    Incomplete,
    #[serde(other)]
    Unknown,
}

impl InteractionStatus {
    const fn as_str(self) -> &'static str {
        match self {
            Self::InProgress => "in_progress",
            Self::RequiresAction => "requires_action",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Incomplete => "incomplete",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Deserialize)]
struct InteractionUsageWire {
    #[serde(default)]
    total_input_tokens: Option<u64>,
    #[serde(default)]
    total_output_tokens: Option<u64>,
    #[serde(default)]
    total_thought_tokens: Option<u64>,
    #[serde(default)]
    total_cached_tokens: Option<u64>,
    #[serde(default)]
    total_tool_use_tokens: Option<u64>,
    #[serde(default)]
    total_tokens: Option<u64>,
    #[serde(default)]
    input_tokens_by_modality: Option<Vec<ModalityTokensWire>>,
    #[serde(default)]
    output_tokens_by_modality: Option<Vec<ModalityTokensWire>>,
    #[serde(default)]
    cached_tokens_by_modality: Option<Vec<ModalityTokensWire>>,
    #[serde(default)]
    tool_use_tokens_by_modality: Option<Vec<ModalityTokensWire>>,
}

#[derive(Serialize, Deserialize)]
struct ModalityTokensWire {
    modality: String,
    tokens: u64,
}

fn validate_terminal_status(status: InteractionStatus, image_count: usize) -> Result<(), Error> {
    match status {
        InteractionStatus::Completed => Ok(()),
        InteractionStatus::Cancelled => {
            Err(Error::cancelled("Gemini cancelled the image interaction"))
        }
        InteractionStatus::Incomplete => Err(Error::partial_result(
            ResourceKind::ImageOutputs,
            1,
            as_u64(image_count.min(1)),
        )),
        InteractionStatus::Failed => Err(Error::new(
            ErrorKind::Provider,
            "Gemini reported a failed image interaction",
        )),
        InteractionStatus::InProgress | InteractionStatus::RequiresAction => Err(
            Error::protocol_violation("Gemini returned a non-terminal image interaction"),
        ),
        InteractionStatus::Unknown => Err(Error::protocol_violation(
            "Gemini returned an unknown interaction status",
        )),
    }
}

fn decode_image(
    data: Option<String>,
    mime_type: Option<String>,
    uri: Option<String>,
) -> Result<ImageArtifact, Error> {
    let media_type = match mime_type {
        Some(media_type) => checked_media_type(media_type)?,
        None => DEFAULT_IMAGE_MEDIA_TYPE.to_string(),
    };
    let data = match (data, uri) {
        (Some(encoded), _) => {
            if encoded.len() > MAX_IMAGE_ENCODED_BYTES {
                return Err(Error::new(
                    ErrorKind::ResponseLimit,
                    "Gemini inline image exceeded the encoded image limit",
                ));
            }
            let decoded = base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .map_err(|source| {
                    Error::protocol_violation(
                        "Gemini returned invalid base64 data in an image block",
                    )
                    .with_source(source)
                })?;
            if decoded.is_empty() {
                return Err(Error::protocol_violation(
                    "Gemini returned an empty inline image",
                ));
            }
            if decoded.len() > MAX_IMAGE_BYTES {
                return Err(Error::new(
                    ErrorKind::ResponseLimit,
                    "Gemini inline image exceeded the decoded image limit",
                ));
            }
            MediaData::Bytes(decoded.into())
        }
        (None, Some(uri)) => MediaData::Url(checked_uri(uri)?),
        (None, None) => {
            return Err(Error::protocol_violation(
                "Gemini returned an image block without data or URI",
            ));
        }
    };
    Ok(ImageArtifact {
        media_type,
        data,
        revised_prompt: None,
    })
}

fn decode_usage(wire: Option<InteractionUsageWire>) -> Result<Usage, Error> {
    let Some(wire) = wire else {
        return Ok(Usage::default());
    };

    let mut usage = Usage::default()
        .with_input_tokens(wire.total_input_tokens)
        .with_output_tokens(wire.total_output_tokens)
        .with_total_tokens(wire.total_tokens)
        .with_reasoning_tokens(wire.total_thought_tokens)
        .with_cache_read_tokens(wire.total_cached_tokens)
        .with_orchestration_tokens(wire.total_tool_use_tokens);
    for (name, modalities) in [
        (
            "google.input_tokens_by_modality",
            wire.input_tokens_by_modality,
        ),
        (
            "google.output_tokens_by_modality",
            wire.output_tokens_by_modality,
        ),
        (
            "google.cached_tokens_by_modality",
            wire.cached_tokens_by_modality,
        ),
        (
            "google.tool_use_tokens_by_modality",
            wire.tool_use_tokens_by_modality,
        ),
    ] {
        if let Some(modalities) = modalities {
            let value = checked_modalities(modalities)?;
            usage = usage.with_provider_value(name, value);
        }
    }
    Ok(usage)
}

fn checked_modalities(modalities: Vec<ModalityTokensWire>) -> Result<Value, Error> {
    if modalities.len() > MAX_MODALITY_ENTRIES {
        return Err(Error::new(
            ErrorKind::ResponseLimit,
            "Gemini usage exceeded the modality-entry limit",
        ));
    }
    if modalities.iter().any(|entry| {
        !valid_bounded_text(&entry.modality, MAX_MODALITY_BYTES) || !entry.modality.is_ascii()
    }) {
        return Err(Error::protocol_violation(
            "Gemini returned an invalid usage modality",
        ));
    }
    serde_json::to_value(modalities).map_err(|source| {
        Error::new(
            ErrorKind::Internal,
            "Gemini usage modalities could not be retained",
        )
        .with_source(source)
    })
}

fn checked_response_id(value: String) -> Result<String, Error> {
    if valid_bounded_text(&value, MAX_RESPONSE_ID_BYTES) {
        Ok(value)
    } else {
        Err(Error::protocol_violation(
            "Gemini returned an invalid interaction identifier",
        ))
    }
}

fn checked_request_id(value: &str) -> Result<String, Error> {
    if valid_bounded_text(value, MAX_REQUEST_ID_BYTES) {
        Ok(value.to_string())
    } else {
        Err(Error::protocol_violation(
            "Gemini returned an invalid request identifier",
        ))
    }
}

fn checked_service_tier(value: String) -> Result<String, Error> {
    if valid_bounded_text(&value, MAX_SERVICE_TIER_BYTES) && value.is_ascii() {
        Ok(value)
    } else {
        Err(Error::protocol_violation(
            "Gemini returned an invalid service tier",
        ))
    }
}

fn checked_media_type(value: String) -> Result<String, Error> {
    let valid = valid_bounded_text(&value, MAX_MEDIA_TYPE_BYTES)
        && value.is_ascii()
        && value
            .split_once('/')
            .is_some_and(|(kind, subtype)| kind == "image" && !subtype.is_empty());
    if valid {
        Ok(value)
    } else {
        Err(Error::protocol_violation(
            "Gemini returned an invalid image media type",
        ))
    }
}

fn checked_uri(value: String) -> Result<String, Error> {
    if valid_bounded_text(&value, MAX_URI_BYTES) {
        Ok(value)
    } else {
        Err(Error::protocol_violation(
            "Gemini returned an invalid image URI",
        ))
    }
}

fn valid_bounded_text(value: &str, maximum: usize) -> bool {
    !value.is_empty()
        && value.len() <= maximum
        && value == value.trim()
        && !value.chars().any(char::is_control)
}

fn as_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use siumai_core::{ErrorKind, UsageValue};

    fn model() -> ModelId {
        ModelId::new("gemini-3.1-flash-image-preview").unwrap()
    }

    #[test]
    fn encodes_stable_v1_current_image_response_format() {
        let format = ImageResponseFormat::new()
            .with_mime_type(ImageMimeType::Jpeg)
            .with_aspect_ratio(ImageAspectRatio::LandscapeSixteenNine)
            .with_image_size(ImageSize::TwoK)
            .with_delivery(ImageDelivery::Uri);

        let value = encode_image_request(&model(), "Draw a lighthouse", &format).unwrap();

        assert_eq!(STABLE_V1_CREATE_TARGET, "v1/interactions");
        assert_eq!(
            value,
            serde_json::json!({
                "model": "gemini-3.1-flash-image-preview",
                "input": "Draw a lighthouse",
                "response_format": {
                    "type": "image",
                    "mime_type": "image/jpeg",
                    "aspect_ratio": "16:9",
                    "image_size": "2K",
                    "delivery": "uri"
                }
            })
        );
        assert!(value.get("outputs").is_none());
        assert!(value.get("response_mime_type").is_none());
        assert!(value.get("response_modalities").is_none());

        let default =
            encode_image_request(&model(), "Draw a lighthouse", &ImageResponseFormat::new())
                .unwrap();
        assert!(default["response_format"].get("mime_type").is_none());
    }

    #[test]
    fn decodes_completed_inline_image_and_usage() {
        let body = serde_json::to_vec(&serde_json::json!({
            "id": "interaction-1",
            "model": "gemini-3.1-flash-image-preview",
            "status": "completed",
            "steps": [{
                "type": "model_output",
                "content": [{
                    "type": "image",
                    "mime_type": "image/png",
                    "data": base64::engine::general_purpose::STANDARD.encode(b"image")
                }]
            }],
            "usage": {
                "total_input_tokens": 7,
                "total_output_tokens": 3,
                "total_tokens": 10,
                "input_tokens_by_modality": [{"modality": "text", "tokens": 7}]
            },
            "service_tier": "standard"
        }))
        .unwrap();

        let response =
            decode_image_response(&body, &model(), Some("request-1"), Vec::new()).unwrap();

        assert_eq!(
            response.metadata.response_id.as_deref(),
            Some("interaction-1")
        );
        assert_eq!(response.metadata.request_id.as_deref(), Some("request-1"));
        assert_eq!(response.usage.input_tokens, UsageValue::Known(7));
        assert_eq!(response.usage.output_tokens, UsageValue::Known(3));
        assert_eq!(response.usage.total_tokens, UsageValue::Known(10));
        assert_eq!(response.images[0].media_type, "image/png");
        assert!(matches!(
            &response.images[0].data,
            MediaData::Bytes(bytes) if bytes.as_ref() == b"image"
        ));
        assert_eq!(response.provider["google.service_tier"], "standard");
    }

    #[test]
    fn keeps_unknown_usage_and_selects_the_final_uri_image() {
        let body = serde_json::to_vec(&serde_json::json!({
            "status": "completed",
            "steps": [{
                "type": "model_output",
                "content": [
                    {"type": "image", "uri": "https://example.test/first.png"},
                    {"type": "image", "uri": "https://example.test/final.png"}
                ]
            }]
        }))
        .unwrap();

        let response = decode_image_response(&body, &model(), None, Vec::new()).unwrap();

        assert_eq!(response.usage.input_tokens, UsageValue::Unknown);
        assert_eq!(response.usage.output_tokens, UsageValue::Unknown);
        assert!(matches!(
            &response.images[0].data,
            MediaData::Url(uri) if uri == "https://example.test/final.png"
        ));
        assert_eq!(response.warnings.len(), 1);
        assert_eq!(
            response.provider["google.image_block_count"],
            Value::from(2_u64)
        );
    }

    #[test]
    fn rejects_non_terminal_failed_and_unknown_statuses() {
        for (status, expected) in [
            ("in_progress", ErrorKind::ProtocolViolation),
            ("failed", ErrorKind::Provider),
            ("cancelled", ErrorKind::Cancelled),
            ("incomplete", ErrorKind::PartialResult),
            ("future_status", ErrorKind::ProtocolViolation),
        ] {
            let body = serde_json::to_vec(&serde_json::json!({
                "status": status,
                "steps": []
            }))
            .unwrap();
            let error = decode_image_response(&body, &model(), None, Vec::new()).unwrap_err();
            assert_eq!(error.kind(), expected, "status {status}");
        }

        let error = decode_image_response(b"{not-json", &model(), None, Vec::new()).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Protocol);
        assert!(!format!("{error:?}").contains("not-json"));
    }
}
