//! OpenAI image-generation request and response codecs.

use std::collections::BTreeMap;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use siumai_core::{
    Error, ErrorKind, ImageArtifact, ImageRequest, ImageResponse, MediaData, ModelId,
    ResponseMetadata, Usage,
};

/// OpenAI Images API mode identifier.
pub const API_MODE_ID: &str = "image-generations";
/// OpenAI Images protocol identifier.
pub const PROTOCOL_ID: &str = "openai.images";
/// Relative OpenAI image-generation endpoint.
pub const TARGET: &str = "images/generations";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageQuality {
    Auto,
    Low,
    Medium,
    High,
    Standard,
    Hd,
}

impl ImageQuality {
    const fn as_wire(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Standard => "standard",
            Self::Hd => "hd",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageBackground {
    Auto,
    Opaque,
    Transparent,
}

impl ImageBackground {
    const fn as_wire(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Opaque => "opaque",
            Self::Transparent => "transparent",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageModeration {
    Auto,
    Low,
}

impl ImageModeration {
    const fn as_wire(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Low => "low",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageOutputFormat {
    Png,
    Jpeg,
    Webp,
}

impl ImageOutputFormat {
    pub const fn as_wire(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpeg",
            Self::Webp => "webp",
        }
    }

    pub const fn media_type(self) -> &'static str {
        match self {
            Self::Png => "image/png",
            Self::Jpeg => "image/jpeg",
            Self::Webp => "image/webp",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageResponseFormat {
    Base64Json,
    Url,
}

impl ImageResponseFormat {
    const fn as_wire(self) -> &'static str {
        match self {
            Self::Base64Json => "b64_json",
            Self::Url => "url",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageStyle {
    Natural,
    Vivid,
}

impl ImageStyle {
    const fn as_wire(self) -> &'static str {
        match self {
            Self::Natural => "natural",
            Self::Vivid => "vivid",
        }
    }
}

/// Checked provider-owned fields for one non-streaming image generation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ImageGenerationConfig {
    pub quality: Option<ImageQuality>,
    pub background: Option<ImageBackground>,
    pub moderation: Option<ImageModeration>,
    pub output_format: Option<ImageOutputFormat>,
    pub output_compression: Option<u8>,
    pub response_format: Option<ImageResponseFormat>,
    pub style: Option<ImageStyle>,
    pub user: Option<String>,
}

/// Encode one portable text-to-image operation.
pub fn encode_image_request(
    request: &ImageRequest,
    model: &ModelId,
    config: &ImageGenerationConfig,
) -> Result<Value, Error> {
    let size = request
        .size()
        .map(|size| format!("{}x{}", size.width(), size.height()));
    serde_json::to_value(ImageRequestWire {
        model: model.as_str(),
        prompt: request.prompt(),
        n: request.count(),
        size: size.as_deref(),
        quality: config.quality.map(ImageQuality::as_wire),
        background: config.background.map(ImageBackground::as_wire),
        moderation: config.moderation.map(ImageModeration::as_wire),
        output_format: config.output_format.map(ImageOutputFormat::as_wire),
        output_compression: config.output_compression,
        response_format: config.response_format.map(ImageResponseFormat::as_wire),
        style: config.style.map(ImageStyle::as_wire),
        user: config.user.as_deref(),
    })
    .map_err(|source| {
        Error::new(
            ErrorKind::Internal,
            "OpenAI image request could not be serialized",
        )
        .with_source(source)
    })
}

/// Decode base64 or temporary-URL image artifacts.
pub fn decode_image_response(
    body: &[u8],
    request: &ImageRequest,
    requested_model: &ModelId,
    requested_output_format: Option<ImageOutputFormat>,
) -> Result<ImageResponse, Error> {
    let response = serde_json::from_slice::<ImageResponseWire>(body).map_err(|source| {
        Error::protocol_violation("OpenAI image response is not valid JSON").with_source(source)
    })?;
    let response_output_format = response
        .output_format
        .as_deref()
        .map(parse_output_format)
        .transpose()?;
    let output_format = response_output_format
        .or(requested_output_format)
        .unwrap_or(ImageOutputFormat::Png);
    let images = response
        .data
        .into_iter()
        .map(|item| {
            let data = match (item.b64_json, item.url) {
                (Some(encoded), None) => MediaData::Bytes(
                    STANDARD
                        .decode(encoded)
                        .map_err(|source| {
                            Error::protocol_violation(
                                "OpenAI image response contains invalid base64 data",
                            )
                            .with_source(source)
                        })?
                        .into(),
                ),
                (None, Some(url)) if !url.trim().is_empty() => MediaData::Url(url),
                _ => {
                    return Err(Error::protocol_violation(
                        "OpenAI image response must contain exactly one image representation",
                    ));
                }
            };
            Ok(ImageArtifact {
                media_type: output_format.media_type().to_string(),
                data,
                revised_prompt: item.revised_prompt,
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let usage = response.usage.unwrap_or_default();
    let mut provider = BTreeMap::new();
    if let Some(created) = response.created {
        provider.insert("created".to_string(), Value::from(created));
    }
    insert_bounded_metadata(&mut provider, "size", response.size)?;
    insert_bounded_metadata(&mut provider, "quality", response.quality)?;
    insert_bounded_metadata(&mut provider, "background", response.background)?;
    insert_bounded_metadata(&mut provider, "output_format", response.output_format)?;
    if let Some(details) = usage.input_tokens_details {
        provider.insert(
            "input_tokens_details".to_string(),
            serde_json::to_value(details).map_err(|source| {
                Error::new(
                    ErrorKind::Internal,
                    "OpenAI image token details could not be serialized",
                )
                .with_source(source)
            })?,
        );
    }
    let decoded = ImageResponse {
        images,
        metadata: ResponseMetadata {
            response_id: None,
            request_id: None,
            model: Some(requested_model.clone()),
        },
        usage: Usage::default()
            .with_input_tokens(usage.input_tokens)
            .with_output_tokens(usage.output_tokens)
            .with_total_tokens(usage.total_tokens),
        warnings: Vec::new(),
        provider,
    };
    decoded.validate(request)?;
    Ok(decoded)
}

#[derive(Debug, Serialize)]
struct ImageRequestWire<'a> {
    model: &'a str,
    prompt: &'a str,
    n: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quality: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    background: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    moderation: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_format: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_compression: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    style: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<&'a str>,
}

#[derive(Debug, Deserialize)]
struct ImageResponseWire {
    #[serde(default)]
    created: Option<u64>,
    data: Vec<ImageItemWire>,
    #[serde(default)]
    background: Option<String>,
    #[serde(default)]
    output_format: Option<String>,
    #[serde(default)]
    size: Option<String>,
    #[serde(default)]
    quality: Option<String>,
    #[serde(default)]
    usage: Option<ImageUsageWire>,
}

#[derive(Debug, Deserialize)]
struct ImageItemWire {
    #[serde(default)]
    b64_json: Option<String>,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    revised_prompt: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct ImageUsageWire {
    #[serde(default)]
    input_tokens: Option<u64>,
    #[serde(default)]
    output_tokens: Option<u64>,
    #[serde(default)]
    total_tokens: Option<u64>,
    #[serde(default)]
    input_tokens_details: Option<ImageInputTokenDetailsWire>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ImageInputTokenDetailsWire {
    #[serde(default)]
    image_tokens: Option<u64>,
    #[serde(default)]
    text_tokens: Option<u64>,
}

fn parse_output_format(value: &str) -> Result<ImageOutputFormat, Error> {
    match value {
        "png" => Ok(ImageOutputFormat::Png),
        "jpeg" | "jpg" => Ok(ImageOutputFormat::Jpeg),
        "webp" => Ok(ImageOutputFormat::Webp),
        _ => Err(Error::protocol_violation(
            "OpenAI image response contains an unknown output format",
        )),
    }
}

fn insert_bounded_metadata(
    target: &mut BTreeMap<String, Value>,
    name: &'static str,
    value: Option<String>,
) -> Result<(), Error> {
    if let Some(value) = value {
        if value.len() > 128 || value.chars().any(char::is_control) {
            return Err(Error::protocol_violation(
                "OpenAI image response metadata is invalid",
            ));
        }
        target.insert(name.to_string(), Value::String(value));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_maps_portable_size_and_decodes_base64_and_url_images() {
        let request = ImageRequest::new("Draw a lighthouse")
            .unwrap()
            .with_count(2)
            .unwrap()
            .with_size(1536, 1024)
            .unwrap();
        let model = ModelId::new("gpt-image-2").unwrap();
        let config = ImageGenerationConfig {
            output_format: Some(ImageOutputFormat::Webp),
            quality: Some(ImageQuality::High),
            ..Default::default()
        };
        let encoded = encode_image_request(&request, &model, &config).unwrap();
        assert_eq!(encoded["size"], "1536x1024");
        assert_eq!(encoded["output_format"], "webp");

        let decoded = decode_image_response(
            br#"{"data":[{"b64_json":"aW1hZ2U="},{"url":"https://example.com/image.webp"}],"output_format":"webp","usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}"#,
            &request,
            &model,
            config.output_format,
        )
        .unwrap();
        assert_eq!(decoded.images.len(), 2);
        assert_eq!(decoded.images[0].media_type, "image/webp");
        assert!(matches!(decoded.images[0].data, MediaData::Bytes(_)));
        assert!(matches!(decoded.images[1].data, MediaData::Url(_)));
    }

    #[test]
    fn decoder_rejects_ambiguous_image_representations() {
        let request = ImageRequest::new("Draw a lighthouse").unwrap();
        let model = ModelId::new("future-image-model").unwrap();
        let error = decode_image_response(
            br#"{"data":[{"b64_json":"aW1hZ2U=","url":"https://example.com/image.png"}]}"#,
            &request,
            &model,
            None,
        )
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ProtocolViolation);
    }
}
