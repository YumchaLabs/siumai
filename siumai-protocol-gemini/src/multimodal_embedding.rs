//! Gemini v1beta multimodal embedding request and response codecs.

use std::fmt;

use base64::Engine as _;
use bytes::Bytes;
use serde::Deserialize;
use serde_json::{Value, json};
use siumai_core::{Error, ErrorKind, ModelId, Usage};

const V1BETA_MODELS_PREFIX: &str = "v1beta/models/";
const EMBED_CONTENT_SUFFIX: &str = ":embedContent";
const MAX_PARTS: usize = 64;
const MAX_FILE_URI_BYTES: usize = 8 * 1024;
const MAX_INLINE_REQUEST_BYTES: usize = 20_000_000;
const REQUEST_ENVELOPE_BYTES: usize = 1024;
const PART_ENVELOPE_BYTES: usize = 256;
const MIN_OUTPUT_DIMENSIONALITY: u32 = 128;
const MAX_OUTPUT_DIMENSIONALITY: u32 = 3_072;
const MAX_SHAPE_RANK: usize = 16;
const MAX_MODALITY_USAGE_ITEMS: usize = 32;

/// One ordered multimodal input part accepted by Gemini Embedding 2.
#[derive(Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GeminiEmbeddingContentPart {
    Text(String),
    InlineData {
        media_type: String,
        data: Bytes,
    },
    FileData {
        media_type: String,
        file_uri: String,
    },
}

impl GeminiEmbeddingContentPart {
    pub fn text(text: impl Into<String>) -> Result<Self, Error> {
        let part = Self::Text(text.into());
        validate_part(&part)?;
        Ok(part)
    }

    pub fn inline_data(
        media_type: impl Into<String>,
        data: impl Into<Bytes>,
    ) -> Result<Self, Error> {
        let part = Self::InlineData {
            media_type: media_type.into(),
            data: data.into(),
        };
        validate_part(&part)?;
        Ok(part)
    }

    pub fn file_data(
        media_type: impl Into<String>,
        file_uri: impl Into<String>,
    ) -> Result<Self, Error> {
        let part = Self::FileData {
            media_type: media_type.into(),
            file_uri: file_uri.into(),
        };
        validate_part(&part)?;
        Ok(part)
    }
}

impl fmt::Debug for GeminiEmbeddingContentPart {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text(text) => formatter
                .debug_struct("Text")
                .field("bytes", &text.len())
                .finish(),
            Self::InlineData { media_type, data } => formatter
                .debug_struct("InlineData")
                .field("media_type_bytes", &media_type.len())
                .field("data_bytes", &data.len())
                .finish(),
            Self::FileData {
                media_type,
                file_uri,
            } => formatter
                .debug_struct("FileData")
                .field("media_type_bytes", &media_type.len())
                .field("file_uri_bytes", &file_uri.len())
                .finish(),
        }
    }
}

/// One ordered Gemini Embedding 2 multimodal request.
#[derive(Clone, PartialEq, Eq)]
pub struct GeminiMultimodalEmbeddingRequest {
    parts: Vec<GeminiEmbeddingContentPart>,
    output_dimensionality: Option<u32>,
    auto_truncate: bool,
}

impl GeminiMultimodalEmbeddingRequest {
    pub fn new(parts: impl IntoIterator<Item = GeminiEmbeddingContentPart>) -> Result<Self, Error> {
        let request = Self {
            parts: parts.into_iter().collect(),
            output_dimensionality: None,
            auto_truncate: false,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn with_output_dimensionality(mut self, dimensions: u32) -> Result<Self, Error> {
        self.output_dimensionality = Some(dimensions);
        self.validate()?;
        Ok(self)
    }

    pub fn parts(&self) -> &[GeminiEmbeddingContentPart] {
        &self.parts
    }

    pub const fn output_dimensionality(&self) -> Option<u32> {
        self.output_dimensionality
    }

    /// Allow Gemini to silently truncate input beyond the model context limit.
    pub const fn with_auto_truncate(mut self, auto_truncate: bool) -> Self {
        self.auto_truncate = auto_truncate;
        self
    }

    pub const fn auto_truncate(&self) -> bool {
        self.auto_truncate
    }

    pub fn validate(&self) -> Result<(), Error> {
        if self.parts.is_empty() || self.parts.len() > MAX_PARTS {
            return Err(invalid_input(
                "Gemini multimodal embedding requires between 1 and 64 ordered content parts",
            ));
        }
        let mut encoded_input_bytes = REQUEST_ENVELOPE_BYTES;
        for part in &self.parts {
            validate_part(part)?;
            encoded_input_bytes = encoded_input_bytes
                .checked_add(encoded_part_bytes(part).saturating_add(PART_ENVELOPE_BYTES))
                .ok_or_else(|| {
                    invalid_input("Gemini multimodal embedding input size overflowed")
                })?;
            if encoded_input_bytes > MAX_INLINE_REQUEST_BYTES {
                return Err(invalid_input(
                    "Gemini inline multimodal embedding input exceeds the 20 MB request budget",
                ));
            }
        }
        if self.output_dimensionality.is_some_and(|dimensions| {
            !(MIN_OUTPUT_DIMENSIONALITY..=MAX_OUTPUT_DIMENSIONALITY).contains(&dimensions)
        }) {
            return Err(invalid_input(
                "Gemini multimodal embedding output dimensionality must be between 128 and 3072",
            ));
        }
        Ok(())
    }
}

impl fmt::Debug for GeminiMultimodalEmbeddingRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GeminiMultimodalEmbeddingRequest")
            .field("part_count", &self.parts.len())
            .field("output_dimensionality", &self.output_dimensionality)
            .field("auto_truncate", &self.auto_truncate)
            .finish()
    }
}

/// Fully encoded Gemini v1beta multimodal embedding request.
#[derive(Clone, PartialEq)]
pub struct EncodedMultimodalEmbeddingRequest {
    target: String,
    body: Value,
}

impl fmt::Debug for EncodedMultimodalEmbeddingRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EncodedMultimodalEmbeddingRequest")
            .field("target_bytes", &self.target.len())
            .field(
                "body_field_count",
                &self.body.as_object().map_or(0, serde_json::Map::len),
            )
            .finish()
    }
}

/// One provider-reported token count for an input modality.
#[derive(Clone, PartialEq, Eq)]
pub struct GeminiEmbeddingModalityUsage {
    modality: Option<String>,
    token_count: Option<u64>,
}

impl GeminiEmbeddingModalityUsage {
    pub fn modality(&self) -> Option<&str> {
        self.modality.as_deref()
    }

    pub const fn token_count(&self) -> Option<u64> {
        self.token_count
    }
}

impl fmt::Debug for GeminiEmbeddingModalityUsage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GeminiEmbeddingModalityUsage")
            .field("modality_bytes", &self.modality.as_ref().map(String::len))
            .field("token_count", &self.token_count)
            .finish()
    }
}

impl EncodedMultimodalEmbeddingRequest {
    pub fn target(&self) -> &str {
        &self.target
    }

    pub fn body(&self) -> &Value {
        &self.body
    }

    pub fn into_parts(self) -> (String, Value) {
        (self.target, self.body)
    }
}

/// One native Gemini multimodal embedding result.
#[derive(Clone, PartialEq)]
pub struct GeminiMultimodalEmbeddingResponse {
    embedding: Vec<f32>,
    shape: Vec<u32>,
    usage: Usage,
    modality_usage: Vec<GeminiEmbeddingModalityUsage>,
}

impl GeminiMultimodalEmbeddingResponse {
    pub fn embedding(&self) -> &[f32] {
        &self.embedding
    }

    pub fn shape(&self) -> &[u32] {
        &self.shape
    }

    pub fn usage(&self) -> &Usage {
        &self.usage
    }

    pub fn modality_usage(&self) -> &[GeminiEmbeddingModalityUsage] {
        &self.modality_usage
    }
}

impl fmt::Debug for GeminiMultimodalEmbeddingResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GeminiMultimodalEmbeddingResponse")
            .field("embedding_dimensions", &self.embedding.len())
            .field("shape_rank", &self.shape.len())
            .field("usage", &self.usage)
            .field("modality_usage_count", &self.modality_usage.len())
            .finish()
    }
}

/// Encode one native Gemini v1beta multimodal embedding request.
pub fn encode_multimodal_embedding_request(
    request: &GeminiMultimodalEmbeddingRequest,
    model: &ModelId,
) -> Result<EncodedMultimodalEmbeddingRequest, Error> {
    request.validate()?;
    validate_model_segment(model)?;
    let parts = request.parts().iter().map(encode_part).collect::<Vec<_>>();
    let mut body = serde_json::Map::from_iter([
        (
            "model".to_string(),
            json!(format!("models/{}", model.as_str())),
        ),
        ("content".to_string(), json!({ "parts": parts })),
    ]);
    let mut config = serde_json::Map::from_iter([(
        "autoTruncate".to_string(),
        Value::Bool(request.auto_truncate()),
    )]);
    if let Some(dimensions) = request.output_dimensionality() {
        config.insert("outputDimensionality".to_string(), json!(dimensions));
    }
    body.insert("embedContentConfig".to_string(), Value::Object(config));
    Ok(EncodedMultimodalEmbeddingRequest {
        target: format!(
            "{V1BETA_MODELS_PREFIX}{}{EMBED_CONTENT_SUFFIX}",
            model.as_str()
        ),
        body: Value::Object(body),
    })
}

/// Decode one native Gemini v1beta multimodal embedding response.
pub fn decode_multimodal_embedding_response(
    body: &[u8],
) -> Result<GeminiMultimodalEmbeddingResponse, Error> {
    let wire = serde_json::from_slice::<EmbedContentResponseWire>(body).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            "Gemini returned an invalid v1beta multimodal embedding response",
        )
        .with_source(source)
    })?;
    if wire.embedding.values.is_empty()
        || wire.embedding.shape.len() > MAX_SHAPE_RANK
        || wire.usage_metadata.prompt_token_details.len() > MAX_MODALITY_USAGE_ITEMS
    {
        return Err(Error::protocol_violation(
            "Gemini multimodal embedding response exceeds the bounded native result shape",
        ));
    }
    Ok(GeminiMultimodalEmbeddingResponse {
        embedding: wire.embedding.values,
        shape: wire.embedding.shape,
        usage: Usage::default().with_input_tokens(wire.usage_metadata.prompt_token_count),
        modality_usage: wire
            .usage_metadata
            .prompt_token_details
            .into_iter()
            .map(|detail| GeminiEmbeddingModalityUsage {
                modality: detail.modality,
                token_count: detail.token_count,
            })
            .collect(),
    })
}

fn encode_part(part: &GeminiEmbeddingContentPart) -> Value {
    match part {
        GeminiEmbeddingContentPart::Text(text) => json!({ "text": text }),
        GeminiEmbeddingContentPart::InlineData { media_type, data } => json!({
            "inlineData": {
                "mimeType": media_type,
                "data": base64::engine::general_purpose::STANDARD.encode(data),
            }
        }),
        GeminiEmbeddingContentPart::FileData {
            media_type,
            file_uri,
        } => json!({
            "fileData": {
                "mimeType": media_type,
                "fileUri": file_uri,
            }
        }),
    }
}

fn validate_part(part: &GeminiEmbeddingContentPart) -> Result<(), Error> {
    match part {
        GeminiEmbeddingContentPart::Text(text) => {
            if text.trim().is_empty() {
                return Err(invalid_input(
                    "Gemini multimodal embedding text must be non-empty",
                ));
            }
        }
        GeminiEmbeddingContentPart::InlineData { media_type, data } => {
            validate_media_type(media_type)?;
            if data.is_empty() {
                return Err(invalid_input(
                    "Gemini multimodal embedding inline data must not be empty",
                ));
            }
        }
        GeminiEmbeddingContentPart::FileData {
            media_type,
            file_uri,
        } => {
            validate_media_type(media_type)?;
            if file_uri.trim().is_empty()
                || file_uri.len() > MAX_FILE_URI_BYTES
                || file_uri.chars().any(char::is_control)
            {
                return Err(invalid_input(
                    "Gemini multimodal embedding file URI is invalid or exceeds 8 KiB",
                ));
            }
        }
    }
    Ok(())
}

fn validate_media_type(media_type: &str) -> Result<(), Error> {
    let supported = media_type == "application/pdf"
        || media_type.starts_with("image/")
        || media_type.starts_with("audio/")
        || media_type.starts_with("video/");
    if media_type.trim() != media_type
        || media_type.len() > 256
        || media_type.chars().any(char::is_control)
        || !supported
    {
        return Err(invalid_input(
            "Gemini multimodal embedding media type must be image, audio, video, or PDF",
        ));
    }
    Ok(())
}

fn encoded_part_bytes(part: &GeminiEmbeddingContentPart) -> usize {
    match part {
        GeminiEmbeddingContentPart::Text(text) => text.len(),
        GeminiEmbeddingContentPart::InlineData { media_type, data } => media_type
            .len()
            .saturating_add((data.len().saturating_add(2) / 3).saturating_mul(4)),
        GeminiEmbeddingContentPart::FileData {
            media_type,
            file_uri,
        } => media_type.len().saturating_add(file_uri.len()),
    }
}

fn validate_model_segment(model: &ModelId) -> Result<(), Error> {
    let safe = model
        .as_str()
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~'));
    if safe {
        Ok(())
    } else {
        Err(invalid_input(
            "Gemini multimodal embedding model ID is not a valid resource path segment",
        ))
    }
}

fn invalid_input(message: &'static str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct EmbedContentResponseWire {
    embedding: ContentEmbeddingWire,
    #[serde(default)]
    usage_metadata: EmbeddingUsageMetadataWire,
}

#[derive(Deserialize)]
struct ContentEmbeddingWire {
    values: Vec<f32>,
    #[serde(default)]
    shape: Vec<u32>,
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EmbeddingUsageMetadataWire {
    prompt_token_count: Option<u64>,
    #[serde(default)]
    prompt_token_details: Vec<ModalityTokenCountWire>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModalityTokenCountWire {
    #[serde(default)]
    modality: Option<String>,
    #[serde(default)]
    token_count: Option<u64>,
}

#[cfg(test)]
mod tests {
    use siumai_core::{ErrorKind, UsageValue};

    use super::*;

    #[test]
    fn mixed_parts_encode_in_order_with_v1beta_config() {
        let request = GeminiMultimodalEmbeddingRequest::new([
            GeminiEmbeddingContentPart::text("caption").unwrap(),
            GeminiEmbeddingContentPart::inline_data("image/png", vec![0_u8, 1, 2]).unwrap(),
            GeminiEmbeddingContentPart::file_data(
                "application/pdf",
                "https://files.example/document",
            )
            .unwrap(),
        ])
        .unwrap()
        .with_output_dimensionality(768)
        .unwrap();
        let encoded = encode_multimodal_embedding_request(
            &request,
            &ModelId::new("gemini-embedding-2").unwrap(),
        )
        .unwrap();

        assert_eq!(
            encoded.target(),
            "v1beta/models/gemini-embedding-2:embedContent"
        );
        assert_eq!(
            encoded.body(),
            &json!({
                "model": "models/gemini-embedding-2",
                "content": {
                    "parts": [
                        { "text": "caption" },
                        { "inlineData": { "mimeType": "image/png", "data": "AAEC" } },
                        { "fileData": {
                            "mimeType": "application/pdf",
                            "fileUri": "https://files.example/document"
                        } }
                    ]
                },
                "embedContentConfig": {
                    "autoTruncate": false,
                    "outputDimensionality": 768
                }
            })
        );
    }

    #[test]
    fn request_and_part_debug_redact_caller_content() {
        let part = GeminiEmbeddingContentPart::file_data(
            "application/pdf",
            "https://files.example/canary-secret",
        )
        .unwrap();
        let request = GeminiMultimodalEmbeddingRequest::new([
            GeminiEmbeddingContentPart::text("canary-text").unwrap(),
            part.clone(),
        ])
        .unwrap();
        assert!(!format!("{part:?}").contains("canary-secret"));
        assert!(!format!("{request:?}").contains("canary-text"));
        let encoded = encode_multimodal_embedding_request(
            &request,
            &ModelId::new("gemini-embedding-2").unwrap(),
        )
        .unwrap();
        let debug = format!("{encoded:?}");
        assert!(!debug.contains("canary-secret"));
        assert!(!debug.contains("canary-text"));
    }

    #[test]
    fn invalid_media_dimensions_and_oversized_inline_input_fail() {
        assert_eq!(
            GeminiEmbeddingContentPart::inline_data("text/plain", vec![1_u8])
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidInput
        );
        assert_eq!(
            GeminiMultimodalEmbeddingRequest::new([
                GeminiEmbeddingContentPart::text("hello").unwrap()
            ])
            .unwrap()
            .with_output_dimensionality(64)
            .unwrap_err()
            .kind(),
            ErrorKind::InvalidInput
        );
        let too_large = vec![0_u8; 16 * 1024 * 1024];
        assert_eq!(
            GeminiMultimodalEmbeddingRequest::new([GeminiEmbeddingContentPart::inline_data(
                "image/png",
                too_large
            )
            .unwrap()])
            .unwrap_err()
            .kind(),
            ErrorKind::InvalidInput
        );
    }

    #[test]
    fn response_preserves_vector_shape_and_unknown_usage() {
        let response = decode_multimodal_embedding_response(
            br#"{
            "embedding":{"values":[0.1,0.2],"shape":[2]},
            "usageMetadata": {
                "promptTokenDetails": [{"modality":"IMAGE","tokenCount":4}]
            }
        }"#,
        )
        .unwrap();
        assert_eq!(response.embedding(), &[0.1_f32, 0.2_f32]);
        assert_eq!(response.shape(), &[2]);
        assert_eq!(response.usage().input_tokens, UsageValue::Unknown);
        assert_eq!(response.modality_usage()[0].modality(), Some("IMAGE"));
        assert_eq!(response.modality_usage()[0].token_count(), Some(4));
    }
}
