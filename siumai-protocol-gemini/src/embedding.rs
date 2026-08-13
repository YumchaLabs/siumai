//! Stable Gemini v1 embedding request and response codecs.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use siumai_core::{
    EmbeddingRequest, EmbeddingResponse, Error, ErrorKind, ModelId, ResponseMetadata, Usage,
    Warning,
};

const STABLE_V1_MODELS_PREFIX: &str = "v1/models/";
const EMBED_CONTENT_SUFFIX: &str = ":embedContent";
const BATCH_EMBED_CONTENTS_SUFFIX: &str = ":batchEmbedContents";
const GEMINI_EMBEDDING_2: &str = "gemini-embedding-2";
const GEMINI_EMBEDDING_001: &str = "gemini-embedding-001";
const MIN_KNOWN_OUTPUT_DIMENSIONALITY: u32 = 128;
const MAX_KNOWN_OUTPUT_DIMENSIONALITY: u32 = 3_072;
const MAX_TITLE_BYTES: usize = 16 * 1024;
const MAX_PROVIDER_METADATA_BYTES: usize = 64 * 1024;
const MAX_PROMPT_TOKEN_DETAILS: usize = 32;
const MAX_MODALITY_BYTES: usize = 64;
const MAX_SHAPE_RANK: usize = 16;

/// Task types accepted by the stable `EmbedContentConfig` schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EmbedContentTaskType {
    /// Embed a query for asymmetric retrieval.
    RetrievalQuery,
    /// Embed a document for asymmetric retrieval.
    RetrievalDocument,
    /// Embed text for symmetric semantic comparison.
    SemanticSimilarity,
    /// Embed text for classification.
    Classification,
    /// Embed text for clustering.
    Clustering,
    /// Embed text for question answering.
    QuestionAnswering,
    /// Embed text for fact verification.
    FactVerification,
    /// Embed a natural-language query for code retrieval.
    CodeRetrievalQuery,
}

impl EmbedContentTaskType {
    const fn as_wire(self) -> &'static str {
        match self {
            Self::RetrievalQuery => "RETRIEVAL_QUERY",
            Self::RetrievalDocument => "RETRIEVAL_DOCUMENT",
            Self::SemanticSimilarity => "SEMANTIC_SIMILARITY",
            Self::Classification => "CLASSIFICATION",
            Self::Clustering => "CLUSTERING",
            Self::QuestionAnswering => "QUESTION_ANSWERING",
            Self::FactVerification => "FACT_VERIFICATION",
            Self::CodeRetrievalQuery => "CODE_RETRIEVAL_QUERY",
        }
    }
}

/// Checked stable v1 `EmbedContentConfig` values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedContentConfig {
    task_type: Option<EmbedContentTaskType>,
    title: Option<String>,
    auto_truncate: bool,
}

impl Default for EmbedContentConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbedContentConfig {
    /// Create a fail-closed configuration that disables silent truncation.
    pub const fn new() -> Self {
        Self {
            task_type: None,
            title: None,
            auto_truncate: false,
        }
    }

    pub const fn with_task_type(mut self, task_type: EmbedContentTaskType) -> Self {
        self.task_type = Some(task_type);
        self
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub const fn with_auto_truncate(mut self, auto_truncate: bool) -> Self {
        self.auto_truncate = auto_truncate;
        self
    }

    pub const fn task_type(&self) -> Option<EmbedContentTaskType> {
        self.task_type
    }

    pub fn title(&self) -> Option<&str> {
        self.title.as_deref()
    }

    pub const fn auto_truncate(&self) -> bool {
        self.auto_truncate
    }
}

/// Stable v1 synchronous embedding operation selected by input cardinality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingRequestMode {
    /// One `embedContent` request and one embedding response.
    Single,
    /// One `batchEmbedContents` request containing multiple inputs.
    Batch,
}

/// Fully encoded stable v1 embedding request.
#[derive(Debug, Clone, PartialEq)]
pub struct EncodedEmbeddingRequest {
    target: String,
    body: Value,
    mode: EmbeddingRequestMode,
}

impl EncodedEmbeddingRequest {
    pub fn target(&self) -> &str {
        &self.target
    }

    pub fn body(&self) -> &Value {
        &self.body
    }

    pub const fn mode(&self) -> EmbeddingRequestMode {
        self.mode
    }

    pub fn into_parts(self) -> (String, Value, EmbeddingRequestMode) {
        (self.target, self.body, self.mode)
    }
}

/// Encode one portable text embedding request using stable Gemini v1.
pub fn encode_embedding_request(
    request: &EmbeddingRequest,
    model: &ModelId,
    config: &EmbedContentConfig,
) -> Result<EncodedEmbeddingRequest, Error> {
    validate_model_segment(model)?;
    validate_config(
        model,
        request.dimensions().map(std::num::NonZeroU32::get),
        config,
    )?;

    let resource = format!("models/{}", model.as_str());
    let mode = if request.inputs().len() == 1 {
        EmbeddingRequestMode::Single
    } else {
        EmbeddingRequestMode::Batch
    };
    let suffix = match mode {
        EmbeddingRequestMode::Single => EMBED_CONTENT_SUFFIX,
        EmbeddingRequestMode::Batch => BATCH_EMBED_CONTENTS_SUFFIX,
    };
    let target = format!("{STABLE_V1_MODELS_PREFIX}{}{suffix}", model.as_str());
    let body = match mode {
        EmbeddingRequestMode::Single => {
            let input = request.inputs().first().ok_or_else(|| {
                Error::new(
                    ErrorKind::Internal,
                    "validated embedding request unexpectedly contains no inputs",
                )
            })?;
            serde_json::to_value(EmbedContentRequestWire::new(
                &resource,
                input,
                request.dimensions().map(std::num::NonZeroU32::get),
                config,
            ))
        }
        EmbeddingRequestMode::Batch => {
            let requests = request
                .inputs()
                .iter()
                .map(|input| {
                    EmbedContentRequestWire::new(
                        &resource,
                        input,
                        request.dimensions().map(std::num::NonZeroU32::get),
                        config,
                    )
                })
                .collect::<Vec<_>>();
            serde_json::to_value(BatchEmbedContentsRequestWire { requests })
        }
    }
    .map_err(|source| {
        Error::new(
            ErrorKind::Internal,
            "Gemini embedding request could not be serialized",
        )
        .with_source(source)
    })?;

    Ok(EncodedEmbeddingRequest { target, body, mode })
}

/// Decode one stable Gemini v1 embedding response into the portable contract.
pub fn decode_embedding_response(
    body: &[u8],
    mode: EmbeddingRequestMode,
    request: &EmbeddingRequest,
    model: &ModelId,
) -> Result<EmbeddingResponse, Error> {
    let (embeddings, usage) = match mode {
        EmbeddingRequestMode::Single => {
            let response = serde_json::from_slice::<EmbedContentResponseWire>(body)
                .map_err(invalid_response)?;
            (vec![response.embedding], response.usage_metadata)
        }
        EmbeddingRequestMode::Batch => {
            let response = serde_json::from_slice::<BatchEmbedContentsResponseWire>(body)
                .map_err(invalid_response)?;
            (response.embeddings, response.usage_metadata)
        }
    };

    let (values, shapes): (Vec<_>, Vec<_>) = embeddings
        .into_iter()
        .map(|embedding| (embedding.values, embedding.shape))
        .unzip();
    let (provider, warnings) = bounded_provider_metadata(&shapes, &usage.prompt_token_details);
    let response = EmbeddingResponse {
        embeddings: values,
        metadata: ResponseMetadata {
            response_id: None,
            request_id: None,
            model: Some(model.clone()),
        },
        usage: Usage::default().with_input_tokens(usage.prompt_token_count),
        warnings,
        provider,
    };
    response.validate(request)?;
    Ok(response)
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EmbedContentRequestWire<'a> {
    model: &'a str,
    content: ContentWire<'a>,
    embed_content_config: EmbedContentConfigWire<'a>,
}

impl<'a> EmbedContentRequestWire<'a> {
    fn new(
        model: &'a str,
        input: &'a str,
        output_dimensionality: Option<u32>,
        config: &'a EmbedContentConfig,
    ) -> Self {
        Self {
            model,
            content: ContentWire {
                parts: [TextPartWire { text: input }],
            },
            embed_content_config: EmbedContentConfigWire {
                output_dimensionality,
                task_type: config.task_type.map(EmbedContentTaskType::as_wire),
                title: config.title.as_deref(),
                auto_truncate: config.auto_truncate,
            },
        }
    }
}

#[derive(Debug, Serialize)]
struct ContentWire<'a> {
    parts: [TextPartWire<'a>; 1],
}

#[derive(Debug, Serialize)]
struct TextPartWire<'a> {
    text: &'a str,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EmbedContentConfigWire<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    output_dimensionality: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    task_type: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<&'a str>,
    auto_truncate: bool,
}

#[derive(Debug, Serialize)]
struct BatchEmbedContentsRequestWire<'a> {
    requests: Vec<EmbedContentRequestWire<'a>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EmbedContentResponseWire {
    embedding: ContentEmbeddingWire,
    #[serde(default)]
    usage_metadata: EmbeddingUsageMetadataWire,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BatchEmbedContentsResponseWire {
    embeddings: Vec<ContentEmbeddingWire>,
    #[serde(default)]
    usage_metadata: EmbeddingUsageMetadataWire,
}

#[derive(Debug, Deserialize)]
struct ContentEmbeddingWire {
    values: Vec<f32>,
    #[serde(default)]
    shape: Vec<u32>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EmbeddingUsageMetadataWire {
    prompt_token_count: Option<u64>,
    #[serde(default)]
    prompt_token_details: Vec<ModalityTokenCountWire>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModalityTokenCountWire {
    #[serde(default)]
    modality: Option<String>,
    #[serde(default)]
    token_count: Option<u64>,
}

fn validate_model_segment(model: &ModelId) -> Result<(), Error> {
    let safe = model
        .as_str()
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~'));
    if safe {
        Ok(())
    } else {
        Err(Error::new(
            ErrorKind::InvalidInput,
            "Gemini embedding model ID is not a valid resource path segment",
        ))
    }
}

fn validate_config(
    model: &ModelId,
    output_dimensionality: Option<u32>,
    config: &EmbedContentConfig,
) -> Result<(), Error> {
    if model.as_str() == GEMINI_EMBEDDING_2 && config.task_type.is_some() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "gemini-embedding-2 does not accept the taskType field",
        ));
    }
    if let Some(dimensions) = output_dimensionality
        && matches!(model.as_str(), GEMINI_EMBEDDING_2 | GEMINI_EMBEDDING_001)
        && !(MIN_KNOWN_OUTPUT_DIMENSIONALITY..=MAX_KNOWN_OUTPUT_DIMENSIONALITY)
            .contains(&dimensions)
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "requested dimensions are not supported by this known Gemini embedding model",
        ));
    }
    if let Some(title) = config.title.as_deref()
        && (title.trim().is_empty() || title.len() > MAX_TITLE_BYTES)
    {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Gemini embedding title must be non-empty and within the local option budget",
        ));
    }
    Ok(())
}

fn invalid_response(source: serde_json::Error) -> Error {
    Error::new(
        ErrorKind::Protocol,
        "Gemini returned an invalid stable v1 embedding response",
    )
    .with_source(source)
}

fn bounded_provider_metadata(
    shapes: &[Vec<u32>],
    prompt_token_details: &[ModalityTokenCountWire],
) -> (BTreeMap<String, Value>, Vec<Warning>) {
    let metadata_is_bounded = shapes.iter().all(|shape| shape.len() <= MAX_SHAPE_RANK)
        && prompt_token_details.len() <= MAX_PROMPT_TOKEN_DETAILS
        && prompt_token_details.iter().all(|detail| {
            detail
                .modality
                .as_ref()
                .is_none_or(|modality| modality.len() <= MAX_MODALITY_BYTES)
        });
    if !metadata_is_bounded {
        return omitted_provider_metadata();
    }

    let mut google = serde_json::Map::new();
    if shapes.iter().any(|shape| !shape.is_empty()) {
        google.insert("embeddingShapes".to_string(), serde_json::json!(shapes));
    }
    if !prompt_token_details.is_empty() {
        google.insert(
            "promptTokenDetails".to_string(),
            serde_json::json!(prompt_token_details),
        );
    }
    if google.is_empty() {
        return (BTreeMap::new(), Vec::new());
    }

    let value = Value::Object(google);
    let within_byte_limit = serde_json::to_vec(&value)
        .is_ok_and(|encoded| encoded.len() <= MAX_PROVIDER_METADATA_BYTES);
    if within_byte_limit {
        (BTreeMap::from([("google".to_string(), value)]), Vec::new())
    } else {
        omitted_provider_metadata()
    }
}

fn omitted_provider_metadata() -> (BTreeMap<String, Value>, Vec<Warning>) {
    (
        BTreeMap::new(),
        vec![Warning::provider(
            "provider_metadata_omitted",
            "Gemini embedding metadata exceeded the portable response budget",
        )],
    )
}

#[cfg(test)]
mod tests {
    use siumai_core::{EmbeddingRequest, ErrorKind, ModelId, UsageValue};

    use super::*;

    #[test]
    fn single_embedding_uses_stable_v1_nested_config_and_retains_usage() {
        let model = ModelId::new(GEMINI_EMBEDDING_2).unwrap();
        let request = EmbeddingRequest::single("hello")
            .unwrap()
            .with_dimensions(128)
            .unwrap();
        let encoded = encode_embedding_request(&request, &model, &EmbedContentConfig::new())
            .expect("encode single embedding request");

        assert_eq!(
            encoded.target(),
            "v1/models/gemini-embedding-2:embedContent"
        );
        assert_eq!(encoded.mode(), EmbeddingRequestMode::Single);
        assert_eq!(encoded.body()["model"], "models/gemini-embedding-2");
        assert_eq!(
            encoded.body()["embedContentConfig"]["outputDimensionality"],
            128
        );
        assert_eq!(encoded.body()["embedContentConfig"]["autoTruncate"], false);
        assert!(encoded.body().get("outputDimensionality").is_none());

        let body = serde_json::to_vec(&serde_json::json!({
            "embedding": {
                "values": vec![0.25_f32; 128],
                "shape": [128]
            },
            "usageMetadata": {
                "promptTokenCount": 7,
                "promptTokenDetails": [{"modality": "TEXT", "tokenCount": 7}]
            }
        }))
        .unwrap();
        let response = decode_embedding_response(&body, encoded.mode(), &request, &model).unwrap();

        assert_eq!(response.embeddings.len(), 1);
        assert_eq!(response.embeddings[0].len(), 128);
        assert_eq!(response.usage.input_tokens, UsageValue::Known(7));
        assert_eq!(response.usage.total_tokens, UsageValue::Unknown);
        assert_eq!(
            response.provider["google"]["promptTokenDetails"][0]["modality"],
            "TEXT"
        );
    }

    #[test]
    fn batch_embedding_preserves_request_and_response_order_for_future_models() {
        let model = ModelId::new("future-gemini-embedding").unwrap();
        let request = EmbeddingRequest::new(["first", "second"]).unwrap();
        let encoded = encode_embedding_request(&request, &model, &EmbedContentConfig::new())
            .expect("encode batch embedding request");

        assert_eq!(
            encoded.target(),
            "v1/models/future-gemini-embedding:batchEmbedContents"
        );
        assert_eq!(encoded.mode(), EmbeddingRequestMode::Batch);
        assert_eq!(
            encoded.body()["requests"][0]["content"]["parts"][0]["text"],
            "first"
        );
        assert_eq!(
            encoded.body()["requests"][1]["content"]["parts"][0]["text"],
            "second"
        );
        assert_eq!(
            encoded.body()["requests"][0]["embedContentConfig"]["autoTruncate"],
            false
        );

        let body = serde_json::to_vec(&serde_json::json!({
            "embeddings": [
                {"values": [0.1, 0.2]},
                {"values": [0.3, 0.4]}
            ],
            "usageMetadata": {"promptTokenCount": 5}
        }))
        .unwrap();
        let response = decode_embedding_response(&body, encoded.mode(), &request, &model).unwrap();

        assert_eq!(
            response.embeddings,
            vec![vec![0.1_f32, 0.2_f32], vec![0.3_f32, 0.4_f32]]
        );
        assert_eq!(response.usage.input_tokens, UsageValue::Known(5));
    }

    #[test]
    fn gemini_embedding_2_rejects_task_type_before_submission() {
        let model = ModelId::new(GEMINI_EMBEDDING_2).unwrap();
        let request = EmbeddingRequest::single("hello").unwrap();
        let error = encode_embedding_request(
            &request,
            &model,
            &EmbedContentConfig::new().with_task_type(EmbedContentTaskType::SemanticSimilarity),
        )
        .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::InvalidInput);
    }
}
