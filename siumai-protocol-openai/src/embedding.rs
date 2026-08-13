//! OpenAI text embedding request and response codecs.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use siumai_core::{
    EmbeddingRequest, EmbeddingResponse, Error, ErrorKind, ModelId, ResponseMetadata, Usage,
};

/// OpenAI Embeddings API mode identifier.
pub const API_MODE_ID: &str = "embeddings";
/// OpenAI Embeddings protocol identifier.
pub const PROTOCOL_ID: &str = "openai.embeddings";
/// Relative OpenAI Embeddings endpoint.
pub const TARGET: &str = "embeddings";

/// Checked provider-owned fields for one embedding request.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EmbeddingConfig {
    user: Option<String>,
}

impl EmbeddingConfig {
    pub const fn new() -> Self {
        Self { user: None }
    }

    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }
}

/// Encode one portable text embedding request using float vectors.
pub fn encode_embedding_request(
    request: &EmbeddingRequest,
    model: &ModelId,
    config: &EmbeddingConfig,
) -> Result<Value, Error> {
    serde_json::to_value(EmbeddingRequestWire {
        model: model.as_str(),
        input: request.inputs(),
        encoding_format: "float",
        dimensions: request.dimensions().map(std::num::NonZeroU32::get),
        user: config.user.as_deref(),
    })
    .map_err(|source| {
        Error::new(
            ErrorKind::Internal,
            "OpenAI embedding request could not be serialized",
        )
        .with_source(source)
    })
}

/// Decode an OpenAI embedding response and restore input ordering by index.
pub fn decode_embedding_response(
    body: &[u8],
    request: &EmbeddingRequest,
    requested_model: &ModelId,
) -> Result<EmbeddingResponse, Error> {
    let response = serde_json::from_slice::<EmbeddingResponseWire>(body).map_err(|source| {
        Error::protocol_violation("OpenAI embedding response is not valid JSON").with_source(source)
    })?;
    if response
        .object
        .as_deref()
        .is_some_and(|value| value != "list")
    {
        return Err(Error::protocol_violation(
            "OpenAI embedding response has an unexpected object type",
        ));
    }

    let expected = request.inputs().len();
    let mut ordered = vec![None; expected];
    for item in response.data {
        if item
            .object
            .as_deref()
            .is_some_and(|value| value != "embedding")
        {
            return Err(Error::protocol_violation(
                "OpenAI embedding response contains an unexpected item type",
            ));
        }
        let slot = ordered.get_mut(item.index).ok_or_else(|| {
            Error::protocol_violation("OpenAI embedding response index is out of range")
        })?;
        if slot.replace(item.embedding).is_some() {
            return Err(Error::protocol_violation(
                "OpenAI embedding response contains a duplicate index",
            ));
        }
    }
    let embeddings = ordered
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            Error::protocol_violation("OpenAI embedding response is missing an input index")
        })?;
    let model = match response.model {
        Some(model) => Some(ModelId::new(model).map_err(|source| {
            Error::protocol_violation("OpenAI embedding response contains an invalid model ID")
                .with_source(source)
        })?),
        None => Some(requested_model.clone()),
    };
    let usage = response.usage.unwrap_or_default();
    let decoded = EmbeddingResponse {
        embeddings,
        metadata: ResponseMetadata {
            response_id: None,
            request_id: None,
            model,
        },
        usage: Usage::default()
            .with_input_tokens(usage.prompt_tokens)
            .with_total_tokens(usage.total_tokens),
        warnings: Vec::new(),
        provider: Default::default(),
    };
    decoded.validate(request)?;
    Ok(decoded)
}

#[derive(Debug, Serialize)]
struct EmbeddingRequestWire<'a> {
    model: &'a str,
    input: &'a [String],
    encoding_format: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<&'a str>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponseWire {
    #[serde(default)]
    object: Option<String>,
    data: Vec<EmbeddingItemWire>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    usage: Option<EmbeddingUsageWire>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingItemWire {
    #[serde(default)]
    object: Option<String>,
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Debug, Default, Deserialize)]
struct EmbeddingUsageWire {
    #[serde(default)]
    prompt_tokens: Option<u64>,
    #[serde(default)]
    total_tokens: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_uses_float_encoding_and_restores_index_order() {
        let request = EmbeddingRequest::new(["first", "second"])
            .unwrap()
            .with_dimensions(2)
            .unwrap();
        let model = ModelId::new("text-embedding-3-small").unwrap();
        let encoded = encode_embedding_request(
            &request,
            &model,
            &EmbeddingConfig::new().with_user("user-1"),
        )
        .unwrap();
        assert_eq!(encoded["encoding_format"], "float");
        assert_eq!(encoded["dimensions"], 2);
        assert_eq!(encoded["user"], "user-1");

        let decoded = decode_embedding_response(
            br#"{"object":"list","data":[{"object":"embedding","embedding":[3.0,4.0],"index":1},{"object":"embedding","embedding":[1.0,2.0],"index":0}],"model":"text-embedding-3-small","usage":{"prompt_tokens":5,"total_tokens":5}}"#,
            &request,
            &model,
        )
        .unwrap();
        assert_eq!(decoded.embeddings, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    }

    #[test]
    fn decoder_rejects_duplicate_indices() {
        let request = EmbeddingRequest::new(["first", "second"]).unwrap();
        let model = ModelId::new("future-embedding-model").unwrap();
        let error = decode_embedding_response(
            br#"{"data":[{"embedding":[1.0],"index":0},{"embedding":[2.0],"index":0}]}"#,
            &request,
            &model,
        )
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ProtocolViolation);
    }
}
