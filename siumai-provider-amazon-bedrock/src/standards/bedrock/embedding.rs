//! Amazon Bedrock embedding standard aligned with the AI SDK runtime contract.
//!
//! Reference:
//! `repo-ref/ai/packages/amazon-bedrock/src/bedrock-embedding-model.ts`

use crate::core::{EmbeddingTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::transformers::request::RequestTransformer;
use crate::execution::transformers::response::ResponseTransformer;
use crate::provider_options::{
    BedrockEmbeddingInputType, BedrockEmbeddingOptions, BedrockEmbeddingPurpose,
    BedrockEmbeddingTruncate,
};
use crate::types::{EmbeddingRequest, EmbeddingResponse, EmbeddingTaskType, EmbeddingUsage};
use reqwest::header::HeaderMap;
use std::collections::HashMap;
use std::sync::Arc;

const TITAN_DIMENSIONS: &[u32] = &[1024, 512, 256];
const NOVA_DIMENSIONS: &[u32] = &[256, 384, 1024, 3072];
const COHERE_OUTPUT_DIMENSIONS: &[u32] = &[256, 512, 1024, 1536];

#[derive(Clone, Default)]
pub struct BedrockEmbeddingStandard;

impl BedrockEmbeddingStandard {
    pub fn new() -> Self {
        Self
    }

    pub fn create_transformers(&self, provider_id: &str) -> EmbeddingTransformers {
        EmbeddingTransformers {
            request: Arc::new(BedrockEmbeddingRequestTransformer {
                provider_id: provider_id.to_string(),
            }),
            response: Arc::new(BedrockEmbeddingResponseTransformer {
                provider_id: provider_id.to_string(),
            }),
        }
    }

    pub fn create_spec(&self, provider_id: &'static str) -> BedrockEmbeddingSpec {
        BedrockEmbeddingSpec { provider_id }
    }
}

pub struct BedrockEmbeddingSpec {
    provider_id: &'static str,
}

impl ProviderSpec for BedrockEmbeddingSpec {
    fn id(&self) -> &'static str {
        self.provider_id
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new().with_embedding()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        crate::standards::bedrock::headers::build_bedrock_json_headers(ctx)
    }

    fn classify_http_error(
        &self,
        status: u16,
        body_text: &str,
        headers: &HeaderMap,
    ) -> Option<LlmError> {
        crate::standards::bedrock::errors::classify_bedrock_http_error(
            self.provider_id,
            status,
            body_text,
            headers,
        )
    }

    fn try_embedding_url(
        &self,
        req: &EmbeddingRequest,
        ctx: &ProviderContext,
    ) -> Result<String, LlmError> {
        let model = req.model.as_deref().unwrap_or_default();
        let encoded_model = urlencoding::encode(model);
        Ok(crate::utils::url::join_url(
            &ctx.base_url,
            &format!("/model/{encoded_model}/invoke"),
        ))
    }

    fn choose_embedding_transformers(
        &self,
        _req: &EmbeddingRequest,
        ctx: &ProviderContext,
    ) -> EmbeddingTransformers {
        BedrockEmbeddingStandard::new().create_transformers(&ctx.provider_id)
    }
}

struct BedrockEmbeddingRequestTransformer {
    provider_id: String,
}

impl BedrockEmbeddingRequestTransformer {
    fn parse_options(req: &EmbeddingRequest) -> Result<BedrockEmbeddingOptions, LlmError> {
        let Some(value) = req.provider_options_map.get("bedrock") else {
            return Ok(BedrockEmbeddingOptions::default());
        };

        serde_json::from_value::<BedrockEmbeddingOptions>(value.clone()).map_err(|error| {
            LlmError::InvalidParameter(format!(
                "Invalid Amazon Bedrock embedding options in providerOptions.bedrock: {error}"
            ))
        })
    }

    fn resolved_model(req: &EmbeddingRequest) -> Result<&str, LlmError> {
        req.model
            .as_deref()
            .filter(|model| !model.trim().is_empty())
            .ok_or_else(|| {
                LlmError::ConfigurationError(
                    "Bedrock embedding request requires a non-empty model id".to_string(),
                )
            })
    }

    fn validate_single_input(req: &EmbeddingRequest) -> Result<&str, LlmError> {
        if req.input.len() != 1 {
            return Err(LlmError::InvalidInput(format!(
                "Amazon Bedrock embedding requests support exactly 1 input per call, got {}",
                req.input.len()
            )));
        }

        Ok(req.input[0].as_str())
    }

    fn validate_dimension(field: &str, value: u32, allowed: &[u32]) -> Result<u32, LlmError> {
        if allowed.contains(&value) {
            Ok(value)
        } else {
            Err(LlmError::InvalidParameter(format!(
                "Invalid Amazon Bedrock embedding option `{field}={value}`. Supported values: {}",
                allowed
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            )))
        }
    }

    fn is_nova_model(model: &str) -> bool {
        model.starts_with("amazon.nova-") && model.contains("embed")
    }

    fn is_cohere_model(model: &str) -> bool {
        model.starts_with("cohere.embed-")
    }

    fn cohere_input_type(
        req: &EmbeddingRequest,
        options: &BedrockEmbeddingOptions,
    ) -> BedrockEmbeddingInputType {
        if let Some(input_type) = options.input_type {
            return input_type;
        }

        match req.task_type {
            Some(EmbeddingTaskType::RetrievalDocument) => BedrockEmbeddingInputType::SearchDocument,
            Some(EmbeddingTaskType::Classification) => BedrockEmbeddingInputType::Classification,
            Some(EmbeddingTaskType::Clustering) => BedrockEmbeddingInputType::Clustering,
            _ => BedrockEmbeddingInputType::SearchQuery,
        }
    }

    fn nova_purpose(
        req: &EmbeddingRequest,
        options: &BedrockEmbeddingOptions,
    ) -> BedrockEmbeddingPurpose {
        options.embedding_purpose.unwrap_or(match req.task_type {
            Some(EmbeddingTaskType::Classification) => BedrockEmbeddingPurpose::Classification,
            Some(EmbeddingTaskType::Clustering) => BedrockEmbeddingPurpose::Clustering,
            _ => BedrockEmbeddingPurpose::GenericIndex,
        })
    }

    fn titan_dimensions(
        req: &EmbeddingRequest,
        options: &BedrockEmbeddingOptions,
    ) -> Result<Option<u32>, LlmError> {
        options
            .dimensions
            .or(req.dimensions)
            .map(|value| Self::validate_dimension("dimensions", value, TITAN_DIMENSIONS))
            .transpose()
    }

    fn nova_dimension(
        req: &EmbeddingRequest,
        options: &BedrockEmbeddingOptions,
    ) -> Result<u32, LlmError> {
        Self::validate_dimension(
            "embeddingDimension",
            options
                .embedding_dimension
                .or(req.dimensions)
                .unwrap_or(1024),
            NOVA_DIMENSIONS,
        )
    }

    fn cohere_output_dimension(
        req: &EmbeddingRequest,
        options: &BedrockEmbeddingOptions,
    ) -> Result<Option<u32>, LlmError> {
        options
            .output_dimension
            .or(req.dimensions)
            .map(|value| {
                Self::validate_dimension("outputDimension", value, COHERE_OUTPUT_DIMENSIONS)
            })
            .transpose()
    }
}

impl RequestTransformer for BedrockEmbeddingRequestTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_chat(
        &self,
        _req: &crate::types::ChatRequest,
    ) -> Result<serde_json::Value, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "bedrock embedding transformer does not implement chat".to_string(),
        ))
    }

    fn transform_embedding(&self, req: &EmbeddingRequest) -> Result<serde_json::Value, LlmError> {
        let model = Self::resolved_model(req)?;
        let input = Self::validate_single_input(req)?;
        let options = Self::parse_options(req)?;

        if Self::is_nova_model(model) {
            return Ok(serde_json::json!({
                "taskType": "SINGLE_EMBEDDING",
                "singleEmbeddingParams": {
                    "embeddingPurpose": Self::nova_purpose(req, &options),
                    "embeddingDimension": Self::nova_dimension(req, &options)?,
                    "text": {
                        "truncationMode": options.truncate.unwrap_or(BedrockEmbeddingTruncate::End),
                        "value": input,
                    },
                },
            }));
        }

        if Self::is_cohere_model(model) {
            let mut body = serde_json::Map::new();
            body.insert(
                "input_type".to_string(),
                serde_json::to_value(Self::cohere_input_type(req, &options))
                    .expect("serialize Cohere input type"),
            );
            body.insert("texts".to_string(), serde_json::json!([input]));
            if let Some(truncate) = options.truncate {
                body.insert(
                    "truncate".to_string(),
                    serde_json::to_value(truncate).expect("serialize BedrockEmbeddingTruncate"),
                );
            }
            if let Some(output_dimension) = Self::cohere_output_dimension(req, &options)? {
                body.insert(
                    "output_dimension".to_string(),
                    serde_json::json!(output_dimension),
                );
            }
            return Ok(serde_json::Value::Object(body));
        }

        let mut body = serde_json::Map::new();
        body.insert("inputText".to_string(), serde_json::json!(input));
        if let Some(dimensions) = Self::titan_dimensions(req, &options)? {
            body.insert("dimensions".to_string(), serde_json::json!(dimensions));
        }
        if let Some(normalize) = options.normalize {
            body.insert("normalize".to_string(), serde_json::json!(normalize));
        }

        Ok(serde_json::Value::Object(body))
    }
}

struct BedrockEmbeddingResponseTransformer {
    provider_id: String,
}

impl BedrockEmbeddingResponseTransformer {
    fn parse_vector(value: &serde_json::Value, field: &str) -> Result<Vec<f32>, LlmError> {
        let array = value.as_array().ok_or_else(|| {
            LlmError::ParseError(format!(
                "Amazon Bedrock embedding response field `{field}` must be an array"
            ))
        })?;

        array
            .iter()
            .map(|item| {
                item.as_f64().map(|value| value as f32).ok_or_else(|| {
                    LlmError::ParseError(format!(
                        "Amazon Bedrock embedding response field `{field}` must contain numbers"
                    ))
                })
            })
            .collect()
    }

    fn parse_usage(value: Option<&serde_json::Value>) -> Result<Option<EmbeddingUsage>, LlmError> {
        let Some(value) = value else {
            return Ok(None);
        };

        let tokens = value.as_u64().ok_or_else(|| {
            LlmError::ParseError(
                "Amazon Bedrock embedding token count must be an unsigned integer".to_string(),
            )
        })?;
        let tokens = u32::try_from(tokens).map_err(|_| {
            LlmError::ParseError(
                "Amazon Bedrock embedding token count exceeds u32 range".to_string(),
            )
        })?;
        Ok(Some(EmbeddingUsage::new(tokens, tokens)))
    }
}

impl ResponseTransformer for BedrockEmbeddingResponseTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_embedding_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<EmbeddingResponse, LlmError> {
        let (embedding, usage) = if let Some(vector) = raw.get("embedding") {
            (
                Self::parse_vector(vector, "embedding")?,
                Self::parse_usage(raw.get("inputTextTokenCount"))?,
            )
        } else if let Some(array) = raw.get("embeddings").and_then(|value| value.as_array()) {
            let first = array.first().ok_or_else(|| {
                LlmError::ParseError(
                    "Amazon Bedrock embedding response `embeddings` array cannot be empty"
                        .to_string(),
                )
            })?;

            if first
                .get("embeddingType")
                .or_else(|| first.get("embedding_type"))
                .is_some()
            {
                let vector = first.get("embedding").ok_or_else(|| {
                    LlmError::ParseError(
                        "Amazon Bedrock Nova embedding response is missing `embedding`".to_string(),
                    )
                })?;
                (
                    Self::parse_vector(vector, "embeddings[0].embedding")?,
                    Self::parse_usage(raw.get("inputTokenCount"))?,
                )
            } else {
                (Self::parse_vector(first, "embeddings[0]")?, None)
            }
        } else if let Some(array) = raw
            .get("embeddings")
            .and_then(|value| value.get("float"))
            .and_then(|value| value.as_array())
        {
            let first = array.first().ok_or_else(|| {
                LlmError::ParseError(
                    "Amazon Bedrock Cohere v4 embedding response `embeddings.float` cannot be empty"
                        .to_string(),
                )
            })?;
            (Self::parse_vector(first, "embeddings.float[0]")?, None)
        } else {
            return Err(LlmError::ParseError(
                "Unrecognized Amazon Bedrock embedding response shape".to_string(),
            ));
        };

        Ok(EmbeddingResponse {
            embeddings: vec![embedding],
            model: String::new(),
            usage,
            metadata: HashMap::new(),
            response: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source_section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
        let start_index = source.find(start).expect("section start marker");
        let end_index = source[start_index..]
            .find(end)
            .map(|offset| start_index + offset)
            .expect("section end marker");
        &source[start_index..end_index]
    }

    #[test]
    fn embedding_request_and_response_transformers_keep_provider_maps_directional() {
        let source = include_str!("embedding.rs");

        let request_transformer = source_section(
            source,
            "impl RequestTransformer for BedrockEmbeddingRequestTransformer",
            "struct BedrockEmbeddingResponseTransformer",
        );
        assert!(
            !request_transformer.contains("provider_metadata"),
            "Bedrock embedding request transformer must not read legacy provider_metadata"
        );
        assert!(
            !request_transformer.contains("providerMetadata"),
            "Bedrock embedding request transformer must not read legacy providerMetadata"
        );

        let response_transformer = source_section(
            source,
            "impl ResponseTransformer for BedrockEmbeddingResponseTransformer",
            "#[cfg(test)]",
        );
        assert!(
            !response_transformer.contains("provider_options"),
            "Bedrock embedding response transformer must not read request provider_options"
        );
        assert!(
            !response_transformer.contains("providerOptions"),
            "Bedrock embedding response transformer must not read request providerOptions"
        );
    }

    fn transformer() -> BedrockEmbeddingRequestTransformer {
        BedrockEmbeddingRequestTransformer {
            provider_id: "bedrock".to_string(),
        }
    }

    #[test]
    fn titan_embedding_transformer_uses_ai_sdk_shape() {
        let request = EmbeddingRequest::single("sunny day at the beach")
            .with_model("amazon.titan-embed-text-v2:0")
            .with_provider_option(
                "bedrock",
                serde_json::json!({
                    "dimensions": 512,
                    "normalize": true,
                }),
            );

        let body = transformer()
            .transform_embedding(&request)
            .expect("transform Titan request");

        assert_eq!(
            body,
            serde_json::json!({
                "inputText": "sunny day at the beach",
                "dimensions": 512,
                "normalize": true,
            })
        );
    }

    #[test]
    fn nova_embedding_transformer_uses_single_embedding_shape() {
        let request = EmbeddingRequest::single("sunny day at the beach")
            .with_model("amazon.nova-2-multimodal-embeddings-v1:0")
            .with_provider_option(
                "bedrock",
                serde_json::json!({
                    "embeddingDimension": 256,
                    "embeddingPurpose": "GENERIC_INDEX",
                }),
            );

        let body = transformer()
            .transform_embedding(&request)
            .expect("transform Nova request");

        assert_eq!(
            body,
            serde_json::json!({
                "taskType": "SINGLE_EMBEDDING",
                "singleEmbeddingParams": {
                    "embeddingPurpose": "GENERIC_INDEX",
                    "embeddingDimension": 256,
                    "text": {
                        "truncationMode": "END",
                        "value": "sunny day at the beach",
                    }
                }
            })
        );
    }

    #[test]
    fn cohere_embedding_transformer_uses_input_type_and_output_dimension() {
        let request = EmbeddingRequest::document("rainy day in the city")
            .with_model("cohere.embed-v4:0")
            .with_provider_option(
                "bedrock",
                serde_json::json!({
                    "truncate": "END",
                    "outputDimension": 256,
                }),
            );

        let body = transformer()
            .transform_embedding(&request)
            .expect("transform Cohere request");

        assert_eq!(
            body,
            serde_json::json!({
                "input_type": "search_document",
                "texts": ["rainy day in the city"],
                "truncate": "END",
                "output_dimension": 256,
            })
        );
    }

    #[test]
    fn embedding_response_transformer_accepts_all_ai_sdk_shapes() {
        let transformer = BedrockEmbeddingResponseTransformer {
            provider_id: "bedrock".to_string(),
        };

        let titan = transformer
            .transform_embedding_response(&serde_json::json!({
                "embedding": [0.1, 0.2, 0.3],
                "inputTextTokenCount": 8,
            }))
            .expect("transform Titan response");
        assert_eq!(titan.embeddings, vec![vec![0.1, 0.2, 0.3]]);
        assert_eq!(
            titan.usage.as_ref().map(|usage| usage.prompt_tokens),
            Some(8)
        );

        let nova = transformer
            .transform_embedding_response(&serde_json::json!({
                "embeddings": [
                    {
                        "embeddingType": "TEXT",
                        "embedding": [0.4, 0.5, 0.6],
                    }
                ],
                "inputTokenCount": 5,
            }))
            .expect("transform Nova response");
        assert_eq!(nova.embeddings, vec![vec![0.4, 0.5, 0.6]]);
        assert_eq!(
            nova.usage.as_ref().map(|usage| usage.prompt_tokens),
            Some(5)
        );

        let cohere_v3 = transformer
            .transform_embedding_response(&serde_json::json!({
                "embeddings": [[0.7, 0.8, 0.9]],
            }))
            .expect("transform Cohere v3 response");
        assert_eq!(cohere_v3.embeddings, vec![vec![0.7, 0.8, 0.9]]);
        assert!(cohere_v3.usage.is_none());

        let cohere_v4 = transformer
            .transform_embedding_response(&serde_json::json!({
                "embeddings": {
                    "float": [[1.0, 1.1, 1.2]],
                },
            }))
            .expect("transform Cohere v4 response");
        assert_eq!(cohere_v4.embeddings, vec![vec![1.0, 1.1, 1.2]]);
        assert!(cohere_v4.usage.is_none());
    }
}
