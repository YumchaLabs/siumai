//! Cohere native embedding standard aligned with AI SDK provider behavior.

use super::shared;
use crate::core::EmbeddingTransformers;
use crate::error::LlmError;
use crate::execution::transformers::request::RequestTransformer;
use crate::execution::transformers::response::ResponseTransformer;
use crate::provider_options::{
    CohereEmbeddingInputType, CohereEmbeddingOptions, CohereEmbeddingTruncate,
};
use crate::types::{EmbeddingRequest, EmbeddingResponse, EmbeddingTaskType, EmbeddingUsage};
use serde_json::{Value, json};
use std::sync::Arc;

#[derive(Clone, Default)]
pub struct CohereEmbeddingStandard;
const VALID_OUTPUT_DIMENSIONS: &[u32] = &[256, 512, 1024, 1536];

impl CohereEmbeddingStandard {
    pub fn new() -> Self {
        Self
    }

    pub fn create_transformers(
        &self,
        provider_id: &str,
        request: &EmbeddingRequest,
    ) -> EmbeddingTransformers {
        EmbeddingTransformers {
            request: Arc::new(CohereEmbeddingRequestTransformer {
                provider_id: provider_id.to_string(),
            }),
            response: Arc::new(CohereEmbeddingResponseTransformer {
                provider_id: provider_id.to_string(),
                default_model: request.model.clone().unwrap_or_default(),
            }),
        }
    }
}

#[derive(Clone)]
struct CohereEmbeddingRequestTransformer {
    provider_id: String,
}

impl RequestTransformer for CohereEmbeddingRequestTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_chat(&self, _req: &crate::types::ChatRequest) -> Result<Value, LlmError> {
        unreachable!("chat transformer not used for Cohere embedding standard")
    }

    fn transform_embedding(&self, req: &EmbeddingRequest) -> Result<Value, LlmError> {
        let options =
            shared::cohere_provider_options::<CohereEmbeddingOptions>(&req.provider_options_map)?;
        let model = req
            .model
            .as_deref()
            .map(str::trim)
            .filter(|model| !model.is_empty())
            .ok_or_else(|| {
                LlmError::ConfigurationError(
                    "Cohere embedding request requires a non-empty model id".to_string(),
                )
            })?;

        let input_type = options
            .as_ref()
            .and_then(|options| options.input_type)
            .unwrap_or_else(|| default_input_type(req.task_type.as_ref()));
        let truncate = options.as_ref().and_then(|options| options.truncate);
        let output_dimension = options
            .as_ref()
            .and_then(|options| options.output_dimension)
            .or(req.dimensions);

        let mut body = serde_json::Map::new();
        body.insert("model".to_string(), Value::String(model.to_string()));
        body.insert("embedding_types".to_string(), json!(["float"]));
        body.insert("texts".to_string(), json!(req.input));
        body.insert(
            "input_type".to_string(),
            Value::String(serialize_input_type(input_type).to_string()),
        );

        if let Some(truncate) = truncate {
            body.insert(
                "truncate".to_string(),
                Value::String(serialize_truncate(truncate).to_string()),
            );
        }
        if let Some(output_dimension) = output_dimension {
            validate_output_dimension(output_dimension)?;
            body.insert("output_dimension".to_string(), json!(output_dimension));
        }

        Ok(Value::Object(body))
    }
}

#[derive(Clone)]
struct CohereEmbeddingResponseTransformer {
    provider_id: String,
    default_model: String,
}

impl ResponseTransformer for CohereEmbeddingResponseTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_embedding_response(&self, raw: &Value) -> Result<EmbeddingResponse, LlmError> {
        let embeddings = raw
            .get("embeddings")
            .and_then(|embeddings| embeddings.get("float"))
            .and_then(Value::as_array)
            .ok_or_else(|| {
                LlmError::ParseError("Missing Cohere embeddings.float response field".to_string())
            })?
            .iter()
            .map(|embedding| {
                embedding
                    .as_array()
                    .ok_or_else(|| {
                        LlmError::ParseError(
                            "Cohere embedding item must be a float array".to_string(),
                        )
                    })?
                    .iter()
                    .map(|value| {
                        value.as_f64().map(|value| value as f32).ok_or_else(|| {
                            LlmError::ParseError(
                                "Cohere embedding vector value must be numeric".to_string(),
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, LlmError>>()
            })
            .collect::<Result<Vec<_>, LlmError>>()?;

        let input_tokens = raw
            .get("meta")
            .and_then(|meta| meta.get("billed_units"))
            .and_then(|units| units.get("input_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32;

        Ok(
            EmbeddingResponse::new(embeddings, self.default_model.clone())
                .with_usage(EmbeddingUsage::new(input_tokens, input_tokens)),
        )
    }
}

fn default_input_type(task_type: Option<&EmbeddingTaskType>) -> CohereEmbeddingInputType {
    match task_type {
        Some(EmbeddingTaskType::RetrievalDocument) => CohereEmbeddingInputType::SearchDocument,
        Some(EmbeddingTaskType::Classification) => CohereEmbeddingInputType::Classification,
        Some(EmbeddingTaskType::Clustering) => CohereEmbeddingInputType::Clustering,
        _ => CohereEmbeddingInputType::SearchQuery,
    }
}

fn serialize_input_type(input_type: CohereEmbeddingInputType) -> &'static str {
    match input_type {
        CohereEmbeddingInputType::SearchDocument => "search_document",
        CohereEmbeddingInputType::SearchQuery => "search_query",
        CohereEmbeddingInputType::Classification => "classification",
        CohereEmbeddingInputType::Clustering => "clustering",
    }
}

fn serialize_truncate(truncate: CohereEmbeddingTruncate) -> &'static str {
    match truncate {
        CohereEmbeddingTruncate::None => "NONE",
        CohereEmbeddingTruncate::Start => "START",
        CohereEmbeddingTruncate::End => "END",
    }
}

fn validate_output_dimension(output_dimension: u32) -> Result<(), LlmError> {
    if VALID_OUTPUT_DIMENSIONS.contains(&output_dimension) {
        Ok(())
    } else {
        Err(LlmError::ConfigurationError(format!(
            "Cohere output_dimension must be one of 256, 512, 1024, 1536, got {output_dimension}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider_options::CohereEmbeddingOptions;
    use crate::providers::cohere::CohereEmbeddingRequestExt;

    #[test]
    fn cohere_embedding_transform_accepts_supported_output_dimension() {
        let transformer = CohereEmbeddingRequestTransformer {
            provider_id: "cohere".to_string(),
        };
        let request = EmbeddingRequest::single("hello embedding")
            .with_model("embed-v4.0")
            .with_cohere_options(CohereEmbeddingOptions::new().with_output_dimension(1024));

        let body = transformer
            .transform_embedding(&request)
            .expect("transform request");

        assert_eq!(body["output_dimension"], json!(1024));
    }

    #[test]
    fn cohere_embedding_transform_rejects_invalid_output_dimension() {
        let transformer = CohereEmbeddingRequestTransformer {
            provider_id: "cohere".to_string(),
        };
        let request = EmbeddingRequest::single("hello embedding")
            .with_model("embed-v4.0")
            .with_cohere_options(CohereEmbeddingOptions::new().with_output_dimension(2048));

        let err = transformer
            .transform_embedding(&request)
            .expect_err("invalid output dimension should fail");

        assert!(
            matches!(err, LlmError::ConfigurationError(message) if message.contains("256, 512, 1024, 1536"))
        );
    }
}
