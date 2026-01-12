//! Response transformers for OpenAI-compatible protocol (Chat/Embedding/Image) and OpenAI Responses API

use crate::error::LlmError;
use crate::execution::transformers::response::ResponseTransformer;
use crate::types::{
    ChatResponse, EmbeddingResponse, EmbeddingUsage, GeneratedImage, ImageGenerationResponse,
    ModerationResponse, ModerationResult,
};

#[derive(Clone)]
pub struct OpenAiResponseTransformer;

impl ResponseTransformer for OpenAiResponseTransformer {
    fn provider_id(&self) -> &str {
        "openai"
    }

    fn transform_chat_response(&self, raw: &serde_json::Value) -> Result<ChatResponse, LlmError> {
        // Delegate to OpenAI-compatible response transformer for robust mapping
        let model = raw
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let adapter: std::sync::Arc<
            dyn crate::standards::openai::compat::adapter::ProviderAdapter,
        > = std::sync::Arc::new(
            crate::standards::openai::compat::adapter::OpenAiStandardAdapter {
                base_url: String::new(),
            },
        );
        let cfg = crate::standards::openai::compat::openai_config::OpenAiCompatibleConfig::new(
            "openai",
            "",
            "",
            adapter.clone(),
        )
        .with_model(&model);
        let compat = crate::standards::openai::compat::transformers::CompatResponseTransformer {
            config: cfg,
            adapter,
        };
        compat.transform_chat_response(raw)
    }

    fn transform_embedding_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<EmbeddingResponse, LlmError> {
        #[derive(serde::Deserialize)]
        struct OpenAiEmbeddingObject {
            embedding: Vec<f32>,
            index: usize,
        }
        #[derive(serde::Deserialize)]
        struct OpenAiEmbeddingUsage {
            prompt_tokens: u32,
            total_tokens: u32,
        }
        #[derive(serde::Deserialize)]
        struct OpenAiEmbeddingResponse {
            data: Vec<OpenAiEmbeddingObject>,
            model: String,
            usage: OpenAiEmbeddingUsage,
        }

        let mut r: OpenAiEmbeddingResponse = serde_json::from_value(raw.clone())
            .map_err(|e| LlmError::ParseError(format!("Invalid OpenAI embedding response: {e}")))?;
        r.data.sort_by_key(|o| o.index);
        let vectors = r.data.into_iter().map(|o| o.embedding).collect();
        let mut resp = EmbeddingResponse::new(vectors, r.model);
        resp.usage = Some(EmbeddingUsage::new(
            r.usage.prompt_tokens,
            r.usage.total_tokens,
        ));
        Ok(resp)
    }

    fn transform_image_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<ImageGenerationResponse, LlmError> {
        #[derive(serde::Deserialize)]
        struct OpenAiImageData {
            url: Option<String>,
            b64_json: Option<String>,
            revised_prompt: Option<String>,
        }
        #[derive(serde::Deserialize)]
        struct OpenAiImageResponse {
            created: u64,
            data: Vec<OpenAiImageData>,
        }

        let r: OpenAiImageResponse = serde_json::from_value(raw.clone())
            .map_err(|e| LlmError::ParseError(format!("Invalid OpenAI image response: {e}")))?;
        let images: Vec<GeneratedImage> = r
            .data
            .into_iter()
            .map(|img| GeneratedImage {
                url: img.url,
                b64_json: img.b64_json,
                format: None,
                width: None,
                height: None,
                revised_prompt: img.revised_prompt,
                metadata: std::collections::HashMap::new(),
            })
            .collect();
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("created".to_string(), serde_json::json!(r.created));
        Ok(ImageGenerationResponse {
            images,
            metadata,
            warnings: None,
            response: None,
        })
    }

    fn transform_moderation_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<ModerationResponse, LlmError> {
        let model = raw
            .get("model")
            .and_then(|v| v.as_str())
            .ok_or_else(|| LlmError::ParseError("Missing OpenAI moderation response model".into()))?
            .to_string();

        let results = raw
            .get("results")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                LlmError::ParseError("Missing OpenAI moderation response results".into())
            })?;

        let mut out: Vec<ModerationResult> = Vec::with_capacity(results.len());
        for item in results {
            let flagged = item
                .get("flagged")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            let categories_obj = item
                .get("categories")
                .and_then(|v| v.as_object())
                .ok_or_else(|| {
                    LlmError::ParseError("Missing OpenAI moderation response categories".into())
                })?;
            let mut categories: std::collections::HashMap<String, bool> =
                std::collections::HashMap::new();
            for (k, v) in categories_obj {
                categories.insert(k.clone(), v.as_bool().unwrap_or(false));
            }

            let scores_obj = item
                .get("category_scores")
                .and_then(|v| v.as_object())
                .ok_or_else(|| {
                    LlmError::ParseError(
                        "Missing OpenAI moderation response category_scores".into(),
                    )
                })?;
            let mut category_scores: std::collections::HashMap<String, f32> =
                std::collections::HashMap::new();
            for (k, v) in scores_obj {
                let score = v.as_f64().unwrap_or(0.0) as f32;
                category_scores.insert(k.clone(), score);
            }

            out.push(ModerationResult {
                flagged,
                categories,
                category_scores,
            });
        }

        Ok(ModerationResponse {
            results: out,
            model,
        })
    }
}

#[cfg(test)]
mod moderation_tests {
    use super::*;
    use crate::execution::transformers::response::ResponseTransformer;

    #[test]
    fn openai_moderation_response_maps_dynamic_category_keys() {
        let raw = serde_json::json!({
            "id": "modr_123",
            "model": "text-moderation-latest",
            "results": [
                {
                    "flagged": true,
                    "categories": {
                        "hate": false,
                        "hate/threatening": true
                    },
                    "category_scores": {
                        "hate": 0.01,
                        "hate/threatening": 0.99
                    }
                }
            ]
        });

        let tx = OpenAiResponseTransformer;
        let resp = tx
            .transform_moderation_response(&raw)
            .expect("moderation response");
        assert_eq!(resp.model, "text-moderation-latest");
        assert_eq!(resp.results.len(), 1);
        assert!(resp.results[0].flagged);
        assert!(!resp.results[0].categories["hate"]);
        assert!(resp.results[0].categories["hate/threatening"]);
        assert!(resp.results[0].category_scores["hate/threatening"] > 0.9);
    }
}

/// Extract thinking content from multiple possible field names with priority order
/// Priority order: reasoning_content > thinking > reasoning
pub fn extract_thinking_from_multiple_fields(value: &serde_json::Value) -> Option<String> {
    let field_names = ["reasoning_content", "thinking", "reasoning"];
    for field in field_names {
        if let Some(s) = value.get(field).and_then(|v| v.as_str()) {
            let trimmed = s.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

#[cfg(feature = "openai-responses")]
mod responses;

#[cfg(feature = "openai-responses")]
pub use responses::{OpenAiResponsesResponseTransformer, ResponsesTransformStyle};

#[cfg(feature = "openai-responses")]
pub(crate) use responses::extract_responses_output_text_logprobs;
