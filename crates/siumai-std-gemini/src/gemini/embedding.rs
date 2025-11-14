//! Gemini Embedding standard.
//!
//! 该模块实现了基于 `siumai-core` 的 Gemini Embedding 标准映射：
//! - 请求：`EmbeddingInput` → Gemini `embedContent` / `batchEmbedContents` JSON
//! - 响应：Gemini embedding JSON → `EmbeddingResult`

use siumai_core::error::LlmError;
use siumai_core::execution::embedding::{
    EmbeddingInput, EmbeddingRequestTransformer, EmbeddingResponseTransformer, EmbeddingResult,
    EmbeddingUsage,
};
use std::sync::Arc;

/// Core-level Gemini Embedding standard.
#[derive(Clone, Default)]
pub struct GeminiEmbeddingStandard;

impl GeminiEmbeddingStandard {
    /// Create a new standard.
    pub fn new() -> Self {
        Self
    }

    /// Create request transformer.
    pub fn create_request_transformer(
        &self,
        provider_id: &str,
    ) -> Arc<dyn EmbeddingRequestTransformer> {
        Arc::new(GeminiEmbeddingRequestTx {
            provider_id: provider_id.to_string(),
        })
    }

    /// Create response transformer.
    pub fn create_response_transformer(
        &self,
        provider_id: &str,
    ) -> Arc<dyn EmbeddingResponseTransformer> {
        Arc::new(GeminiEmbeddingResponseTx {
            provider_id: provider_id.to_string(),
        })
    }
}

/// 内部请求结构，与 Gemini embedContent/batchEmbedContents 对齐（简化版）。
#[derive(serde::Serialize)]
struct GeminiPart {
    text: String,
}

#[derive(serde::Serialize)]
struct GeminiContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    parts: Vec<GeminiPart>,
}

#[derive(serde::Serialize)]
struct GeminiEmbeddingRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    content: GeminiContent,
    #[serde(skip_serializing_if = "Option::is_none", rename = "taskType")]
    task_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "outputDimensionality"
    )]
    output_dimensionality: Option<u32>,
}

#[derive(serde::Serialize)]
struct GeminiBatchEmbeddingRequest {
    requests: Vec<GeminiEmbeddingRequest>,
}

#[derive(Clone)]
struct GeminiEmbeddingRequestTx {
    provider_id: String,
}

impl EmbeddingRequestTransformer for GeminiEmbeddingRequestTx {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_embedding(&self, input: &EmbeddingInput) -> Result<serde_json::Value, LlmError> {
        // 目前 EmbeddingInput 里没有标准化的 task_type 概念，
        // 这里统一使用 TASK_TYPE_UNSPECIFIED；具体优化由聚合层处理。
        let task_type: Option<String> = None;
        let title = input.title.clone();
        let output_dimensionality = input.dimensions;

        // Gemini 要求 model 形如 "models/{id}"；如果上层只传了裸 id，
        // 这里做一个保守的补前缀。
        let model_str = input.model.clone().unwrap_or_default();
        let model = if model_str.is_empty() {
            None
        } else if model_str.starts_with("models/") {
            Some(model_str)
        } else {
            Some(format!("models/{}", model_str))
        };

        if input.input.len() == 1 {
            let content = GeminiContent {
                role: None,
                parts: vec![GeminiPart {
                    text: input.input[0].clone(),
                }],
            };
            let body = GeminiEmbeddingRequest {
                model,
                content,
                task_type,
                title,
                output_dimensionality,
            };
            serde_json::to_value(body).map_err(|e| {
                LlmError::ParseError(format!("Serialize Gemini embedding request failed: {e}"))
            })
        } else {
            let requests: Vec<GeminiEmbeddingRequest> = input
                .input
                .iter()
                .map(|text| {
                    let content = GeminiContent {
                        role: Some("user".to_string()),
                        parts: vec![GeminiPart { text: text.clone() }],
                    };
                    GeminiEmbeddingRequest {
                        model: model.clone(),
                        content,
                        task_type: task_type.clone(),
                        title: title.clone(),
                        output_dimensionality,
                    }
                })
                .collect();
            let batch = GeminiBatchEmbeddingRequest { requests };
            serde_json::to_value(batch).map_err(|e| {
                LlmError::ParseError(format!(
                    "Serialize Gemini batch embedding request failed: {e}"
                ))
            })
        }
    }
}

#[derive(Clone)]
struct GeminiEmbeddingResponseTx {
    provider_id: String,
}

impl EmbeddingResponseTransformer for GeminiEmbeddingResponseTx {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_embedding_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<EmbeddingResult, LlmError> {
        // 支持两种常见返回形态：
        // - { "embedding": { "values": [...] } }
        // - { "embeddings": [ { "values": [...] }, ... ] }
        if let Some(obj) = raw.get("embedding") {
            let vals = obj
                .get("values")
                .and_then(|v| v.as_array())
                .ok_or_else(|| LlmError::ParseError("missing embedding.values".to_string()))?;
            let mut vec = Vec::with_capacity(vals.len());
            for v in vals {
                vec.push(v.as_f64().unwrap_or(0.0) as f32);
            }
            return Ok(EmbeddingResult {
                embeddings: vec![vec],
                model: String::new(),
                usage: None,
                metadata: Default::default(),
            });
        }

        if let Some(arr) = raw.get("embeddings").and_then(|v| v.as_array()) {
            let mut all = Vec::with_capacity(arr.len());
            for e in arr {
                let vals = e.get("values").and_then(|v| v.as_array()).ok_or_else(|| {
                    LlmError::ParseError("missing embeddings[i].values".to_string())
                })?;
                let mut vec = Vec::with_capacity(vals.len());
                for v in vals {
                    vec.push(v.as_f64().unwrap_or(0.0) as f32);
                }
                all.push(vec);
            }
            return Ok(EmbeddingResult {
                embeddings: all,
                model: String::new(),
                usage: None,
                metadata: Default::default(),
            });
        }

        Err(LlmError::ParseError(
            "Unrecognized Gemini embedding response shape".to_string(),
        ))
    }
}
