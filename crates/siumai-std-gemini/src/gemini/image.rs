//! Gemini Image standard.
//!
//! 基于 `siumai-core` 的 Gemini 图片生成标准：
//! - 请求：`ImageGenerationRequest` → Gemini `generateContent`(IMAGE) JSON
//! - 响应：Gemini `GenerateContentResponse` → `ImageGenerationResponse`

use siumai_core::error::LlmError;
use siumai_core::execution::image::{
    ImageHttpBody, ImageRequestTransformer, ImageResponseTransformer,
};
use siumai_core::types::image::{
    GeneratedImage, ImageEditRequest, ImageGenerationRequest, ImageGenerationResponse,
    ImageVariationRequest,
};
use std::sync::Arc;

/// Core-level Gemini Image standard.
#[derive(Clone, Default)]
pub struct GeminiImageStandard;

impl GeminiImageStandard {
    /// Create a new standard.
    pub fn new() -> Self {
        Self
    }

    /// Create request transformer.
    pub fn create_request_transformer(
        &self,
        provider_id: &str,
    ) -> Arc<dyn ImageRequestTransformer> {
        Arc::new(GeminiImageRequestTx {
            provider_id: provider_id.to_string(),
        })
    }

    /// Create response transformer.
    pub fn create_response_transformer(
        &self,
        provider_id: &str,
    ) -> Arc<dyn ImageResponseTransformer> {
        Arc::new(GeminiImageResponseTx {
            provider_id: provider_id.to_string(),
        })
    }
}

/// 下面的类型是 `GenerateContentRequest/Response` 的精简版，只保留图片相关字段。

#[derive(serde::Serialize)]
struct Content {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    parts: Vec<Part>,
}

#[derive(serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Part {
    #[serde(rename = "text")]
    Text { text: String },
}

#[derive(serde::Serialize)]
struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none", rename = "candidateCount")]
    candidate_count: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "responseModalities")]
    response_modalities: Option<Vec<String>>,
}

#[derive(serde::Serialize)]
struct GenerateContentRequest {
    model: String,
    contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "generationConfig")]
    generation_config: Option<GenerationConfig>,
}

#[derive(serde::Deserialize)]
struct InlineData {
    #[serde(rename = "mimeType")]
    mime_type: String,
    data: String,
}

#[derive(serde::Deserialize)]
struct FileData {
    #[serde(rename = "fileUri")]
    file_uri: String,
    #[serde(rename = "mimeType")]
    mime_type: Option<String>,
}

#[derive(serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RespPart {
    #[serde(rename = "inline_data")]
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: InlineData,
    },
    #[serde(rename = "file_data")]
    FileData {
        #[serde(rename = "fileData")]
        file_data: FileData,
    },
    #[serde(other)]
    Other,
}

#[derive(serde::Deserialize)]
struct RespContent {
    parts: Vec<RespPart>,
}

#[derive(serde::Deserialize)]
struct Candidate {
    content: Option<RespContent>,
}

#[derive(serde::Deserialize)]
struct GenerateContentResponse {
    #[serde(default)]
    candidates: Vec<Candidate>,
}

#[derive(Clone)]
struct GeminiImageRequestTx {
    provider_id: String,
}

impl ImageRequestTransformer for GeminiImageRequestTx {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_image(&self, req: &ImageGenerationRequest) -> Result<serde_json::Value, LlmError> {
        if req.model.as_deref().unwrap_or("").is_empty() {
            return Err(LlmError::InvalidParameter(
                "Model must be specified for Gemini image generation".into(),
            ));
        }

        // 将 prompt 映射到 contents/parts，与文本 chat 一致，但强制 IMAGE 模态。
        let contents = vec![Content {
            role: Some("user".to_string()),
            parts: vec![Part::Text {
                text: req.prompt.clone(),
            }],
        }];

        let mut modalities = vec!["IMAGE".to_string()];
        // 预留：如果未来需要 TEXT+IMAGE，可以在这里扩展。
        if modalities.is_empty() {
            modalities.push("IMAGE".to_string());
        }

        let gen_cfg = GenerationConfig {
            candidate_count: if req.count > 0 {
                Some(req.count as i32)
            } else {
                None
            },
            response_modalities: Some(modalities),
        };

        let model = req.model.clone().unwrap_or_default();
        let model = if model.starts_with("models/") {
            model
        } else {
            format!("models/{}", model)
        };

        let body = GenerateContentRequest {
            model,
            contents,
            generation_config: Some(gen_cfg),
        };

        serde_json::to_value(body).map_err(|e| {
            LlmError::ParseError(format!("Serialize Gemini image request failed: {e}"))
        })
    }

    fn transform_image_edit(&self, _req: &ImageEditRequest) -> Result<ImageHttpBody, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Gemini image edit not implemented in std-gemini".to_string(),
        ))
    }

    fn transform_image_variation(
        &self,
        _req: &ImageVariationRequest,
    ) -> Result<ImageHttpBody, LlmError> {
        Err(LlmError::UnsupportedOperation(
            "Gemini image variation not implemented in std-gemini".to_string(),
        ))
    }
}

#[derive(Clone)]
struct GeminiImageResponseTx {
    provider_id: String,
}

impl ImageResponseTransformer for GeminiImageResponseTx {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_image_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<ImageGenerationResponse, LlmError> {
        let response: GenerateContentResponse = serde_json::from_value(raw.clone())
            .map_err(|e| LlmError::ParseError(format!("Invalid Gemini image response: {e}")))?;

        let mut images = Vec::new();
        if let Some(candidate) = response.candidates.first()
            && let Some(content) = &candidate.content
        {
            for part in &content.parts {
                match part {
                    RespPart::InlineData { inline_data } => {
                        if inline_data.mime_type.starts_with("image/") {
                            images.push(GeneratedImage {
                                url: None,
                                b64_json: Some(inline_data.data.clone()),
                                format: Some(inline_data.mime_type.clone()),
                                width: None,
                                height: None,
                                revised_prompt: None,
                                metadata: std::collections::HashMap::new(),
                            });
                        }
                    }
                    RespPart::FileData { file_data } => {
                        let mime = file_data
                            .mime_type
                            .as_deref()
                            .unwrap_or("application/octet-stream");
                        if !mime.starts_with("image/") {
                            continue;
                        }
                        images.push(GeneratedImage {
                            url: Some(file_data.file_uri.clone()),
                            b64_json: None,
                            format: Some(mime.to_string()),
                            width: None,
                            height: None,
                            revised_prompt: None,
                            metadata: std::collections::HashMap::new(),
                        });
                    }
                    RespPart::Other => {}
                }
            }
        }

        Ok(ImageGenerationResponse {
            images,
            metadata: std::collections::HashMap::new(),
        })
    }
}
