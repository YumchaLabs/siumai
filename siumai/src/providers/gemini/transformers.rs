//! Transformers for Google Gemini
//!
//! Centralizes request/response transformation for Gemini to reduce duplication
//! between non-streaming and streaming paths.

use crate::error::LlmError;
use crate::execution::transformers::files::{FilesHttpBody, FilesTransformer};
use crate::execution::transformers::request::{
    GenericRequestTransformer, MappingProfile, ProviderRequestHooks, RangeMode, Rule,
};
use crate::execution::transformers::{
    request::RequestTransformer, response::ResponseTransformer, stream::StreamChunkTransformer,
};
use crate::streaming::SseEventConverter;
use crate::types::EmbeddingRequest;
use crate::types::ImageGenerationRequest;
use crate::types::{ChatRequest, ChatResponse, ContentPart, FinishReason, MessageContent, Usage};
use eventsource_stream::Event;
use std::future::Future;
use std::pin::Pin;

use super::types::{CreateFileResponse, GeminiFile, GeminiFileState, ListFilesResponse};
use super::types::{GeminiConfig, GenerateContentRequest, GenerateContentResponse, Part};
// No longer depend on chat capability for request construction

/// Request transformer for Gemini
#[derive(Clone)]
pub struct GeminiRequestTransformer {
    pub config: GeminiConfig,
}

impl RequestTransformer for GeminiRequestTransformer {
    fn provider_id(&self) -> &str {
        "gemini"
    }

    fn transform_chat(&self, req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
        // Minimal validation: require model
        if req.common_params.model.is_empty() {
            return Err(LlmError::InvalidParameter(
                "Model must be specified".to_string(),
            ));
        }

        // Hooks + rules via Generic transformer
        struct GeminiChatHooks(super::types::GeminiConfig);
        impl ProviderRequestHooks for GeminiChatHooks {
            fn build_base_chat_body(
                &self,
                req: &ChatRequest,
            ) -> Result<serde_json::Value, LlmError> {
                // Start from typed builder (includes content/messages/tools)
                let typed: GenerateContentRequest = super::convert::build_request_body(
                    &self.0,
                    &req.messages,
                    req.tools.as_deref(),
                )?;
                let mut body = serde_json::to_value(typed)
                    .map_err(|e| LlmError::ParseError(format!("Serialize request failed: {e}")))?;

                // Put common params at top-level for rule-based moving into generationConfig
                if let Some(t) = req.common_params.temperature {
                    let v = (t as f64 * 1_000_000.0).round() / 1_000_000.0;
                    body["temperature"] = serde_json::json!(v);
                }
                if let Some(tp) = req.common_params.top_p {
                    let v = (tp as f64 * 1_000_000.0).round() / 1_000_000.0;
                    body["top_p"] = serde_json::json!(v);
                }
                if let Some(max) = req.common_params.max_tokens {
                    body["max_tokens"] = serde_json::json!(max);
                }
                if let Some(stops) = &req.common_params.stop_sequences {
                    body["stop_sequences"] = serde_json::json!(stops);
                }
                Ok(body)
            }

            fn post_process_chat(
                &self,
                req: &ChatRequest,
                body: &mut serde_json::Value,
            ) -> Result<(), LlmError> {
                // Add tool_choice if specified
                if req.tools.is_some()
                    && req.tool_choice.is_some()
                    && let Some(choice) = &req.tool_choice
                {
                    body["tool_config"] =
                        crate::providers::gemini::convert::convert_tool_choice(choice);
                }

                // All provider-specific features are now handled via provider_options
                // in ProviderSpec::chat_before_send()
                Ok(())
            }
        }

        let hooks = GeminiChatHooks(self.config.clone());
        let profile = MappingProfile {
            provider_id: "gemini",
            rules: vec![
                // Stable ranges only
                Rule::Range {
                    field: "temperature",
                    min: 0.0,
                    max: 2.0,
                    mode: RangeMode::Error,
                    message: Some("temperature must be between 0.0 and 2.0"),
                },
                Rule::Range {
                    field: "top_p",
                    min: 0.0,
                    max: 1.0,
                    mode: RangeMode::Error,
                    message: Some("top_p must be between 0.0 and 1.0"),
                },
                // Move top-level common params into generationConfig (camelCase)
                Rule::Move {
                    from: "temperature",
                    to: "generationConfig.temperature",
                },
                Rule::Move {
                    from: "top_p",
                    to: "generationConfig.topP",
                },
                Rule::Move {
                    from: "max_tokens",
                    to: "generationConfig.maxOutputTokens",
                },
                Rule::Move {
                    from: "stop_sequences",
                    to: "generationConfig.stopSequences",
                },
            ],
            // Provider options are injected via ProviderSpec::chat_before_send()
            merge_strategy:
                crate::execution::transformers::request::ProviderParamsMergeStrategy::Flatten,
        };
        let generic = GenericRequestTransformer { profile, hooks };
        generic.transform_chat(req)
    }

    fn transform_embedding(&self, req: &EmbeddingRequest) -> Result<serde_json::Value, LlmError> {
        // Use Generic transformer hooks to build typed JSON; no extra rules for now
        struct GeminiEmbeddingHooks(super::types::GeminiConfig);
        impl crate::execution::transformers::request::ProviderRequestHooks for GeminiEmbeddingHooks {
            fn build_base_embedding_body(
                &self,
                req: &EmbeddingRequest,
            ) -> Result<serde_json::Value, LlmError> {
                // Map to Gemini embedContent / batchEmbedContents request model
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

                let task_type = req.task_type.as_ref().map(|tt| match tt {
                    crate::types::EmbeddingTaskType::RetrievalQuery => {
                        "RETRIEVAL_QUERY".to_string()
                    }
                    crate::types::EmbeddingTaskType::RetrievalDocument => {
                        "RETRIEVAL_DOCUMENT".to_string()
                    }
                    crate::types::EmbeddingTaskType::SemanticSimilarity => {
                        "SEMANTIC_SIMILARITY".to_string()
                    }
                    crate::types::EmbeddingTaskType::Classification => "CLASSIFICATION".to_string(),
                    crate::types::EmbeddingTaskType::Clustering => "CLUSTERING".to_string(),
                    crate::types::EmbeddingTaskType::QuestionAnswering => {
                        "QUESTION_ANSWERING".to_string()
                    }
                    crate::types::EmbeddingTaskType::FactVerification => {
                        "FACT_VERIFICATION".to_string()
                    }
                    crate::types::EmbeddingTaskType::Unspecified => {
                        "TASK_TYPE_UNSPECIFIED".to_string()
                    }
                });
                let title = req.title.clone();
                let output_dimensionality = req.dimensions;

                if req.input.len() == 1 {
                    let content = GeminiContent {
                        role: None,
                        parts: vec![GeminiPart {
                            text: req.input[0].clone(),
                        }],
                    };
                    let body = GeminiEmbeddingRequest {
                        model: Some(format!("models/{}", self.0.model)),
                        content,
                        task_type,
                        title,
                        output_dimensionality,
                    };
                    serde_json::to_value(body)
                        .map_err(|e| LlmError::ParseError(format!("Serialize request failed: {e}")))
                } else {
                    let requests: Vec<GeminiEmbeddingRequest> = req
                        .input
                        .iter()
                        .map(|text| {
                            let content = GeminiContent {
                                role: Some("user".to_string()),
                                parts: vec![GeminiPart { text: text.clone() }],
                            };
                            GeminiEmbeddingRequest {
                                model: Some(format!("models/{}", self.0.model)),
                                content,
                                task_type: task_type.clone(),
                                title: title.clone(),
                                output_dimensionality,
                            }
                        })
                        .collect();
                    let batch = GeminiBatchEmbeddingRequest { requests };
                    serde_json::to_value(batch)
                        .map_err(|e| LlmError::ParseError(format!("Serialize request failed: {e}")))
                }
            }
        }
        let hooks = GeminiEmbeddingHooks(self.config.clone());
        let profile = crate::execution::transformers::request::MappingProfile {
            provider_id: "gemini",
            rules: vec![], // no generic rules; hook builds typed JSON
            merge_strategy:
                crate::execution::transformers::request::ProviderParamsMergeStrategy::Flatten,
        };
        let generic =
            crate::execution::transformers::request::GenericRequestTransformer { profile, hooks };
        generic.transform_embedding(req)
    }

    fn transform_image(&self, req: &ImageGenerationRequest) -> Result<serde_json::Value, LlmError> {
        // Use Generic hooks to build typed JSON for generateContent (IMAGE)
        struct GeminiImageHooks(super::types::GeminiConfig);
        impl crate::execution::transformers::request::ProviderRequestHooks for GeminiImageHooks {
            fn build_base_image_body(
                &self,
                req: &ImageGenerationRequest,
            ) -> Result<serde_json::Value, LlmError> {
                use super::types::{Content, GenerateContentRequest, Part};
                if self.0.model.is_empty() {
                    return Err(LlmError::InvalidParameter("Model must be specified".into()));
                }
                let prompt = req.prompt.clone();
                let contents = vec![Content {
                    role: Some("user".to_string()),
                    parts: vec![Part::Text {
                        text: prompt,
                        thought: None,
                    }],
                }];
                let mut gcfg = self.0.generation_config.clone().unwrap_or_default();
                if req.count > 0 {
                    gcfg.candidate_count = Some(req.count as i32);
                }
                // Ensure IMAGE modality present
                let mut modalities = gcfg.response_modalities.take().unwrap_or_default();
                if !modalities.iter().any(|m| m == "IMAGE") {
                    modalities.push("IMAGE".to_string());
                }
                if !modalities.is_empty() {
                    gcfg.response_modalities = Some(modalities);
                }
                let body = GenerateContentRequest {
                    model: self.0.model.clone(),
                    contents,
                    system_instruction: None,
                    tools: None,
                    tool_config: None,
                    safety_settings: self.0.safety_settings.clone(),
                    generation_config: Some(gcfg),
                    cached_content: None,
                };
                serde_json::to_value(body).map_err(|e| {
                    LlmError::ParseError(format!("Serialize image request failed: {e}"))
                })
            }
        }
        let hooks = GeminiImageHooks(self.config.clone());
        let profile = crate::execution::transformers::request::MappingProfile {
            provider_id: "gemini",
            rules: vec![],
            merge_strategy:
                crate::execution::transformers::request::ProviderParamsMergeStrategy::Flatten,
        };
        let generic =
            crate::execution::transformers::request::GenericRequestTransformer { profile, hooks };
        generic.transform_image(req)
    }
}

#[cfg(test)]
mod tests_gemini_rules {
    use super::*;

    #[test]
    fn move_common_params_into_generation_config() {
        let cfg = GeminiConfig::default()
            .with_model("gemini-1.5-flash".into())
            .with_base_url("https://example".into());
        let tx = GeminiRequestTransformer { config: cfg };
        let mut req = ChatRequest::new(vec![]);
        req.common_params.model = "gemini-1.5-flash".to_string();
        req.common_params.temperature = Some(0.4);
        req.common_params.top_p = Some(0.9);
        req.common_params.max_tokens = Some(1024);
        req.common_params.stop_sequences = Some(vec!["END".into()]);
        let body = tx.transform_chat(&req).expect("transform");
        let got_temp = body["generationConfig"]["temperature"].as_f64().unwrap();
        assert!((got_temp - 0.4).abs() < 1e-6);
        let got_top_p = body["generationConfig"]["topP"].as_f64().unwrap();
        assert!((got_top_p - 0.9).abs() < 1e-6);
        assert_eq!(
            body["generationConfig"]["maxOutputTokens"],
            serde_json::json!(1024)
        );
        assert_eq!(
            body["generationConfig"]["stopSequences"],
            serde_json::json!(["END"])
        );
    }
}

/// Response transformer for Gemini
#[derive(Clone)]
pub struct GeminiResponseTransformer {
    pub config: GeminiConfig,
}

impl ResponseTransformer for GeminiResponseTransformer {
    fn provider_id(&self) -> &str {
        "gemini"
    }

    fn transform_chat_response(&self, raw: &serde_json::Value) -> Result<ChatResponse, LlmError> {
        // Parse typed response and convert to unified ChatResponse (mirrors chat::convert_response)
        let response: GenerateContentResponse = serde_json::from_value(raw.clone())
            .map_err(|e| LlmError::ParseError(format!("Invalid Gemini response: {e}")))?;

        if response.candidates.is_empty() {
            return Err(LlmError::api_error(400, "No candidates in response"));
        }

        let candidate = &response.candidates[0];
        let content = candidate
            .content
            .as_ref()
            .ok_or_else(|| LlmError::api_error(400, "No content in candidate"))?;

        let mut text_content = String::new();
        let mut content_parts = Vec::new();
        // Track if any non-text parts are present (kept for future heuristics)
        let mut _has_multimodal_content = false;

        for part in &content.parts {
            match part {
                Part::Text { text, thought } => {
                    if thought.unwrap_or(false) {
                        // Add reasoning content
                        content_parts.push(ContentPart::reasoning(text));
                    } else {
                        if !text_content.is_empty() {
                            text_content.push('\n');
                        }
                        text_content.push_str(text);
                        content_parts.push(ContentPart::text(text));
                    }
                }
                Part::InlineData { inline_data } => {
                    _has_multimodal_content = true;
                    if inline_data.mime_type.starts_with("image/") {
                        content_parts.push(crate::types::ContentPart::Image {
                            source: crate::types::chat::MediaSource::Base64 {
                                data: inline_data.data.clone(),
                            },
                            detail: None,
                        });
                    } else if inline_data.mime_type.starts_with("audio/") {
                        content_parts.push(crate::types::ContentPart::Audio {
                            source: crate::types::chat::MediaSource::Base64 {
                                data: inline_data.data.clone(),
                            },
                            media_type: Some(inline_data.mime_type.clone()),
                        });
                    } else {
                        // Other file types
                        content_parts.push(crate::types::ContentPart::File {
                            source: crate::types::chat::MediaSource::Base64 {
                                data: inline_data.data.clone(),
                            },
                            media_type: inline_data.mime_type.clone(),
                            filename: None,
                        });
                    }
                }
                Part::FileData { file_data } => {
                    _has_multimodal_content = true;
                    let mime_type = file_data
                        .mime_type
                        .as_deref()
                        .unwrap_or("application/octet-stream");
                    if mime_type.starts_with("image/") {
                        content_parts.push(crate::types::ContentPart::Image {
                            source: crate::types::chat::MediaSource::Url {
                                url: file_data.file_uri.clone(),
                            },
                            detail: None,
                        });
                    } else if mime_type.starts_with("audio/") {
                        content_parts.push(crate::types::ContentPart::Audio {
                            source: crate::types::chat::MediaSource::Url {
                                url: file_data.file_uri.clone(),
                            },
                            media_type: Some(mime_type.to_string()),
                        });
                    } else {
                        // Other file types
                        content_parts.push(crate::types::ContentPart::File {
                            source: crate::types::chat::MediaSource::Url {
                                url: file_data.file_uri.clone(),
                            },
                            media_type: mime_type.to_string(),
                            filename: None,
                        });
                    }
                }
                Part::FunctionCall { function_call } => {
                    let arguments = function_call
                        .args
                        .clone()
                        .unwrap_or_else(|| serde_json::json!({}));
                    content_parts.push(ContentPart::tool_call(
                        format!("call_{}", uuid::Uuid::new_v4()),
                        function_call.name.clone(),
                        arguments,
                        None,
                    ));
                }
                _ => {}
            }
        }

        let usage = response.usage_metadata.as_ref().map(|m| Usage {
            prompt_tokens: m.prompt_token_count.unwrap_or(0) as u32,
            completion_tokens: m.candidates_token_count.unwrap_or(0) as u32,
            total_tokens: m.total_token_count.unwrap_or(0) as u32,
            #[allow(deprecated)]
            cached_tokens: m.cached_content_token_count.map(|t| t as u32),
            #[allow(deprecated)]
            reasoning_tokens: m.thoughts_token_count.map(|t| t as u32),
            prompt_tokens_details: m.cached_content_token_count.map(|cached| {
                crate::types::PromptTokensDetails {
                    audio_tokens: None,
                    cached_tokens: Some(cached as u32),
                }
            }),
            completion_tokens_details: m.thoughts_token_count.map(|reasoning| {
                crate::types::CompletionTokensDetails {
                    reasoning_tokens: Some(reasoning as u32),
                    audio_tokens: None,
                    accepted_prediction_tokens: None,
                    rejected_prediction_tokens: None,
                }
            }),
        });

        let finish_reason = candidate.finish_reason.as_ref().map(|reason| match reason {
            super::types::FinishReason::Stop => FinishReason::Stop,
            super::types::FinishReason::MaxTokens => FinishReason::Length,
            super::types::FinishReason::Safety => FinishReason::ContentFilter,
            _ => FinishReason::Other("unknown".to_string()),
        });

        // If we collected any content parts (text, tool calls, media), prefer MultiModal
        // unless all parts are plain text, in which case return a single Text.
        let content = if !content_parts.is_empty() {
            if content_parts.iter().all(|p| p.is_text()) {
                MessageContent::Text(text_content)
            } else {
                MessageContent::MultiModal(content_parts)
            }
        } else {
            MessageContent::Text(text_content)
        };

        Ok(ChatResponse {
            id: response.response_id.clone(),
            content,
            model: response.model_version.clone(),
            usage,
            finish_reason,
            audio: None, // Gemini doesn't support audio output in this format
            system_fingerprint: None,
            service_tier: None,
            warnings: None,
            provider_metadata: None,
        })
    }

    fn transform_embedding_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<crate::types::EmbeddingResponse, LlmError> {
        // Handle both single and batch embeddings
        // Single: { "embedding": { "values": [f32,..] } }
        // Batch:  { "embeddings": [ { "values": [...] }, ... ] }
        let model = self.config.model.clone();
        if let Some(obj) = raw.get("embedding") {
            let vals = obj
                .get("values")
                .and_then(|v| v.as_array())
                .ok_or_else(|| LlmError::ParseError("missing embedding.values".to_string()))?;
            let mut vec = Vec::with_capacity(vals.len());
            for v in vals {
                vec.push(v.as_f64().unwrap_or(0.0) as f32);
            }
            return Ok(crate::types::EmbeddingResponse::new(vec![vec], model));
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
            return Ok(crate::types::EmbeddingResponse::new(all, model));
        }
        Err(LlmError::ParseError(
            "Unrecognized Gemini embedding response shape".to_string(),
        ))
    }

    fn transform_image_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<crate::types::ImageGenerationResponse, LlmError> {
        let response: GenerateContentResponse = serde_json::from_value(raw.clone())
            .map_err(|e| LlmError::ParseError(format!("Invalid Gemini image response: {e}")))?;
        let mut images = Vec::new();
        if let Some(candidate) = response.candidates.first()
            && let Some(content) = &candidate.content
        {
            for part in &content.parts {
                match part {
                    Part::InlineData { inline_data } => {
                        if inline_data.mime_type.starts_with("image/") {
                            images.push(crate::types::GeneratedImage {
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
                    Part::FileData { file_data } => {
                        if let Some(m) = &file_data.mime_type
                            && !m.starts_with("image/")
                        {
                            continue;
                        }
                        images.push(crate::types::GeneratedImage {
                            url: Some(file_data.file_uri.clone()),
                            b64_json: None,
                            format: file_data.mime_type.clone(),
                            width: None,
                            height: None,
                            revised_prompt: None,
                            metadata: std::collections::HashMap::new(),
                        });
                    }
                    _ => {}
                }
            }
        }
        Ok(crate::types::ImageGenerationResponse {
            images,
            metadata: std::collections::HashMap::new(),
        })
    }
}

/// Files transformer for Gemini
#[derive(Clone)]
pub struct GeminiFilesTransformer {
    pub config: GeminiConfig,
}

impl GeminiFilesTransformer {
    fn convert_file(&self, gemini_file: &GeminiFile) -> crate::types::FileObject {
        let id = gemini_file
            .name
            .as_ref()
            .and_then(|n| n.strip_prefix("files/"))
            .unwrap_or("")
            .to_string();
        let bytes = gemini_file
            .size_bytes
            .as_ref()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        let created_at = gemini_file
            .create_time
            .as_ref()
            .and_then(|t| chrono::DateTime::parse_from_rfc3339(t).ok())
            .map(|dt| dt.timestamp() as u64)
            .unwrap_or(0);
        let status = match gemini_file.state {
            Some(GeminiFileState::Active) => "active".to_string(),
            Some(GeminiFileState::Processing) => "processing".to_string(),
            Some(GeminiFileState::Failed) => "failed".to_string(),
            _ => "unknown".to_string(),
        };
        let filename = gemini_file
            .display_name
            .clone()
            .unwrap_or_else(|| format!("file_{id}"));
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("provider".to_string(), serde_json::json!("gemini"));
        if let Some(uri) = &gemini_file.uri {
            metadata.insert("uri".to_string(), serde_json::json!(uri));
        }
        if let Some(hash) = &gemini_file.sha256_hash {
            metadata.insert("sha256_hash".to_string(), serde_json::json!(hash));
        }
        if let Some(exp) = &gemini_file.expiration_time {
            metadata.insert("expiration_time".to_string(), serde_json::json!(exp));
        }
        crate::types::FileObject {
            id,
            filename,
            bytes,
            created_at,
            purpose: "general".to_string(),
            status,
            mime_type: gemini_file.mime_type.clone(),
            metadata,
        }
    }
}

impl FilesTransformer for GeminiFilesTransformer {
    fn provider_id(&self) -> &str {
        "gemini"
    }

    fn build_upload_body(
        &self,
        req: &crate::types::FileUploadRequest,
    ) -> Result<FilesHttpBody, LlmError> {
        let detected = req
            .mime_type
            .clone()
            .unwrap_or_else(|| crate::utils::guess_mime(Some(&req.content), Some(&req.filename)));
        let part = reqwest::multipart::Part::bytes(req.content.clone())
            .file_name(req.filename.clone())
            .mime_str(&detected)
            .map_err(|e| LlmError::HttpError(format!("Invalid MIME type: {e}")))?;
        let mut form = reqwest::multipart::Form::new().part("file", part);
        if let Some(name) = req.metadata.get("display_name") {
            form = form.text("display_name", name.clone());
        }
        Ok(FilesHttpBody::Multipart(form))
    }

    fn list_endpoint(&self, query: &Option<crate::types::FileListQuery>) -> String {
        let mut endpoint = "files".to_string();
        let mut params = Vec::new();
        if let Some(q) = query {
            if let Some(limit) = q.limit {
                params.push(format!("pageSize={limit}"));
            }
            if let Some(after) = &q.after {
                params.push(format!("pageToken={after}"));
            }
        }
        if !params.is_empty() {
            endpoint.push('?');
            endpoint.push_str(&params.join("&"));
        }
        endpoint
    }

    fn retrieve_endpoint(&self, file_id: &str) -> String {
        if file_id.starts_with("files/") {
            file_id.to_string()
        } else {
            format!("files/{file_id}")
        }
    }

    fn delete_endpoint(&self, file_id: &str) -> String {
        if file_id.starts_with("files/") {
            file_id.to_string()
        } else {
            format!("files/{file_id}")
        }
    }

    fn transform_file_object(
        &self,
        raw: &serde_json::Value,
    ) -> Result<crate::types::FileObject, LlmError> {
        // Gemini upload returns CreateFileResponse; retrieve returns GeminiFile directly
        if raw.get("file").is_some() {
            let resp: CreateFileResponse = serde_json::from_value(raw.clone()).map_err(|e| {
                LlmError::ParseError(format!("Failed to parse upload response: {e}"))
            })?;
            let file = resp
                .file
                .ok_or_else(|| LlmError::ParseError("No file in upload response".to_string()))?;
            Ok(self.convert_file(&file))
        } else {
            let file: GeminiFile = serde_json::from_value(raw.clone())
                .map_err(|e| LlmError::ParseError(format!("Failed to parse file response: {e}")))?;
            Ok(self.convert_file(&file))
        }
    }

    fn transform_list_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<crate::types::FileListResponse, LlmError> {
        let resp: ListFilesResponse = serde_json::from_value(raw.clone())
            .map_err(|e| LlmError::ParseError(format!("Failed to parse list response: {e}")))?;
        let files: Vec<crate::types::FileObject> = resp
            .files
            .into_iter()
            .map(|f| self.convert_file(&f))
            .collect();
        Ok(crate::types::FileListResponse {
            files,
            has_more: resp.next_page_token.is_some(),
            next_cursor: resp.next_page_token,
        })
    }

    fn content_url_from_file_object(&self, file: &crate::types::FileObject) -> Option<String> {
        file.metadata
            .get("uri")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }
}

#[cfg(test)]
mod files_tests {
    use super::*;
    use crate::execution::transformers::files::{FilesHttpBody, FilesTransformer};

    fn sample_config() -> GeminiConfig {
        use secrecy::SecretString;
        GeminiConfig {
            api_key: SecretString::from("x".to_string()),
            base_url: "https://example.com".into(),
            model: "gemini-1.5-flash".into(),
            common_params: crate::types::CommonParams::default(),
            generation_config: None,
            safety_settings: None,
            timeout: Some(30),
            http_config: crate::types::HttpConfig::default(),
            token_provider: None,
        }
    }

    #[test]
    fn test_gemini_files_endpoints() {
        let tx = GeminiFilesTransformer {
            config: sample_config(),
        };
        let q = crate::types::FileListQuery {
            limit: Some(20),
            after: Some("page123".into()),
            ..Default::default()
        };
        let ep = tx.list_endpoint(&Some(q));
        assert!(ep.starts_with("files?"));
        assert!(ep.contains("pageSize=20"));
        assert!(ep.contains("pageToken=page123"));
        assert_eq!(tx.retrieve_endpoint("abc"), "files/abc");
        assert_eq!(tx.delete_endpoint("files/def"), "files/def");
        assert!(tx.content_endpoint("ignored").is_none());
    }

    #[test]
    fn test_gemini_files_upload_and_parse() {
        let tx = GeminiFilesTransformer {
            config: sample_config(),
        };
        let req = crate::types::FileUploadRequest {
            content: b"hi".to_vec(),
            filename: "hi.txt".into(),
            mime_type: Some("text/plain".into()),
            purpose: "general".into(),
            metadata: std::collections::HashMap::new(),
            http_config: None,
        };
        match tx.build_upload_body(&req).unwrap() {
            FilesHttpBody::Multipart(_) => {}
            _ => panic!("expected multipart form for Gemini upload"),
        }

        // upload response shape (CreateFileResponse)
        let upload_json = serde_json::json!({
            "file": {
                "name": "files/abc",
                "display_name": "hi.txt",
                "mime_type": "text/plain",
                "size_bytes": "2",
                "create_time": "2024-01-01T00:00:00Z",
                "state": "ACTIVE",
                "uri": "https://content.example/abc"
            }
        });
        let fo = tx.transform_file_object(&upload_json).unwrap();
        assert_eq!(fo.id, "abc");
        assert_eq!(fo.filename, "hi.txt");
        assert_eq!(
            tx.content_url_from_file_object(&fo).as_deref(),
            Some("https://content.example/abc")
        );

        // list response
        let list_json = serde_json::json!({
            "files": [ upload_json["file"].clone() ],
            "next_page_token": null
        });
        let lr = tx.transform_list_response(&list_json).unwrap();
        assert_eq!(lr.files.len(), 1);
        assert!(!lr.has_more);
    }
}

#[cfg(test)]
mod images_tests {
    use super::*;
    use crate::execution::transformers::request::RequestTransformer;
    use crate::execution::transformers::response::ResponseTransformer;

    fn cfg() -> GeminiConfig {
        use secrecy::SecretString;
        GeminiConfig {
            api_key: SecretString::from("x".to_string()),
            base_url: "https://example.com".into(),
            model: "gemini-1.5-flash".into(),
            common_params: crate::types::CommonParams::default(),
            generation_config: None,
            safety_settings: None,
            timeout: Some(30),
            http_config: crate::types::HttpConfig::default(),
            token_provider: None,
        }
    }

    #[test]
    fn test_transform_image_builds_generate_content_body() {
        let tx = GeminiRequestTransformer { config: cfg() };
        let req = crate::types::ImageGenerationRequest {
            prompt: "a cat".into(),
            count: 1,
            ..Default::default()
        };
        let body = tx.transform_image(&req).unwrap();
        // Basic presence checks
        assert_eq!(body["model"], "gemini-1.5-flash");
        assert!(body.get("contents").is_some());
        // Ensure modalities include IMAGE via generationConfig if present after transform
        // Note: we don't require presence if not set; behavior depends on config merging
    }

    #[test]
    fn test_transform_image_response_extracts_images() {
        let tx = GeminiResponseTransformer { config: cfg() };
        let json = serde_json::json!({
            "candidates": [
                {
                    "content": {
                        "parts": [
                            { "inlineData": { "mime_type": "image/png", "data": "iVBORw0..." } },
                            { "fileData": { "file_uri": "https://storage.example/image.png", "mime_type": "image/png" } }
                        ]
                    }
                }
            ]
        });
        let out = tx.transform_image_response(&json).unwrap();
        assert_eq!(out.images.len(), 2);
        assert!(out.images.iter().any(|i| i.b64_json.is_some()));
        assert!(out.images.iter().any(|i| i.url.is_some()));
    }
}

#[cfg(test)]
mod embeddings_tests {
    use super::*;
    use crate::execution::transformers::request::RequestTransformer;
    use crate::execution::transformers::response::ResponseTransformer;

    fn cfg() -> GeminiConfig {
        use secrecy::SecretString;
        GeminiConfig {
            api_key: SecretString::from("x".to_string()),
            base_url: "https://example.com".into(),
            model: "gemini-embedding-001".into(),
            common_params: crate::types::CommonParams::default(),
            generation_config: None,
            safety_settings: None,
            timeout: Some(30),
            http_config: crate::types::HttpConfig::default(),
            token_provider: None,
        }
    }

    #[test]
    fn test_transform_embedding_response_single() {
        let tx = GeminiResponseTransformer { config: cfg() };
        let json = serde_json::json!({
            "embedding": { "values": [0.1, 0.2, 0.3] }
        });
        let out = tx.transform_embedding_response(&json).unwrap();
        assert_eq!(out.embeddings.len(), 1);
        assert_eq!(out.embeddings[0].len(), 3);
        assert_eq!(out.model, "gemini-embedding-001");
    }

    #[test]
    fn test_transform_embedding_response_batch() {
        let tx = GeminiResponseTransformer { config: cfg() };
        let json = serde_json::json!({
            "embeddings": [
                { "values": [0.1, 0.2] },
                { "values": [0.3, 0.4] }
            ]
        });
        let out = tx.transform_embedding_response(&json).unwrap();
        assert_eq!(out.embeddings.len(), 2);
        assert_eq!(out.embeddings[0], vec![0.1_f32, 0.2_f32]);
        assert_eq!(out.embeddings[1], vec![0.3_f32, 0.4_f32]);
    }

    #[test]
    fn test_transform_embedding_request_single_flattened() {
        let tx = GeminiRequestTransformer { config: cfg() };
        let req = crate::types::EmbeddingRequest::new(vec!["Hello".to_string()])
            .with_dimensions(768)
            .with_task_type(crate::types::EmbeddingTaskType::RetrievalQuery)
            .with_title("My Title");

        let body = tx.transform_embedding(&req).expect("serialize request");

        // Ensure there is no nested embeddingConfig and fields are flattened
        assert!(body.get("embeddingConfig").is_none());
        assert_eq!(body["model"], "models/gemini-embedding-001");
        assert_eq!(body["taskType"], "RETRIEVAL_QUERY");
        assert_eq!(body["title"], "My Title");
        assert_eq!(body["outputDimensionality"], 768);
        assert_eq!(body["content"]["parts"][0]["text"], "Hello");
        // role is optional for single
        assert!(body["content"].get("role").is_none());
    }

    #[test]
    fn test_transform_embedding_request_batch_flattened_with_role() {
        let tx = GeminiRequestTransformer { config: cfg() };
        let req = crate::types::EmbeddingRequest::new(vec!["A".to_string(), "B".to_string()])
            .with_dimensions(64)
            .with_task_type(crate::types::EmbeddingTaskType::SemanticSimilarity);

        let body = tx.transform_embedding(&req).expect("serialize request");

        let requests = body["requests"].as_array().expect("requests array");
        assert_eq!(requests.len(), 2);
        for (i, value) in ["A", "B"].iter().enumerate() {
            let item = &requests[i];
            assert_eq!(item["model"], "models/gemini-embedding-001");
            assert_eq!(item["taskType"], "SEMANTIC_SIMILARITY");
            assert_eq!(item["outputDimensionality"], 64);
            assert_eq!(item["content"]["role"], "user");
            assert_eq!(item["content"]["parts"][0]["text"], *value);
            assert!(item.get("embeddingConfig").is_none());
        }
    }
}

/// Stream chunk transformer wrapping the existing GeminiEventConverter
#[derive(Clone)]
pub struct GeminiStreamChunkTransformer {
    pub provider_id: String,
    pub inner: super::streaming::GeminiEventConverter,
}

impl StreamChunkTransformer for GeminiStreamChunkTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn convert_event(
        &self,
        event: Event,
    ) -> Pin<
        Box<
            dyn Future<Output = Vec<Result<crate::streaming::ChatStreamEvent, LlmError>>>
                + Send
                + Sync
                + '_,
        >,
    > {
        self.inner.convert_event(event)
    }

    fn handle_stream_end(&self) -> Option<Result<crate::streaming::ChatStreamEvent, LlmError>> {
        self.inner.handle_stream_end()
    }
}
