//! Transformers for Google Gemini
//!
//! Centralizes request/response transformation for Gemini to reduce duplication
//! between non-streaming and streaming paths.

use crate::error::LlmError;
use crate::transformers::files::{FilesHttpBody, FilesTransformer};
use crate::transformers::request::{
    GenericRequestTransformer, MappingProfile, ProviderRequestHooks, RangeMode, Rule,
};
use crate::transformers::{
    request::RequestTransformer, response::ResponseTransformer, stream::StreamChunkTransformer,
};
use crate::types::EmbeddingRequest;
use crate::types::ImageGenerationRequest;
use crate::types::{
    ChatRequest, ChatResponse, FinishReason, FunctionCall, MessageContent, ToolCall, Usage,
};
use crate::utils::streaming::SseEventConverter;
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
                    body["temperature"] = serde_json::json!(t);
                }
                if let Some(tp) = req.common_params.top_p {
                    body["top_p"] = serde_json::json!(tp);
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
                // Structured output hints via provider_params. Map into generationConfig.
                if let Some(pp) = &req.provider_params {
                    if let Some(so) = pp
                        .params
                        .get("structured_output")
                        .and_then(|v| v.as_object())
                    {
                        let mut mime = "application/json".to_string();
                        // Enum output special-case
                        if let Some(out) = so.get("output").and_then(|v| v.as_str()) {
                            if out.eq_ignore_ascii_case("enum") {
                                mime = "text/x.enum".to_string();
                            }
                        }
                        let schema_opt = so.get("schema").cloned().or_else(|| {
                            // If enum with values provided, synthesize schema
                            so.get("enum").and_then(|arr| arr.as_array()).map(|vals| {
                                let strings: Vec<String> = vals
                                    .iter()
                                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                    .collect();
                                serde_json::json!({"type": "STRING", "enum": strings})
                            })
                        });
                        // Ensure generationConfig object exists
                        if !body.get("generationConfig").is_some() {
                            body["generationConfig"] = serde_json::json!({});
                        }
                        body["generationConfig"]["responseMimeType"] = serde_json::json!(mime);
                        if let Some(schema) = schema_opt {
                            body["generationConfig"]["responseSchema"] = schema;
                        }
                    }
                }
                // Remove any merged hint at top-level to avoid unknown field
                if let Some(obj) = body.as_object_mut() {
                    obj.remove("structured_output");
                }
                // Map provider_params.function_calling -> toolConfig.functionCallingConfig
                if let Some(pp) = &req.provider_params
                    && let Some(fc) = pp
                        .params
                        .get("function_calling")
                        .and_then(|v| v.as_object())
                {
                    let mut cfg = super::types::FunctionCallingConfig {
                        mode: None,
                        allowed_function_names: None,
                    };
                    if let Some(mode_str) = fc.get("mode").and_then(|v| v.as_str()) {
                        let mode = match mode_str.to_ascii_uppercase().as_str() {
                            "AUTO" => super::types::FunctionCallingMode::Auto,
                            "ANY" => super::types::FunctionCallingMode::Any,
                            "NONE" => super::types::FunctionCallingMode::None,
                            _ => super::types::FunctionCallingMode::Unspecified,
                        };
                        cfg.mode = Some(mode);
                    }
                    if let Some(arr) = fc
                        .get("allowed")
                        .and_then(|v| v.as_array())
                        .or_else(|| fc.get("allowed_function_names").and_then(|v| v.as_array()))
                    {
                        let names: Vec<String> = arr
                            .iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect();
                        if !names.is_empty() {
                            cfg.allowed_function_names = Some(names);
                        }
                    }
                    // Write toolConfig.functionCallingConfig
                    let tool_cfg = super::types::ToolConfig {
                        function_calling_config: Some(cfg),
                    };
                    body["toolConfig"] = serde_json::to_value(tool_cfg).map_err(|e| {
                        LlmError::ParseError(format!("Serialize tool config failed: {e}"))
                    })?;
                }
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
            // provider_params merged via hooks when needed (function_calling)
            merge_strategy: crate::transformers::request::ProviderParamsMergeStrategy::Flatten,
        };
        let generic = GenericRequestTransformer { profile, hooks };
        generic.transform_chat(req)
    }

    fn transform_embedding(&self, req: &EmbeddingRequest) -> Result<serde_json::Value, LlmError> {
        // Use Generic transformer hooks to build typed JSON; no extra rules for now
        struct GeminiEmbeddingHooks(super::types::GeminiConfig);
        impl crate::transformers::request::ProviderRequestHooks for GeminiEmbeddingHooks {
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

                let task_type = req
                    .provider_params
                    .get("task_type")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let title = req
                    .provider_params
                    .get("title")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
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
        let profile = crate::transformers::request::MappingProfile {
            provider_id: "gemini",
            rules: vec![], // no generic rules; hook builds typed JSON
            merge_strategy: crate::transformers::request::ProviderParamsMergeStrategy::Flatten,
        };
        let generic = crate::transformers::request::GenericRequestTransformer { profile, hooks };
        generic.transform_embedding(req)
    }

    fn transform_image(&self, req: &ImageGenerationRequest) -> Result<serde_json::Value, LlmError> {
        // Use Generic hooks to build typed JSON for generateContent (IMAGE)
        struct GeminiImageHooks(super::types::GeminiConfig);
        impl crate::transformers::request::ProviderRequestHooks for GeminiImageHooks {
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
        let profile = crate::transformers::request::MappingProfile {
            provider_id: "gemini",
            rules: vec![],
            merge_strategy: crate::transformers::request::ProviderParamsMergeStrategy::Flatten,
        };
        let generic = crate::transformers::request::GenericRequestTransformer { profile, hooks };
        generic.transform_image(req)
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
        let mut tool_calls = Vec::new();
        let mut content_parts = Vec::new();
        let mut thinking_content = String::new();
        let mut has_multimodal_content = false;

        for part in &content.parts {
            match part {
                Part::Text { text, thought } => {
                    if thought.unwrap_or(false) {
                        if !thinking_content.is_empty() {
                            thinking_content.push('\n');
                        }
                        thinking_content.push_str(text);
                    } else {
                        if !text_content.is_empty() {
                            text_content.push('\n');
                        }
                        text_content.push_str(text);
                        content_parts.push(crate::types::ContentPart::Text { text: text.clone() });
                    }
                }
                Part::InlineData { inline_data } => {
                    has_multimodal_content = true;
                    let data_url =
                        format!("data:{};base64,{}", inline_data.mime_type, inline_data.data);
                    if inline_data.mime_type.starts_with("image/") {
                        content_parts.push(crate::types::ContentPart::Image {
                            image_url: data_url,
                            detail: None,
                        });
                    } else if inline_data.mime_type.starts_with("audio/") {
                        content_parts.push(crate::types::ContentPart::Audio {
                            audio_url: data_url,
                            format: inline_data.mime_type.clone(),
                        });
                    }
                }
                Part::FileData { file_data } => {
                    has_multimodal_content = true;
                    let mime_type = file_data
                        .mime_type
                        .as_deref()
                        .unwrap_or("application/octet-stream");
                    if mime_type.starts_with("image/") {
                        content_parts.push(crate::types::ContentPart::Image {
                            image_url: file_data.file_uri.clone(),
                            detail: None,
                        });
                    } else if mime_type.starts_with("audio/") {
                        content_parts.push(crate::types::ContentPart::Audio {
                            audio_url: file_data.file_uri.clone(),
                            format: mime_type.to_string(),
                        });
                    }
                }
                Part::FunctionCall { function_call } => {
                    let arguments = if let Some(args) = &function_call.args {
                        serde_json::to_string(args).unwrap_or_default()
                    } else {
                        "{}".to_string()
                    };
                    tool_calls.push(ToolCall {
                        id: format!("call_{}", uuid::Uuid::new_v4()),
                        r#type: "function".to_string(),
                        function: Some(FunctionCall {
                            name: function_call.name.clone(),
                            arguments,
                        }),
                    });
                }
                _ => {}
            }
        }

        let usage = response.usage_metadata.as_ref().map(|m| Usage {
            prompt_tokens: m.prompt_token_count.unwrap_or(0) as u32,
            completion_tokens: m.candidates_token_count.unwrap_or(0) as u32,
            total_tokens: m.total_token_count.unwrap_or(0) as u32,
            cached_tokens: None,
            reasoning_tokens: m.thoughts_token_count.map(|t| t as u32),
        });

        let finish_reason = candidate.finish_reason.as_ref().map(|reason| match reason {
            super::types::FinishReason::Stop => FinishReason::Stop,
            super::types::FinishReason::MaxTokens => FinishReason::Length,
            super::types::FinishReason::Safety => FinishReason::ContentFilter,
            _ => FinishReason::Other("unknown".to_string()),
        });

        let content = if has_multimodal_content && !content_parts.is_empty() {
            MessageContent::MultiModal(content_parts)
        } else if text_content.is_empty() {
            MessageContent::Text(String::new())
        } else {
            MessageContent::Text(text_content)
        };

        Ok(ChatResponse {
            id: None,
            content,
            model: None,
            usage,
            finish_reason,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            thinking: if thinking_content.is_empty() {
                None
            } else {
                Some(thinking_content)
            },
            metadata: std::collections::HashMap::new(),
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
    use crate::transformers::files::{FilesHttpBody, FilesTransformer};

    fn sample_config() -> GeminiConfig {
        GeminiConfig {
            api_key: "x".into(),
            base_url: "https://example.com".into(),
            model: "gemini-1.5-flash".into(),
            generation_config: None,
            safety_settings: None,
            timeout: Some(30),
            http_config: Some(crate::types::HttpConfig::default()),
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
    use crate::transformers::request::RequestTransformer;
    use crate::transformers::response::ResponseTransformer;

    fn cfg() -> GeminiConfig {
        GeminiConfig {
            api_key: "x".into(),
            base_url: "https://example.com".into(),
            model: "gemini-1.5-flash".into(),
            generation_config: None,
            safety_settings: None,
            timeout: Some(30),
            http_config: Some(crate::types::HttpConfig::default()),
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
    use crate::transformers::request::RequestTransformer;
    use crate::transformers::response::ResponseTransformer;

    fn cfg() -> GeminiConfig {
        GeminiConfig {
            api_key: "x".into(),
            base_url: "https://example.com".into(),
            model: "gemini-embedding-001".into(),
            generation_config: None,
            safety_settings: None,
            timeout: Some(30),
            http_config: Some(crate::types::HttpConfig::default()),
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
            .with_provider_param("title", serde_json::Value::String("My Title".into()));

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
            dyn Future<Output = Vec<Result<crate::stream::ChatStreamEvent, LlmError>>>
                + Send
                + Sync
                + '_,
        >,
    > {
        self.inner.convert_event(event)
    }

    fn handle_stream_end(&self) -> Option<Result<crate::stream::ChatStreamEvent, LlmError>> {
        self.inner.handle_stream_end()
    }
}
