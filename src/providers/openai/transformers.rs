//! OpenAI native transformers for Chat, Embedding, Image, and Stream chunks

use crate::error::LlmError;
use crate::transformers::files::{FilesHttpBody, FilesTransformer};
use crate::transformers::{
    request::{ImageHttpBody, RequestTransformer},
    response::ResponseTransformer,
    stream::StreamChunkTransformer,
};
use crate::types::{
    ChatRequest, ChatResponse, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage, GeneratedImage,
    ImageEditRequest, ImageGenerationRequest, ImageGenerationResponse, ImageVariationRequest,
};
use crate::utils::streaming::SseEventConverter;
use reqwest::multipart::{Form, Part};
use std::future::Future;
use std::pin::Pin;

#[derive(Clone)]
pub struct OpenAiRequestTransformer;

impl RequestTransformer for OpenAiRequestTransformer {
    fn provider_id(&self) -> &str {
        "openai"
    }

    fn transform_chat(&self, req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
        // Minimal provider-specific validations centralized here
        if req.common_params.model.is_empty() {
            return Err(LlmError::InvalidParameter(
                "Model must be specified".to_string(),
            ));
        }
        // OpenAI tools upper bound (align with legacy checks)
        if let Some(tools) = &req.tools
            && tools.len() > 128
        {
            return Err(LlmError::InvalidParameter(
                "OpenAI supports maximum 128 tools per request".to_string(),
            ));
        }
        // o1-* models do not support temperature / top_p
        if req.common_params.model.starts_with("o1-") {
            if req.common_params.temperature.is_some() {
                return Err(LlmError::InvalidParameter(
                    "o1 models do not support temperature parameter".to_string(),
                ));
            }
            if req.common_params.top_p.is_some() {
                return Err(LlmError::InvalidParameter(
                    "o1 models do not support top_p parameter".to_string(),
                ));
            }
        }

        // Build body directly (no ParameterMapper)
        use crate::providers::openai::utils::convert_messages;
        let mut body = serde_json::json!({ "model": req.common_params.model });
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
            body["stop"] = serde_json::json!(stops);
        }
        // Messages
        let messages = convert_messages(&req.messages)?;
        body["messages"] = serde_json::to_value(messages)?;

        // Tools
        if let Some(tools) = &req.tools
            && !tools.is_empty()
        {
            body["tools"] = serde_json::to_value(tools)?;
        }

        // Streaming flags
        if req.stream {
            body["stream"] = serde_json::Value::Bool(true);
            body["stream_options"] = serde_json::json!({ "include_usage": true });
        }
        // Merge provider_params (flat into body)
        if let Some(pp) = &req.provider_params
            && let Some(obj) = body.as_object_mut()
        {
            for (k, v) in &pp.params {
                obj.insert(k.clone(), v.clone());
            }
        }

        // Clean nulls that can cause API errors
        if let serde_json::Value::Object(obj) = &mut body {
            let keys: Vec<String> = obj
                .iter()
                .filter_map(|(k, v)| if v.is_null() { Some(k.clone()) } else { None })
                .collect();
            for k in keys {
                obj.remove(&k);
            }
        }
        Ok(body)
    }

    fn transform_embedding(&self, req: &EmbeddingRequest) -> Result<serde_json::Value, LlmError> {
        let model = req
            .model
            .clone()
            .unwrap_or_else(|| "text-embedding-3-small".to_string());
        let encoding_format = req.encoding_format.as_ref().map(|f| match f {
            crate::types::EmbeddingFormat::Float => "float".to_string(),
            crate::types::EmbeddingFormat::Base64 => "base64".to_string(),
        });
        let mut json = serde_json::json!({
            "input": req.input,
            "model": model,
        });
        if let Some(fmt) = encoding_format {
            json["encoding_format"] = serde_json::json!(fmt);
        }
        if let Some(dim) = req.dimensions {
            json["dimensions"] = serde_json::json!(dim);
        }
        if let Some(user) = &req.user {
            json["user"] = serde_json::json!(user);
        }
        // merge provider params
        if let Some(obj) = json.as_object_mut() {
            for (k, v) in &req.provider_params {
                obj.insert(k.clone(), v.clone());
            }
        }
        Ok(json)
    }

    fn transform_image(
        &self,
        request: &ImageGenerationRequest,
    ) -> Result<serde_json::Value, LlmError> {
        // Map to OpenAI Images API request (dall-e/gpt-image-1) format
        let mut body = serde_json::json!({
            "prompt": request.prompt,
        });
        if let Some(n) = Some(request.count).filter(|c| *c > 0) {
            body["n"] = serde_json::json!(n);
        }
        if let Some(size) = &request.size {
            body["size"] = serde_json::json!(size);
        }
        if let Some(q) = &request.quality {
            body["quality"] = serde_json::json!(q);
        }
        if let Some(style) = &request.style {
            body["style"] = serde_json::json!(style);
        }
        if let Some(fmt) = &request.response_format {
            body["response_format"] = serde_json::json!(fmt);
        }
        if let Some(model) = &request.model {
            body["model"] = serde_json::json!(model);
        }
        if let Some(neg) = &request.negative_prompt {
            body["negative_prompt"] = serde_json::json!(neg);
        }
        // extra params
        if let Some(obj) = body.as_object_mut() {
            for (k, v) in &request.extra_params {
                obj.insert(k.clone(), v.clone());
            }
        }
        Ok(body)
    }

    fn transform_image_edit(&self, req: &ImageEditRequest) -> Result<ImageHttpBody, LlmError> {
        // Build multipart form for OpenAI Images Edit
        let mut form = Form::new().text("prompt", req.prompt.clone());
        let image_mime = crate::utils::mime::guess_mime(Some(&req.image), None);
        let image_part = Part::bytes(req.image.clone())
            .file_name("image")
            .mime_str(&image_mime)?;
        form = form.part("image", image_part);
        if let Some(mask) = &req.mask {
            let mask_mime = crate::utils::mime::guess_mime(Some(mask), None);
            let mask_part = Part::bytes(mask.clone())
                .file_name("mask")
                .mime_str(&mask_mime)?;
            form = form.part("mask", mask_part);
        }
        if let Some(size) = &req.size {
            form = form.text("size", size.clone());
        }
        if let Some(n) = req.count {
            form = form.text("n", n.to_string());
        }
        if let Some(fmt) = &req.response_format {
            form = form.text("response_format", fmt.clone());
        }
        Ok(ImageHttpBody::Multipart(form))
    }

    fn transform_image_variation(
        &self,
        req: &ImageVariationRequest,
    ) -> Result<ImageHttpBody, LlmError> {
        // Build multipart form for OpenAI Images Variation
        let mut form = Form::new();
        let image_mime = crate::utils::mime::guess_mime(Some(&req.image), None);
        let image_part = Part::bytes(req.image.clone())
            .file_name("image")
            .mime_str(&image_mime)?;
        form = form.part("image", image_part);
        if let Some(size) = &req.size {
            form = form.text("size", size.clone());
        }
        if let Some(n) = req.count {
            form = form.text("n", n.to_string());
        }
        if let Some(fmt) = &req.response_format {
            form = form.text("response_format", fmt.clone());
        }
        Ok(ImageHttpBody::Multipart(form))
    }
}

#[derive(Clone)]
pub struct OpenAiResponseTransformer;

impl ResponseTransformer for OpenAiResponseTransformer {
    fn provider_id(&self) -> &str {
        "openai"
    }

    fn transform_chat_response(&self, raw: &serde_json::Value) -> Result<ChatResponse, LlmError> {
        // Delegate to OpenAI-compatible response transformer for robust mapping
        use crate::providers::openai_compatible::adapter::{
            ProviderAdapter, ProviderCompatibility,
        };
        use crate::providers::openai_compatible::types::FieldMappings;
        use crate::traits::ProviderCapabilities;
        #[derive(Debug, Clone)]
        struct OpenAiStandardAdapter {
            base_url: String,
        }
        impl ProviderAdapter for OpenAiStandardAdapter {
            fn provider_id(&self) -> &'static str {
                "openai"
            }
            fn transform_request_params(
                &self,
                _params: &mut serde_json::Value,
                _model: &str,
                _ty: crate::providers::openai_compatible::types::RequestType,
            ) -> Result<(), LlmError> {
                Ok(())
            }
            fn get_field_mappings(&self, _model: &str) -> FieldMappings {
                FieldMappings::standard()
            }
            fn get_model_config(
                &self,
                _model: &str,
            ) -> crate::providers::openai_compatible::types::ModelConfig {
                crate::providers::openai_compatible::types::ModelConfig::default()
            }
            fn capabilities(&self) -> ProviderCapabilities {
                ProviderCapabilities::new()
                    .with_chat()
                    .with_streaming()
                    .with_tools()
            }
            fn compatibility(&self) -> ProviderCompatibility {
                ProviderCompatibility::openai_standard()
            }
            fn base_url(&self) -> &str {
                &self.base_url
            }
            fn clone_adapter(&self) -> Box<dyn ProviderAdapter> {
                Box::new(self.clone())
            }
        }
        let model = raw
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let adapter: std::sync::Arc<
            dyn crate::providers::openai_compatible::adapter::ProviderAdapter,
        > = std::sync::Arc::new(OpenAiStandardAdapter {
            base_url: String::new(),
        });
        let cfg = crate::providers::openai_compatible::openai_config::OpenAiCompatibleConfig::new(
            "openai",
            "",
            "",
            adapter.clone(),
        )
        .with_model(&model);
        let compat = crate::providers::openai_compatible::transformers::CompatResponseTransformer {
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
        Ok(ImageGenerationResponse { images, metadata })
    }
}

/// Stream chunk transformer wrapping the OpenAI-compatible converter for OpenAI
#[derive(Clone)]
pub struct OpenAiStreamChunkTransformer {
    pub provider_id: String,
    pub inner: crate::providers::openai_compatible::streaming::OpenAiCompatibleEventConverter,
}

impl StreamChunkTransformer for OpenAiStreamChunkTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn convert_event(
        &self,
        event: eventsource_stream::Event,
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

/// Files transformer for OpenAI
#[derive(Clone)]
pub struct OpenAiFilesTransformer;

impl FilesTransformer for OpenAiFilesTransformer {
    fn provider_id(&self) -> &str {
        "openai"
    }

    fn build_upload_body(
        &self,
        req: &crate::types::FileUploadRequest,
    ) -> Result<FilesHttpBody, LlmError> {
        let detected = req.mime_type.clone().unwrap_or_else(|| {
            crate::utils::mime::guess_mime(Some(&req.content), Some(&req.filename))
        });
        let part = reqwest::multipart::Part::bytes(req.content.clone())
            .file_name(req.filename.clone())
            .mime_str(&detected)
            .map_err(|e| LlmError::HttpError(format!("Invalid MIME type: {e}")))?;
        let form = reqwest::multipart::Form::new()
            .text("purpose", req.purpose.clone())
            .part("file", part);
        Ok(FilesHttpBody::Multipart(form))
    }

    fn list_endpoint(&self, query: &Option<crate::types::FileListQuery>) -> String {
        let mut endpoint = "files".to_string();
        if let Some(q) = query {
            let mut params = Vec::new();
            if let Some(purpose) = &q.purpose {
                params.push(format!("purpose={}", urlencoding::encode(purpose)));
            }
            if let Some(limit) = q.limit {
                params.push(format!("limit={limit}"));
            }
            if let Some(after) = &q.after {
                params.push(format!("after={}", urlencoding::encode(after)));
            }
            if let Some(order) = &q.order {
                params.push(format!("order={}", urlencoding::encode(order)));
            }
            if !params.is_empty() {
                endpoint.push('?');
                endpoint.push_str(&params.join("&"));
            }
        }
        endpoint
    }

    fn retrieve_endpoint(&self, file_id: &str) -> String {
        format!("files/{file_id}")
    }
    fn delete_endpoint(&self, file_id: &str) -> String {
        format!("files/{file_id}")
    }
    fn content_endpoint(&self, file_id: &str) -> Option<String> {
        Some(format!("files/{file_id}/content"))
    }

    fn transform_file_object(
        &self,
        raw: &serde_json::Value,
    ) -> Result<crate::types::FileObject, LlmError> {
        #[derive(serde::Deserialize)]
        struct OpenAiFileResponse {
            id: String,
            object: String,
            bytes: u64,
            created_at: u64,
            filename: String,
            purpose: String,
            status: String,
            status_details: Option<String>,
        }
        let f: OpenAiFileResponse = serde_json::from_value(raw.clone())
            .map_err(|e| LlmError::ParseError(format!("Failed to parse file: {e}")))?;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("object".to_string(), serde_json::json!(f.object));
        metadata.insert("status".to_string(), serde_json::json!(f.status));
        if let Some(d) = f.status_details {
            metadata.insert("status_details".to_string(), serde_json::json!(d));
        }
        Ok(crate::types::FileObject {
            id: f.id,
            filename: f.filename,
            bytes: f.bytes,
            created_at: f.created_at,
            purpose: f.purpose,
            status: "uploaded".to_string(),
            mime_type: None,
            metadata,
        })
    }

    fn transform_list_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<crate::types::FileListResponse, LlmError> {
        #[derive(serde::Deserialize)]
        struct OpenAiFileResponse {
            id: String,
            object: String,
            bytes: u64,
            created_at: u64,
            filename: String,
            purpose: String,
            status: String,
            status_details: Option<String>,
        }
        #[derive(serde::Deserialize)]
        struct OpenAiFileListResponse {
            data: Vec<OpenAiFileResponse>,
            has_more: Option<bool>,
        }
        let r: OpenAiFileListResponse = serde_json::from_value(raw.clone())
            .map_err(|e| LlmError::ParseError(format!("Failed to parse list: {e}")))?;
        let files = r
            .data
            .into_iter()
            .map(|f| {
                let mut metadata = std::collections::HashMap::new();
                metadata.insert("object".to_string(), serde_json::json!(f.object));
                metadata.insert("status".to_string(), serde_json::json!(f.status));
                if let Some(d) = f.status_details.clone() {
                    metadata.insert("status_details".to_string(), serde_json::json!(d));
                }
                crate::types::FileObject {
                    id: f.id,
                    filename: f.filename,
                    bytes: f.bytes,
                    created_at: f.created_at,
                    purpose: f.purpose,
                    status: "uploaded".to_string(),
                    mime_type: None,
                    metadata,
                }
            })
            .collect();
        Ok(crate::types::FileListResponse {
            files,
            has_more: r.has_more.unwrap_or(false),
            next_cursor: None,
        })
    }
}

#[cfg(test)]
mod files_tests {
    use super::*;
    use crate::transformers::files::{FilesHttpBody, FilesTransformer};

    #[test]
    fn test_openai_files_endpoints() {
        let tx = OpenAiFilesTransformer;
        // list endpoint with params
        let q = crate::types::FileListQuery {
            purpose: Some("assistants".to_string()),
            limit: Some(10),
            after: Some("cursor123".to_string()),
            order: Some("desc".to_string()),
        };
        let ep = tx.list_endpoint(&Some(q));
        assert!(ep.starts_with("files?"));
        assert!(ep.contains("purpose=assistants"));
        assert!(ep.contains("limit=10"));
        assert!(ep.contains("after=cursor123"));
        assert!(ep.contains("order=desc"));

        assert_eq!(tx.retrieve_endpoint("file_1"), "files/file_1");
        assert_eq!(tx.delete_endpoint("file_2"), "files/file_2");
        assert_eq!(
            tx.content_endpoint("file_3"),
            Some("files/file_3/content".to_string())
        );
    }

    #[test]
    fn test_openai_files_upload_and_parse() {
        let tx = OpenAiFilesTransformer;
        let req = crate::types::FileUploadRequest {
            content: b"hello".to_vec(),
            filename: "hello.txt".to_string(),
            mime_type: Some("text/plain".to_string()),
            purpose: "assistants".to_string(),
            metadata: std::collections::HashMap::new(),
        };
        match tx.build_upload_body(&req).unwrap() {
            FilesHttpBody::Multipart(_) => {}
            _ => panic!("expected multipart form for OpenAI upload"),
        }

        // file object
        let json = serde_json::json!({
            "id": "file_123",
            "object": "file",
            "bytes": 12,
            "created_at": 1710000000u64,
            "filename": "hello.txt",
            "purpose": "assistants",
            "status": "uploaded",
            "status_details": null
        });
        let fo = tx.transform_file_object(&json).unwrap();
        assert_eq!(fo.id, "file_123");
        assert_eq!(fo.filename, "hello.txt");
        assert_eq!(fo.purpose, "assistants");

        // list response
        let list = serde_json::json!({
            "object": "list",
            "data": [json],
            "has_more": false
        });
        let lr = tx.transform_list_response(&list).unwrap();
        assert_eq!(lr.files.len(), 1);
        assert!(!lr.has_more);
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

/// Request transformer for OpenAI Responses API
#[derive(Clone)]
pub struct OpenAiResponsesRequestTransformer;

impl OpenAiResponsesRequestTransformer {
    fn convert_message(msg: &crate::types::ChatMessage) -> Result<serde_json::Value, LlmError> {
        use crate::types::{ContentPart, MessageContent, MessageRole};

        // Tool role message becomes function_call_output item
        if matches!(msg.role, MessageRole::Tool) {
            let call_id = msg.tool_call_id.as_ref().ok_or_else(|| {
                LlmError::InvalidInput("Tool message missing tool_call_id".into())
            })?;
            let output_text = match &msg.content {
                MessageContent::Text(t) => t.clone(),
                MessageContent::MultiModal(_) => String::new(),
            };
            return Ok(serde_json::json!({
                "type": "function_call_output",
                "call_id": call_id,
                "output": output_text,
            }));
        }

        // Base message with role
        let role = match msg.role {
            crate::types::MessageRole::System => "system",
            crate::types::MessageRole::User => "user",
            crate::types::MessageRole::Assistant => "assistant",
            crate::types::MessageRole::Developer => "developer",
            crate::types::MessageRole::Tool => "user",
        };
        let mut api_message = serde_json::json!({ "role": role });

        // Assistant tool calls → tool_use content parts
        if matches!(msg.role, MessageRole::Assistant) && msg.tool_calls.is_some() {
            let mut content_parts: Vec<serde_json::Value> = Vec::new();
            match &msg.content {
                MessageContent::Text(text) => {
                    if !text.is_empty() {
                        content_parts
                            .push(serde_json::json!({ "type": "input_text", "text": text }));
                    }
                }
                MessageContent::MultiModal(parts) => {
                    for part in parts {
                        if let ContentPart::Text { text } = part {
                            content_parts
                                .push(serde_json::json!({ "type": "input_text", "text": text }));
                        }
                    }
                }
            }
            if let Some(tool_calls) = &msg.tool_calls {
                for call in tool_calls {
                    let (name, args_str) = if let Some(func) = &call.function {
                        (func.name.clone(), func.arguments.clone())
                    } else {
                        (String::new(), String::new())
                    };
                    if !name.is_empty() {
                        let input_json = serde_json::from_str::<serde_json::Value>(&args_str)
                            .unwrap_or_else(|_| serde_json::json!({}));
                        content_parts.push(serde_json::json!({
                            "type": "tool_use",
                            "id": call.id,
                            "name": name,
                            "input": input_json
                        }));
                    }
                }
            }
            api_message["content"] = serde_json::Value::Array(content_parts);
            if let Some(tool_call_id) = &msg.tool_call_id {
                api_message["tool_call_id"] = serde_json::Value::String(tool_call_id.clone());
            }
            return Ok(api_message);
        }

        // Default content handling
        match &msg.content {
            MessageContent::Text(text) => {
                api_message["content"] = serde_json::Value::Array(vec![serde_json::json!({
                    "type": "input_text",
                    "text": text
                })]);
            }
            MessageContent::MultiModal(parts) => {
                let mut content_parts = Vec::new();
                for part in parts {
                    match part {
                        ContentPart::Text { text } => {
                            content_parts
                                .push(serde_json::json!({ "type": "input_text", "text": text }));
                        }
                        ContentPart::Image { image_url, detail } => {
                            let mut image_part = serde_json::json!({
                                "type": "image_url",
                                "image_url": { "url": image_url }
                            });
                            if let Some(d) = detail {
                                image_part["image_url"]["detail"] =
                                    serde_json::Value::String(d.clone());
                            }
                            content_parts.push(image_part);
                        }
                        ContentPart::Audio { audio_url, format } => {
                            content_parts.push(serde_json::json!({ "type": "audio", "audio_url": audio_url, "format": format }));
                        }
                    }
                }
                api_message["content"] = serde_json::Value::Array(content_parts);
            }
        }

        if let Some(tool_call_id) = &msg.tool_call_id {
            api_message["tool_call_id"] = serde_json::Value::String(tool_call_id.clone());
        }
        Ok(api_message)
    }
}

impl RequestTransformer for OpenAiResponsesRequestTransformer {
    fn provider_id(&self) -> &str {
        "openai_responses"
    }

    fn transform_chat(&self, req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
        // Build base body
        let mut body = serde_json::json!({
            "model": req.common_params.model,
            "stream": req.stream,
        });

        // input
        let mut input_items = Vec::with_capacity(req.messages.len());
        for m in &req.messages {
            input_items.push(Self::convert_message(m)?);
        }
        body["input"] = serde_json::Value::Array(input_items);

        // tools (flattened)
        if let Some(tools) = &req.tools {
            let t: Vec<serde_json::Value> = tools
                .iter()
                .map(|tool| {
                    serde_json::json!({
                        "type": tool.r#type,
                        "name": tool.function.name,
                        "description": tool.function.description,
                        "parameters": tool.function.parameters
                    })
                })
                .collect();
            body["tools"] = serde_json::Value::Array(t);
        }

        // stream options
        if req.stream {
            body["stream_options"] = serde_json::json!({ "include_usage": true });
        }

        // temperature
        if let Some(temp) = req.common_params.temperature {
            body["temperature"] = serde_json::json!(temp);
        }

        // max_output_tokens (prefer common max_tokens)
        if let Some(max_tokens) = req.common_params.max_tokens {
            body["max_output_tokens"] = serde_json::json!(max_tokens);
        }

        // seed
        if let Some(seed) = req.common_params.seed {
            body["seed"] = serde_json::json!(seed);
        }

        Ok(body)
    }
}

/// Response transformer for OpenAI Responses API
#[derive(Clone)]
pub struct OpenAiResponsesResponseTransformer;

impl ResponseTransformer for OpenAiResponsesResponseTransformer {
    fn provider_id(&self) -> &str {
        "openai_responses"
    }

    fn transform_chat_response(&self, raw: &serde_json::Value) -> Result<ChatResponse, LlmError> {
        use crate::types::{FinishReason, FunctionCall, MessageContent, ToolCall, Usage};
        let root = raw.get("response").unwrap_or(raw);

        // Extract text content from output[*].content[*].text
        let mut text_content = String::new();
        if let Some(output) = root.get("output").and_then(|v| v.as_array()) {
            for item in output {
                if let Some(parts) = item.get("content").and_then(|c| c.as_array()) {
                    for p in parts {
                        if let Some(t) = p.get("text").and_then(|v| v.as_str()) {
                            if !text_content.is_empty() {
                                text_content.push('\n');
                            }
                            text_content.push_str(t);
                        }
                    }
                }
            }
        }

        // Tool calls (support nested function object or flattened)
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        if let Some(output) = root.get("output").and_then(|v| v.as_array()) {
            for item in output {
                if let Some(calls) = item.get("tool_calls").and_then(|tc| tc.as_array()) {
                    for call in calls {
                        let id = call
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let (name, arguments) = if let Some(f) = call.get("function") {
                            (
                                f.get("name")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string(),
                                f.get("arguments")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string(),
                            )
                        } else {
                            (
                                call.get("name")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string(),
                                call.get("arguments")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string(),
                            )
                        };
                        if !name.is_empty() {
                            tool_calls.push(ToolCall {
                                id,
                                r#type: "function".into(),
                                function: Some(FunctionCall { name, arguments }),
                            });
                        }
                    }
                }
            }
        }

        // Usage
        let usage = root.get("usage").map(|u| Usage {
            prompt_tokens: u
                .get("input_tokens")
                .or_else(|| u.get("prompt_tokens"))
                .or_else(|| u.get("inputTokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            completion_tokens: u
                .get("output_tokens")
                .or_else(|| u.get("completion_tokens"))
                .or_else(|| u.get("outputTokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            total_tokens: u
                .get("total_tokens")
                .or_else(|| u.get("totalTokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            reasoning_tokens: u
                .get("reasoning_tokens")
                .or_else(|| u.get("reasoningTokens"))
                .and_then(|v| v.as_u64())
                .map(|v| v as u32),
            cached_tokens: None,
        });

        // Finish reason
        let finish_reason = root
            .get("finish_reason")
            .or_else(|| root.get("stop_reason"))
            .and_then(|v| v.as_str())
            .map(|s| match s {
                "stop" => FinishReason::Stop,
                "length" | "max_tokens" => FinishReason::Length,
                "tool_calls" | "tool_use" | "function_call" => FinishReason::ToolCalls,
                "content_filter" | "safety" => FinishReason::ContentFilter,
                other => FinishReason::Other(other.to_string()),
            });

        Ok(ChatResponse {
            id: root
                .get("id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            content: MessageContent::Text(text_content),
            model: root
                .get("model")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            usage,
            finish_reason,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            thinking: None,
            metadata: std::collections::HashMap::new(),
        })
    }
}

/// Stream transformer for OpenAI Responses API using existing converter
#[derive(Clone)]
pub struct OpenAiResponsesStreamChunkTransformer {
    pub provider_id: String,
    pub inner: crate::providers::openai::responses::OpenAiResponsesEventConverter,
}

impl StreamChunkTransformer for OpenAiResponsesStreamChunkTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }
    fn convert_event(
        &self,
        event: eventsource_stream::Event,
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
        None
    }
}
