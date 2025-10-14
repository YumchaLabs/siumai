//! Request transformers for OpenAI (Chat/Embedding/Images) and OpenAI Responses API

use crate::error::LlmError;
use crate::transformers::request::{ImageHttpBody, RequestTransformer};
use crate::types::{
    ChatRequest, EmbeddingRequest, ImageEditRequest, ImageGenerationRequest, ImageVariationRequest,
};
use reqwest::multipart::{Form, Part};

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
        let mut json = serde_json::json!({ "input": req.input, "model": model });
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
        let mut body = serde_json::json!({ "prompt": request.prompt });
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
