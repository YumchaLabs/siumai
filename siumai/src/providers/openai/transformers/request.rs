//! Request transformers for OpenAI (Chat/Embedding/Images) and OpenAI Responses API

use crate::error::LlmError;
use crate::transformers::request::{
    Condition, GenericRequestTransformer, MappingProfile, ProviderParamsMergeStrategy,
    ProviderRequestHooks, RangeMode, Rule,
};
use crate::transformers::request::{ImageHttpBody, RequestTransformer};
use crate::types::{
    ChatRequest, EmbeddingRequest, ImageEditRequest, ImageGenerationRequest, ImageVariationRequest,
    ModerationRequest, RerankRequest,
};
use reqwest::multipart::{Form, Part};

#[derive(Clone)]
pub struct OpenAiRequestTransformer;

impl RequestTransformer for OpenAiRequestTransformer {
    fn provider_id(&self) -> &str {
        "openai"
    }

    fn transform_chat(&self, req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
        // Minimal provider-agnostic validation
        if req.common_params.model.is_empty() {
            return Err(LlmError::InvalidParameter(
                "Model must be specified".to_string(),
            ));
        }

        // Build via GenericRequestTransformer (profile + hooks)
        struct OpenAiChatHooks;
        impl ProviderRequestHooks for OpenAiChatHooks {
            fn build_base_chat_body(
                &self,
                req: &ChatRequest,
            ) -> Result<serde_json::Value, LlmError> {
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
                    body["stop_sequences"] = serde_json::json!(stops);
                }

                let messages = convert_messages(&req.messages)?;
                body["messages"] = serde_json::to_value(messages)?;

                if let Some(tools) = &req.tools
                    && !tools.is_empty()
                {
                    body["tools"] = serde_json::to_value(tools)?;
                }

                if req.stream {
                    body["stream"] = serde_json::Value::Bool(true);
                    body["stream_options"] = serde_json::json!({ "include_usage": true });
                }
                Ok(body)
            }

            fn post_process_chat(
                &self,
                req: &ChatRequest,
                body: &mut serde_json::Value,
            ) -> Result<(), LlmError> {
                // Map structured_output provider hint to OpenAI chat `response_format`
                if let Some(pp) = &req.provider_params {
                    if let Some(so) = pp
                        .params
                        .get("structured_output")
                        .and_then(|v| v.as_object())
                    {
                        let mode = so.get("mode").and_then(|v| v.as_str()).unwrap_or("auto");
                        let schema = so.get("schema");
                        let name = so.get("schema_name").and_then(|v| v.as_str());
                        if let Some(schema_v) = schema.cloned() {
                            let rf = if let Some(n) = name {
                                serde_json::json!({
                                    "type": "json_schema",
                                    "json_schema": {"name": n, "schema": schema_v, "strict": true}
                                })
                            } else {
                                serde_json::json!({
                                    "type": "json_object",
                                    "json_schema": {"schema": schema_v, "strict": true}
                                })
                            };
                            body["response_format"] = rf;
                        } else if mode == "json" {
                            body["response_format"] = serde_json::json!({"type":"json_object"});
                        }
                    }
                }
                // Drop leftover hint if merged by rules
                if let Some(obj) = body.as_object_mut() {
                    obj.remove("structured_output");
                }
                Ok(())
            }
        }

        let profile = MappingProfile {
            provider_id: "openai",
            rules: vec![
                // stop_sequences -> stop (OpenAI specific)
                Rule::Move {
                    from: "stop_sequences",
                    to: "stop",
                },
                // Stable ranges: temperature and top_p
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
                // Model condition: o1-* models forbid temperature and top_p
                Rule::ForbidWhen {
                    field: "temperature",
                    condition: Condition::ModelPrefix("o1-"),
                    message: "o1 models do not support temperature parameter",
                },
                Rule::ForbidWhen {
                    field: "top_p",
                    condition: Condition::ModelPrefix("o1-"),
                    message: "o1 models do not support top_p parameter",
                },
                // Tools upper bound (stable, per official docs)
                Rule::MaxLen {
                    field: "tools",
                    max: 128,
                    message: "OpenAI supports maximum 128 tools per request",
                },
                // Flatten provider_params into the top-level body
                Rule::MergeProviderParams {
                    strategy: ProviderParamsMergeStrategy::Flatten,
                },
                // Drop high-level structured_output hint for chat/completions path
                Rule::Drop {
                    field: "structured_output",
                },
            ],
            merge_strategy: ProviderParamsMergeStrategy::Flatten,
        };

        let generic = GenericRequestTransformer {
            profile,
            hooks: OpenAiChatHooks,
        };
        generic.transform_chat(req)
    }

    fn transform_embedding(&self, req: &EmbeddingRequest) -> Result<serde_json::Value, LlmError> {
        // Reuse GenericRequestTransformer via embedding hooks (behavior unchanged)
        struct OpenAiEmbeddingHooks;
        impl ProviderRequestHooks for OpenAiEmbeddingHooks {
            fn build_base_embedding_body(
                &self,
                req: &EmbeddingRequest,
            ) -> Result<serde_json::Value, LlmError> {
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
                Ok(json)
            }
        }

        let hooks = OpenAiEmbeddingHooks;
        let profile = MappingProfile {
            provider_id: "openai",
            rules: vec![
                // Flatten provider params into top-level for embeddings
                Rule::MergeProviderParams {
                    strategy: ProviderParamsMergeStrategy::Flatten,
                },
            ],
            merge_strategy: ProviderParamsMergeStrategy::Flatten,
        };
        let generic = GenericRequestTransformer { profile, hooks };
        generic.transform_embedding(req)
    }

    fn transform_image(
        &self,
        request: &ImageGenerationRequest,
    ) -> Result<serde_json::Value, LlmError> {
        // Use Generic transformer with image hooks (preserve existing behavior)
        struct OpenAiImageHooks;
        impl ProviderRequestHooks for OpenAiImageHooks {
            fn build_base_image_body(
                &self,
                req: &ImageGenerationRequest,
            ) -> Result<serde_json::Value, LlmError> {
                let mut body = serde_json::json!({ "prompt": req.prompt });
                if let Some(n) = Some(req.count).filter(|c| *c > 0) {
                    body["n"] = serde_json::json!(n);
                }
                if let Some(size) = &req.size {
                    body["size"] = serde_json::json!(size);
                }
                if let Some(q) = &req.quality {
                    body["quality"] = serde_json::json!(q);
                }
                if let Some(style) = &req.style {
                    body["style"] = serde_json::json!(style);
                }
                if let Some(fmt) = &req.response_format {
                    body["response_format"] = serde_json::json!(fmt);
                }
                if let Some(model) = &req.model {
                    body["model"] = serde_json::json!(model);
                }
                if let Some(neg) = &req.negative_prompt {
                    body["negative_prompt"] = serde_json::json!(neg);
                }
                Ok(body)
            }

            fn post_process_image(
                &self,
                req: &ImageGenerationRequest,
                body: &mut serde_json::Value,
            ) -> Result<(), LlmError> {
                if let Some(obj) = body.as_object_mut() {
                    for (k, v) in &req.extra_params {
                        obj.insert(k.clone(), v.clone());
                    }
                }
                Ok(())
            }
        }

        let hooks = OpenAiImageHooks;
        let profile = MappingProfile {
            provider_id: "openai",
            rules: vec![
                // Flatten provider params for image generation as well
                Rule::MergeProviderParams {
                    strategy: ProviderParamsMergeStrategy::Flatten,
                },
            ],
            merge_strategy: ProviderParamsMergeStrategy::Flatten,
        };
        let generic = GenericRequestTransformer { profile, hooks };
        generic.transform_image(request)
    }

    fn transform_image_edit(&self, req: &ImageEditRequest) -> Result<ImageHttpBody, LlmError> {
        // Build multipart form for OpenAI Images Edit
        let mut form = Form::new().text("prompt", req.prompt.clone());
        let image_mime = crate::utils::guess_mime(Some(&req.image), None);
        let image_part = Part::bytes(req.image.clone())
            .file_name("image")
            .mime_str(&image_mime)?;
        form = form.part("image", image_part);
        if let Some(mask) = &req.mask {
            let mask_mime = crate::utils::guess_mime(Some(mask), None);
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
        let image_mime = crate::utils::guess_mime(Some(&req.image), None);
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

    fn transform_rerank(&self, req: &RerankRequest) -> Result<serde_json::Value, LlmError> {
        let mut payload = serde_json::json!({
            "model": req.model,
            "query": req.query,
            "documents": req.documents,
        });
        if let Some(instr) = &req.instruction {
            payload["instruction"] = serde_json::json!(instr);
        }
        if let Some(top_n) = req.top_n {
            payload["top_n"] = serde_json::json!(top_n);
        }
        if let Some(rd) = req.return_documents {
            payload["return_documents"] = serde_json::json!(rd);
        }
        if let Some(maxc) = req.max_chunks_per_doc {
            payload["max_chunks_per_doc"] = serde_json::json!(maxc);
        }
        if let Some(over) = req.overlap_tokens {
            payload["overlap_tokens"] = serde_json::json!(over);
        }
        Ok(payload)
    }

    fn transform_moderation(&self, req: &ModerationRequest) -> Result<serde_json::Value, LlmError> {
        let model = req
            .model
            .clone()
            .unwrap_or_else(|| "text-moderation-latest".to_string());

        let input_value = if let Some(arr) = &req.inputs {
            serde_json::Value::Array(
                arr.iter()
                    .map(|s| serde_json::Value::String(s.clone()))
                    .collect(),
            )
        } else {
            serde_json::Value::String(req.input.clone())
        };

        Ok(serde_json::json!({ "model": model, "input": input_value }))
    }
}

#[cfg(test)]
mod mapping_tests {
    use super::*;
    use crate::types::{ChatMessage, ChatRequest, ProviderParams};

    #[test]
    fn chat_transform_injects_response_format_from_structured_output_named() {
        let tx = OpenAiRequestTransformer;
        let mut req = ChatRequest::new(vec![ChatMessage::user("hi").build()]);
        req.common_params.model = "gpt-4o-mini".into();
        let mut hint = serde_json::Map::new();
        hint.insert("mode".into(), serde_json::json!("auto"));
        hint.insert("schema_name".into(), serde_json::json!("User"));
        hint.insert("schema".into(), serde_json::json!({"type":"object"}));
        req = req.with_provider_params(
            ProviderParams::new().with_param("structured_output", serde_json::Value::Object(hint)),
        );
        let body = tx.transform_chat(&req).expect("ok");
        assert_eq!(
            body.get("response_format")
                .and_then(|v| v.get("type"))
                .and_then(|v| v.as_str()),
            Some("json_schema")
        );
    }

    #[test]
    fn chat_transform_injects_response_format_from_structured_output_object() {
        let tx = OpenAiRequestTransformer;
        let mut req = ChatRequest::new(vec![ChatMessage::user("hi").build()]);
        req.common_params.model = "gpt-4o-mini".into();
        let mut hint = serde_json::Map::new();
        hint.insert("mode".into(), serde_json::json!("auto"));
        hint.insert("schema".into(), serde_json::json!({"type":"object"}));
        req = req.with_provider_params(
            ProviderParams::new().with_param("structured_output", serde_json::Value::Object(hint)),
        );
        let body = tx.transform_chat(&req).expect("ok");
        assert_eq!(
            body.get("response_format")
                .and_then(|v| v.get("type"))
                .and_then(|v| v.as_str()),
            Some("json_object")
        );
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
            // Prefer JSON output when structured content is used
            #[cfg(feature = "structured-messages")]
            if let MessageContent::Json(v) = &msg.content {
                return Ok(serde_json::json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output_json": v,
                }));
            }
            let output_text = match &msg.content {
                MessageContent::Text(t) => t.clone(),
                MessageContent::MultiModal(_) => String::new(),
                #[cfg(feature = "structured-messages")]
                MessageContent::Json(v) => serde_json::to_string(v).unwrap_or_default(),
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
                #[cfg(feature = "structured-messages")]
                MessageContent::Json(v) => {
                    // Prefer native JSON input item for Responses API
                    content_parts.push(serde_json::json!({
                        "type": "input_json",
                        "json": v
                    }));
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
                            // Responses API prefers `input_image` items
                            let mut image_part = serde_json::json!({
                                "type": "input_image",
                                "image_url": { "url": image_url }
                            });
                            if let Some(d) = detail {
                                image_part["image_url"]["detail"] =
                                    serde_json::Value::String(d.clone());
                            }
                            content_parts.push(image_part);
                        }
                        ContentPart::Audio { audio_url, format } => {
                            // Responses API prefers `input_audio` items
                            content_parts.push(serde_json::json!({ "type": "input_audio", "audio_url": audio_url, "format": format }));
                        }
                    }
                }
                api_message["content"] = serde_json::Value::Array(content_parts);
            }
            #[cfg(feature = "structured-messages")]
            MessageContent::Json(v) => {
                // Prefer native JSON input item for Responses API
                api_message["content"] = serde_json::Value::Array(vec![serde_json::json!({
                    "type": "input_json",
                    "json": v
                })]);
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
        struct ResponsesHooks;
        impl crate::transformers::request::ProviderRequestHooks for ResponsesHooks {
            fn build_base_chat_body(
                &self,
                req: &ChatRequest,
            ) -> Result<serde_json::Value, LlmError> {
                // Build base body
                let mut body = serde_json::json!({
                    "model": req.common_params.model,
                    "stream": req.stream,
                });

                // input
                let mut input_items = Vec::with_capacity(req.messages.len());
                for m in &req.messages {
                    input_items.push(OpenAiResponsesRequestTransformer::convert_message(m)?);
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

            fn post_process_chat(
                &self,
                req: &crate::types::ChatRequest,
                body: &mut serde_json::Value,
            ) -> Result<(), LlmError> {
                if let Some(pp) = &req.provider_params {
                    if let Some(so) = pp
                        .params
                        .get("structured_output")
                        .and_then(|v| v.as_object())
                    {
                        let mode = so.get("mode").and_then(|v| v.as_str()).unwrap_or("auto");
                        let schema = so.get("schema");
                        let name = so.get("schema_name").and_then(|v| v.as_str());
                        // Prefer schema if provided
                        if let Some(schema_v) = schema.cloned() {
                            let rf = if let Some(n) = name {
                                serde_json::json!({
                                    "type": "json_schema",
                                    "json_schema": {"name": n, "schema": schema_v, "strict": true}
                                })
                            } else {
                                serde_json::json!({
                                    "type": "json_object",
                                    "json_schema": {"schema": schema_v, "strict": true}
                                })
                            };
                            body["response_format"] = rf;
                        } else if mode == "json" {
                            body["response_format"] = serde_json::json!({"type":"json_object"});
                        }
                    }
                }
                // Safety: drop any leftover hint merged into body
                if let Some(obj) = body.as_object_mut() {
                    obj.remove("structured_output");
                }
                Ok(())
            }
        }
        let hooks = ResponsesHooks;
        let profile = crate::transformers::request::MappingProfile {
            provider_id: "openai_responses",
            rules: vec![
                crate::transformers::request::Rule::Range {
                    field: "temperature",
                    min: 0.0,
                    max: 2.0,
                    mode: crate::transformers::request::RangeMode::Error,
                    message: None,
                },
                crate::transformers::request::Rule::MergeProviderParams {
                    strategy: crate::transformers::request::ProviderParamsMergeStrategy::Flatten,
                },
            ],
            merge_strategy: crate::transformers::request::ProviderParamsMergeStrategy::Flatten,
        };
        let generic = crate::transformers::request::GenericRequestTransformer { profile, hooks };
        generic.transform_chat(req)
    }

    fn transform_rerank(&self, req: &RerankRequest) -> Result<serde_json::Value, LlmError> {
        let mut payload = serde_json::json!({
            "model": req.model,
            "query": req.query,
            "documents": req.documents,
        });
        if let Some(instr) = &req.instruction {
            payload["instruction"] = serde_json::json!(instr);
        }
        if let Some(top_n) = req.top_n {
            payload["top_n"] = serde_json::json!(top_n);
        }
        if let Some(rd) = req.return_documents {
            payload["return_documents"] = serde_json::json!(rd);
        }
        if let Some(maxc) = req.max_chunks_per_doc {
            payload["max_chunks_per_doc"] = serde_json::json!(maxc);
        }
        if let Some(over) = req.overlap_tokens {
            payload["overlap_tokens"] = serde_json::json!(over);
        }
        Ok(payload)
    }

    fn transform_moderation(&self, req: &ModerationRequest) -> Result<serde_json::Value, LlmError> {
        let model = req
            .model
            .clone()
            .unwrap_or_else(|| "text-moderation-latest".to_string());

        // OpenAI Moderations accepts either string or string[] for `input`.
        // Prefer array when provided in request.
        let input_value = if let Some(arr) = &req.inputs {
            serde_json::Value::Array(
                arr.iter()
                    .map(|s| serde_json::Value::String(s.clone()))
                    .collect(),
            )
        } else {
            serde_json::Value::String(req.input.clone())
        };

        let json = serde_json::json!({ "model": model, "input": input_value });
        Ok(json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "structured-messages")]
    #[test]
    fn convert_message_json_maps_to_input_json() {
        use crate::types::{ChatMessage, MessageContent, MessageMetadata, MessageRole};
        let msg = ChatMessage {
            role: MessageRole::User,
            content: MessageContent::Json(serde_json::json!({"a":1})),
            metadata: MessageMetadata::default(),
            tool_calls: None,
            tool_call_id: None,
        };
        let v = OpenAiResponsesRequestTransformer::convert_message(&msg).expect("convert");
        // Expect content array with input_json item
        let content = v
            .get("content")
            .and_then(|x| x.as_array())
            .cloned()
            .unwrap();
        assert!(!content.is_empty());
        let first = &content[0];
        assert_eq!(
            first.get("type").and_then(|t| t.as_str()).unwrap_or(""),
            "input_json"
        );
        assert_eq!(
            first
                .get("json")
                .unwrap()
                .get("a")
                .and_then(|x| x.as_i64())
                .unwrap_or(0),
            1
        );
    }

    #[cfg(feature = "structured-messages")]
    #[test]
    fn convert_tool_output_json_maps_to_output_json() {
        use crate::types::{ChatMessage, MessageContent, MessageMetadata, MessageRole};
        let mut msg = ChatMessage {
            role: MessageRole::Tool,
            content: MessageContent::Json(serde_json::json!({"r":42})),
            metadata: MessageMetadata::default(),
            tool_calls: None,
            tool_call_id: Some("call-1".into()),
        };
        let v = OpenAiResponsesRequestTransformer::convert_message(&msg).expect("convert");
        assert_eq!(
            v.get("type").and_then(|t| t.as_str()).unwrap_or(""),
            "function_call_output"
        );
        assert_eq!(
            v.get("call_id").and_then(|x| x.as_str()).unwrap_or(""),
            "call-1"
        );
        assert_eq!(
            v.get("output_json")
                .unwrap()
                .get("r")
                .and_then(|x| x.as_i64())
                .unwrap_or(0),
            42
        );

        // Fallback to output text for Text content
        msg.content = MessageContent::Text("ok".into());
        let v2 = OpenAiResponsesRequestTransformer::convert_message(&msg).expect("convert");
        assert_eq!(
            v2.get("output").and_then(|x| x.as_str()).unwrap_or(""),
            "ok"
        );
    }
}
