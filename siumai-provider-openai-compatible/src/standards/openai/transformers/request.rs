//! Request transformers for OpenAI-compatible protocol (Chat/Embedding/Images) and OpenAI Responses API

use crate::error::LlmError;
use crate::execution::transformers::request::{
    Condition, GenericRequestTransformer, MappingProfile, ProviderParamsMergeStrategy,
    ProviderRequestHooks, RangeMode, Rule,
};
use crate::execution::transformers::request::{ImageHttpBody, RequestTransformer};
use crate::types::{
    ChatRequest, EmbeddingRequest, ImageEditRequest, ImageGenerationRequest, ImageVariationRequest,
    ModerationRequest, RerankRequest,
};
#[cfg(feature = "openai-responses")]
use base64::Engine;
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
                use crate::standards::openai::utils::convert_messages;
                let mut body = serde_json::json!({ "model": req.common_params.model });
                if let Some(t) = req.common_params.temperature {
                    body["temperature"] = serde_json::json!(t);
                }
                if let Some(tp) = req.common_params.top_p {
                    body["top_p"] = serde_json::json!(tp);
                }
                // Prefer max_completion_tokens for o1/o3 models, fallback to max_tokens
                if let Some(max) = req.common_params.max_completion_tokens {
                    body["max_completion_tokens"] = serde_json::json!(max);
                } else if let Some(max) = req.common_params.max_tokens {
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
                    let openai_tools =
                        crate::standards::openai::utils::convert_tools_to_openai_format(tools)?;
                    if !openai_tools.is_empty() {
                        body["tools"] = serde_json::Value::Array(openai_tools);

                        // Add tool_choice if specified
                        if let Some(choice) = &req.tool_choice {
                            body["tool_choice"] =
                                crate::standards::openai::utils::convert_tool_choice(choice);
                        }
                    }
                }

                if req.stream {
                    body["stream"] = serde_json::Value::Bool(true);
                    body["stream_options"] = serde_json::json!({ "include_usage": true });
                }
                Ok(body)
            }

            fn post_process_chat(
                &self,
                _req: &ChatRequest,
                _body: &mut serde_json::Value,
            ) -> Result<(), LlmError> {
                // All provider-specific features are now handled via provider_options
                // in ProviderSpec::chat_before_send()
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
                // For o1-* models, prefer max_completion_tokens by moving max_tokens
                Rule::When {
                    condition: Condition::ModelPrefix("o1-"),
                    rules: vec![
                        Rule::Move {
                            from: "max_tokens",
                            to: "max_completion_tokens",
                        },
                        Rule::Drop {
                            field: "max_tokens",
                        },
                    ],
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
            rules: vec![],
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
                _req: &ImageGenerationRequest,
                _body: &mut serde_json::Value,
            ) -> Result<(), LlmError> {
                // Merging extra parameters is handled by GenericRequestTransformer via merge_strategy
                Ok(())
            }
        }

        let hooks = OpenAiImageHooks;
        let profile = MappingProfile {
            provider_id: "openai",
            rules: vec![],
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
mod tests_openai_rules {
    use super::*;

    #[test]
    fn when_model_prefix_o1_moves_max_tokens() {
        let tx = OpenAiRequestTransformer;
        let mut req = ChatRequest::new(vec![]);
        req.common_params.model = "o1-mini".to_string();
        req.common_params.max_tokens = Some(123);
        let body = tx.transform_chat(&req).expect("transform");
        // max_tokens should be moved to max_completion_tokens
        assert!(body.get("max_tokens").is_none());
        assert_eq!(body["max_completion_tokens"], serde_json::json!(123));
    }
}

// Tests for structured_output via provider_params have been removed
// as this functionality is now handled via provider_options in ProviderSpec::chat_before_send()

/// Request transformer for OpenAI Responses API
#[derive(Clone)]
#[cfg(feature = "openai-responses")]
pub struct OpenAiResponsesRequestTransformer;

#[cfg(feature = "openai-responses")]
impl OpenAiResponsesRequestTransformer {
    fn should_include_item_reference(req: &ChatRequest) -> bool {
        // Vercel alignment:
        // - `convertToOpenAIResponsesInput` takes `store` as a parameter.
        // - In Siumai, `store` lives in `providerOptions.openai` (Responses API config).
        // - Default to true when unspecified (matches Vercel fixtures expectations).
        let openai = req.provider_options_map.get_object("openai");

        let store = openai
            .and_then(|m| m.get("store"))
            .and_then(|v| v.as_bool())
            .or_else(|| {
                openai
                    .and_then(|m| m.get("responsesApi").or_else(|| m.get("responses_api")))
                    .and_then(|v| v.as_object())
                    .and_then(|m| m.get("store"))
                    .and_then(|v| v.as_bool())
            });

        store != Some(false)
    }

    fn convert_message(
        req: &ChatRequest,
        msg: &crate::types::ChatMessage,
    ) -> Result<Vec<serde_json::Value>, LlmError> {
        use crate::types::{ContentPart, MessageContent, MessageRole};

        // Tool role message becomes one or many `function_call_output` items (one per tool result).
        if matches!(msg.role, MessageRole::Tool) {
            let store = Self::should_include_item_reference(req);
            let mut items: Vec<serde_json::Value> = Vec::new();
            let mut processed_approval_ids: std::collections::HashSet<String> =
                std::collections::HashSet::new();

            if let MessageContent::MultiModal(parts) = &msg.content {
                for part in parts {
                    match part {
                        ContentPart::ToolApprovalResponse {
                            approval_id,
                            approved,
                        } => {
                            if !processed_approval_ids.insert(approval_id.clone()) {
                                continue;
                            }

                            if store {
                                items.push(serde_json::json!({
                                    "type": "item_reference",
                                    "id": approval_id,
                                }));
                            }

                            items.push(serde_json::json!({
                                "type": "mcp_approval_response",
                                "approval_request_id": approval_id,
                                "approve": approved,
                            }));
                        }
                        ContentPart::ToolResult {
                            tool_call_id,
                            output,
                            ..
                        } => {
                            // OpenAI Responses expects `output` (string or output content list). Keep it stable by
                            // sending a string form for all outputs.
                            let output_text = match output {
                                crate::types::ToolResultOutput::Text { value } => value.clone(),
                                crate::types::ToolResultOutput::Json { value } => {
                                    serde_json::to_string(value).unwrap_or_default()
                                }
                                crate::types::ToolResultOutput::ErrorText { value } => {
                                    value.clone()
                                }
                                crate::types::ToolResultOutput::ErrorJson { value } => {
                                    serde_json::to_string(value).unwrap_or_default()
                                }
                                crate::types::ToolResultOutput::ExecutionDenied { reason } => {
                                    reason
                                        .clone()
                                        .unwrap_or_else(|| "Execution denied".to_string())
                                }
                                crate::types::ToolResultOutput::Content { value } => value
                                    .iter()
                                    .filter_map(|part| {
                                        if let crate::types::ToolResultContentPart::Text { text } =
                                            part
                                        {
                                            Some(text.as_str())
                                        } else {
                                            None
                                        }
                                    })
                                    .collect::<Vec<_>>()
                                    .join("\n"),
                            };

                            items.push(serde_json::json!({
                                "type": "function_call_output",
                                "call_id": tool_call_id,
                                "output": output_text,
                            }));
                        }
                        _ => {}
                    }
                }
            }

            if items.is_empty() {
                return Err(LlmError::InvalidInput(
                    "Tool message missing tool result".into(),
                ));
            }

            return Ok(items);
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
                        ContentPart::Image { source, detail } => {
                            // Responses API prefers `input_image` items
                            let url = match source {
                                crate::types::chat::MediaSource::Url { url } => url.clone(),
                                crate::types::chat::MediaSource::Base64 { data } => {
                                    format!("data:image/jpeg;base64,{}", data)
                                }
                                crate::types::chat::MediaSource::Binary { data } => {
                                    let encoded =
                                        base64::engine::general_purpose::STANDARD.encode(data);
                                    format!("data:image/jpeg;base64,{}", encoded)
                                }
                            };

                            // OpenAI Responses `input_image`: `image_url` is a string and `detail` is top-level.
                            let image_part = serde_json::json!({
                                "type": "input_image",
                                "image_url": url,
                                "detail": detail.clone().unwrap_or(crate::types::ImageDetail::Auto),
                            });
                            content_parts.push(image_part);
                        }
                        ContentPart::Audio { source, media_type } => {
                            // Responses API input does not currently accept audio inside message content.
                            // Keep a stable fallback representation.
                            let hint = match source {
                                crate::types::chat::MediaSource::Url { url } => {
                                    format!("[Audio: {}]", url)
                                }
                                crate::types::chat::MediaSource::Base64 { .. }
                                | crate::types::chat::MediaSource::Binary { .. } => {
                                    let mt =
                                        media_type.clone().unwrap_or_else(|| "audio".to_string());
                                    format!("[Audio: {mt}]")
                                }
                            };
                            content_parts
                                .push(serde_json::json!({ "type": "input_text", "text": hint }));
                        }
                        ContentPart::File {
                            source,
                            media_type,
                            filename,
                            ..
                        } => {
                            // Responses API file support
                            if media_type == "application/pdf" {
                                let filename = filename
                                    .clone()
                                    .unwrap_or_else(|| "document.pdf".to_string());

                                match source {
                                    crate::types::chat::MediaSource::Url { url } => {
                                        content_parts.push(serde_json::json!({
                                            "type": "input_file",
                                            "filename": filename,
                                            "file_url": url,
                                        }));
                                    }
                                    crate::types::chat::MediaSource::Base64 { data } => {
                                        content_parts.push(serde_json::json!({
                                            "type": "input_file",
                                            "filename": filename,
                                            "file_data": data,
                                        }));
                                    }
                                    crate::types::chat::MediaSource::Binary { data } => {
                                        let encoded =
                                            base64::engine::general_purpose::STANDARD.encode(data);
                                        content_parts.push(serde_json::json!({
                                            "type": "input_file",
                                            "filename": filename,
                                            "file_data": encoded,
                                        }));
                                    }
                                }
                            } else {
                                content_parts.push(serde_json::json!({
                                    "type": "input_text",
                                    "text": format!("[Unsupported file type: {}]", media_type)
                                }));
                            }
                        }
                        ContentPart::ToolCall { .. } => {
                            if let ContentPart::ToolCall {
                                provider_executed, ..
                            } = part
                                && provider_executed == &Some(true)
                            {
                                continue;
                            }

                            // Assistant tool calls are represented as `tool_use` content parts with
                            // structured `input` (aligned with the official Responses API semantics).
                            if let ContentPart::ToolCall {
                                tool_call_id,
                                tool_name,
                                arguments,
                                ..
                            } = part
                            {
                                content_parts.push(serde_json::json!({
                                    "type": "tool_use",
                                    "id": tool_call_id,
                                    "name": tool_name,
                                    "input": arguments,
                                }));
                            }
                        }
                        ContentPart::ToolResult { .. } => {}
                        ContentPart::Reasoning { text } => {
                            // Reasoning content as text
                            content_parts.push(serde_json::json!({
                                "type": "input_text",
                                "text": format!("<thinking>{}</thinking>", text)
                            }));
                        }
                        ContentPart::ToolApprovalResponse { .. } => {}
                    }
                }

                // Vercel alignment: if a message only contained provider-executed tool calls
                // (or other skipped parts), omit the message entirely.
                if content_parts.is_empty() {
                    return Ok(vec![]);
                }

                api_message["content"] = serde_json::Value::Array(content_parts);
            }
            #[cfg(feature = "structured-messages")]
            MessageContent::Json(v) => {
                // Responses API does not define an `input_json` content part; serialize as text.
                api_message["content"] = serde_json::Value::Array(vec![serde_json::json!({
                    "type": "input_text",
                    "text": serde_json::to_string(v).unwrap_or_default()
                })]);
            }
        }

        Ok(vec![api_message])
    }
}

#[cfg(feature = "openai-responses")]
impl RequestTransformer for OpenAiResponsesRequestTransformer {
    fn provider_id(&self) -> &str {
        "openai_responses"
    }

    fn transform_chat(&self, req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
        struct ResponsesHooks;
        impl crate::execution::transformers::request::ProviderRequestHooks for ResponsesHooks {
            fn build_base_chat_body(
                &self,
                req: &ChatRequest,
            ) -> Result<serde_json::Value, LlmError> {
                // Build base body
                let mut body = serde_json::json!({
                    "model": req.common_params.model,
                });

                if req.stream {
                    body["stream"] = serde_json::Value::Bool(true);
                }

                // input
                let mut input_items: Vec<serde_json::Value> = Vec::new();
                for m in &req.messages {
                    input_items.extend(OpenAiResponsesRequestTransformer::convert_message(req, m)?);
                }
                body["input"] = serde_json::Value::Array(input_items);

                // tools (flattened)
                if let Some(tools) = &req.tools {
                    let openai_tools =
                        crate::standards::openai::utils::convert_tools_to_responses_format(tools)?;
                    if !openai_tools.is_empty() {
                        body["tools"] = serde_json::Value::Array(openai_tools);

                        // Add tool_choice if specified
                        if let Some(choice) = &req.tool_choice {
                            body["tool_choice"] =
                                crate::standards::openai::utils::convert_responses_tool_choice(
                                    choice,
                                    req.tools.as_deref(),
                                );
                        }
                    }
                }

                // stream options
                if req.stream {
                    body["stream_options"] = serde_json::json!({ "include_usage": true });
                }

                // temperature
                if let Some(temp) = req.common_params.temperature {
                    body["temperature"] = serde_json::json!(temp);
                }

                // top_p
                if let Some(tp) = req.common_params.top_p {
                    body["top_p"] = serde_json::json!(tp);
                }

                // max_output_tokens (prefer max_completion_tokens, fallback to max_tokens)
                if let Some(max_tokens) = req.common_params.max_completion_tokens {
                    body["max_output_tokens"] = serde_json::json!(max_tokens);
                } else if let Some(max_tokens) = req.common_params.max_tokens {
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
                _req: &crate::types::ChatRequest,
                _body: &mut serde_json::Value,
            ) -> Result<(), LlmError> {
                // All provider-specific features are now handled via provider_options
                // in ProviderSpec::chat_before_send()
                Ok(())
            }
        }
        let hooks = ResponsesHooks;
        let profile = crate::execution::transformers::request::MappingProfile {
            provider_id: "openai_responses",
            rules: vec![crate::execution::transformers::request::Rule::Range {
                field: "temperature",
                min: 0.0,
                max: 2.0,
                mode: crate::execution::transformers::request::RangeMode::Error,
                message: None,
            }],
            merge_strategy:
                crate::execution::transformers::request::ProviderParamsMergeStrategy::Flatten,
        };
        let generic =
            crate::execution::transformers::request::GenericRequestTransformer { profile, hooks };
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

    #[test]
    fn transform_chat_stream_includes_stream_options_include_usage() {
        use crate::types::ChatMessage;

        let tx = OpenAiRequestTransformer;
        let req = ChatRequest::builder()
            .message(ChatMessage::user("hi").build())
            .model("gpt-4o-mini")
            .stream(true)
            .build();

        let body = tx.transform_chat(&req).expect("transform chat");
        assert_eq!(body["stream"], true);
        assert_eq!(body["stream_options"]["include_usage"], true);
    }

    #[test]
    fn transform_rerank_includes_optional_fields() {
        use crate::types::RerankRequest;

        let tx = OpenAiRequestTransformer;
        let req = RerankRequest::new(
            "bge-reranker".to_string(),
            "rust async".to_string(),
            vec!["doc1".into(), "doc2".into()],
        )
        .with_instruction("rank by semantic relevance".to_string())
        .with_top_n(3)
        .with_return_documents(true)
        .with_max_chunks_per_doc(8)
        .with_overlap_tokens(16);

        let body = tx.transform_rerank(&req).expect("transform rerank");

        assert_eq!(body["model"], "bge-reranker");
        assert_eq!(body["query"], "rust async");
        let docs = body["documents"].as_array().expect("documents array");
        assert_eq!(docs.len(), 2);
        assert_eq!(body["top_n"], 3);
        assert_eq!(body["return_documents"], true);
        assert_eq!(body["max_chunks_per_doc"], 8);
        assert_eq!(body["overlap_tokens"], 16);
        // instruction is optional and may be adapted per provider; if present, assert matches
        assert_eq!(body["instruction"], "rank by semantic relevance");
    }

    #[cfg(feature = "structured-messages")]
    #[test]
    fn convert_message_json_maps_to_input_text_json_string() {
        use crate::types::{ChatMessage, MessageContent, MessageMetadata, MessageRole};
        let msg = ChatMessage {
            role: MessageRole::User,
            content: MessageContent::Json(serde_json::json!({"a":1})),
            metadata: MessageMetadata::default(),
        };
        let items =
            super::OpenAiResponsesRequestTransformer::convert_message(&msg).expect("convert");
        assert_eq!(items.len(), 1);
        let v = &items[0];
        // Expect content array with input_text JSON string
        let content = v
            .get("content")
            .and_then(|x| x.as_array())
            .cloned()
            .unwrap();
        assert!(!content.is_empty());
        let first = &content[0];
        assert_eq!(
            first.get("type").and_then(|t| t.as_str()).unwrap_or(""),
            "input_text"
        );
        let s = first.get("text").and_then(|t| t.as_str()).unwrap_or("");
        let parsed: serde_json::Value = serde_json::from_str(s).unwrap();
        assert_eq!(parsed.get("a").and_then(|x| x.as_i64()).unwrap_or(0), 1);
    }

    #[cfg(feature = "structured-messages")]
    #[test]
    fn convert_tool_output_json_maps_to_output_string() {
        use crate::types::{
            ChatMessage, ContentPart, MessageContent, MessageMetadata, MessageRole,
        };
        let msg = ChatMessage {
            role: MessageRole::Tool,
            content: MessageContent::MultiModal(vec![ContentPart::tool_result_json(
                "call-1",
                "test_tool",
                serde_json::json!({"r":42}),
            )]),
            metadata: MessageMetadata::default(),
        };
        let items =
            super::OpenAiResponsesRequestTransformer::convert_message(&msg).expect("convert");
        assert_eq!(items.len(), 1);
        let v = &items[0];
        assert_eq!(
            v.get("type").and_then(|t| t.as_str()).unwrap_or(""),
            "function_call_output"
        );
        assert_eq!(
            v.get("call_id").and_then(|x| x.as_str()).unwrap_or(""),
            "call-1"
        );
        let s = v.get("output").and_then(|o| o.as_str()).unwrap_or("");
        let parsed: serde_json::Value = serde_json::from_str(s).unwrap();
        assert_eq!(parsed.get("r").and_then(|x| x.as_i64()).unwrap_or(0), 42);

        // Fallback to output text for Text output
        let msg2 = ChatMessage {
            role: MessageRole::Tool,
            content: MessageContent::MultiModal(vec![ContentPart::tool_result_text(
                "call-2",
                "test_tool",
                "ok",
            )]),
            metadata: MessageMetadata::default(),
        };
        let items2 =
            super::OpenAiResponsesRequestTransformer::convert_message(&msg2).expect("convert");
        assert_eq!(items2.len(), 1);
        let v2 = &items2[0];
        assert_eq!(
            v2.get("output").and_then(|x| x.as_str()).unwrap_or(""),
            "ok"
        );
    }
}
