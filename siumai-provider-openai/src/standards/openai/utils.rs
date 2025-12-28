//! OpenAI-compatible protocol utilities
//!
//! This module contains wire-format conversion helpers that are shared across
//! multiple providers that implement OpenAI-style APIs.

use crate::error::LlmError;
use crate::types::*;
use base64::Engine;

use super::types::{OpenAiFunction, OpenAiMessage, OpenAiToolCall};

/// Convert tools to OpenAI Chat Completions format.
pub fn convert_tools_to_openai_format(
    tools: &[crate::types::Tool],
) -> Result<Vec<serde_json::Value>, LlmError> {
    let mut openai_tools = Vec::new();

    for tool in tools {
        match tool {
            crate::types::Tool::Function { function } => {
                openai_tools.push(serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": function.name,
                        "description": function.description,
                        "parameters": function.parameters
                    }
                }));
            }
            crate::types::Tool::ProviderDefined(provider_tool) => {
                if provider_tool.provider() == Some("openai") {
                    let tool_type = provider_tool.tool_type().unwrap_or("unknown");
                    let mut openai_tool = serde_json::json!({
                        "type": tool_type,
                    });

                    if let serde_json::Value::Object(args_map) = &provider_tool.args
                        && let serde_json::Value::Object(tool_map) = &mut openai_tool
                    {
                        for (k, v) in args_map {
                            tool_map.insert(k.clone(), v.clone());
                        }
                    }

                    openai_tools.push(openai_tool);
                }
            }
        }
    }

    Ok(openai_tools)
}

/// Convert tools to OpenAI Responses API format (flattened).
pub fn convert_tools_to_responses_format(
    tools: &[crate::types::Tool],
) -> Result<Vec<serde_json::Value>, LlmError> {
    let mut openai_tools = Vec::new();

    for tool in tools {
        match tool {
            crate::types::Tool::Function { function } => {
                openai_tools.push(serde_json::json!({
                    "type": "function",
                    "name": function.name,
                    "description": function.description,
                    "parameters": function.parameters
                }));
            }
            crate::types::Tool::ProviderDefined(provider_tool) => {
                if provider_tool.provider() != Some("openai") {
                    continue;
                }

                let raw = provider_tool.tool_type().unwrap_or("unknown");

                // Vercel alignment:
                // - provider tool args live in SDK-shaped camelCase (e.g., searchContextSize),
                //   while OpenAI Responses API expects snake_case fields in the tool object.
                // - Accept both shapes for backward compatibility.
                let args_obj = provider_tool.args.as_object();

                let mut openai_tool = serde_json::json!({
                    "type": raw,
                });

                let Some(args_obj) = args_obj else {
                    openai_tools.push(openai_tool);
                    continue;
                };

                match raw {
                    "web_search" | "web_search_preview" => {
                        if let Some(v) = args_obj
                            .get("searchContextSize")
                            .or_else(|| args_obj.get("search_context_size"))
                        {
                            openai_tool["search_context_size"] = v.clone();
                        }
                        if let Some(v) = args_obj
                            .get("userLocation")
                            .or_else(|| args_obj.get("user_location"))
                        {
                            openai_tool["user_location"] = v.clone();
                        }

                        if raw == "web_search" {
                            if let Some(v) = args_obj
                                .get("externalWebAccess")
                                .or_else(|| args_obj.get("external_web_access"))
                            {
                                openai_tool["external_web_access"] = v.clone();
                            }

                            // Map filters.allowedDomains -> filters.allowed_domains
                            if let Some(filters) = args_obj.get("filters") {
                                if let Some(obj) = filters.as_object() {
                                    if let Some(allowed) = obj
                                        .get("allowedDomains")
                                        .or_else(|| obj.get("allowed_domains"))
                                    {
                                        openai_tool["filters"] =
                                            serde_json::json!({ "allowed_domains": allowed });
                                    } else {
                                        // Best-effort passthrough
                                        openai_tool["filters"] = filters.clone();
                                    }
                                } else {
                                    openai_tool["filters"] = filters.clone();
                                }
                            }
                        }
                    }
                    "code_interpreter" => {
                        // Vercel alignment:
                        // - tool args use `container`:
                        //   - string container ID
                        //   - { fileIds: [...] } (SDK shape)
                        // - API expects either string or { type: "auto", file_ids: [...] }.
                        let container = args_obj.get("container");
                        match container {
                            None => {
                                openai_tool["container"] = serde_json::json!({ "type": "auto" });
                            }
                            Some(serde_json::Value::String(id)) => {
                                openai_tool["container"] = serde_json::json!(id);
                            }
                            Some(serde_json::Value::Object(map)) => {
                                let file_ids =
                                    map.get("fileIds").or_else(|| map.get("file_ids")).cloned();
                                let mut out = serde_json::Map::new();
                                out.insert("type".to_string(), serde_json::json!("auto"));
                                if let Some(ids) = file_ids {
                                    out.insert("file_ids".to_string(), ids);
                                }
                                openai_tool["container"] = serde_json::Value::Object(out);
                            }
                            Some(other) => {
                                // Best-effort passthrough
                                openai_tool["container"] = other.clone();
                            }
                        }
                    }
                    "image_generation" => {
                        // Vercel alignment: camelCase args → snake_case tool fields.
                        if let Some(v) = args_obj.get("background") {
                            openai_tool["background"] = v.clone();
                        }
                        if let Some(v) = args_obj
                            .get("inputFidelity")
                            .or_else(|| args_obj.get("input_fidelity"))
                        {
                            openai_tool["input_fidelity"] = v.clone();
                        }
                        if let Some(mask) = args_obj
                            .get("inputImageMask")
                            .or_else(|| args_obj.get("input_image_mask"))
                            && let Some(obj) = mask.as_object()
                        {
                            let mut out = serde_json::Map::new();
                            if let Some(v) = obj.get("fileId").or_else(|| obj.get("file_id")) {
                                out.insert("file_id".to_string(), v.clone());
                            }
                            if let Some(v) = obj.get("imageUrl").or_else(|| obj.get("image_url")) {
                                out.insert("image_url".to_string(), v.clone());
                            }
                            if !out.is_empty() {
                                openai_tool["input_image_mask"] = serde_json::Value::Object(out);
                            }
                        }
                        if let Some(v) = args_obj.get("model") {
                            openai_tool["model"] = v.clone();
                        }
                        if let Some(v) = args_obj.get("moderation") {
                            openai_tool["moderation"] = v.clone();
                        }
                        if let Some(v) = args_obj
                            .get("partialImages")
                            .or_else(|| args_obj.get("partial_images"))
                        {
                            openai_tool["partial_images"] = v.clone();
                        }
                        if let Some(v) = args_obj.get("quality") {
                            openai_tool["quality"] = v.clone();
                        }
                        if let Some(v) = args_obj
                            .get("outputCompression")
                            .or_else(|| args_obj.get("output_compression"))
                        {
                            openai_tool["output_compression"] = v.clone();
                        }
                        if let Some(v) = args_obj
                            .get("outputFormat")
                            .or_else(|| args_obj.get("output_format"))
                        {
                            openai_tool["output_format"] = v.clone();
                        }
                        if let Some(v) = args_obj.get("size") {
                            openai_tool["size"] = v.clone();
                        }
                    }
                    "computer_use" => {
                        // Vercel alignment:
                        // - Tool ID: "openai.computer_use"
                        // - OpenAI Responses API tool type: "computer_use_preview"
                        // - Accept both camelCase (SDK shape) and snake_case (compat) args.
                        openai_tool["type"] = serde_json::json!("computer_use_preview");

                        if let Some(v) = args_obj
                            .get("displayWidth")
                            .or_else(|| args_obj.get("display_width"))
                        {
                            openai_tool["display_width"] = v.clone();
                        }
                        if let Some(v) = args_obj
                            .get("displayHeight")
                            .or_else(|| args_obj.get("display_height"))
                        {
                            openai_tool["display_height"] = v.clone();
                        }
                        if let Some(v) = args_obj.get("environment") {
                            openai_tool["environment"] = v.clone();
                        }

                        if let Some(v) = args_obj
                            .get("displayScale")
                            .or_else(|| args_obj.get("display_scale"))
                        {
                            openai_tool["display_scale"] = v.clone();
                        }
                    }
                    "file_search" => {
                        if let Some(v) = args_obj
                            .get("vectorStoreIds")
                            .or_else(|| args_obj.get("vector_store_ids"))
                        {
                            openai_tool["vector_store_ids"] = v.clone();
                        }
                        if let Some(v) = args_obj
                            .get("maxNumResults")
                            .or_else(|| args_obj.get("max_num_results"))
                        {
                            openai_tool["max_num_results"] = v.clone();
                        }

                        if let Some(ranking) = args_obj
                            .get("ranking")
                            .or_else(|| args_obj.get("ranking_options"))
                            && let Some(obj) = ranking.as_object()
                        {
                            let ranker = obj.get("ranker").cloned();
                            let score_threshold = obj
                                .get("scoreThreshold")
                                .or_else(|| obj.get("score_threshold"))
                                .cloned();
                            let mut out = serde_json::Map::new();
                            if let Some(r) = ranker {
                                out.insert("ranker".to_string(), r);
                            }
                            if let Some(st) = score_threshold {
                                out.insert("score_threshold".to_string(), st);
                            }
                            if !out.is_empty() {
                                openai_tool["ranking_options"] = serde_json::Value::Object(out);
                            }
                        }

                        if let Some(filters) = args_obj.get("filters") {
                            openai_tool["filters"] = filters.clone();
                        }
                    }
                    _ => {
                        // Best-effort passthrough for unknown tools:
                        // merge args into the tool definition.
                        if let serde_json::Value::Object(tool_map) = &mut openai_tool {
                            for (k, v) in args_obj {
                                tool_map.insert(k.clone(), v.clone());
                            }
                        }
                    }
                }

                openai_tools.push(openai_tool);
            }
        }
    }

    Ok(openai_tools)
}

fn convert_message_content(content: &MessageContent) -> Result<serde_json::Value, LlmError> {
    match content {
        MessageContent::Text(text) => Ok(serde_json::Value::String(text.clone())),
        MessageContent::MultiModal(parts) => {
            let mut content_parts = Vec::new();

            for part in parts {
                match part {
                    ContentPart::Text { text } => {
                        content_parts.push(serde_json::json!({
                            "type": "text",
                            "text": text
                        }));
                    }
                    ContentPart::Image { source, detail } => {
                        let url = match source {
                            crate::types::chat::MediaSource::Url { url } => url.clone(),
                            crate::types::chat::MediaSource::Base64 { data } => {
                                if data.starts_with("data:") {
                                    data.clone()
                                } else {
                                    format!("data:image/jpeg;base64,{}", data)
                                }
                            }
                            crate::types::chat::MediaSource::Binary { data } => {
                                let encoded =
                                    base64::engine::general_purpose::STANDARD.encode(data);
                                format!("data:image/jpeg;base64,{}", encoded)
                            }
                        };

                        let mut image_obj = serde_json::json!({
                            "type": "image_url",
                            "image_url": {
                                "url": url
                            }
                        });

                        if let Some(detail) = detail {
                            image_obj["image_url"]["detail"] = serde_json::json!(detail);
                        }

                        content_parts.push(image_obj);
                    }
                    ContentPart::Audio { source, media_type } => match source {
                        crate::types::chat::MediaSource::Base64 { data } => {
                            let format = infer_audio_format(media_type.as_deref());
                            content_parts.push(serde_json::json!({
                                "type": "input_audio",
                                "input_audio": {
                                    "data": data,
                                    "format": format
                                }
                            }));
                        }
                        crate::types::chat::MediaSource::Binary { data } => {
                            let encoded = base64::engine::general_purpose::STANDARD.encode(data);
                            let format = infer_audio_format(media_type.as_deref());
                            content_parts.push(serde_json::json!({
                                "type": "input_audio",
                                "input_audio": {
                                    "data": encoded,
                                    "format": format
                                }
                            }));
                        }
                        crate::types::chat::MediaSource::Url { url } => {
                            content_parts.push(serde_json::json!({
                                "type": "text",
                                "text": format!("[Audio: {}]", url)
                            }));
                        }
                    },
                    ContentPart::File {
                        source, media_type, ..
                    } => {
                        if media_type == "application/pdf" {
                            let data = match source {
                                crate::types::chat::MediaSource::Base64 { data } => data.clone(),
                                crate::types::chat::MediaSource::Binary { data } => {
                                    base64::engine::general_purpose::STANDARD.encode(data)
                                }
                                crate::types::chat::MediaSource::Url { url } => {
                                    content_parts.push(serde_json::json!({
                                        "type": "text",
                                        "text": format!("[PDF: {}]", url)
                                    }));
                                    continue;
                                }
                            };
                            content_parts.push(serde_json::json!({
                                "type": "file",
                                "file": {
                                    "data": data,
                                    "media_type": media_type
                                }
                            }));
                        } else {
                            content_parts.push(serde_json::json!({
                                "type": "text",
                                "text": format!("[Unsupported file type: {}]", media_type)
                            }));
                        }
                    }
                    ContentPart::ToolCall { .. } => {}
                    ContentPart::ToolResult { .. } => {}
                    ContentPart::Reasoning { .. } => {}
                }
            }

            Ok(serde_json::Value::Array(content_parts))
        }
        #[cfg(feature = "structured-messages")]
        MessageContent::Json(v) => Ok(serde_json::Value::String(
            serde_json::to_string(v).unwrap_or_default(),
        )),
    }
}

/// Convert a message content value to the OpenAI(-compatible) wire format.
///
/// This is primarily used by provider-layer compatibility shims that still expose
/// historical helper functions (e.g. `providers::openai::utils::convert_message_content`).
pub fn convert_message_content_to_openai_value(
    content: &MessageContent,
) -> Result<serde_json::Value, LlmError> {
    convert_message_content(content)
}

/// Convert Siumai messages into OpenAI(-compatible) wire format.
pub fn convert_messages(messages: &[ChatMessage]) -> Result<Vec<OpenAiMessage>, LlmError> {
    let mut openai_messages = Vec::new();

    for message in messages {
        let openai_message = match message.role {
            MessageRole::System => OpenAiMessage {
                role: "system".to_string(),
                content: Some(convert_message_content(&message.content)?),
                tool_calls: None,
                tool_call_id: None,
            },
            MessageRole::User => OpenAiMessage {
                role: "user".to_string(),
                content: Some(convert_message_content(&message.content)?),
                tool_calls: None,
                tool_call_id: None,
            },
            MessageRole::Assistant => {
                let tool_calls_vec = message.tool_calls();
                let tool_calls_openai = if !tool_calls_vec.is_empty() {
                    Some(
                        tool_calls_vec
                            .iter()
                            .filter_map(|part| {
                                if let crate::types::ContentPart::ToolCall {
                                    tool_call_id,
                                    tool_name,
                                    arguments,
                                    ..
                                } = part
                                {
                                    Some(OpenAiToolCall {
                                        id: tool_call_id.clone(),
                                        r#type: "function".to_string(),
                                        function: Some(OpenAiFunction {
                                            name: tool_name.clone(),
                                            arguments: serde_json::to_string(arguments)
                                                .unwrap_or_default(),
                                        }),
                                    })
                                } else {
                                    None
                                }
                            })
                            .collect(),
                    )
                } else {
                    None
                };

                OpenAiMessage {
                    role: "assistant".to_string(),
                    content: Some(convert_message_content(&message.content)?),
                    tool_calls: tool_calls_openai,
                    tool_call_id: None,
                }
            }
            MessageRole::Tool => {
                let tool_results = message.tool_results();
                let tool_call_id = tool_results.first().and_then(|part| {
                    if let crate::types::ContentPart::ToolResult { tool_call_id, .. } = part {
                        Some(tool_call_id.clone())
                    } else {
                        None
                    }
                });

                OpenAiMessage {
                    role: "tool".to_string(),
                    content: Some(convert_message_content(&message.content)?),
                    tool_calls: None,
                    tool_call_id,
                }
            }
            MessageRole::Developer => OpenAiMessage {
                role: "developer".to_string(),
                content: Some(convert_message_content(&message.content)?),
                tool_calls: None,
                tool_call_id: None,
            },
        };

        openai_messages.push(openai_message);
    }

    Ok(openai_messages)
}

/// Convert Siumai tool choice to OpenAI wire format.
pub fn convert_tool_choice(choice: &crate::types::ToolChoice) -> serde_json::Value {
    match choice {
        ToolChoice::Auto => serde_json::json!("auto"),
        ToolChoice::Required => serde_json::json!("required"),
        ToolChoice::None => serde_json::json!("none"),
        ToolChoice::Tool { name } => {
            serde_json::json!({ "type": "function", "function": { "name": name } })
        }
    }
}

/// Parse OpenAI finish reason to unified FinishReason.
pub fn parse_finish_reason(reason: Option<&str>) -> Option<FinishReason> {
    match reason {
        Some("stop") => Some(FinishReason::Stop),
        Some("length") => Some(FinishReason::Length),
        Some("tool_calls") => Some(FinishReason::ToolCalls),
        Some("content_filter") => Some(FinishReason::ContentFilter),
        Some("function_call") => Some(FinishReason::ToolCalls),
        Some(other) => Some(FinishReason::Other(other.to_string())),
        None => None,
    }
}

/// Infer audio format from media type.
pub(crate) fn infer_audio_format(media_type: Option<&str>) -> &'static str {
    match media_type {
        Some("audio/wav") | Some("audio/wave") | Some("audio/x-wav") => "wav",
        Some("audio/mp3") | Some("audio/mpeg") => "mp3",
        _ => "wav",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn responses_tools_map_computer_use_to_preview_type() {
        let tool = crate::types::Tool::provider_defined("openai.computer_use", "computer_use")
            .with_args(serde_json::json!({
                "displayWidth": 1920,
                "displayHeight": 1080,
                "environment": "headless",
            }));

        let out = convert_tools_to_responses_format(&[tool]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["type"], serde_json::json!("computer_use_preview"));
        assert_eq!(out[0]["display_width"], serde_json::json!(1920));
        assert_eq!(out[0]["display_height"], serde_json::json!(1080));
        assert_eq!(out[0]["environment"], serde_json::json!("headless"));
    }

    #[test]
    fn responses_tools_map_code_interpreter_container_shape() {
        let tool =
            crate::types::Tool::provider_defined("openai.code_interpreter", "code_interpreter")
                .with_args(serde_json::json!({
                    "container": { "fileIds": ["file_1", "file_2"] }
                }));

        let out = convert_tools_to_responses_format(&[tool]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["type"], serde_json::json!("code_interpreter"));
        assert_eq!(out[0]["container"]["type"], serde_json::json!("auto"));
        assert_eq!(
            out[0]["container"]["file_ids"],
            serde_json::json!(["file_1", "file_2"])
        );
    }

    #[test]
    fn responses_tools_map_image_generation_keys() {
        let tool = crate::types::Tool::provider_defined(
            "openai.image_generation",
            "image_generation",
        )
        .with_args(serde_json::json!({
            "background": "transparent",
            "inputFidelity": "high",
            "inputImageMask": { "fileId": "file_mask", "imageUrl": "data:image/png;base64,..." },
            "model": "gpt-image-1",
            "outputFormat": "png",
            "outputCompression": 80,
            "partialImages": 2,
            "quality": "high",
            "size": "1024x1024",
        }));

        let out = convert_tools_to_responses_format(&[tool]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["type"], serde_json::json!("image_generation"));
        assert_eq!(out[0]["background"], serde_json::json!("transparent"));
        assert_eq!(out[0]["input_fidelity"], serde_json::json!("high"));
        assert_eq!(
            out[0]["input_image_mask"]["file_id"],
            serde_json::json!("file_mask")
        );
        assert_eq!(
            out[0]["input_image_mask"]["image_url"],
            serde_json::json!("data:image/png;base64,...")
        );
        assert_eq!(out[0]["model"], serde_json::json!("gpt-image-1"));
        assert_eq!(out[0]["output_format"], serde_json::json!("png"));
        assert_eq!(out[0]["output_compression"], serde_json::json!(80));
        assert_eq!(out[0]["partial_images"], serde_json::json!(2));
        assert_eq!(out[0]["quality"], serde_json::json!("high"));
        assert_eq!(out[0]["size"], serde_json::json!("1024x1024"));
    }
}
