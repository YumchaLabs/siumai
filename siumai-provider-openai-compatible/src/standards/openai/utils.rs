//! OpenAI-compatible protocol utilities
//!
//! This module contains wire-format conversion helpers that are shared across
//! multiple providers that implement OpenAI-style APIs.

use super::types::OpenAiMessage;
use crate::error::LlmError;
use crate::types::*;

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
                    "mcp" => {
                        // Vercel alignment (OpenAI Responses MCP tool):
                        // - Tool args are SDK-shaped camelCase, Responses expects snake_case.
                        // - Default require_approval to "never" when omitted.
                        if let Some(v) = args_obj
                            .get("serverLabel")
                            .or_else(|| args_obj.get("server_label"))
                        {
                            openai_tool["server_label"] = v.clone();
                        }
                        if let Some(v) = args_obj
                            .get("serverUrl")
                            .or_else(|| args_obj.get("server_url"))
                        {
                            openai_tool["server_url"] = v.clone();
                        }
                        if let Some(v) = args_obj
                            .get("serverDescription")
                            .or_else(|| args_obj.get("server_description"))
                        {
                            openai_tool["server_description"] = v.clone();
                        }

                        if let Some(v) = args_obj
                            .get("requireApproval")
                            .or_else(|| args_obj.get("require_approval"))
                        {
                            openai_tool["require_approval"] = v.clone();
                        } else {
                            openai_tool["require_approval"] = serde_json::json!("never");
                        }

                        if let Some(v) = args_obj
                            .get("allowedTools")
                            .or_else(|| args_obj.get("allowed_tools"))
                        {
                            openai_tool["allowed_tools"] = v.clone();
                        }
                        if let Some(v) = args_obj.get("headers") {
                            openai_tool["headers"] = v.clone();
                        }
                    }
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

/// Convert a message content value to the OpenAI(-compatible) wire format.
///
/// This is primarily used by provider-layer compatibility shims that still expose
/// historical helper functions (e.g. `providers::openai::utils::convert_message_content`).
pub fn convert_message_content_to_openai_value(
    content: &MessageContent,
) -> Result<serde_json::Value, LlmError> {
    siumai_core::standards::openai::utils::convert_message_content_to_openai_value(content)
}

/// Convert Siumai messages into OpenAI(-compatible) wire format.
pub fn convert_messages(messages: &[ChatMessage]) -> Result<Vec<OpenAiMessage>, LlmError> {
    siumai_core::standards::openai::utils::convert_messages(messages)
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

/// Convert Siumai tool choice to OpenAI Responses API wire format.
///
/// Vercel AI SDK mapping (Responses API):
/// - `"auto"`, `"none"`, `"required"`
/// - `{ "type": "<builtin>" }` for provider-defined builtins (e.g. `web_search`)
/// - `{ "type": "function", "name": "<toolName>" }` for function tools
///
/// Note: This helper also supports custom names for provider-defined tools by
/// resolving the selected tool name against the provided tool list.
pub fn convert_responses_tool_choice(
    choice: &crate::types::ToolChoice,
    tools: Option<&[crate::types::Tool]>,
) -> serde_json::Value {
    match choice {
        ToolChoice::Auto => serde_json::json!("auto"),
        ToolChoice::Required => serde_json::json!("required"),
        ToolChoice::None => serde_json::json!("none"),
        ToolChoice::Tool { name } => {
            // Vercel parity: recognize built-in responses tools by name.
            if matches!(
                name.as_str(),
                "code_interpreter"
                    | "file_search"
                    | "image_generation"
                    | "web_search_preview"
                    | "web_search"
                    | "mcp"
                    | "apply_patch"
            ) {
                return serde_json::json!({ "type": name });
            }

            // Optional: resolve custom tool names (e.g. "generateImage") to the built-in type
            // by matching against the request tool list.
            if let Some(tools) = tools {
                for tool in tools {
                    let crate::types::Tool::ProviderDefined(provider_tool) = tool else {
                        continue;
                    };
                    if provider_tool.name != *name {
                        continue;
                    }

                    if provider_tool.provider() != Some("openai") {
                        continue;
                    }

                    match provider_tool.tool_type() {
                        Some(
                            "code_interpreter" | "file_search" | "image_generation"
                            | "web_search_preview" | "web_search" | "mcp" | "apply_patch",
                        ) => {
                            return serde_json::json!({ "type": provider_tool.tool_type().unwrap() });
                        }
                        _ => {}
                    }
                }
            }

            serde_json::json!({ "type": "function", "name": name })
        }
    }
}

/// Parse OpenAI finish reason to unified FinishReason.
pub fn parse_finish_reason(reason: Option<&str>) -> Option<FinishReason> {
    siumai_core::standards::openai::utils::parse_finish_reason(reason)
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

    #[test]
    fn responses_tool_choice_maps_builtins_by_name() {
        let choice = crate::types::ToolChoice::tool("web_search");
        let out = convert_responses_tool_choice(&choice, None);
        assert_eq!(out, serde_json::json!({ "type": "web_search" }));
    }

    #[test]
    fn responses_tool_choice_maps_function_by_name() {
        let choice = crate::types::ToolChoice::tool("testFunction");
        let out = convert_responses_tool_choice(&choice, None);
        assert_eq!(
            out,
            serde_json::json!({ "type": "function", "name": "testFunction" })
        );
    }

    #[test]
    fn responses_tool_choice_resolves_custom_provider_tool_name() {
        let choice = crate::types::ToolChoice::tool("generateImage");
        let tools = vec![crate::types::Tool::provider_defined(
            "openai.image_generation",
            "generateImage",
        )];
        let out = convert_responses_tool_choice(&choice, Some(&tools));
        assert_eq!(out, serde_json::json!({ "type": "image_generation" }));
    }
}
