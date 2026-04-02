//! OpenAI-compatible protocol utilities
//!
//! This module contains wire-format conversion helpers that are shared across
//! multiple providers that implement OpenAI-style APIs.

use super::types::OpenAiMessage;
use crate::error::LlmError;
use crate::types::*;

fn object_value_by_aliases<'a>(
    obj: &'a serde_json::Map<String, serde_json::Value>,
    aliases: &[&str],
) -> Option<&'a serde_json::Value> {
    aliases.iter().find_map(|alias| obj.get(*alias))
}

fn copy_object_value(
    target: &mut serde_json::Value,
    target_key: &str,
    obj: &serde_json::Map<String, serde_json::Value>,
    aliases: &[&str],
) {
    if let Some(value) = object_value_by_aliases(obj, aliases) {
        target[target_key] = value.clone();
    }
}

fn required_object_value(
    obj: &serde_json::Map<String, serde_json::Value>,
    aliases: &[&str],
    message: &str,
) -> Result<serde_json::Value, LlmError> {
    object_value_by_aliases(obj, aliases)
        .cloned()
        .ok_or_else(|| LlmError::InvalidInput(message.to_string()))
}

fn convert_xai_provider_tool_to_responses_format(
    provider_tool: &crate::types::ProviderDefinedTool,
) -> Result<Option<serde_json::Value>, LlmError> {
    let raw = provider_tool.tool_type().unwrap_or("unknown");
    let args_obj = provider_tool.args.as_object();

    let tool = match raw {
        "web_search" => {
            let mut tool = serde_json::json!({ "type": "web_search" });
            if let Some(args_obj) = args_obj {
                copy_object_value(
                    &mut tool,
                    "allowed_domains",
                    args_obj,
                    &["allowedDomains", "allowed_domains"],
                );
                copy_object_value(
                    &mut tool,
                    "excluded_domains",
                    args_obj,
                    &["excludedDomains", "excluded_domains"],
                );
                copy_object_value(
                    &mut tool,
                    "enable_image_understanding",
                    args_obj,
                    &["enableImageUnderstanding", "enable_image_understanding"],
                );
            }
            tool
        }
        "x_search" => {
            let mut tool = serde_json::json!({ "type": "x_search" });
            if let Some(args_obj) = args_obj {
                copy_object_value(
                    &mut tool,
                    "allowed_x_handles",
                    args_obj,
                    &["allowedXHandles", "allowed_x_handles"],
                );
                copy_object_value(
                    &mut tool,
                    "excluded_x_handles",
                    args_obj,
                    &["excludedXHandles", "excluded_x_handles"],
                );
                copy_object_value(&mut tool, "from_date", args_obj, &["fromDate", "from_date"]);
                copy_object_value(&mut tool, "to_date", args_obj, &["toDate", "to_date"]);
                copy_object_value(
                    &mut tool,
                    "enable_image_understanding",
                    args_obj,
                    &["enableImageUnderstanding", "enable_image_understanding"],
                );
                copy_object_value(
                    &mut tool,
                    "enable_video_understanding",
                    args_obj,
                    &["enableVideoUnderstanding", "enable_video_understanding"],
                );
            }
            tool
        }
        "code_execution" => serde_json::json!({ "type": "code_interpreter" }),
        "view_image" => serde_json::json!({ "type": "view_image" }),
        "view_x_video" => serde_json::json!({ "type": "view_x_video" }),
        "file_search" => {
            let Some(args_obj) = args_obj else {
                return Err(LlmError::InvalidInput(
                    "xAI file_search requires vectorStoreIds".to_string(),
                ));
            };

            let vector_store_ids = required_object_value(
                args_obj,
                &["vectorStoreIds", "vector_store_ids"],
                "xAI file_search requires vectorStoreIds",
            )?;
            let mut tool = serde_json::json!({
                "type": "file_search",
                "vector_store_ids": vector_store_ids,
            });
            copy_object_value(
                &mut tool,
                "max_num_results",
                args_obj,
                &["maxNumResults", "max_num_results"],
            );
            tool
        }
        "mcp" => {
            let Some(args_obj) = args_obj else {
                return Err(LlmError::InvalidInput(
                    "xAI mcp requires serverUrl".to_string(),
                ));
            };

            let server_url = required_object_value(
                args_obj,
                &["serverUrl", "server_url"],
                "xAI mcp requires serverUrl",
            )?;
            let mut tool = serde_json::json!({
                "type": "mcp",
                "server_url": server_url,
            });
            copy_object_value(
                &mut tool,
                "server_label",
                args_obj,
                &["serverLabel", "server_label"],
            );
            copy_object_value(
                &mut tool,
                "server_description",
                args_obj,
                &["serverDescription", "server_description"],
            );
            copy_object_value(
                &mut tool,
                "allowed_tools",
                args_obj,
                &["allowedTools", "allowed_tools"],
            );
            copy_object_value(&mut tool, "headers", args_obj, &["headers"]);
            copy_object_value(&mut tool, "authorization", args_obj, &["authorization"]);
            tool
        }
        _ => return Ok(None),
    };

    Ok(Some(tool))
}

/// Convert tools to OpenAI Chat Completions format.
pub fn convert_tools_to_openai_format(
    tools: &[crate::types::Tool],
) -> Result<Vec<serde_json::Value>, LlmError> {
    let mut openai_tools = Vec::new();

    for tool in tools {
        match tool {
            crate::types::Tool::Function { function } => {
                let mut tool = serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": function.name,
                        "description": function.description,
                        "parameters": function.parameters
                    }
                });

                if let Some(strict) = function.strict
                    && let serde_json::Value::Object(obj) = &mut tool["function"]
                {
                    obj.insert("strict".to_string(), serde_json::Value::Bool(strict));
                }

                openai_tools.push(tool);
            }
            crate::types::Tool::ProviderDefined(_) => {}
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
                let mut tool = serde_json::json!({
                    "type": "function",
                    "name": function.name,
                    "description": function.description,
                    "parameters": function.parameters
                });

                if let Some(strict) = function.strict {
                    tool["strict"] = serde_json::Value::Bool(strict);
                }

                openai_tools.push(tool);
            }
            crate::types::Tool::ProviderDefined(provider_tool) => {
                let provider = provider_tool.provider().unwrap_or("");
                if provider != "openai" && provider != "xai" {
                    continue;
                }

                if provider == "xai" {
                    if let Some(xai_tool) =
                        convert_xai_provider_tool_to_responses_format(provider_tool)?
                    {
                        openai_tools.push(xai_tool);
                    }
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

/// Convert a message content value to the OpenAI Chat Completions wire format.
///
/// This is aligned with Vercel `@ai-sdk/openai` behavior (PDF/audio file parts).
pub fn convert_message_content_to_openai_chat_value(
    content: &MessageContent,
) -> Result<serde_json::Value, LlmError> {
    siumai_core::standards::openai::utils::convert_message_content_to_openai_chat_value(content)
}

/// Convert Siumai messages into OpenAI(-compatible) wire format.
pub fn convert_messages(messages: &[ChatMessage]) -> Result<Vec<OpenAiMessage>, LlmError> {
    siumai_core::standards::openai::utils::convert_messages(messages)
}

/// Convert Siumai messages into OpenAI Chat Completions wire format.
///
/// This is aligned with Vercel `@ai-sdk/openai` behavior (PDF/audio file parts).
pub fn convert_messages_openai_chat(
    messages: &[ChatMessage],
) -> Result<Vec<OpenAiMessage>, LlmError> {
    siumai_core::standards::openai::utils::convert_messages_openai_chat(messages)
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

/// Convert Siumai response format into the OpenAI Chat Completions `response_format` wire shape.
///
/// Vercel AI SDK parity:
/// - `responseFormat: { type: "json", schema }` => `{ type: "json_schema", json_schema: { name, schema, strict } }`
pub fn convert_chat_completions_response_format(
    fmt: &crate::types::chat::ResponseFormat,
    strict_json_schema: bool,
) -> serde_json::Value {
    match fmt {
        crate::types::chat::ResponseFormat::Json {
            schema,
            name,
            description,
            strict,
        } => {
            let strict = strict.unwrap_or(strict_json_schema);
            let name = name.as_deref().unwrap_or("response");
            let mut out = serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "schema": schema,
                    "strict": strict,
                }
            });

            if let Some(desc) = description.as_deref()
                && !desc.trim().is_empty()
                && let Some(obj) = out.get_mut("json_schema").and_then(|v| v.as_object_mut())
            {
                obj.insert("description".to_string(), serde_json::json!(desc));
            }

            out
        }
    }
}

/// Convert Siumai response format into the OpenAI Responses `text.format` wire shape.
pub fn convert_responses_response_format(
    fmt: &crate::types::chat::ResponseFormat,
) -> serde_json::Value {
    match fmt {
        crate::types::chat::ResponseFormat::Json {
            schema,
            name,
            description,
            strict,
        } => {
            let mut out = serde_json::json!({
                "type": "json_schema",
                "schema": schema,
                "strict": strict.unwrap_or(true),
            });

            if let Some(name) = name.as_deref().filter(|value| !value.trim().is_empty()) {
                out["name"] = serde_json::json!(name);
            }

            if let Some(description) = description
                .as_deref()
                .filter(|value| !value.trim().is_empty())
            {
                out["description"] = serde_json::json!(description);
            }

            out
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
) -> Option<serde_json::Value> {
    match choice {
        ToolChoice::Auto => Some(serde_json::json!("auto")),
        ToolChoice::Required => Some(serde_json::json!("required")),
        ToolChoice::None => Some(serde_json::json!("none")),
        ToolChoice::Tool { name } => {
            if let Some(tools) = tools {
                for tool in tools.iter().rev() {
                    match tool {
                        crate::types::Tool::Function { function } if function.name == *name => {
                            return Some(serde_json::json!({
                                "type": "function",
                                "name": name,
                            }));
                        }
                        crate::types::Tool::ProviderDefined(provider_tool)
                            if provider_tool.name == *name =>
                        {
                            match provider_tool.provider() {
                                Some("openai") => {
                                    if let Some(tool_type) = provider_tool.tool_type()
                                        && let Some(t) =
                                            siumai_core::tools::openai::responses_builtin_type_for_tool_type(
                                                tool_type,
                                            )
                                    {
                                        return Some(serde_json::json!({ "type": t }));
                                    }
                                }
                                Some("xai") => {
                                    if matches!(
                                        provider_tool.tool_type(),
                                        Some(
                                            "web_search"
                                                | "x_search"
                                                | "code_execution"
                                                | "view_image"
                                                | "view_x_video"
                                                | "file_search"
                                                | "mcp"
                                        )
                                    ) {
                                        return None;
                                    }
                                }
                                _ => {}
                            }

                            return Some(serde_json::json!({
                                "type": "function",
                                "name": name,
                            }));
                        }
                        _ => {}
                    }
                }
            }

            if let Some(t) =
                siumai_core::tools::openai::responses_builtin_type_for_choice_name(name.as_str())
            {
                return Some(serde_json::json!({ "type": t }));
            }

            Some(serde_json::json!({ "type": "function", "name": name }))
        }
    }
}

/// Parse OpenAI finish reason to unified FinishReason.
pub fn parse_finish_reason(reason: Option<&str>) -> Option<FinishReason> {
    siumai_core::standards::openai::utils::parse_finish_reason(reason)
}

/// Parse OpenAI chat/responses/AI SDK usage payloads into unified `Usage`.
pub fn parse_openai_usage_value(value: &serde_json::Value) -> Option<Usage> {
    siumai_core::standards::openai::utils::parse_openai_usage_value(value)
}

/// Convert unified `Usage` into OpenAI Chat Completions usage JSON.
pub fn openai_chat_usage_value(usage: &Usage) -> serde_json::Value {
    siumai_core::standards::openai::utils::openai_chat_usage_value(usage)
}

/// Convert unified `Usage` into OpenAI Responses usage JSON.
pub fn openai_responses_usage_value(usage: &Usage) -> serde_json::Value {
    siumai_core::standards::openai::utils::openai_responses_usage_value(usage)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convert_tool_choice_matches_openai_chat_completions_wire_format() {
        use crate::types::ToolChoice;

        // auto
        let out = convert_tool_choice(&ToolChoice::Auto);
        assert_eq!(out, serde_json::json!("auto"));

        // required
        let out = convert_tool_choice(&ToolChoice::Required);
        assert_eq!(out, serde_json::json!("required"));

        // none
        let out = convert_tool_choice(&ToolChoice::None);
        assert_eq!(out, serde_json::json!("none"));

        // specific function tool
        let out = convert_tool_choice(&ToolChoice::tool("weather"));
        assert_eq!(
            out,
            serde_json::json!({
                "type": "function",
                "function": { "name": "weather" }
            })
        );
    }

    #[test]
    fn convert_response_format_maps_json_schema_like_vercel() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
            "required": ["value"],
            "additionalProperties": false
        });
        let fmt = crate::types::chat::ResponseFormat::json_schema(schema.clone())
            .with_name("mySchema")
            .with_description("desc")
            .with_strict(false);

        let out = convert_chat_completions_response_format(&fmt, true);
        assert_eq!(
            out,
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "mySchema",
                    "schema": schema,
                    "strict": false,
                    "description": "desc"
                }
            })
        );
    }

    #[test]
    fn responses_tools_map_computer_use_to_preview_type() {
        let tool = crate::tools::openai::computer_use().with_args(serde_json::json!({
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
        let tool = crate::tools::openai::code_interpreter().with_args(serde_json::json!({
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
        let tool = crate::tools::openai::image_generation().with_args(serde_json::json!({
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
    fn responses_tools_map_xai_server_tools_to_sdk_aligned_shapes() {
        let tools = vec![
            crate::tools::xai::web_search().with_args(serde_json::json!({
                "allowedDomains": ["wikipedia.org"],
                "enableImageUnderstanding": true,
            })),
            crate::tools::xai::x_search().with_args(serde_json::json!({
                "allowedXHandles": ["xai"],
                "fromDate": "2025-01-01",
                "enableVideoUnderstanding": true,
            })),
            crate::tools::xai::view_image(),
            crate::tools::xai::view_x_video(),
            crate::tools::xai::file_search(vec!["collection_1".to_string()]).with_args(
                serde_json::json!({
                    "vectorStoreIds": ["collection_1"],
                    "maxNumResults": 5,
                }),
            ),
            crate::tools::xai::mcp("https://example.com/mcp").with_args(serde_json::json!({
                "serverUrl": "https://example.com/mcp",
                "serverLabel": "docs",
                "allowedTools": ["search_docs"],
                "authorization": "Bearer token",
            })),
            crate::tools::xai::code_execution(),
        ];

        let out = convert_tools_to_responses_format(&tools).unwrap();
        assert_eq!(out.len(), 7);
        assert_eq!(out[0]["type"], serde_json::json!("web_search"));
        assert_eq!(
            out[0]["allowed_domains"],
            serde_json::json!(["wikipedia.org"])
        );
        assert_eq!(
            out[0]["enable_image_understanding"],
            serde_json::json!(true)
        );
        assert_eq!(out[1]["type"], serde_json::json!("x_search"));
        assert_eq!(out[1]["allowed_x_handles"], serde_json::json!(["xai"]));
        assert_eq!(out[1]["from_date"], serde_json::json!("2025-01-01"));
        assert_eq!(
            out[1]["enable_video_understanding"],
            serde_json::json!(true)
        );
        assert_eq!(out[2], serde_json::json!({ "type": "view_image" }));
        assert_eq!(out[3], serde_json::json!({ "type": "view_x_video" }));
        assert_eq!(out[4]["type"], serde_json::json!("file_search"));
        assert_eq!(
            out[4]["vector_store_ids"],
            serde_json::json!(["collection_1"])
        );
        assert_eq!(out[4]["max_num_results"], serde_json::json!(5));
        assert_eq!(out[5]["type"], serde_json::json!("mcp"));
        assert_eq!(
            out[5]["server_url"],
            serde_json::json!("https://example.com/mcp")
        );
        assert_eq!(out[5]["server_label"], serde_json::json!("docs"));
        assert_eq!(out[5]["allowed_tools"], serde_json::json!(["search_docs"]));
        assert_eq!(out[5]["authorization"], serde_json::json!("Bearer token"));
        assert_eq!(out[6], serde_json::json!({ "type": "code_interpreter" }));
    }

    #[test]
    fn responses_tool_choice_maps_builtins_by_name() {
        let choice = crate::types::ToolChoice::tool("web_search");
        let out = convert_responses_tool_choice(&choice, None);
        assert_eq!(out, Some(serde_json::json!({ "type": "web_search" })));
    }

    #[test]
    fn responses_tool_choice_maps_computer_use_alias_to_preview_type() {
        let choice = crate::types::ToolChoice::tool("computer_use");
        let out = convert_responses_tool_choice(&choice, None);
        assert_eq!(
            out,
            Some(serde_json::json!({ "type": "computer_use_preview" }))
        );
    }

    #[test]
    fn responses_tool_choice_maps_function_by_name() {
        let choice = crate::types::ToolChoice::tool("testFunction");
        let out = convert_responses_tool_choice(&choice, None);
        assert_eq!(
            out,
            Some(serde_json::json!({ "type": "function", "name": "testFunction" }))
        );
    }

    #[test]
    fn responses_tool_choice_resolves_custom_provider_tool_name() {
        let choice = crate::types::ToolChoice::tool("generateImage");
        let tools = vec![crate::tools::openai::image_generation_named(
            "generateImage",
        )];
        let out = convert_responses_tool_choice(&choice, Some(&tools));
        assert_eq!(out, Some(serde_json::json!({ "type": "image_generation" })));
    }

    #[test]
    fn responses_tool_choice_resolves_custom_provider_tool_name_for_computer_use() {
        let choice = crate::types::ToolChoice::tool("myComputer");
        let tools = vec![crate::tools::openai::computer_use_named("myComputer")];
        let out = convert_responses_tool_choice(&choice, Some(&tools));
        assert_eq!(
            out,
            Some(serde_json::json!({ "type": "computer_use_preview" }))
        );
    }

    #[test]
    fn responses_tool_choice_drops_xai_server_side_tools() {
        let choice = crate::types::ToolChoice::tool("web_search");
        let tools = vec![crate::tools::xai::web_search()];
        let out = convert_responses_tool_choice(&choice, Some(&tools));
        assert_eq!(out, None);
    }

    #[test]
    fn responses_tool_choice_keeps_xai_function_tool_names() {
        let choice = crate::types::ToolChoice::tool("weather");
        let tools = vec![crate::types::Tool::function(
            "weather",
            "weather lookup",
            serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        )];
        let out = convert_responses_tool_choice(&choice, Some(&tools));
        assert_eq!(
            out,
            Some(serde_json::json!({ "type": "function", "name": "weather" }))
        );
    }
}
