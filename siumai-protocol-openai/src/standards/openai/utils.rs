//! OpenAI-compatible protocol utilities
//!
//! This module contains wire-format conversion helpers that are shared across
//! multiple providers that implement OpenAI-style APIs.

use std::collections::HashMap;

use base64::Engine;
use serde_json::{Map, Value};

use super::types::{OpenAiFunction, OpenAiMessage, OpenAiToolCall};
use crate::error::LlmError;
use crate::types::*;

mod message_dialect;

pub use message_dialect::{
    convert_message_content_to_openai_chat_value, convert_message_content_to_openai_value,
    convert_messages, convert_messages_deepseek_chat, convert_messages_mistral_chat,
    convert_messages_openai_chat, convert_messages_perplexity_chat, convert_messages_xai_chat,
};

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

fn map_mcp_tool_filter(value: &serde_json::Value, include_read_only: bool) -> serde_json::Value {
    let Some(obj) = value.as_object() else {
        return value.clone();
    };

    let mut out = serde_json::Map::new();
    if include_read_only && let Some(v) = obj.get("readOnly").or_else(|| obj.get("read_only")) {
        out.insert("read_only".to_string(), v.clone());
    }
    if let Some(v) = obj.get("toolNames").or_else(|| obj.get("tool_names")) {
        out.insert("tool_names".to_string(), v.clone());
    }

    if out.is_empty() {
        value.clone()
    } else {
        serde_json::Value::Object(out)
    }
}

fn map_mcp_require_approval(value: &serde_json::Value) -> serde_json::Value {
    let Some(obj) = value.as_object() else {
        return value.clone();
    };

    if let Some(never) = obj.get("never") {
        return serde_json::json!({
            "never": map_mcp_tool_filter(never, false),
        });
    }

    value.clone()
}

fn map_mcp_allowed_tools(value: &serde_json::Value) -> serde_json::Value {
    if value.is_array() {
        return value.clone();
    }

    map_mcp_tool_filter(value, true)
}

fn map_shell_skills(value: &serde_json::Value) -> Result<serde_json::Value, LlmError> {
    let Some(skills) = value.as_array() else {
        return Ok(value.clone());
    };

    let mut out = Vec::with_capacity(skills.len());
    for skill in skills {
        let Some(obj) = skill.as_object() else {
            out.push(skill.clone());
            continue;
        };

        match obj.get("type").and_then(|v| v.as_str()) {
            Some("skillReference") | Some("skill_reference") => {
                let provider_reference = obj
                    .get("providerReference")
                    .or_else(|| obj.get("provider_reference"))
                    .and_then(|v| v.as_object());
                let skill_id = provider_reference
                    .and_then(|reference| reference.get("openai"))
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        LlmError::InvalidInput(
                            "OpenAI shell skillReference requires providerReference.openai"
                                .to_string(),
                        )
                    })?;

                let mut mapped = serde_json::Map::new();
                mapped.insert("type".to_string(), serde_json::json!("skill_reference"));
                mapped.insert("skill_id".to_string(), serde_json::json!(skill_id));
                mapped.insert(
                    "version".to_string(),
                    obj.get("version")
                        .cloned()
                        .unwrap_or_else(|| serde_json::json!("latest")),
                );
                out.push(serde_json::Value::Object(mapped));
            }
            Some("inline") => {
                let mut mapped = serde_json::Map::new();
                mapped.insert("type".to_string(), serde_json::json!("inline"));

                if let Some(v) = obj.get("name") {
                    mapped.insert("name".to_string(), v.clone());
                }
                if let Some(v) = obj.get("description") {
                    mapped.insert("description".to_string(), v.clone());
                }
                if let Some(source) = obj.get("source") {
                    if let Some(source_obj) = source.as_object() {
                        let mut mapped_source = serde_json::Map::new();
                        if let Some(v) = source_obj.get("type") {
                            mapped_source.insert("type".to_string(), v.clone());
                        }
                        if let Some(v) = source_obj
                            .get("mediaType")
                            .or_else(|| source_obj.get("media_type"))
                        {
                            mapped_source.insert("media_type".to_string(), v.clone());
                        }
                        if let Some(v) = source_obj.get("data") {
                            mapped_source.insert("data".to_string(), v.clone());
                        }
                        mapped.insert(
                            "source".to_string(),
                            serde_json::Value::Object(mapped_source),
                        );
                    } else {
                        mapped.insert("source".to_string(), source.clone());
                    }
                }
                out.push(serde_json::Value::Object(mapped));
            }
            _ => out.push(skill.clone()),
        }
    }

    Ok(serde_json::Value::Array(out))
}

fn map_shell_network_policy(value: &serde_json::Value) -> serde_json::Value {
    let Some(obj) = value.as_object() else {
        return value.clone();
    };

    match obj.get("type").and_then(|v| v.as_str()) {
        Some("disabled") => serde_json::json!({ "type": "disabled" }),
        Some("allowlist") => {
            let mut out = serde_json::Map::new();
            out.insert("type".to_string(), serde_json::json!("allowlist"));
            if let Some(v) = obj
                .get("allowedDomains")
                .or_else(|| obj.get("allowed_domains"))
            {
                out.insert("allowed_domains".to_string(), v.clone());
            }
            if let Some(v) = obj
                .get("domainSecrets")
                .or_else(|| obj.get("domain_secrets"))
            {
                out.insert("domain_secrets".to_string(), v.clone());
            }
            serde_json::Value::Object(out)
        }
        _ => value.clone(),
    }
}

fn map_shell_environment(value: &serde_json::Value) -> Result<serde_json::Value, LlmError> {
    let Some(obj) = value.as_object() else {
        return Ok(value.clone());
    };

    match obj.get("type").and_then(|v| v.as_str()) {
        Some("containerReference") | Some("container_reference") => {
            let mut out = serde_json::Map::new();
            out.insert("type".to_string(), serde_json::json!("container_reference"));
            if let Some(v) = obj.get("containerId").or_else(|| obj.get("container_id")) {
                out.insert("container_id".to_string(), v.clone());
            }
            Ok(serde_json::Value::Object(out))
        }
        Some("containerAuto") | Some("container_auto") => {
            let mut out = serde_json::Map::new();
            out.insert("type".to_string(), serde_json::json!("container_auto"));
            if let Some(v) = obj.get("fileIds").or_else(|| obj.get("file_ids")) {
                out.insert("file_ids".to_string(), v.clone());
            }
            if let Some(v) = obj.get("memoryLimit").or_else(|| obj.get("memory_limit")) {
                out.insert("memory_limit".to_string(), v.clone());
            }
            if let Some(v) = obj
                .get("networkPolicy")
                .or_else(|| obj.get("network_policy"))
            {
                out.insert("network_policy".to_string(), map_shell_network_policy(v));
            }
            if let Some(v) = obj.get("skills") {
                out.insert("skills".to_string(), map_shell_skills(v)?);
            }
            Ok(serde_json::Value::Object(out))
        }
        _ => {
            let mut out = serde_json::Map::new();
            out.insert("type".to_string(), serde_json::json!("local"));
            if let Some(v) = obj.get("skills") {
                out.insert("skills".to_string(), v.clone());
            }
            Ok(serde_json::Value::Object(out))
        }
    }
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

                if let Some(defer_loading) = function
                    .provider_options_map
                    .get_object("openai")
                    .and_then(|openai| {
                        openai
                            .get("deferLoading")
                            .or_else(|| openai.get("defer_loading"))
                    })
                {
                    tool["defer_loading"] = defer_loading.clone();
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
                            openai_tool["require_approval"] = map_mcp_require_approval(v);
                        } else {
                            openai_tool["require_approval"] = serde_json::json!("never");
                        }

                        if let Some(v) = args_obj
                            .get("allowedTools")
                            .or_else(|| args_obj.get("allowed_tools"))
                        {
                            openai_tool["allowed_tools"] = map_mcp_allowed_tools(v);
                        }
                        if let Some(v) = args_obj.get("authorization") {
                            openai_tool["authorization"] = v.clone();
                        }
                        if let Some(v) = args_obj
                            .get("connectorId")
                            .or_else(|| args_obj.get("connector_id"))
                        {
                            openai_tool["connector_id"] = v.clone();
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
                    "local_shell" | "apply_patch" => {
                        // These hosted tools do not forward args in the AI SDK shaper.
                    }
                    "shell" => {
                        if let Some(environment) = args_obj.get("environment") {
                            openai_tool["environment"] = map_shell_environment(environment)?;
                        }
                    }
                    "custom" => {
                        openai_tool["name"] = serde_json::json!(&provider_tool.name);
                        if let Some(v) = args_obj.get("description") {
                            openai_tool["description"] = v.clone();
                        }
                        if let Some(v) = args_obj.get("format") {
                            openai_tool["format"] = v.clone();
                        }
                    }
                    "tool_search" => {
                        if let Some(v) = args_obj.get("execution") {
                            openai_tool["execution"] = v.clone();
                        }
                        if let Some(v) = args_obj.get("description") {
                            openai_tool["description"] = v.clone();
                        }
                        if let Some(v) = args_obj.get("parameters") {
                            openai_tool["parameters"] = v.clone();
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

fn openai_compatible_options_object(
    provider_options: Option<&ProviderOptionsMap>,
) -> Option<&serde_json::Map<String, serde_json::Value>> {
    provider_options
        .and_then(|provider_options| provider_options.get("openaiCompatible"))
        .and_then(|value| value.as_object())
}

fn merge_openai_compatible_extra(
    extra: &mut HashMap<String, serde_json::Value>,
    provider_options: Option<&ProviderOptionsMap>,
) {
    let Some(obj) = openai_compatible_options_object(provider_options) else {
        return;
    };

    for (k, v) in obj {
        extra.insert(k.clone(), v.clone());
    }
}

fn merge_openai_compatible_json(
    obj: &mut serde_json::Map<String, serde_json::Value>,
    provider_options: Option<&ProviderOptionsMap>,
) {
    let Some(extra) = openai_compatible_options_object(provider_options) else {
        return;
    };

    for (k, v) in extra {
        obj.insert(k.clone(), v.clone());
    }
}

/// Parse OpenAI(-compatible) finish reason to unified `FinishReason`.
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

/// Parse provider-specific OpenAI-compatible finish reasons to unified `FinishReason`.
pub fn parse_provider_openai_finish_reason(
    provider_id: &str,
    reason: Option<&str>,
) -> Option<FinishReason> {
    let reason = reason?;

    Some(match provider_id {
        "deepseek" => match reason {
            "stop" => FinishReason::Stop,
            "length" => FinishReason::Length,
            "content_filter" => FinishReason::ContentFilter,
            "tool_calls" => FinishReason::ToolCalls,
            "insufficient_system_resource" => FinishReason::Error,
            other => FinishReason::Other(other.to_string()),
        },
        "mistral" => match reason {
            "stop" => FinishReason::Stop,
            "length" | "model_length" => FinishReason::Length,
            "tool_calls" => FinishReason::ToolCalls,
            other => FinishReason::Other(other.to_string()),
        },
        "perplexity" => match reason {
            "stop" => FinishReason::Stop,
            "length" => FinishReason::Length,
            other => FinishReason::Other(other.to_string()),
        },
        "cohere" => match reason {
            "COMPLETE" | "STOP_SEQUENCE" => FinishReason::Stop,
            "MAX_TOKENS" => FinishReason::Length,
            "ERROR" => FinishReason::Error,
            "TOOL_CALL" => FinishReason::ToolCalls,
            other => FinishReason::Other(other.to_string()),
        },
        _ => parse_finish_reason(Some(reason))?,
    })
}

pub(crate) fn usage_u32(value: Option<&Value>) -> Option<u32> {
    value
        .and_then(Value::as_u64)
        .map(|value| value.min(u32::MAX as u64) as u32)
}

pub(crate) fn usage_value<'a>(object: &'a Map<String, Value>, keys: &[&str]) -> Option<&'a Value> {
    keys.iter().find_map(|key| object.get(*key))
}

pub(crate) fn usage_object<'a>(
    object: &'a Map<String, Value>,
    keys: &[&str],
) -> Option<&'a Map<String, Value>> {
    usage_value(object, keys).and_then(Value::as_object)
}

pub(crate) fn parse_input_tokens_value(
    tokens: Option<&Value>,
) -> (Option<u32>, Option<u32>, Option<u32>, Option<u32>) {
    let Some(tokens) = tokens else {
        return (None, None, None, None);
    };

    if let Some(total) = usage_u32(Some(tokens)) {
        return (Some(total), Some(total), None, None);
    }

    let Some(object) = tokens.as_object() else {
        return (None, None, None, None);
    };

    let no_cache = usage_u32(usage_value(object, &["noCache", "no_cache"]));
    let cache_read = usage_u32(usage_value(object, &["cacheRead", "cache_read"]));
    let cache_write = usage_u32(usage_value(object, &["cacheWrite", "cache_write"]));
    let total = usage_u32(usage_value(object, &["total", "totalTokens"])).or_else(|| {
        if no_cache.is_none() && cache_read.is_none() && cache_write.is_none() {
            None
        } else {
            Some(
                no_cache
                    .unwrap_or(0)
                    .saturating_add(cache_read.unwrap_or(0))
                    .saturating_add(cache_write.unwrap_or(0)),
            )
        }
    });

    let no_cache = no_cache.or_else(|| {
        total.map(|total| {
            total
                .saturating_sub(cache_read.unwrap_or(0))
                .saturating_sub(cache_write.unwrap_or(0))
        })
    });

    (total, no_cache, cache_read, cache_write)
}

pub(crate) fn parse_output_tokens_value(
    tokens: Option<&Value>,
) -> (Option<u32>, Option<u32>, Option<u32>) {
    let Some(tokens) = tokens else {
        return (None, None, None);
    };

    if let Some(total) = usage_u32(Some(tokens)) {
        return (Some(total), Some(total), None);
    }

    let Some(object) = tokens.as_object() else {
        return (None, None, None);
    };

    let reasoning = usage_u32(usage_value(object, &["reasoning", "reasoningTokens"]));
    let text = usage_u32(usage_value(object, &["text", "textTokens"]));
    let total = usage_u32(usage_value(object, &["total", "totalTokens"])).or_else(|| {
        if text.is_none() && reasoning.is_none() {
            None
        } else {
            Some(text.unwrap_or(0).saturating_add(reasoning.unwrap_or(0)))
        }
    });
    let text = text.or_else(|| total.map(|total| total.saturating_sub(reasoning.unwrap_or(0))));

    (total, text, reasoning)
}

fn stripped_raw_usage_object(usage: &Usage, removed_keys: &[&str]) -> Map<String, Value> {
    let mut object = usage.raw.clone().unwrap_or_default();
    for key in removed_keys {
        object.remove(*key);
    }
    object
}

pub(crate) fn ensure_object_entry<'a>(
    object: &'a mut Map<String, Value>,
    key: &str,
) -> &'a mut Map<String, Value> {
    if !object.get(key).is_some_and(Value::is_object) {
        object.insert(key.to_string(), Value::Object(Map::new()));
    }

    object
        .get_mut(key)
        .and_then(Value::as_object_mut)
        .expect("usage detail entry must be an object")
}

pub(crate) fn set_usage_detail_number(
    object: &mut Map<String, Value>,
    detail_key: &str,
    token_key: &str,
    value: u32,
) {
    ensure_object_entry(object, detail_key).insert(token_key.to_string(), serde_json::json!(value));
}

/// Parse OpenAI chat/responses/AI SDK usage payloads into the unified `Usage` shape.
pub fn parse_openai_usage_value(value: &Value) -> Option<Usage> {
    let object = value.as_object()?;

    let (input_total, input_no_cache, input_cache_read, input_cache_write) =
        parse_input_tokens_value(object.get("inputTokens"));
    let (output_total, output_text, output_reasoning) =
        parse_output_tokens_value(object.get("outputTokens"));

    let prompt_tokens = usage_u32(usage_value(object, &["prompt_tokens", "input_tokens"]))
        .or(input_total)
        .or_else(|| {
            if input_no_cache.is_none() && input_cache_read.is_none() && input_cache_write.is_none()
            {
                None
            } else {
                Some(
                    input_no_cache
                        .unwrap_or(0)
                        .saturating_add(input_cache_read.unwrap_or(0))
                        .saturating_add(input_cache_write.unwrap_or(0)),
                )
            }
        });

    let completion_tokens = usage_u32(usage_value(object, &["completion_tokens", "output_tokens"]))
        .or(output_total)
        .or_else(|| {
            if output_text.is_none() && output_reasoning.is_none() {
                None
            } else {
                Some(
                    output_text
                        .unwrap_or(0)
                        .saturating_add(output_reasoning.unwrap_or(0)),
                )
            }
        });

    let total_tokens = usage_u32(usage_value(object, &["total_tokens", "totalTokens"]));

    let prompt_details = usage_object(object, &["prompt_tokens_details", "input_tokens_details"]);
    let completion_details = usage_object(
        object,
        &["completion_tokens_details", "output_tokens_details"],
    );

    let cached_tokens = usage_u32(
        prompt_details.and_then(|details| usage_value(details, &["cached_tokens", "cachedTokens"])),
    )
    .or(input_cache_read);
    let reasoning_tokens = usage_u32(usage_value(
        object,
        &["reasoning_tokens", "reasoningTokens"],
    ))
    .or_else(|| {
        usage_u32(
            completion_details
                .and_then(|details| usage_value(details, &["reasoning_tokens", "reasoningTokens"])),
        )
    })
    .or(output_reasoning);

    let prompt_audio_tokens = usage_u32(
        prompt_details.and_then(|details| usage_value(details, &["audio_tokens", "audioTokens"])),
    );
    let completion_audio_tokens = usage_u32(
        completion_details
            .and_then(|details| usage_value(details, &["audio_tokens", "audioTokens"])),
    );
    let accepted_prediction_tokens = usage_u32(completion_details.and_then(|details| {
        usage_value(
            details,
            &["accepted_prediction_tokens", "acceptedPredictionTokens"],
        )
    }));
    let rejected_prediction_tokens = usage_u32(completion_details.and_then(|details| {
        usage_value(
            details,
            &["rejected_prediction_tokens", "rejectedPredictionTokens"],
        )
    }));

    let mut builder = Usage::builder();

    if let Some(prompt_tokens) = prompt_tokens {
        builder = builder.prompt_tokens(prompt_tokens);
    }
    if let Some(completion_tokens) = completion_tokens {
        builder = builder.completion_tokens(completion_tokens);
    }
    if let Some(total_tokens) = total_tokens {
        builder = builder.total_tokens(total_tokens);
    }
    if let Some(cached_tokens) = cached_tokens {
        builder = builder.with_cached_tokens(cached_tokens);
    }
    if let Some(reasoning_tokens) = reasoning_tokens {
        builder = builder.with_reasoning_tokens(reasoning_tokens);
    }
    if let Some(prompt_audio_tokens) = prompt_audio_tokens {
        builder = builder.with_prompt_audio_tokens(prompt_audio_tokens);
    }
    if let Some(completion_audio_tokens) = completion_audio_tokens {
        builder = builder.with_completion_audio_tokens(completion_audio_tokens);
    }
    if let Some(accepted_prediction_tokens) = accepted_prediction_tokens {
        builder = builder.with_accepted_prediction_tokens(accepted_prediction_tokens);
    }
    if let Some(rejected_prediction_tokens) = rejected_prediction_tokens {
        builder = builder.with_rejected_prediction_tokens(rejected_prediction_tokens);
    }
    if let Some(input_total) = input_total {
        builder = builder.with_input_total_tokens(input_total);
    }
    if let Some(input_no_cache) = input_no_cache {
        builder = builder.with_input_no_cache_tokens(input_no_cache);
    }
    if let Some(input_cache_read) = input_cache_read {
        builder = builder.with_input_cache_read_tokens(input_cache_read);
    }
    if let Some(input_cache_write) = input_cache_write {
        builder = builder.with_input_cache_write_tokens(input_cache_write);
    }
    if let Some(output_total) = output_total {
        builder = builder.with_output_total_tokens(output_total);
    }
    if let Some(output_text) = output_text {
        builder = builder.with_output_text_tokens(output_text);
    }
    if let Some(output_reasoning) = output_reasoning {
        builder = builder.with_output_reasoning_tokens(output_reasoning);
    }

    let raw_usage = object
        .get("raw")
        .and_then(Value::as_object)
        .cloned()
        .or_else(|| {
            if object.contains_key("raw") {
                None
            } else {
                Some(object.clone())
            }
        });
    if let Some(raw_usage) = raw_usage {
        builder = builder.with_raw_usage(raw_usage);
    }

    Some(builder.build())
}

/// Convert unified `Usage` into OpenAI Chat Completions usage JSON.
pub fn openai_chat_usage_value(usage: &Usage) -> Value {
    let normalized_input = usage.normalized_input_tokens();
    let normalized_output = usage.normalized_output_tokens();
    let prompt_tokens = normalized_input
        .total
        .or_else(|| usage.prompt_tokens_value());
    let completion_tokens = normalized_output
        .total
        .or_else(|| usage.completion_tokens_value());
    let total_tokens = usage.total_tokens_value().or_else(|| {
        prompt_tokens
            .zip(completion_tokens)
            .map(|(prompt, completion)| prompt.saturating_add(completion))
    });
    let mut object = stripped_raw_usage_object(
        usage,
        &[
            "input_tokens",
            "input_tokens_details",
            "output_tokens",
            "output_tokens_details",
            "inputTokens",
            "outputTokens",
            "raw",
        ],
    );

    object.insert(
        "prompt_tokens".to_string(),
        prompt_tokens
            .map(serde_json::Value::from)
            .unwrap_or(serde_json::Value::Null),
    );
    object.insert(
        "completion_tokens".to_string(),
        completion_tokens
            .map(serde_json::Value::from)
            .unwrap_or(serde_json::Value::Null),
    );
    object.insert(
        "total_tokens".to_string(),
        total_tokens
            .map(serde_json::Value::from)
            .unwrap_or(serde_json::Value::Null),
    );

    let raw_prompt_details_present = object
        .get("prompt_tokens_details")
        .and_then(Value::as_object)
        .is_some();
    let explicit_cached_tokens = usage
        .prompt_tokens_details
        .as_ref()
        .and_then(|details| details.cached_tokens);
    let chat_cached_tokens = if raw_prompt_details_present
        || explicit_cached_tokens.is_some()
        || normalized_input
            .cache_read
            .is_some_and(|tokens| tokens != 0)
    {
        normalized_input.cache_read
    } else {
        None
    };

    if chat_cached_tokens.is_some()
        || usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|details| details.audio_tokens)
            .is_some()
        || raw_prompt_details_present
    {
        let details = ensure_object_entry(&mut object, "prompt_tokens_details");
        if let Some(cached_tokens) = chat_cached_tokens {
            details.insert(
                "cached_tokens".to_string(),
                serde_json::json!(cached_tokens),
            );
        }
        if let Some(audio_tokens) = usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|details| details.audio_tokens)
        {
            details.insert("audio_tokens".to_string(), serde_json::json!(audio_tokens));
        }
        if details.is_empty() {
            object.remove("prompt_tokens_details");
        }
    }

    let raw_completion_details_present = object
        .get("completion_tokens_details")
        .and_then(Value::as_object)
        .is_some();
    let explicit_reasoning_tokens = usage
        .completion_tokens_details
        .as_ref()
        .and_then(|details| details.reasoning_tokens);
    let chat_reasoning_tokens = if raw_completion_details_present
        || explicit_reasoning_tokens.is_some()
        || normalized_output
            .reasoning
            .is_some_and(|tokens| tokens != 0)
    {
        normalized_output.reasoning
    } else {
        None
    };

    if chat_reasoning_tokens.is_some()
        || usage
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.audio_tokens)
            .is_some()
        || usage
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.accepted_prediction_tokens)
            .is_some()
        || usage
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.rejected_prediction_tokens)
            .is_some()
        || raw_completion_details_present
    {
        let details = ensure_object_entry(&mut object, "completion_tokens_details");
        if let Some(reasoning_tokens) = chat_reasoning_tokens {
            details.insert(
                "reasoning_tokens".to_string(),
                serde_json::json!(reasoning_tokens),
            );
        }
        if let Some(audio_tokens) = usage
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.audio_tokens)
        {
            details.insert("audio_tokens".to_string(), serde_json::json!(audio_tokens));
        }
        if let Some(accepted_prediction_tokens) = usage
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.accepted_prediction_tokens)
        {
            details.insert(
                "accepted_prediction_tokens".to_string(),
                serde_json::json!(accepted_prediction_tokens),
            );
        }
        if let Some(rejected_prediction_tokens) = usage
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.rejected_prediction_tokens)
        {
            details.insert(
                "rejected_prediction_tokens".to_string(),
                serde_json::json!(rejected_prediction_tokens),
            );
        }
        if details.is_empty() {
            object.remove("completion_tokens_details");
        }
    }

    Value::Object(object)
}

/// Convert unified `Usage` into OpenAI Responses usage JSON.
pub fn openai_responses_usage_value(usage: &Usage) -> Value {
    let normalized_input = usage.normalized_input_tokens();
    let normalized_output = usage.normalized_output_tokens();
    let input_total = normalized_input
        .total
        .or_else(|| usage.prompt_tokens_value());
    let output_total = normalized_output
        .total
        .or_else(|| usage.completion_tokens_value());
    let total_tokens = usage.total_tokens_value().or_else(|| {
        input_total
            .zip(output_total)
            .map(|(input, output)| input.saturating_add(output))
    });
    let mut object = stripped_raw_usage_object(
        usage,
        &[
            "prompt_tokens",
            "prompt_tokens_details",
            "completion_tokens",
            "completion_tokens_details",
            "reasoning_tokens",
            "reasoningTokens",
            "inputTokens",
            "outputTokens",
            "raw",
        ],
    );

    object.insert(
        "input_tokens".to_string(),
        input_total
            .map(serde_json::Value::from)
            .unwrap_or(serde_json::Value::Null),
    );
    object.insert(
        "output_tokens".to_string(),
        output_total
            .map(serde_json::Value::from)
            .unwrap_or(serde_json::Value::Null),
    );
    object.insert(
        "total_tokens".to_string(),
        total_tokens
            .map(serde_json::Value::from)
            .unwrap_or(serde_json::Value::Null),
    );

    if normalized_input.cache_read.is_some()
        || object
            .get("input_tokens_details")
            .and_then(Value::as_object)
            .is_some()
    {
        let details = ensure_object_entry(&mut object, "input_tokens_details");
        if let Some(cached_tokens) = normalized_input.cache_read {
            details.insert(
                "cached_tokens".to_string(),
                serde_json::json!(cached_tokens),
            );
        }
        if details.is_empty() {
            object.remove("input_tokens_details");
        }
    }

    if normalized_output.reasoning.is_some()
        || object
            .get("output_tokens_details")
            .and_then(Value::as_object)
            .is_some()
    {
        let details = ensure_object_entry(&mut object, "output_tokens_details");
        if let Some(reasoning_tokens) = normalized_output.reasoning {
            details.insert(
                "reasoning_tokens".to_string(),
                serde_json::json!(reasoning_tokens),
            );
        }
        if details.is_empty() {
            object.remove("output_tokens_details");
        }
    }

    Value::Object(object)
}

#[cfg(test)]
mod usage_tests {
    use super::*;

    #[test]
    fn parse_openai_usage_value_supports_ai_sdk_wrapper_with_raw_usage() {
        let usage = parse_openai_usage_value(&serde_json::json!({
            "inputTokens": {
                "total": 12,
                "noCache": 9,
                "cacheRead": 3
            },
            "outputTokens": {
                "total": 8,
                "text": 5,
                "reasoning": 3
            },
            "raw": {
                "input_tokens": 12,
                "input_tokens_details": {
                    "cached_tokens": 3
                },
                "output_tokens": 8,
                "output_tokens_details": {
                    "reasoning_tokens": 3
                },
                "total_tokens": 20
            }
        }))
        .expect("parse usage");

        assert_eq!(usage.prompt_tokens(), Some(12));
        assert_eq!(usage.completion_tokens(), Some(8));
        assert_eq!(usage.total_tokens(), Some(20));
        assert_eq!(usage.normalized_input_tokens().no_cache, Some(9));
        assert_eq!(usage.normalized_input_tokens().cache_read, Some(3));
        assert_eq!(usage.normalized_output_tokens().text, Some(5));
        assert_eq!(usage.normalized_output_tokens().reasoning, Some(3));
        assert_eq!(
            usage.raw_usage_value().expect("raw usage")["input_tokens"],
            serde_json::json!(12)
        );
    }

    #[test]
    fn parse_provider_openai_finish_reason_matches_ai_sdk_vendor_mappings() {
        assert_eq!(
            parse_provider_openai_finish_reason("deepseek", Some("insufficient_system_resource")),
            Some(FinishReason::Error)
        );
        assert_eq!(
            parse_provider_openai_finish_reason("mistral", Some("model_length")),
            Some(FinishReason::Length)
        );
        assert_eq!(
            parse_provider_openai_finish_reason("perplexity", Some("content_filter")),
            Some(FinishReason::Other("content_filter".to_string()))
        );
        assert_eq!(
            parse_provider_openai_finish_reason("cohere", Some("STOP_SEQUENCE")),
            Some(FinishReason::Stop)
        );
        assert_eq!(
            parse_provider_openai_finish_reason("groq", Some("function_call")),
            Some(FinishReason::ToolCalls)
        );
    }

    #[test]
    fn openai_chat_usage_value_preserves_vendor_extensions_and_strips_responses_keys() {
        let usage = Usage::builder()
            .prompt_tokens(10)
            .completion_tokens(8)
            .total_tokens(18)
            .with_input_total_tokens(10)
            .with_input_no_cache_tokens(6)
            .with_input_cache_read_tokens(4)
            .with_output_total_tokens(8)
            .with_output_text_tokens(5)
            .with_output_reasoning_tokens(3)
            .with_raw_usage_value(serde_json::json!({
                "citation_tokens": 7,
                "input_tokens": 99,
                "output_tokens": 42
            }))
            .build();

        let value = openai_chat_usage_value(&usage);
        assert_eq!(value["prompt_tokens"], serde_json::json!(10));
        assert_eq!(value["completion_tokens"], serde_json::json!(8));
        assert_eq!(
            value["prompt_tokens_details"]["cached_tokens"],
            serde_json::json!(4)
        );
        assert_eq!(
            value["completion_tokens_details"]["reasoning_tokens"],
            serde_json::json!(3)
        );
        assert_eq!(value["citation_tokens"], serde_json::json!(7));
        assert!(value.get("input_tokens").is_none());
        assert!(value.get("output_tokens").is_none());
    }

    #[test]
    fn openai_chat_usage_value_omits_synthetic_zero_details() {
        let usage = Usage::builder()
            .prompt_tokens(10)
            .completion_tokens(8)
            .total_tokens(18)
            .with_input_total_tokens(10)
            .with_input_no_cache_tokens(10)
            .with_input_cache_read_tokens(0)
            .with_output_total_tokens(8)
            .with_output_text_tokens(8)
            .with_output_reasoning_tokens(0)
            .with_raw_usage_value(serde_json::json!({
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18
            }))
            .build();

        let value = openai_chat_usage_value(&usage);

        assert_eq!(value["prompt_tokens"], serde_json::json!(10));
        assert_eq!(value["completion_tokens"], serde_json::json!(8));
        assert!(value.get("prompt_tokens_details").is_none());
        assert!(value.get("completion_tokens_details").is_none());
    }

    #[test]
    fn openai_responses_usage_value_preserves_vendor_extensions_and_strips_chat_keys() {
        let usage = Usage::builder()
            .prompt_tokens(11)
            .completion_tokens(9)
            .total_tokens(20)
            .with_cached_tokens(2)
            .with_reasoning_tokens(4)
            .with_raw_usage_value(serde_json::json!({
                "citation_tokens": 5,
                "prompt_tokens": 111,
                "completion_tokens": 222
            }))
            .build();

        let value = openai_responses_usage_value(&usage);
        assert_eq!(value["input_tokens"], serde_json::json!(11));
        assert_eq!(value["output_tokens"], serde_json::json!(9));
        assert_eq!(
            value["input_tokens_details"]["cached_tokens"],
            serde_json::json!(2)
        );
        assert_eq!(
            value["output_tokens_details"]["reasoning_tokens"],
            serde_json::json!(4)
        );
        assert_eq!(value["citation_tokens"], serde_json::json!(5));
        assert!(value.get("prompt_tokens").is_none());
        assert!(value.get("completion_tokens").is_none());
    }

    #[test]
    fn openai_responses_usage_value_preserves_unknown_totals_as_null() {
        let usage = Usage::builder()
            .with_raw_usage_value(serde_json::json!({
                "input_tokens": null,
                "output_tokens": null,
                "total_tokens": null
            }))
            .build();

        let value = openai_responses_usage_value(&usage);
        assert_eq!(value["input_tokens"], serde_json::Value::Null);
        assert_eq!(value["output_tokens"], serde_json::Value::Null);
        assert_eq!(value["total_tokens"], serde_json::Value::Null);
    }
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
/// - `responseFormat: { type: "json" }` => `{ type: "json_object" }`
pub fn convert_chat_completions_response_format(
    fmt: &crate::types::chat::ResponseFormat,
    strict_json_schema: bool,
) -> serde_json::Value {
    match fmt {
        crate::types::chat::ResponseFormat::JsonObject { .. } => {
            serde_json::json!({ "type": "json_object" })
        }
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
        crate::types::chat::ResponseFormat::JsonObject { .. } => {
            serde_json::json!({ "type": "json_object" })
        }
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
                                    if provider_tool.tool_type() == Some("custom") {
                                        return Some(serde_json::json!({
                                            "type": "custom",
                                            "name": name,
                                        }));
                                    }

                                    if let Some(tool_type) = provider_tool.tool_type()
                                        && let Some(t) = crate::tool_catalog::openai::responses_builtin_type_for_tool_type(
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
                crate::tool_catalog::openai::responses_builtin_type_for_choice_name(name.as_str())
            {
                return Some(serde_json::json!({ "type": t }));
            }

            Some(serde_json::json!({ "type": "function", "name": name }))
        }
    }
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
        let tool = crate::tool_catalog::openai::computer_use().with_args(serde_json::json!({
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
        let tool = crate::tool_catalog::openai::code_interpreter().with_args(serde_json::json!({
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
    fn responses_tools_map_function_defer_loading_option() {
        let mut provider_options = crate::types::ProviderOptionsMap::default();
        provider_options.insert("openai", serde_json::json!({ "deferLoading": true }));
        let tool = crate::types::Tool::function(
            "get_weather",
            "Get weather",
            serde_json::json!({
                "type": "object",
                "properties": {},
            }),
        )
        .with_provider_options_map(provider_options);

        let out = convert_tools_to_responses_format(&[tool]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["type"], serde_json::json!("function"));
        assert_eq!(out[0]["defer_loading"], serde_json::json!(true));
    }

    #[test]
    fn responses_tools_map_image_generation_keys() {
        let tool = crate::tool_catalog::openai::image_generation().with_args(serde_json::json!({
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
    fn responses_tools_map_mcp_filters_like_ai_sdk() {
        let tool = crate::tool_catalog::openai::mcp().with_args(serde_json::json!({
            "serverLabel": "docs",
            "serverUrl": "https://example.com/mcp",
            "allowedTools": {
                "readOnly": true,
                "toolNames": ["search_docs"],
            },
            "connectorId": "conn_123",
            "authorization": "Bearer token",
            "requireApproval": {
                "never": {
                    "toolNames": ["safe_tool"],
                }
            },
        }));

        let out = convert_tools_to_responses_format(&[tool]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["type"], serde_json::json!("mcp"));
        assert_eq!(out[0]["server_label"], serde_json::json!("docs"));
        assert_eq!(
            out[0]["server_url"],
            serde_json::json!("https://example.com/mcp")
        );
        assert_eq!(
            out[0]["allowed_tools"],
            serde_json::json!({
                "read_only": true,
                "tool_names": ["search_docs"],
            })
        );
        assert_eq!(out[0]["connector_id"], serde_json::json!("conn_123"));
        assert_eq!(out[0]["authorization"], serde_json::json!("Bearer token"));
        assert_eq!(
            out[0]["require_approval"],
            serde_json::json!({
                "never": { "tool_names": ["safe_tool"] }
            })
        );
    }

    #[test]
    fn responses_tools_map_shell_environment_like_ai_sdk() {
        let tool = crate::tool_catalog::openai::shell().with_args(serde_json::json!({
            "environment": {
                "type": "containerAuto",
                "fileIds": ["file_1"],
                "memoryLimit": "16g",
                "networkPolicy": {
                    "type": "allowlist",
                    "allowedDomains": ["example.com"],
                    "domainSecrets": [
                        { "domain": "example.com", "name": "API_KEY", "value": "secret" }
                    ]
                },
                "skills": [
                    {
                        "type": "skillReference",
                        "providerReference": { "openai": "skill_abc" },
                    },
                    {
                        "type": "inline",
                        "name": "my-skill",
                        "description": "A test skill",
                        "source": {
                            "type": "base64",
                            "mediaType": "application/zip",
                            "data": "dGVzdA=="
                        }
                    }
                ]
            }
        }));

        let out = convert_tools_to_responses_format(&[tool]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["type"], serde_json::json!("shell"));
        assert_eq!(
            out[0]["environment"],
            serde_json::json!({
                "type": "container_auto",
                "file_ids": ["file_1"],
                "memory_limit": "16g",
                "network_policy": {
                    "type": "allowlist",
                    "allowed_domains": ["example.com"],
                    "domain_secrets": [
                        { "domain": "example.com", "name": "API_KEY", "value": "secret" }
                    ]
                },
                "skills": [
                    {
                        "type": "skill_reference",
                        "skill_id": "skill_abc",
                        "version": "latest"
                    },
                    {
                        "type": "inline",
                        "name": "my-skill",
                        "description": "A test skill",
                        "source": {
                            "type": "base64",
                            "media_type": "application/zip",
                            "data": "dGVzdA=="
                        }
                    }
                ]
            })
        );
    }

    #[test]
    fn responses_tools_map_custom_and_tool_search() {
        let tools = vec![
            crate::tool_catalog::openai::custom("write_sql").with_args(serde_json::json!({
                "description": "Write SQL.",
                "format": {
                    "type": "grammar",
                    "syntax": "regex",
                    "definition": "SELECT .+"
                }
            })),
            crate::tool_catalog::openai::tool_search().with_args(serde_json::json!({
                "execution": "client",
                "description": "Search for deferred tools",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "goal": { "type": "string" }
                    }
                }
            })),
        ];

        let out = convert_tools_to_responses_format(&tools).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(
            out[0],
            serde_json::json!({
                "type": "custom",
                "name": "write_sql",
                "description": "Write SQL.",
                "format": {
                    "type": "grammar",
                    "syntax": "regex",
                    "definition": "SELECT .+"
                }
            })
        );
        assert_eq!(
            out[1],
            serde_json::json!({
                "type": "tool_search",
                "execution": "client",
                "description": "Search for deferred tools",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "goal": { "type": "string" }
                    }
                }
            })
        );
    }

    #[test]
    fn responses_tools_map_xai_server_tools_to_sdk_aligned_shapes() {
        let tools = vec![
            crate::types::Tool::provider_defined("xai.web_search", "web_search").with_args(
                serde_json::json!({
                    "allowedDomains": ["wikipedia.org"],
                    "enableImageUnderstanding": true,
                }),
            ),
            crate::types::Tool::provider_defined("xai.x_search", "x_search").with_args(
                serde_json::json!({
                    "allowedXHandles": ["xai"],
                    "fromDate": "2025-01-01",
                    "enableVideoUnderstanding": true,
                }),
            ),
            crate::types::Tool::provider_defined("xai.view_image", "view_image"),
            crate::types::Tool::provider_defined("xai.view_x_video", "view_x_video"),
            crate::types::Tool::provider_defined("xai.file_search", "file_search").with_args(
                serde_json::json!({
                    "vectorStoreIds": ["collection_1"],
                    "maxNumResults": 5,
                }),
            ),
            crate::types::Tool::provider_defined("xai.mcp", "mcp").with_args(serde_json::json!({
                "serverUrl": "https://example.com/mcp",
                "serverLabel": "docs",
                "allowedTools": ["search_docs"],
                "authorization": "Bearer token",
            })),
            crate::types::Tool::provider_defined("xai.code_execution", "code_execution"),
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
        let tools = vec![crate::tool_catalog::openai::image_generation_named(
            "generateImage",
        )];
        let out = convert_responses_tool_choice(&choice, Some(&tools));
        assert_eq!(out, Some(serde_json::json!({ "type": "image_generation" })));
    }

    #[test]
    fn responses_tool_choice_resolves_custom_provider_tool_name_for_computer_use() {
        let choice = crate::types::ToolChoice::tool("myComputer");
        let tools = vec![crate::tool_catalog::openai::computer_use_named(
            "myComputer",
        )];
        let out = convert_responses_tool_choice(&choice, Some(&tools));
        assert_eq!(
            out,
            Some(serde_json::json!({ "type": "computer_use_preview" }))
        );
    }

    #[test]
    fn responses_tool_choice_resolves_openai_custom_provider_tool_name() {
        let choice = crate::types::ToolChoice::tool("write_sql");
        let tools = vec![crate::tool_catalog::openai::custom("write_sql")];
        let out = convert_responses_tool_choice(&choice, Some(&tools));
        assert_eq!(
            out,
            Some(serde_json::json!({ "type": "custom", "name": "write_sql" }))
        );
    }

    #[test]
    fn responses_tool_choice_drops_xai_server_side_tools() {
        let choice = crate::types::ToolChoice::tool("web_search");
        let tools = vec![crate::types::Tool::provider_defined(
            "xai.web_search",
            "web_search",
        )];
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
