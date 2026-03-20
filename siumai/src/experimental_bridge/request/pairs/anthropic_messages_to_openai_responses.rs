//! Direct request bridge for `Anthropic Messages -> OpenAI Responses`.
//!
//! The direct path keeps the reusable OpenAI Responses transformer in charge of
//! most payload shaping, while applying pair-specific preprocessing that the
//! normalized route cannot infer safely:
//!
//! - flatten leading Anthropic system/developer instructions into a single
//!   Responses `system` item
//! - force `store=false` request semantics during input conversion
//! - synthesize replayable reasoning item ids for Anthropic assistant thinking
//! - translate the highest-value Anthropic server tools into Responses tools

use std::collections::{BTreeSet, HashMap};

use serde_json::{Map, Value, json};
use siumai_core::LlmError;
use siumai_core::bridge::{BridgeReport, BridgeWarning, BridgeWarningKind};
use siumai_core::execution::transformers::request::RequestTransformer;
use siumai_core::types::{
    ChatMessage, ChatRequest, ContentPart, MessageContent, MessageRole, Tool, ToolChoice,
};

pub(crate) fn serialize_anthropic_messages_to_openai_responses(
    request: &ChatRequest,
    report: &mut BridgeReport,
) -> Result<serde_json::Value, LlmError> {
    let prepared = prepare_direct_request(request, report);
    let tx =
        siumai_protocol_openai::standards::openai::transformers::request::OpenAiResponsesRequestTransformer;
    let mut body = tx.transform_chat(&prepared)?;

    body["store"] = Value::Bool(false);
    ensure_include_entry(&mut body, "reasoning.encrypted_content");
    ensure_auto_include_from_tools(&prepared, &mut body);
    apply_anthropic_reasoning_options(request, &mut body);
    apply_anthropic_parallel_tool_policy(request, &mut body);

    Ok(body)
}

fn prepare_direct_request(request: &ChatRequest, report: &mut BridgeReport) -> ChatRequest {
    let mut prepared = request.clone();

    flatten_leading_system_messages(&mut prepared, report);
    translate_anthropic_tools(&mut prepared, report);
    annotate_anthropic_reasoning_for_openai(&mut prepared, report);
    force_openai_responses_input_mode(&mut prepared);

    prepared
}

fn flatten_leading_system_messages(request: &mut ChatRequest, report: &mut BridgeReport) {
    let Some(first) = request.messages.first() else {
        return;
    };

    let mut prefix_len = 0usize;
    let mut flattened_parts = Vec::new();
    let mut needs_flatten = false;

    for message in &request.messages {
        match message.role {
            MessageRole::System | MessageRole::Developer => {
                prefix_len += 1;
                if !matches!(message.role, MessageRole::System) {
                    needs_flatten = true;
                }

                let text = message.content.all_text();
                if !text.trim().is_empty() {
                    flattened_parts.push(text);
                }
            }
            _ => break,
        }
    }

    if prefix_len == 0 {
        return;
    }

    if prefix_len == 1 && matches!(first.role, MessageRole::System) && !needs_flatten {
        return;
    }

    let mut rewritten = Vec::with_capacity(request.messages.len() - prefix_len + 1);
    let flattened = flattened_parts.join("\n\n");
    if !flattened.trim().is_empty() {
        rewritten.push(ChatMessage::system(flattened).build());
    }
    rewritten.extend(request.messages.iter().skip(prefix_len).cloned());
    request.messages = rewritten;

    report.add_warning(BridgeWarning::new(
        BridgeWarningKind::Custom,
        "anthropic direct bridge flattened leading system/developer messages into one OpenAI Responses system item",
    ));
}

fn translate_anthropic_tools(request: &mut ChatRequest, report: &mut BridgeReport) {
    let Some(original_tools) = request.tools.as_ref() else {
        return;
    };

    let mut translated = Vec::new();
    let mut available_names = BTreeSet::new();

    for (index, tool) in original_tools.iter().enumerate() {
        let Some(tool) = translate_tool(index, tool, report) else {
            continue;
        };

        if let Some(name) = tool_name(&tool) {
            available_names.insert(name.to_string());
        }
        translated.push(tool);
    }

    request.tools = (!translated.is_empty()).then_some(translated);
    repair_tool_choice(request, &available_names, report);
}

fn translate_tool(index: usize, tool: &Tool, report: &mut BridgeReport) -> Option<Tool> {
    match tool {
        Tool::Function { .. } => Some(tool.clone()),
        Tool::ProviderDefined(provider_tool) => match provider_tool.provider() {
            Some("anthropic") => map_anthropic_provider_tool(index, provider_tool, report),
            Some("openai") | Some("xai") => Some(tool.clone()),
            Some(other) => {
                report.record_dropped_field(
                    format!("tools[{index}]"),
                    format!(
                        "direct anthropic-to-responses bridge does not translate provider-defined tool provider `{other}`"
                    ),
                );
                None
            }
            None => Some(tool.clone()),
        },
    }
}

fn map_anthropic_provider_tool(
    index: usize,
    provider_tool: &siumai_core::types::ProviderDefinedTool,
    report: &mut BridgeReport,
) -> Option<Tool> {
    let tool_type = provider_tool.tool_type().unwrap_or_default();

    match tool_type {
        "web_search_20250305" => Some(
            Tool::provider_defined("openai.web_search", provider_tool.name.clone())
                .with_args(map_web_search_args(index, &provider_tool.args, report)),
        ),
        "code_execution_20250522" | "code_execution_20250825" => Some(
            Tool::provider_defined("openai.code_interpreter", provider_tool.name.clone())
                .with_args(map_code_execution_args(index, &provider_tool.args, report)),
        ),
        other => {
            report.record_dropped_field(
                format!("tools[{index}]"),
                format!(
                    "OpenAI Responses direct pair does not yet have a stable mapping for Anthropic server tool `{other}`"
                ),
            );
            None
        }
    }
}

fn map_web_search_args(index: usize, args: &Value, report: &mut BridgeReport) -> Value {
    let Some(obj) = args.as_object() else {
        return Value::Object(Map::new());
    };

    let mut out = Map::new();

    if let Some(value) = obj.get("userLocation").or_else(|| obj.get("user_location")) {
        out.insert("userLocation".to_string(), value.clone());
    }

    if let Some(value) = obj
        .get("allowedDomains")
        .or_else(|| obj.get("allowed_domains"))
    {
        out.insert(
            "filters".to_string(),
            json!({
                "allowedDomains": value.clone(),
            }),
        );
    }

    if obj
        .get("blockedDomains")
        .or_else(|| obj.get("blocked_domains"))
        .is_some()
    {
        report.record_dropped_field(
            format!("tools[{index}].args.blockedDomains"),
            "OpenAI Responses web_search does not support blocked domain filters",
        );
    }

    if obj.get("maxUses").or_else(|| obj.get("max_uses")).is_some() {
        report.record_dropped_field(
            format!("tools[{index}].args.maxUses"),
            "OpenAI Responses web_search does not support Anthropic maxUses",
        );
    }

    Value::Object(out)
}

fn map_code_execution_args(index: usize, args: &Value, report: &mut BridgeReport) -> Value {
    if let Some(obj) = args.as_object() {
        for key in obj.keys() {
            report.record_dropped_field(
                format!("tools[{index}].args.{key}"),
                "OpenAI code_interpreter does not reuse Anthropic code_execution tool arguments directly",
            );
        }
    }

    Value::Object(Map::new())
}

fn repair_tool_choice(
    request: &mut ChatRequest,
    available_names: &BTreeSet<String>,
    report: &mut BridgeReport,
) {
    match request.tool_choice.as_ref() {
        Some(ToolChoice::Required) if available_names.is_empty() => {
            report.record_lossy_field(
                "tool_choice",
                "required tool choice was dropped because no tool survived direct pair translation",
            );
            request.tool_choice = None;
        }
        Some(ToolChoice::Tool { name }) if !available_names.contains(name) => {
            report.record_lossy_field(
                "tool_choice",
                format!(
                    "specific tool choice `{name}` could not be preserved because the target tool definition was not translated"
                ),
            );
            request.tool_choice = if available_names.is_empty() {
                None
            } else {
                Some(ToolChoice::Auto)
            };
        }
        Some(ToolChoice::Auto) if available_names.is_empty() => {
            request.tool_choice = None;
        }
        _ => {}
    }
}

fn annotate_anthropic_reasoning_for_openai(request: &mut ChatRequest, report: &mut BridgeReport) {
    let mut carried_redacted_payload = false;

    for (message_index, message) in request.messages.iter_mut().enumerate() {
        if !matches!(message.role, MessageRole::Assistant) {
            continue;
        }

        let redacted_payload = message
            .metadata
            .custom
            .get("anthropic_redacted_thinking_data")
            .cloned();

        let MessageContent::MultiModal(parts) = &mut message.content else {
            continue;
        };

        for (part_index, part) in parts.iter_mut().enumerate() {
            let ContentPart::Reasoning {
                provider_metadata, ..
            } = part
            else {
                continue;
            };

            let provider_metadata = provider_metadata.get_or_insert_with(HashMap::new);
            let openai_entry = provider_metadata
                .entry("openai".to_string())
                .or_insert_with(|| Value::Object(Map::new()));

            if !openai_entry.is_object() {
                *openai_entry = Value::Object(Map::new());
            }

            let openai = openai_entry
                .as_object_mut()
                .expect("openai metadata entry was normalized to an object");

            openai.entry("itemId".to_string()).or_insert_with(|| {
                json!(format!("anthropic_reasoning_{message_index}_{part_index}"))
            });

            if let Some(redacted_payload) = redacted_payload.as_ref() {
                if !openai.contains_key("reasoningEncryptedContent")
                    && !openai.contains_key("reasoning_encrypted_content")
                {
                    openai.insert(
                        "reasoningEncryptedContent".to_string(),
                        redacted_payload.clone(),
                    );
                    carried_redacted_payload = true;
                }
            }
        }
    }

    if carried_redacted_payload {
        report.record_carried_provider_metadata(
            "anthropic.redacted_thinking_data",
            "anthropic redacted thinking payload was carried into OpenAI reasoning encrypted content",
        );
    }
}

fn force_openai_responses_input_mode(request: &mut ChatRequest) {
    let mut openai = request
        .provider_options_map
        .get("openai")
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();

    openai.insert("store".to_string(), Value::Bool(false));
    openai.insert(
        "systemMessageMode".to_string(),
        Value::String("system".to_string()),
    );

    request
        .provider_options_map
        .insert("openai", Value::Object(openai));
}

fn apply_anthropic_reasoning_options(request: &ChatRequest, body: &mut Value) {
    let Some(effort) = anthropic_option_string(request, "effort") else {
        return;
    };

    let mapped_effort = match effort {
        "medium" => "high",
        "high" => "xhigh",
        other => other,
    };

    let reasoning = body
        .as_object_mut()
        .expect("responses request body must be an object")
        .entry("reasoning".to_string())
        .or_insert_with(|| Value::Object(Map::new()));

    if !reasoning.is_object() {
        *reasoning = Value::Object(Map::new());
    }

    let reasoning = reasoning
        .as_object_mut()
        .expect("reasoning entry was normalized to object");
    reasoning
        .entry("effort".to_string())
        .or_insert_with(|| Value::String(mapped_effort.to_string()));
    reasoning
        .entry("summary".to_string())
        .or_insert_with(|| Value::String("auto".to_string()));
}

fn apply_anthropic_parallel_tool_policy(request: &ChatRequest, body: &mut Value) {
    let Some(value) = anthropic_option_bool(
        request,
        &["disableParallelToolUse", "disable_parallel_tool_use"],
    ) else {
        return;
    };

    if value {
        body["parallel_tool_calls"] = Value::Bool(false);
    }
}

fn ensure_auto_include_from_tools(request: &ChatRequest, body: &mut Value) {
    let Some(tools) = request.tools.as_ref() else {
        return;
    };

    for tool in tools {
        let Tool::ProviderDefined(provider_tool) = tool else {
            continue;
        };

        if provider_tool.provider() != Some("openai") {
            continue;
        }

        match provider_tool.tool_type() {
            Some("code_interpreter") => {
                ensure_include_entry(body, "code_interpreter_call.outputs");
            }
            Some("web_search") | Some("web_search_preview") => {
                ensure_include_entry(body, "web_search_call.action.sources");
            }
            _ => {}
        }
    }
}

fn ensure_include_entry(body: &mut Value, include_value: &str) {
    let include = body
        .as_object_mut()
        .expect("responses request body must be an object")
        .entry("include".to_string())
        .or_insert_with(|| Value::Array(Vec::new()));

    if !include.is_array() {
        *include = Value::Array(Vec::new());
    }

    let include = include
        .as_array_mut()
        .expect("include entry was normalized to array");

    if include
        .iter()
        .any(|value| value.as_str() == Some(include_value))
    {
        return;
    }

    include.push(Value::String(include_value.to_string()));
}

fn anthropic_option_string<'a>(request: &'a ChatRequest, key: &str) -> Option<&'a str> {
    let obj = request.provider_options_map.get("anthropic")?.as_object()?;
    obj.get(key).and_then(|value| value.as_str())
}

fn anthropic_option_bool(request: &ChatRequest, keys: &[&str]) -> Option<bool> {
    let obj = request.provider_options_map.get("anthropic")?.as_object()?;
    for key in keys {
        if let Some(value) = obj.get(*key).and_then(|value| value.as_bool()) {
            return Some(value);
        }
    }
    None
}

fn tool_name(tool: &Tool) -> Option<&str> {
    match tool {
        Tool::Function { function } => Some(function.name.as_str()),
        Tool::ProviderDefined(provider_tool) => Some(provider_tool.name.as_str()),
    }
}
