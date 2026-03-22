//! Direct request bridge for `OpenAI Responses -> Anthropic Messages`.
//!
//! Like the forward bridge, this route keeps the existing Anthropic request
//! transformer as the backbone and only patches semantics that the normalized
//! path cannot recover on its own:
//!
//! - lift Responses `instructions` into Anthropic `system`
//! - translate the highest-value OpenAI Responses built-in tools into
//!   Anthropic server tools / MCP server config
//! - replay OpenAI reasoning items as Anthropic redacted thinking when an
//!   encrypted payload is available
//! - map Responses-only request knobs such as `parallel_tool_calls=false`,
//!   `reasoning_effort`, and provider-level `response_format`

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value, json};
use siumai_core::LlmError;
use siumai_core::bridge::{BridgeReport, BridgeWarning, BridgeWarningKind};
use siumai_core::execution::transformers::request::RequestTransformer;
use siumai_core::types::chat::ResponseFormat;
use siumai_core::types::{
    ChatMessage, ChatRequest, ContentPart, MessageContent, MessageRole, Tool, ToolChoice,
};

use super::tool_rules::{
    ProviderToolTranslationRule, TargetToolNamePolicy, find_provider_tool_translation_rule,
};

const OPENAI_TO_ANTHROPIC_TOOL_RULES: &[ProviderToolTranslationRule] = &[
    ProviderToolTranslationRule {
        source_tool_types: &["web_search", "web_search_preview"],
        target_tool_id: siumai_core::tools::anthropic::WEB_SEARCH_20250305_ID,
        target_tool_name: TargetToolNamePolicy::Fixed("web_search"),
        choice_name: Some("web_search"),
        aliases: &["web_search", "web_search_preview"],
        args_mapper: map_web_search_args,
    },
    ProviderToolTranslationRule {
        source_tool_types: &["code_interpreter"],
        target_tool_id: siumai_core::tools::anthropic::CODE_EXECUTION_20250825_ID,
        target_tool_name: TargetToolNamePolicy::Fixed("code_execution"),
        choice_name: Some("code_execution"),
        aliases: &["code_interpreter"],
        args_mapper: map_code_execution_args,
    },
    ProviderToolTranslationRule {
        source_tool_types: &["computer_use"],
        target_tool_id: siumai_core::tools::anthropic::COMPUTER_20250124_ID,
        target_tool_name: TargetToolNamePolicy::Fixed("computer"),
        choice_name: Some("computer"),
        aliases: &["computer_use", "computer_use_preview"],
        args_mapper: map_computer_use_args,
    },
];

pub(crate) fn serialize_openai_responses_to_anthropic_messages(
    request: &ChatRequest,
    report: &mut BridgeReport,
) -> Result<serde_json::Value, LlmError> {
    let prepared = prepare_direct_request(request, report);
    let tx =
        siumai_protocol_anthropic::standards::anthropic::transformers::AnthropicRequestTransformer::default();
    let mut body = tx.transform_chat(&prepared)?;

    apply_mcp_servers_overlay(&prepared, &mut body);
    apply_effort_overlay(&prepared, &mut body);

    Ok(body)
}

fn prepare_direct_request(request: &ChatRequest, report: &mut BridgeReport) -> ChatRequest {
    let mut prepared = request.clone();

    lift_openai_instructions_to_system(&mut prepared, report);
    translate_openai_tools(&mut prepared, report);
    lift_openai_response_format(&mut prepared, report);
    annotate_openai_reasoning_for_anthropic(&mut prepared, report);
    apply_openai_parallel_tool_policy(&mut prepared);
    apply_openai_reasoning_effort_policy(&mut prepared, report);

    prepared
}

fn lift_openai_instructions_to_system(request: &mut ChatRequest, report: &mut BridgeReport) {
    let Some(instructions) = openai_instruction_text(request) else {
        return;
    };

    if instructions.trim().is_empty() {
        return;
    }

    request
        .messages
        .insert(0, ChatMessage::system(instructions.clone()).build());

    report.add_warning(BridgeWarning::new(
        BridgeWarningKind::Custom,
        "openai direct bridge lifted Responses instructions into Anthropic system blocks",
    ));
}

fn translate_openai_tools(request: &mut ChatRequest, report: &mut BridgeReport) {
    let Some(original_tools) = request.tools.as_ref() else {
        return;
    };

    let mut translated = Vec::new();
    let mut mcp_servers = Vec::new();
    let mut available_names = BTreeSet::new();
    let mut choice_aliases = BTreeMap::new();

    for (index, tool) in original_tools.iter().enumerate() {
        match tool {
            Tool::Function { function } => {
                translated.push(tool.clone());
                available_names.insert(function.name.clone());
                choice_aliases.insert(function.name.clone(), function.name.clone());
            }
            Tool::ProviderDefined(provider_tool) => match provider_tool.provider() {
                Some("openai") => match map_openai_provider_tool(index, provider_tool, report) {
                    ToolTranslation::AnthropicTool {
                        tool,
                        choice_name,
                        aliases,
                    } => {
                        available_names.insert(choice_name.clone());
                        choice_aliases.insert(choice_name.clone(), choice_name.clone());
                        for alias in aliases {
                            choice_aliases.insert(alias, choice_name.clone());
                        }
                        translated.push(tool);
                    }
                    ToolTranslation::McpServer { server, aliases } => {
                        mcp_servers.push(server);
                        for alias in aliases {
                            choice_aliases.insert(alias, "__anthropic_mcp_server__".to_string());
                        }
                    }
                    ToolTranslation::Dropped => {}
                },
                Some("anthropic") => {
                    translated.push(tool.clone());
                    if let Some(name) = tool_name(tool) {
                        available_names.insert(name.to_string());
                        choice_aliases.insert(name.to_string(), name.to_string());
                    }
                }
                Some(other) => {
                    report.record_dropped_field(
                        format!("tools[{index}]"),
                        format!(
                            "direct responses-to-anthropic bridge does not translate provider-defined tool provider `{other}`"
                        ),
                    );
                }
                None => {
                    translated.push(tool.clone());
                    if let Some(name) = tool_name(tool) {
                        available_names.insert(name.to_string());
                        choice_aliases.insert(name.to_string(), name.to_string());
                    }
                }
            },
        }
    }

    if !mcp_servers.is_empty() {
        set_anthropic_option(request, "mcpServers", Value::Array(mcp_servers));
    }

    request.tools = (!translated.is_empty()).then_some(translated);
    repair_tool_choice(request, &available_names, &choice_aliases, report);
}

enum ToolTranslation {
    AnthropicTool {
        tool: Tool,
        choice_name: String,
        aliases: Vec<String>,
    },
    McpServer {
        server: Value,
        aliases: Vec<String>,
    },
    Dropped,
}

fn map_openai_provider_tool(
    index: usize,
    provider_tool: &siumai_core::types::ProviderDefinedTool,
    report: &mut BridgeReport,
) -> ToolTranslation {
    let tool_type = provider_tool.tool_type().unwrap_or_default();

    match tool_type {
        "mcp" => match map_mcp_server(index, provider_tool, report) {
            Some(server) => ToolTranslation::McpServer {
                server,
                aliases: vec![provider_tool.name.clone(), "mcp".to_string()],
            },
            None => ToolTranslation::Dropped,
        },
        other => {
            match find_provider_tool_translation_rule(provider_tool, OPENAI_TO_ANTHROPIC_TOOL_RULES)
            {
                Some(rule) => ToolTranslation::AnthropicTool {
                    tool: rule.translate_tool(index, provider_tool, report),
                    choice_name: rule.choice_name(provider_tool),
                    aliases: rule.aliases(provider_tool),
                },
                None => {
                    report.record_dropped_field(
                format!("tools[{index}]"),
                format!(
                    "Anthropic direct pair does not yet have a stable mapping for OpenAI Responses tool `{other}`"
                ),
            );
                    ToolTranslation::Dropped
                }
            }
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

    if let Some(filters) = obj.get("filters").and_then(|value| value.as_object())
        && let Some(value) = filters
            .get("allowedDomains")
            .or_else(|| filters.get("allowed_domains"))
    {
        out.insert("allowedDomains".to_string(), value.clone());
    }

    if obj
        .get("externalWebAccess")
        .or_else(|| obj.get("external_web_access"))
        .is_some()
    {
        report.record_dropped_field(
            format!("tools[{index}].args.externalWebAccess"),
            "Anthropic web_search does not support OpenAI external_web_access",
        );
    }

    if obj
        .get("searchContextSize")
        .or_else(|| obj.get("search_context_size"))
        .is_some()
    {
        report.record_dropped_field(
            format!("tools[{index}].args.searchContextSize"),
            "Anthropic web_search does not expose OpenAI search context size",
        );
    }

    Value::Object(out)
}

fn map_code_execution_args(index: usize, args: &Value, report: &mut BridgeReport) -> Value {
    if let Some(obj) = args.as_object() {
        for key in obj.keys() {
            report.record_dropped_field(
                format!("tools[{index}].args.{key}"),
                "Anthropic code_execution does not reuse OpenAI code_interpreter arguments directly",
            );
        }
    }

    Value::Object(Map::new())
}

fn map_computer_use_args(index: usize, args: &Value, report: &mut BridgeReport) -> Value {
    let Some(obj) = args.as_object() else {
        return Value::Object(Map::new());
    };

    let mut out = Map::new();

    if let Some(value) = obj.get("displayWidth").or_else(|| obj.get("display_width")) {
        out.insert("displayWidthPx".to_string(), value.clone());
    }
    if let Some(value) = obj
        .get("displayHeight")
        .or_else(|| obj.get("display_height"))
    {
        out.insert("displayHeightPx".to_string(), value.clone());
    }

    if obj
        .get("displayScale")
        .or_else(|| obj.get("display_scale"))
        .is_some()
    {
        report.record_dropped_field(
            format!("tools[{index}].args.displayScale"),
            "Anthropic computer tool does not expose OpenAI display scale in the same shape",
        );
    }

    if obj.get("environment").is_some() {
        report.record_dropped_field(
            format!("tools[{index}].args.environment"),
            "Anthropic computer tool does not expose OpenAI environment directly",
        );
    }

    Value::Object(out)
}

fn map_mcp_server(
    index: usize,
    provider_tool: &siumai_core::types::ProviderDefinedTool,
    report: &mut BridgeReport,
) -> Option<Value> {
    let Some(obj) = provider_tool.args.as_object() else {
        report.record_dropped_field(
            format!("tools[{index}]"),
            "OpenAI MCP tool is missing an args object and could not be mapped to Anthropic mcp_servers",
        );
        return None;
    };

    let url = obj
        .get("serverUrl")
        .or_else(|| obj.get("server_url"))
        .and_then(|value| value.as_str())
        .map(str::to_string);

    let Some(url) = url else {
        report.record_dropped_field(
            format!("tools[{index}].args.serverUrl"),
            "Anthropic mcp_servers requires a server URL",
        );
        return None;
    };

    let mut server = Map::new();
    server.insert(
        "serverName".to_string(),
        Value::String(
            obj.get("serverLabel")
                .or_else(|| obj.get("server_label"))
                .and_then(|value| value.as_str())
                .unwrap_or(provider_tool.name.as_str())
                .to_string(),
        ),
    );
    server.insert("serverUrl".to_string(), Value::String(url));

    if let Some(allowed_tools) = obj.get("allowedTools").or_else(|| obj.get("allowed_tools")) {
        server.insert(
            "toolConfiguration".to_string(),
            json!({
                "allowedTools": allowed_tools.clone(),
            }),
        );
    }

    if let Some(token) = extract_authorization_token(obj.get("headers")) {
        server.insert("authorizationToken".to_string(), Value::String(token));
    } else if obj.get("headers").is_some() {
        report.record_dropped_field(
            format!("tools[{index}].args.headers"),
            "only Authorization bearer headers are mapped into Anthropic authorization_token",
        );
    }

    if obj
        .get("requireApproval")
        .or_else(|| obj.get("require_approval"))
        .is_some()
    {
        report.record_dropped_field(
            format!("tools[{index}].args.requireApproval"),
            "Anthropic mcp_servers does not expose OpenAI require_approval",
        );
    }

    if obj
        .get("serverDescription")
        .or_else(|| obj.get("server_description"))
        .is_some()
    {
        report.record_dropped_field(
            format!("tools[{index}].args.serverDescription"),
            "Anthropic mcp_servers does not expose OpenAI server description directly",
        );
    }

    Some(Value::Object(server))
}

fn extract_authorization_token(headers: Option<&Value>) -> Option<String> {
    let headers = headers?.as_object()?;
    let value = headers
        .get("Authorization")
        .or_else(|| headers.get("authorization"))
        .and_then(|value| value.as_str())?;

    if let Some(token) = value.strip_prefix("Bearer ") {
        return Some(token.to_string());
    }

    Some(value.to_string())
}

fn repair_tool_choice(
    request: &mut ChatRequest,
    available_names: &BTreeSet<String>,
    choice_aliases: &BTreeMap<String, String>,
    report: &mut BridgeReport,
) {
    match request.tool_choice.clone() {
        Some(ToolChoice::Required) if available_names.is_empty() => {
            report.record_lossy_field(
                "tool_choice",
                "required tool choice was dropped because no Anthropic tool survived direct pair translation",
            );
            request.tool_choice = None;
        }
        Some(ToolChoice::Tool { name }) => {
            if let Some(mapped) = choice_aliases.get(&name) {
                if mapped == "__anthropic_mcp_server__" {
                    report.record_lossy_field(
                        "tool_choice",
                        "specific MCP tool choice cannot be enforced on Anthropic mcp_servers",
                    );
                    request.tool_choice = None;
                } else {
                    request.tool_choice = Some(ToolChoice::tool(mapped.clone()));
                }
            } else if !available_names.contains(&name) {
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
        }
        Some(ToolChoice::Auto) if available_names.is_empty() => {
            request.tool_choice = None;
        }
        _ => {}
    }
}

fn lift_openai_response_format(request: &mut ChatRequest, report: &mut BridgeReport) {
    if request.response_format.is_some() {
        return;
    }

    let Some(format) = openai_response_format_value(request) else {
        return;
    };

    let Some(parsed) = parse_openai_response_format(format, report) else {
        return;
    };

    request.response_format = Some(parsed);
}

fn parse_openai_response_format(
    format: &Value,
    report: &mut BridgeReport,
) -> Option<ResponseFormat> {
    let type_name = format.get("type").and_then(|value| value.as_str());

    let (schema_owner, default_name, default_desc, default_strict) =
        if let Some(owner) = format.get("json_schema") {
            (
                owner,
                owner.get("name").and_then(|value| value.as_str()),
                owner.get("description").and_then(|value| value.as_str()),
                owner.get("strict").and_then(|value| value.as_bool()),
            )
        } else {
            (
                format,
                format.get("name").and_then(|value| value.as_str()),
                format.get("description").and_then(|value| value.as_str()),
                format.get("strict").and_then(|value| value.as_bool()),
            )
        };

    if !matches!(type_name, Some("json_schema")) && schema_owner.get("schema").is_none() {
        report.record_dropped_field(
            "provider_options_map.openai.responses_api.response_format",
            "only OpenAI JSON schema response formats are mapped into Anthropic response_format",
        );
        return None;
    }

    let schema = schema_owner.get("schema")?.clone();
    let mut out = ResponseFormat::json_schema(schema);
    if let Some(name) = default_name {
        out = out.with_name(name.to_string());
    }
    if let Some(description) = default_desc {
        out = out.with_description(description.to_string());
    }
    if let Some(strict) = default_strict {
        out = out.with_strict(strict);
    }
    Some(out)
}

fn annotate_openai_reasoning_for_anthropic(request: &mut ChatRequest, report: &mut BridgeReport) {
    for (message_index, message) in request.messages.iter_mut().enumerate() {
        if !matches!(message.role, MessageRole::Assistant) {
            continue;
        }

        let MessageContent::MultiModal(parts) = &message.content else {
            continue;
        };

        let mut encrypted_values = BTreeSet::new();
        for part in parts {
            let ContentPart::Reasoning {
                provider_metadata, ..
            } = part
            else {
                continue;
            };

            let encrypted = provider_metadata
                .as_ref()
                .and_then(|meta| meta.get("openai"))
                .and_then(|value| value.as_object())
                .and_then(|meta| {
                    meta.get("reasoningEncryptedContent")
                        .or_else(|| meta.get("reasoning_encrypted_content"))
                })
                .and_then(|value| value.as_str());

            if let Some(encrypted) = encrypted {
                encrypted_values.insert(encrypted.to_string());
            }
        }

        let Some(first) = encrypted_values.iter().next().cloned() else {
            continue;
        };

        if encrypted_values.len() > 1 {
            report.record_lossy_field(
                format!("messages[{message_index}].content[*].reasoning.encrypted_content"),
                "Anthropic request replay only carries one redacted thinking payload per assistant message",
            );
        }

        message.metadata.custom.insert(
            "anthropic_redacted_thinking_data".to_string(),
            Value::String(first),
        );
        report.record_carried_provider_metadata(
            "openai.reasoning_encrypted_content",
            "OpenAI encrypted reasoning payload was carried into Anthropic redacted thinking metadata",
        );
    }
}

fn apply_openai_parallel_tool_policy(request: &mut ChatRequest) {
    let Some(false) = openai_parallel_tool_calls(request) else {
        return;
    };

    set_anthropic_option(request, "disableParallelToolUse", Value::Bool(true));
}

fn apply_openai_reasoning_effort_policy(request: &mut ChatRequest, report: &mut BridgeReport) {
    let Some(effort) = openai_reasoning_effort(request) else {
        return;
    };

    let mapped = match effort {
        "none" => {
            report.record_dropped_field(
                "provider_options_map.openai.reasoning_effort",
                "Anthropic does not expose an exact equivalent for OpenAI reasoning_effort=none",
            );
            return;
        }
        "minimal" => {
            report.record_lossy_field(
                "provider_options_map.openai.reasoning_effort",
                "OpenAI reasoning_effort=minimal was approximated to Anthropic effort=low",
            );
            "low"
        }
        "xhigh" => {
            report.record_lossy_field(
                "provider_options_map.openai.reasoning_effort",
                "OpenAI reasoning_effort=xhigh was approximated to Anthropic effort=high",
            );
            "high"
        }
        "low" => "low",
        "medium" => "medium",
        "high" => "high",
        _ => return,
    };

    set_anthropic_option(request, "effort", Value::String(mapped.to_string()));
}

fn apply_mcp_servers_overlay(request: &ChatRequest, body: &mut Value) {
    let Some(servers) = anthropic_option(request, "mcpServers")
        .or_else(|| anthropic_option(request, "mcp_servers"))
        .and_then(|value| value.as_array())
    else {
        return;
    };

    let normalized: Vec<Value> = servers.iter().filter_map(normalize_mcp_server).collect();

    if normalized.is_empty() {
        return;
    }

    body["mcp_servers"] = Value::Array(normalized);
}

fn normalize_mcp_server(value: &Value) -> Option<Value> {
    let obj = value.as_object()?;
    let mut out = Map::new();

    for (key, value) in obj {
        let normalized_key = match key.as_str() {
            "serverName" => "name",
            "serverUrl" => "url",
            "authorizationToken" => "authorization_token",
            "toolConfiguration" => "tool_configuration",
            "allowedTools" => "allowed_tools",
            other => other,
        };

        if normalized_key == "tool_configuration" && value.is_object() {
            let mut tool_configuration = Map::new();
            if let Some(inner) = value.as_object() {
                for (inner_key, inner_value) in inner {
                    let normalized_inner = match inner_key.as_str() {
                        "allowedTools" => "allowed_tools",
                        other => other,
                    };
                    tool_configuration.insert(normalized_inner.to_string(), inner_value.clone());
                }
            }
            out.insert(
                "tool_configuration".to_string(),
                Value::Object(tool_configuration),
            );
            continue;
        }

        out.insert(normalized_key.to_string(), value.clone());
    }

    Some(Value::Object(out))
}

fn apply_effort_overlay(request: &ChatRequest, body: &mut Value) {
    let Some(effort) = anthropic_option(request, "effort").and_then(|value| value.as_str()) else {
        return;
    };

    body["output_config"] = json!({
        "effort": effort,
    });
}

fn set_anthropic_option(request: &mut ChatRequest, key: &str, value: Value) {
    let mut anthropic = request
        .provider_options_map
        .get("anthropic")
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    anthropic.insert(key.to_string(), value);
    request
        .provider_options_map
        .insert("anthropic", Value::Object(anthropic));
}

fn anthropic_option<'a>(request: &'a ChatRequest, key: &str) -> Option<&'a Value> {
    request
        .provider_options_map
        .get("anthropic")?
        .as_object()?
        .get(key)
}

fn openai_instruction_text(request: &ChatRequest) -> Option<String> {
    let root = request.provider_options_map.get("openai")?.as_object()?;

    root.get("instructions")
        .and_then(|value| value.as_str())
        .map(str::to_string)
        .or_else(|| {
            root.get("responsesApi")
                .or_else(|| root.get("responses_api"))
                .and_then(|value| value.as_object())
                .and_then(|value| value.get("instructions"))
                .and_then(|value| value.as_str())
                .map(str::to_string)
        })
}

fn openai_response_format_value<'a>(request: &'a ChatRequest) -> Option<&'a Value> {
    let root = request.provider_options_map.get("openai")?.as_object()?;

    root.get("responseFormat")
        .or_else(|| root.get("response_format"))
        .or_else(|| {
            root.get("responsesApi")
                .or_else(|| root.get("responses_api"))
                .and_then(|value| value.as_object())
                .and_then(|value| {
                    value
                        .get("responseFormat")
                        .or_else(|| value.get("response_format"))
                })
        })
}

fn openai_parallel_tool_calls(request: &ChatRequest) -> Option<bool> {
    let root = request.provider_options_map.get("openai")?.as_object()?;

    root.get("parallelToolCalls")
        .or_else(|| root.get("parallel_tool_calls"))
        .and_then(|value| value.as_bool())
        .or_else(|| {
            root.get("responsesApi")
                .or_else(|| root.get("responses_api"))
                .and_then(|value| value.as_object())
                .and_then(|value| {
                    value
                        .get("parallelToolCalls")
                        .or_else(|| value.get("parallel_tool_calls"))
                })
                .and_then(|value| value.as_bool())
        })
}

fn openai_reasoning_effort<'a>(request: &'a ChatRequest) -> Option<&'a str> {
    let root = request.provider_options_map.get("openai")?.as_object()?;
    root.get("reasoningEffort")
        .or_else(|| root.get("reasoning_effort"))
        .and_then(|value| value.as_str())
}

fn tool_name(tool: &Tool) -> Option<&str> {
    match tool {
        Tool::Function { function } => Some(function.name.as_str()),
        Tool::ProviderDefined(provider_tool) => Some(provider_tool.name.as_str()),
    }
}
