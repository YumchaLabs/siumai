use serde_json::{Map, Value, json};
use siumai_core::bridge::BridgeReport;
use siumai_core::types::{ChatRequest, ProviderDefinedTool};

pub(crate) const ANTHROPIC_MCP_SERVER_CHOICE_ALIAS: &str = "__anthropic_mcp_server__";

pub(crate) fn openai_mcp_choice_aliases(provider_tool: &ProviderDefinedTool) -> Vec<String> {
    vec![provider_tool.name.clone(), "mcp".to_string()]
}

pub(crate) fn is_anthropic_mcp_server_choice_alias(alias: &str) -> bool {
    alias == ANTHROPIC_MCP_SERVER_CHOICE_ALIAS
}

pub(crate) fn map_openai_mcp_server(
    index: usize,
    provider_tool: &ProviderDefinedTool,
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

pub(crate) fn apply_anthropic_mcp_servers_option(request: &mut ChatRequest, servers: Vec<Value>) {
    if servers.is_empty() {
        return;
    }

    let mut anthropic = request
        .provider_options_map
        .get("anthropic")
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    anthropic.insert("mcpServers".to_string(), Value::Array(servers));
    request
        .provider_options_map
        .insert("anthropic".to_string(), Value::Object(anthropic));
}

pub(crate) fn apply_anthropic_mcp_servers_overlay(request: &ChatRequest, body: &mut Value) {
    let Some(servers) = request
        .provider_options_map
        .get("anthropic")
        .and_then(|value| value.as_object())
        .and_then(|value| {
            value
                .get("mcpServers")
                .or_else(|| value.get("mcp_servers"))
                .and_then(|value| value.as_array())
        })
    else {
        return;
    };

    let normalized: Vec<Value> = servers
        .iter()
        .filter_map(normalize_anthropic_mcp_server)
        .collect();

    if normalized.is_empty() {
        return;
    }

    body["mcp_servers"] = Value::Array(normalized);
}

fn normalize_anthropic_mcp_server(value: &Value) -> Option<Value> {
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
