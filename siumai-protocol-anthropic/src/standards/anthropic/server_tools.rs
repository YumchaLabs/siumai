//! Anthropic server tool helpers (Vercel-aligned).
//!
//! "Server tools" are provider-hosted tools that Anthropic executes automatically (e.g. web search,
//! tool search, code execution). Anthropic streaming surfaces the provider-native tool name, while
//! Siumai/Vercel use a stable custom tool name in events.

use serde_json::Value;

/// Normalize Anthropic server tool names to stable custom tool names (Vercel-aligned).
pub fn normalize_server_tool_name(name_raw: &str) -> &str {
    match name_raw {
        "tool_search_tool_regex" | "tool_search_tool_bm25" => "tool_search",
        "text_editor_code_execution" | "bash_code_execution" | "code_execution" => "code_execution",
        other => other,
    }
}

/// Normalize Anthropic server tool input payloads to match Vercel's event shape.
pub fn normalize_server_tool_input(name_raw: &str, input: Value) -> Value {
    match name_raw {
        "text_editor_code_execution" | "bash_code_execution" | "code_execution" => {
            wrap_code_execution_input(name_raw, input)
        }
        _ => input,
    }
}

fn wrap_code_execution_input(name_raw: &str, input: Value) -> Value {
    let mut obj = serde_json::Map::new();
    let kind = if name_raw == "code_execution" {
        "programmatic-tool-call"
    } else {
        name_raw
    };
    obj.insert("type".to_string(), serde_json::json!(kind));

    if let Value::Object(m) = input {
        for (k, v) in m {
            obj.insert(k, v);
        }
    }

    Value::Object(obj)
}
