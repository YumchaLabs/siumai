//! Anthropic server tool helpers (Vercel-aligned).
//!
//! "Server tools" are provider-hosted tools that Anthropic executes automatically (e.g. web search,
//! tool search, code execution). Anthropic streaming surfaces the provider-native tool name, while
//! Siumai/Vercel use a stable custom tool name in events.

use crate::types::ToolResultOutput;
use serde_json::{Map, Value};

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

/// Normalize Anthropic server tool result block types to stable custom tool names.
pub fn normalize_server_tool_result_name(block_type: &str) -> &str {
    match block_type {
        "tool_search_tool_result" => "tool_search",
        "text_editor_code_execution_tool_result"
        | "bash_code_execution_tool_result"
        | "code_execution_tool_result" => "code_execution",
        other => other.strip_suffix("_tool_result").unwrap_or(other),
    }
}

/// Normalize Anthropic provider-hosted tool results to the unified/Vercel-aligned result shape.
pub fn normalize_server_tool_result(block_type: &str, raw_result: &Value) -> Option<(Value, bool)> {
    match block_type {
        "web_search_tool_result" => Some(normalize_web_search_tool_result(raw_result)),
        "web_fetch_tool_result" => Some(normalize_web_fetch_tool_result(raw_result)),
        "tool_search_tool_result" => Some(normalize_tool_search_tool_result(raw_result)),
        "code_execution_tool_result" => Some(normalize_code_execution_tool_result(raw_result)),
        _ => None,
    }
}

/// Resolve the Anthropic provider-hosted tool name to replay on the wire.
pub fn replay_server_tool_name(
    tool_name: &str,
    raw_server_tool_name: Option<&str>,
    normalized_input: Option<&Value>,
) -> String {
    match tool_name {
        "tool_search" | "code_execution" => raw_server_tool_name
            .map(ToString::to_string)
            .or_else(|| {
                if tool_name == "code_execution" {
                    infer_code_execution_server_tool_name_from_input(normalized_input)
                        .map(ToString::to_string)
                } else {
                    None
                }
            })
            .unwrap_or_else(|| tool_name.to_string()),
        _ => tool_name.to_string(),
    }
}

/// Resolve the Anthropic provider-hosted tool result block type to replay on the wire.
pub fn replay_server_tool_result_block_type(
    tool_name: &str,
    raw_server_tool_name: Option<&str>,
    output: &ToolResultOutput,
) -> String {
    match tool_name {
        "web_search" => "web_search_tool_result".to_string(),
        "web_fetch" => "web_fetch_tool_result".to_string(),
        "tool_search" => "tool_search_tool_result".to_string(),
        "code_execution" => {
            let raw_name = raw_server_tool_name
                .map(ToString::to_string)
                .or_else(|| infer_code_execution_server_tool_name_from_output(output));
            format!(
                "{}_tool_result",
                raw_name.unwrap_or_else(|| "code_execution".to_string())
            )
        }
        _ => "mcp_tool_result".to_string(),
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

fn infer_code_execution_server_tool_name_from_input(
    normalized_input: Option<&Value>,
) -> Option<&'static str> {
    let input = normalized_input?.as_object()?;
    let kind = input.get("type")?.as_str()?;
    match kind {
        "text_editor_code_execution" => Some("text_editor_code_execution"),
        "bash_code_execution" => Some("bash_code_execution"),
        "programmatic-tool-call" | "code_execution" => Some("code_execution"),
        _ => None,
    }
}

fn infer_code_execution_server_tool_name_from_output(output: &ToolResultOutput) -> Option<String> {
    let type_name = match output {
        ToolResultOutput::Json { value } | ToolResultOutput::ErrorJson { value } => {
            value.get("type").and_then(|v| v.as_str())?
        }
        _ => return None,
    };

    infer_code_execution_server_tool_name_from_type_name(type_name).map(ToString::to_string)
}

fn infer_code_execution_server_tool_name_from_type_name(type_name: &str) -> Option<&'static str> {
    if type_name == "code_execution_result" || type_name == "code_execution_tool_result_error" {
        Some("code_execution")
    } else if type_name.starts_with("bash_code_execution_") {
        Some("bash_code_execution")
    } else if type_name.starts_with("text_editor_code_execution_") {
        Some("text_editor_code_execution")
    } else {
        None
    }
}

fn normalize_web_search_tool_result(raw_result: &Value) -> (Value, bool) {
    if let Some(arr) = raw_result.as_array() {
        let normalized = arr
            .iter()
            .map(normalize_web_search_result_item)
            .collect::<Vec<_>>();
        (Value::Array(normalized), false)
    } else if let Some(obj) = raw_result.as_object() {
        let error_code = obj.get("error_code").cloned().unwrap_or(Value::Null);
        (
            serde_json::json!({
                "type": "web_search_tool_result_error",
                "errorCode": error_code,
            }),
            true,
        )
    } else {
        (
            serde_json::json!({
                "type": "web_search_tool_result_error",
                "errorCode": Value::Null,
            }),
            true,
        )
    }
}

fn normalize_web_search_result_item(item: &Value) -> Value {
    let Some(obj) = item.as_object() else {
        return item.clone();
    };

    let mut out = Map::new();
    out.insert(
        "type".to_string(),
        obj.get("type")
            .cloned()
            .unwrap_or_else(|| serde_json::json!("web_search_result")),
    );
    out.insert(
        "url".to_string(),
        obj.get("url").cloned().unwrap_or(Value::Null),
    );
    out.insert(
        "title".to_string(),
        obj.get("title").cloned().unwrap_or(Value::Null),
    );
    out.insert(
        "pageAge".to_string(),
        obj.get("page_age")
            .or_else(|| obj.get("pageAge"))
            .cloned()
            .unwrap_or(Value::Null),
    );
    out.insert(
        "encryptedContent".to_string(),
        obj.get("encrypted_content")
            .or_else(|| obj.get("encryptedContent"))
            .cloned()
            .unwrap_or(Value::Null),
    );

    for (key, value) in obj {
        if key != "type"
            && key != "url"
            && key != "title"
            && key != "page_age"
            && key != "pageAge"
            && key != "encrypted_content"
            && key != "encryptedContent"
        {
            out.insert(key.clone(), value.clone());
        }
    }

    Value::Object(out)
}

fn normalize_web_fetch_tool_result(raw_result: &Value) -> (Value, bool) {
    let Some(obj) = raw_result.as_object() else {
        return (raw_result.clone(), false);
    };

    let tpe = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
    if tpe == "web_fetch_result" {
        let url = obj.get("url").cloned().unwrap_or(Value::Null);
        let retrieved_at = obj
            .get("retrieved_at")
            .or_else(|| obj.get("retrievedAt"))
            .cloned()
            .unwrap_or(Value::Null);
        let content = obj.get("content").and_then(|v| v.as_object());

        let mut out_content = Map::new();
        if let Some(content) = content {
            if let Some(v) = content.get("type") {
                out_content.insert("type".to_string(), v.clone());
            }
            if let Some(v) = content.get("title") {
                out_content.insert("title".to_string(), v.clone());
            }
            if let Some(v) = content.get("citations") {
                out_content.insert("citations".to_string(), v.clone());
            }
            if let Some(source) = content.get("source").and_then(|v| v.as_object()) {
                let mut out_source = Map::new();
                if let Some(v) = source.get("type") {
                    out_source.insert("type".to_string(), v.clone());
                }
                if let Some(v) = source.get("media_type").or_else(|| source.get("mediaType")) {
                    out_source.insert("mediaType".to_string(), v.clone());
                }
                if let Some(v) = source.get("data") {
                    out_source.insert("data".to_string(), v.clone());
                }
                for (key, value) in source {
                    if key != "type" && key != "media_type" && key != "mediaType" && key != "data" {
                        out_source.insert(key.clone(), value.clone());
                    }
                }
                out_content.insert("source".to_string(), Value::Object(out_source));
            }
            for (key, value) in content {
                if key != "type" && key != "title" && key != "citations" && key != "source" {
                    out_content.insert(key.clone(), value.clone());
                }
            }
        }

        (
            serde_json::json!({
                "type": "web_fetch_result",
                "url": url,
                "retrievedAt": retrieved_at,
                "content": Value::Object(out_content),
            }),
            false,
        )
    } else if tpe == "web_fetch_tool_result_error" {
        let error_code = obj.get("error_code").cloned().unwrap_or(Value::Null);
        (
            serde_json::json!({
                "type": "web_fetch_tool_result_error",
                "errorCode": error_code,
            }),
            true,
        )
    } else {
        (raw_result.clone(), false)
    }
}

fn normalize_tool_search_tool_result(raw_result: &Value) -> (Value, bool) {
    let Some(obj) = raw_result.as_object() else {
        return (raw_result.clone(), false);
    };

    let tpe = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
    if tpe == "tool_search_tool_search_result" {
        let refs = obj
            .get("tool_references")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|v| v.as_object().cloned())
            .map(|ref_obj| {
                serde_json::json!({
                    "type": ref_obj
                        .get("type")
                        .cloned()
                        .unwrap_or_else(|| serde_json::json!("tool_reference")),
                    "toolName": ref_obj.get("tool_name").cloned().unwrap_or(Value::Null),
                })
            })
            .collect::<Vec<_>>();
        (Value::Array(refs), false)
    } else {
        let error_code = obj.get("error_code").cloned().unwrap_or(Value::Null);
        (
            serde_json::json!({
                "type": "tool_search_tool_result_error",
                "errorCode": error_code,
            }),
            true,
        )
    }
}

fn normalize_code_execution_tool_result(raw_result: &Value) -> (Value, bool) {
    let Some(obj) = raw_result.as_object() else {
        return (raw_result.clone(), false);
    };

    let tpe = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
    if tpe == "code_execution_result" {
        let mut out = serde_json::Map::new();
        out.insert(
            "type".to_string(),
            serde_json::json!("code_execution_result"),
        );
        out.insert(
            "stdout".to_string(),
            obj.get("stdout").cloned().unwrap_or(Value::Null),
        );
        out.insert(
            "stderr".to_string(),
            obj.get("stderr").cloned().unwrap_or(Value::Null),
        );
        out.insert(
            "return_code".to_string(),
            obj.get("return_code").cloned().unwrap_or(Value::Null),
        );
        if let Some(v) = obj.get("content") {
            out.insert("content".to_string(), v.clone());
        }
        (Value::Object(out), false)
    } else if tpe == "code_execution_tool_result_error" {
        let error_code = obj.get("error_code").cloned().unwrap_or(Value::Null);
        (
            serde_json::json!({
                "type": "code_execution_tool_result_error",
                "errorCode": error_code,
            }),
            true,
        )
    } else {
        (raw_result.clone(), false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_server_tool_result_names() {
        assert_eq!(
            normalize_server_tool_result_name("web_search_tool_result"),
            "web_search"
        );
        assert_eq!(
            normalize_server_tool_result_name("tool_search_tool_result"),
            "tool_search"
        );
        assert_eq!(
            normalize_server_tool_result_name("bash_code_execution_tool_result"),
            "code_execution"
        );
    }

    #[test]
    fn replays_code_execution_server_tool_name_from_metadata_or_input() {
        assert_eq!(
            replay_server_tool_name("code_execution", Some("bash_code_execution"), None),
            "bash_code_execution"
        );
        assert_eq!(
            replay_server_tool_name(
                "code_execution",
                None,
                Some(&serde_json::json!({ "type": "text_editor_code_execution" })),
            ),
            "text_editor_code_execution"
        );
        assert_eq!(
            replay_server_tool_name(
                "code_execution",
                None,
                Some(&serde_json::json!({ "type": "programmatic-tool-call" })),
            ),
            "code_execution"
        );
    }

    #[test]
    fn replays_code_execution_result_block_type_from_metadata_or_output() {
        assert_eq!(
            replay_server_tool_result_block_type(
                "code_execution",
                Some("bash_code_execution"),
                &ToolResultOutput::json(serde_json::json!({
                    "type": "code_execution_result"
                })),
            ),
            "bash_code_execution_tool_result"
        );
        assert_eq!(
            replay_server_tool_result_block_type(
                "code_execution",
                None,
                &ToolResultOutput::json(serde_json::json!({
                    "type": "text_editor_code_execution_create_result"
                })),
            ),
            "text_editor_code_execution_tool_result"
        );
    }

    #[test]
    fn normalizes_web_search_tool_result_shape() {
        let (value, is_error) = normalize_server_tool_result(
            "web_search_tool_result",
            &serde_json::json!([
                {
                    "type": "web_search_result",
                    "url": "https://example.com",
                    "title": "Example",
                    "page_age": "2 days ago",
                    "encrypted_content": "secret"
                }
            ]),
        )
        .expect("normalized result");

        assert!(!is_error);
        assert_eq!(value[0]["pageAge"], serde_json::json!("2 days ago"));
        assert_eq!(value[0]["encryptedContent"], serde_json::json!("secret"));
        assert!(value[0].get("page_age").is_none());
        assert!(value[0].get("encrypted_content").is_none());
    }

    #[test]
    fn normalizes_web_fetch_tool_result_shape() {
        let (value, is_error) = normalize_server_tool_result(
            "web_fetch_tool_result",
            &serde_json::json!({
                "type": "web_fetch_result",
                "url": "https://example.com",
                "retrieved_at": "2025-01-01T00:00:00Z",
                "content": {
                    "type": "document",
                    "title": "Example",
                    "source": {
                        "type": "text",
                        "media_type": "text/plain",
                        "data": "hello"
                    }
                }
            }),
        )
        .expect("normalized result");

        assert!(!is_error);
        assert_eq!(
            value["retrievedAt"],
            serde_json::json!("2025-01-01T00:00:00Z")
        );
        assert_eq!(
            value["content"]["source"]["mediaType"],
            serde_json::json!("text/plain")
        );
    }

    #[test]
    fn normalizes_tool_search_tool_result_shape() {
        let (value, is_error) = normalize_server_tool_result(
            "tool_search_tool_result",
            &serde_json::json!({
                "type": "tool_search_tool_search_result",
                "tool_references": [{ "type": "tool_reference", "tool_name": "get_weather" }]
            }),
        )
        .expect("normalized result");

        assert!(!is_error);
        assert!(value.is_array());
        assert_eq!(value[0]["toolName"], serde_json::json!("get_weather"));
    }

    #[test]
    fn normalizes_code_execution_tool_result_shape() {
        let (value, is_error) = normalize_server_tool_result(
            "code_execution_tool_result",
            &serde_json::json!({
                "type": "code_execution_result",
                "stdout": "2\n",
                "stderr": "",
                "return_code": 0
            }),
        )
        .expect("normalized result");

        assert!(!is_error);
        assert_eq!(value["type"], serde_json::json!("code_execution_result"));
        assert_eq!(value["return_code"], serde_json::json!(0));
    }
}
