//! Shared Anthropic request option normalization and body overlays.

use crate::types::ChatRequest;
use serde_json::{Map, Value, json};

/// Normalize known Anthropic provider-option keys into the snake_case shapes used by the HTTP API.
pub fn normalize_anthropic_provider_options_json(value: &Value) -> Value {
    fn normalize_key(key: &str) -> Option<&'static str> {
        Some(match key {
            "promptCaching" => "prompt_caching",
            "thinkingMode" => "thinking_mode",
            "responseFormat" => "response_format",
            "structuredOutputMode" => "structured_output_mode",
            "contextManagement" => "context_management",
            "toolStreaming" => "tool_streaming",
            "sendReasoning" => "send_reasoning",
            "disableParallelToolUse" => "disable_parallel_tool_use",
            "anthropicBeta" => "anthropic_beta",
            "expiresAt" => "expires_at",
            "containerId" => "container_id",
            "cacheControl" => "cache_control",
            "cacheType" => "cache_type",
            "messageIndex" => "message_index",
            "thinkingBudget" => "thinking_budget",
            "mcpServers" => "mcp_servers",
            "serverName" => "name",
            "serverUrl" => "url",
            "authorizationToken" => "authorization_token",
            "toolConfiguration" => "tool_configuration",
            "allowedTools" => "allowed_tools",
            "skillId" => "skill_id",
            "userId" => "user_id",
            "clearAtLeast" => "clear_at_least",
            "clearToolInputs" => "clear_tool_inputs",
            "excludeTools" => "exclude_tools",
            "pauseAfterCompaction" => "pause_after_compaction",
            _ => return None,
        })
    }

    fn inner(value: &Value) -> Value {
        match value {
            Value::Object(map) => {
                let mut out = Map::new();
                for (key, value) in map {
                    if key == "thinking"
                        && let Some(obj) = value.as_object()
                    {
                        let thinking_type = obj
                            .get("type")
                            .and_then(|value| value.as_str())
                            .unwrap_or("disabled");
                        let enabled = matches!(thinking_type, "enabled" | "adaptive");
                        let budget = obj
                            .get("budgetTokens")
                            .or_else(|| obj.get("budget_tokens"))
                            .and_then(|value| value.as_u64())
                            .and_then(|value| u32::try_from(value).ok());

                        let mut thinking_mode = Map::new();
                        thinking_mode.insert("type".to_string(), json!(thinking_type));
                        thinking_mode.insert("enabled".to_string(), Value::Bool(enabled));
                        if let Some(budget) = budget {
                            thinking_mode
                                .insert("thinking_budget".to_string(), serde_json::json!(budget));
                        }
                        out.insert("thinking_mode".to_string(), Value::Object(thinking_mode));
                        continue;
                    }

                    let normalized_key = normalize_key(key).unwrap_or(key);
                    out.insert(normalized_key.to_string(), inner(value));
                }
                Value::Object(out)
            }
            Value::Array(values) => Value::Array(values.iter().map(inner).collect()),
            other => other.clone(),
        }
    }

    inner(value)
}

/// Returns whether the Anthropic request body needs provider-option overlays or token capping.
pub fn anthropic_request_body_overlays_needed(req: &ChatRequest) -> bool {
    let max_output_tokens = known_max_output_tokens(&req.common_params.model);
    let needs_cap = max_output_tokens.is_some()
        && req
            .common_params
            .max_tokens
            .is_some_and(|max_tokens| max_tokens > max_output_tokens.unwrap_or(max_tokens));

    let Some(options) = normalized_anthropic_provider_options(req) else {
        return needs_cap;
    };

    needs_cap
        || thinking_overlay(&options).is_some()
        || response_format_overlay(&options).is_some()
        || cache_control_overlay(&options).is_some()
        || metadata_overlay(&options).is_some()
        || mcp_servers_overlay(&options).is_some()
        || container_overlay(&options).is_some()
        || context_management_overlay(&options).is_some()
        || effort_overlay(&options).is_some()
        || speed_overlay(&options).is_some()
}

/// Apply provider-option-driven Anthropic body overlays after the base request is built.
pub fn apply_anthropic_request_body_overlays(req: &ChatRequest, body: &mut Value) {
    if let Some(options) = normalized_anthropic_provider_options(req) {
        if let Some((thinking, _budget_tokens)) = thinking_overlay(&options) {
            body["thinking"] = thinking;
        }

        if let Some(output_format) = response_format_overlay(&options) {
            body["output_format"] = output_format;
        }

        if let Some(cache_control) = cache_control_overlay(&options) {
            body["cache_control"] = cache_control;
        }

        if let Some(metadata) = metadata_overlay(&options) {
            body["metadata"] = metadata;
        }

        if let Some(servers) = mcp_servers_overlay(&options) {
            body["mcp_servers"] = Value::Array(servers);
        }

        if let Some(container) = container_overlay(&options) {
            body["container"] = container;
        }

        if let Some(context_management) = context_management_overlay(&options) {
            body["context_management"] = context_management;
        }

        if let Some(effort) = effort_overlay(&options) {
            body["output_config"] = json!({
                "effort": effort,
            });
        }

        if let Some(speed) = speed_overlay(&options) {
            body["speed"] = speed;
        }
    }

    finalize_anthropic_thinking_body(req, body);
    cap_max_tokens_for_known_model(&req.common_params.model, body);
}

fn normalized_anthropic_provider_options(req: &ChatRequest) -> Option<Map<String, Value>> {
    let value = req.provider_options_map.get("anthropic")?;
    let normalized = normalize_anthropic_provider_options_json(value);
    normalized.as_object().cloned()
}

pub(crate) fn finalize_anthropic_thinking_body(req: &ChatRequest, body: &mut Value) {
    let Some(thinking) = body.get("thinking").and_then(|value| value.as_object()) else {
        return;
    };

    let thinking_type = thinking
        .get("type")
        .and_then(|value| value.as_str())
        .unwrap_or("disabled")
        .to_string();
    let budget_tokens = thinking
        .get("budget_tokens")
        .or_else(|| thinking.get("budgetTokens"))
        .and_then(|value| value.as_u64());

    if !matches!(thinking_type.as_str(), "enabled" | "adaptive") {
        return;
    }

    if let Some(obj) = body.as_object_mut() {
        obj.remove("temperature");
        obj.remove("top_p");
        obj.remove("top_k");
    }

    if thinking_type == "enabled" {
        let budget_tokens = budget_tokens.unwrap_or(1024);

        body["thinking"] = json!({
            "type": "enabled",
            "budget_tokens": budget_tokens,
        });

        let base_max_tokens = req
            .common_params
            .max_tokens
            .map(u64::from)
            .or_else(|| body.get("max_tokens").and_then(|value| value.as_u64()));
        if let Some(base_max_tokens) = base_max_tokens {
            body["max_tokens"] = json!(base_max_tokens.saturating_add(budget_tokens));
        }
    } else {
        body["thinking"] = json!({
            "type": "adaptive",
        });
    }
}

fn thinking_overlay(options: &Map<String, Value>) -> Option<(Value, Option<u64>)> {
    let thinking = options.get("thinking_mode")?.as_object()?;
    let thinking_type = thinking
        .get("type")
        .and_then(|value| value.as_str())
        .or_else(|| {
            thinking
                .get("enabled")
                .and_then(|value| value.as_bool())
                .map(|enabled| if enabled { "enabled" } else { "disabled" })
        })
        .unwrap_or("disabled");

    match thinking_type {
        "adaptive" => Some((json!({ "type": "adaptive" }), None)),
        "enabled" => {
            let budget_tokens = thinking
                .get("thinking_budget")
                .and_then(|value| value.as_u64())
                .unwrap_or(1024);

            Some((
                json!({
                    "type": "enabled",
                    "budget_tokens": budget_tokens,
                }),
                Some(budget_tokens),
            ))
        }
        _ => None,
    }
}

fn response_format_overlay(options: &Map<String, Value>) -> Option<Value> {
    let value = options.get("response_format")?;

    if let Some(kind) = value.as_str() {
        if is_json_object_kind(kind) {
            return Some(json!({ "type": "json_object" }));
        }
        return None;
    }

    let obj = value.as_object()?;

    if let Some(kind) = obj.get("type").and_then(|value| value.as_str()) {
        if is_json_object_kind(kind) {
            return Some(json!({ "type": "json_object" }));
        }
        if is_json_schema_kind(kind) {
            return json_schema_output_format(obj);
        }
    }

    if obj.contains_key("JsonObject")
        || obj.contains_key("jsonObject")
        || obj.contains_key("json_object")
        || obj.contains_key("json-object")
    {
        return Some(json!({ "type": "json_object" }));
    }

    let schema_value = obj
        .get("JsonSchema")
        .or_else(|| obj.get("jsonSchema"))
        .or_else(|| obj.get("json_schema"))
        .and_then(|value| value.as_object())?;

    json_schema_output_format(schema_value)
}

fn cache_control_overlay(options: &Map<String, Value>) -> Option<Value> {
    let value = options.get("cache_control")?.clone();
    if value.is_null() || value.as_object().is_some_and(|obj| obj.is_empty()) {
        None
    } else {
        Some(value)
    }
}

fn metadata_overlay(options: &Map<String, Value>) -> Option<Value> {
    let metadata = options.get("metadata")?.as_object()?;
    let mut out = Map::new();

    if let Some(user_id) = metadata
        .get("user_id")
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
    {
        out.insert("user_id".to_string(), json!(user_id));
    }

    if out.is_empty() {
        None
    } else {
        Some(Value::Object(out))
    }
}

fn json_schema_output_format(source: &Map<String, Value>) -> Option<Value> {
    let schema = source.get("schema")?.clone();
    let mut out = Map::new();
    out.insert("type".to_string(), json!("json_schema"));
    out.insert("schema".to_string(), schema);
    if let Some(strict) = source.get("strict").and_then(|value| value.as_bool()) {
        out.insert("strict".to_string(), Value::Bool(strict));
    }
    Some(Value::Object(out))
}

fn mcp_servers_overlay(options: &Map<String, Value>) -> Option<Vec<Value>> {
    let servers = options.get("mcp_servers")?.as_array()?;
    if servers.is_empty() {
        return None;
    }
    Some(servers.clone())
}

fn container_overlay(options: &Map<String, Value>) -> Option<Value> {
    let container = options
        .get("container")
        .or_else(|| options.get("container_id"))?;

    match container {
        Value::String(id) => {
            if id.trim().is_empty() {
                None
            } else {
                Some(Value::String(id.clone()))
            }
        }
        Value::Object(map) => {
            let id = map
                .get("id")
                .and_then(|value| value.as_str())
                .map(ToString::to_string);
            let skills = map.get("skills").and_then(|value| value.as_array());

            let normalized_skills: Vec<Value> = skills
                .into_iter()
                .flatten()
                .filter_map(|value| value.as_object())
                .map(|skill| {
                    let mut out = Map::new();
                    if let Some(value) = skill.get("type") {
                        out.insert("type".to_string(), value.clone());
                    }
                    let skill_id = skill
                        .get("skill_id")
                        .and_then(|value| value.as_str())
                        .map(ToString::to_string);
                    if let Some(skill_id) = skill_id {
                        out.insert("skill_id".to_string(), json!(skill_id));
                    }
                    if let Some(value) = skill.get("version") {
                        out.insert("version".to_string(), value.clone());
                    }
                    Value::Object(out)
                })
                .filter(|value| value.as_object().is_some_and(|obj| !obj.is_empty()))
                .collect();

            if normalized_skills.is_empty() && id.as_ref().is_some_and(|value| !value.is_empty()) {
                return Some(Value::String(id.unwrap()));
            }

            let mut out = Map::new();
            if let Some(id) = id
                && !id.is_empty()
            {
                out.insert("id".to_string(), Value::String(id));
            }
            if !normalized_skills.is_empty() {
                out.insert("skills".to_string(), Value::Array(normalized_skills));
            }

            if out.is_empty() {
                None
            } else {
                Some(Value::Object(out))
            }
        }
        _ => None,
    }
}

fn context_management_overlay(options: &Map<String, Value>) -> Option<Value> {
    let value = options.get("context_management")?.clone();
    if value.is_null() || value.as_object().is_some_and(|obj| obj.is_empty()) {
        None
    } else {
        Some(value)
    }
}

fn effort_overlay(options: &Map<String, Value>) -> Option<Value> {
    options.get("effort").cloned()
}

fn speed_overlay(options: &Map<String, Value>) -> Option<Value> {
    options.get("speed").cloned()
}

pub(crate) fn cap_max_tokens_for_known_model(model_id: &str, body: &mut Value) {
    let Some(max_output_tokens) = known_max_output_tokens(model_id) else {
        return;
    };

    if let Some(max_tokens) = body.get("max_tokens").and_then(|value| value.as_u64())
        && max_tokens > max_output_tokens as u64
    {
        body["max_tokens"] = json!(max_output_tokens);
    }
}

fn known_max_output_tokens(model_id: &str) -> Option<u32> {
    if !model_id.starts_with("claude-") {
        return None;
    }

    Some(match model_id {
        id if id.starts_with("claude-sonnet-4-5")
            || id.starts_with("claude-opus-4-5")
            || id.starts_with("claude-haiku-4-5") =>
        {
            64_000
        }
        id if id.contains("claude-opus-4") || id.contains("claude-sonnet-4") => 32_000,
        id if id.contains("claude-3-7-sonnet") => 64_000,
        id if id.contains("claude-3-5") => 8192,
        id if id.contains("claude-3") => 4096,
        _ => 8192,
    })
}

fn is_json_object_kind(value: &str) -> bool {
    matches!(
        value,
        "JsonObject" | "jsonObject" | "json_object" | "json-object"
    )
}

fn is_json_schema_kind(value: &str) -> bool {
    matches!(
        value,
        "JsonSchema" | "jsonSchema" | "json_schema" | "json-schema"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_provider_options_maps_extended_anthropic_request_keys() {
        let normalized = normalize_anthropic_provider_options_json(&json!({
            "thinking": {
                "type": "adaptive"
            },
            "metadata": {
                "userId": "user-1"
            },
            "contextManagement": {
                "edits": [{
                    "type": "compact_20260112",
                    "pauseAfterCompaction": true
                }]
            },
            "anthropicBeta": ["beta-1"],
            "disableParallelToolUse": true,
            "sendReasoning": false
        }));

        assert_eq!(normalized["thinking_mode"]["type"], json!("adaptive"));
        assert_eq!(normalized["thinking_mode"]["enabled"], json!(true));
        assert_eq!(normalized["metadata"]["user_id"], json!("user-1"));
        assert_eq!(
            normalized["context_management"]["edits"][0]["pause_after_compaction"],
            json!(true)
        );
        assert_eq!(normalized["anthropic_beta"], json!(["beta-1"]));
        assert_eq!(normalized["disable_parallel_tool_use"], json!(true));
        assert_eq!(normalized["send_reasoning"], json!(false));
    }

    #[test]
    fn apply_body_overlays_preserves_adaptive_thinking_and_request_metadata() {
        let req = ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
            .with_provider_option(
                "anthropic",
                json!({
                    "thinking": {
                        "type": "adaptive"
                    },
                    "cacheControl": {
                        "type": "ephemeral",
                        "ttl": "1h"
                    },
                    "metadata": {
                        "userId": "user-1"
                    },
                    "speed": "fast"
                }),
            );

        let mut body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [],
            "max_tokens": 2000,
            "temperature": 0.5,
            "top_p": 0.8
        });

        apply_anthropic_request_body_overlays(&req, &mut body);

        assert_eq!(body["thinking"], json!({ "type": "adaptive" }));
        assert_eq!(body["max_tokens"], json!(2000));
        assert!(body.get("temperature").is_none());
        assert!(body.get("top_p").is_none());
        assert_eq!(
            body["cache_control"],
            json!({
                "type": "ephemeral",
                "ttl": "1h"
            })
        );
        assert_eq!(
            body["metadata"],
            json!({
                "user_id": "user-1"
            })
        );
        assert_eq!(body["speed"], json!("fast"));
    }

    #[test]
    fn apply_body_overlays_recompute_enabled_thinking_max_tokens_from_request_input() {
        let mut req = ChatRequest::new(vec![crate::types::ChatMessage::user("hi").build()])
            .with_provider_option(
                "anthropic",
                json!({
                    "thinking": {
                        "type": "enabled",
                        "budgetTokens": 1000
                    }
                }),
            );
        req.common_params.model = "claude-sonnet-4-5".to_string();
        req.common_params.max_tokens = Some(2000);
        req.common_params.temperature = Some(0.5);
        req.common_params.top_p = Some(0.8);

        let mut body = json!({
            "model": "claude-sonnet-4-5",
            "messages": [],
            "max_tokens": 9000,
            "temperature": 0.5,
            "top_p": 0.8,
            "thinking": {
                "type": "enabled",
                "budget_tokens": 5000
            }
        });

        apply_anthropic_request_body_overlays(&req, &mut body);

        assert_eq!(
            body["thinking"],
            json!({
                "type": "enabled",
                "budget_tokens": 1000
            })
        );
        assert_eq!(body["max_tokens"], json!(3000));
        assert!(body.get("temperature").is_none());
        assert!(body.get("top_p").is_none());
    }
}
