use crate::types::ChatRequest;

/// Anthropic request option helpers for `ChatRequest`.
///
/// This is a provider-owned extension trait so `siumai-core` stays provider-agnostic.
pub trait AnthropicChatRequestExt {
    /// Convenience: attach Anthropic-specific options to `provider_options_map["anthropic"]`.
    fn with_anthropic_options<T: serde::Serialize>(self, options: T) -> Self;
}

fn denormalize_anthropic_option_key(key: &str) -> Option<&'static str> {
    Some(match key {
        "prompt_caching" => "promptCaching",
        "thinking_mode" => "thinkingMode",
        "budget_tokens" => "budgetTokens",
        "send_reasoning" => "sendReasoning",
        "response_format" => "responseFormat",
        "structured_output_mode" => "structuredOutputMode",
        "disable_parallel_tool_use" => "disableParallelToolUse",
        "cache_control" => "cacheControl",
        "user_id" => "userId",
        "mcp_servers" => "mcpServers",
        "allowed_tools" => "allowedTools",
        "authorization_token" => "authorizationToken",
        "tool_configuration" => "toolConfiguration",
        "context_management" => "contextManagement",
        "pause_after_compaction" => "pauseAfterCompaction",
        "tool_streaming" => "toolStreaming",
        "anthropic_beta" => "anthropicBeta",
        _ => return None,
    })
}

fn denormalize_anthropic_options_json(value: &serde_json::Value) -> serde_json::Value {
    fn inner(value: &serde_json::Value) -> Option<serde_json::Value> {
        match value {
            serde_json::Value::Null => None,
            serde_json::Value::Object(map) => {
                let mut out = serde_json::Map::new();
                for (key, value) in map {
                    if let Some(value) = inner(value) {
                        let key = denormalize_anthropic_option_key(key).unwrap_or(key);
                        out.insert(key.to_string(), value);
                    }
                }
                Some(serde_json::Value::Object(out))
            }
            serde_json::Value::Array(values) => Some(serde_json::Value::Array(
                values.iter().filter_map(inner).collect(),
            )),
            other => Some(other.clone()),
        }
    }

    inner(value).unwrap_or(serde_json::Value::Null)
}

pub(crate) fn merge_anthropic_provider_option_object(
    map: &mut crate::types::ProviderOptionsMap,
    value: serde_json::Value,
) {
    if let serde_json::Value::Object(new_options) = value {
        let mut merged = map
            .get("anthropic")
            .and_then(|value| value.as_object())
            .cloned()
            .unwrap_or_default();

        for (key, value) in new_options {
            merged.insert(key, value);
        }

        map.insert("anthropic", serde_json::Value::Object(merged));
    } else {
        map.insert("anthropic", value);
    }
}

impl AnthropicChatRequestExt for ChatRequest {
    fn with_anthropic_options<T: serde::Serialize>(mut self, options: T) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        let value = denormalize_anthropic_options_json(&value);
        merge_anthropic_provider_option_object(&mut self.provider_options_map, value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider_options::anthropic::{
        AnthropicOptions, AnthropicToolAllowedCaller, AnthropicToolOptions,
    };
    use crate::types::ChatMessage;

    fn source_section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
        let start_index = source.find(start).expect("section start marker");
        let end_index = source[start_index..]
            .find(end)
            .map(|offset| start_index + offset)
            .expect("section end marker");
        &source[start_index..end_index]
    }

    #[test]
    fn anthropic_request_option_extension_source_does_not_read_response_metadata() {
        let source = include_str!("request_options.rs");
        let request_source =
            source_section(source, "pub trait AnthropicChatRequestExt", "#[cfg(test)]");

        for disallowed in ["provider_metadata", "ProviderMetadata", "ContentPart::"] {
            assert!(
                !request_source.contains(disallowed),
                "Anthropic request option extension helpers must stay request-only"
            );
        }
    }

    #[test]
    fn with_anthropic_options_merges_existing_provider_options() {
        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_provider_option(
                "anthropic",
                serde_json::json!({
                    "existing": true,
                    "sendReasoning": false
                }),
            )
            .with_anthropic_options(AnthropicOptions::new().with_send_reasoning(true));

        let options = request
            .provider_option("anthropic")
            .and_then(|value| value.as_object())
            .expect("anthropic provider options");

        assert_eq!(options.get("existing"), Some(&serde_json::json!(true)));
        assert_eq!(options.get("sendReasoning"), Some(&serde_json::json!(true)));
    }

    #[test]
    fn merge_helper_supports_tool_options_shape() {
        let mut map = crate::types::ProviderOptionsMap::new();
        map.insert("anthropic", serde_json::json!({ "existing": true }));

        merge_anthropic_provider_option_object(
            &mut map,
            serde_json::to_value(
                AnthropicToolOptions::new()
                    .with_defer_loading(true)
                    .with_allowed_callers([AnthropicToolAllowedCaller::Direct]),
            )
            .expect("serialize anthropic tool options"),
        );

        let options = map
            .get("anthropic")
            .and_then(|value| value.as_object())
            .expect("anthropic provider options");
        assert_eq!(options.get("existing"), Some(&serde_json::json!(true)));
        assert_eq!(options.get("deferLoading"), Some(&serde_json::json!(true)));
        assert_eq!(
            options.get("allowedCallers"),
            Some(&serde_json::json!(["direct"]))
        );
    }
}
