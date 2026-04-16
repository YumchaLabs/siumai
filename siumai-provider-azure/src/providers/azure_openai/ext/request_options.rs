use crate::types::ChatRequest;

fn merge_values(base: &mut serde_json::Value, override_value: serde_json::Value) {
    match (base, override_value) {
        (serde_json::Value::Object(base_obj), serde_json::Value::Object(override_obj)) => {
            for (key, value) in override_obj {
                if let Some(existing) = base_obj.get_mut(&key) {
                    merge_values(existing, value);
                } else {
                    base_obj.insert(key, value);
                }
            }
        }
        (base_slot, value) => {
            *base_slot = value;
        }
    }
}

fn is_responses_api_field(key: &str) -> bool {
    matches!(
        key,
        "conversation"
            | "previous_response_id"
            | "background"
            | "include"
            | "instructions"
            | "max_tool_calls"
            | "reasoning_summary"
            | "truncation"
    )
}

fn normalize_azure_option_key(key: &str) -> Option<&'static str> {
    Some(match key {
        "responsesApi" => "responses_api",
        "providerTools" => "provider_tools",
        "logitBias" => "logit_bias",
        "reasoningEffort" => "reasoning_effort",
        "serviceTier" => "service_tier",
        "webSearchOptions" => "web_search_options",
        "conversation" => "conversation",
        "previousResponseId" => "previous_response_id",
        "promptCacheKey" => "prompt_cache_key",
        "promptCacheRetention" => "prompt_cache_retention",
        "responseFormat" => "response_format",
        "strictJsonSchema" => "strict_json_schema",
        "background" => "background",
        "include" => "include",
        "instructions" => "instructions",
        "maxToolCalls" => "max_tool_calls",
        "logprobs" => "logprobs",
        "reasoningSummary" => "reasoning_summary",
        "safetyIdentifier" => "safety_identifier",
        "store" => "store",
        "truncation" => "truncation",
        "parallelToolCalls" => "parallel_tool_calls",
        "textVerbosity" => "text_verbosity",
        "maxCompletionTokens" => "max_completion_tokens",
        "metadata" => "metadata",
        "user" => "user",
        "prediction" => "prediction",
        "forceReasoning" => "force_reasoning",
        _ => return None,
    })
}

fn normalize_azure_provider_options_json(value: &serde_json::Value) -> serde_json::Value {
    fn inner(value: &serde_json::Value, parent_key: Option<&str>) -> serde_json::Value {
        match value {
            serde_json::Value::Object(map) => {
                let is_top = parent_key.is_none();
                let mut out = serde_json::Map::new();
                let mut responses_overlay = serde_json::Map::new();

                for (key, value) in map {
                    let normalized_key = normalize_azure_option_key(key).unwrap_or(key);
                    let normalized_value = inner(value, Some(normalized_key));

                    if is_top
                        && normalized_key != "responses_api"
                        && is_responses_api_field(normalized_key)
                    {
                        responses_overlay.insert(normalized_key.to_string(), normalized_value);
                    } else if let Some(existing) = out.get_mut(normalized_key) {
                        merge_values(existing, normalized_value);
                    } else {
                        out.insert(normalized_key.to_string(), normalized_value);
                    }
                }

                if is_top && !responses_overlay.is_empty() {
                    let mut responses_api = match out.remove("responses_api") {
                        Some(serde_json::Value::Object(obj)) => obj,
                        Some(other) => {
                            out.insert("responses_api".to_string(), other);
                            serde_json::Map::new()
                        }
                        None => serde_json::Map::new(),
                    };

                    for (key, value) in responses_overlay {
                        responses_api.entry(key).or_insert(value);
                    }

                    if !responses_api.contains_key("enabled") {
                        responses_api.insert("enabled".to_string(), serde_json::Value::Bool(true));
                    }

                    out.insert(
                        "responses_api".to_string(),
                        serde_json::Value::Object(responses_api),
                    );
                }

                if parent_key == Some("responses_api") && !out.contains_key("enabled") {
                    out.insert("enabled".to_string(), serde_json::Value::Bool(true));
                }

                serde_json::Value::Object(out)
            }
            serde_json::Value::Array(values) => serde_json::Value::Array(
                values
                    .iter()
                    .map(|value| inner(value, parent_key))
                    .collect(),
            ),
            other => other.clone(),
        }
    }

    inner(value, None)
}

fn merge_provider_option_value(
    mut request: ChatRequest,
    provider_id: &str,
    value: serde_json::Value,
) -> ChatRequest {
    let mut overrides = crate::types::ProviderOptionsMap::new();
    overrides.insert(provider_id, value);
    request.provider_options_map.merge_overrides(overrides);
    request
}

/// Azure request option helpers for `ChatRequest`.
pub trait AzureOpenAiChatRequestExt {
    /// Convenience: attach Azure-specific options to `provider_options_map["azure"]`.
    fn with_azure_options<T: serde::Serialize>(self, options: T) -> Self;
}

impl AzureOpenAiChatRequestExt for ChatRequest {
    fn with_azure_options<T: serde::Serialize>(self, options: T) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        let value = normalize_azure_provider_options_json(&value);
        merge_provider_option_value(self, "azure", value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider_options::{
        AzureOpenAiOptions, AzureReasoningEffort, OpenAILanguageModelResponsesOptions,
    };
    use crate::types::ChatMessage;

    #[test]
    fn chat_request_ext_attaches_azure_options() {
        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()]).with_azure_options(
            AzureOpenAiOptions::new()
                .with_force_reasoning(true)
                .with_reasoning_effort(AzureReasoningEffort::High),
        );

        let value = request
            .provider_options_map
            .get("azure")
            .expect("azure options present");
        assert_eq!(value["force_reasoning"], serde_json::json!(true));
        assert_eq!(value["reasoning_effort"], serde_json::json!("high"));
    }

    #[test]
    fn chat_request_ext_merges_existing_azure_options_recursively() {
        let typed: OpenAILanguageModelResponsesOptions =
            serde_json::from_value(serde_json::json!({
                "reasoningEffort": "high",
                "responsesApi": {
                    "reasoningSummary": "detailed"
                }
            }))
            .expect("deserialize responses alias");

        let request = ChatRequest::new(vec![ChatMessage::user("hi").build()])
            .with_provider_option(
                "azure",
                serde_json::json!({
                    "existing": true,
                    "responses_api": {
                        "custom_nested": true
                    }
                }),
            )
            .with_azure_options(typed);

        let value = request
            .provider_options_map
            .get("azure")
            .expect("azure options present");
        assert_eq!(value["existing"], serde_json::json!(true));
        assert_eq!(value["reasoning_effort"], serde_json::json!("high"));
        assert_eq!(
            value["responses_api"]["custom_nested"],
            serde_json::json!(true)
        );
        assert_eq!(value["responses_api"]["enabled"], serde_json::json!(true));
        assert_eq!(
            value["responses_api"]["reasoning_summary"],
            serde_json::json!("detailed")
        );
    }
}
