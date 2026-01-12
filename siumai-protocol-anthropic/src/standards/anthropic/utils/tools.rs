use super::*;

pub fn convert_tools_to_anthropic_format(
    tools: &[crate::types::Tool],
) -> Result<Vec<serde_json::Value>, LlmError> {
    let mut anthropic_tools = Vec::new();
    let mut cache_control_breakpoints: usize = 0;

    for tool in tools {
        match tool {
            crate::types::Tool::Function { function } => {
                let mut tool_map = serde_json::Map::new();
                tool_map.insert("name".to_string(), serde_json::json!(function.name));
                if !function.description.is_empty() {
                    tool_map.insert(
                        "description".to_string(),
                        serde_json::json!(function.description),
                    );
                }
                tool_map.insert(
                    "input_schema".to_string(),
                    serde_json::json!(function.parameters),
                );

                let mut anthropic_tool = serde_json::Value::Object(tool_map);

                if let Some(strict) = function.strict
                    && let Some(map) = anthropic_tool.as_object_mut()
                {
                    map.insert("strict".to_string(), serde_json::json!(strict));
                }

                // Vercel-aligned: tool-level provider options for Anthropic.
                // Example: `{ providerOptions: { anthropic: { deferLoading: true } } }`
                if let Some(opts) = function.provider_options_map.get("anthropic")
                    && let Some(obj) = opts.as_object()
                    && let Some(v) = obj
                        .get("deferLoading")
                        .or_else(|| obj.get("defer_loading"))
                        .and_then(|v| v.as_bool())
                    && let Some(map) = anthropic_tool.as_object_mut()
                {
                    map.insert("defer_loading".to_string(), serde_json::json!(v));
                }

                // Vercel-aligned: tool-level cache control for Anthropic.
                // Example: `{ providerOptions: { anthropic: { cacheControl: { type: "ephemeral" } } } }`
                if let Some(opts) = function.provider_options_map.get("anthropic")
                    && let Some(obj) = opts.as_object()
                    && let Some(cache) = obj
                        .get("cacheControl")
                        .or_else(|| obj.get("cache_control"))
                        .and_then(|v| v.as_object())
                    && let Some(map) = anthropic_tool.as_object_mut()
                    && cache_control_breakpoints < 4
                {
                    map.insert(
                        "cache_control".to_string(),
                        serde_json::Value::Object(cache.clone()),
                    );
                    cache_control_breakpoints += 1;
                }

                // Vercel-aligned: allowed callers for programmatic tool calling.
                // Example: `{ providerOptions: { anthropic: { allowedCallers: ["code_execution_20250825"] } } }`
                if let Some(opts) = function.provider_options_map.get("anthropic")
                    && let Some(obj) = opts.as_object()
                    && let Some(allowed) = obj
                        .get("allowedCallers")
                        .or_else(|| obj.get("allowed_callers"))
                        .and_then(|v| v.as_array())
                    && let Some(map) = anthropic_tool.as_object_mut()
                {
                    let values: Vec<serde_json::Value> = allowed
                        .iter()
                        .filter_map(|v| {
                            v.as_str().map(|s| serde_json::Value::String(s.to_string()))
                        })
                        .collect();
                    if !values.is_empty() {
                        map.insert(
                            "allowed_callers".to_string(),
                            serde_json::Value::Array(values),
                        );
                    }
                }

                // Vercel-aligned: tool input examples.
                if let Some(examples) = function.input_examples.as_ref()
                    && !examples.is_empty()
                    && let Some(map) = anthropic_tool.as_object_mut()
                {
                    let out: Vec<serde_json::Value> = examples
                        .iter()
                        .map(|v| {
                            if let Some(obj) = v.as_object()
                                && let Some(input) = obj.get("input")
                            {
                                return input.clone();
                            }
                            v.clone()
                        })
                        .collect();
                    map.insert("input_examples".to_string(), serde_json::Value::Array(out));
                }

                anthropic_tools.push(anthropic_tool);
            }
            crate::types::Tool::ProviderDefined(provider_tool) => {
                // Check if this is an Anthropic provider-defined tool
                if provider_tool.provider() == Some("anthropic") {
                    let Some(spec) = crate::tools::anthropic::server_tool_spec(&provider_tool.id)
                    else {
                        continue;
                    };

                    let tool_type = provider_tool.tool_type().unwrap_or("unknown");

                    // Vercel alignment:
                    // - provider tool args live in SDK-shaped camelCase (e.g., maxUses),
                    //   while Anthropic Messages API expects snake_case fields in the tool object.
                    // - Accept both shapes for backward compatibility.
                    let mut anthropic_tool = serde_json::json!({
                        "type": spec.tool_type,
                        "name": spec.tool_name,
                    });

                    if let serde_json::Value::Object(args_map) = &provider_tool.args
                        && let serde_json::Value::Object(tool_map) = &mut anthropic_tool
                    {
                        match tool_type {
                            "computer_20241022" | "computer_20250124" => {
                                if let Some(v) = args_map
                                    .get("displayWidthPx")
                                    .or_else(|| args_map.get("display_width_px"))
                                {
                                    tool_map.insert("display_width_px".to_string(), v.clone());
                                }
                                if let Some(v) = args_map
                                    .get("displayHeightPx")
                                    .or_else(|| args_map.get("display_height_px"))
                                {
                                    tool_map.insert("display_height_px".to_string(), v.clone());
                                }
                                if let Some(v) = args_map
                                    .get("displayNumber")
                                    .or_else(|| args_map.get("display_number"))
                                {
                                    tool_map.insert("display_number".to_string(), v.clone());
                                }
                            }
                            "text_editor_20250728" => {
                                if let Some(v) = args_map
                                    .get("maxCharacters")
                                    .or_else(|| args_map.get("max_characters"))
                                {
                                    tool_map.insert("max_characters".to_string(), v.clone());
                                }
                            }
                            "web_fetch_20250910" => {
                                if let Some(v) =
                                    args_map.get("maxUses").or_else(|| args_map.get("max_uses"))
                                {
                                    tool_map.insert("max_uses".to_string(), v.clone());
                                }
                                if let Some(v) = args_map
                                    .get("allowedDomains")
                                    .or_else(|| args_map.get("allowed_domains"))
                                {
                                    tool_map.insert("allowed_domains".to_string(), v.clone());
                                }
                                if let Some(v) = args_map
                                    .get("blockedDomains")
                                    .or_else(|| args_map.get("blocked_domains"))
                                {
                                    tool_map.insert("blocked_domains".to_string(), v.clone());
                                }
                                if let Some(v) = args_map.get("citations") {
                                    tool_map.insert("citations".to_string(), v.clone());
                                }
                                if let Some(v) = args_map
                                    .get("maxContentTokens")
                                    .or_else(|| args_map.get("max_content_tokens"))
                                {
                                    tool_map.insert("max_content_tokens".to_string(), v.clone());
                                }
                            }
                            "web_search_20250305" => {
                                if let Some(v) =
                                    args_map.get("maxUses").or_else(|| args_map.get("max_uses"))
                                {
                                    tool_map.insert("max_uses".to_string(), v.clone());
                                }
                                if let Some(v) = args_map
                                    .get("allowedDomains")
                                    .or_else(|| args_map.get("allowed_domains"))
                                {
                                    tool_map.insert("allowed_domains".to_string(), v.clone());
                                }
                                if let Some(v) = args_map
                                    .get("blockedDomains")
                                    .or_else(|| args_map.get("blocked_domains"))
                                {
                                    tool_map.insert("blocked_domains".to_string(), v.clone());
                                }
                                if let Some(v) = args_map
                                    .get("userLocation")
                                    .or_else(|| args_map.get("user_location"))
                                {
                                    tool_map.insert("user_location".to_string(), v.clone());
                                }
                            }
                            "memory_20250818" => {
                                for (k, v) in args_map {
                                    tool_map.insert(k.clone(), v.clone());
                                }
                            }
                            _ => {}
                        }
                    }

                    anthropic_tools.push(anthropic_tool);
                } else {
                    // Ignore provider-defined tools from other providers
                    // This allows users to mix tools for different providers
                    continue;
                }
            }
        }
    }

    Ok(anthropic_tools)
}

#[cfg(test)]
mod provider_tool_tests {
    use super::*;

    #[test]
    fn maps_anthropic_provider_defined_web_search() {
        let t = crate::tools::anthropic::web_search_20250305().with_args(serde_json::json!({
            "maxUses": 2,
            "allowedDomains": ["example.com"],
            "blockedDomains": ["bad.com"]
        }));
        let mapped = convert_tools_to_anthropic_format(&[t]).expect("map ok");
        let obj = mapped.first().and_then(|v| v.as_object()).expect("obj");
        assert_eq!(
            obj.get("type").and_then(|v| v.as_str()),
            Some("web_search_20250305")
        );
        assert_eq!(obj.get("name").and_then(|v| v.as_str()), Some("web_search"));
        assert_eq!(obj.get("max_uses").and_then(|v| v.as_u64()), Some(2));
        assert!(obj.get("allowed_domains").is_some());
        assert!(obj.get("blocked_domains").is_some());
    }

    #[test]
    fn maps_anthropic_provider_defined_web_fetch() {
        let t = crate::tools::anthropic::web_fetch_20250910().with_args(serde_json::json!({
            "maxUses": 1,
            "allowedDomains": ["example.com"],
            "citations": { "enabled": true },
            "maxContentTokens": 2048
        }));
        let mapped = convert_tools_to_anthropic_format(&[t]).expect("map ok");
        let obj = mapped.first().and_then(|v| v.as_object()).expect("obj");
        assert_eq!(
            obj.get("type").and_then(|v| v.as_str()),
            Some("web_fetch_20250910")
        );
        assert_eq!(obj.get("name").and_then(|v| v.as_str()), Some("web_fetch"));
        assert_eq!(obj.get("max_uses").and_then(|v| v.as_u64()), Some(1));
        assert!(obj.get("allowed_domains").is_some());
        assert!(obj.get("citations").is_some());
        assert_eq!(
            obj.get("max_content_tokens").and_then(|v| v.as_u64()),
            Some(2048)
        );
    }

    #[test]
    fn maps_anthropic_provider_defined_tool_search_regex() {
        let t = crate::tools::anthropic::tool_search_regex_20251119();
        let mapped = convert_tools_to_anthropic_format(&[t]).expect("map ok");
        let obj = mapped.first().and_then(|v| v.as_object()).expect("obj");
        assert_eq!(
            obj.get("type").and_then(|v| v.as_str()),
            Some("tool_search_tool_regex_20251119")
        );
        assert_eq!(
            obj.get("name").and_then(|v| v.as_str()),
            Some("tool_search_tool_regex")
        );
    }

    #[test]
    fn maps_anthropic_provider_defined_tool_search_bm25() {
        let t = crate::tools::anthropic::tool_search_bm25_20251119();
        let mapped = convert_tools_to_anthropic_format(&[t]).expect("map ok");
        let obj = mapped.first().and_then(|v| v.as_object()).expect("obj");
        assert_eq!(
            obj.get("type").and_then(|v| v.as_str()),
            Some("tool_search_tool_bm25_20251119")
        );
        assert_eq!(
            obj.get("name").and_then(|v| v.as_str()),
            Some("tool_search_tool_bm25")
        );
    }

    #[test]
    fn maps_anthropic_provider_defined_code_execution() {
        let t = crate::tools::anthropic::code_execution_20250522();
        let mapped = convert_tools_to_anthropic_format(&[t]).expect("map ok");
        let obj = mapped.first().and_then(|v| v.as_object()).expect("obj");
        assert_eq!(
            obj.get("type").and_then(|v| v.as_str()),
            Some("code_execution_20250522")
        );
        assert_eq!(
            obj.get("name").and_then(|v| v.as_str()),
            Some("code_execution")
        );
    }

    #[test]
    fn maps_anthropic_provider_defined_code_execution_20250825() {
        let t = crate::tools::anthropic::code_execution_20250825();
        let mapped = convert_tools_to_anthropic_format(&[t]).expect("map ok");
        let obj = mapped.first().and_then(|v| v.as_object()).expect("obj");
        assert_eq!(
            obj.get("type").and_then(|v| v.as_str()),
            Some("code_execution_20250825")
        );
        assert_eq!(
            obj.get("name").and_then(|v| v.as_str()),
            Some("code_execution")
        );
    }

    #[test]
    fn maps_anthropic_provider_defined_computer_use() {
        let t = crate::tools::anthropic::computer_20241022().with_args(serde_json::json!({
            "displayWidthPx": 800,
            "displayHeightPx": 600,
            "displayNumber": 1
        }));
        let mapped = convert_tools_to_anthropic_format(&[t]).expect("map ok");
        let obj = mapped.first().and_then(|v| v.as_object()).expect("obj");
        assert_eq!(
            obj.get("type").and_then(|v| v.as_str()),
            Some("computer_20241022")
        );
        assert_eq!(obj.get("name").and_then(|v| v.as_str()), Some("computer"));
        assert_eq!(
            obj.get("display_width_px").and_then(|v| v.as_u64()),
            Some(800)
        );
        assert_eq!(
            obj.get("display_height_px").and_then(|v| v.as_u64()),
            Some(600)
        );
        assert_eq!(obj.get("display_number").and_then(|v| v.as_u64()), Some(1));
    }

    #[test]
    fn maps_anthropic_provider_defined_computer_use_20250124() {
        let t = crate::tools::anthropic::computer_20250124().with_args(serde_json::json!({
            "displayWidthPx": 800,
            "displayHeightPx": 600,
            "displayNumber": 1
        }));
        let mapped = convert_tools_to_anthropic_format(&[t]).expect("map ok");
        let obj = mapped.first().and_then(|v| v.as_object()).expect("obj");
        assert_eq!(
            obj.get("type").and_then(|v| v.as_str()),
            Some("computer_20250124")
        );
        assert_eq!(obj.get("name").and_then(|v| v.as_str()), Some("computer"));
        assert_eq!(
            obj.get("display_width_px").and_then(|v| v.as_u64()),
            Some(800)
        );
        assert_eq!(
            obj.get("display_height_px").and_then(|v| v.as_u64()),
            Some(600)
        );
        assert_eq!(obj.get("display_number").and_then(|v| v.as_u64()), Some(1));
    }

    #[test]
    fn maps_anthropic_provider_defined_text_editor() {
        let t = crate::tools::anthropic::text_editor_20241022();
        let mapped = convert_tools_to_anthropic_format(&[t]).expect("map ok");
        let obj = mapped.first().and_then(|v| v.as_object()).expect("obj");
        assert_eq!(
            obj.get("type").and_then(|v| v.as_str()),
            Some("text_editor_20241022")
        );
        assert_eq!(
            obj.get("name").and_then(|v| v.as_str()),
            Some("str_replace_editor")
        );
    }

    #[test]
    fn maps_anthropic_provider_defined_text_editor_20250124() {
        let t = crate::tools::anthropic::text_editor_20250124();
        let mapped = convert_tools_to_anthropic_format(&[t]).expect("map ok");
        let obj = mapped.first().and_then(|v| v.as_object()).expect("obj");
        assert_eq!(
            obj.get("type").and_then(|v| v.as_str()),
            Some("text_editor_20250124")
        );
        assert_eq!(
            obj.get("name").and_then(|v| v.as_str()),
            Some("str_replace_editor")
        );
    }

    #[test]
    fn maps_anthropic_provider_defined_text_editor_20250429() {
        let t = crate::tools::anthropic::text_editor_20250429();
        let mapped = convert_tools_to_anthropic_format(&[t]).expect("map ok");
        let obj = mapped.first().and_then(|v| v.as_object()).expect("obj");
        assert_eq!(
            obj.get("type").and_then(|v| v.as_str()),
            Some("text_editor_20250429")
        );
        assert_eq!(
            obj.get("name").and_then(|v| v.as_str()),
            Some("str_replace_based_edit_tool")
        );
    }

    #[test]
    fn maps_anthropic_provider_defined_text_editor_20250728_max_characters() {
        let t = crate::tools::anthropic::text_editor_20250728().with_args(serde_json::json!({
            "maxCharacters": 10000
        }));
        let mapped = convert_tools_to_anthropic_format(&[t]).expect("map ok");
        let obj = mapped.first().and_then(|v| v.as_object()).expect("obj");
        assert_eq!(
            obj.get("type").and_then(|v| v.as_str()),
            Some("text_editor_20250728")
        );
        assert_eq!(
            obj.get("name").and_then(|v| v.as_str()),
            Some("str_replace_based_edit_tool")
        );
        assert_eq!(
            obj.get("max_characters").and_then(|v| v.as_u64()),
            Some(10000)
        );
    }

    #[test]
    fn maps_anthropic_provider_defined_bash() {
        let t = crate::tools::anthropic::bash_20241022();
        let mapped = convert_tools_to_anthropic_format(&[t]).expect("map ok");
        let obj = mapped.first().and_then(|v| v.as_object()).expect("obj");
        assert_eq!(
            obj.get("type").and_then(|v| v.as_str()),
            Some("bash_20241022")
        );
        assert_eq!(obj.get("name").and_then(|v| v.as_str()), Some("bash"));
    }

    #[test]
    fn maps_anthropic_provider_defined_bash_20250124() {
        let t = crate::tools::anthropic::bash_20250124();
        let mapped = convert_tools_to_anthropic_format(&[t]).expect("map ok");
        let obj = mapped.first().and_then(|v| v.as_object()).expect("obj");
        assert_eq!(
            obj.get("type").and_then(|v| v.as_str()),
            Some("bash_20250124")
        );
        assert_eq!(obj.get("name").and_then(|v| v.as_str()), Some("bash"));
    }

    #[test]
    fn ignores_unknown_anthropic_provider_defined_tools() {
        let t = crate::types::Tool::provider_defined("anthropic.unknown_tool", "unknown_tool")
            .with_args(serde_json::json!({ "foo": "bar" }));
        let mapped = convert_tools_to_anthropic_format(&[t]).expect("map ok");
        assert!(mapped.is_empty());
    }
}

#[cfg(test)]
mod function_tool_tests {
    use super::*;

    #[test]
    fn function_tools_preserve_strict_and_defer_loading() {
        let mut tool = crate::types::Tool::function(
            "testFunction",
            "A test function",
            serde_json::json!({ "type": "object", "properties": {} }),
        );
        match &mut tool {
            crate::types::Tool::Function { function } => {
                function.strict = Some(true);
                function
                    .provider_options_map
                    .insert("anthropic", serde_json::json!({ "deferLoading": true }));
            }
            _ => panic!("expected function tool"),
        }

        let mapped = convert_tools_to_anthropic_format(&[tool]).expect("map ok");
        let obj = mapped.first().and_then(|v| v.as_object()).expect("obj");

        assert_eq!(
            obj.get("name").and_then(|v| v.as_str()),
            Some("testFunction")
        );
        assert_eq!(obj.get("strict").and_then(|v| v.as_bool()), Some(true));
        assert_eq!(
            obj.get("defer_loading").and_then(|v| v.as_bool()),
            Some(true)
        );
    }
}
