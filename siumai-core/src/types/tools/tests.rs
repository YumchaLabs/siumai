use super::*;

#[test]
#[cfg(any())]
fn web_search_advanced_merges_extra() {
    let t = OpenAiBuiltInTool::WebSearchAdvanced {
        extra: serde_json::json!({"region":"us","safe":"moderate"}),
    };
    let v = t.to_json();
    assert_eq!(
        v.get("type").and_then(|s| s.as_str()),
        Some("web_search_preview")
    );
    assert_eq!(v.get("region").and_then(|s| s.as_str()), Some("us"));
    assert_eq!(v.get("safe").and_then(|s| s.as_str()), Some("moderate"));
}

#[test]
#[cfg(any())]
fn file_search_advanced_includes_ids_and_extra() {
    let t = OpenAiBuiltInTool::FileSearchAdvanced {
        vector_store_ids: Some(vec!["vs1".into(), "vs2".into()]),
        extra: serde_json::json!({"ranker":"semantic"}),
    };
    let v = t.to_json();
    assert_eq!(v.get("type").and_then(|s| s.as_str()), Some("file_search"));
    let ids = v
        .get("vector_store_ids")
        .and_then(|a| a.as_array())
        .cloned()
        .unwrap();
    assert_eq!(ids.len(), 2);
    assert_eq!(v.get("ranker").and_then(|s| s.as_str()), Some("semantic"));
}

#[test]
#[cfg(any())]
fn computer_use_advanced_includes_scale_and_extra() {
    let t = OpenAiBuiltInTool::ComputerUseAdvanced {
        display_width: 1280,
        display_height: 720,
        environment: "headless".into(),
        display_scale: Some(1.5),
        extra: serde_json::json!({"cursor":"enabled"}),
    };
    let v = t.to_json();
    assert_eq!(
        v.get("type").and_then(|s| s.as_str()),
        Some("computer_use_preview")
    );
    assert_eq!(v.get("display_width").and_then(|x| x.as_u64()), Some(1280));
    assert_eq!(v.get("display_height").and_then(|x| x.as_u64()), Some(720));
    assert_eq!(
        v.get("environment").and_then(|s| s.as_str()),
        Some("headless")
    );
    assert_eq!(v.get("display_scale").and_then(|x| x.as_f64()), Some(1.5));
    assert_eq!(v.get("cursor").and_then(|s| s.as_str()), Some("enabled"));
}

#[test]
#[cfg(any())]
fn web_search_options_includes_country_timezone() {
    let t = OpenAiBuiltInTool::WebSearchOptions {
        options: WebSearchOptions {
            search_context_size: Some("high".into()),
            user_location: Some(WebSearchUserLocation {
                r#type: "approximate".into(),
                country: Some("US".into()),
                city: Some("SF".into()),
                region: Some("CA".into()),
                timezone: Some("America/Los_Angeles".into()),
            }),
        },
    };
    let v = t.to_json();
    let loc = v.get("userLocation").and_then(|o| o.as_object()).unwrap();
    assert_eq!(loc.get("country").and_then(|x| x.as_str()), Some("US"));
    assert_eq!(
        loc.get("timezone").and_then(|x| x.as_str()),
        Some("America/Los_Angeles")
    );
}

#[test]
#[cfg(any())]
fn file_search_options_includes_max_num_results_ranking_filters() {
    let t = OpenAiBuiltInTool::FileSearchOptions {
        vector_store_ids: Some(vec!["vs1".into()]),
        max_num_results: Some(15),
        ranking_options: Some(FileSearchRankingOptions {
            ranker: Some("auto".into()),
            score_threshold: Some(0.5),
        }),
        filters: Some(FileSearchFilter::And {
            filters: vec![
                FileSearchFilter::Eq {
                    key: "doc_type".into(),
                    value: serde_json::json!("pdf"),
                },
                FileSearchFilter::Gt {
                    key: "score".into(),
                    value: serde_json::json!(0.1),
                },
            ],
        }),
    };
    let v = t.to_json();
    assert_eq!(v.get("type").and_then(|s| s.as_str()), Some("file_search"));
    assert_eq!(v.get("max_num_results").and_then(|x| x.as_u64()), Some(15));
    let ro = v
        .get("ranking_options")
        .and_then(|o| o.as_object())
        .unwrap();
    assert_eq!(ro.get("ranker").and_then(|x| x.as_str()), Some("auto"));
    assert_eq!(
        ro.get("score_threshold").and_then(|x| x.as_f64()),
        Some(0.5)
    );
    assert!(v.get("filters").is_some());
}

#[test]
fn provider_defined_tool_new() {
    let tool = ProviderDefinedTool::new("openai.web_search", "web_search");
    assert_eq!(tool.id, "openai.web_search");
    assert_eq!(tool.name, "web_search");
    assert_eq!(tool.args, serde_json::json!({}));
}

#[test]
fn provider_defined_tool_with_args() {
    let tool =
        ProviderDefinedTool::new("openai.web_search", "web_search").with_args(serde_json::json!({
            "searchContextSize": "high"
        }));
    assert_eq!(tool.id, "openai.web_search");
    assert_eq!(tool.name, "web_search");
    assert_eq!(
        tool.args.get("searchContextSize").and_then(|v| v.as_str()),
        Some("high")
    );
}

#[test]
fn provider_defined_tool_provider() {
    let tool = ProviderDefinedTool::new("openai.web_search", "web_search");
    assert_eq!(tool.provider(), Some("openai"));

    let tool2 = ProviderDefinedTool::new("anthropic.web_search_20250305", "web_search");
    assert_eq!(tool2.provider(), Some("anthropic"));

    let tool3 = ProviderDefinedTool::new("invalid", "test");
    assert_eq!(tool3.provider(), Some("invalid"));
}

#[test]
fn provider_defined_tool_tool_type() {
    let tool = ProviderDefinedTool::new("openai.web_search", "web_search");
    assert_eq!(tool.tool_type(), Some("web_search"));

    let tool2 = ProviderDefinedTool::new("google.code_execution", "code_execution");
    assert_eq!(tool2.tool_type(), Some("code_execution"));

    let tool3 = ProviderDefinedTool::new("invalid", "test");
    assert_eq!(tool3.tool_type(), None);
}

#[test]
fn tool_enum_function_variant() {
    let tool = Tool::function(
        "weather".to_string(),
        "Get weather".to_string(),
        serde_json::json!({}),
    );
    match tool {
        Tool::Function { function } => {
            assert_eq!(function.name, "weather");
            assert_eq!(function.description, "Get weather");
        }
        _ => panic!("Expected Function variant"),
    }
}

#[test]
fn tool_enum_provider_defined_variant() {
    let provider_tool = ProviderDefinedTool::new("openai.web_search", "web_search");
    let tool = Tool::ProviderDefined(provider_tool.clone());
    match tool {
        Tool::ProviderDefined(pt) => {
            assert_eq!(pt.id, "openai.web_search");
            assert_eq!(pt.name, "web_search");
        }
        _ => panic!("Expected ProviderDefined variant"),
    }
}

#[test]
fn tool_enum_serialization() {
    // Test Function variant serialization
    let func_tool = Tool::function(
        "weather".to_string(),
        "Get weather".to_string(),
        serde_json::json!({}),
    );
    let json = serde_json::to_value(&func_tool).unwrap();
    assert_eq!(json.get("type").and_then(|v| v.as_str()), Some("function"));
    assert_eq!(json.get("name").and_then(|v| v.as_str()), Some("weather"));

    // Test ProviderDefined variant serialization
    let provider_tool = ProviderDefinedTool::new("openai.web_search", "web_search");
    let pd_tool = Tool::ProviderDefined(provider_tool);
    let json = serde_json::to_value(&pd_tool).unwrap();
    assert_eq!(
        json.get("type").and_then(|v| v.as_str()),
        Some("provider-defined")
    );
    assert_eq!(
        json.get("id").and_then(|v| v.as_str()),
        Some("openai.web_search")
    );
    assert_eq!(
        json.get("name").and_then(|v| v.as_str()),
        Some("web_search")
    );
    assert_eq!(json.get("args"), Some(&serde_json::json!({})));
}

#[test]
fn provider_defined_tool_deserializes_vercel_shape() {
    let v = serde_json::json!({
        "type": "provider",
        "id": "openai.web_search_preview",
        "name": "web_search_preview",
        "args": {
            "search_context_size": "low"
        }
    });

    let tool: Tool = serde_json::from_value(v).expect("deserialize provider tool");
    let Tool::ProviderDefined(tool) = tool else {
        panic!("expected ProviderDefined");
    };
    assert_eq!(tool.id, "openai.web_search_preview");
    assert_eq!(tool.name, "web_search_preview");
    assert_eq!(
        tool.args
            .get("search_context_size")
            .and_then(|v| v.as_str()),
        Some("low")
    );
}

#[test]
fn provider_defined_tool_deserializes_legacy_flatten_shape() {
    let v = serde_json::json!({
        "type": "provider-defined",
        "id": "openai.web_search",
        "name": "web_search",
        "searchContextSize": "high",
        "userLocation": {
            "type": "approximate",
            "country": "US"
        }
    });

    let tool: Tool = serde_json::from_value(v).expect("deserialize legacy provider tool");
    let Tool::ProviderDefined(tool) = tool else {
        panic!("expected ProviderDefined");
    };
    assert_eq!(tool.id, "openai.web_search");
    assert_eq!(tool.name, "web_search");
    assert_eq!(
        tool.args.get("searchContextSize").and_then(|v| v.as_str()),
        Some("high")
    );
    assert_eq!(
        tool.args
            .get("userLocation")
            .and_then(|v| v.get("country"))
            .and_then(|v| v.as_str()),
        Some("US")
    );
}

#[test]
fn tool_enum_deserialization() {
    // Test Function variant deserialization
    let json = serde_json::json!({
        "type": "function",
        "name": "weather",
        "description": "Get weather",
        "parameters": {}
    });
    let tool: Tool = serde_json::from_value(json).unwrap();
    match tool {
        Tool::Function { function } => {
            assert_eq!(function.name, "weather");
        }
        _ => panic!("Expected Function variant"),
    }

    // Test ProviderDefined variant deserialization
    let json = serde_json::json!({
        "type": "provider-defined",
        "id": "openai.web_search",
        "name": "web_search"
    });
    let tool: Tool = serde_json::from_value(json).unwrap();
    match tool {
        Tool::ProviderDefined(pt) => {
            assert_eq!(pt.id, "openai.web_search");
            assert_eq!(pt.name, "web_search");
        }
        _ => panic!("Expected ProviderDefined variant"),
    }
}
