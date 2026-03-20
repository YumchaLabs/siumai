#![cfg(feature = "anthropic")]

use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use siumai::experimental::bridge::bridge_anthropic_messages_json_to_chat_request;
use siumai::prelude::unified::ResponseFormat;
use std::path::{Path, PathBuf};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("anthropic")
        .join("messages")
}

fn read_text(path: impl AsRef<Path>) -> String {
    std::fs::read_to_string(path).expect("read fixture text")
}

fn read_json<T: DeserializeOwned>(path: impl AsRef<Path>) -> T {
    let text = read_text(path);
    serde_json::from_str(&text).expect("parse fixture json")
}

#[test]
fn anthropic_messages_request_normalization_restores_basic_settings_from_fixtures() {
    let body: Value = read_json(
        fixtures_dir()
            .join("anthropic-settings.1")
            .join("expected_body.json"),
    );
    let request = bridge_anthropic_messages_json_to_chat_request(&body).expect("normalize");

    assert_eq!(request.common_params.model, "claude-3-haiku-20240307");
    assert_eq!(request.common_params.temperature, Some(0.5));
    assert_eq!(request.common_params.max_tokens, Some(100));
    assert_eq!(request.common_params.top_k, Some(0.1));
    assert_eq!(
        request.common_params.stop_sequences,
        Some(vec!["abc".to_string(), "def".to_string()])
    );
    assert_eq!(request.messages.len(), 1);
    assert_eq!(request.messages[0].content_text(), Some("Hello"));
}

#[test]
fn anthropic_messages_request_normalization_restores_tool_choice_and_provider_tools() {
    let body: Value = read_json(
        fixtures_dir()
            .join("anthropic-tool-choice-tool.1")
            .join("expected_body.json"),
    );
    let request = bridge_anthropic_messages_json_to_chat_request(&body).expect("normalize");
    assert_eq!(
        request.tool_choice,
        Some(siumai::prelude::unified::ToolChoice::tool("testFunction"))
    );
    let tools = request.tools.expect("tools");
    assert_eq!(tools.len(), 1);
    let siumai::prelude::unified::Tool::Function { function } = &tools[0] else {
        panic!("expected function tool");
    };
    assert_eq!(function.name, "testFunction");
    assert_eq!(function.description, "Test");

    let body: Value = read_json(
        fixtures_dir()
            .join("anthropic-web-search-tool.1")
            .join("expected_body.json"),
    );
    let request = bridge_anthropic_messages_json_to_chat_request(&body).expect("normalize");
    let tools = request.tools.expect("tools");
    let siumai::prelude::unified::Tool::ProviderDefined(tool) = &tools[0] else {
        panic!("expected provider-defined tool");
    };
    assert_eq!(tool.id, "anthropic.web_search_20250305");
    assert_eq!(tool.name, "web_search");
    assert_eq!(
        tool.args,
        json!({
            "max_uses": 1,
            "user_location": {
                "country": "US",
                "type": "approximate"
            }
        })
    );
}

#[test]
fn anthropic_messages_request_normalization_restores_structured_output_and_mcp_servers() {
    let body: Value = read_json(
        fixtures_dir()
            .join("anthropic-json-output-format.1")
            .join("expected_body.json"),
    );
    let request = bridge_anthropic_messages_json_to_chat_request(&body).expect("normalize");
    let anthropic_options = request
        .provider_option("anthropic")
        .and_then(Value::as_object)
        .expect("anthropic options");
    assert_eq!(
        anthropic_options["structuredOutputMode"],
        json!("outputFormat")
    );
    let format = request.response_format.expect("response format");
    let ResponseFormat::Json {
        name,
        strict,
        schema,
        ..
    } = format;
    assert_eq!(name, None);
    assert_eq!(strict, None);
    assert_eq!(
        schema,
        json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            },
            "required": ["name"],
            "additionalProperties": false,
            "$schema": "http://json-schema.org/draft-07/schema#"
        })
    );

    let body: Value = read_json(
        fixtures_dir()
            .join("anthropic-mcp.1")
            .join("expected_body.json"),
    );
    let request = bridge_anthropic_messages_json_to_chat_request(&body).expect("normalize");
    let anthropic_options = request
        .provider_option("anthropic")
        .and_then(Value::as_object)
        .expect("anthropic options");
    assert_eq!(
        anthropic_options["mcpServers"],
        json!([
            {
                "name": "echo",
                "type": "url",
                "url": "https://echo.mcp.inevitable.fyi/mcp"
            }
        ])
    );
}

#[test]
fn anthropic_messages_request_normalization_restores_thinking_body_semantics() {
    let body: Value = read_json(
        fixtures_dir()
            .join("anthropic-thinking-enabled.1")
            .join("expected_body.json"),
    );
    let request = bridge_anthropic_messages_json_to_chat_request(&body).expect("normalize");

    assert_eq!(request.common_params.model, "claude-sonnet-4-5");
    assert_eq!(request.common_params.max_tokens, Some(21000));
    assert_eq!(
        request.provider_option("anthropic"),
        Some(&json!({
            "thinking": {
                "budget_tokens": 1000,
                "type": "enabled"
            }
        }))
    );
}
