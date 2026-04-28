#![cfg(feature = "anthropic")]

use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use siumai::experimental::bridge::bridge_anthropic_messages_json_to_chat_request;
use siumai::prelude::unified::{ChatRequest, ResponseFormat};
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

fn request_fixture(case: &str) -> ChatRequest {
    read_json(fixtures_dir().join(case).join("request.json"))
}

fn expected_body_fixture(case: &str) -> Value {
    read_json(fixtures_dir().join(case).join("expected_body.json"))
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
fn anthropic_messages_request_normalization_restores_message_and_part_provider_options() {
    let body = expected_body_fixture("anthropic-message-and-part-provider-options.1");
    let request = bridge_anthropic_messages_json_to_chat_request(&body).expect("normalize");

    assert_eq!(request.messages.len(), 1);
    assert_eq!(
        request.messages[0].role,
        siumai::prelude::unified::MessageRole::User
    );
    assert!(
        request.messages[0].provider_options.is_empty(),
        "wire cache_control should normalize onto the canonical Anthropic part lane"
    );

    let parts = request.messages[0]
        .content
        .as_multimodal()
        .expect("multimodal content");

    let file_anthropic = parts[0]
        .provider_options()
        .and_then(|provider_options| provider_options.get_object("anthropic"))
        .expect("file anthropic provider options");
    assert_eq!(file_anthropic["citations"], json!({ "enabled": true }));
    assert_eq!(file_anthropic["title"], json!("My Doc"));
    assert_eq!(file_anthropic["context"], json!("background"));

    let text_anthropic = parts[1]
        .provider_options()
        .and_then(|provider_options| provider_options.get_object("anthropic"))
        .expect("text anthropic provider options");
    assert_eq!(
        text_anthropic["cacheControl"],
        json!({ "type": "ephemeral" })
    );
}

#[test]
fn anthropic_messages_request_normalization_restores_tool_choice_and_provider_tools() {
    let body = expected_body_fixture("anthropic-tool-choice-tool.1");
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

    for case in ["anthropic-web-search-tool.1", "anthropic-web-fetch-tool.1"] {
        let body = expected_body_fixture(case);
        let request = bridge_anthropic_messages_json_to_chat_request(&body).expect("normalize");
        let expected = request_fixture(case);
        let request_tools = request.tools.expect("normalized tools");
        let expected_tools = expected.tools.expect("fixture tools");
        assert_eq!(
            request_tools.len(),
            expected_tools.len(),
            "fixture case: {}",
            case
        );

        let siumai::prelude::unified::Tool::ProviderDefined(request_tool) = &request_tools[0]
        else {
            panic!("expected provider-defined tool");
        };
        let siumai::prelude::unified::Tool::ProviderDefined(expected_tool) = &expected_tools[0]
        else {
            panic!("expected fixture provider-defined tool");
        };

        assert_eq!(request_tool.id, expected_tool.id, "fixture case: {}", case);
        assert_eq!(
            request_tool.name, expected_tool.name,
            "fixture case: {}",
            case
        );
        assert_eq!(
            request_tool.args, expected_tool.args,
            "fixture case: {}",
            case
        );
    }
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
    } = format
    else {
        panic!("expected JSON response format, got {format:?}");
    };
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
    let expected = request_fixture("anthropic-mcp.1");
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
    assert_eq!(
        request.provider_option("anthropic"),
        expected.provider_option("anthropic")
    );
}

#[test]
fn anthropic_messages_request_normalization_restores_thinking_body_semantics() {
    let body = expected_body_fixture("anthropic-thinking-enabled.1");
    let request = bridge_anthropic_messages_json_to_chat_request(&body).expect("normalize");
    let expected = request_fixture("anthropic-thinking-enabled.1");

    assert_eq!(request.common_params.model, "claude-sonnet-4-5");
    assert_eq!(request.common_params.max_tokens, Some(21000));
    assert_eq!(
        request.provider_option("anthropic"),
        expected.provider_option("anthropic")
    );
}

#[test]
fn anthropic_messages_request_normalization_restores_context_management_and_container_options() {
    for case in [
        "anthropic-context-management.1",
        "anthropic-context-management-full.1",
        "anthropic-code-execution-20250825.pptx-skill",
    ] {
        let body = expected_body_fixture(case);
        let request = bridge_anthropic_messages_json_to_chat_request(&body).expect("normalize");
        let expected = request_fixture(case);
        assert_eq!(
            request.provider_option("anthropic"),
            expected.provider_option("anthropic"),
            "fixture case: {}",
            case
        );
    }
}
