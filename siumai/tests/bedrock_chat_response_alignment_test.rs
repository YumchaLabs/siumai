#![cfg(feature = "bedrock")]

//! Alignment tests for Vercel `@ai-sdk/amazon-bedrock` Converse response fixtures.

use serde_json::Value;
use siumai::prelude::unified::*;
use std::path::Path;

fn fixtures_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("bedrock")
        .join("chat")
}

fn read_json(path: impl AsRef<Path>) -> Value {
    let text = std::fs::read_to_string(path).expect("read fixture json");
    serde_json::from_str(&text).expect("parse fixture json")
}

#[test]
fn bedrock_tool_use_maps_to_tool_call_part() {
    let raw = read_json(fixtures_dir().join("bedrock-tool-call.1.json"));
    let standard = siumai::experimental::standards::bedrock::chat::BedrockChatStandard::new();
    let tx = standard.create_transformers("bedrock", false);

    let resp = tx
        .response
        .transform_chat_response(&raw)
        .expect("transform response");

    assert_eq!(resp.finish_reason, Some(FinishReason::ToolCalls));

    let parts = resp
        .content
        .as_multimodal()
        .expect("expected multimodal content");
    let tool_call = parts.iter().find_map(|p| p.as_tool_call());
    let tool_call = tool_call.expect("expected tool call part");
    assert_eq!(tool_call.tool_name, "bash");
    assert_eq!(
        tool_call.arguments.get("command").and_then(|v| v.as_str()),
        Some("ls -l")
    );
}

#[test]
fn bedrock_json_tool_response_format_returns_json_as_text() {
    let raw = read_json(fixtures_dir().join("bedrock-json-tool.1.json"));
    let standard = siumai::experimental::standards::bedrock::chat::BedrockChatStandard::new();
    let tx = standard.create_transformers("bedrock", true);

    let resp = tx
        .response
        .transform_chat_response(&raw)
        .expect("transform response");

    assert_eq!(resp.finish_reason, Some(FinishReason::Stop));

    let text = resp.content_text().unwrap_or_default();
    assert!(
        text.contains("\"elements\"") && text.contains("San Francisco"),
        "expected JSON text to be preserved"
    );

    let meta = resp
        .provider_metadata
        .as_ref()
        .and_then(|m| m.get("bedrock"))
        .and_then(|m| m.get("isJsonResponseFromTool"))
        .and_then(|v| v.as_bool());
    assert_eq!(meta, Some(true));
}

#[test]
fn bedrock_json_other_tool_stays_as_tool_call() {
    let raw = read_json(fixtures_dir().join("bedrock-json-other-tool.1.json"));
    let standard = siumai::experimental::standards::bedrock::chat::BedrockChatStandard::new();
    let tx = standard.create_transformers("bedrock", true);

    let resp = tx
        .response
        .transform_chat_response(&raw)
        .expect("transform response");

    assert_eq!(resp.finish_reason, Some(FinishReason::ToolCalls));
    let parts = resp
        .content
        .as_multimodal()
        .expect("expected multimodal content");
    let tool_call = parts.iter().find_map(|p| p.as_tool_call());
    let tool_call = tool_call.expect("expected tool call part");
    assert_eq!(tool_call.tool_name, "get-weather");
    assert_eq!(
        tool_call.arguments.get("location").and_then(|v| v.as_str()),
        Some("San Francisco")
    );
}
