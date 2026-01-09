#![cfg(feature = "bedrock")]

//! Alignment tests for Vercel `@ai-sdk/amazon-bedrock` Converse streaming fixtures.

use siumai::prelude::unified::*;
use std::path::Path;

fn fixtures_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("bedrock")
        .join("chat")
}

fn read_fixture_lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .expect("read fixture file")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.to_string())
        .collect()
}

fn run_converter(lines: Vec<String>, uses_json_tool: bool) -> Vec<ChatStreamEvent> {
    let standard = siumai::experimental::standards::bedrock::chat::BedrockChatStandard::new();
    let tx = standard.create_transformers("bedrock", uses_json_tool);
    let conv = tx.json.expect("bedrock json event converter");

    let mut out: Vec<ChatStreamEvent> = Vec::new();
    for line in lines {
        let events = futures::executor::block_on(
            siumai::experimental::providers::amazon_bedrock::streaming::JsonEventConverter::convert_json(
                conv.as_ref(),
                &line,
            ),
        );
        for item in events {
            match item {
                Ok(e) => out.push(e),
                Err(err) => panic!("failed to convert chunk: {err:?}"),
            }
        }
    }

    if let Some(item) =
        siumai::experimental::providers::amazon_bedrock::streaming::JsonEventConverter::handle_stream_end(
            conv.as_ref(),
        )
    {
        match item {
            Ok(e) => out.push(e),
            Err(err) => panic!("failed to finalize stream: {err:?}"),
        }
    }

    out
}

#[test]
fn bedrock_tool_call_stream_emits_tool_call_deltas() {
    let path = fixtures_dir().join("bedrock-tool-call.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);

    let lines = read_fixture_lines(&path);
    let events = run_converter(lines, false);

    assert!(
        events
            .iter()
            .any(|e| matches!(e, ChatStreamEvent::StreamStart { .. })),
        "expected StreamStart"
    );

    let mut args = String::new();
    let mut name: Option<String> = None;
    let mut id: Option<String> = None;
    for e in &events {
        if let ChatStreamEvent::ToolCallDelta {
            id: call_id,
            function_name,
            arguments_delta,
            ..
        } = e
        {
            id.get_or_insert_with(|| call_id.clone());
            if let Some(n) = function_name {
                name.get_or_insert_with(|| n.clone());
            }
            if let Some(delta) = arguments_delta {
                args.push_str(delta);
            }
        }
    }

    assert_eq!(name.as_deref(), Some("test-tool"));
    assert_eq!(id.as_deref(), Some("tool-use-id"));
    assert!(
        args.contains("Sparkle Day"),
        "expected tool input JSON fragments"
    );

    let ended = events
        .iter()
        .any(|e| matches!(e, ChatStreamEvent::StreamEnd { .. }));
    assert!(ended, "expected StreamEnd");

    let end_resp = events.iter().find_map(|e| match e {
        ChatStreamEvent::StreamEnd { response } => Some(response),
        _ => None,
    });
    let end_resp = end_resp.expect("StreamEnd response");
    assert_eq!(end_resp.finish_reason, Some(FinishReason::ToolCalls));
}

#[test]
fn bedrock_json_tool_stream_emits_json_as_text() {
    let path = fixtures_dir().join("bedrock-json-tool.1.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);

    let lines = read_fixture_lines(&path);
    let events = run_converter(lines, true);

    let text: String = events
        .iter()
        .filter_map(|e| match e {
            ChatStreamEvent::ContentDelta { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        text.contains("\"value\":\"test\""),
        "expected JSON text to be emitted from json tool"
    );

    let end_resp = events.iter().find_map(|e| match e {
        ChatStreamEvent::StreamEnd { response } => Some(response),
        _ => None,
    });
    let end_resp = end_resp.expect("StreamEnd response");
    assert_eq!(end_resp.finish_reason, Some(FinishReason::Stop));

    let meta = end_resp
        .provider_metadata
        .as_ref()
        .and_then(|m| m.get("bedrock"))
        .and_then(|m| m.get("isJsonResponseFromTool"))
        .and_then(|v| v.as_bool());
    assert_eq!(meta, Some(true));
}
