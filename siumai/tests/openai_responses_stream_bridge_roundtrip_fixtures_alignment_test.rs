#![cfg(feature = "openai")]

use eventsource_stream::Event;
use futures_util::{StreamExt, stream};
use serde::Serialize;
use serde_json::Value;
use siumai::experimental::bridge::{
    BridgeMode, BridgeTarget, bridge_chat_stream_to_openai_responses_sse,
};
use siumai::prelude::unified::{ChatByteStream, ChatStreamEvent, SseEventConverter};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("openai")
        .join("responses-stream")
}

fn read_fixture_lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .expect("read fixture file")
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.to_string())
        .collect()
}

fn extract_sse_data_payload_lines(bytes: &[u8]) -> Vec<String> {
    let text = String::from_utf8_lossy(bytes);
    text.split("\n\n")
        .filter_map(|frame| {
            let frame = frame.trim();
            if frame.is_empty() {
                return None;
            }
            let data = frame
                .lines()
                .find(|line| line.starts_with("data: "))
                .map(|line| line.trim_start_matches("data: ").trim())?;
            if data.is_empty() || data == "[DONE]" {
                return None;
            }
            Some(data.to_string())
        })
        .collect()
}

fn decode_openai_responses(lines: Vec<String>) -> Vec<ChatStreamEvent> {
    let conv = siumai::protocol::openai::responses_sse::OpenAiResponsesEventConverter::new();
    let mut events = Vec::new();

    for (index, line) in lines.into_iter().enumerate() {
        let event = Event {
            event: "".to_string(),
            data: line,
            id: index.to_string(),
            retry: None,
        };
        let out = futures::executor::block_on(conv.convert_event(event));
        for item in out {
            match item {
                Ok(evt) => events.push(evt),
                Err(err) => panic!("failed to convert chunk: {err:?}"),
            }
        }
    }

    while let Some(item) = conv.handle_stream_end() {
        match item {
            Ok(evt) => events.push(evt),
            Err(err) => panic!("failed to finalize stream: {err:?}"),
        }
    }

    events
}

async fn collect_bytes(mut stream: ChatByteStream) -> Vec<u8> {
    let mut out = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.expect("stream chunk");
        out.extend_from_slice(&chunk);
    }
    out
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct ReasoningBoundary {
    id: Option<String>,
    item_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct OpenAiStreamSummary {
    has_stream_start: bool,
    has_finish: bool,
    has_metadata_id: bool,
    metadata_model: Option<String>,
    finish_reason: Option<String>,
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    text: String,
    reasoning_starts: Vec<ReasoningBoundary>,
    reasoning_deltas: Vec<String>,
    reasoning_ends: Vec<ReasoningBoundary>,
    provider_tool_calls: BTreeMap<String, usize>,
    provider_tool_results: BTreeMap<String, usize>,
    tool_approval_requests: BTreeMap<String, usize>,
    source_urls: BTreeMap<String, usize>,
}

fn push_adjacent_unique<T: PartialEq>(items: &mut Vec<T>, value: T) {
    if items.last() != Some(&value) {
        items.push(value);
    }
}

fn increment_count(counts: &mut BTreeMap<String, usize>, key: Option<&str>) {
    if let Some(key) = key {
        *counts.entry(key.to_string()).or_default() += 1;
    }
}

fn finish_reason_to_string(reason: &siumai::prelude::unified::FinishReason) -> Option<String> {
    serde_json::to_value(reason)
        .ok()
        .and_then(|value| value.as_str().map(ToString::to_string))
}

fn reasoning_boundary_from_custom(data: &Value) -> ReasoningBoundary {
    ReasoningBoundary {
        id: data
            .get("id")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        item_id: data
            .pointer("/providerMetadata/openai/itemId")
            .and_then(Value::as_str)
            .map(ToString::to_string),
    }
}

fn summarize_openai_events(events: &[ChatStreamEvent]) -> OpenAiStreamSummary {
    let mut summary = OpenAiStreamSummary {
        has_stream_start: false,
        has_finish: false,
        has_metadata_id: false,
        metadata_model: None,
        finish_reason: None,
        prompt_tokens: None,
        completion_tokens: None,
        text: String::new(),
        reasoning_starts: Vec::new(),
        reasoning_deltas: Vec::new(),
        reasoning_ends: Vec::new(),
        provider_tool_calls: BTreeMap::new(),
        provider_tool_results: BTreeMap::new(),
        tool_approval_requests: BTreeMap::new(),
        source_urls: BTreeMap::new(),
    };
    let mut text_deltas = Vec::new();

    for event in events {
        match event {
            ChatStreamEvent::StreamStart { .. } => {
                summary.has_stream_start = true;
            }
            ChatStreamEvent::ContentDelta { delta, .. } => {
                push_adjacent_unique(&mut text_deltas, delta.clone());
            }
            ChatStreamEvent::UsageUpdate { usage } => {
                if usage.prompt_tokens > 0 || usage.completion_tokens > 0 {
                    summary.prompt_tokens = Some(usage.prompt_tokens);
                    summary.completion_tokens = Some(usage.completion_tokens);
                }
            }
            ChatStreamEvent::StreamEnd { response } => {
                if summary.finish_reason.is_none() {
                    summary.finish_reason = response
                        .finish_reason
                        .as_ref()
                        .and_then(finish_reason_to_string);
                }
                if summary.prompt_tokens.is_none()
                    && let Some(usage) = response.usage.as_ref()
                {
                    summary.prompt_tokens = Some(usage.prompt_tokens);
                    summary.completion_tokens = Some(usage.completion_tokens);
                }
            }
            ChatStreamEvent::Custom { data, .. } => {
                match data.get("type").and_then(Value::as_str) {
                    Some("response-metadata") => {
                        summary.has_metadata_id = data.get("id").and_then(Value::as_str).is_some();
                        summary.metadata_model = data
                            .get("modelId")
                            .and_then(Value::as_str)
                            .map(ToString::to_string);
                    }
                    Some("finish") => {
                        summary.has_finish = true;
                        summary.finish_reason = data
                            .pointer("/finishReason/unified")
                            .and_then(Value::as_str)
                            .map(ToString::to_string)
                            .or_else(|| summary.finish_reason.clone());
                        summary.prompt_tokens = data
                            .pointer("/usage/inputTokens/total")
                            .and_then(Value::as_u64)
                            .and_then(|value| u32::try_from(value).ok())
                            .or(summary.prompt_tokens);
                        summary.completion_tokens = data
                            .pointer("/usage/outputTokens/total")
                            .and_then(Value::as_u64)
                            .and_then(|value| u32::try_from(value).ok())
                            .or(summary.completion_tokens);
                    }
                    Some("reasoning-start") => {
                        push_adjacent_unique(
                            &mut summary.reasoning_starts,
                            reasoning_boundary_from_custom(data),
                        );
                    }
                    Some("reasoning-delta") => {
                        if let Some(delta) = data.get("delta").and_then(Value::as_str) {
                            push_adjacent_unique(&mut summary.reasoning_deltas, delta.to_string());
                        }
                    }
                    Some("reasoning-end") => {
                        push_adjacent_unique(
                            &mut summary.reasoning_ends,
                            reasoning_boundary_from_custom(data),
                        );
                    }
                    Some("tool-call") => {
                        if data
                            .get("providerExecuted")
                            .and_then(Value::as_bool)
                            .unwrap_or(false)
                        {
                            increment_count(
                                &mut summary.provider_tool_calls,
                                data.get("toolName").and_then(Value::as_str),
                            );
                        }
                    }
                    Some("tool-result") => {
                        if data
                            .get("providerExecuted")
                            .and_then(Value::as_bool)
                            .unwrap_or(false)
                        {
                            increment_count(
                                &mut summary.provider_tool_results,
                                data.get("toolName").and_then(Value::as_str),
                            );
                        }
                    }
                    Some("tool-approval-request") => {
                        increment_count(
                            &mut summary.tool_approval_requests,
                            data.get("toolCallId").and_then(Value::as_str),
                        );
                    }
                    Some("source") => {
                        increment_count(
                            &mut summary.source_urls,
                            data.get("url").and_then(Value::as_str),
                        );
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    summary.text = text_deltas.concat();
    summary
}

async fn roundtrip_summary(path: &str) -> OpenAiStreamSummary {
    let fixture = fixtures_dir().join(path);
    let original = decode_openai_responses(read_fixture_lines(&fixture));
    let bridged = bridge_chat_stream_to_openai_responses_sse(
        stream::iter(original.clone().into_iter().map(Ok)),
        Some(BridgeTarget::OpenAiResponses),
        BridgeMode::Strict,
    )
    .expect("bridge stream");

    assert!(
        !bridged.is_rejected(),
        "bridge rejected fixture case {} with report {:?}",
        fixture.display(),
        bridged.report
    );

    let bytes = collect_bytes(bridged.value.expect("bridged byte stream")).await;
    let roundtripped = decode_openai_responses(extract_sse_data_payload_lines(&bytes));
    summarize_openai_events(&roundtripped)
}

fn fixture_summary(path: &str) -> OpenAiStreamSummary {
    let fixture = fixtures_dir().join(path);
    summarize_openai_events(&decode_openai_responses(read_fixture_lines(&fixture)))
}

#[tokio::test]
async fn openai_responses_stream_bridge_roundtrip_fixture_summary_cases_match() {
    let summary_cases = [
        "text/openai-text-deltas.1.chunks.txt",
        "reasoning/openai-reasoning-encrypted-content.1.chunks.txt",
        "reasoning/openai-reasoning-encrypted-empty-summary.1.chunks.txt",
        "mcp/openai-mcp-tool-approval.1.chunks.txt",
        "web-search/openai-web-search-tool.1.chunks.txt",
    ];

    for case in summary_cases {
        assert_eq!(
            roundtrip_summary(case).await,
            fixture_summary(case),
            "fixture case: {}",
            fixtures_dir().join(case).display()
        );
    }
}
