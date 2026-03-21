#![cfg(feature = "anthropic")]

use eventsource_stream::Event;
use futures_util::{StreamExt, stream};
use serde::Serialize;
use serde_json::Value;
use siumai::experimental::bridge::{
    BridgeMode, BridgeTarget, bridge_chat_stream_to_anthropic_messages_sse,
};
use siumai::prelude::unified::{ChatByteStream, ChatStreamEvent, SseEventConverter};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("anthropic")
        .join("messages-stream")
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
            if data.is_empty() {
                return None;
            }
            Some(data.to_string())
        })
        .collect()
}

fn decode_anthropic(lines: Vec<String>) -> Vec<ChatStreamEvent> {
    let conv = siumai::protocol::anthropic::streaming::AnthropicEventConverter::new(
        siumai::protocol::anthropic::params::AnthropicParams::default(),
    );
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
struct AnthropicStreamSummary {
    has_stream_start: bool,
    has_finish: bool,
    start_id: Option<String>,
    start_model: Option<String>,
    finish_reason: Option<String>,
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    text: String,
    provider_tool_calls: BTreeMap<String, usize>,
    provider_tool_results: BTreeMap<String, usize>,
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

fn summarize_anthropic_events(events: &[ChatStreamEvent]) -> AnthropicStreamSummary {
    let mut summary = AnthropicStreamSummary {
        has_stream_start: false,
        has_finish: false,
        start_id: None,
        start_model: None,
        finish_reason: None,
        prompt_tokens: None,
        completion_tokens: None,
        text: String::new(),
        provider_tool_calls: BTreeMap::new(),
        provider_tool_results: BTreeMap::new(),
        source_urls: BTreeMap::new(),
    };
    let mut text_deltas = Vec::new();

    for event in events {
        match event {
            ChatStreamEvent::StreamStart { metadata } => {
                summary.has_stream_start = true;
                summary.start_id = metadata.id.clone();
                summary.start_model = metadata.model.clone();
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
                        if summary.start_id.is_none() {
                            summary.start_id = data
                                .get("id")
                                .and_then(Value::as_str)
                                .map(ToString::to_string);
                        }
                        if summary.start_model.is_none() {
                            summary.start_model = data
                                .get("modelId")
                                .and_then(Value::as_str)
                                .map(ToString::to_string);
                        }
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

async fn roundtrip_summary(path: &str) -> AnthropicStreamSummary {
    let fixture = fixtures_dir().join(path);
    let original = decode_anthropic(read_fixture_lines(&fixture));
    let bridged = bridge_chat_stream_to_anthropic_messages_sse(
        stream::iter(original.clone().into_iter().map(Ok)),
        Some(BridgeTarget::AnthropicMessages),
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
    let roundtripped = decode_anthropic(extract_sse_data_payload_lines(&bytes));
    summarize_anthropic_events(&roundtripped)
}

fn fixture_summary(path: &str) -> AnthropicStreamSummary {
    let fixture = fixtures_dir().join(path);
    summarize_anthropic_events(&decode_anthropic(read_fixture_lines(&fixture)))
}

#[tokio::test]
async fn anthropic_messages_stream_bridge_roundtrip_fixture_summary_cases_match() {
    let summary_cases = [
        "anthropic-message-delta-input-tokens.chunks.txt",
        "anthropic-json-output-format.1.chunks.txt",
        "anthropic-json-tool.1.chunks.txt",
        "anthropic-web-search-tool.1.chunks.txt",
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
