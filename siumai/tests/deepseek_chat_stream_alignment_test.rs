#![cfg(feature = "deepseek")]

//! Alignment tests for Vercel `@ai-sdk/deepseek` Chat Completions streaming fixtures.

use eventsource_stream::Event;
use siumai::prelude::unified::*;
use siumai_provider_openai_compatible::providers::openai_compatible::streaming::OpenAiCompatibleEventConverter;
use siumai_provider_openai_compatible::providers::openai_compatible::{
    ConfigurableAdapter, OpenAiCompatibleConfig, get_provider_config,
};
use std::path::Path;
use std::sync::Arc;

fn fixtures_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("deepseek")
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

fn make_deepseek_converter(model: &str) -> OpenAiCompatibleEventConverter {
    let provider_config = get_provider_config("deepseek").expect("deepseek provider config");
    let adapter = Arc::new(ConfigurableAdapter::new(provider_config.clone()));

    let cfg = OpenAiCompatibleConfig::new(
        "deepseek",
        "sk-test",
        &provider_config.base_url,
        adapter.clone(),
    )
    .with_model(model);

    OpenAiCompatibleEventConverter::new(cfg, adapter)
}

fn run_converter(lines: Vec<String>, model: &str) -> Vec<ChatStreamEvent> {
    let conv = make_deepseek_converter(model);

    let mut events: Vec<ChatStreamEvent> = Vec::new();
    for (i, line) in lines.into_iter().enumerate() {
        let ev = Event {
            event: "".to_string(),
            data: line,
            id: i.to_string(),
            retry: None,
        };

        let out = futures::executor::block_on(conv.convert_event(ev));
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

#[test]
fn deepseek_text_stream_emits_content_usage_and_stream_end() {
    let path = fixtures_dir().join("deepseek-text.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);

    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let events = run_converter(lines, "deepseek-chat");

    let has_start = events
        .iter()
        .any(|e| matches!(e, ChatStreamEvent::StreamStart { .. }));
    assert!(has_start, "expected StreamStart");

    let content: String = events
        .iter()
        .filter_map(|e| match e {
            ChatStreamEvent::ContentDelta { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert!(!content.is_empty(), "expected non-empty content stream");

    let usage = events.iter().find_map(|e| match e {
        ChatStreamEvent::UsageUpdate { usage } => Some(usage),
        _ => None,
    });
    assert!(
        usage.is_some_and(|u| u.prompt_tokens == 13
            && u.completion_tokens == 400
            && u.total_tokens == 413),
        "expected final usage to match fixture"
    );

    let ended = events
        .iter()
        .any(|e| matches!(e, ChatStreamEvent::StreamEnd { .. }));
    assert!(ended, "expected StreamEnd");
}

#[test]
fn deepseek_reasoning_stream_emits_thinking_then_text() {
    let path = fixtures_dir().join("deepseek-reasoning.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);

    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let events = run_converter(lines, "deepseek-reasoner");

    let first_thinking = events
        .iter()
        .position(|e| matches!(e, ChatStreamEvent::ThinkingDelta { .. }));
    let first_text = events
        .iter()
        .position(|e| matches!(e, ChatStreamEvent::ContentDelta { .. }));

    assert!(first_thinking.is_some(), "expected ThinkingDelta");
    assert!(first_text.is_some(), "expected ContentDelta");
    assert!(
        first_thinking.unwrap() < first_text.unwrap(),
        "expected thinking deltas before text deltas"
    );

    let usage = events.iter().find_map(|e| match e {
        ChatStreamEvent::UsageUpdate { usage } => Some(usage),
        _ => None,
    });
    assert!(
        usage.is_some_and(|u| {
            u.prompt_tokens == 18
                && u.completion_tokens == 219
                && u.total_tokens == 237
                && u.completion_tokens_details
                    .as_ref()
                    .and_then(|d| d.reasoning_tokens)
                    == Some(205)
        }),
        "expected reasoning token details from fixture"
    );
}

#[test]
fn deepseek_tool_call_stream_emits_tool_call_deltas() {
    let path = fixtures_dir().join("deepseek-tool-call.chunks.txt");
    assert!(path.exists(), "fixture missing: {:?}", path);

    let lines = read_fixture_lines(&path);
    assert!(!lines.is_empty(), "fixture empty");

    let events = run_converter(lines, "deepseek-reasoner");

    let mut tool_call_id: Option<String> = None;
    let mut tool_name: Option<String> = None;
    let mut args = String::new();

    for e in &events {
        if let ChatStreamEvent::ToolCallDelta {
            id,
            function_name,
            arguments_delta,
            ..
        } = e
        {
            tool_call_id.get_or_insert_with(|| id.clone());
            if let Some(name) = function_name {
                tool_name.get_or_insert_with(|| name.clone());
            }
            if let Some(delta) = arguments_delta {
                args.push_str(delta);
            }
        }
    }

    assert_eq!(
        tool_name.as_deref(),
        Some("weather"),
        "expected weather tool call"
    );
    assert!(
        tool_call_id.is_some(),
        "expected tool call id from stream fixture"
    );
    assert!(
        args.contains("San Francisco"),
        "expected tool arguments to contain location, got: {args}"
    );

    let usage = events.iter().find_map(|e| match e {
        ChatStreamEvent::UsageUpdate { usage } => Some(usage),
        _ => None,
    });
    assert!(
        usage.is_some_and(|u| {
            u.prompt_tokens == 339
                && u.completion_tokens == 83
                && u.total_tokens == 422
                && u.prompt_tokens_details
                    .as_ref()
                    .and_then(|d| d.cached_tokens)
                    == Some(320)
        }),
        "expected cached token details from fixture"
    );
}
