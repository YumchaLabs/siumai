#![cfg(feature = "openai")]

use eventsource_stream::Event;
use futures_util::{StreamExt, stream};
use serde::Serialize;
use siumai::experimental::bridge::{
    BridgeMode, BridgeTarget, bridge_chat_stream_to_openai_chat_completions_sse,
};
use siumai::prelude::unified::{ChatByteStream, ChatStreamEvent, SseEventConverter};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("openai")
        .join("chat-completions")
}

fn read_fixture_lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .expect("read fixture file")
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() {
                return None;
            }

            Some(
                line.strip_prefix("data: ")
                    .unwrap_or(line)
                    .trim()
                    .to_string(),
            )
        })
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

fn make_openai_chat_converter(
    model: &str,
) -> siumai::protocol::openai::compat::streaming::OpenAiCompatibleEventConverter {
    use siumai::protocol::openai::compat::openai_config::OpenAiCompatibleConfig;
    use siumai::protocol::openai::compat::provider_registry::{
        ConfigurableAdapter, ProviderConfig, ProviderFieldMappings,
    };

    let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
        id: "openai".to_string(),
        name: "OpenAI".to_string(),
        base_url: "https://api.openai.com/v1".to_string(),
        field_mappings: ProviderFieldMappings::default(),
        capabilities: vec!["tools".to_string()],
        default_model: Some(model.to_string()),
        supports_reasoning: false,
        api_key_env: None,
        api_key_env_aliases: vec![],
    }));

    let cfg = OpenAiCompatibleConfig::new(
        "openai",
        "sk-siumai-fixture",
        "https://api.openai.com/v1",
        adapter.clone(),
    )
    .with_model(model);

    siumai::protocol::openai::compat::streaming::OpenAiCompatibleEventConverter::new(cfg, adapter)
}

fn decode_openai_chat_completions(model: &str, lines: Vec<String>) -> Vec<ChatStreamEvent> {
    let conv = make_openai_chat_converter(model);
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

    for item in conv.handle_stream_end_events() {
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
struct ToolCallSummary {
    id: String,
    name: Option<String>,
    arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct OpenAiChatStreamSummary {
    has_stream_start: bool,
    start_id: Option<String>,
    start_model: Option<String>,
    finish_reason: Option<String>,
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    total_tokens: Option<u32>,
    text: String,
    tool_calls: Vec<ToolCallSummary>,
}

fn finish_reason_to_string(reason: &siumai::prelude::unified::FinishReason) -> Option<String> {
    serde_json::to_value(reason)
        .ok()
        .and_then(|value| value.as_str().map(ToString::to_string))
}

fn summarize_openai_chat_events(events: &[ChatStreamEvent]) -> OpenAiChatStreamSummary {
    let mut summary = OpenAiChatStreamSummary {
        has_stream_start: false,
        start_id: None,
        start_model: None,
        finish_reason: None,
        prompt_tokens: None,
        completion_tokens: None,
        total_tokens: None,
        text: String::new(),
        tool_calls: Vec::new(),
    };
    let mut tool_calls: BTreeMap<String, ToolCallSummary> = BTreeMap::new();

    for event in events {
        match event {
            ChatStreamEvent::StreamStart { metadata } => {
                summary.has_stream_start = true;
                summary.start_id = metadata.id.clone();
                summary.start_model = metadata.model.clone();
            }
            ChatStreamEvent::ContentDelta { delta, .. } => {
                summary.text.push_str(delta);
            }
            ChatStreamEvent::ToolCallDelta {
                id,
                function_name,
                arguments_delta,
                ..
            } => {
                let entry = tool_calls
                    .entry(id.clone())
                    .or_insert_with(|| ToolCallSummary {
                        id: id.clone(),
                        name: None,
                        arguments: String::new(),
                    });

                if let Some(function_name) = function_name.clone()
                    && !function_name.is_empty()
                {
                    entry.name = Some(function_name);
                }
                if let Some(arguments_delta) = arguments_delta {
                    entry.arguments.push_str(arguments_delta);
                }
            }
            ChatStreamEvent::UsageUpdate { usage } => {
                if usage.prompt_tokens > 0 || usage.completion_tokens > 0 {
                    summary.prompt_tokens = Some(usage.prompt_tokens);
                    summary.completion_tokens = Some(usage.completion_tokens);
                    summary.total_tokens = Some(usage.total_tokens);
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
                    summary.total_tokens = Some(usage.total_tokens);
                }
            }
            _ => {}
        }
    }

    summary.tool_calls = tool_calls.into_values().collect();
    summary
}

async fn roundtrip_summary(path: &str, model: &str) -> OpenAiChatStreamSummary {
    let fixture = fixtures_dir().join(path);
    let original = decode_openai_chat_completions(model, read_fixture_lines(&fixture));
    let bridged = bridge_chat_stream_to_openai_chat_completions_sse(
        stream::iter(original.into_iter().map(Ok)),
        Some(BridgeTarget::OpenAiChatCompletions),
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
    let roundtripped =
        decode_openai_chat_completions(model, extract_sse_data_payload_lines(&bytes));
    summarize_openai_chat_events(&roundtripped)
}

fn fixture_summary(path: &str, model: &str) -> OpenAiChatStreamSummary {
    let fixture = fixtures_dir().join(path);
    summarize_openai_chat_events(&decode_openai_chat_completions(
        model,
        read_fixture_lines(&fixture),
    ))
}

#[tokio::test]
async fn openai_chat_completions_stream_bridge_roundtrip_fixture_summary_cases_match() {
    let summary_cases = [
        ("role_content_finish_reason.sse", "gpt-4o-mini"),
        ("tool_calls_arguments_usage.sse", "gpt-4o-mini"),
        ("multiple_tool_calls.sse", "gpt-4o-mini"),
        ("azure-model-router.1.chunks.txt", "test-azure-model-router"),
    ];

    for (case, model) in summary_cases {
        assert_eq!(
            roundtrip_summary(case, model).await,
            fixture_summary(case, model),
            "fixture case: {}",
            fixtures_dir().join(case).display()
        );
    }
}
