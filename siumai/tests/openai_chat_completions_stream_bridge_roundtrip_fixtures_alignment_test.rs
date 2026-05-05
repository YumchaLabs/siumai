#![cfg(feature = "openai")]

use eventsource_stream::Event;
use futures_util::{StreamExt, stream};
use serde::Serialize;
use siumai::experimental::bridge::{
    BridgeMode, BridgeTarget, bridge_chat_stream_to_openai_chat_completions_sse,
};
use siumai::prelude::unified::{ChatByteStream, ChatStreamEvent, SseEventConverter};
use siumai_core::types::ChatStreamPart;
use siumai_core::types::SourcePart;
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
struct SourceSummary {
    id: String,
    url: String,
    title: Option<String>,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct ResponseMetadataSummary {
    id: Option<String>,
    model: Option<String>,
    created_unix: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct ResponseEnvelopeSummary {
    id: Option<String>,
    model: Option<String>,
    system_fingerprint: Option<String>,
    service_tier: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct PredictionTokensSummary {
    accepted_prediction_tokens: Option<u64>,
    rejected_prediction_tokens: Option<u64>,
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
                    summary.prompt_tokens = usage.prompt_tokens();
                    summary.completion_tokens = usage.completion_tokens();
                    summary.total_tokens = usage.total_tokens();
                }
            }
            ChatStreamEvent::Part { part } | ChatStreamEvent::PartWithReplay { part, .. } => {
                match part {
                    ChatStreamPart::TextDelta { delta, .. } => {
                        summary.text.push_str(delta);
                    }
                    ChatStreamPart::ToolInputStart { id, tool_name, .. } => {
                        let entry =
                            tool_calls
                                .entry(id.clone())
                                .or_insert_with(|| ToolCallSummary {
                                    id: id.clone(),
                                    name: None,
                                    arguments: String::new(),
                                });
                        if !tool_name.is_empty() {
                            entry.name = Some(tool_name.clone());
                        }
                    }
                    ChatStreamPart::ToolInputDelta { id, delta, .. } => {
                        let entry =
                            tool_calls
                                .entry(id.clone())
                                .or_insert_with(|| ToolCallSummary {
                                    id: id.clone(),
                                    name: None,
                                    arguments: String::new(),
                                });
                        entry.arguments.push_str(delta);
                    }
                    ChatStreamPart::ToolCall(call) => {
                        let entry =
                            tool_calls
                                .entry(call.tool_call_id.clone())
                                .or_insert_with(|| ToolCallSummary {
                                    id: call.tool_call_id.clone(),
                                    name: None,
                                    arguments: String::new(),
                                });
                        entry.name = Some(call.tool_name.clone());
                        entry.arguments = call.input.clone();
                    }
                    ChatStreamPart::Finish {
                        usage,
                        finish_reason,
                        ..
                    } => {
                        summary.finish_reason = finish_reason_to_string(&finish_reason.unified)
                            .or_else(|| summary.finish_reason.clone());
                        summary.prompt_tokens = usage.prompt_tokens().or(summary.prompt_tokens);
                        summary.completion_tokens =
                            usage.completion_tokens().or(summary.completion_tokens);
                        summary.total_tokens = usage.total_tokens().or(summary.total_tokens);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    summary.tool_calls = tool_calls.into_values().collect();
    summary
}

fn summarize_openai_chat_sources(events: &[ChatStreamEvent]) -> Vec<SourceSummary> {
    events
        .iter()
        .filter_map(|event| match event {
            ChatStreamEvent::Part {
                part:
                    siumai::prelude::unified::ChatStreamPart::Source {
                        id,
                        source: SourcePart::Url { url, title },
                        ..
                    },
            } => Some(SourceSummary {
                id: id.clone(),
                url: url.clone(),
                title: title.clone(),
            }),
            _ => None,
        })
        .collect()
}

fn summarize_openai_chat_response_metadata(
    events: &[ChatStreamEvent],
) -> Vec<ResponseMetadataSummary> {
    events
        .iter()
        .filter_map(|event| match event {
            ChatStreamEvent::Part {
                part: ChatStreamPart::ResponseMetadata(metadata),
            } => Some(ResponseMetadataSummary {
                id: metadata.id.clone(),
                model: metadata.model.clone(),
                created_unix: metadata.created.map(|value| value.timestamp()),
            }),
            _ => None,
        })
        .collect()
}

fn summarize_openai_chat_logprobs(events: &[ChatStreamEvent]) -> Option<serde_json::Value> {
    events.iter().find_map(|event| match event {
        ChatStreamEvent::StreamEnd { response } => response
            .provider_metadata
            .as_ref()
            .and_then(|providers| providers.get("openai"))
            .and_then(|meta| meta.get("logprobs"))
            .cloned(),
        _ => None,
    })
}

fn summarize_openai_chat_prediction_tokens(
    events: &[ChatStreamEvent],
) -> Option<PredictionTokensSummary> {
    events.iter().find_map(|event| match event {
        ChatStreamEvent::StreamEnd { response } => {
            let openai = response
                .provider_metadata
                .as_ref()
                .and_then(|providers| providers.get("openai"))?;
            let accepted_prediction_tokens = openai
                .get("acceptedPredictionTokens")
                .and_then(serde_json::Value::as_u64);
            let rejected_prediction_tokens = openai
                .get("rejectedPredictionTokens")
                .and_then(serde_json::Value::as_u64);

            if accepted_prediction_tokens.is_none() && rejected_prediction_tokens.is_none() {
                None
            } else {
                Some(PredictionTokensSummary {
                    accepted_prediction_tokens,
                    rejected_prediction_tokens,
                })
            }
        }
        _ => None,
    })
}

fn summarize_openai_chat_terminal_envelope(
    events: &[ChatStreamEvent],
) -> Option<ResponseEnvelopeSummary> {
    events.iter().find_map(|event| match event {
        ChatStreamEvent::StreamEnd { response } => Some(ResponseEnvelopeSummary {
            id: response.id.clone(),
            model: response.model.clone(),
            system_fingerprint: response.system_fingerprint.clone(),
            service_tier: response.service_tier.clone(),
        }),
        _ => None,
    })
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
        ("annotations_url_sources.sse", "gpt-4o-mini"),
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

#[tokio::test]
async fn openai_chat_completions_stream_bridge_roundtrip_preserves_url_source_annotations() {
    let case = "annotations_url_sources.sse";
    let model = "gpt-4o-mini";
    let fixture = fixtures_dir().join(case);

    let original = decode_openai_chat_completions(model, read_fixture_lines(&fixture));
    let expected_sources = summarize_openai_chat_sources(&original);
    assert_eq!(
        expected_sources,
        vec![
            SourceSummary {
                id: "source_chatcmpl-annotations-1_0".to_string(),
                url: "https://example.com/rust".to_string(),
                title: Some("Rust".to_string()),
            },
            SourceSummary {
                id: "source_chatcmpl-annotations-1_1".to_string(),
                url: "https://example.com/book".to_string(),
                title: Some("Book".to_string()),
            },
        ]
    );

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

    assert_eq!(
        summarize_openai_chat_sources(&roundtripped),
        expected_sources,
        "fixture case: {}",
        fixture.display()
    );
}

#[tokio::test]
async fn openai_chat_completions_stream_bridge_roundtrip_preserves_response_metadata_and_logprobs()
{
    let case = "metadata_logprobs.sse";
    let model = "gpt-4o-mini";
    let fixture = fixtures_dir().join(case);

    let original = decode_openai_chat_completions(model, read_fixture_lines(&fixture));
    let expected_metadata = summarize_openai_chat_response_metadata(&original);
    let expected_logprobs = summarize_openai_chat_logprobs(&original);
    let expected_prediction_tokens = summarize_openai_chat_prediction_tokens(&original);
    let expected_terminal = summarize_openai_chat_terminal_envelope(&original);

    assert_eq!(
        expected_metadata,
        vec![ResponseMetadataSummary {
            id: Some("chatcmpl-meta-logprobs-1".to_string()),
            model: Some("gpt-4o-mini".to_string()),
            created_unix: Some(1731234567),
        }]
    );
    assert_eq!(
        expected_logprobs,
        Some(serde_json::json!([
            {
                "token": "Hello",
                "logprob": -0.01,
                "bytes": [72, 101, 108, 108, 111],
                "top_logprobs": [
                    { "token": "Hello", "logprob": -0.01 },
                    { "token": "Hi", "logprob": -1.5 }
                ]
            },
            {
                "token": "!",
                "logprob": -0.2,
                "bytes": [33],
                "top_logprobs": [
                    { "token": "!", "logprob": -0.2 }
                ]
            }
        ]))
    );
    assert_eq!(
        expected_prediction_tokens,
        Some(PredictionTokensSummary {
            accepted_prediction_tokens: Some(5),
            rejected_prediction_tokens: Some(6),
        })
    );
    assert_eq!(
        expected_terminal,
        Some(ResponseEnvelopeSummary {
            id: Some("chatcmpl-meta-logprobs-1".to_string()),
            model: Some("gpt-4o-mini".to_string()),
            system_fingerprint: Some("fp_meta_logprobs".to_string()),
            service_tier: Some("priority".to_string()),
        })
    );

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

    assert_eq!(
        summarize_openai_chat_response_metadata(&roundtripped),
        expected_metadata,
        "fixture case: {}",
        fixture.display()
    );
    assert_eq!(
        summarize_openai_chat_logprobs(&roundtripped),
        expected_logprobs,
        "fixture case: {}",
        fixture.display()
    );
    assert_eq!(
        summarize_openai_chat_prediction_tokens(&roundtripped),
        expected_prediction_tokens,
        "fixture case: {}",
        fixture.display()
    );
    assert_eq!(
        summarize_openai_chat_terminal_envelope(&roundtripped),
        expected_terminal,
        "fixture case: {}",
        fixture.display()
    );
}
