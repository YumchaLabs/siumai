#![cfg(any(feature = "google", feature = "google-vertex"))]

use eventsource_stream::Event;
use futures_util::{StreamExt, stream};
use serde::Serialize;
use serde_json::Value;
use siumai::experimental::bridge::{
    BridgeMode, BridgeTarget, bridge_chat_stream_to_gemini_generate_content_sse,
};
use siumai::experimental::core::{ProviderContext, ProviderSpec};
use siumai::experimental::execution::transformers::stream::StreamChunkTransformer;
use siumai::prelude::unified::{ChatByteStream, ChatRequest, ChatStreamEvent, CommonParams};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("google")
        .join("generative-ai-stream")
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

fn run_converter(
    lines: Vec<String>,
    converter: &dyn StreamChunkTransformer,
) -> Vec<ChatStreamEvent> {
    let mut events = Vec::new();

    for (index, line) in lines.into_iter().enumerate() {
        let event = Event {
            event: "".to_string(),
            data: line,
            id: index.to_string(),
            retry: None,
        };
        let out = futures::executor::block_on(converter.convert_event(event));
        for item in out {
            match item {
                Ok(evt) => events.push(evt),
                Err(err) => panic!("failed to convert chunk: {err:?}"),
            }
        }
    }

    for item in converter.handle_stream_end_events() {
        match item {
            Ok(evt) => events.push(evt),
            Err(err) => panic!("failed to finalize stream: {err:?}"),
        }
    }

    events
}

#[cfg(feature = "google")]
fn decode_google(lines: Vec<String>) -> Vec<ChatStreamEvent> {
    let ctx = ProviderContext::new(
        "google",
        "https://generativelanguage.googleapis.com/v1beta",
        Some("test-api-key".to_string()),
        HashMap::new(),
    );
    let req = ChatRequest {
        stream: true,
        common_params: CommonParams {
            model: "gemini-2.5-pro".to_string(),
            ..Default::default()
        },
        ..Default::default()
    };
    let spec =
        siumai::experimental::providers::gemini::standards::gemini::GeminiChatStandard::new()
            .create_spec("google");
    let bundle = spec.choose_chat_transformers(&req, &ctx);
    let converter = bundle.stream.expect("expected google stream transformer");

    run_converter(lines, converter.as_ref())
}

#[cfg(feature = "google-vertex")]
fn decode_vertex(lines: Vec<String>) -> Vec<ChatStreamEvent> {
    let mut extra_headers = HashMap::new();
    extra_headers.insert("Authorization".to_string(), "Bearer token".to_string());
    let ctx = ProviderContext::new(
        "vertex",
        "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/test-project/locations/us-central1/publishers/google",
        None,
        extra_headers,
    );
    let req = ChatRequest {
        stream: true,
        common_params: CommonParams {
            model: "gemini-2.5-pro".to_string(),
            ..Default::default()
        },
        ..Default::default()
    };
    let spec = siumai::experimental::providers::google_vertex::standards::vertex_generative_ai::VertexGenerativeAiStandard::new()
        .create_spec("vertex");
    let bundle = spec.choose_chat_transformers(&req, &ctx);
    let converter = bundle.stream.expect("expected vertex stream transformer");

    run_converter(lines, converter.as_ref())
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
struct ReasoningMarkerSummary {
    id: Option<String>,
    provider_namespace: Option<String>,
    thought_signature: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct ReasoningDeltaSummary {
    id: Option<String>,
    delta: String,
    provider_namespace: Option<String>,
    thought_signature: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct GeminiStreamSummary {
    has_stream_start: bool,
    finish_reason: Option<String>,
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    total_tokens: Option<u32>,
    text: String,
    reasoning_starts: Vec<ReasoningMarkerSummary>,
    reasoning_deltas: Vec<ReasoningDeltaSummary>,
    reasoning_ends: Vec<Option<String>>,
    provider_tool_calls: BTreeMap<String, usize>,
    provider_tool_results: BTreeMap<String, usize>,
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

fn thought_signature_metadata(data: &Value) -> (Option<String>, Option<String>) {
    let Some(provider_metadata) = data.get("providerMetadata").and_then(Value::as_object) else {
        return (None, None);
    };

    for preferred in ["google", "vertex"] {
        if let Some(value) = provider_metadata.get(preferred) {
            return (
                Some(preferred.to_string()),
                value
                    .get("thoughtSignature")
                    .and_then(Value::as_str)
                    .map(ToString::to_string),
            );
        }
    }

    provider_metadata
        .iter()
        .find_map(|(namespace, value)| {
            value
                .get("thoughtSignature")
                .and_then(Value::as_str)
                .map(|sig| (Some(namespace.clone()), Some(sig.to_string())))
        })
        .unwrap_or((None, None))
}

fn reasoning_marker_from_custom(data: &Value) -> ReasoningMarkerSummary {
    let (provider_namespace, thought_signature) = thought_signature_metadata(data);
    ReasoningMarkerSummary {
        id: data
            .get("id")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        provider_namespace,
        thought_signature,
    }
}

fn reasoning_delta_from_custom(data: &Value) -> ReasoningDeltaSummary {
    let (provider_namespace, thought_signature) = thought_signature_metadata(data);
    ReasoningDeltaSummary {
        id: data
            .get("id")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        delta: data
            .get("delta")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        provider_namespace,
        thought_signature,
    }
}

fn summarize_gemini_events(events: &[ChatStreamEvent]) -> GeminiStreamSummary {
    let mut summary = GeminiStreamSummary {
        has_stream_start: false,
        finish_reason: None,
        prompt_tokens: None,
        completion_tokens: None,
        total_tokens: None,
        text: String::new(),
        reasoning_starts: Vec::new(),
        reasoning_deltas: Vec::new(),
        reasoning_ends: Vec::new(),
        provider_tool_calls: BTreeMap::new(),
        provider_tool_results: BTreeMap::new(),
    };

    for event in events {
        match event {
            ChatStreamEvent::StreamStart { .. } => {
                summary.has_stream_start = true;
            }
            ChatStreamEvent::ContentDelta { delta, .. } => {
                summary.text.push_str(delta);
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
            ChatStreamEvent::Custom { data, .. } => {
                match data.get("type").and_then(Value::as_str) {
                    Some("reasoning-start") => {
                        summary
                            .reasoning_starts
                            .push(reasoning_marker_from_custom(data));
                    }
                    Some("reasoning-delta") => {
                        summary
                            .reasoning_deltas
                            .push(reasoning_delta_from_custom(data));
                    }
                    Some("reasoning-end") => {
                        summary.reasoning_ends.push(
                            data.get("id")
                                .and_then(Value::as_str)
                                .map(ToString::to_string),
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
                    _ => {}
                }
            }
            _ => {}
        }
    }

    summary
}

async fn roundtrip_summary(
    path: &str,
    decode: fn(Vec<String>) -> Vec<ChatStreamEvent>,
) -> GeminiStreamSummary {
    let fixture = fixtures_dir().join(path);
    let original = decode(read_fixture_lines(&fixture));
    let bridged = bridge_chat_stream_to_gemini_generate_content_sse(
        stream::iter(original.into_iter().map(Ok)),
        Some(BridgeTarget::GeminiGenerateContent),
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
    let roundtripped = decode(extract_sse_data_payload_lines(&bytes));
    summarize_gemini_events(&roundtripped)
}

fn fixture_summary(
    path: &str,
    decode: fn(Vec<String>) -> Vec<ChatStreamEvent>,
) -> GeminiStreamSummary {
    let fixture = fixtures_dir().join(path);
    summarize_gemini_events(&decode(read_fixture_lines(&fixture)))
}

#[cfg(feature = "google")]
#[tokio::test]
async fn google_generate_content_stream_bridge_roundtrip_fixture_summary_cases_match() {
    let summary_cases = [
        "google-code-execution.1.chunks.txt",
        "google-thought-signature-reasoning.1.chunks.txt",
    ];

    for case in summary_cases {
        assert_eq!(
            roundtrip_summary(case, decode_google).await,
            fixture_summary(case, decode_google),
            "fixture case: {}",
            fixtures_dir().join(case).display()
        );
    }
}

#[cfg(feature = "google-vertex")]
#[tokio::test]
async fn vertex_generate_content_stream_bridge_roundtrip_fixture_summary_cases_match() {
    let summary_cases = [
        "google-code-execution.1.chunks.txt",
        "google-thought-signature-reasoning.1.chunks.txt",
    ];

    for case in summary_cases {
        assert_eq!(
            roundtrip_summary(case, decode_vertex).await,
            fixture_summary(case, decode_vertex),
            "fixture case: {}",
            fixtures_dir().join(case).display()
        );
    }
}
