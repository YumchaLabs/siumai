#![cfg(all(feature = "server", feature = "openai"))]

use axum::{
    Router,
    body::{Body, to_bytes},
    extract::State,
    http::{Request, StatusCode},
    response::Response,
    routing::get,
};
#[cfg(any(feature = "anthropic", feature = "google"))]
use eventsource_stream::Event;
use futures::stream;
#[cfg(feature = "anthropic")]
use siumai::prelude::unified::ProviderDefinedTool;
#[cfg(any(feature = "anthropic", feature = "google"))]
use siumai::prelude::unified::SseEventConverter;
#[cfg(any(feature = "anthropic", feature = "google"))]
use siumai::prelude::unified::Tool;
use siumai::prelude::unified::{ChatResponse, ChatStream, ChatStreamEvent, MessageContent};
use siumai_extras::server::{
    GatewayBridgePolicy,
    axum::{
        TargetJsonFormat, TargetSseFormat, TranscodeJsonOptions, TranscodeSseOptions,
        to_transcoded_json_response, to_transcoded_sse_response,
    },
};
#[cfg(any(feature = "anthropic", feature = "google"))]
use std::path::{Path, PathBuf};
use tower::ServiceExt;

#[derive(Clone)]
struct AppState {
    policy: GatewayBridgePolicy,
}

fn openai_app() -> Router {
    let state = AppState {
        policy: GatewayBridgePolicy::new(siumai::experimental::bridge::BridgeMode::BestEffort)
            .with_bridge_headers(true)
            .with_bridge_warning_headers(true),
    };

    Router::new()
        .route("/json", get(json_route))
        .route("/sse", get(sse_route))
        .with_state(state)
}

#[cfg(feature = "anthropic")]
fn anthropic_app() -> Router {
    let state = AppState {
        policy: GatewayBridgePolicy::new(siumai::experimental::bridge::BridgeMode::BestEffort)
            .with_bridge_headers(true)
            .with_bridge_warning_headers(true),
    };

    Router::new()
        .route("/json", get(anthropic_json_route))
        .route("/sse", get(anthropic_sse_route))
        .with_state(state)
}

#[cfg(feature = "google")]
fn gemini_app() -> Router {
    let state = AppState {
        policy: GatewayBridgePolicy::new(siumai::experimental::bridge::BridgeMode::BestEffort)
            .with_bridge_headers(true)
            .with_bridge_warning_headers(true),
    };

    Router::new()
        .route("/json", get(gemini_json_route))
        .route("/sse", get(gemini_sse_route))
        .with_state(state)
}

#[cfg(feature = "anthropic")]
fn cross_protocol_app() -> Router {
    let state = AppState {
        policy: GatewayBridgePolicy::new(siumai::experimental::bridge::BridgeMode::BestEffort)
            .with_bridge_headers(true)
            .with_bridge_warning_headers(true),
    };

    Router::new()
        .route(
            "/anthropic-to-openai",
            get(anthropic_fixture_to_openai_sse_route),
        )
        .route(
            "/openai-to-anthropic",
            get(openai_fixture_to_anthropic_sse_route),
        )
        .with_state(state)
}

#[cfg(feature = "google")]
fn gemini_cross_protocol_app() -> Router {
    let state = AppState {
        policy: GatewayBridgePolicy::new(siumai::experimental::bridge::BridgeMode::BestEffort)
            .with_bridge_headers(true)
            .with_bridge_warning_headers(true),
    };

    Router::new()
        .route("/gemini-to-openai", get(gemini_fixture_to_openai_sse_route))
        .route("/openai-to-gemini", get(openai_fixture_to_gemini_sse_route))
        .route(
            "/openai-to-gemini-strict",
            get(openai_fixture_to_gemini_strict_sse_route),
        )
        .with_state(state)
}

#[cfg(all(feature = "anthropic", feature = "google"))]
fn anthropic_gemini_cross_protocol_app() -> Router {
    let state = AppState {
        policy: GatewayBridgePolicy::new(siumai::experimental::bridge::BridgeMode::BestEffort)
            .with_bridge_headers(true)
            .with_bridge_warning_headers(true),
    };

    Router::new()
        .route(
            "/anthropic-to-gemini",
            get(anthropic_fixture_to_gemini_sse_route),
        )
        .route(
            "/gemini-to-anthropic",
            get(gemini_fixture_to_anthropic_sse_route),
        )
        .with_state(state)
}

async fn json_route(State(state): State<AppState>) -> Response<Body> {
    to_transcoded_json_response(
        ChatResponse::new(MessageContent::Text("gateway ok".to_string())),
        TargetJsonFormat::OpenAiResponses,
        TranscodeJsonOptions::default().with_policy(state.policy),
    )
}

async fn sse_route(State(state): State<AppState>) -> Response<Body> {
    let response = ChatResponse::new(MessageContent::Text("gateway ok".to_string()));
    let stream: ChatStream = Box::pin(stream::iter(vec![
        Ok(ChatStreamEvent::ContentDelta {
            delta: "gateway ok".to_string(),
            index: None,
        }),
        Ok(ChatStreamEvent::StreamEnd { response }),
    ]));

    to_transcoded_sse_response(
        stream,
        TargetSseFormat::OpenAiResponses,
        TranscodeSseOptions::default().with_policy(state.policy),
    )
}

#[cfg(feature = "anthropic")]
async fn anthropic_json_route(State(state): State<AppState>) -> Response<Body> {
    to_transcoded_json_response(
        ChatResponse::new(MessageContent::Text("gateway ok".to_string())),
        TargetJsonFormat::AnthropicMessages,
        TranscodeJsonOptions::default().with_policy(state.policy),
    )
}

#[cfg(feature = "anthropic")]
async fn anthropic_sse_route(State(state): State<AppState>) -> Response<Body> {
    let response = ChatResponse::new(MessageContent::Text("gateway ok".to_string()));
    let stream: ChatStream = Box::pin(stream::iter(vec![
        Ok(ChatStreamEvent::ContentDelta {
            delta: "gateway ok".to_string(),
            index: None,
        }),
        Ok(ChatStreamEvent::StreamEnd { response }),
    ]));

    to_transcoded_sse_response(
        stream,
        TargetSseFormat::AnthropicMessages,
        TranscodeSseOptions::default().with_policy(state.policy),
    )
}

#[cfg(feature = "google")]
async fn gemini_json_route(State(state): State<AppState>) -> Response<Body> {
    to_transcoded_json_response(
        ChatResponse::new(MessageContent::Text("gateway ok".to_string())),
        TargetJsonFormat::GeminiGenerateContent,
        TranscodeJsonOptions::default().with_policy(state.policy),
    )
}

#[cfg(feature = "google")]
async fn gemini_sse_route(State(state): State<AppState>) -> Response<Body> {
    let response = ChatResponse::new(MessageContent::Text("gateway ok".to_string()));
    let stream: ChatStream = Box::pin(stream::iter(vec![
        Ok(ChatStreamEvent::ContentDelta {
            delta: "gateway ok".to_string(),
            index: None,
        }),
        Ok(ChatStreamEvent::StreamEnd { response }),
    ]));

    to_transcoded_sse_response(
        stream,
        TargetSseFormat::GeminiGenerateContent,
        TranscodeSseOptions::default().with_policy(state.policy),
    )
}

#[cfg(feature = "anthropic")]
async fn anthropic_fixture_to_openai_sse_route(State(state): State<AppState>) -> Response<Body> {
    let upstream = decode_anthropic_fixture_stream(
        repo_fixtures_dir()
            .join("anthropic")
            .join("messages-stream")
            .join("anthropic-web-fetch-tool.1.chunks.txt"),
    );
    let stream: ChatStream = Box::pin(stream::iter(upstream.into_iter().map(Ok)));

    to_transcoded_sse_response(
        stream,
        TargetSseFormat::OpenAiResponses,
        TranscodeSseOptions::default().with_policy(state.policy),
    )
}

#[cfg(feature = "anthropic")]
async fn openai_fixture_to_anthropic_sse_route(State(state): State<AppState>) -> Response<Body> {
    let tools = vec![Tool::ProviderDefined(ProviderDefinedTool::new(
        "openai.web_search",
        "webSearch",
    ))];
    let upstream = decode_openai_fixture_stream(
        repo_fixtures_dir()
            .join("openai")
            .join("responses-stream")
            .join("web-search")
            .join("openai-web-search-tool.1.chunks.txt"),
        tools,
    );
    let stream: ChatStream = Box::pin(stream::iter(upstream.into_iter().map(Ok)));

    to_transcoded_sse_response(
        stream,
        TargetSseFormat::AnthropicMessages,
        TranscodeSseOptions::default().with_policy(state.policy),
    )
}

#[cfg(feature = "google")]
async fn gemini_fixture_to_openai_sse_route(State(state): State<AppState>) -> Response<Body> {
    let upstream = decode_gemini_fixture_stream(
        repo_fixtures_dir()
            .join("gemini")
            .join("simple_text_then_finish.sse"),
    );
    let stream: ChatStream = Box::pin(stream::iter(upstream.into_iter().map(Ok)));

    to_transcoded_sse_response(
        stream,
        TargetSseFormat::OpenAiResponses,
        TranscodeSseOptions::default()
            .with_bridge_source(siumai::experimental::bridge::BridgeTarget::GeminiGenerateContent)
            .with_policy(state.policy),
    )
}

#[cfg(feature = "google")]
async fn openai_fixture_to_gemini_sse_route(State(state): State<AppState>) -> Response<Body> {
    let upstream = decode_openai_fixture_stream(
        repo_fixtures_dir()
            .join("openai")
            .join("responses-stream")
            .join("text")
            .join("openai-text-deltas.1.chunks.txt"),
        Vec::new(),
    );
    let stream: ChatStream = Box::pin(stream::iter(upstream.into_iter().map(Ok)));

    to_transcoded_sse_response(
        stream,
        TargetSseFormat::GeminiGenerateContent,
        TranscodeSseOptions::default()
            .with_bridge_source(siumai::experimental::bridge::BridgeTarget::OpenAiResponses)
            .with_policy(state.policy),
    )
}

#[cfg(feature = "google")]
async fn openai_fixture_to_gemini_strict_sse_route(
    State(state): State<AppState>,
) -> Response<Body> {
    let upstream = decode_openai_fixture_stream(
        repo_fixtures_dir()
            .join("openai")
            .join("responses-stream")
            .join("text")
            .join("openai-text-deltas.1.chunks.txt"),
        Vec::new(),
    );
    let stream: ChatStream = Box::pin(stream::iter(upstream.into_iter().map(Ok)));

    to_transcoded_sse_response(
        stream,
        TargetSseFormat::GeminiGenerateContent,
        TranscodeSseOptions::default()
            .with_bridge_source(siumai::experimental::bridge::BridgeTarget::OpenAiResponses)
            .with_policy(state.policy)
            .with_bridge_mode_override(siumai::experimental::bridge::BridgeMode::Strict),
    )
}

#[cfg(all(feature = "anthropic", feature = "google"))]
async fn anthropic_fixture_to_gemini_sse_route(State(state): State<AppState>) -> Response<Body> {
    let upstream = decode_anthropic_fixture_stream(
        repo_fixtures_dir()
            .join("anthropic")
            .join("messages-stream")
            .join("anthropic-web-search-tool.1.chunks.txt"),
    );
    let stream: ChatStream = Box::pin(stream::iter(upstream.into_iter().map(Ok)));

    to_transcoded_sse_response(
        stream,
        TargetSseFormat::GeminiGenerateContent,
        TranscodeSseOptions::default()
            .with_bridge_source(siumai::experimental::bridge::BridgeTarget::AnthropicMessages)
            .with_policy(state.policy),
    )
}

#[cfg(all(feature = "anthropic", feature = "google"))]
async fn gemini_fixture_to_anthropic_sse_route(State(state): State<AppState>) -> Response<Body> {
    let upstream = decode_gemini_fixture_stream(
        repo_fixtures_dir()
            .join("gemini")
            .join("thought_then_text_stop.sse"),
    );
    let stream: ChatStream = Box::pin(stream::iter(upstream.into_iter().map(Ok)));

    to_transcoded_sse_response(
        stream,
        TargetSseFormat::AnthropicMessages,
        TranscodeSseOptions::default()
            .with_bridge_source(siumai::experimental::bridge::BridgeTarget::GeminiGenerateContent)
            .with_policy(state.policy),
    )
}

#[cfg(any(feature = "anthropic", feature = "google"))]
fn repo_fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .join("siumai")
        .join("tests")
        .join("fixtures")
}

#[cfg(any(feature = "anthropic", feature = "google"))]
fn read_fixture_lines(path: impl AsRef<Path>) -> Vec<String> {
    std::fs::read_to_string(path)
        .expect("read fixture file")
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.to_string())
        .collect()
}

#[cfg(feature = "google")]
fn read_gemini_sse_data_lines(path: impl AsRef<Path>) -> Vec<String> {
    std::fs::read_to_string(path)
        .expect("read fixture file")
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() {
                return None;
            }
            Some(line.trim_start_matches("data: ").trim().to_string())
        })
        .collect()
}

#[cfg(any(feature = "anthropic", feature = "google"))]
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

#[cfg(feature = "google")]
fn decode_gemini_sse_body(bytes: &[u8]) -> Vec<ChatStreamEvent> {
    decode_gemini_stream_lines(extract_sse_data_payload_lines(bytes))
}

#[cfg(feature = "google")]
fn decode_gemini_fixture_stream(path: impl AsRef<Path>) -> Vec<ChatStreamEvent> {
    decode_gemini_stream_lines(read_gemini_sse_data_lines(path))
}

#[cfg(feature = "google")]
fn decode_gemini_stream_lines(lines: Vec<String>) -> Vec<ChatStreamEvent> {
    let conv = siumai::protocol::gemini::streaming::GeminiEventConverter::new(
        siumai::protocol::gemini::types::GeminiConfig::default(),
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
                Err(err) => panic!("failed to convert gemini chunk: {err:?}"),
            }
        }
    }

    while let Some(item) = conv.handle_stream_end() {
        match item {
            Ok(evt) => events.push(evt),
            Err(err) => panic!("failed to finalize gemini stream: {err:?}"),
        }
    }

    events
}

#[cfg(feature = "anthropic")]
fn decode_anthropic_fixture_stream(path: impl AsRef<Path>) -> Vec<ChatStreamEvent> {
    let conv = siumai::protocol::anthropic::streaming::AnthropicEventConverter::new(
        siumai::protocol::anthropic::params::AnthropicParams::default(),
    );
    let mut events = Vec::new();

    for (index, line) in read_fixture_lines(path).into_iter().enumerate() {
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
                Err(err) => panic!("failed to convert anthropic chunk: {err:?}"),
            }
        }
    }

    while let Some(item) = conv.handle_stream_end() {
        match item {
            Ok(evt) => events.push(evt),
            Err(err) => panic!("failed to finalize anthropic stream: {err:?}"),
        }
    }

    events
}

#[cfg(any(feature = "anthropic", feature = "google"))]
fn decode_openai_fixture_stream(path: impl AsRef<Path>, tools: Vec<Tool>) -> Vec<ChatStreamEvent> {
    let conv = siumai::provider_ext::openai::ext::OpenAiResponsesEventConverter::new()
        .with_request_tools(&tools);
    let mut events = Vec::new();

    for (index, line) in read_fixture_lines(path).into_iter().enumerate() {
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
                Err(err) => panic!("failed to convert openai chunk: {err:?}"),
            }
        }
    }

    while let Some(item) = conv.handle_stream_end() {
        match item {
            Ok(evt) => events.push(evt),
            Err(err) => panic!("failed to finalize openai stream: {err:?}"),
        }
    }

    events
}

#[cfg(feature = "anthropic")]
fn parse_anthropic_sse_json_frames(bytes: &[u8]) -> Vec<serde_json::Value> {
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
            serde_json::from_str::<serde_json::Value>(data).ok()
        })
        .collect()
}

fn parse_sse_json_frames(bytes: &[u8]) -> Vec<serde_json::Value> {
    extract_sse_data_payload_lines(bytes)
        .into_iter()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(&line).ok())
        .collect()
}

#[cfg(any(feature = "anthropic", feature = "google"))]
fn decode_openai_responses_sse_body(bytes: &[u8]) -> Vec<ChatStreamEvent> {
    let conv = siumai::protocol::openai::responses_sse::OpenAiResponsesEventConverter::new();
    let mut events = Vec::new();

    for (index, line) in extract_sse_data_payload_lines(bytes)
        .into_iter()
        .enumerate()
    {
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
                Err(err) => panic!("failed to decode openai response body: {err:?}"),
            }
        }
    }

    while let Some(item) = conv.handle_stream_end() {
        match item {
            Ok(evt) => events.push(evt),
            Err(err) => panic!("failed to finalize decoded openai response body: {err:?}"),
        }
    }

    events
}

fn collect_text_deltas(events: &[ChatStreamEvent]) -> String {
    events
        .iter()
        .filter_map(|event| match event {
            ChatStreamEvent::ContentDelta { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

#[cfg(feature = "anthropic")]
fn v3_tool_parts(
    events: &[ChatStreamEvent],
    kind: &str,
    tool_name: &str,
) -> Vec<serde_json::Value> {
    events
        .iter()
        .filter_map(|event| match event {
            ChatStreamEvent::Custom { data, .. }
                if data.get("type") == Some(&serde_json::json!(kind))
                    && data.get("toolName") == Some(&serde_json::json!(tool_name)) =>
            {
                Some(data.clone())
            }
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn gateway_json_route_smoke_emits_provider_json() {
    let response = openai_app()
        .oneshot(
            Request::builder()
                .uri("/json")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("router response");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "application/json");
    assert_eq!(
        response.headers()["x-siumai-bridge-target"],
        "openai-responses"
    );
    assert_eq!(response.headers()["x-siumai-bridge-mode"], "best-effort");

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("json body");
    assert_eq!(json["object"], "response");
    assert_eq!(json["status"], "completed");
    assert_eq!(json["output"][0]["content"][0]["text"], "gateway ok");
}

#[tokio::test]
async fn gateway_sse_route_smoke_emits_provider_sse() {
    let response = openai_app()
        .oneshot(
            Request::builder()
                .uri("/sse")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("router response");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    assert_eq!(
        response.headers()["x-siumai-bridge-target"],
        "openai-responses"
    );
    assert_eq!(response.headers()["x-siumai-bridge-mode"], "best-effort");

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let text = String::from_utf8(body.to_vec()).expect("utf8");
    assert!(text.contains("event: response.output_text.delta"));
    assert!(text.contains("gateway ok"));
    assert!(text.contains("event: response.completed"));
    assert!(text.contains("data: [DONE]"));
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn gateway_json_route_smoke_emits_anthropic_json() {
    let response = anthropic_app()
        .oneshot(
            Request::builder()
                .uri("/json")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("router response");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "application/json");
    assert_eq!(
        response.headers()["x-siumai-bridge-target"],
        "anthropic-messages"
    );
    assert_eq!(response.headers()["x-siumai-bridge-mode"], "best-effort");

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("json body");
    assert_eq!(json["type"], "message");
    assert_eq!(json["role"], "assistant");
    assert_eq!(json["content"][0]["type"], "text");
    assert_eq!(json["content"][0]["text"], "gateway ok");
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn gateway_sse_route_smoke_emits_anthropic_sse() {
    let response = anthropic_app()
        .oneshot(
            Request::builder()
                .uri("/sse")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("router response");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    assert_eq!(
        response.headers()["x-siumai-bridge-target"],
        "anthropic-messages"
    );
    assert_eq!(response.headers()["x-siumai-bridge-mode"], "best-effort");

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let text = String::from_utf8(body.to_vec()).expect("utf8");
    assert!(text.contains("event: content_block_delta"));
    assert!(text.contains("gateway ok"));
    assert!(text.contains("event: message_stop"));
}

#[cfg(feature = "google")]
#[tokio::test]
async fn gateway_json_route_smoke_emits_gemini_json() {
    let response = gemini_app()
        .oneshot(
            Request::builder()
                .uri("/json")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("router response");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "application/json");
    assert_eq!(
        response.headers()["x-siumai-bridge-target"],
        "gemini-generate-content"
    );
    assert_eq!(response.headers()["x-siumai-bridge-mode"], "best-effort");

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("json body");
    assert_eq!(json["candidates"][0]["content"]["role"], "model");
    assert_eq!(
        json["candidates"][0]["content"]["parts"][0]["text"],
        "gateway ok"
    );
}

#[cfg(feature = "google")]
#[tokio::test]
async fn gateway_sse_route_smoke_emits_gemini_sse() {
    let response = gemini_app()
        .oneshot(
            Request::builder()
                .uri("/sse")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("router response");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    assert_eq!(
        response.headers()["x-siumai-bridge-target"],
        "gemini-generate-content"
    );
    assert_eq!(response.headers()["x-siumai-bridge-mode"], "best-effort");

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let downstream = decode_gemini_sse_body(&body);

    assert!(downstream.iter().any(|event| matches!(
        event,
        ChatStreamEvent::ContentDelta { delta, .. } if delta == "gateway ok"
    )));
    assert!(downstream.iter().any(|event| matches!(
        event,
        ChatStreamEvent::StreamEnd { response }
            if response.finish_reason == Some(siumai::prelude::unified::FinishReason::Stop)
    )));
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn gateway_route_smoke_transcodes_anthropic_fixture_to_openai_sse() {
    let response = cross_protocol_app()
        .oneshot(
            Request::builder()
                .uri("/anthropic-to-openai")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("router response");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    assert_eq!(
        response.headers()["x-siumai-bridge-target"],
        "openai-responses"
    );

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let downstream = decode_openai_responses_sse_body(&body);
    let calls = v3_tool_parts(&downstream, "tool-call", "web_fetch");
    let results = v3_tool_parts(&downstream, "tool-result", "web_fetch");

    assert!(!calls.is_empty(), "expected downstream web_fetch tool-call");
    assert!(
        !results.is_empty(),
        "expected downstream web_fetch tool-result"
    );
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn gateway_route_smoke_transcodes_openai_fixture_to_anthropic_sse() {
    let response = cross_protocol_app()
        .oneshot(
            Request::builder()
                .uri("/openai-to-anthropic")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("router response");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    assert_eq!(
        response.headers()["x-siumai-bridge-target"],
        "anthropic-messages"
    );

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let frames = parse_anthropic_sse_json_frames(&body);

    assert!(
        frames.iter().any(|frame| frame["type"] == "message_stop"),
        "expected anthropic message_stop frame"
    );
    assert!(
        frames.iter().any(|frame| {
            frame["type"] == "content_block_start"
                && frame["content_block"]["type"] == "server_tool_use"
                && frame["content_block"]["name"] == "webSearch"
        }),
        "expected anthropic server_tool_use block for webSearch"
    );
}

#[cfg(feature = "google")]
#[tokio::test]
async fn gateway_route_smoke_transcodes_gemini_fixture_to_openai_sse() {
    let response = gemini_cross_protocol_app()
        .oneshot(
            Request::builder()
                .uri("/gemini-to-openai")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("router response");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    assert_eq!(
        response.headers()["x-siumai-bridge-target"],
        "openai-responses"
    );
    assert_eq!(response.headers()["x-siumai-bridge-decision"], "lossy");

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let downstream = decode_openai_responses_sse_body(&body);

    assert_eq!(collect_text_deltas(&downstream), "Hello world");
    assert!(downstream.iter().any(|event| matches!(
        event,
        ChatStreamEvent::StreamEnd { response }
            if response.finish_reason == Some(siumai::prelude::unified::FinishReason::Stop)
    )));
}

#[cfg(feature = "google")]
#[tokio::test]
async fn gateway_route_smoke_transcodes_openai_fixture_to_gemini_sse_with_lossy_headers() {
    let response = gemini_cross_protocol_app()
        .oneshot(
            Request::builder()
                .uri("/openai-to-gemini")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("router response");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    assert_eq!(
        response.headers()["x-siumai-bridge-target"],
        "gemini-generate-content"
    );
    assert_eq!(response.headers()["x-siumai-bridge-mode"], "best-effort");
    assert_eq!(response.headers()["x-siumai-bridge-decision"], "lossy");
    assert_eq!(response.headers()["x-siumai-bridge-warnings"], "1");
    assert_eq!(response.headers()["x-siumai-bridge-lossy-fields"], "1");
    assert_eq!(response.headers()["x-siumai-bridge-dropped-fields"], "0");

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let downstream = decode_gemini_sse_body(&body);
    let text = collect_text_deltas(&downstream);

    assert!(
        text.contains("Hello, World!"),
        "unexpected decoded text: {text}"
    );
    assert!(downstream.iter().any(|event| matches!(
        event,
        ChatStreamEvent::StreamEnd { response }
            if response.finish_reason == Some(siumai::prelude::unified::FinishReason::Stop)
    )));
}

#[cfg(feature = "google")]
#[tokio::test]
async fn gateway_route_smoke_rejects_openai_fixture_to_gemini_sse_in_strict_mode() {
    let response = gemini_cross_protocol_app()
        .oneshot(
            Request::builder()
                .uri("/openai-to-gemini-strict")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("router response");

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(response.headers()["content-type"], "application/json");
    assert_eq!(
        response.headers()["x-siumai-bridge-target"],
        "gemini-generate-content"
    );
    assert_eq!(response.headers()["x-siumai-bridge-mode"], "strict");
    assert_eq!(response.headers()["x-siumai-bridge-decision"], "rejected");
    assert_eq!(response.headers()["x-siumai-bridge-lossy-fields"], "1");

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let json: serde_json::Value = serde_json::from_slice(&body).expect("json body");
    assert_eq!(json["error"], "bridge rejected");
    assert_eq!(json["report"]["lossy_fields"][0], "stream.protocol");
}

#[cfg(all(feature = "anthropic", feature = "google"))]
#[tokio::test]
async fn gateway_route_smoke_transcodes_anthropic_fixture_to_gemini_sse() {
    let response = anthropic_gemini_cross_protocol_app()
        .oneshot(
            Request::builder()
                .uri("/anthropic-to-gemini")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("router response");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    assert_eq!(
        response.headers()["x-siumai-bridge-target"],
        "gemini-generate-content"
    );
    assert_eq!(response.headers()["x-siumai-bridge-decision"], "lossy");

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let frames = parse_sse_json_frames(&body);

    assert!(frames.iter().any(|frame| {
        frame["candidates"][0]["content"]["parts"][0]["functionCall"]["name"]
            == serde_json::json!("web_search")
    }));
    assert!(
        frames
            .iter()
            .any(|frame| { frame["candidates"][0]["finishReason"].as_str().is_some() })
    );
}

#[cfg(all(feature = "anthropic", feature = "google"))]
#[tokio::test]
async fn gateway_route_smoke_transcodes_gemini_fixture_to_anthropic_sse() {
    let response = anthropic_gemini_cross_protocol_app()
        .oneshot(
            Request::builder()
                .uri("/gemini-to-anthropic")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("router response");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    assert_eq!(
        response.headers()["x-siumai-bridge-target"],
        "anthropic-messages"
    );
    assert_eq!(response.headers()["x-siumai-bridge-decision"], "lossy");

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let frames = parse_anthropic_sse_json_frames(&body);

    assert!(frames.iter().any(|frame| frame["type"] == "message_stop"));
    assert!(frames.iter().any(|frame| {
        frame["type"] == "content_block_delta"
            && frame["delta"]["type"] == "thinking_delta"
            && frame["delta"]["thinking"].as_str().is_some()
    }));
    assert!(frames.iter().any(|frame| {
        frame["type"] == "content_block_delta"
            && frame["delta"]["type"] == "text_delta"
            && frame["delta"]["text"]
                .as_str()
                .is_some_and(|text| text.contains("Final answer."))
    }));
}
