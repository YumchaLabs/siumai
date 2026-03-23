#![cfg(all(feature = "server", feature = "openai"))]

use std::time::Duration;

use async_stream::stream;
use axum::{
    Router,
    body::{Body, to_bytes},
    extract::State,
    http::{Request, StatusCode},
    response::Response,
    routing::get,
};
use serde_json::Value;
use siumai::experimental::bridge::BridgeMode;
use siumai::prelude::unified::{ChatResponse, ChatStream, ChatStreamEvent, MessageContent};
use siumai_extras::server::{
    GatewayBridgePolicy,
    axum::{
        TargetJsonFormat, TargetSseFormat, TranscodeJsonOptions, TranscodeSseOptions,
        read_upstream_body_with_policy, read_upstream_json_with_policy,
        to_transcoded_json_response, to_transcoded_sse_response,
    },
};
use tower::ServiceExt;

#[derive(Clone)]
struct UpstreamProxyState {
    policy: GatewayBridgePolicy,
    upstream_body: String,
}

#[derive(Clone)]
struct UpstreamSseState {
    policy: GatewayBridgePolicy,
}

fn base_policy() -> GatewayBridgePolicy {
    GatewayBridgePolicy::new(BridgeMode::BestEffort)
        .with_bridge_headers(true)
        .with_bridge_warning_headers(true)
}

fn upstream_body_proxy_app(
    policy: GatewayBridgePolicy,
    upstream_body: impl Into<String>,
) -> Router {
    Router::new()
        .route("/proxy-body", get(upstream_body_proxy_route))
        .with_state(UpstreamProxyState {
            policy,
            upstream_body: upstream_body.into(),
        })
}

fn upstream_json_proxy_app(
    policy: GatewayBridgePolicy,
    upstream_body: impl Into<String>,
) -> Router {
    Router::new()
        .route("/proxy-json", get(upstream_json_proxy_route))
        .with_state(UpstreamProxyState {
            policy,
            upstream_body: upstream_body.into(),
        })
}

fn upstream_keepalive_sse_app(policy: GatewayBridgePolicy) -> Router {
    Router::new()
        .route("/proxy-sse-keepalive", get(upstream_keepalive_sse_route))
        .with_state(UpstreamSseState { policy })
}

fn upstream_idle_timeout_sse_app(policy: GatewayBridgePolicy) -> Router {
    Router::new()
        .route("/proxy-sse-timeout", get(upstream_idle_timeout_sse_route))
        .with_state(UpstreamSseState { policy })
}

async fn upstream_body_proxy_route(State(state): State<UpstreamProxyState>) -> Response<Body> {
    let bytes = match read_upstream_body_with_policy(Body::from(state.upstream_body), &state.policy)
        .await
    {
        Ok(bytes) => bytes,
        Err(error) => return error.to_response(&state.policy),
    };

    to_transcoded_json_response(
        ChatResponse::new(MessageContent::Text(format!("bytes={}", bytes.len()))),
        TargetJsonFormat::OpenAiResponses,
        TranscodeJsonOptions::default().with_policy(state.policy),
    )
}

async fn upstream_json_proxy_route(State(state): State<UpstreamProxyState>) -> Response<Body> {
    let value: Value = match read_upstream_json_with_policy(
        Body::from(state.upstream_body),
        &state.policy,
    )
    .await
    {
        Ok(value) => value,
        Err(error) => return error.to_response(&state.policy),
    };

    let message = value
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("missing");

    to_transcoded_json_response(
        ChatResponse::new(MessageContent::Text(format!("upstream={message}"))),
        TargetJsonFormat::OpenAiResponses,
        TranscodeJsonOptions::default().with_policy(state.policy),
    )
}

async fn upstream_keepalive_sse_route(State(state): State<UpstreamSseState>) -> Response<Body> {
    let chat_stream: ChatStream = Box::pin(stream! {
        tokio::time::sleep(Duration::from_millis(25)).await;
        yield Ok(ChatStreamEvent::ContentDelta {
            delta: "proxy-stream-ok".to_string(),
            index: None,
        });
        yield Ok(ChatStreamEvent::StreamEnd {
            response: ChatResponse::new(MessageContent::Text("proxy-stream-ok".to_string())),
        });
    });

    to_transcoded_sse_response(
        chat_stream,
        TargetSseFormat::OpenAiResponses,
        TranscodeSseOptions::default().with_policy(state.policy),
    )
}

async fn upstream_idle_timeout_sse_route(State(state): State<UpstreamSseState>) -> Response<Body> {
    let chat_stream: ChatStream = Box::pin(stream! {
        tokio::time::sleep(Duration::from_millis(30)).await;
        yield Ok(ChatStreamEvent::ContentDelta {
            delta: "late".to_string(),
            index: None,
        });
    });

    to_transcoded_sse_response(
        chat_stream,
        TargetSseFormat::OpenAiResponses,
        TranscodeSseOptions::default().with_policy(state.policy),
    )
}

fn decode_openai_responses_text(bytes: &[u8]) -> String {
    let json: Value = serde_json::from_slice(bytes).expect("json body");
    json["output"][0]["content"][0]["text"]
        .as_str()
        .expect("openai responses text")
        .to_string()
}

#[tokio::test]
async fn gateway_runtime_proxy_route_smoke_reads_upstream_body_before_transcoding() {
    let response = upstream_body_proxy_app(base_policy(), "hello-upstream")
        .oneshot(
            Request::builder()
                .uri("/proxy-body")
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

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let text = decode_openai_responses_text(&body);
    assert_eq!(text, "bytes=14");
}

#[tokio::test]
async fn gateway_runtime_proxy_route_smoke_reads_upstream_json_before_transcoding() {
    let response = upstream_json_proxy_app(base_policy(), r#"{"message":"upstream ok"}"#)
        .oneshot(
            Request::builder()
                .uri("/proxy-json")
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
    let text = decode_openai_responses_text(&body);
    assert_eq!(text, "upstream=upstream ok");
}

#[tokio::test]
async fn gateway_runtime_proxy_route_smoke_enforces_upstream_read_limit() {
    let response = upstream_json_proxy_app(
        base_policy().with_upstream_read_limit_bytes(8),
        r#"{"message":"this upstream body is too large"}"#,
    )
    .oneshot(
        Request::builder()
            .uri("/proxy-json")
            .body(Body::empty())
            .expect("request"),
    )
    .await
    .expect("router response");

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(
        response.headers()["content-type"],
        "text/plain; charset=utf-8"
    );

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    assert_eq!(
        String::from_utf8(body.to_vec()).expect("utf8"),
        "upstream body exceeded limit of 8 bytes"
    );
}

#[tokio::test]
async fn gateway_runtime_proxy_route_smoke_masks_invalid_upstream_json() {
    let response = upstream_json_proxy_app(
        base_policy().with_passthrough_runtime_errors(false),
        "not-json",
    )
    .oneshot(
        Request::builder()
            .uri("/proxy-json")
            .body(Body::empty())
            .expect("request"),
    )
    .await
    .expect("router response");

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(
        response.headers()["content-type"],
        "text/plain; charset=utf-8"
    );

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    assert_eq!(
        String::from_utf8(body.to_vec()).expect("utf8"),
        "invalid upstream response json"
    );
}

#[tokio::test]
async fn gateway_runtime_proxy_sse_route_smoke_emits_keepalive_comments() {
    let response =
        upstream_keepalive_sse_app(base_policy().with_keepalive_interval(Duration::from_millis(5)))
            .oneshot(
                Request::builder()
                    .uri("/proxy-sse-keepalive")
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

    assert!(text.contains(": keep-alive"));
    assert!(text.contains("proxy-stream-ok"));
    assert!(text.contains("data: [DONE]"));
}

#[tokio::test]
async fn gateway_runtime_proxy_sse_route_smoke_masks_idle_timeout() {
    let response = upstream_idle_timeout_sse_app(
        base_policy()
            .with_stream_idle_timeout(Duration::from_millis(5))
            .with_passthrough_runtime_errors(false),
    )
    .oneshot(
        Request::builder()
            .uri("/proxy-sse-timeout")
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
    let text = String::from_utf8(body.to_vec()).expect("utf8");

    assert!(text.contains("event: response.error"));
    assert!(text.contains("gateway stream idle timeout"));
    assert!(!text.contains("after 5 ms"));
    assert!(!text.contains("late"));
}
