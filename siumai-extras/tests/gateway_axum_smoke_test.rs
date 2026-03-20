#![cfg(all(feature = "server", feature = "openai"))]

use axum::{
    Router,
    body::{Body, to_bytes},
    extract::State,
    http::{Request, StatusCode},
    response::Response,
    routing::get,
};
use futures::stream;
use siumai::prelude::unified::{ChatResponse, ChatStream, ChatStreamEvent, MessageContent};
use siumai_extras::server::{
    GatewayBridgePolicy,
    axum::{
        TargetJsonFormat, TargetSseFormat, TranscodeJsonOptions, TranscodeSseOptions,
        to_transcoded_json_response, to_transcoded_sse_response,
    },
};
use tower::ServiceExt;

#[derive(Clone)]
struct AppState {
    policy: GatewayBridgePolicy,
}

fn app() -> Router {
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

#[tokio::test]
async fn gateway_json_route_smoke_emits_provider_json() {
    let response = app()
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
    let response = app()
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
