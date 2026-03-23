#![cfg(all(feature = "server", feature = "openai"))]

use std::sync::Arc;

use axum::{
    Router,
    body::{Body, to_bytes},
    extract::{Request as AxumRequest, State},
    http::{Request, StatusCode},
    response::Response,
    routing::post,
};
use futures::stream;
use serde_json::{Value, json};
use siumai::experimental::bridge::{
    BridgeMode, BridgeReport, BridgeTarget, ProviderToolRewriteCustomization,
};
use siumai::prelude::unified::{
    ChatRequest, ChatResponse, ChatStream, ChatStreamEvent, MessageContent, Tool,
};
use siumai_extras::bridge::ClosureBridgeCustomization;
use siumai_extras::server::{
    GatewayBridgePolicy,
    axum::{
        NormalizeRequestOptions, SourceRequestFormat, TargetSseFormat, TranscodeSseOptions,
        normalize_request_json_with_options, read_request_json_with_policy,
        to_transcoded_sse_response,
    },
};
use tower::ServiceExt;

#[derive(Clone)]
struct IngressSseAppState {
    source: SourceRequestFormat,
    target: TargetSseFormat,
    policy: GatewayBridgePolicy,
    normalize_options: NormalizeRequestOptions,
}

fn base_policy() -> GatewayBridgePolicy {
    GatewayBridgePolicy::new(BridgeMode::BestEffort)
        .with_bridge_headers(true)
        .with_bridge_warning_headers(true)
}

fn ingress_sse_app(state: IngressSseAppState) -> Router {
    Router::new()
        .route("/ingress-sse", post(ingress_sse_route))
        .with_state(state)
}

fn openai_responses_ingress_sse_app() -> Router {
    ingress_sse_app(IngressSseAppState {
        source: SourceRequestFormat::OpenAiResponses,
        target: TargetSseFormat::OpenAiResponses,
        policy: base_policy(),
        normalize_options: NormalizeRequestOptions::default(),
    })
}

fn limited_openai_responses_ingress_sse_app() -> Router {
    ingress_sse_app(IngressSseAppState {
        source: SourceRequestFormat::OpenAiResponses,
        target: TargetSseFormat::OpenAiResponses,
        policy: base_policy().with_request_body_limit_bytes(8),
        normalize_options: NormalizeRequestOptions::default(),
    })
}

fn strict_openai_responses_ingress_sse_app() -> Router {
    let policy = base_policy()
        .with_route_label("tests.request-ingress.sse.strict")
        .with_customization(Arc::new(
            ClosureBridgeCustomization::default().with_request(|ctx, request, report| {
                assert_eq!(ctx.source, Some(BridgeTarget::OpenAiResponses));
                assert_eq!(ctx.target, BridgeTarget::OpenAiResponses);
                assert_eq!(ctx.mode, BridgeMode::Strict);
                assert_eq!(
                    ctx.route_label.as_deref(),
                    Some("tests.request-ingress.sse.strict")
                );
                assert_eq!(ctx.path_label.as_deref(), Some("source-normalize"));

                request.common_params.max_tokens = Some(88);
                report.record_lossy_field(
                    "normalize.custom",
                    "gateway ingress sse route forced lossy normalization",
                );
                Ok(())
            }),
        ));

    ingress_sse_app(IngressSseAppState {
        source: SourceRequestFormat::OpenAiResponses,
        target: TargetSseFormat::OpenAiResponses,
        policy,
        normalize_options: NormalizeRequestOptions::default()
            .with_bridge_mode_override(BridgeMode::Strict),
    })
}

#[cfg(feature = "anthropic")]
fn anthropic_ingress_sse_app() -> Router {
    let policy = base_policy().with_customization(Arc::new(
        ProviderToolRewriteCustomization::new()
            .map_provider_tool_id("anthropic.web_fetch_20250910", "openai.web_search"),
    ));

    ingress_sse_app(IngressSseAppState {
        source: SourceRequestFormat::AnthropicMessages,
        target: TargetSseFormat::OpenAiResponses,
        policy,
        normalize_options: NormalizeRequestOptions::default(),
    })
}

async fn ingress_sse_route(
    State(state): State<IngressSseAppState>,
    request: AxumRequest,
) -> Response<Body> {
    let body: Value = match read_request_json_with_policy(request.into_body(), &state.policy).await
    {
        Ok(body) => body,
        Err(error) => return error.to_response(&state.policy),
    };

    let normalize_options = state
        .normalize_options
        .clone()
        .with_policy(state.policy.clone());
    let bridged = match normalize_request_json_with_options(&body, state.source, &normalize_options)
    {
        Ok(bridged) => bridged,
        Err(error) => return plain_text_response(StatusCode::BAD_REQUEST, error.to_string()),
    };

    let (normalized, report) = match bridged.into_result() {
        Ok(result) => result,
        Err(report) => return rejected_request_response(report),
    };

    let summary = summarize_request(&normalized, &report);
    let response = ChatResponse::new(MessageContent::Text(summary.clone()));
    let stream: ChatStream = Box::pin(stream::iter(vec![
        Ok(ChatStreamEvent::ContentDelta {
            delta: summary,
            index: None,
        }),
        Ok(ChatStreamEvent::StreamEnd { response }),
    ]));

    to_transcoded_sse_response(
        stream,
        state.target,
        TranscodeSseOptions::default().with_policy(state.policy.clone()),
    )
}

fn summarize_request(request: &ChatRequest, report: &BridgeReport) -> String {
    let tools = request
        .tools
        .as_ref()
        .map(|tools| {
            tools
                .iter()
                .map(summarize_tool)
                .collect::<Vec<_>>()
                .join(",")
        })
        .filter(|tools| !tools.is_empty())
        .unwrap_or_else(|| "none".to_string());

    format!(
        "model={};messages={};tools={};decision={};warnings={}",
        request.common_params.model,
        request.messages.len(),
        tools,
        bridge_decision_label(report),
        report.warnings.len()
    )
}

fn summarize_tool(tool: &Tool) -> String {
    match tool {
        Tool::Function { function } => format!("function:{}", function.name),
        Tool::ProviderDefined(tool) => format!("provider:{}", tool.id),
    }
}

fn bridge_decision_label(report: &BridgeReport) -> &'static str {
    if report.is_rejected() {
        "rejected"
    } else if report.is_lossy() {
        "lossy"
    } else {
        "exact"
    }
}

fn plain_text_response(status: StatusCode, body: impl Into<String>) -> Response<Body> {
    Response::builder()
        .status(status)
        .header("content-type", "text/plain; charset=utf-8")
        .body(Body::from(body.into()))
        .unwrap_or_else(|_| Response::new(Body::from("internal error")))
}

fn rejected_request_response(report: BridgeReport) -> Response<Body> {
    Response::builder()
        .status(StatusCode::UNPROCESSABLE_ENTITY)
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "error": "bridge rejected",
                "report": report,
            }))
            .unwrap_or_else(|_| b"{\"error\":\"bridge rejected\"}".to_vec()),
        ))
        .unwrap_or_else(|_| Response::new(Body::from("internal error")))
}

#[tokio::test]
async fn gateway_request_ingress_sse_route_smoke_streams_openai_responses_after_normalization() {
    let body = json!({
        "model": "gpt-5-mini",
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "hello"
                    }
                ]
            }
        ]
    });

    let response = openai_responses_ingress_sse_app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/ingress-sse")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
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
    let raw = String::from_utf8(body.to_vec()).expect("utf8");

    assert!(raw.contains("event: response.output_text.delta"));
    assert!(
        raw.contains("model=gpt-5-mini"),
        "unexpected sse body: {raw}"
    );
    assert!(raw.contains("messages=1"), "unexpected sse body: {raw}");
    assert!(raw.contains("tools=none"), "unexpected sse body: {raw}");
    assert!(raw.contains("decision=exact"), "unexpected sse body: {raw}");
    assert!(raw.contains("event: response.completed"));
    assert!(raw.contains("data: [DONE]"));
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn gateway_request_ingress_sse_route_smoke_applies_provider_tool_rewrite_before_streaming() {
    let body = json!({
        "model": "claude-3-5-haiku-20241022",
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": "fetch docs"
            }
        ],
        "tools": [
            {
                "type": "web_fetch_20250910",
                "name": "web_fetch"
            }
        ]
    });

    let response = anthropic_ingress_sse_app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/ingress-sse")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
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
    let raw = String::from_utf8(body.to_vec()).expect("utf8");

    assert!(
        raw.contains("provider:openai.web_search"),
        "unexpected sse body: {raw}"
    );
    assert!(raw.contains("warnings=1"), "unexpected sse body: {raw}");
    assert!(raw.contains("decision=exact"), "unexpected sse body: {raw}");
    assert!(raw.contains("data: [DONE]"));
}

#[tokio::test]
async fn gateway_request_ingress_sse_route_smoke_rejects_lossy_normalization_in_strict_mode() {
    let body = json!({
        "model": "gpt-5-mini",
        "input": [
            {
                "role": "user",
                "content": [
                    { "type": "input_text", "text": "hi" }
                ]
            }
        ]
    });

    let response = strict_openai_responses_ingress_sse_app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/ingress-sse")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request"),
        )
        .await
        .expect("router response");

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(response.headers()["content-type"], "application/json");

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let json: Value = serde_json::from_slice(&body).expect("json body");

    assert_eq!(json["error"], "bridge rejected");
    assert_eq!(json["report"]["mode"], "Strict");
    assert_eq!(json["report"]["lossy_fields"][0], "normalize.custom");
}

#[tokio::test]
async fn gateway_request_ingress_sse_route_smoke_enforces_request_body_limit() {
    let response = limited_openai_responses_ingress_sse_app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/ingress-sse")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"this":"request body is too large"}"#))
                .expect("request"),
        )
        .await
        .expect("router response");

    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(
        response.headers()["content-type"],
        "text/plain; charset=utf-8"
    );

    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    assert_eq!(
        String::from_utf8(body.to_vec()).expect("utf8"),
        "request body exceeded limit of 8 bytes"
    );
}
