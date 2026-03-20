//! Gateway Custom Transform Example
//!
//! This example demonstrates the recommended bridge customization surface for gateways:
//! - `GatewayBridgePolicy` for runtime policy and route defaults
//! - `BridgeOptions` for typed bridge customization
//! - `response_bridge_hook(...)` for non-streaming response mutation before serialization
//! - `stream_bridge_hook(...)` for streaming event mutation before SSE serialization
//! - `ClosurePrimitiveRemapper` for reusable tool-name / tool-call-id rewrites
//!
//! Upstream backend: Gemini.
//! Downstream surface: OpenAI Responses (SSE + JSON).
//!
//! ## Setup
//! ```bash
//! export GOOGLE_API_KEY="your-key"
//! ```
//!
//! ## Run
//! ```bash
//! cargo run -p siumai-extras --example gateway-custom-transform --features "server,google,openai"
//! ```
//!
//! ## Test
//! ```bash
//! curl -N "http://127.0.0.1:3000/v1/responses?redact=1"
//! curl -N "http://127.0.0.1:3000/v1/responses?tool_prefix=tenant"
//! curl "http://127.0.0.1:3000/v1/responses.json?redact=1"
//! curl "http://127.0.0.1:3000/v1/responses.json?tool_prefix=tenant"
//! ```

use std::{sync::Arc, time::Duration};

use axum::{
    Router,
    extract::{Query, State},
    response::Response,
    routing::get,
};
use serde::Deserialize;
use siumai::experimental::bridge::{BridgeMode, BridgeOptionsOverride};
use siumai::prelude::unified::*;
use siumai_extras::server::{
    GatewayBridgePolicy,
    axum::{
        ClosurePrimitiveRemapper, TargetJsonFormat, TargetSseFormat, TranscodeJsonOptions,
        TranscodeSseOptions, response_bridge_hook, stream_bridge_hook, to_transcoded_json_response,
        to_transcoded_sse_response,
    },
};

#[derive(Clone)]
struct AppState {
    client: Arc<registry::LanguageModelHandle>,
}

#[derive(Debug, Clone, Deserialize)]
struct GatewayQuery {
    /// Prompt override.
    prompt: Option<String>,
    /// Enable lossy downgrade for unsupported v3 parts.
    lossy: Option<bool>,
    /// Enable OpenAI Responses bridge (tool/source/reasoning v3 parts -> openai:*).
    bridge: Option<bool>,
    /// Redact assistant output through typed bridge hooks.
    redact: Option<bool>,
    /// Prefix tool names / tool-call ids when the bridge sees tool primitives.
    tool_prefix: Option<String>,
}

fn internal_error_response(error: &LlmError) -> Response {
    let body = format!("failed to start stream: {}", error.user_message());
    Response::builder()
        .status(500)
        .header("content-type", "text/plain")
        .body(axum::body::Body::from(body))
        .unwrap_or_else(|_| Response::new(axum::body::Body::from("internal error")))
}

fn build_bridge_options(q: &GatewayQuery) -> BridgeOptionsOverride {
    let tool_name_prefix = q.tool_prefix.clone().unwrap_or_else(|| "gw".to_string());
    let tool_call_prefix = tool_name_prefix.clone();

    let mut bridge_options = BridgeOptionsOverride::new()
        .with_route_label("examples.gateway-custom-transform")
        .with_primitive_remapper(Arc::new(
            ClosurePrimitiveRemapper::default()
                .with_tool_name(move |_, name| Some(format!("{tool_name_prefix}_{name}")))
                .with_tool_call_id(move |_, id| Some(format!("{tool_call_prefix}_{id}"))),
        ));

    if q.redact.unwrap_or(false) {
        bridge_options = bridge_options
            .with_response_hook(response_bridge_hook(|_, response, report| {
                response.content = MessageContent::Text("[REDACTED]".to_string());
                report.record_lossy_field(
                    "response.content",
                    "route policy redacted assistant output before target serialization",
                );
                Ok(())
            }))
            .with_stream_hook(stream_bridge_hook(|_, event| match event {
                ChatStreamEvent::ContentDelta { index, .. } => {
                    vec![ChatStreamEvent::ContentDelta {
                        delta: "[REDACTED]\n".to_string(),
                        index,
                    }]
                }
                ChatStreamEvent::ThinkingDelta { .. } => vec![ChatStreamEvent::ThinkingDelta {
                    delta: "[REDACTED]\n".to_string(),
                }],
                other => vec![other],
            }));
    }

    bridge_options
}

fn gateway_policy(q: &GatewayQuery) -> GatewayBridgePolicy {
    GatewayBridgePolicy::new(BridgeMode::BestEffort)
        .with_bridge_options_override(build_bridge_options(q))
        .with_bridge_headers(true)
        .with_bridge_warning_headers(true)
        .with_keepalive_interval(Duration::from_secs(15))
        .with_stream_idle_timeout(Duration::from_secs(60))
}

fn sse_transcode_opts(q: &GatewayQuery) -> TranscodeSseOptions {
    let mut opts = if q.lossy.unwrap_or(false) {
        TranscodeSseOptions::lossy_text()
    } else {
        TranscodeSseOptions::strict()
    };
    if let Some(bridge) = q.bridge {
        opts.bridge_openai_responses_stream_parts = bridge;
    }
    opts.with_policy(gateway_policy(q))
}

fn json_transcode_opts(q: &GatewayQuery) -> TranscodeJsonOptions {
    TranscodeJsonOptions::default().with_policy(gateway_policy(q))
}

async fn responses(State(state): State<AppState>, Query(q): Query<GatewayQuery>) -> Response {
    let prompt = q
        .prompt
        .clone()
        .unwrap_or_else(|| "Explain Rust lifetimes in one short paragraph.".to_string());

    let stream = match text::stream(
        &*state.client,
        ChatRequest::new(vec![user!(prompt)]),
        text::StreamOptions::default(),
    )
    .await
    {
        Ok(s) => s,
        Err(e) => return internal_error_response(&e),
    };

    to_transcoded_sse_response(
        stream,
        TargetSseFormat::OpenAiResponses,
        sse_transcode_opts(&q),
    )
}

async fn responses_json(State(state): State<AppState>, Query(q): Query<GatewayQuery>) -> Response {
    let prompt = q
        .prompt
        .clone()
        .unwrap_or_else(|| "Write one sentence about Rust.".to_string());

    let resp = match text::generate(
        &*state.client,
        ChatRequest::new(vec![user!(prompt)]),
        text::GenerateOptions::default(),
    )
    .await
    {
        Ok(r) => r,
        Err(e) => return internal_error_response(&e),
    };

    to_transcoded_json_response(
        resp,
        TargetJsonFormat::OpenAiResponses,
        json_transcode_opts(&q),
    )
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let reg = registry::global();
    let client = reg.language_model("gemini:gemini-2.0-flash-exp")?;

    let state = AppState {
        client: Arc::new(client),
    };

    let app = Router::<AppState>::new()
        .route("/v1/responses", get(responses))
        .route("/v1/responses.json", get(responses_json))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000").await?;
    println!("Listening on http://127.0.0.1:3000/v1/responses");
    println!("Listening on http://127.0.0.1:3000/v1/responses.json");
    axum::serve(listener, app).await?;

    Ok(())
}
