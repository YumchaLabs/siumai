//! Gateway Custom Transform Example
//!
//! This example demonstrates how to implement **user-defined conversion logic** for:
//! - Streaming (SSE): `ChatStreamEvent` transform hook
//! - Non-streaming (JSON): `ChatResponse` transform hook (preferred, no JSON round-trip)
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
//! curl -N "http://127.0.0.1:3000/v1/responses"
//! curl "http://127.0.0.1:3000/v1/responses.json?redact=1"
//! curl "http://127.0.0.1:3000/v1/responses.json"
//! ```

use std::sync::Arc;

use axum::{
    Router,
    extract::{Query, State},
    response::Response,
    routing::get,
};
use serde::Deserialize;
use siumai::prelude::*;
use siumai_extras::server::axum::{
    TargetJsonFormat, TargetSseFormat, TranscodeJsonOptions, TranscodeSseOptions,
    to_transcoded_json_response, to_transcoded_json_response_with_response_transform,
    to_transcoded_sse_response, to_transcoded_sse_response_with_transform,
};

#[derive(Clone)]
struct AppState {
    client: Arc<Siumai>,
}

#[derive(Debug, Clone, Deserialize)]
struct GatewayQuery {
    /// Prompt override.
    prompt: Option<String>,
    /// Enable lossy downgrade for unsupported v3 parts.
    lossy: Option<bool>,
    /// Enable OpenAI Responses bridge (tool/source/reasoning v3 parts -> openai:*).
    bridge: Option<bool>,
    /// Redact output (example of user-defined conversion).
    redact: Option<bool>,
}

fn internal_error_response(error: &LlmError) -> Response {
    let body = format!("failed to start stream: {}", error.user_message());
    Response::builder()
        .status(500)
        .header("content-type", "text/plain")
        .body(axum::body::Body::from(body))
        .unwrap_or_else(|_| Response::new(axum::body::Body::from("internal error")))
}

fn transcode_opts(q: &GatewayQuery) -> TranscodeSseOptions {
    let mut opts = if q.lossy.unwrap_or(false) {
        TranscodeSseOptions::lossy_text()
    } else {
        TranscodeSseOptions::strict()
    };
    if let Some(bridge) = q.bridge {
        opts.bridge_openai_responses_stream_parts = bridge;
    }
    opts
}

async fn responses(State(state): State<AppState>, Query(q): Query<GatewayQuery>) -> Response {
    let prompt = q
        .prompt
        .clone()
        .unwrap_or_else(|| "Explain Rust lifetimes in one short paragraph.".to_string());

    let stream = match state.client.chat_stream(vec![user!(prompt)], None).await {
        Ok(s) => s,
        Err(e) => return internal_error_response(&e),
    };

    if q.redact.unwrap_or(false) {
        return to_transcoded_sse_response_with_transform(
            stream,
            TargetSseFormat::OpenAiResponses,
            transcode_opts(&q),
            |ev| match ev {
                ChatStreamEvent::ContentDelta { .. } => vec![ChatStreamEvent::ContentDelta {
                    delta: "[REDACTED]\n".to_string(),
                    index: None,
                }],
                other => vec![other],
            },
        );
    }

    to_transcoded_sse_response(stream, TargetSseFormat::OpenAiResponses, transcode_opts(&q))
}

async fn responses_json(State(state): State<AppState>, Query(q): Query<GatewayQuery>) -> Response {
    let prompt = q
        .prompt
        .clone()
        .unwrap_or_else(|| "Write one sentence about Rust.".to_string());

    let resp = match state.client.chat(vec![user!(prompt)]).await {
        Ok(r) => r,
        Err(e) => return internal_error_response(&e),
    };

    if q.redact.unwrap_or(false) {
        return to_transcoded_json_response_with_response_transform(
            resp,
            TargetJsonFormat::OpenAiResponses,
            TranscodeJsonOptions::default(),
            |r| {
                r.content = MessageContent::Text("[REDACTED]".to_string());
            },
        );
    }

    to_transcoded_json_response(
        resp,
        TargetJsonFormat::OpenAiResponses,
        TranscodeJsonOptions::default(),
    )
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("GOOGLE_API_KEY")?;

    // Gemini backend (upstream).
    let client = Siumai::builder()
        .gemini()
        .api_key(&api_key)
        .model("gemini-2.0-flash-exp")
        .build()
        .await?;

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
