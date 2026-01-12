//! Multi-protocol SSE Gateway (Gemini backend)
//!
//! This example demonstrates how to:
//! - Stream from a non-OpenAI provider (Gemini) using `siumai` unified streaming
//! - Expose multiple downstream protocol surfaces from the same upstream backend
//!   (OpenAI Responses SSE / OpenAI Chat Completions SSE / Anthropic Messages SSE / Gemini SSE)
//! - Apply a simple transcoding policy (strict drop vs lossy text fallback)
//! - Serve the result over HTTP using Axum
//!
//! ## Setup
//! ```bash
//! export GOOGLE_API_KEY="your-key"
//! ```
//!
//! ## Run
//! ```bash
//! cargo run -p siumai-extras --example openai-responses-gateway --features "server,google,openai,anthropic"
//! ```
//!
//! ## Test
//! ```bash
//! curl -N http://127.0.0.1:3000/v1/responses
//! curl -N "http://127.0.0.1:3000/v1/responses?lossy=1"
//! curl -N http://127.0.0.1:3000/v1/chat/completions
//! curl -N "http://127.0.0.1:3000/anthropic/messages?lossy=1"
//! curl -N http://127.0.0.1:3000/gemini/generateContent
//! curl http://127.0.0.1:3000/v1/responses.json
//! curl http://127.0.0.1:3000/v1/chat/completions.json
//! curl http://127.0.0.1:3000/anthropic/messages.json
//! curl http://127.0.0.1:3000/gemini/generateContent.json
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
    to_transcoded_json_response, to_transcoded_sse_response,
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

    to_transcoded_sse_response(stream, TargetSseFormat::OpenAiResponses, transcode_opts(&q))
}

async fn chat_completions(
    State(state): State<AppState>,
    Query(q): Query<GatewayQuery>,
) -> Response {
    let prompt = q
        .prompt
        .clone()
        .unwrap_or_else(|| "Write a short haiku about Rust.".to_string());

    let stream = match state.client.chat_stream(vec![user!(prompt)], None).await {
        Ok(s) => s,
        Err(e) => return internal_error_response(&e),
    };

    to_transcoded_sse_response(
        stream,
        TargetSseFormat::OpenAiChatCompletions,
        transcode_opts(&q),
    )
}

#[cfg(feature = "anthropic")]
async fn anthropic_messages(
    State(state): State<AppState>,
    Query(q): Query<GatewayQuery>,
) -> Response {
    let prompt = q
        .prompt
        .clone()
        .unwrap_or_else(|| "Answer in JSON: {\"summary\":\"...\"}. Use one sentence.".to_string());

    let stream = match state.client.chat_stream(vec![user!(prompt)], None).await {
        Ok(s) => s,
        Err(e) => return internal_error_response(&e),
    };

    to_transcoded_sse_response(
        stream,
        TargetSseFormat::AnthropicMessages,
        transcode_opts(&q),
    )
}

#[cfg(feature = "google")]
async fn gemini_generate_content(
    State(state): State<AppState>,
    Query(q): Query<GatewayQuery>,
) -> Response {
    let prompt = q
        .prompt
        .clone()
        .unwrap_or_else(|| "List three practical tips for learning Rust.".to_string());

    let stream = match state.client.chat_stream(vec![user!(prompt)], None).await {
        Ok(s) => s,
        Err(e) => return internal_error_response(&e),
    };

    to_transcoded_sse_response(
        stream,
        TargetSseFormat::GeminiGenerateContent,
        transcode_opts(&q),
    )
}

async fn responses_json(State(state): State<AppState>, Query(q): Query<GatewayQuery>) -> Response {
    let prompt = q
        .prompt
        .clone()
        .unwrap_or_else(|| "Explain Rust lifetimes in one short paragraph.".to_string());

    let resp = match state.client.chat(vec![user!(prompt)]).await {
        Ok(r) => r,
        Err(e) => return internal_error_response(&e),
    };

    to_transcoded_json_response(
        resp,
        TargetJsonFormat::OpenAiResponses,
        TranscodeJsonOptions::default(),
    )
}

async fn chat_completions_json(
    State(state): State<AppState>,
    Query(q): Query<GatewayQuery>,
) -> Response {
    let prompt = q
        .prompt
        .clone()
        .unwrap_or_else(|| "Write a short haiku about Rust.".to_string());

    let resp = match state.client.chat(vec![user!(prompt)]).await {
        Ok(r) => r,
        Err(e) => return internal_error_response(&e),
    };

    to_transcoded_json_response(
        resp,
        TargetJsonFormat::OpenAiChatCompletions,
        TranscodeJsonOptions::default(),
    )
}

#[cfg(feature = "anthropic")]
async fn anthropic_messages_json(
    State(state): State<AppState>,
    Query(q): Query<GatewayQuery>,
) -> Response {
    let prompt = q
        .prompt
        .clone()
        .unwrap_or_else(|| "Answer in JSON: {\"summary\":\"...\"}. Use one sentence.".to_string());

    let resp = match state.client.chat(vec![user!(prompt)]).await {
        Ok(r) => r,
        Err(e) => return internal_error_response(&e),
    };

    to_transcoded_json_response(
        resp,
        TargetJsonFormat::AnthropicMessages,
        TranscodeJsonOptions::default(),
    )
}

#[cfg(feature = "google")]
async fn gemini_generate_content_json(
    State(state): State<AppState>,
    Query(q): Query<GatewayQuery>,
) -> Response {
    let prompt = q
        .prompt
        .clone()
        .unwrap_or_else(|| "List three practical tips for learning Rust.".to_string());

    let resp = match state.client.chat(vec![user!(prompt)]).await {
        Ok(r) => r,
        Err(e) => return internal_error_response(&e),
    };

    to_transcoded_json_response(
        resp,
        TargetJsonFormat::GeminiGenerateContent,
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
        .route("/v1/chat/completions", get(chat_completions))
        .route("/v1/responses.json", get(responses_json))
        .route("/v1/chat/completions.json", get(chat_completions_json));

    #[cfg(feature = "anthropic")]
    let app = app
        .route("/anthropic/messages", get(anthropic_messages))
        .route("/anthropic/messages.json", get(anthropic_messages_json));

    #[cfg(feature = "google")]
    let app = app
        .route("/gemini/generateContent", get(gemini_generate_content))
        .route(
            "/gemini/generateContent.json",
            get(gemini_generate_content_json),
        );

    let app: Router = app.with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000").await?;
    println!("Listening on http://127.0.0.1:3000/v1/responses");
    println!("Listening on http://127.0.0.1:3000/v1/chat/completions");
    println!("Listening on http://127.0.0.1:3000/anthropic/messages");
    println!("Listening on http://127.0.0.1:3000/gemini/generateContent");
    axum::serve(listener, app).await?;

    Ok(())
}
