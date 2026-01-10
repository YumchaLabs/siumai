//! OpenAI Responses SSE Gateway (Gemini backend)
//!
//! This example demonstrates how to:
//! - Stream from a non-OpenAI provider (Gemini) using `siumai` unified streaming
//! - Bridge provider-specific Vercel-aligned stream parts into `openai:*` parts
//! - Serialize the stream into OpenAI Responses SSE frames
//! - Serve the result over HTTP using Axum
//!
//! ## Setup
//! ```bash
//! export GOOGLE_API_KEY="your-key"
//! ```
//!
//! ## Run
//! ```bash
//! cargo run -p siumai-extras --example openai-responses-gateway --features "server,google,openai"
//! ```
//!
//! ## Test
//! ```bash
//! curl -N http://127.0.0.1:3000/v1/responses
//! ```
//!
//! curl -N http://127.0.0.1:3000/v1/chat/completions

use std::sync::Arc;

use axum::{Router, extract::State, response::Response, routing::get};
use siumai::prelude::*;
use siumai_extras::server::axum::{
    to_openai_chat_completions_sse_response, to_openai_responses_sse_response,
};

#[derive(Clone)]
struct AppState {
    client: Arc<Siumai>,
}

async fn responses(State(state): State<AppState>) -> Response {
    let stream = match state
        .client
        .chat_stream(
            vec![user!("Explain Rust lifetimes in one short paragraph.")],
            None,
        )
        .await
    {
        Ok(s) => s,
        Err(e) => {
            let body = format!("failed to start stream: {}", e.user_message());
            return Response::builder()
                .status(500)
                .header("content-type", "text/plain")
                .body(axum::body::Body::from(body))
                .unwrap_or_else(|_| Response::new(axum::body::Body::from("internal error")));
        }
    };

    to_openai_responses_sse_response(stream)
}

async fn chat_completions(State(state): State<AppState>) -> Response {
    let stream = match state
        .client
        .chat_stream(vec![user!("Write a short haiku about Rust.")], None)
        .await
    {
        Ok(s) => s,
        Err(e) => {
            let body = format!("failed to start stream: {}", e.user_message());
            return Response::builder()
                .status(500)
                .header("content-type", "text/plain")
                .body(axum::body::Body::from(body))
                .unwrap_or_else(|_| Response::new(axum::body::Body::from("internal error")));
        }
    };

    to_openai_chat_completions_sse_response(stream)
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

    let app = Router::new()
        .route("/v1/responses", get(responses))
        .route("/v1/chat/completions", get(chat_completions))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000").await?;
    println!("Listening on http://127.0.0.1:3000/v1/responses");
    println!("Listening on http://127.0.0.1:3000/v1/chat/completions");
    axum::serve(listener, app).await?;

    Ok(())
}
