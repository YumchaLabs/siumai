//! API Server - HTTP API for LLM access
//!
//! This example demonstrates building an HTTP API server that
//! exposes LLM capabilities via REST endpoints.
//!
//! ## Run
//! ```bash
//! cargo run --example api-server --features openai
//! ```
//!
//! Then test with:
//! ```bash
//! curl -X POST http://localhost:3000/chat \
//!   -H "Content-Type: application/json" \
//!   -d '{"message": "Hello!"}'
//! ```
//!
//! ## Learn More
//! See `siumai/examples/05_use_cases/api_integration.rs` for the complete
//! implementation with authentication, rate limiting, and more.

use axum::{Router, extract::State, http::StatusCode, response::Json, routing::post};
use serde::{Deserialize, Serialize};
use siumai::prelude::unified::*;
use std::sync::Arc;

#[derive(Clone)]
struct AppState {
    client: Arc<dyn siumai::text::TextModelV3 + Send + Sync>,
}

#[derive(Deserialize)]
struct ChatInput {
    message: String,
}

#[derive(Serialize)]
struct ChatOutput {
    response: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 API Server starting...\n");

    // Build LLM client
    // Note: API key is automatically read from `OPENAI_API_KEY`.
    let client = registry::global().language_model("openai:gpt-4o-mini")?;

    let state = AppState {
        client: Arc::new(client),
    };

    // Build router
    let app = Router::new()
        .route("/chat", post(chat_handler))
        .with_state(state);

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("Listening on http://{}", addr);
    println!(
        "Try: curl -X POST http://localhost:3000/chat -H 'Content-Type: application/json' -d '{{\"message\": \"Hello!\"}}'"
    );

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn chat_handler(
    State(state): State<AppState>,
    Json(req): Json<ChatInput>,
) -> Result<Json<ChatOutput>, StatusCode> {
    let response = text::generate(
        state.client.as_ref(),
        ChatRequest::new(vec![user!(&req.message)]),
        text::GenerateOptions::default(),
    )
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let text = response.content_text().unwrap_or("No response").to_string();

    Ok(Json(ChatOutput { response: text }))
}
