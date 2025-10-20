//! API Server - HTTP API for LLM access
//!
//! This example demonstrates building an HTTP API server that
//! exposes LLM capabilities via REST endpoints.
//!
//! ## Run
//! ```bash
//! cargo run --example api-server --features "openai,server-adapters"
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
use siumai::prelude::*;
use std::sync::Arc;

#[derive(Clone)]
struct AppState {
    client: Arc<Box<dyn ChatCapability + Send + Sync>>,
}

#[derive(Deserialize)]
struct ChatRequest {
    message: String,
}

#[derive(Serialize)]
struct ChatResponse {
    response: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 API Server starting...\n");

    // Build LLM client
    let client = Siumai::builder()
        .openai()
        .api_key(&std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o-mini")
        .build()
        .await?;

    let state = AppState {
        client: Arc::new(Box::new(client)),
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
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, StatusCode> {
    let response = state
        .client
        .chat(vec![user!(&req.message)])
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let text = response
        .content_text()
        .unwrap_or_else(|| "No response".to_string());

    Ok(Json(ChatResponse { response: text }))
}
