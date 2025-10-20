//! Axum SSE Server - Stream responses via HTTP
//!
//! This example demonstrates building an HTTP server with Axum
//! that streams LLM responses using Server-Sent Events (SSE).
//!
//! ## Run
//! ```bash
//! cargo run --example axum-sse --features "openai,server-adapters"
//! ```
//!
//! Then visit: http://localhost:8080/chat?q=hello
//!
//! ## Learn More
//! See `siumai/examples/07_server_adapters/` for complete examples:
//! - axum_sse.rs - Axum with SSE streaming
//! - actix_sse.rs - Actix-web with SSE streaming

use axum::{Router, extract::Query, response::sse::Sse, routing::get};
use futures::{Stream, StreamExt};
use serde::Deserialize;
use siumai::prelude::*;
use siumai::server_adapters::SseOptions;
use siumai::server_adapters::axum::to_sse_response;
use std::convert::Infallible;

#[derive(Debug, Deserialize)]
struct ChatQuery {
    q: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app = Router::new().route("/chat", get(chat_handler));

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], 8080));
    println!("🚀 Server running on http://{}/chat?q=hello", addr);
    println!("💡 Try: curl http://localhost:8080/chat?q=hello");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn chat_handler(
    Query(ChatQuery { q }): Query<ChatQuery>,
) -> Sse<impl Stream<Item = Result<axum::response::sse::Event, Infallible>> + Send> {
    let prompt = q.unwrap_or_else(|| "Say hello!".to_string());

    // Build client
    let client = match Siumai::builder()
        .openai()
        .model("gpt-4o-mini")
        .build()
        .await
    {
        Ok(c) => c,
        Err(e) => {
            return to_sse_response(
                futures::stream::once(
                    async move { Err(LlmError::ConfigurationError(e.to_string())) },
                ),
                SseOptions::default(),
            );
        }
    };

    // Start streaming
    let stream = match client.chat_stream(vec![user!(&prompt)], None).await {
        Ok(s) => s,
        Err(e) => {
            return to_sse_response(
                futures::stream::once(async move { Err(e) }),
                SseOptions::default(),
            );
        }
    };

    // Convert to SSE response
    to_sse_response(stream, SseOptions::default())
}
