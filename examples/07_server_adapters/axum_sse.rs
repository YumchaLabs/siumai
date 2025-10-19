// Axum SSE adapter example for ChatStream (English comments)
// Run: OPENAI_API_KEY=sk-... cargo run --example axum_sse --features server-adapters

use axum::Router;
use axum::extract::Query;
use axum::response::sse::Sse;
use axum::routing::get;
use futures::{Stream, StreamExt};
use serde::Deserialize;
use std::convert::Infallible;

use siumai::orchestrator::{OrchestratorStreamOptions, generate_stream};
use siumai::prelude::*;
use siumai::server_adapters::SseOptions;
use siumai::server_adapters::axum::to_sse_response;

#[derive(Debug, Deserialize)]
struct ChatQuery {
    q: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build axum app
    let app = Router::new()
        .route("/chat", get(chat_handler))
        .route("/chat/simple", get(chat_handler_simple));

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], 8080));
    println!("Listening on http://{addr}/chat?q=hello");
    println!("Simple version: http://{addr}/chat/simple?q=hello");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

/// Simple chat handler using the new `to_sse_response` helper
async fn chat_handler_simple(
    Query(ChatQuery { q }): Query<ChatQuery>,
) -> Sse<impl Stream<Item = Result<axum::response::sse::Event, Infallible>> + Send> {
    // Fallback prompt
    let prompt = q.unwrap_or_else(|| "Say hello in one sentence.".to_string());

    // Build OpenAI client directly (requires OPENAI_API_KEY)
    let client = match siumai::builder::LlmBuilder::new()
        .openai()
        .model("gpt-4o-mini")
        .build()
        .await
    {
        Ok(c) => c,
        Err(e) => {
            // Return error stream
            let error_stream = futures::stream::once(async move {
                Ok(ChatStreamEvent::Error {
                    error: format!("client error: {}", e.user_message()),
                })
            });
            return to_sse_response(Box::pin(error_stream), SseOptions::production());
        }
    };

    // Start orchestrated streaming (no tools for simplicity)
    let orchestration = match generate_stream(
        &client,
        vec![siumai::types::ChatMessage::user(prompt).build()],
        None,
        None,
        OrchestratorStreamOptions::default(),
    )
    .await
    {
        Ok(o) => o,
        Err(e) => {
            // Return error stream
            let error_stream = futures::stream::once(async move {
                Ok(ChatStreamEvent::Error {
                    error: format!("start error: {}", e.user_message()),
                })
            });
            return to_sse_response(Box::pin(error_stream), SseOptions::production());
        }
    };

    // Use the new helper function with production settings (errors masked)
    to_sse_response(orchestration.stream, SseOptions::production())
}

/// Original chat handler (kept for comparison)
async fn chat_handler(
    Query(ChatQuery { q }): Query<ChatQuery>,
) -> Sse<impl Stream<Item = Result<axum::response::sse::Event, Infallible>> + Send> {
    // Fallback prompt
    let prompt = q.unwrap_or_else(|| "Say hello in one sentence.".to_string());

    // Build OpenAI client directly (requires OPENAI_API_KEY)
    let client = match siumai::builder::LlmBuilder::new()
        .openai()
        .model("gpt-4o-mini")
        .build()
        .await
    {
        Ok(c) => c,
        Err(e) => {
            let error_stream = futures::stream::once(async move {
                Ok(ChatStreamEvent::Error {
                    error: format!("client error: {}", e.user_message()),
                })
            });
            return to_sse_response(Box::pin(error_stream), SseOptions::development());
        }
    };

    // Start orchestrated streaming (no tools for simplicity)
    let orchestration = match generate_stream(
        &client,
        vec![siumai::types::ChatMessage::user(prompt).build()],
        None,
        None,
        OrchestratorStreamOptions::default(),
    )
    .await
    {
        Ok(o) => o,
        Err(e) => {
            let error_stream = futures::stream::once(async move {
                Ok(ChatStreamEvent::Error {
                    error: format!("start error: {}", e.user_message()),
                })
            });
            return to_sse_response(Box::pin(error_stream), SseOptions::development());
        }
    };

    // Use development settings (errors not masked) for debugging
    to_sse_response(orchestration.stream, SseOptions::development())
}
