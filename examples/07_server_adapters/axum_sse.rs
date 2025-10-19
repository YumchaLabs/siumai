// Axum SSE adapter example for ChatStream (English comments)
// Run: OPENAI_API_KEY=sk-... cargo run --example axum_sse

use axum::Router;
use axum::extract::Query;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::routing::get;
use futures::{Stream, StreamExt};
use serde::Deserialize;
use serde_json::json;
use std::convert::Infallible;
use std::pin::Pin;

use siumai::error::LlmError;
use siumai::orchestrator::{OrchestratorStreamOptions, generate_stream};
use siumai::prelude::*;
use siumai::stream::{ChatStream, ChatStreamEvent};

#[derive(Debug, Deserialize)]
struct ChatQuery {
    q: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build axum app
    let app = Router::new().route("/chat", get(chat_handler));
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], 8080));
    println!("Listening on http://{addr}/chat?q=hello");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

type BoxEventStream = Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>>;

async fn chat_handler(Query(ChatQuery { q }): Query<ChatQuery>) -> Sse<BoxEventStream> {
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
        Err(e) => return sse_error(format!("client error: {}", e.user_message())),
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
        Err(e) => return sse_error(format!("start error: {}", e.user_message())),
    };

    let stream = orchestration.stream.map(|item| Ok(chat_event_to_sse(item)));
    Sse::new(Box::pin(stream) as BoxEventStream)
}

fn sse_error(msg: String) -> Sse<BoxEventStream> {
    let ev = Event::default().event("error").data(msg);
    let stream = futures::stream::once(async move { Ok(ev) });
    Sse::new(Box::pin(stream) as BoxEventStream)
}

fn chat_event_to_sse(item: Result<ChatStreamEvent, LlmError>) -> Event {
    match item {
        Ok(ChatStreamEvent::ContentDelta { delta, index }) => {
            json_event("content", json!({"delta": delta, "index": index}))
        }
        Ok(ChatStreamEvent::ThinkingDelta { delta }) => {
            json_event("thinking", json!({"delta": delta}))
        }
        Ok(ChatStreamEvent::ToolCallDelta {
            id,
            function_name,
            arguments_delta,
            index,
        }) => json_event(
            "tool_call",
            json!({"id": id, "function_name": function_name, "arguments_delta": arguments_delta, "index": index}),
        ),
        Ok(ChatStreamEvent::UsageUpdate { usage }) => json_event("usage", json!({"usage": usage})),
        Ok(ChatStreamEvent::StreamStart { metadata }) => {
            json_event("start", json!({"metadata": metadata}))
        }
        Ok(ChatStreamEvent::StreamEnd { response }) => {
            let text = response
                .content_text()
                .map(|s| s.to_string())
                .or_else(|| response.text());
            json_event(
                "end",
                json!({"text": text, "finish_reason": response.finish_reason, "usage": response.usage}),
            )
        }
        Ok(ChatStreamEvent::Error { error }) => json_event("error", json!({"message": error})),
        Err(e) => json_event("error", json!({"message": e.user_message()})),
    }
}

fn json_event(name: &str, payload: serde_json::Value) -> Event {
    // Use .data with serialized JSON to avoid dealing with Result from .json_data
    let data = serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_string());
    Event::default().event(name).data(data)
}
