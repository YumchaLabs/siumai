// Actix SSE adapter example for ChatStream
// Run: OPENAI_API_KEY=sk-... cargo run --example actix_sse

use actix_web::{App, HttpResponse, HttpServer, Responder, get, web};
use futures::{Stream, StreamExt};
use serde::Deserialize;
use serde_json::json;
use std::pin::Pin;

use siumai::error::LlmError;
use siumai::orchestrator::{OrchestratorStreamOptions, StepResult, generate_stream};
use siumai::stream::ChatStreamEvent;

#[derive(Debug, Deserialize)]
struct ChatQuery {
    q: Option<String>,
}

#[get("/chat")]
async fn chat(query: web::Query<ChatQuery>) -> impl Responder {
    let prompt = query
        .q
        .clone()
        .unwrap_or_else(|| "Say hello in one sentence.".to_string());
    let client = match siumai::builder::LlmBuilder::new()
        .openai()
        .model("gpt-4o-mini")
        .build()
        .await
    {
        Ok(c) => c,
        Err(e) => return sse_error(format!("client error: {}", e.user_message())),
    };

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

    // Convert ChatStreamEvent -> SSE text (event: name + data: json)
    let event_stream = orchestration.stream.map(|item| chat_event_to_sse(item));

    // Append a summary frame after stream ends (step usage aggregation)
    let steps_rx = orchestration.steps;
    let summary = async_stream::stream! {
        if let Ok(steps) = steps_rx.await {
            let total = StepResult::merge_usage(&steps);
            let data = json!({"total_usage": total});
            let frame = format!("event: summary\ndata: {}\n\n", serde_json::to_string(&data).unwrap_or("{}".into()));
            yield Ok::<_, actix_web::Error>(actix_web::web::Bytes::from(frame));
        }
    };

    let sse_stream = event_stream.chain(summary);

    HttpResponse::Ok()
        .append_header(("Content-Type", "text/event-stream"))
        .append_header(("Cache-Control", "no-cache"))
        .append_header(("Connection", "keep-alive"))
        .streaming(sse_stream)
}

fn sse_error(msg: String) -> HttpResponse {
    let frame = format!(
        "event: error\ndata: {}\n\n",
        serde_json::to_string(&json!({"message": msg})).unwrap()
    );
    HttpResponse::Ok()
        .append_header(("Content-Type", "text/event-stream"))
        .body(frame)
}

fn chat_event_to_sse(
    item: Result<ChatStreamEvent, LlmError>,
) -> Result<actix_web::web::Bytes, actix_web::Error> {
    let (name, payload) = match item {
        Ok(ChatStreamEvent::ContentDelta { delta, index }) => {
            ("content", json!({"delta": delta, "index": index}))
        }
        Ok(ChatStreamEvent::ThinkingDelta { delta }) => ("thinking", json!({"delta": delta})),
        Ok(ChatStreamEvent::ToolCallDelta {
            id,
            function_name,
            arguments_delta,
            index,
        }) => (
            "tool_call",
            json!({"id": id, "function_name": function_name, "arguments_delta": arguments_delta, "index": index}),
        ),
        Ok(ChatStreamEvent::UsageUpdate { usage }) => ("usage", json!({"usage": usage})),
        Ok(ChatStreamEvent::StreamStart { metadata }) => ("start", json!({"metadata": metadata})),
        Ok(ChatStreamEvent::StreamEnd { response }) => {
            let text = response
                .content_text()
                .map(|s| s.to_string())
                .or_else(|| response.text());
            (
                "end",
                json!({"text": text, "finish_reason": response.finish_reason, "usage": response.usage}),
            )
        }
        Ok(ChatStreamEvent::Error { error }) => ("error", json!({"message": error})),
        Err(e) => ("error", json!({"message": e.user_message()})),
    };
    let data = serde_json::to_string(&payload).unwrap_or("{}".into());
    let frame = format!("event: {}\ndata: {}\n\n", name, data);
    Ok(actix_web::web::Bytes::from(frame))
}

#[cfg(all(feature = "server-adapters", feature = "openai"))]
#[tokio::main]
async fn main() -> std::io::Result<()> {
    let addr = ("127.0.0.1", 8081);
    println!("Listening on http://{}:{}/chat?q=hello", addr.0, addr.1);
    HttpServer::new(|| App::new().service(chat))
        .bind(addr)?
        .run()
        .await
}

#[cfg(not(all(feature = "server-adapters", feature = "openai")))]
fn main() {
    eprintln!("Enable features: server-adapters, openai to run this example");
}
