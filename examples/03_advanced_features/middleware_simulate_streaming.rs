// Simulate streaming middleware demo
// This example shows how to use the SimulateStreamingMiddleware in isolation
// by wrapping a minimal base stream function that only emits a final StreamEnd.

use futures::StreamExt;
use siumai::middleware::language_model::StreamAsyncFn;
use siumai::middleware::samples::simulate_streaming_middleware;
use siumai::types::{ChatRequest, ChatResponse, MessageContent};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Base stream: no deltas, only a final StreamEnd with some text
    let base: std::sync::Arc<StreamAsyncFn> = std::sync::Arc::new(|_req: ChatRequest| {
        Box::pin(async move {
            let s = async_stream::try_stream! {
                yield siumai::stream::ChatStreamEvent::StreamEnd { response: ChatResponse::new(MessageContent::Text("Hello from final only".into())) };
            };
            Ok(Box::pin(s) as siumai::stream::ChatStream)
        })
    });

    // Build the middleware: split into 5-char chunks (no delay)
    let mw = simulate_streaming_middleware(5, None);
    let wrapped = mw.wrap_stream_async(base);

    // Run wrapped stream
    let req = ChatRequest::new(vec![]);
    let stream = wrapped(req).await?;
    futures::pin_mut!(stream);
    println!("Simulated streaming chunks:");
    while let Some(ev) = stream.next().await {
        match ev? {
            siumai::stream::ChatStreamEvent::ContentDelta { delta, .. } => println!("delta: {}", delta),
            siumai::stream::ChatStreamEvent::StreamEnd { .. } => println!("<end>"),
            _ => {}
        }
    }
    Ok(())
}

