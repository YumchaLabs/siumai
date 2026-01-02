//! OpenAI Responses API - Streaming provider-hosted tools events
//!
//! This example demonstrates how to observe provider-hosted tool execution in streaming mode:
//! - Web search tool emits `ChatStreamEvent::Custom` events (`openai:tool-call` / `openai:tool-result`)
//! - Use `siumai::provider_ext::openai::responses::OpenAiResponsesCustomEvent` to parse them
//!
//! Run:
//! ```bash
//! cargo run --example openai-responses-streaming-tools --features openai
//! ```

use futures_util::StreamExt;
use siumai::prelude::*;
use siumai::provider_ext::openai::{OpenAiChatRequestExt, OpenAiOptions, ResponsesApiConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder()
        .openai()
        .api_key(&std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o-mini")
        .build()
        .await?;

    let tool = siumai::hosted_tools::openai::web_search()
        .with_search_context_size("high")
        .build();

    let mut request = ChatRequest::new(vec![user!(
        "Find the latest Rust release notes and summarize."
    )])
    .with_tools(vec![tool])
    .with_openai_options(OpenAiOptions::new().with_responses_api(ResponsesApiConfig::new()));
    request.stream = true;

    let mut stream = client.chat_stream_request(request).await?;

    while let Some(ev) = stream.next().await {
        let ev = ev?;
        if let Some(custom) =
            siumai::provider_ext::openai::responses::OpenAiResponsesCustomEvent::from_stream_event(
                &ev,
            )
        {
            match custom {
                siumai::provider_ext::openai::responses::OpenAiResponsesCustomEvent::ProviderToolCall(e) => {
                    println!("provider tool call: {} ({})", e.tool_name, e.tool_call_id);
                }
                siumai::provider_ext::openai::responses::OpenAiResponsesCustomEvent::ProviderToolResult(e) => {
                    println!("provider tool result: {} ({})", e.tool_name, e.tool_call_id);
                }
                siumai::provider_ext::openai::responses::OpenAiResponsesCustomEvent::Source(e) => {
                    println!("source: {} ({})", e.url, e.id);
                }
            }
            continue;
        }

        match ev {
            ChatStreamEvent::ContentDelta { delta, .. } => print!("{delta}"),
            ChatStreamEvent::StreamEnd { .. } => break,
            _ => {}
        }
    }

    Ok(())
}
