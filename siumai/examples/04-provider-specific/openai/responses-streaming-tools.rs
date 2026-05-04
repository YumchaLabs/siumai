//! OpenAI Responses API - Streaming provider-hosted tools events
//!
//! This example demonstrates how to observe provider-hosted tool execution in streaming mode:
//! - Web search tool emits stable `Part` / `PartWithReplay` events, with legacy custom-event
//!   compatibility where needed
//! - Use `siumai::provider_ext::openai::ext::responses::OpenAiResponsesCustomEvent` to parse the
//!   normalized extension semantics
//!
//! Run:
//! ```bash
//! cargo run --example openai-responses-streaming-tools --features openai
//! ```

use futures_util::StreamExt;
use siumai::prelude::unified::*;
use siumai::provider_ext::openai::{OpenAiChatRequestExt, OpenAiOptions, ResponsesApiConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Recommended construction: resolve a model handle from the registry.
    // Note: API key is automatically read from `OPENAI_API_KEY`.
    let model = registry::global().language_model("openai:gpt-4o-mini")?;

    let tool = siumai::hosted_tools::openai::web_search()
        .with_search_context_size("high")
        .build();

    let request = ChatRequest::new(vec![user!(
        "Find the latest Rust release notes and summarize."
    )])
    .with_tools(vec![tool])
    .with_openai_options(OpenAiOptions::new().with_responses_api(ResponsesApiConfig::new()));

    let mut stream = text::stream(&model, request, text::StreamOptions::default()).await?;
    let mut deltas = text::StreamDeltaExtractor::new();

    while let Some(ev) = stream.next().await {
        let ev = ev?;
        if let Some(custom) =
            siumai::provider_ext::openai::ext::responses::OpenAiResponsesCustomEvent::from_stream_event(
                &ev,
            )
        {
            match custom {
                siumai::provider_ext::openai::ext::responses::OpenAiResponsesCustomEvent::ProviderToolCall(e) => {
                    println!("provider tool call: {} ({})", e.tool_name, e.tool_call_id);
                }
                siumai::provider_ext::openai::ext::responses::OpenAiResponsesCustomEvent::ProviderToolResult(e) => {
                    println!("provider tool result: {} ({})", e.tool_name, e.tool_call_id);
                }
                siumai::provider_ext::openai::ext::responses::OpenAiResponsesCustomEvent::Source(e) => {
                    println!("source: {} ({})", e.url, e.id);
                }
            }
            continue;
        }

        if let Some(delta) = deltas.text_delta(&ev) {
            print!("{delta}");
            continue;
        }

        match ev {
            ChatStreamEvent::StreamEnd { .. } => break,
            _ => {}
        }
    }

    Ok(())
}
