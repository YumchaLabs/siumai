//! Anthropic Extensions - Thinking Replay (explicit extension API)
//!
//! This example demonstrates how to replay Anthropic thinking blocks across turns,
//! aligned with the Vercel AI SDK behavior:
//! - The first response captures `thinking_signature` / `redacted_thinking_data` in `provider_metadata`.
//! - We convert that response into a replayable assistant message by copying those fields into
//!   `ChatMessage.metadata.custom`.
//! - The next request includes that assistant message so the provider can validate/replay reasoning blocks.
//!
//! ## Run
//! ```bash
//! cargo run --example thinking-replay-ext --features anthropic
//! ```

use siumai::prelude::unified::*;
use siumai::provider_ext::anthropic::ThinkingModeConfig;
use siumai::provider_ext::anthropic::thinking;
use siumai::user;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Siumai::builder()
        .anthropic()
        .api_key(&std::env::var("ANTHROPIC_API_KEY")?)
        .model("claude-3-7-sonnet-20250219")
        .build()
        .await?;

    let question =
        user!("Prove that the sum of the first n odd numbers equals n^2. Show your reasoning.");

    let request = ChatRequest::builder().message(question.clone()).build();

    let response = thinking::chat_with_thinking(
        &client,
        request,
        ThinkingModeConfig {
            enabled: true,
            thinking_budget: Some(5_000),
        },
    )
    .await?;

    println!("Answer:\n{}\n", response.content_text().unwrap_or_default());

    let replayable_assistant = thinking::assistant_message_with_thinking_metadata(&response);

    let follow_up = user!("Now summarize the proof in 3 bullet points.");
    let request = ChatRequest::builder()
        .messages(vec![question, replayable_assistant, follow_up])
        .build();

    let response = client.chat_request(request).await?;
    println!("Summary:\n{}", response.content_text().unwrap_or_default());

    Ok(())
}
