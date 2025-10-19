// Structured JSON messages example (requires features: openai, structured-messages)
// English comments as requested.

#[cfg(all(feature = "openai", feature = "structured-messages"))]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use serde_json::json;
    use siumai::prelude::*;

    // Ensure API key exists
    if std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!("Set OPENAI_API_KEY to run this example");
        return Ok(());
    }

    // Build OpenAI client using Responses API to echo/handle structured inputs more naturally
    let client = siumai::builder::LlmBuilder::new()
        .openai()
        .model("gpt-4o-mini")
        .use_responses_api(true)
        .build()
        .await?;

    // Compose messages: system + user(JSON)
    let system = siumai::types::ChatMessage::system(
        "You are a helpful assistant. If user sends JSON, summarize keys.",
    )
    .build();

    // Use structured JSON content (MessageContent::Json)
    let user_json = json!({
        "action": "greet",
        "name": "Alice",
        "details": {"lang": "en", "caps": ["concise", "polite"]}
    });
    let user = siumai::types::ChatMessage {
        role: siumai::types::MessageRole::User,
        content: siumai::types::MessageContent::Json(user_json),
        metadata: Default::default(),
        tool_calls: None,
        tool_call_id: None,
    };

    let resp = client.chat(vec![system, user]).await?;
    println!("Assistant> {}", resp.content_text().unwrap_or("<no text>"));
    Ok(())
}

#[cfg(not(all(feature = "openai", feature = "structured-messages")))]
fn main() {
    eprintln!("Enable features: openai, structured-messages to run this example");
}
