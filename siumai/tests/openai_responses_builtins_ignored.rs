//! Ignored integration test: OpenAI Responses API with built-in tools (advanced)
//! Requires network and OPENAI_API_KEY; run with:
//!   cargo test --test openai_responses_builtins_ignored --features openai -- --ignored

#[cfg(feature = "openai")]
#[tokio::test]
#[ignore]
async fn responses_with_advanced_builtins_smoke() {
    use siumai::prelude::ChatCapability;
    use siumai::prelude::unified::ChatMessage;
    use siumai::provider_ext::openai::{OpenAiClient, OpenAiConfig};

    // Skip if no key set
    if std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!("OPENAI_API_KEY not set; skipping");
        return;
    }

    let config =
        OpenAiConfig::new(std::env::var("OPENAI_API_KEY").unwrap()).with_model("gpt-4.1-mini");
    let client = OpenAiClient::new_with_config(config);

    // JSON input will be mapped as input_json
    #[cfg(feature = "structured-messages")]
    let msg = {
        use siumai::prelude::unified::{MessageContent, MessageMetadata, MessageRole};
        ChatMessage {
            role: MessageRole::User,
            content: MessageContent::Json(serde_json::json!({"query":"hello"})),
            metadata: MessageMetadata::default(),
        }
    };
    #[cfg(not(feature = "structured-messages"))]
    let msg = ChatMessage::user("hello").build();

    let _ = client.chat(vec![msg]).await.expect("responses chat");
}
