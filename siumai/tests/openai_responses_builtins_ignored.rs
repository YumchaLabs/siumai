//! Ignored integration test: OpenAI Responses API with built-in tools (advanced)
//! Requires network and OPENAI_API_KEY; run with:
//!   cargo test --test openai_responses_builtins_ignored --features openai -- --ignored

use siumai::ChatCapability;

#[cfg(feature = "openai")]
#[tokio::test]
#[ignore]
async fn responses_with_advanced_builtins_smoke() {
    use siumai::providers::openai::{OpenAiClient, OpenAiConfig};
    use siumai::types::{ChatMessage, OpenAiBuiltInTool};

    // Skip if no key set
    if std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!("OPENAI_API_KEY not set; skipping");
        return;
    }

    let config = OpenAiConfig::new(std::env::var("OPENAI_API_KEY").unwrap())
        .with_model("gpt-4.1-mini")
        .with_responses_api(true)
        .with_built_in_tools(vec![
            // web search with provider hints
            OpenAiBuiltInTool::WebSearchAdvanced {
                extra: serde_json::json!({"region":"us","safe":"strict"}),
            },
            // file search (no vector stores provided here)
            OpenAiBuiltInTool::FileSearchAdvanced {
                vector_store_ids: None,
                extra: serde_json::json!({}),
            },
        ]);
    let client = OpenAiClient::new_with_config(config);

    // JSON input will be mapped as input_json
    #[cfg(feature = "structured-messages")]
    let msg = ChatMessage::user("")
        .with_json(serde_json::json!({"query":"hello"}))
        .build();
    #[cfg(not(feature = "structured-messages"))]
    let msg = ChatMessage::user("hello").build();

    let _ = client.chat(vec![msg]).await.expect("responses chat");
}
