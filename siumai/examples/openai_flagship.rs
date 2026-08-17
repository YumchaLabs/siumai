use std::time::Duration;

use siumai::families::language;
use siumai::providers::openai::models::GPT_5_6;
use siumai::providers::openai::responses::{
    OpenAiReasoning, OpenAiReasoningEffort, OpenAiResponsesOptions,
};
use siumai::providers::openai::{OpenAiCredential, OpenAiProvider};
use siumai::transport::ProviderHttpTransportSettings;
use siumai::{CallOptions, LanguageRequest, Message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Ok(api_key) = std::env::var("OPENAI_API_KEY") else {
        eprintln!("Set OPENAI_API_KEY to run the OpenAI flagship example.");
        return Ok(());
    };

    let http_settings =
        ProviderHttpTransportSettings::default().with_call_timeout(Duration::from_secs(120))?;
    let provider = OpenAiProvider::builder(OpenAiCredential::api_key(api_key))
        .with_http_transport_settings(http_settings)
        .build()?;
    let model = provider.responses(GPT_5_6)?;
    let provider_options = OpenAiResponsesOptions::default()
        .with_reasoning(OpenAiReasoning::default().with_effort(OpenAiReasoningEffort::High));
    let call_options =
        CallOptions::default().with_provider_options_for(&model, &provider_options)?;

    let response = language::generate_with_options(
        &model,
        LanguageRequest::new(vec![Message::user(
            "Explain why exact-target provider options are useful in one sentence.",
        )]),
        call_options,
    )
    .await?;
    println!(
        "received {} portable content parts",
        response.content().len()
    );

    if let Ok(conversation_id) = std::env::var("OPENAI_CONVERSATION_ID") {
        let conversation = provider.conversations().retrieve(&conversation_id).await?;
        println!(
            "retrieved a conversation with {} metadata entries",
            conversation.metadata.len()
        );
    }

    Ok(())
}
