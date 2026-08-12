use siumai::core::MessagePart;
use siumai::families::language;
use siumai::providers::anthropic::annotations::{
    AnthropicCacheTtl, AnthropicContentOptions, AnthropicMessageFile,
};
use siumai::providers::anthropic::models::CLAUDE_SONNET_5;
use siumai::providers::anthropic::options::AnthropicMessagesOptions;
use siumai::providers::anthropic::resources::AnthropicSkillListQuery;
use siumai::providers::anthropic::{AnthropicCredential, AnthropicProvider};
use siumai::{CallOptions, LanguageRequest, Message, MessageRole, ReplayDomainId};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") else {
        eprintln!("Set ANTHROPIC_API_KEY to run the Anthropic flagship example.");
        return Ok(());
    };

    let caller_scope =
        std::env::var("ANTHROPIC_CALLER_SCOPE").unwrap_or_else(|_| "flagship-example".to_string());
    let provider = AnthropicProvider::builder(AnthropicCredential::api_key(api_key))
        .with_caller_scope(ReplayDomainId::new(caller_scope)?)
        .build()?;
    let model = provider.language(CLAUDE_SONNET_5)?;
    let user_message = if let Ok(file_id) = std::env::var("ANTHROPIC_FILE_ID") {
        let reference = provider.files().reference(file_id)?;
        let file_part =
            AnthropicContentOptions::file_part(AnthropicMessageFile::document(reference))?;
        Message::new(
            MessageRole::User,
            [
                file_part,
                MessagePart::text("Summarize the referenced document in one sentence."),
            ],
        )
    } else {
        Message::user("Explain provider-owned replay in one sentence.")
    };

    let provider_options = AnthropicMessagesOptions::new()
        .with_adaptive_thinking()
        .with_automatic_cache(AnthropicCacheTtl::OneHour);
    let call_options =
        CallOptions::default().with_provider_options_for(&model, &provider_options)?;
    let response = language::generate_with_options(
        &model,
        LanguageRequest::new(vec![user_message]),
        call_options,
    )
    .await?;

    let (assistant_history, omissions) = response.project_assistant_history().into_parts();
    if let Some(assistant_history) = assistant_history {
        let continuation = LanguageRequest::new(vec![
            assistant_history,
            Message::user("Continue while preserving provider-owned replay state."),
        ]);
        println!(
            "prepared a {}-message continuation with {} explicit omissions",
            continuation.messages.len(),
            omissions.len()
        );
    }

    let skills = provider
        .skills()
        .list(AnthropicSkillListQuery::default())
        .await?;
    println!(
        "listed {} provider-owned Skill metadata records",
        skills.data.len()
    );

    Ok(())
}
