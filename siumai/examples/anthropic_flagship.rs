use siumai::providers::anthropic::annotations::{
    AnthropicCacheTtl, AnthropicContentOptions, AnthropicMessageFile,
};
use siumai::providers::anthropic::models::CLAUDE_SONNET_5;
use siumai::providers::anthropic::options::AnthropicMessagesOptions;
use siumai::providers::anthropic::resources::{AnthropicFiles, AnthropicMessageBatches};
use siumai::{LanguageRequest, Message, MessagePart, MessageRole, Siumai};

async fn flagship() -> Result<(), Box<dyn std::error::Error>> {
    let ai = Siumai::builder()
        .anthropic()
        .api_key("example-anthropic-key")
        .build()?;
    let client = ai.language(CLAUDE_SONNET_5)?;

    let files: AnthropicFiles = client.provider().files();
    let file_reference = files.reference("file_fixture")?;
    let file_part =
        AnthropicContentOptions::file_part(AnthropicMessageFile::document(file_reference))?;
    let cached_prefix = MessagePart::text("Keep this stable prefix cached.")
        .with_provider_annotation(&AnthropicContentOptions::one_hour())?;
    let request = LanguageRequest::new(vec![Message::new(
        MessageRole::User,
        [
            file_part,
            cached_prefix,
            MessagePart::text("Summarize the referenced document in one sentence."),
        ],
    )]);
    let provider_options = AnthropicMessagesOptions::new()
        .with_adaptive_thinking()
        .with_automatic_cache(AnthropicCacheTtl::OneHour);
    let response = client
        .call(request.clone())
        .with_provider_options(&provider_options)?
        .generate()
        .await?;

    // The portable result remains complete rather than collapsing to text.
    let _content = response.content();
    let _termination = response.termination();
    let _usage = response.usage();
    let _warnings = response.warnings();
    let _provider_metadata = response.provider_metadata();

    // Portable execution and the model-native cache prewarm path share the
    // provider-owned request policy.
    drop(client.model().prewarm_cache(request, provider_options));

    let _batches: AnthropicMessageBatches = client.provider().message_batches();

    Ok(())
}

fn main() {
    // The future is compiled but never polled, so the synthetic credential
    // cannot produce a provider request or billable work.
    drop(flagship());
}
