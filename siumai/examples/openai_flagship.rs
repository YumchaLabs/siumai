use siumai::providers::openai::models::GPT_5_6;
use siumai::providers::openai::prompt_cache::OpenAiContentOptions;
use siumai::providers::openai::resources::files::OpenAiFiles;
use siumai::providers::openai::responses::{
    OpenAiReasoning, OpenAiReasoningEffort, OpenAiResponsesOptions, OpenAiResponsesResource,
};
use siumai::{LanguageRequest, Message, MessagePart, MessageRole, Siumai};

async fn flagship() -> Result<(), Box<dyn std::error::Error>> {
    let ai = Siumai::builder()
        .openai()
        .api_key("example-openai-key")
        .build()?;
    let client = ai.language(GPT_5_6)?;

    let cached_prefix = MessagePart::text("Keep this stable prefix cached.")
        .with_provider_annotation(&OpenAiContentOptions::prompt_cache_breakpoint())?;
    let request = LanguageRequest::new(vec![Message::new(
        MessageRole::User,
        [
            cached_prefix,
            MessagePart::text(
                "Explain why exact-target provider options are useful in one sentence.",
            ),
        ],
    )]);
    let provider_options = OpenAiResponsesOptions::default()
        .with_reasoning(OpenAiReasoning::default().with_effort(OpenAiReasoningEffort::High));
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

    // Provider-wide and model-native APIs remain concrete and discoverable.
    let _files: OpenAiFiles = client.provider().files();
    let _responses: OpenAiResponsesResource = client.provider().responses_resource();
    drop(
        client
            .model()
            .generate_native(request, siumai::CallOptions::default()),
    );

    Ok(())
}

fn main() {
    // The future is compiled but never polled, so the synthetic credential
    // cannot produce a provider request or billable work.
    drop(flagship());
}
