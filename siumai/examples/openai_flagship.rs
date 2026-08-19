use siumai::providers::openai::models::GPT_5_6;
use siumai::providers::openai::prompt_cache::OpenAiContentOptions;
use siumai::providers::openai::resources::files::OpenAiFiles;
use siumai::providers::openai::responses::{
    OpenAiReasoning, OpenAiReasoningEffort, OpenAiResponsesOptions, OpenAiResponsesResource,
};
use siumai::{LanguageRequest, Message, MessagePart, MessageRole, Siumai};

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    let call = client
        .call(request.clone())
        .with_provider_options(&provider_options)?;

    // Portable execution still uses the existing family call path. The future
    // is not polled so this flagship remains an offline compile contract.
    drop(call.generate());

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
