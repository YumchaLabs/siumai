use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::json;
use siumai::language;
use siumai::providers::anthropic::annotations::AnthropicContentOptions;
use siumai::providers::anthropic::models::CLAUDE_SONNET_5;
use siumai::providers::anthropic::options::AnthropicMessagesOptions;
use siumai::providers::anthropic::{
    AnthropicCredential, AnthropicLanguageResponseExt, AnthropicProvider,
};
use siumai::providers::openai::models::GPT_5_6;
use siumai::providers::openai::prompt_cache::OpenAiContentOptions;
use siumai::providers::openai::responses::OpenAiResponsesOptions;
use siumai::providers::openai::{OpenAiCredential, OpenAiProvider};
use siumai::registry::{Registry, RegistryBuilderExt};
use siumai::transport::EndpointConfig;
use siumai::{
    ContentPart, LanguageCallError, LanguageCompletionReason, LanguageInput, LanguageModel,
    LanguageRequest, LanguageResponse, Message, MessagePart, MessageRole, ModelId, ReplayDomain,
    ReplayDomainId, Usage,
};

async fn application_call<M, I>(model: &M, input: I) -> Result<LanguageResponse, LanguageCallError>
where
    M: LanguageModel + ?Sized,
    I: Into<LanguageInput>,
{
    language::call(model, input).generate().await
}

fn inspect_complete_response(
    response: &LanguageResponse,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("text: {:?}", response.output_text());
    println!("content parts: {}", response.content().len());
    println!("usage: {:?}", response.usage());
    println!(
        "provider metadata namespaces: {:?}",
        response.provider_metadata().keys().collect::<Vec<_>>()
    );

    if let Some(metadata) = response.anthropic_metadata()? {
        println!("Anthropic metadata fields: {}", metadata.raw().len());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Explicit local endpoints ensure that an accidentally polled future can
    // target only loopback while this construction-only example evolves.
    let openai = OpenAiProvider::builder(OpenAiCredential::unauthenticated())
        .with_endpoint(EndpointConfig::local_explicit("http://127.0.0.1:9/v1")?)
        .with_replay_domain(ReplayDomain::custom(ReplayDomainId::new(
            "provider-switching-openai",
        )?))
        .build()?;
    let anthropic = AnthropicProvider::builder(AnthropicCredential::unauthenticated())
        .with_endpoint(EndpointConfig::local_explicit("http://127.0.0.1:9/v1/")?)
        .with_replay_domain(ReplayDomain::custom(ReplayDomainId::new(
            "provider-switching-anthropic",
        )?))
        .build()?;

    let openai_model = openai.responses(GPT_5_6)?;
    let anthropic_model = anthropic.language(CLAUDE_SONNET_5)?;
    let erased_openai: Arc<dyn LanguageModel> = Arc::new(openai_model.clone());

    let mut registry = Registry::builder();
    registry
        .register_provider("openai", &openai)?
        .register_provider("anthropic", &anthropic)?;
    let registry = registry.build()?;
    let registry_model = registry.language_model(format!("openai:{GPT_5_6}"))?;

    let openai_part = MessagePart::text("Keep this OpenAI prefix cached.")
        .with_provider_annotation(&OpenAiContentOptions::prompt_cache_breakpoint())?;
    let anthropic_part = MessagePart::text("Keep this Anthropic prefix cached.")
        .with_provider_annotation(&AnthropicContentOptions::one_hour())?;
    let openai_request = LanguageRequest::new(vec![Message::new(MessageRole::User, [openai_part])]);
    let anthropic_request =
        LanguageRequest::new(vec![Message::new(MessageRole::User, [anthropic_part])]);

    // One application function accepts concrete, explicitly erased, and
    // Registry-resolved handles without provider matching. Dropping these
    // futures keeps the example deterministic and network-free.
    drop(application_call(&openai_model, openai_request.clone()));
    drop(application_call(
        erased_openai.as_ref(),
        openai_request.clone(),
    ));
    drop(application_call(
        &anthropic_model,
        anthropic_request.clone(),
    ));
    drop(application_call(
        registry_model.as_ref(),
        openai_request.clone(),
    ));

    let openai_options = OpenAiResponsesOptions {
        instructions: Some("Preserve exact OpenAI intent.".to_string()),
        ..OpenAiResponsesOptions::default()
    };
    let anthropic_options = AnthropicMessagesOptions::new().with_adaptive_thinking();
    drop(
        language::call(&openai_model, openai_request.clone())
            .with_provider_options(&openai_options)?
            .generate(),
    );
    drop(
        language::call(erased_openai.as_ref(), openai_request.clone())
            .with_provider_options(&openai_options)?
            .generate(),
    );
    drop(
        language::call(registry_model.as_ref(), openai_request)
            .with_provider_options(&openai_options)?
            .generate(),
    );
    drop(
        language::call(&anthropic_model, anthropic_request)
            .with_provider_options(&anthropic_options)?
            .generate(),
    );

    // Native APIs stay on the retained concrete providers. The erased handles
    // above intentionally expose only the provider-neutral LanguageModel API.
    let _openai_responses = openai.responses_resource();
    let _openai_files = openai.files();
    let _anthropic_files = anthropic.files();
    let _anthropic_batches = anthropic.message_batches();

    // Portable calls return the complete response, so generic and typed
    // provider metadata inspection remains available without a facade wrapper.
    let fixture_response = LanguageResponse::completed(
        vec![
            ContentPart::Text {
                text: "complete response".to_string(),
            },
            ContentPart::Reasoning {
                text: "preserved reasoning".to_string(),
            },
        ],
        LanguageCompletionReason::Stop,
        Usage::default()
            .with_input_tokens(4_u64)
            .with_output_tokens(2_u64),
    )?
    .with_id("msg_fixture")
    .with_model(ModelId::new(CLAUDE_SONNET_5)?)
    .with_provider_metadata(BTreeMap::from([(
        "anthropic-messages".to_string(),
        json!({
            "usage": {"service_tier": "priority"},
            "future_metadata": true
        }),
    )]));
    inspect_complete_response(&fixture_response)?;

    Ok(())
}
