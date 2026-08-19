use super::{Siumai, accepts_language_model, error_diagnostics};
use siumai::providers::anthropic::annotations::{AnthropicCacheTtl, AnthropicContentOptions};
use siumai::providers::anthropic::options::AnthropicMessagesOptions;
use siumai::providers::anthropic::resources::{AnthropicFiles, AnthropicMessageBatches};
use siumai::providers::anthropic::{
    AnthropicCredential, AnthropicLanguageModel, AnthropicProvider, AnthropicProviderBuilder,
};
use siumai::{LanguageRequest, Message, MessagePart, MessageRole, Model, ModelFamily};

const CREDENTIAL_CANARY: &str = "anthropic-facade-secret-canary";

fn accepts_anthropic_provider(provider: &AnthropicProvider) {
    let _ = provider;
}

fn accepts_anthropic_model(model: &AnthropicLanguageModel) {
    let _ = model;
}

#[test]
fn api_key_and_provider_owned_credential_build_the_same_typed_hub()
-> Result<(), Box<dyn std::error::Error>> {
    let api_key_hub = Siumai::builder()
        .anthropic()
        .api_key("test-api-key")
        .build()?;
    let credential_hub = Siumai::builder()
        .anthropic()
        .credential(AnthropicCredential::api_key("test-api-key"))
        .build()?;

    for hub in [&api_key_hub, &credential_hub] {
        let client = hub.language("future-anthropic-model")?;
        accepts_language_model(&client);
        accepts_anthropic_provider(hub.provider());
        accepts_anthropic_model(client.model());
        assert_eq!(
            client.descriptor().model().as_str(),
            "future-anthropic-model"
        );
    }

    Ok(())
}

#[test]
fn real_builder_configuration_and_secret_diagnostics_are_preserved()
-> Result<(), Box<dyn std::error::Error>> {
    let stage = Siumai::builder().anthropic().api_key(CREDENTIAL_CANARY);
    assert!(!format!("{stage:?}").contains(CREDENTIAL_CANARY));
    let hub = stage.build()?;
    let client = hub.language("future-anthropic-model")?;
    assert!(!format!("{hub:?}").contains(CREDENTIAL_CANARY));
    assert!(!format!("{client:?}").contains(CREDENTIAL_CANARY));

    let error = Siumai::builder()
        .anthropic()
        .api_key(CREDENTIAL_CANARY)
        .configure_provider(|builder: AnthropicProviderBuilder| {
            builder.with_base_url("not a valid URL")
        })
        .build()
        .expect_err("the real Anthropic builder must validate its endpoint");
    assert!(!error_diagnostics(&error).contains(CREDENTIAL_CANARY));

    Ok(())
}

#[test]
fn native_resources_annotations_and_model_methods_remain_typed_after_binding()
-> Result<(), Box<dyn std::error::Error>> {
    let hub = Siumai::builder()
        .anthropic()
        .api_key(CREDENTIAL_CANARY)
        .build()?;
    let client = hub.language("future-anthropic-model")?;
    let files: AnthropicFiles = client.provider().files();
    let batches: AnthropicMessageBatches = client.provider().message_batches();

    assert!(std::ptr::eq(hub.provider(), client.provider()));
    assert!(std::ptr::eq(
        Model::descriptor(&client),
        client.model().descriptor()
    ));
    assert_eq!(client.family(), ModelFamily::Language);
    assert_eq!(client.descriptor().api_mode(), Some("messages"));
    assert!(client.route_id().is_none());
    assert!(!format!("{files:?}").contains(CREDENTIAL_CANARY));
    assert!(!format!("{batches:?}").contains(CREDENTIAL_CANARY));

    let annotated = MessagePart::text("cache this prefix")
        .with_provider_annotation(&AnthropicContentOptions::one_hour())?;
    assert_eq!(
        annotated
            .annotations()
            .decode::<AnthropicContentOptions>()?,
        Some(AnthropicContentOptions::one_hour())
    );
    let request = LanguageRequest::new(vec![Message::new(MessageRole::User, [annotated])]);
    let options = AnthropicMessagesOptions::new().with_automatic_cache(AnthropicCacheTtl::OneHour);
    drop(client.model().prewarm_cache(request, options));

    Ok(())
}
