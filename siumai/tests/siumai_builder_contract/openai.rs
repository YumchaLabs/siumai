use super::{Siumai, accepts_language_model, error_diagnostics};
use siumai::providers::openai::chat_completions::{
    OpenAiChatCompletionsModel, OpenAiChatCompletionsOptions,
};
use siumai::providers::openai::embeddings::{
    OpenAiEmbeddingModel, OpenAiEmbeddingOptions, TEXT_EMBEDDING_3_SMALL,
};
use siumai::providers::openai::models::GPT_5_6;
use siumai::providers::openai::resources::files::OpenAiFiles;
use siumai::providers::openai::responses::{
    OpenAiResponsesModel, OpenAiResponsesOptions, OpenAiResponsesResource,
};
use siumai::providers::openai::{
    OpenAiConfigError, OpenAiCredential, OpenAiProvider, OpenAiProviderBuilder,
};
use siumai::{
    CallOptions, EmbeddingLimits, EmbeddingModel, Model, ModelFamily, ProviderOptionError,
};

const CREDENTIAL_CANARY: &str = "openai-facade-secret-canary";

fn accepts_embedding_model<M>(model: &M)
where
    M: EmbeddingModel + ?Sized,
{
    let _ = model;
}

fn accepts_model<M>(model: &M)
where
    M: Model + ?Sized,
{
    let _ = model;
}

fn accepts_openai_provider(provider: &OpenAiProvider) {
    let _ = provider;
}

fn accepts_openai_responses_model(model: &OpenAiResponsesModel) {
    let _ = model;
}

fn accepts_openai_embedding_model(model: &OpenAiEmbeddingModel) {
    let _ = model;
}

fn accepts_openai_chat_model(model: &OpenAiChatCompletionsModel) {
    let _ = model;
}

#[test]
fn api_key_chain_builds_a_reusable_multi_family_hub() -> Result<(), Box<dyn std::error::Error>> {
    let hub = Siumai::builder().openai().api_key("test-api-key").build()?;

    let language = hub.language(GPT_5_6)?;
    let embedding = hub.embedding(TEXT_EMBEDDING_3_SMALL)?;

    accepts_language_model(&language);
    accepts_embedding_model(&embedding);
    accepts_openai_provider(hub.provider());
    accepts_openai_provider(language.provider());
    accepts_openai_responses_model(language.model());
    accepts_openai_embedding_model(embedding.model());
    accepts_model(language.model());
    accepts_model(embedding.model());
    assert!(std::ptr::eq(hub.provider(), language.provider()));
    assert!(std::ptr::eq(hub.provider(), embedding.provider()));

    Ok(())
}

#[cfg(feature = "registry")]
#[test]
fn registry_registration_preserves_openai_identity_and_route()
-> Result<(), Box<dyn std::error::Error>> {
    use siumai::registry::{Registry, RegistryBuilderExt};

    let hub = Siumai::builder().openai().api_key("test-api-key").build()?;
    let direct = hub.language(GPT_5_6)?;
    let mut registry = Registry::builder();
    registry.register_provider("production", hub.provider())?;
    let registry = registry.build()?;
    let routed = registry.language_model(format!("production:{GPT_5_6}"))?;

    assert_eq!(direct.descriptor().scope(), routed.descriptor().scope());
    assert_eq!(direct.model_id(), routed.model_id());
    assert_eq!(direct.family(), routed.family());
    assert_eq!(
        direct.descriptor().instance_id(),
        routed.descriptor().instance_id()
    );
    assert_eq!(
        routed.route_id().map(siumai::core::RouteId::as_str),
        Some("production")
    );

    Ok(())
}

#[test]
fn provider_owned_credential_uses_the_same_public_shape() -> Result<(), Box<dyn std::error::Error>>
{
    let credential = OpenAiCredential::api_key("test-api-key");
    let hub = Siumai::builder().openai().credential(credential).build()?;
    let language = hub.language(GPT_5_6)?;

    accepts_language_model(&language);
    accepts_openai_provider(language.provider());
    accepts_model(language.model());

    Ok(())
}

#[test]
fn native_access_identity_limits_and_exact_options_survive_typed_wrapping()
-> Result<(), Box<dyn std::error::Error>> {
    let hub = Siumai::builder()
        .openai()
        .api_key(CREDENTIAL_CANARY)
        .build()?;
    let responses = hub.language(GPT_5_6)?;
    let chat = hub.chat_completions(GPT_5_6)?;
    let embedding = hub.embedding(TEXT_EMBEDDING_3_SMALL)?;

    let files: OpenAiFiles = responses.provider().files();
    let responses_resource: OpenAiResponsesResource = responses.provider().responses_resource();
    assert!(!format!("{files:?}").contains(CREDENTIAL_CANARY));
    assert!(!format!("{responses_resource:?}").contains(CREDENTIAL_CANARY));

    assert!(std::ptr::eq(hub.provider(), responses.provider()));
    assert!(std::ptr::eq(hub.provider(), chat.provider()));
    assert!(std::ptr::eq(hub.provider(), embedding.provider()));
    assert!(std::ptr::eq(
        Model::descriptor(&responses),
        responses.model().descriptor()
    ));
    assert!(responses.route_id().is_none());
    assert_eq!(responses.family(), ModelFamily::Language);
    assert_eq!(embedding.family(), ModelFamily::Embedding);
    assert_eq!(responses.descriptor().api_mode(), Some("responses"));
    assert_eq!(chat.descriptor().api_mode(), Some("chat-completions"));
    assert_eq!(embedding.descriptor().api_mode(), Some("embeddings"));
    assert_eq!(
        responses.descriptor().instance_id(),
        chat.descriptor().instance_id()
    );
    assert_eq!(
        responses.descriptor().instance_id(),
        embedding.descriptor().instance_id()
    );
    assert_eq!(
        EmbeddingModel::limits(&embedding),
        EmbeddingLimits {
            max_inputs: Some(2_048),
            max_input_tokens: Some(8_192),
        }
    );
    assert_eq!(
        EmbeddingModel::limits(&embedding),
        EmbeddingModel::limits(embedding.model())
    );

    let matching = CallOptions::default().with_provider_options_for(
        responses.model(),
        &OpenAiResponsesOptions {
            instructions: Some("preserve exact target".to_string()),
            ..OpenAiResponsesOptions::default()
        },
    )?;
    let matching_call = responses.call("hello").with_options(matching)?;
    assert_eq!(
        matching_call
            .base_options()
            .provider_options_for(&responses)?
            .typed()
            .count(),
        1
    );

    let independent = Siumai::builder()
        .openai()
        .api_key("independent-openai-key")
        .build()?
        .language(GPT_5_6)?;
    assert_ne!(
        responses.descriptor().instance_id(),
        independent.descriptor().instance_id()
    );
    let foreign_instance = CallOptions::default()
        .with_provider_options_for(independent.model(), &OpenAiResponsesOptions::default())?;
    let instance_error = match responses.call("hello").with_options(foreign_instance) {
        Ok(_) => panic!("options from another configured instance must be rejected"),
        Err(error) => error,
    };
    assert!(matches!(
        instance_error,
        ProviderOptionError::ExactTargetMismatch { .. }
    ));

    let foreign_mode = CallOptions::default()
        .with_provider_options_for(chat.model(), &OpenAiChatCompletionsOptions::default())?;
    let mode_error = match responses.call("hello").with_options(foreign_mode) {
        Ok(_) => panic!("options for Chat Completions must not target Responses"),
        Err(error) => error,
    };
    assert!(matches!(
        mode_error,
        ProviderOptionError::TargetMismatch { .. }
    ));

    let foreign_family = CallOptions::default().with_provider_options_for(
        embedding.model(),
        &OpenAiEmbeddingOptions::new().with_user("facade-test")?,
    )?;
    let family_error = match responses.call("hello").with_options(foreign_family) {
        Ok(_) => panic!("embedding options must not target a language client"),
        Err(error) => error,
    };
    assert!(matches!(
        family_error,
        ProviderOptionError::TargetMismatch { .. }
    ));

    Ok(())
}

#[cfg(feature = "openai-responses-websocket")]
#[test]
fn responses_model_exposes_the_provider_owned_websocket_configuration()
-> Result<(), Box<dyn std::error::Error>> {
    use siumai::providers::openai::experimental::responses_websocket::OpenAiResponsesWebSocketConfig;

    let client = Siumai::builder()
        .openai()
        .api_key(CREDENTIAL_CANARY)
        .build()?
        .language(GPT_5_6)?;
    let websocket: OpenAiResponsesWebSocketConfig = client.model().websocket()?;

    assert_eq!(websocket.model().descriptor(), client.descriptor());
    assert_eq!(
        websocket.model().descriptor().instance_id(),
        client.descriptor().instance_id()
    );
    assert!(!format!("{websocket:?}").contains(CREDENTIAL_CANARY));

    Ok(())
}

#[test]
fn explicit_chat_mode_and_future_model_ids_remain_open() -> Result<(), Box<dyn std::error::Error>> {
    let hub = Siumai::builder().openai().api_key("test-api-key").build()?;
    let responses = hub.language("future-openai-model")?;
    let chat = hub.chat_completions("future-openai-model")?;

    accepts_openai_responses_model(responses.model());
    accepts_openai_chat_model(chat.model());
    assert_eq!(
        responses.descriptor().model().as_str(),
        "future-openai-model"
    );
    assert_eq!(chat.descriptor().model().as_str(), "future-openai-model");
    assert_ne!(
        responses.descriptor().api_mode(),
        chat.descriptor().api_mode()
    );

    Ok(())
}

#[test]
fn real_builder_configuration_and_secret_diagnostics_are_preserved()
-> Result<(), Box<dyn std::error::Error>> {
    let required = Siumai::builder().openai();
    assert!(!format!("{required:?}").contains(CREDENTIAL_CANARY));

    let stage = Siumai::builder().openai().api_key(CREDENTIAL_CANARY);
    assert!(!format!("{stage:?}").contains(CREDENTIAL_CANARY));
    let hub = stage.build()?;
    let client = hub.language("future-openai-model")?;
    assert!(!format!("{hub:?}").contains(CREDENTIAL_CANARY));
    assert!(!format!("{client:?}").contains(CREDENTIAL_CANARY));

    let error = Siumai::builder()
        .openai()
        .api_key(CREDENTIAL_CANARY)
        .configure_provider(|builder: OpenAiProviderBuilder| {
            builder.with_project("project-without-replay-scope")
        })
        .build()
        .expect_err("the real OpenAI builder must validate project replay scope");
    assert!(matches!(
        error,
        OpenAiConfigError::AccountScopeRequiresReplayCallerScope
    ));
    assert!(!error_diagnostics(&error).contains(CREDENTIAL_CANARY));

    Ok(())
}
