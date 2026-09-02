use super::{Siumai, accepts_language_model, error_diagnostics};
use siumai::providers::google::{
    GEMINI_GENERATE_CONTENT_API_MODE_ID, GeminiCredential, GeminiFiles, GeminiGenerateContentModel,
    GeminiLanguageModel, GeminiProvider, GeminiProviderBuilder, GeminiVeo,
};
use siumai::{CallOptions, LanguageRequest, Model, ModelFamily};

const CREDENTIAL_CANARY: &str = "gemini-facade-secret-canary";

fn accepts_gemini_provider(provider: &GeminiProvider) {
    let _ = provider;
}

fn accepts_interactions_model(model: &GeminiLanguageModel) {
    let _ = model;
}

fn accepts_generate_content_model(model: &GeminiGenerateContentModel) {
    let _ = model;
}

#[test]
fn canonical_interactions_and_explicit_generate_content_are_distinct()
-> Result<(), Box<dyn std::error::Error>> {
    let hub = Siumai::builder().gemini().api_key("test-api-key").build()?;
    let interactions = hub.language("future-gemini-model")?;
    let generate_content = hub.generate_content("future-gemini-model")?;

    accepts_language_model(&interactions);
    accepts_language_model(&generate_content);
    accepts_gemini_provider(hub.provider());
    accepts_interactions_model(interactions.model());
    accepts_generate_content_model(generate_content.model());
    assert_eq!(
        generate_content.descriptor().api_mode(),
        Some(GEMINI_GENERATE_CONTENT_API_MODE_ID)
    );
    assert_ne!(
        interactions.descriptor().api_mode(),
        generate_content.descriptor().api_mode()
    );

    Ok(())
}

#[test]
fn credential_path_real_builder_configuration_and_redaction_are_preserved()
-> Result<(), Box<dyn std::error::Error>> {
    let hub = Siumai::builder()
        .gemini()
        .credential(GeminiCredential::api_key(CREDENTIAL_CANARY))
        .build()?;
    let client = hub.language("future-gemini-model")?;
    assert!(!format!("{hub:?}").contains(CREDENTIAL_CANARY));
    assert!(!format!("{client:?}").contains(CREDENTIAL_CANARY));

    let stage = Siumai::builder().gemini().api_key(CREDENTIAL_CANARY);
    assert!(!format!("{stage:?}").contains(CREDENTIAL_CANARY));
    let error = stage
        .configure_provider(|builder: GeminiProviderBuilder| {
            builder.with_base_url("not a valid URL")
        })
        .build()
        .expect_err("the real Gemini builder must validate its endpoint");
    assert!(!error_diagnostics(&error).contains(CREDENTIAL_CANARY));

    Ok(())
}

#[test]
fn native_files_veo_and_interactions_methods_remain_typed_after_binding()
-> Result<(), Box<dyn std::error::Error>> {
    let hub = Siumai::builder()
        .gemini()
        .api_key(CREDENTIAL_CANARY)
        .build()?;
    let interactions = hub.language("future-gemini-model")?;
    let generate_content = hub.generate_content("future-gemini-model")?;
    let files: GeminiFiles = interactions.provider().files();
    let veo: GeminiVeo = interactions.provider().veo();

    assert!(std::ptr::eq(hub.provider(), interactions.provider()));
    assert!(std::ptr::eq(hub.provider(), generate_content.provider()));
    assert!(std::ptr::eq(
        Model::descriptor(&interactions),
        interactions.model().descriptor()
    ));
    assert_eq!(interactions.family(), ModelFamily::Language);
    assert_eq!(interactions.descriptor().api_mode(), Some("interactions"));
    assert_eq!(
        generate_content.descriptor().api_mode(),
        Some(GEMINI_GENERATE_CONTENT_API_MODE_ID)
    );
    assert_eq!(
        interactions.descriptor().instance_id(),
        generate_content.descriptor().instance_id()
    );
    assert!(interactions.route_id().is_none());
    assert!(!format!("{files:?}").contains(CREDENTIAL_CANARY));
    assert!(!format!("{veo:?}").contains(CREDENTIAL_CANARY));

    drop(
        interactions
            .model()
            .generate_native(LanguageRequest::new(Vec::new()), CallOptions::default()),
    );

    Ok(())
}
