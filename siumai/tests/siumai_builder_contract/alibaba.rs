use siumai::providers::alibaba::{
    AlibabaCredential, AlibabaEmbeddingModel, AlibabaLanguageModel, AlibabaProvider,
    AlibabaProviderBuilder,
};
use siumai::{Model, ModelFamily, Siumai};

use super::error_diagnostics;

const CREDENTIAL_CANARY: &str = "alibaba-facade-credential-canary";

fn accepts_provider(provider: &AlibabaProvider) {
    let _ = provider;
}

fn accepts_language_model(model: &AlibabaLanguageModel) {
    let _ = model;
}

fn accepts_embedding_model(model: &AlibabaEmbeddingModel) {
    let _ = model;
}

#[test]
fn required_configuration_builds_one_multi_family_hub() -> Result<(), Box<dyn std::error::Error>> {
    let hub = Siumai::builder()
        .alibaba()
        .credential(AlibabaCredential::api_key("test-api-key"))
        .configure_provider(|builder: AlibabaProviderBuilder| {
            builder
                .with_legacy_singapore_language()
                .with_legacy_singapore_embedding()
        })
        .build()?;

    let language = hub.language("future-alibaba-language-model")?;
    let embedding = hub.embedding("future-alibaba-embedding-model")?;
    accepts_provider(hub.provider());
    accepts_language_model(language.model());
    accepts_embedding_model(embedding.model());
    assert!(std::ptr::eq(hub.provider(), language.provider()));
    assert!(std::ptr::eq(hub.provider(), embedding.provider()));
    assert_eq!(language.family(), ModelFamily::Language);
    assert_eq!(embedding.family(), ModelFamily::Embedding);
    assert_eq!(language.descriptor().api_mode(), Some("responses"));
    assert_eq!(embedding.descriptor().api_mode(), Some("text-embedding"));
    assert_eq!(
        language.descriptor().instance_id(),
        embedding.descriptor().instance_id()
    );
    assert_eq!(
        language.descriptor().model().as_str(),
        "future-alibaba-language-model"
    );
    assert_eq!(
        embedding.descriptor().model().as_str(),
        "future-alibaba-embedding-model"
    );

    Ok(())
}

#[test]
fn stages_and_provider_owned_errors_redact_alibaba_credentials() {
    let configuration = Siumai::builder().alibaba().api_key(CREDENTIAL_CANARY);
    assert!(!format!("{configuration:?}").contains(CREDENTIAL_CANARY));

    let hub = configuration
        .configure_provider(|builder| builder.with_legacy_singapore_language())
        .build()
        .expect("the provider-owned legacy endpoint is valid");
    let client = hub
        .language("future-alibaba-canary-model")
        .expect("future model IDs use baseline construction");
    assert!(!format!("{hub:?}").contains(CREDENTIAL_CANARY));
    assert!(!format!("{client:?}").contains(CREDENTIAL_CANARY));

    let error = Siumai::builder()
        .alibaba()
        .api_key(format!("{CREDENTIAL_CANARY}\n"))
        .configure_provider(|builder| builder.with_legacy_singapore_language())
        .build()
        .expect_err("the provider-owned credential validator must reject control characters");
    assert!(!error_diagnostics(&error).contains(CREDENTIAL_CANARY));
}
