use siumai::providers::moonshotai::{
    MoonshotCredential, MoonshotLanguageModel, MoonshotProvider, MoonshotProviderBuilder,
};
use siumai::{Model, ModelFamily, Siumai};

use super::error_diagnostics;

const CREDENTIAL_CANARY: &str = "moonshot-facade-credential-canary";

#[test]
fn api_key_and_credential_paths_bind_future_language_models()
-> Result<(), Box<dyn std::error::Error>> {
    let api_key_hub = Siumai::builder()
        .moonshot()
        .api_key("test-api-key")
        .build()?;
    let credential_hub = Siumai::builder()
        .moonshot()
        .credential(MoonshotCredential::api_key("test-api-key"))
        .build()?;

    let client = api_key_hub.language("future-moonshot-model")?;
    let _: &MoonshotProvider = api_key_hub.provider();
    let _: &MoonshotLanguageModel = client.model();
    assert!(std::ptr::eq(api_key_hub.provider(), client.provider()));
    assert_eq!(client.family(), ModelFamily::Language);
    assert_eq!(client.descriptor().api_mode(), Some("chat-completions"));
    assert_eq!(
        client.descriptor().model().as_str(),
        "future-moonshot-model"
    );
    assert_eq!(
        credential_hub
            .language("future-moonshot-credential-model")?
            .descriptor()
            .model()
            .as_str(),
        "future-moonshot-credential-model"
    );

    Ok(())
}

#[test]
fn configuration_and_credentials_remain_observable_and_redacted()
-> Result<(), Box<dyn std::error::Error>> {
    let stage = Siumai::builder().moonshot().api_key(CREDENTIAL_CANARY);
    assert!(!format!("{stage:?}").contains(CREDENTIAL_CANARY));
    let hub = stage.build()?;
    let client = hub.language("future-moonshot-canary-model")?;
    assert!(!format!("{hub:?}").contains(CREDENTIAL_CANARY));
    assert!(!format!("{client:?}").contains(CREDENTIAL_CANARY));

    let credential_error = Siumai::builder()
        .moonshot()
        .api_key(format!("{CREDENTIAL_CANARY}\n"))
        .build()
        .expect_err("the provider-owned credential validator must reject control characters");
    assert!(!error_diagnostics(&credential_error).contains(CREDENTIAL_CANARY));

    let configuration_error = Siumai::builder()
        .moonshot()
        .api_key("test-api-key")
        .configure_provider(|builder: MoonshotProviderBuilder| {
            builder.with_base_url("not a valid URL")
        })
        .build()
        .expect_err("the configured endpoint must reach the provider builder");
    assert!(error_diagnostics(&configuration_error).contains("endpoint"));

    Ok(())
}
