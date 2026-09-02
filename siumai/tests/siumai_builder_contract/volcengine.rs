use siumai::providers::volcengine::{
    ArkImageModel, VolcengineCredential, VolcengineLanguageModel, VolcengineProvider,
    VolcengineProviderBuilder,
};
use siumai::{Model, ModelFamily, Siumai};

use super::error_diagnostics;

const CREDENTIAL_CANARY: &str = "volcengine-facade-credential-canary";

#[test]
fn one_hub_binds_language_and_image_models() -> Result<(), Box<dyn std::error::Error>> {
    let hub = Siumai::builder()
        .volcengine()
        .api_key("test-api-key")
        .build()?;
    let _credential_stage = Siumai::builder()
        .volcengine()
        .credential(VolcengineCredential::api_key("test-api-key"));

    let language = hub.language("future-volcengine-language-model")?;
    let image = hub.image("future-volcengine-image-model")?;
    let _: &VolcengineProvider = hub.provider();
    let _: &VolcengineLanguageModel = language.model();
    let _: &ArkImageModel = image.model();
    assert!(std::ptr::eq(hub.provider(), language.provider()));
    assert!(std::ptr::eq(hub.provider(), image.provider()));
    assert_eq!(language.family(), ModelFamily::Language);
    assert_eq!(image.family(), ModelFamily::Image);
    assert_eq!(language.descriptor().api_mode(), Some("responses"));
    assert_eq!(image.descriptor().api_mode(), Some("images-generations"));
    assert_eq!(
        language.descriptor().instance_id(),
        image.descriptor().instance_id()
    );
    assert_eq!(
        language.descriptor().model().as_str(),
        "future-volcengine-language-model"
    );
    assert_eq!(
        image.descriptor().model().as_str(),
        "future-volcengine-image-model"
    );

    Ok(())
}

#[test]
fn configuration_and_credentials_remain_observable_and_redacted()
-> Result<(), Box<dyn std::error::Error>> {
    let stage = Siumai::builder().volcengine().api_key(CREDENTIAL_CANARY);
    assert!(!format!("{stage:?}").contains(CREDENTIAL_CANARY));
    let hub = stage.build()?;
    let client = hub.language("future-volcengine-canary-model")?;
    assert!(!format!("{hub:?}").contains(CREDENTIAL_CANARY));
    assert!(!format!("{client:?}").contains(CREDENTIAL_CANARY));

    let credential_error = Siumai::builder()
        .volcengine()
        .api_key(format!("{CREDENTIAL_CANARY}\n"))
        .build()
        .expect_err("the provider-owned credential validator must reject control characters");
    assert!(!error_diagnostics(&credential_error).contains(CREDENTIAL_CANARY));

    let configuration_error = Siumai::builder()
        .volcengine()
        .api_key("test-api-key")
        .configure_provider(|builder: VolcengineProviderBuilder| {
            builder.with_base_url("not a valid URL")
        })
        .build()
        .expect_err("the configured endpoint must reach the provider builder");
    assert!(error_diagnostics(&configuration_error).contains("endpoint"));

    Ok(())
}
