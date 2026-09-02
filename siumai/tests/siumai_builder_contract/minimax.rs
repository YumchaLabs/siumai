use siumai::providers::minimax::{
    MinimaxCredential, MinimaxImageModel, MinimaxLanguageModel, MinimaxProvider,
    MinimaxProviderBuilder, MinimaxSpeechModel,
};
use siumai::{Model, ModelFamily, Siumai};

use super::error_diagnostics;

const CREDENTIAL_CANARY: &str = "minimax-facade-credential-canary";

#[test]
fn one_hub_binds_language_image_and_speech_models() -> Result<(), Box<dyn std::error::Error>> {
    let hub = Siumai::builder()
        .minimax()
        .api_key("test-api-key")
        .build()?;
    let _credential_stage = Siumai::builder()
        .minimax()
        .credential(MinimaxCredential::api_key("test-api-key"));

    let language = hub.language("future-minimax-language-model")?;
    let image = hub.image("future-minimax-image-model")?;
    let speech = hub.speech("future-minimax-speech-model")?;
    let _: &MinimaxProvider = hub.provider();
    let _: &MinimaxLanguageModel = language.model();
    let _: &MinimaxImageModel = image.model();
    let _: &MinimaxSpeechModel = speech.model();
    assert!(std::ptr::eq(hub.provider(), language.provider()));
    assert!(std::ptr::eq(hub.provider(), image.provider()));
    assert!(std::ptr::eq(hub.provider(), speech.provider()));
    assert_eq!(language.family(), ModelFamily::Language);
    assert_eq!(image.family(), ModelFamily::Image);
    assert_eq!(speech.family(), ModelFamily::Speech);
    assert_eq!(language.descriptor().api_mode(), Some("messages"));
    assert_eq!(image.descriptor().api_mode(), Some("image-generation"));
    assert_eq!(speech.descriptor().api_mode(), Some("speech-http"));
    assert_eq!(
        language.descriptor().instance_id(),
        image.descriptor().instance_id()
    );
    assert_eq!(
        language.descriptor().instance_id(),
        speech.descriptor().instance_id()
    );
    assert_eq!(
        language.descriptor().model().as_str(),
        "future-minimax-language-model"
    );
    assert_eq!(
        image.descriptor().model().as_str(),
        "future-minimax-image-model"
    );
    assert_eq!(
        speech.descriptor().model().as_str(),
        "future-minimax-speech-model"
    );

    Ok(())
}

#[test]
fn configuration_and_credentials_remain_observable_and_redacted()
-> Result<(), Box<dyn std::error::Error>> {
    let stage = Siumai::builder().minimax().api_key(CREDENTIAL_CANARY);
    assert!(!format!("{stage:?}").contains(CREDENTIAL_CANARY));
    let hub = stage.build()?;
    let client = hub.language("future-minimax-canary-model")?;
    assert!(!format!("{hub:?}").contains(CREDENTIAL_CANARY));
    assert!(!format!("{client:?}").contains(CREDENTIAL_CANARY));

    let credential_error = Siumai::builder()
        .minimax()
        .api_key(format!("{CREDENTIAL_CANARY}\n"))
        .build()
        .expect_err("the provider-owned credential validator must reject control characters");
    assert!(!error_diagnostics(&credential_error).contains(CREDENTIAL_CANARY));

    let configuration_error = Siumai::builder()
        .minimax()
        .api_key("test-api-key")
        .configure_provider(|builder: MinimaxProviderBuilder| {
            builder.with_messages_base_url("not a valid URL")
        })
        .build()
        .expect_err("the configured endpoint must reach the provider builder");
    assert!(error_diagnostics(&configuration_error).contains("endpoint"));

    Ok(())
}
