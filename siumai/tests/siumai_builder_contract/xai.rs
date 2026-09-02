use siumai::providers::xai::models;
use siumai::providers::xai::{
    XaiCredential, XaiImageModel, XaiLanguageModel, XaiProvider, XaiProviderBuilder,
    XaiSpeechModel, XaiTranscriptionModel,
};
use siumai::{Model, ModelFamily, Siumai};

use super::error_diagnostics;

const CREDENTIAL_CANARY: &str = "xai-facade-credential-canary";

#[test]
fn one_hub_binds_all_portable_xai_families() -> Result<(), Box<dyn std::error::Error>> {
    let hub = Siumai::builder().xai().api_key("test-api-key").build()?;
    let _credential_stage = Siumai::builder()
        .xai()
        .credential(XaiCredential::api_key("test-api-key"));

    let language = hub.language("future-xai-language-model")?;
    let image = hub.image("future-xai-image-model")?;
    let speech = hub.speech(models::speech::TTS)?;
    let transcription = hub.transcription(models::transcription::STT)?;
    let _: &XaiProvider = hub.provider();
    let _: &XaiLanguageModel = language.model();
    let _: &XaiImageModel = image.model();
    let _: &XaiSpeechModel = speech.model();
    let _: &XaiTranscriptionModel = transcription.model();
    assert!(std::ptr::eq(hub.provider(), language.provider()));
    assert!(std::ptr::eq(hub.provider(), image.provider()));
    assert!(std::ptr::eq(hub.provider(), speech.provider()));
    assert!(std::ptr::eq(hub.provider(), transcription.provider()));
    assert_eq!(language.family(), ModelFamily::Language);
    assert_eq!(image.family(), ModelFamily::Image);
    assert_eq!(speech.family(), ModelFamily::Speech);
    assert_eq!(transcription.family(), ModelFamily::Transcription);
    assert_eq!(language.descriptor().api_mode(), Some("responses"));
    assert_eq!(image.descriptor().api_mode(), Some("image-generations"));
    assert_eq!(speech.descriptor().api_mode(), Some("tts"));
    assert_eq!(transcription.descriptor().api_mode(), Some("stt"));
    assert_eq!(
        language.descriptor().instance_id(),
        image.descriptor().instance_id()
    );
    assert_eq!(
        language.descriptor().instance_id(),
        speech.descriptor().instance_id()
    );
    assert_eq!(
        language.descriptor().instance_id(),
        transcription.descriptor().instance_id()
    );
    assert_eq!(
        language.descriptor().model().as_str(),
        "future-xai-language-model"
    );
    assert_eq!(
        image.descriptor().model().as_str(),
        "future-xai-image-model"
    );

    Ok(())
}

#[test]
fn configuration_and_credentials_remain_observable_and_redacted()
-> Result<(), Box<dyn std::error::Error>> {
    let stage = Siumai::builder().xai().api_key(CREDENTIAL_CANARY);
    assert!(!format!("{stage:?}").contains(CREDENTIAL_CANARY));
    let hub = stage.build()?;
    let client = hub.language("future-xai-canary-model")?;
    assert!(!format!("{hub:?}").contains(CREDENTIAL_CANARY));
    assert!(!format!("{client:?}").contains(CREDENTIAL_CANARY));

    let credential_error = Siumai::builder()
        .xai()
        .api_key(format!("{CREDENTIAL_CANARY}\n"))
        .build()
        .expect_err("the provider-owned credential validator must reject control characters");
    assert!(!error_diagnostics(&credential_error).contains(CREDENTIAL_CANARY));

    let configuration_error = Siumai::builder()
        .xai()
        .api_key("test-api-key")
        .configure_provider(|builder: XaiProviderBuilder| builder.with_base_url("not a valid URL"))
        .build()
        .expect_err("the configured endpoint must reach the provider builder");
    assert!(error_diagnostics(&configuration_error).contains("endpoint"));

    Ok(())
}
