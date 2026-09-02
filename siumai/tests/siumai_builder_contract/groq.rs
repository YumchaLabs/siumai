use siumai::providers::groq::{
    GroqCredential, GroqLanguageModel, GroqProvider, GroqProviderBuilder, GroqSpeechModel,
    GroqTranscriptionModel,
};
use siumai::{Model, ModelFamily, Siumai};

use super::error_diagnostics;

const CREDENTIAL_CANARY: &str = "groq-facade-credential-canary";

#[test]
fn one_hub_binds_language_speech_and_transcription_models() -> Result<(), Box<dyn std::error::Error>>
{
    let hub = Siumai::builder().groq().api_key("test-api-key").build()?;
    let _credential_stage = Siumai::builder()
        .groq()
        .credential(GroqCredential::api_key("test-api-key"));

    let language = hub.language("future-groq-language-model")?;
    let speech = hub.speech("future-groq-speech-model")?;
    let transcription = hub.transcription("future-groq-transcription-model")?;
    let _: &GroqProvider = hub.provider();
    let _: &GroqLanguageModel = language.model();
    let _: &GroqSpeechModel = speech.model();
    let _: &GroqTranscriptionModel = transcription.model();
    assert!(std::ptr::eq(hub.provider(), language.provider()));
    assert!(std::ptr::eq(hub.provider(), speech.provider()));
    assert!(std::ptr::eq(hub.provider(), transcription.provider()));
    assert_eq!(language.family(), ModelFamily::Language);
    assert_eq!(speech.family(), ModelFamily::Speech);
    assert_eq!(transcription.family(), ModelFamily::Transcription);
    assert_eq!(language.descriptor().api_mode(), Some("chat-completions"));
    assert_eq!(speech.descriptor().api_mode(), Some("audio-speech"));
    assert_eq!(
        transcription.descriptor().api_mode(),
        Some("audio-transcriptions")
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
        "future-groq-language-model"
    );
    assert_eq!(
        speech.descriptor().model().as_str(),
        "future-groq-speech-model"
    );
    assert_eq!(
        transcription.descriptor().model().as_str(),
        "future-groq-transcription-model"
    );

    Ok(())
}

#[test]
fn configuration_and_credentials_remain_observable_and_redacted()
-> Result<(), Box<dyn std::error::Error>> {
    let stage = Siumai::builder().groq().api_key(CREDENTIAL_CANARY);
    assert!(!format!("{stage:?}").contains(CREDENTIAL_CANARY));
    let hub = stage.build()?;
    let client = hub.language("future-groq-canary-model")?;
    assert!(!format!("{hub:?}").contains(CREDENTIAL_CANARY));
    assert!(!format!("{client:?}").contains(CREDENTIAL_CANARY));

    let credential_error = Siumai::builder()
        .groq()
        .api_key(format!("{CREDENTIAL_CANARY}\n"))
        .build()
        .expect_err("the provider-owned credential validator must reject control characters");
    assert!(!error_diagnostics(&credential_error).contains(CREDENTIAL_CANARY));

    let configuration_error = Siumai::builder()
        .groq()
        .api_key("test-api-key")
        .configure_provider(|builder: GroqProviderBuilder| builder.with_base_url("not a valid URL"))
        .build()
        .expect_err("the configured endpoint must reach the provider builder");
    assert!(error_diagnostics(&configuration_error).contains("endpoint"));

    Ok(())
}
