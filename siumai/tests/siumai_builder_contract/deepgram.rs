use siumai::providers::deepgram::{
    DeepgramCredential, DeepgramProvider, DeepgramProviderBuilder, DeepgramSpeechModel,
    DeepgramTranscriptionModel,
};
use siumai::transport::EndpointConfig;
use siumai::{Model, ModelFamily, Siumai};

use super::error_diagnostics;

const CREDENTIAL_CANARY: &str = "deepgram-facade-credential-canary";

#[test]
fn one_hub_binds_speech_and_transcription_models() -> Result<(), Box<dyn std::error::Error>> {
    let custom_endpoint = EndpointConfig::public_custom("https://deepgram.example.test")?;
    let hub = Siumai::builder()
        .deepgram()
        .api_key("test-api-key")
        .configure_provider(|builder: DeepgramProviderBuilder| {
            builder.with_endpoint(custom_endpoint)
        })
        .build()?;
    let _credential_stage = Siumai::builder()
        .deepgram()
        .credential(DeepgramCredential::api_key("test-api-key"));

    let speech = hub.speech("future-deepgram-speech-model")?;
    let transcription = hub.transcription("future-deepgram-transcription-model")?;
    let _: &DeepgramProvider = hub.provider();
    let _: &DeepgramSpeechModel = speech.model();
    let _: &DeepgramTranscriptionModel = transcription.model();
    assert!(std::ptr::eq(hub.provider(), speech.provider()));
    assert!(std::ptr::eq(hub.provider(), transcription.provider()));
    assert_eq!(speech.family(), ModelFamily::Speech);
    assert_eq!(transcription.family(), ModelFamily::Transcription);
    assert_eq!(speech.descriptor().api_mode(), Some("tts"));
    assert_eq!(transcription.descriptor().api_mode(), Some("prerecorded"));
    assert!(
        hub.provider()
            .profile()
            .provider_profile()
            .verified_claims()
            .is_none()
    );
    assert_eq!(
        speech.descriptor().instance_id(),
        transcription.descriptor().instance_id()
    );
    assert_eq!(
        speech.descriptor().model().as_str(),
        "future-deepgram-speech-model"
    );
    assert_eq!(
        transcription.descriptor().model().as_str(),
        "future-deepgram-transcription-model"
    );

    Ok(())
}

#[test]
fn stages_hubs_clients_and_error_chains_redact_deepgram_credentials()
-> Result<(), Box<dyn std::error::Error>> {
    let stage = Siumai::builder().deepgram().api_key(CREDENTIAL_CANARY);
    assert!(!format!("{stage:?}").contains(CREDENTIAL_CANARY));
    let hub = stage.build()?;
    let client = hub.transcription("future-deepgram-canary-model")?;
    assert!(!format!("{hub:?}").contains(CREDENTIAL_CANARY));
    assert!(!format!("{client:?}").contains(CREDENTIAL_CANARY));

    let error = Siumai::builder()
        .deepgram()
        .api_key(format!("{CREDENTIAL_CANARY}\n"))
        .build()
        .expect_err("the provider-owned credential validator must reject control characters");
    assert!(!error_diagnostics(&error).contains(CREDENTIAL_CANARY));

    Ok(())
}
