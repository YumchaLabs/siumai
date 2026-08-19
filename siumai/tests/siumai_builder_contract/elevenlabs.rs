use siumai::providers::elevenlabs::{
    ElevenLabsCredential, ElevenLabsProfile, ElevenLabsProvider, ElevenLabsProviderBuilder,
    ElevenLabsSpeechModel, ElevenLabsTranscriptionModel,
};
use siumai::{Model, ModelFamily, Siumai, SpeechLimits, SpeechModel};

use super::error_diagnostics;

const CREDENTIAL_CANARY: &str = "elevenlabs-facade-credential-canary";

#[test]
fn profile_then_credential_binds_speech_and_transcription_models()
-> Result<(), Box<dyn std::error::Error>> {
    let configured_limits = SpeechLimits {
        max_text_bytes: Some(4096),
        max_text_chars: Some(2048),
    };
    let hub = Siumai::builder()
        .elevenlabs()
        .profile(ElevenLabsProfile::official()?)
        .credential(ElevenLabsCredential::api_key("test-api-key"))
        .configure_provider(|builder: ElevenLabsProviderBuilder| {
            builder.with_speech_limits(configured_limits)
        })
        .build()?;

    let speech = hub.speech("future-elevenlabs-speech-model")?;
    let transcription = hub.transcription("future-elevenlabs-transcription-model")?;
    let _: &ElevenLabsProvider = hub.provider();
    let _: &ElevenLabsSpeechModel = speech.model();
    let _: &ElevenLabsTranscriptionModel = transcription.model();
    assert!(std::ptr::eq(hub.provider(), speech.provider()));
    assert!(std::ptr::eq(hub.provider(), transcription.provider()));
    assert_eq!(speech.family(), ModelFamily::Speech);
    assert_eq!(transcription.family(), ModelFamily::Transcription);
    assert_eq!(speech.descriptor().api_mode(), Some("text-to-speech"));
    assert_eq!(speech.model().limits(), configured_limits);
    assert_eq!(
        transcription.descriptor().api_mode(),
        Some("batch-transcription")
    );
    assert_eq!(
        speech.descriptor().instance_id(),
        transcription.descriptor().instance_id()
    );
    assert_eq!(
        speech.descriptor().model().as_str(),
        "future-elevenlabs-speech-model"
    );
    assert_eq!(
        transcription.descriptor().model().as_str(),
        "future-elevenlabs-transcription-model"
    );

    Ok(())
}

#[test]
fn stages_hubs_clients_and_error_chains_redact_elevenlabs_credentials()
-> Result<(), Box<dyn std::error::Error>> {
    let stage = Siumai::builder()
        .elevenlabs()
        .profile(ElevenLabsProfile::official()?)
        .api_key(CREDENTIAL_CANARY);
    assert!(!format!("{stage:?}").contains(CREDENTIAL_CANARY));
    let hub = stage.build()?;
    let client = hub.speech("future-elevenlabs-canary-model")?;
    assert!(!format!("{hub:?}").contains(CREDENTIAL_CANARY));
    assert!(!format!("{client:?}").contains(CREDENTIAL_CANARY));

    let error = Siumai::builder()
        .elevenlabs()
        .profile(ElevenLabsProfile::official()?)
        .api_key(format!("{CREDENTIAL_CANARY}\n"))
        .build()
        .expect_err("the provider-owned credential validator must reject control characters");
    assert!(!error_diagnostics(&error).contains(CREDENTIAL_CANARY));

    Ok(())
}
