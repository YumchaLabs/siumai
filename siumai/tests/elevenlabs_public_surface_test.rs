#![cfg(feature = "elevenlabs")]

use siumai::prelude::unified::{SttRequest, TtsRequest};
use siumai::provider_ext::elevenlabs::{
    ApplyTextNormalization, ElevenLabsClient, ElevenLabsConfig, ElevenLabsSpeechModel,
    ElevenLabsSpeechModelOptions, ElevenLabsSttRequestExt, ElevenLabsTranscriptionFileFormat,
    ElevenLabsTranscriptionModel, ElevenLabsTranscriptionModelOptions,
    ElevenLabsTranscriptionTimestampsGranularity, ElevenLabsTtsRequestExt, ElevenLabsVoiceSettings,
    VERSION, create_elevenlabs, elevenlabs, model_sets, speech, transcription,
};

#[test]
fn elevenlabs_provider_ext_exports_package_surface() {
    let config = ElevenLabsConfig::new("test-key")
        .with_base_url("https://api.elevenlabs.test")
        .with_speech_model(speech::ELEVEN_V3)
        .with_transcription_model(transcription::SCRIBE_V1)
        .with_default_voice("voice-123");
    let client = ElevenLabsClient::from_config(config).expect("elevenlabs client config");

    let speech_model: ElevenLabsSpeechModel = client.speech_model(speech::ELEVEN_V3);
    let transcription_model: ElevenLabsTranscriptionModel =
        client.transcription_model(transcription::SCRIBE_V1);

    assert_eq!(ElevenLabsConfig::API_KEY_ENV, "ELEVENLABS_API_KEY");
    assert_eq!(
        ElevenLabsConfig::DEFAULT_BASE_URL,
        "https://api.elevenlabs.io"
    );
    assert_eq!(client.default_voice(), "voice-123");
    assert_eq!(model_sets::DEFAULT_SPEECH, speech::ELEVEN_MULTILINGUAL_V2);
    assert_eq!(model_sets::DEFAULT_TRANSCRIPTION, transcription::SCRIBE_V1);
    assert_eq!(model_sets::DEFAULT_VOICE, "21m00Tcm4TlvDq8ikWAM");
    assert_eq!(
        siumai::models::elevenlabs::DEFAULT_SPEECH,
        speech::ELEVEN_MULTILINGUAL_V2
    );
    assert_eq!(
        siumai::constants::elevenlabs::DEFAULT_TRANSCRIPTION,
        transcription::SCRIBE_V1
    );
    assert!(!VERSION.is_empty());
    drop(speech_model);
    drop(transcription_model);

    let tts_request = TtsRequest::new("hello".to_string()).with_elevenlabs_tts_options(
        ElevenLabsSpeechModelOptions::new()
            .with_language_code("en")
            .with_voice_settings(ElevenLabsVoiceSettings::new().with_stability(0.5))
            .with_apply_text_normalization(ApplyTextNormalization::Auto)
            .with_enable_logging(false),
    );
    assert_eq!(
        tts_request
            .provider_options_map
            .get("elevenlabs")
            .and_then(|value| value.get("applyTextNormalization")),
        Some(&serde_json::json!("auto"))
    );
    assert_eq!(
        tts_request
            .provider_options_map
            .get("elevenlabs")
            .and_then(|value| value.get("enableLogging")),
        Some(&serde_json::json!(false))
    );

    let stt_request = SttRequest::from_audio(vec![1, 2, 3], "audio/wav")
        .with_elevenlabs_stt_options(
            ElevenLabsTranscriptionModelOptions::new()
                .with_language_code("en")
                .with_diarize(true)
                .with_timestamps_granularity(ElevenLabsTranscriptionTimestampsGranularity::Word)
                .with_file_format(ElevenLabsTranscriptionFileFormat::PcmS16le16),
        );
    assert_eq!(
        stt_request
            .provider_options_map
            .get("elevenlabs")
            .and_then(|value| value.get("timestampsGranularity")),
        Some(&serde_json::json!("word"))
    );
    assert_eq!(
        stt_request
            .provider_options_map
            .get("elevenlabs")
            .and_then(|value| value.get("fileFormat")),
        Some(&serde_json::json!("pcm_s16le_16"))
    );
}

#[test]
fn elevenlabs_provider_alias_and_compat_builder_are_available() {
    let client = tokio_test_build(
        create_elevenlabs()
            .api_key("test-key")
            .model(speech::ELEVEN_V3),
    );
    assert_eq!(client.metadata().provider_id, "elevenlabs");
    assert_eq!(
        client.metadata().provider_type,
        siumai::prelude::unified::ProviderType::ElevenLabs
    );

    let alias_client = tokio_test_build(
        siumai::providers::elevenlabs::elevenlabs()
            .api_key("test-key")
            .model(transcription::SCRIBE_V1),
    );
    assert_eq!(alias_client.metadata().provider_id, "elevenlabs");

    let compat_client = tokio_test_build(
        siumai::compat::Provider::elevenlabs()
            .api_key("test-key")
            .model(transcription::SCRIBE_V1),
    );
    assert_eq!(compat_client.metadata().provider_id, "elevenlabs");

    let _builder = elevenlabs();
}

fn tokio_test_build(builder: siumai::compat::SiumaiBuilder) -> siumai::compat::Siumai {
    tokio::runtime::Runtime::new()
        .expect("tokio runtime")
        .block_on(async { builder.build().await.expect("builder should build") })
}
