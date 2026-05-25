#![cfg(feature = "deepgram")]

use siumai::prelude::unified::{SttRequest, TtsRequest};
use siumai::provider_ext::deepgram::{
    DeepgramClient, DeepgramConfig, DeepgramSpeechModel, DeepgramSpeechModelOptions,
    DeepgramSttOptions, DeepgramSttRequestExt, DeepgramSummarizeOption, DeepgramTranscriptionModel,
    DeepgramTranscriptionModelOptions, DeepgramTtsRequestExt, VERSION, create_deepgram, deepgram,
    model_sets, speech, transcription,
};

#[test]
fn deepgram_provider_ext_exports_package_surface() {
    let config = DeepgramConfig::new("test-key")
        .with_base_url("https://api.deepgram.test")
        .with_speech_model(speech::AURA_2_HELENA_EN)
        .with_transcription_model(transcription::NOVA_3);
    let client = DeepgramClient::from_config(config).expect("deepgram client config");

    let speech_model: DeepgramSpeechModel = client.speech_model(speech::AURA_2_HELENA_EN);
    let transcription_model: DeepgramTranscriptionModel =
        client.transcription_model(transcription::NOVA_3);

    assert_eq!(DeepgramConfig::API_KEY_ENV, "DEEPGRAM_API_KEY");
    assert_eq!(DeepgramConfig::DEFAULT_BASE_URL, "https://api.deepgram.com");
    assert_eq!(model_sets::DEFAULT_SPEECH, speech::AURA_2_HELENA_EN);
    assert_eq!(model_sets::DEFAULT_TRANSCRIPTION, transcription::NOVA_3);
    assert_eq!(
        siumai::models::deepgram::DEFAULT_SPEECH,
        speech::AURA_2_HELENA_EN
    );
    assert_eq!(
        siumai::constants::deepgram::DEFAULT_TRANSCRIPTION,
        transcription::NOVA_3
    );
    assert!(!VERSION.is_empty());
    drop(speech_model);
    drop(transcription_model);

    let tts_request = TtsRequest::new("hello".to_string()).with_deepgram_tts_options(
        DeepgramSpeechModelOptions::new()
            .with_encoding("linear16")
            .with_sample_rate(24_000),
    );
    assert_eq!(
        tts_request
            .provider_options_map
            .get("deepgram")
            .and_then(|value| value.get("sampleRate")),
        Some(&serde_json::json!(24_000))
    );

    let stt_request = SttRequest::from_audio(vec![1, 2, 3], "audio/wav").with_deepgram_stt_options(
        DeepgramTranscriptionModelOptions::new()
            .with_language("en")
            .with_summarize(DeepgramSummarizeOption::Bool(true)),
    );
    assert_eq!(
        stt_request
            .provider_options_map
            .get("deepgram")
            .and_then(|value| value.get("summarize")),
        Some(&serde_json::json!(true))
    );

    let _alias_options: DeepgramSttOptions =
        DeepgramTranscriptionModelOptions::new().with_detect_language(true);
}

#[test]
fn deepgram_provider_alias_and_compat_builder_are_available() {
    let client = tokio_test_build(
        create_deepgram()
            .api_key("test-key")
            .model(transcription::NOVA_3),
    );
    assert_eq!(client.metadata().provider_id, "deepgram");
    assert_eq!(
        client.metadata().provider_type,
        siumai::prelude::unified::ProviderType::Deepgram
    );

    let alias_client = tokio_test_build(
        siumai::providers::deepgram::deepgram()
            .api_key("test-key")
            .model(speech::AURA_2_HELENA_EN),
    );
    assert_eq!(alias_client.metadata().provider_id, "deepgram");

    let compat_client = tokio_test_build(
        siumai::compat::Provider::deepgram()
            .api_key("test-key")
            .model(transcription::NOVA_3),
    );
    assert_eq!(compat_client.metadata().provider_id, "deepgram");

    let _builder = deepgram();
}

fn tokio_test_build(builder: siumai::compat::SiumaiBuilder) -> siumai::compat::Siumai {
    tokio::runtime::Runtime::new()
        .expect("tokio runtime")
        .block_on(async { builder.build().await.expect("builder should build") })
}
