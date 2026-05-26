#![cfg(feature = "elevenlabs")]

use siumai::provider_ext::elevenlabs::resources::{
    ElevenLabsAddPvcVoiceSamplesRequest, ElevenLabsCreateIvcVoiceRequest,
    ElevenLabsCreateIvcVoiceResponse, ElevenLabsCreatePronunciationDictionaryFromFileRequest,
    ElevenLabsCreatePronunciationDictionaryFromRulesRequest, ElevenLabsCreatePvcVoiceRequest,
    ElevenLabsPronunciationDictionaries, ElevenLabsPronunciationDictionary,
    ElevenLabsPronunciationDictionaryCreateResponse,
    ElevenLabsPronunciationDictionaryDownloadResponse, ElevenLabsPronunciationDictionaryListQuery,
    ElevenLabsPronunciationDictionaryListResponse, ElevenLabsPronunciationDictionaryRule,
    ElevenLabsPronunciationDictionaryRuleRequest,
    ElevenLabsPronunciationDictionaryRulesMutationRequest,
    ElevenLabsPronunciationDictionaryRulesMutationResponse, ElevenLabsPvcSpeakerAudioResponse,
    ElevenLabsPvcSpeakerResponse, ElevenLabsPvcSpeakerSeparationResponse, ElevenLabsPvcUtterance,
    ElevenLabsPvcVoiceResponse, ElevenLabsPvcVoiceSample, ElevenLabsPvcVoiceSampleAudioQuery,
    ElevenLabsPvcVoiceSampleAudioResponse, ElevenLabsPvcVoiceSampleWaveformResponse,
    ElevenLabsRemovePronunciationDictionaryRulesRequest, ElevenLabsTrainPvcVoiceRequest,
    ElevenLabsUpdatePronunciationDictionaryRequest, ElevenLabsUpdatePvcVoiceRequest,
    ElevenLabsUpdatePvcVoiceSampleRequest, ElevenLabsUpdateVoiceSettingsRequest,
    ElevenLabsVerifiedLanguage, ElevenLabsVoice, ElevenLabsVoiceListQuery,
    ElevenLabsVoiceListResponse, ElevenLabsVoiceSampleFile, ElevenLabsVoiceSettingsResponse,
    ElevenLabsVoiceSettingsUpdateResponse, ElevenLabsVoiceStatusResponse, ElevenLabsVoices,
};
use siumai::provider_ext::elevenlabs::{ElevenLabsClient, ElevenLabsConfig};

fn assert_type<T>() {}

#[test]
fn elevenlabs_voice_resources_are_exported_from_facade_modules() {
    assert_type::<ElevenLabsVoices>();
    assert_type::<ElevenLabsVoiceListQuery>();
    assert_type::<ElevenLabsVoiceListResponse>();
    assert_type::<ElevenLabsVoice>();
    assert_type::<ElevenLabsVoiceSettingsResponse>();
    assert_type::<ElevenLabsUpdateVoiceSettingsRequest>();
    assert_type::<ElevenLabsVoiceSettingsUpdateResponse>();
    assert_type::<ElevenLabsVoiceStatusResponse>();
    assert_type::<ElevenLabsCreateIvcVoiceRequest>();
    assert_type::<ElevenLabsCreateIvcVoiceResponse>();
    assert_type::<ElevenLabsVoiceSampleFile>();
    assert_type::<ElevenLabsCreatePvcVoiceRequest>();
    assert_type::<ElevenLabsUpdatePvcVoiceRequest>();
    assert_type::<ElevenLabsTrainPvcVoiceRequest>();
    assert_type::<ElevenLabsPvcVoiceResponse>();
    assert_type::<ElevenLabsAddPvcVoiceSamplesRequest>();
    assert_type::<ElevenLabsUpdatePvcVoiceSampleRequest>();
    assert_type::<ElevenLabsPvcVoiceSampleAudioQuery>();
    assert_type::<ElevenLabsPvcVoiceSample>();
    assert_type::<ElevenLabsPvcSpeakerSeparationResponse>();
    assert_type::<ElevenLabsPvcSpeakerResponse>();
    assert_type::<ElevenLabsPvcUtterance>();
    assert_type::<ElevenLabsPvcVoiceSampleAudioResponse>();
    assert_type::<ElevenLabsPvcVoiceSampleWaveformResponse>();
    assert_type::<ElevenLabsPvcSpeakerAudioResponse>();
    assert_type::<ElevenLabsVerifiedLanguage>();
    assert_type::<ElevenLabsPronunciationDictionaries>();
    assert_type::<ElevenLabsPronunciationDictionaryListQuery>();
    assert_type::<ElevenLabsPronunciationDictionaryListResponse>();
    assert_type::<ElevenLabsPronunciationDictionary>();
    assert_type::<ElevenLabsPronunciationDictionaryRule>();
    assert_type::<ElevenLabsCreatePronunciationDictionaryFromFileRequest>();
    assert_type::<ElevenLabsCreatePronunciationDictionaryFromRulesRequest>();
    assert_type::<ElevenLabsPronunciationDictionaryRuleRequest>();
    assert_type::<ElevenLabsPronunciationDictionaryCreateResponse>();
    assert_type::<ElevenLabsPronunciationDictionaryDownloadResponse>();
    assert_type::<ElevenLabsUpdatePronunciationDictionaryRequest>();
    assert_type::<ElevenLabsPronunciationDictionaryRulesMutationRequest>();
    assert_type::<ElevenLabsRemovePronunciationDictionaryRulesRequest>();
    assert_type::<ElevenLabsPronunciationDictionaryRulesMutationResponse>();
    assert_type::<siumai::providers::elevenlabs::resources::ElevenLabsVoices>();
    assert_type::<siumai::providers::elevenlabs::resources::ElevenLabsVoiceListQuery>();
    assert_type::<siumai::providers::elevenlabs::resources::ElevenLabsUpdateVoiceSettingsRequest>();
    assert_type::<siumai::providers::elevenlabs::resources::ElevenLabsVoiceStatusResponse>();
    assert_type::<siumai::providers::elevenlabs::resources::ElevenLabsCreateIvcVoiceRequest>();
    assert_type::<siumai::providers::elevenlabs::resources::ElevenLabsCreatePvcVoiceRequest>();
    assert_type::<siumai::providers::elevenlabs::resources::ElevenLabsUpdatePvcVoiceRequest>();
    assert_type::<siumai::providers::elevenlabs::resources::ElevenLabsTrainPvcVoiceRequest>();
    assert_type::<siumai::providers::elevenlabs::resources::ElevenLabsAddPvcVoiceSamplesRequest>();
    assert_type::<siumai::providers::elevenlabs::resources::ElevenLabsUpdatePvcVoiceSampleRequest>(
    );
    assert_type::<siumai::providers::elevenlabs::resources::ElevenLabsPvcVoiceSampleAudioResponse>(
    );
    assert_type::<siumai::providers::elevenlabs::resources::ElevenLabsPronunciationDictionaries>();
    assert_type::<
        siumai::providers::elevenlabs::resources::ElevenLabsCreatePronunciationDictionaryFromRulesRequest,
    >();
    assert_type::<
        siumai::providers::elevenlabs::resources::ElevenLabsCreatePronunciationDictionaryFromFileRequest,
    >();
    assert_type::<
        siumai::providers::elevenlabs::resources::ElevenLabsUpdatePronunciationDictionaryRequest,
    >();
    assert_type::<
        siumai::providers::elevenlabs::resources::ElevenLabsPronunciationDictionaryRulesMutationRequest,
    >();

    let config = ElevenLabsConfig::new("test-key").with_base_url("https://api.elevenlabs.test");
    let client = ElevenLabsClient::from_config(config).expect("elevenlabs client");
    let voices: ElevenLabsVoices = client.voices();
    let pronunciation_dictionaries: ElevenLabsPronunciationDictionaries =
        client.pronunciation_dictionaries();

    let query = ElevenLabsVoiceListQuery::new()
        .with_page_size(10)
        .with_search("rachel")
        .with_include_total_count(true)
        .with_voice_id("voice-1");
    let dictionary_query = ElevenLabsPronunciationDictionaryListQuery::new()
        .with_page_size(10)
        .with_sort("creation_time_unix");
    let update_voice_settings_request = ElevenLabsUpdateVoiceSettingsRequest::new()
        .with_stability(0.45)
        .with_similarity_boost(0.8)
        .with_speed(1.05);
    let create_ivc_voice_request = ElevenLabsCreateIvcVoiceRequest::new(
        "Narrator Clone",
        [ElevenLabsVoiceSampleFile::new(b"audio".to_vec()).with_filename("sample.wav")],
    )
    .with_remove_background_noise(true)
    .with_label("accent", "american");
    let create_pvc_voice_request = ElevenLabsCreatePvcVoiceRequest::new("PVC Voice", "en")
        .with_description("Narration PVC voice")
        .with_label("accent", "american");
    let update_pvc_voice_request = ElevenLabsUpdatePvcVoiceRequest::new()
        .with_name("Updated PVC Voice")
        .with_language("en-US");
    let train_pvc_voice_request =
        ElevenLabsTrainPvcVoiceRequest::new().with_model_id("eleven_multilingual_v2");
    let add_pvc_samples_request =
        ElevenLabsAddPvcVoiceSamplesRequest::new([ElevenLabsVoiceSampleFile::new(
            b"audio".to_vec(),
        )
        .with_filename("sample.wav")])
        .with_remove_background_noise(true);
    let update_pvc_sample_request = ElevenLabsUpdatePvcVoiceSampleRequest::new()
        .with_remove_background_noise(true)
        .with_selected_speaker_id("speaker-1")
        .with_trim_start_time(100)
        .with_trim_end_time(1_000)
        .with_file_name("sample.wav");
    let pvc_sample_audio_query =
        ElevenLabsPvcVoiceSampleAudioQuery::new().with_remove_background_noise(true);
    let create_dictionary_request = ElevenLabsCreatePronunciationDictionaryFromRulesRequest::new(
        "Product terms",
        vec![ElevenLabsPronunciationDictionaryRuleRequest::alias(
            "Siumai", "sue my",
        )],
    );
    let create_dictionary_from_file_request =
        ElevenLabsCreatePronunciationDictionaryFromFileRequest::new(
            "File terms",
            b"<lexicon />".to_vec(),
        )
        .with_filename("terms.pls")
        .with_mime_type("application/pls+xml");
    let update_dictionary_request = ElevenLabsUpdatePronunciationDictionaryRequest::new()
        .with_name("Updated terms")
        .with_archived(false);
    let add_dictionary_rules_request =
        ElevenLabsPronunciationDictionaryRulesMutationRequest::new(vec![
            ElevenLabsPronunciationDictionaryRuleRequest::alias("SQL", "sequel"),
        ]);
    let remove_dictionary_rules_request =
        ElevenLabsRemovePronunciationDictionaryRulesRequest::new(["SQL"]);

    drop(query);
    drop(dictionary_query);
    drop(create_dictionary_request);
    drop(create_dictionary_from_file_request);
    drop(update_dictionary_request);
    drop(add_dictionary_rules_request);
    drop(remove_dictionary_rules_request);
    drop(voices);
    drop(update_voice_settings_request);
    drop(create_ivc_voice_request);
    drop(create_pvc_voice_request);
    drop(update_pvc_voice_request);
    drop(train_pvc_voice_request);
    drop(add_pvc_samples_request);
    drop(update_pvc_sample_request);
    drop(pvc_sample_audio_query);
    drop(pronunciation_dictionaries);
}
