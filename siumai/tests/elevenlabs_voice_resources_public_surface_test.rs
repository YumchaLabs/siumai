#![cfg(feature = "elevenlabs")]

use siumai::provider_ext::elevenlabs::resources::{
    ElevenLabsCreatePronunciationDictionaryFromFileRequest,
    ElevenLabsCreatePronunciationDictionaryFromRulesRequest, ElevenLabsPronunciationDictionaries,
    ElevenLabsPronunciationDictionary, ElevenLabsPronunciationDictionaryCreateResponse,
    ElevenLabsPronunciationDictionaryListQuery, ElevenLabsPronunciationDictionaryListResponse,
    ElevenLabsPronunciationDictionaryRule, ElevenLabsPronunciationDictionaryRuleRequest,
    ElevenLabsPronunciationDictionaryRulesMutationRequest,
    ElevenLabsPronunciationDictionaryRulesMutationResponse,
    ElevenLabsRemovePronunciationDictionaryRulesRequest,
    ElevenLabsUpdatePronunciationDictionaryRequest, ElevenLabsVerifiedLanguage, ElevenLabsVoice,
    ElevenLabsVoiceListQuery, ElevenLabsVoiceListResponse, ElevenLabsVoiceSettingsResponse,
    ElevenLabsVoices,
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
    assert_type::<ElevenLabsUpdatePronunciationDictionaryRequest>();
    assert_type::<ElevenLabsPronunciationDictionaryRulesMutationRequest>();
    assert_type::<ElevenLabsRemovePronunciationDictionaryRulesRequest>();
    assert_type::<ElevenLabsPronunciationDictionaryRulesMutationResponse>();
    assert_type::<siumai::providers::elevenlabs::resources::ElevenLabsVoices>();
    assert_type::<siumai::providers::elevenlabs::resources::ElevenLabsVoiceListQuery>();
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
    drop(pronunciation_dictionaries);
}
