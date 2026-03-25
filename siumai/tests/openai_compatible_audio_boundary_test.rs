#![cfg(feature = "openai")]

use siumai::prelude::unified::LlmError;
use siumai::provider_ext::openai_compatible::{
    get_provider_config, list_provider_ids, provider_supports_capability,
};
use siumai::registry::helpers::create_registry_with_defaults;

fn sample_model_for(provider_id: &str) -> String {
    if let Some(model) = get_provider_config(provider_id).and_then(|config| config.default_model) {
        return model;
    }

    match provider_id {
        "openrouter" => "openai/gpt-4o".to_string(),
        other => panic!("no sample model available for compat provider {other}"),
    }
}

fn has_native_audio_registry_override(provider_id: &str) -> bool {
    matches!(
        provider_id,
        "openai"
            | "azure"
            | "anthropic"
            | "gemini"
            | "vertex"
            | "anthropic-vertex"
            | "groq"
            | "xai"
            | "minimaxi"
            | "deepseek"
            | "cohere"
            | "togetherai"
            | "bedrock"
            | "ollama"
    )
}

#[test]
fn compat_audio_capability_flag_matches_family_enrollment() {
    for provider_id in list_provider_ids() {
        let has_speech = provider_supports_capability(&provider_id, "speech");
        let has_transcription = provider_supports_capability(&provider_id, "transcription");
        let has_audio = provider_supports_capability(&provider_id, "audio");

        assert_eq!(
            has_audio,
            has_speech || has_transcription,
            "compat provider {provider_id} must keep `audio` aligned with speech/transcription enrollment"
        );
    }
}

#[test]
fn compat_registry_audio_handles_follow_declared_capability_split() {
    let registry = create_registry_with_defaults();

    for provider_id in list_provider_ids() {
        if has_native_audio_registry_override(&provider_id) {
            continue;
        }

        let model = sample_model_for(&provider_id);
        let full_model_id = format!("{provider_id}:{model}");

        if !provider_supports_capability(&provider_id, "speech") {
            match registry.speech_model(&full_model_id) {
                Ok(_) => panic!("{full_model_id} registry speech handle should be unsupported"),
                Err(LlmError::UnsupportedOperation(_)) => {}
                Err(err) => panic!(
                    "{full_model_id} should fail with UnsupportedOperation for speech, got {err:?}"
                ),
            }
        }

        if !provider_supports_capability(&provider_id, "transcription") {
            match registry.transcription_model(&full_model_id) {
                Ok(_) => {
                    panic!("{full_model_id} registry transcription handle should be unsupported")
                }
                Err(LlmError::UnsupportedOperation(_)) => {}
                Err(err) => panic!(
                    "{full_model_id} should fail with UnsupportedOperation for transcription, got {err:?}"
                ),
            }
        }
    }
}
