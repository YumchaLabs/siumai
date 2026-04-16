#![cfg(feature = "groq")]

#[test]
fn groq_facade_model_catalog_matches_current_ai_sdk_surface() {
    use siumai::models::groq::{chat, preview, production, speech, transcription};

    assert!(chat::ALL.contains(&production::GEMMA2_9B_IT));
    assert!(chat::ALL.contains(&production::GPT_OSS_120B));
    assert!(chat::ALL.contains(&preview::LLAMA_GUARD_3_8B));
    assert!(chat::ALL.contains(&preview::DEEPSEEK_R1_DISTILL_QWEN_32B));

    assert_eq!(
        transcription::ALL,
        &[
            transcription::WHISPER_LARGE_V3_TURBO,
            transcription::WHISPER_LARGE_V3
        ]
    );
    assert!(!chat::ALL.contains(&speech::PLAYAI_TTS));
}

#[test]
fn groq_facade_model_catalog_excludes_obsolete_package_models() {
    let models = siumai::models::groq::all_models();

    for obsolete in [
        "compound-beta",
        "compound-beta-mini",
        "llama-3.2-11b-vision-preview",
        "llama-3.2-90b-vision-preview",
        "llama3-groq-8b-8192-tool-use-preview",
        "llama3-groq-70b-8192-tool-use-preview",
        "llava-v1.5-7b-4096-preview",
        "gemma-7b-it",
    ] {
        assert!(
            !models.contains(&obsolete),
            "obsolete model should stay removed: {obsolete}"
        );
    }
}
