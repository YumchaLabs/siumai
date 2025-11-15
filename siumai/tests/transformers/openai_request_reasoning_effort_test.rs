//! Verify OpenAI ProviderOptions are mapped into request JSON
//! via the std-openai adapter (Chat Completions path).

use siumai::core::{ProviderContext, ProviderSpec};
use siumai::execution::transformers::request::RequestTransformer;
use siumai::types::{ChatMessage, ChatRequest, CommonParams, OpenAiOptions};

#[test]
fn openai_reasoning_and_audio_options_are_mapped_via_std_openai() {
    // Build a minimal ChatRequest with OpenAI-specific options.
    let req = ChatRequest::builder()
        .messages(vec![ChatMessage::user("hi").build()])
        .common_params(CommonParams {
            model: "gpt-4o-mini".to_string(),
            ..Default::default()
        })
        .openai_options(
            OpenAiOptions::new()
                .with_reasoning_effort(siumai::types::ReasoningEffort::High)
                .with_service_tier(siumai::types::ServiceTier::Default)
                .with_audio_voice(siumai::types::ChatCompletionAudioVoice::Ash),
        )
        .build();

    // Provider context with standard OpenAI base URL.
    let ctx = ProviderContext::new(
        "openai",
        "https://api.openai.com/v1",
        Some("test-key".to_string()),
        std::collections::HashMap::new(),
    );

    // Use the runtime OpenAiSpec, which under std-openai-external bridges
    // through OpenAiChatStandard + OpenAiDefaultChatAdapter.
    let spec = siumai::providers::openai::spec::OpenAiSpec::new();
    let transformers = spec.choose_chat_transformers(&req, &ctx);

    let body = transformers
        .request
        .transform_chat(&req)
        .expect("transform ok");

    // Reasoning effort should be serialized as a lowercase string.
    assert_eq!(
        body.get("reasoning_effort").and_then(|v| v.as_str()),
        Some("high")
    );

    // Service tier should also be serialized as a lowercase string.
    assert_eq!(
        body.get("service_tier").and_then(|v| v.as_str()),
        Some("default")
    );

    // Modalities should include "text" and "audio" when audio is requested.
    let modalities = body
        .get("modalities")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let modalities_strs: Vec<_> = modalities
        .into_iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    assert!(modalities_strs.contains(&"text".to_string()));
    assert!(modalities_strs.contains(&"audio".to_string()));

    // Audio config should be present with the selected voice.
    let audio = body.get("audio").cloned().unwrap_or_default();
    assert_eq!(
        audio
            .get("voice")
            .and_then(|v| v.as_str())
            .unwrap_or_default(),
        "ash"
    );
}
