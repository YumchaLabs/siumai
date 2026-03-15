#![cfg(feature = "openai")]

use siumai::experimental::client::LlmClient;
use siumai::prelude::unified::Provider;
use siumai::provider_ext::openai_compatible::{
    get_provider_config, list_provider_ids, provider_supports_capability,
};

#[test]
fn compat_audio_preset_capability_matrix_matches_documented_policy() {
    let provider_ids = list_provider_ids();

    assert!(
        provider_ids
            .iter()
            .any(|provider_id| provider_id == "siliconflow")
    );
    assert!(
        provider_ids
            .iter()
            .any(|provider_id| provider_id == "together")
    );
    assert!(
        provider_ids
            .iter()
            .any(|provider_id| provider_id == "fireworks")
    );

    let siliconflow = get_provider_config("siliconflow").expect("siliconflow provider config");
    let together = get_provider_config("together").expect("together provider config");
    let fireworks = get_provider_config("fireworks").expect("fireworks provider config");

    assert_eq!(siliconflow.id, "siliconflow");
    assert_eq!(together.id, "together");
    assert_eq!(fireworks.id, "fireworks");

    assert!(provider_supports_capability("siliconflow", "speech"));
    assert!(provider_supports_capability("siliconflow", "transcription"));
    assert!(provider_supports_capability("siliconflow", "audio"));

    assert!(provider_supports_capability("together", "speech"));
    assert!(provider_supports_capability("together", "transcription"));
    assert!(provider_supports_capability("together", "audio"));

    assert!(!provider_supports_capability("fireworks", "speech"));
    assert!(provider_supports_capability("fireworks", "transcription"));
    assert!(provider_supports_capability("fireworks", "audio"));
}

#[test]
fn compat_preset_builder_shortcuts_resolve_shared_registry_defaults() {
    let siliconflow_registry =
        get_provider_config("siliconflow").expect("siliconflow provider config");
    let together_registry = get_provider_config("together").expect("together provider config");
    let fireworks_registry = get_provider_config("fireworks").expect("fireworks provider config");

    let siliconflow_config = Provider::openai()
        .siliconflow()
        .api_key("test-key")
        .into_config()
        .expect("siliconflow config");
    let together_config = Provider::openai()
        .together()
        .api_key("test-key")
        .into_config()
        .expect("together config");
    let fireworks_config = Provider::openai()
        .fireworks()
        .api_key("test-key")
        .into_config()
        .expect("fireworks config");

    assert_eq!(siliconflow_config.provider_id, "siliconflow");
    assert_eq!(together_config.provider_id, "together");
    assert_eq!(fireworks_config.provider_id, "fireworks");

    assert_eq!(siliconflow_config.base_url, siliconflow_registry.base_url);
    assert_eq!(together_config.base_url, together_registry.base_url);
    assert_eq!(fireworks_config.base_url, fireworks_registry.base_url);
}

#[tokio::test]
async fn compat_audio_preset_public_clients_expose_documented_audio_split() {
    let siliconflow = Provider::openai()
        .siliconflow()
        .api_key("test-key")
        .model("FunAudioLLM/SenseVoiceSmall")
        .build()
        .await
        .expect("siliconflow client");
    let together = Provider::openai()
        .together()
        .api_key("test-key")
        .model("openai/whisper-large-v3")
        .build()
        .await
        .expect("together client");
    let fireworks = Provider::openai()
        .fireworks()
        .api_key("test-key")
        .model("whisper-v3")
        .build()
        .await
        .expect("fireworks client");

    let siliconflow_caps = siliconflow.capabilities();
    assert!(siliconflow_caps.supports("audio"));
    assert!(siliconflow_caps.supports("speech"));
    assert!(siliconflow_caps.supports("transcription"));
    assert!(siliconflow.as_audio_capability().is_some());
    assert!(siliconflow.as_speech_capability().is_some());
    assert!(siliconflow.as_transcription_capability().is_some());

    let together_caps = together.capabilities();
    assert!(together_caps.supports("audio"));
    assert!(together_caps.supports("speech"));
    assert!(together_caps.supports("transcription"));
    assert!(together.as_audio_capability().is_some());
    assert!(together.as_speech_capability().is_some());
    assert!(together.as_transcription_capability().is_some());

    let fireworks_caps = fireworks.capabilities();
    assert!(fireworks_caps.supports("audio"));
    assert!(!fireworks_caps.supports("speech"));
    assert!(fireworks_caps.supports("transcription"));
    assert!(fireworks.as_audio_capability().is_some());
    assert!(fireworks.as_speech_capability().is_none());
    assert!(fireworks.as_transcription_capability().is_some());
}

#[tokio::test]
async fn compat_image_preset_public_clients_expose_documented_image_generation_split() {
    let siliconflow = Provider::openai()
        .siliconflow()
        .api_key("test-key")
        .model("stabilityai/stable-diffusion-3.5-large")
        .build()
        .await
        .expect("siliconflow client");
    let together = Provider::openai()
        .together()
        .api_key("test-key")
        .model("black-forest-labs/FLUX.1-schnell-Free")
        .build()
        .await
        .expect("together client");
    let fireworks = Provider::openai()
        .fireworks()
        .api_key("test-key")
        .model("accounts/fireworks/models/llama-v3p1-8b-instruct")
        .build()
        .await
        .expect("fireworks client");

    let siliconflow_caps = siliconflow.capabilities();
    assert!(siliconflow_caps.supports("image_generation"));
    assert!(siliconflow.as_image_generation_capability().is_some());

    let together_caps = together.capabilities();
    assert!(together_caps.supports("image_generation"));
    assert!(together.as_image_generation_capability().is_some());

    let fireworks_caps = fireworks.capabilities();
    assert!(!fireworks_caps.supports("image_generation"));
    assert!(fireworks.as_image_generation_capability().is_none());
}

#[test]
fn compat_embedding_rerank_preset_capability_matrix_matches_documented_policy() {
    let provider_ids = list_provider_ids();

    assert!(provider_ids.iter().any(|provider_id| provider_id == "jina"));
    assert!(
        provider_ids
            .iter()
            .any(|provider_id| provider_id == "voyageai")
    );
    assert!(
        provider_ids
            .iter()
            .any(|provider_id| provider_id == "infini")
    );

    let jina = get_provider_config("jina").expect("jina provider config");
    let voyageai = get_provider_config("voyageai").expect("voyageai provider config");
    let infini = get_provider_config("infini").expect("infini provider config");

    assert_eq!(jina.id, "jina");
    assert_eq!(voyageai.id, "voyageai");
    assert_eq!(infini.id, "infini");

    assert!(provider_supports_capability("jina", "embedding"));
    assert!(provider_supports_capability("jina", "rerank"));
    assert!(!provider_supports_capability("jina", "chat"));
    assert!(!provider_supports_capability("jina", "streaming"));

    assert!(provider_supports_capability("voyageai", "embedding"));
    assert!(provider_supports_capability("voyageai", "rerank"));
    assert!(!provider_supports_capability("voyageai", "chat"));
    assert!(!provider_supports_capability("voyageai", "streaming"));

    assert!(provider_supports_capability("infini", "embedding"));
    assert!(provider_supports_capability("infini", "chat"));
    assert!(provider_supports_capability("infini", "streaming"));
    assert!(!provider_supports_capability("infini", "rerank"));
    assert!(!provider_supports_capability("infini", "image_generation"));
}

#[tokio::test]
async fn compat_embedding_rerank_preset_public_clients_expose_documented_non_text_split() {
    let jina = Provider::openai()
        .jina()
        .api_key("test-key")
        .model("jina-embeddings-v2-base-en")
        .build()
        .await
        .expect("jina client");
    let voyageai = Provider::openai()
        .voyageai()
        .api_key("test-key")
        .model("voyage-3")
        .build()
        .await
        .expect("voyageai client");
    let infini = Provider::openai()
        .infini()
        .api_key("test-key")
        .model("text-embedding-3-small")
        .build()
        .await
        .expect("infini client");
    let openrouter = Provider::openai()
        .openrouter()
        .api_key("test-key")
        .model("openai/text-embedding-3-small")
        .build()
        .await
        .expect("openrouter client");

    let jina_caps = jina.capabilities();
    assert!(!jina_caps.supports("chat"));
    assert!(!jina_caps.supports("streaming"));
    assert!(jina_caps.supports("embedding"));
    assert!(jina_caps.supports("rerank"));
    assert!(jina.as_chat_capability().is_none());
    assert!(jina.as_embedding_capability().is_some());
    assert!(jina.as_rerank_capability().is_some());

    let voyageai_caps = voyageai.capabilities();
    assert!(!voyageai_caps.supports("chat"));
    assert!(!voyageai_caps.supports("streaming"));
    assert!(voyageai_caps.supports("embedding"));
    assert!(voyageai_caps.supports("rerank"));
    assert!(voyageai.as_chat_capability().is_none());
    assert!(voyageai.as_embedding_capability().is_some());
    assert!(voyageai.as_rerank_capability().is_some());

    let infini_caps = infini.capabilities();
    assert!(infini_caps.supports("chat"));
    assert!(infini_caps.supports("streaming"));
    assert!(infini_caps.supports("embedding"));
    assert!(!infini_caps.supports("rerank"));
    assert!(infini.as_chat_capability().is_some());
    assert!(infini.as_embedding_capability().is_some());
    assert!(infini.as_rerank_capability().is_none());

    let openrouter_caps = openrouter.capabilities();
    assert!(openrouter_caps.supports("embedding"));
    assert!(!openrouter_caps.supports("rerank"));
    assert!(openrouter.as_embedding_capability().is_some());
    assert!(openrouter.as_rerank_capability().is_none());
}

#[test]
fn compat_vendor_view_capability_matrix_matches_documented_policy() {
    let provider_ids = list_provider_ids();

    assert!(
        provider_ids
            .iter()
            .any(|provider_id| provider_id == "openrouter")
    );
    assert!(
        provider_ids
            .iter()
            .any(|provider_id| provider_id == "perplexity")
    );

    let openrouter = get_provider_config("openrouter").expect("openrouter provider config");
    let perplexity = get_provider_config("perplexity").expect("perplexity provider config");

    assert_eq!(openrouter.id, "openrouter");
    assert_eq!(perplexity.id, "perplexity");

    assert!(provider_supports_capability("openrouter", "tools"));
    assert!(provider_supports_capability("openrouter", "embedding"));
    assert!(provider_supports_capability("openrouter", "reasoning"));
    assert!(!provider_supports_capability("openrouter", "rerank"));
    assert!(!provider_supports_capability("openrouter", "speech"));
    assert!(!provider_supports_capability("openrouter", "transcription"));
    assert!(!provider_supports_capability("openrouter", "audio"));

    assert!(provider_supports_capability("perplexity", "tools"));
    assert!(!provider_supports_capability("perplexity", "embedding"));
    assert!(!provider_supports_capability("perplexity", "speech"));
    assert!(!provider_supports_capability("perplexity", "transcription"));
    assert!(!provider_supports_capability("perplexity", "audio"));
}

#[test]
fn compat_vendor_view_builder_shortcuts_resolve_shared_registry_defaults() {
    let openrouter_registry =
        get_provider_config("openrouter").expect("openrouter provider config");
    let perplexity_registry =
        get_provider_config("perplexity").expect("perplexity provider config");

    let openrouter_config = Provider::openai()
        .openrouter()
        .api_key("test-key")
        .into_config()
        .expect("openrouter config");
    let perplexity_config = Provider::openai()
        .perplexity()
        .api_key("test-key")
        .into_config()
        .expect("perplexity config");

    assert_eq!(openrouter_config.provider_id, "openrouter");
    assert_eq!(perplexity_config.provider_id, "perplexity");

    assert_eq!(openrouter_config.base_url, openrouter_registry.base_url);
    assert_eq!(perplexity_config.base_url, perplexity_registry.base_url);
}

#[tokio::test]
async fn compat_vendor_view_public_clients_expose_documented_capability_split() {
    let openrouter = Provider::openai()
        .openrouter()
        .api_key("test-key")
        .model("openai/gpt-4o")
        .build()
        .await
        .expect("openrouter client");
    let perplexity = Provider::openai()
        .perplexity()
        .api_key("test-key")
        .model("sonar")
        .build()
        .await
        .expect("perplexity client");

    let openrouter_caps = openrouter.capabilities();
    assert!(openrouter_caps.supports("tools"));
    assert!(openrouter_caps.supports("embedding"));
    assert!(openrouter_caps.supports("reasoning"));
    assert!(!openrouter_caps.supports("rerank"));
    assert!(!openrouter_caps.supports("speech"));
    assert!(!openrouter_caps.supports("transcription"));
    assert!(!openrouter_caps.supports("audio"));
    assert!(openrouter.as_embedding_capability().is_some());
    assert!(openrouter.as_rerank_capability().is_none());
    assert!(openrouter.as_audio_capability().is_none());
    assert!(openrouter.as_speech_capability().is_none());
    assert!(openrouter.as_transcription_capability().is_none());

    let perplexity_caps = perplexity.capabilities();
    assert!(perplexity_caps.supports("tools"));
    assert!(!perplexity_caps.supports("embedding"));
    assert!(!perplexity_caps.supports("speech"));
    assert!(!perplexity_caps.supports("transcription"));
    assert!(!perplexity_caps.supports("audio"));
    assert!(perplexity.as_embedding_capability().is_none());
    assert!(perplexity.as_audio_capability().is_none());
    assert!(perplexity.as_speech_capability().is_none());
    assert!(perplexity.as_transcription_capability().is_none());
}
