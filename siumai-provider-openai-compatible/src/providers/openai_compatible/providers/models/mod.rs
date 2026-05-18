//! OpenAI-Compatible Provider Model Definitions
//!
//! This module contains model definitions for various OpenAI-compatible providers.

pub mod alibaba;
pub mod deepinfra;
pub mod deepseek;
pub mod fireworks;
pub mod google_vertex_xai;
pub mod groq;
pub mod mistral;
pub mod moonshot;
pub mod moonshotai;
pub mod openrouter;
pub mod perplexity;
pub mod qwen;
pub mod siliconflow;
pub mod together;
pub mod togetherai;
pub mod vertex_maas;
pub mod xai;

/// Get models for a specific provider
pub fn get_models_for_provider(provider: &str) -> Vec<String> {
    match provider.to_lowercase().as_str() {
        "deepseek" => deepseek::all_models(),
        "deepinfra" => deepinfra::all_models(),
        "fireworks" => fireworks::all_models(),
        "google-vertex-xai" | "googlevertex.xai" | "vertex-xai" => google_vertex_xai::all_models(),
        "openrouter" => openrouter::all_models(),
        "vertex-maas" => vertex_maas::all_models(),
        "xai" => xai::all_models(),
        "groq" => groq::all_models(),
        "together" | "togetherai" => togetherai::all_models(),
        "siliconflow" => siliconflow::all_models(),
        "moonshot" | "moonshotai" => moonshot::all_models(),
        _ => vec![],
    }
}

/// Check if a model is supported by a provider
pub fn is_model_supported(provider: &str, model: &str) -> bool {
    get_models_for_provider(provider).contains(&model.to_string())
}

/// Model recommendations for different use cases
pub mod recommendations {
    use super::*;

    /// Recommended model for general chat
    pub const fn for_chat() -> &'static str {
        openrouter::openai::GPT_4O
    }

    /// Recommended model for coding tasks
    pub const fn for_coding() -> &'static str {
        deepseek::DEEPSEEK_V3_0324 // Use latest V3 model for coding
    }

    /// Recommended model for reasoning tasks
    pub const fn for_reasoning() -> &'static str {
        deepseek::REASONER
    }

    /// Recommended model for fast responses
    pub const fn for_fast_response() -> &'static str {
        groq::LLAMA_3_1_8B
    }

    /// Recommended model for cost-effective usage
    pub const fn for_cost_effective() -> &'static str {
        deepseek::CHAT
    }

    /// Recommended model for vision tasks
    pub const fn for_vision() -> &'static str {
        openrouter::openai::GPT_4O
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_model_catalog_stays_split_by_provider_family() {
        let source = include_str!("mod.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap_or_default();
        let facade_markers = [
            "pub mod deepseek;",
            "pub mod openrouter;",
            "pub mod deepinfra;",
            "pub mod vertex_maas;",
            "pub mod google_vertex_xai;",
            "pub mod fireworks;",
            "pub mod xai;",
            "pub mod siliconflow;",
            "pub mod groq;",
            "pub mod togetherai;",
            "pub mod moonshot;",
        ];

        for marker in facade_markers {
            assert!(
                source.contains(marker),
                "models facade should keep provider family module `{marker}`"
            );
        }

        let legacy_models_file = std::path::Path::new(file!())
            .parent()
            .and_then(|path| path.parent())
            .map(|path| path.join("models.rs"))
            .expect("models facade should resolve the legacy file path");

        assert!(
            !legacy_models_file.exists(),
            "legacy monolithic provider model catalog should stay deleted"
        );

        assert!(!source.contains("pub const CHAT: &str"));
        assert!(!source.contains("pub const ALL_CHAT: &[&str]"));
        assert!(!source.contains("pub fn all_models() -> Vec<String>"));
    }

    #[test]
    fn test_deepseek_models() {
        let models = deepseek::all_models();
        assert!(!models.is_empty());
        assert!(models.contains(&"deepseek-chat".to_string()));
    }

    #[test]
    fn test_openrouter_models() {
        let models = openrouter::all_models();
        assert!(!models.is_empty());
        assert!(models.contains(&"openai/gpt-4o".to_string()));
    }

    #[test]
    fn test_deepinfra_models() {
        let models = deepinfra::all_models();
        assert!(!models.is_empty());
        assert!(models.contains(&deepinfra::CHAT.to_string()));
        assert!(models.contains(&deepinfra::COMPLETION.to_string()));
        assert!(models.contains(&deepinfra::EMBEDDING.to_string()));
        assert!(models.contains(&deepinfra::IMAGE.to_string()));
    }

    #[test]
    fn test_vertex_maas_models() {
        let models = vertex_maas::all_models();
        assert!(!models.is_empty());
        assert!(models.contains(&vertex_maas::CHAT.to_string()));
        assert!(models.contains(&vertex_maas::COMPLETION.to_string()));
        assert!(models.contains(&vertex_maas::EMBEDDING.to_string()));
    }

    #[test]
    fn test_fireworks_models() {
        let models = fireworks::all_models();
        assert!(!models.is_empty());
        assert!(models.contains(&fireworks::CHAT.to_string()));
        assert!(models.contains(&fireworks::COMPLETION.to_string()));
        assert!(models.contains(&fireworks::EMBEDDING.to_string()));
        assert!(models.contains(&fireworks::IMAGE.to_string()));
    }

    #[test]
    fn test_google_vertex_xai_models() {
        let models = google_vertex_xai::all_models();
        assert_eq!(models.len(), google_vertex_xai::ALL_CHAT.len());
        assert!(models.contains(&google_vertex_xai::chat::GROK_4_1_FAST_REASONING.to_string()));
    }

    #[test]
    fn test_xai_models() {
        let models = xai::all_models();
        assert!(!models.is_empty());
        assert!(models.contains(&xai::CHAT.to_string()));
        assert!(models.contains(&xai::IMAGE.to_string()));
        assert!(models.contains(&xai::VIDEO.to_string()));
        assert!(models.contains(&xai::GROK_BETA.to_string()));
        assert!(models.contains(&xai::GROK_CODE_FAST_1.to_string()));
    }

    #[test]
    fn test_groq_models() {
        let models = groq::all_models();
        assert!(!models.is_empty());
        assert!(models.contains(&groq::CHAT.to_string()));
        assert!(models.contains(&groq::TRANSCRIPTION.to_string()));
        assert!(models.contains(&groq::production::GPT_OSS_20B.to_string()));
        assert!(models.contains(&groq::preview::QWEN3_32B.to_string()));
    }

    #[test]
    fn test_togetherai_models() {
        let models = togetherai::all_models();
        assert!(!models.is_empty());
        assert!(models.contains(&togetherai::CHAT.to_string()));
        assert!(models.contains(&togetherai::COMPLETION.to_string()));
        assert!(models.contains(&togetherai::EMBEDDING.to_string()));
        assert!(models.contains(&togetherai::IMAGE.to_string()));
        assert!(models.contains(&togetherai::RERANK.to_string()));
    }

    #[test]
    fn test_get_models_for_provider() {
        let deepseek_models = get_models_for_provider("deepseek");
        assert!(!deepseek_models.is_empty());

        let deepinfra_models = get_models_for_provider("deepinfra");
        assert!(deepinfra_models.contains(&deepinfra::chat::LLAMA_V3P3_70B_INSTRUCT.to_string()));

        let vertex_maas_models = get_models_for_provider("vertex-maas");
        assert!(vertex_maas_models.contains(&vertex_maas::chat::DEEPSEEK_V3_2_MAAS.to_string()));

        let fireworks_models = get_models_for_provider("fireworks");
        assert!(fireworks_models.contains(&fireworks::chat::DEEPSEEK_V3.to_string()));

        let google_vertex_xai_models = get_models_for_provider("google-vertex-xai");
        assert!(
            google_vertex_xai_models
                .contains(&google_vertex_xai::chat::GROK_4_1_FAST_REASONING.to_string())
        );

        let moonshotai_models = get_models_for_provider("moonshotai");
        assert!(moonshotai_models.contains(&moonshotai::KIMI_K2_0905.to_string()));

        let xai_models = get_models_for_provider("xai");
        assert!(xai_models.contains(&xai::grok_4::GROK_4_LATEST.to_string()));

        let groq_models = get_models_for_provider("groq");
        assert!(groq_models.contains(&groq::production::GPT_OSS_120B.to_string()));

        let togetherai_models = get_models_for_provider("togetherai");
        assert!(togetherai_models.contains(&togetherai::chat::DEEPSEEK_V3.to_string()));

        let unknown_models = get_models_for_provider("unknown");
        assert!(unknown_models.is_empty());
    }

    #[test]
    fn test_is_model_supported() {
        assert!(is_model_supported("deepseek", "deepseek-chat"));
        assert!(is_model_supported(
            "deepinfra",
            deepinfra::image::FLUX_1_KONTEXT_PRO
        ));
        assert!(is_model_supported(
            "vertex-maas",
            vertex_maas::chat::QWEN3_NEXT_80B_A3B_THINKING_MAAS
        ));
        assert!(is_model_supported(
            "fireworks",
            fireworks::image::FLUX_KONTEXT_PRO
        ));
        assert!(is_model_supported(
            "google-vertex-xai",
            google_vertex_xai::chat::GROK_4_20_REASONING
        ));
        assert!(is_model_supported("moonshotai", moonshotai::KIMI_K2P5));
        assert!(is_model_supported("xai", xai::grok_4::GROK_4_LATEST));
        assert!(is_model_supported("groq", groq::production::GPT_OSS_20B));
        assert!(is_model_supported(
            "togetherai",
            togetherai::image::FLUX_1_KONTEXT_PRO
        ));
        assert!(!is_model_supported("deepseek", "unknown-model"));
        assert!(!is_model_supported("unknown", "any-model"));
    }
}
