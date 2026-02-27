use crate::traits::ProviderCapabilities;
use crate::types::ProviderType;
use std::borrow::Cow;

/// Provider Information
#[derive(Debug, Clone)]
pub struct ProviderInfo {
    /// Provider type
    pub provider_type: ProviderType,
    /// Provider name
    pub name: Cow<'static, str>,
    /// Description
    pub description: Cow<'static, str>,
    /// Supported capabilities
    pub capabilities: ProviderCapabilities,
    /// Default base URL
    pub default_base_url: Cow<'static, str>,
    /// Supported models
    pub supported_models: Vec<Cow<'static, str>>,
}

/// Get information for all supported providers
#[allow(clippy::vec_init_then_push)]
pub fn get_supported_providers() -> Vec<ProviderInfo> {
    let providers_iter: Vec<crate::registry::ProviderRecord> =
        if let Ok(guard) = crate::registry::global_registry().read() {
            guard.list_providers().into_iter().cloned().collect()
        } else {
            crate::registry::ProviderRegistry::with_builtin_providers()
                .list_providers()
                .into_iter()
                .cloned()
                .collect()
        };

    let native_metas = crate::native_provider_metadata::native_providers_metadata();

    let mut out = Vec::new();
    for rec in providers_iter {
        let ptype = ProviderType::from_name(&rec.id);
        #[allow(unreachable_patterns)]
        match ptype {
            #[cfg(feature = "openai")]
            ProviderType::OpenAi => {
                let meta = native_metas
                    .iter()
                    .find(|m| m.id == "openai")
                    .expect("OpenAI metadata should be registered");
                use siumai_provider_openai::providers::openai::model_constants as openai;
                let mut models: Vec<Cow<'static, str>> = Vec::new();
                models.extend(openai::gpt_4o::ALL.iter().copied().map(Cow::Borrowed));
                models.extend(openai::gpt_4_1::ALL.iter().copied().map(Cow::Borrowed));
                models.extend(openai::gpt_4_5::ALL.iter().copied().map(Cow::Borrowed));
                models.extend(openai::gpt_4_turbo::ALL.iter().copied().map(Cow::Borrowed));
                models.extend(openai::gpt_4::ALL.iter().copied().map(Cow::Borrowed));
                models.extend(openai::o1::ALL.iter().copied().map(Cow::Borrowed));
                models.extend(openai::o3::ALL.iter().copied().map(Cow::Borrowed));
                models.extend(openai::o4::ALL.iter().copied().map(Cow::Borrowed));
                models.extend(openai::gpt_5::ALL.iter().copied().map(Cow::Borrowed));
                models.extend(openai::gpt_3_5::ALL.iter().copied().map(Cow::Borrowed));
                models.extend(openai::audio::ALL.iter().copied().map(Cow::Borrowed));
                models.extend(openai::images::ALL.iter().copied().map(Cow::Borrowed));
                models.extend(openai::embeddings::ALL.iter().copied().map(Cow::Borrowed));
                models.extend(openai::moderation::ALL.iter().copied().map(Cow::Borrowed));
                out.push(ProviderInfo {
                    provider_type: ptype,
                    name: Cow::Borrowed(meta.name),
                    description: Cow::Borrowed(meta.description),
                    capabilities: rec.capabilities.clone(),
                    default_base_url: Cow::Borrowed(
                        meta.default_base_url.unwrap_or("https://api.openai.com/v1"),
                    ),
                    supported_models: models,
                });
            }
            #[cfg(feature = "anthropic")]
            ProviderType::Anthropic => {
                let meta = native_metas
                    .iter()
                    .find(|m| m.id == "anthropic")
                    .expect("Anthropic metadata should be registered");
                use siumai_provider_anthropic::providers::anthropic::model_constants as anthropic;
                let mut models: Vec<Cow<'static, str>> = Vec::new();
                models.extend(
                    anthropic::claude_opus_4_1::ALL
                        .iter()
                        .copied()
                        .map(Cow::Borrowed),
                );
                models.extend(
                    anthropic::claude_opus_4::ALL
                        .iter()
                        .copied()
                        .map(Cow::Borrowed),
                );
                models.extend(
                    anthropic::claude_sonnet_4::ALL
                        .iter()
                        .copied()
                        .map(Cow::Borrowed),
                );
                models.extend(
                    anthropic::claude_sonnet_3_7::ALL
                        .iter()
                        .copied()
                        .map(Cow::Borrowed),
                );
                models.extend(
                    anthropic::claude_sonnet_3_5::ALL
                        .iter()
                        .copied()
                        .map(Cow::Borrowed),
                );
                models.extend(
                    anthropic::claude_haiku_3_5::ALL
                        .iter()
                        .copied()
                        .map(Cow::Borrowed),
                );
                models.extend(
                    anthropic::claude_haiku_3::ALL
                        .iter()
                        .copied()
                        .map(Cow::Borrowed),
                );
                models.extend(
                    anthropic::claude_opus_3::ALL
                        .iter()
                        .copied()
                        .map(Cow::Borrowed),
                );
                models.extend(
                    anthropic::claude_sonnet_3::ALL
                        .iter()
                        .copied()
                        .map(Cow::Borrowed),
                );
                out.push(ProviderInfo {
                    provider_type: ptype,
                    name: Cow::Borrowed(meta.name),
                    description: Cow::Borrowed(meta.description),
                    capabilities: rec.capabilities.clone(),
                    default_base_url: Cow::Borrowed(
                        meta.default_base_url.unwrap_or("https://api.anthropic.com"),
                    ),
                    supported_models: models,
                });
            }
            #[cfg(feature = "google")]
            ProviderType::Gemini => {
                let meta = native_metas
                    .iter()
                    .find(|m| m.id == "gemini")
                    .expect("Gemini metadata should be registered");
                use siumai_provider_gemini::providers::gemini::model_constants as gemini;
                let mut models: Vec<Cow<'static, str>> = Vec::new();
                models.extend(
                    gemini::gemini_2_5_pro::ALL
                        .iter()
                        .copied()
                        .map(Cow::Borrowed),
                );
                models.extend(
                    gemini::gemini_2_5_flash::ALL
                        .iter()
                        .copied()
                        .map(Cow::Borrowed),
                );
                models.extend(
                    gemini::gemini_2_5_flash_lite::ALL
                        .iter()
                        .copied()
                        .map(Cow::Borrowed),
                );
                models.extend(
                    gemini::gemini_2_0_flash::ALL
                        .iter()
                        .copied()
                        .map(Cow::Borrowed),
                );
                models.extend(
                    gemini::gemini_2_0_flash_lite::ALL
                        .iter()
                        .copied()
                        .map(Cow::Borrowed),
                );
                models.extend(
                    gemini::gemini_1_5_flash::ALL
                        .iter()
                        .copied()
                        .map(Cow::Borrowed),
                );
                models.extend(
                    gemini::gemini_1_5_flash_8b::ALL
                        .iter()
                        .copied()
                        .map(Cow::Borrowed),
                );
                models.extend(
                    gemini::gemini_1_5_pro::ALL
                        .iter()
                        .copied()
                        .map(Cow::Borrowed),
                );
                out.push(ProviderInfo {
                    provider_type: ptype,
                    name: Cow::Borrowed(meta.name),
                    description: Cow::Borrowed(meta.description),
                    capabilities: rec.capabilities.clone(),
                    default_base_url: Cow::Borrowed(
                        "https://generativelanguage.googleapis.com/v1beta",
                    ),
                    supported_models: models,
                });
            }
            #[cfg(feature = "ollama")]
            ProviderType::Ollama => {
                let meta = native_metas
                    .iter()
                    .find(|m| m.id == "ollama")
                    .expect("Ollama metadata should be registered");
                out.push(ProviderInfo {
                    provider_type: ptype,
                    name: Cow::Borrowed(meta.name),
                    description: Cow::Borrowed(meta.description),
                    capabilities: rec.capabilities.clone(),
                    default_base_url: Cow::Borrowed(
                        meta.default_base_url.unwrap_or("http://localhost:11434"),
                    ),
                    supported_models: vec![
                        Cow::Borrowed("llama3.2:latest"),
                        Cow::Borrowed("llama3.2:3b"),
                        Cow::Borrowed("llama3.2:1b"),
                        Cow::Borrowed("llama3.1:latest"),
                        Cow::Borrowed("llama3.1:8b"),
                        Cow::Borrowed("llama3.1:70b"),
                        Cow::Borrowed("mistral:latest"),
                        Cow::Borrowed("mistral:7b"),
                        Cow::Borrowed("codellama:latest"),
                        Cow::Borrowed("codellama:7b"),
                        Cow::Borrowed("codellama:13b"),
                        Cow::Borrowed("codellama:34b"),
                        Cow::Borrowed("phi3:latest"),
                        Cow::Borrowed("phi3:mini"),
                        Cow::Borrowed("phi3:medium"),
                        Cow::Borrowed("gemma:latest"),
                        Cow::Borrowed("gemma:2b"),
                        Cow::Borrowed("gemma:7b"),
                        Cow::Borrowed("qwen2:latest"),
                        Cow::Borrowed("qwen2:0.5b"),
                        Cow::Borrowed("qwen2:1.5b"),
                        Cow::Borrowed("qwen2:7b"),
                        Cow::Borrowed("qwen2:72b"),
                        Cow::Borrowed("deepseek-coder:latest"),
                        Cow::Borrowed("deepseek-coder:6.7b"),
                        Cow::Borrowed("deepseek-coder:33b"),
                        Cow::Borrowed("nomic-embed-text:latest"),
                        Cow::Borrowed("all-minilm:latest"),
                    ],
                });
            }
            #[cfg(feature = "xai")]
            ProviderType::XAI => {
                let meta = native_metas
                    .iter()
                    .find(|m| m.id == "xai")
                    .expect("xAI metadata should be registered");
                out.push(ProviderInfo {
                    provider_type: ptype,
                    name: Cow::Borrowed(meta.name),
                    description: Cow::Borrowed(meta.description),
                    capabilities: rec.capabilities.clone(),
                    default_base_url: Cow::Borrowed(
                        meta.default_base_url.unwrap_or("https://api.x.ai/v1"),
                    ),
                    supported_models: siumai_provider_xai::providers::xai::models::all_models()
                        .into_iter()
                        .map(Cow::Borrowed)
                        .collect(),
                });
            }
            #[cfg(feature = "groq")]
            ProviderType::Groq => {
                let meta = native_metas
                    .iter()
                    .find(|m| m.id == "groq")
                    .expect("Groq metadata should be registered");
                out.push(ProviderInfo {
                    provider_type: ptype,
                    name: Cow::Borrowed(meta.name),
                    description: Cow::Borrowed(meta.description),
                    capabilities: rec.capabilities.clone(),
                    default_base_url: Cow::Borrowed(
                        meta.default_base_url
                            .unwrap_or("https://api.groq.com/openai/v1"),
                    ),
                    supported_models: siumai_provider_groq::providers::groq::models::all_models()
                        .into_iter()
                        .map(Cow::Borrowed)
                        .collect(),
                });
            }
            #[cfg(feature = "minimaxi")]
            ProviderType::MiniMaxi => {
                let meta = native_metas
                    .iter()
                    .find(|m| m.id == "minimaxi")
                    .expect("MiniMaxi metadata should be registered");
                out.push(ProviderInfo {
                    provider_type: ptype,
                    name: Cow::Borrowed(meta.name),
                    description: Cow::Borrowed(meta.description),
                    capabilities: rec.capabilities.clone(),
                    default_base_url: Cow::Borrowed(
                        meta.default_base_url
                            .unwrap_or("https://api.minimaxi.com/v1"),
                    ),
                    supported_models: vec![
                        Cow::Borrowed("MiniMax-M2"),
                        Cow::Borrowed("speech-2.6-hd"),
                        Cow::Borrowed("speech-2.6-turbo"),
                        Cow::Borrowed("hailuo-2.3"),
                        Cow::Borrowed("hailuo-2.3-fast"),
                        Cow::Borrowed("music-2.0"),
                    ],
                });
            }
            ProviderType::Custom(_) => {
                // Treat native providers that aren't represented in `ProviderType` as built-ins
                // when they are registered via the shared native metadata table.
                if let Some(meta) = native_metas.iter().find(|m| m.id == rec.id) {
                    let mut models: Vec<Cow<'static, str>> = Vec::new();
                    if let Some(m) = rec.default_model.clone() {
                        models.push(Cow::Owned(m));
                    }

                    out.push(ProviderInfo {
                        provider_type: ProviderType::Custom(rec.id.clone()),
                        name: Cow::Borrowed(meta.name),
                        description: Cow::Borrowed(meta.description),
                        capabilities: rec.capabilities.clone(),
                        default_base_url: Cow::Borrowed(meta.default_base_url.unwrap_or("N/A")),
                        supported_models: models,
                    });
                    continue;
                }

                // OpenAI-compatible providers are registered as concrete provider ids
                // (e.g. "deepseek", "openrouter", "siliconflow"). Keep this catalog
                // as a discovery helper, not a strict source of truth for model lists.
                #[cfg(feature = "openai")]
                {
                    if let Some(cfg) =
                        siumai_provider_openai_compatible::providers::openai_compatible::get_provider_config(
                            &rec.id,
                        )
                    {
                        let mut models: Vec<Cow<'static, str>> = Vec::new();
                        if let Some(m) = cfg.default_model {
                            models.push(Cow::Owned(m));
                        }

                        // Add a few curated model ids when we have stable constants.
                        use siumai_provider_openai_compatible::providers::openai_compatible::providers::models as oai_models;
                        match rec.id.as_str() {
                            "deepseek" => {
                                models.push(Cow::Borrowed(oai_models::deepseek::CHAT));
                                models.push(Cow::Borrowed(oai_models::deepseek::REASONER));
                            }
                            "openrouter" => {
                                models.push(Cow::Borrowed(oai_models::openrouter::popular::GPT_4O));
                                models.push(Cow::Borrowed(
                                    oai_models::openrouter::popular::CLAUDE_SONNET_4,
                                ));
                                models.push(Cow::Borrowed(
                                    oai_models::openrouter::popular::GEMINI_2_5_PRO,
                                ));
                                models.push(Cow::Borrowed(
                                    oai_models::openrouter::popular::DEEPSEEK_REASONER,
                                ));
                            }
                            "siliconflow" => {
                                models.push(Cow::Borrowed(oai_models::siliconflow::DEEPSEEK_V3_1));
                                models.push(Cow::Borrowed(oai_models::siliconflow::DEEPSEEK_R1));
                            }
                            "moonshot" => {
                                models.push(Cow::Borrowed(
                                    oai_models::moonshot::KIMI_K2_0905_PREVIEW,
                                ));
                            }
                            _ => {}
                        }

                        out.push(ProviderInfo {
                            provider_type: ProviderType::Custom(cfg.id),
                            name: Cow::Owned(cfg.name),
                            description: Cow::Owned(format!(
                                "OpenAI-compatible provider (via adapter): {}",
                                rec.id
                            )),
                            capabilities: rec.capabilities.clone(),
                            default_base_url: Cow::Owned(cfg.base_url),
                            supported_models: models,
                        });
                        continue;
                    }
                }

                // Custom providers registered by users (or builds without OpenAI adapters).
                out.push(ProviderInfo {
                    provider_type: ProviderType::Custom(rec.id.clone()),
                    name: Cow::Owned(rec.name.clone()),
                    description: Cow::Borrowed("Custom provider"),
                    capabilities: rec.capabilities.clone(),
                    default_base_url: Cow::Owned(rec.base_url.unwrap_or_else(|| "N/A".into())),
                    supported_models: Vec::new(),
                });
            }
            // Generic fallback (keeps the catalog useful when provider features are disabled).
            _ => {
                out.push(ProviderInfo {
                    provider_type: ptype,
                    name: Cow::Owned(rec.name),
                    description: Cow::Owned(rec.id.clone()),
                    capabilities: rec.capabilities.clone(),
                    default_base_url: Cow::Owned(rec.base_url.unwrap_or_default()),
                    supported_models: Vec::new(),
                });
            }
        }
    }

    out
}

/// Get provider information by provider type
pub fn get_provider_info(provider_type: &ProviderType) -> Option<ProviderInfo> {
    get_supported_providers()
        .into_iter()
        .find(|info| &info.provider_type == provider_type)
}

/// Get provider information by provider id (canonical id or alias-like string).
pub fn get_provider_info_by_id(provider_id: &str) -> Option<ProviderInfo> {
    get_provider_info(&ProviderType::from_name(provider_id))
}

/// Check if a model is supported by the provider
pub fn is_model_supported(provider_type: &ProviderType, model: &str) -> bool {
    if let Some(info) = get_provider_info(provider_type) {
        info.supported_models.iter().any(|m| m.as_ref() == model)
    } else {
        false
    }
}

/// Check if a model is supported by the provider id.
pub fn is_model_supported_by_id(provider_id: &str, model: &str) -> bool {
    is_model_supported(&ProviderType::from_name(provider_id), model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "openai")]
    fn provider_catalog_includes_openai_compatible_providers() {
        let providers = get_supported_providers();
        assert!(
            providers
                .iter()
                .any(|p| matches!(&p.provider_type, ProviderType::Custom(id) if id == "deepseek")),
            "expected openai-compatible provider 'deepseek' to be listed"
        );
    }

    #[test]
    #[cfg(feature = "openai")]
    fn provider_catalog_lookup_by_id_works_for_openai_compatible() {
        let info = get_provider_info_by_id("deepseek").expect("deepseek should exist");
        assert!(matches!(info.provider_type, ProviderType::Custom(id) if id == "deepseek"));
    }

    #[test]
    #[cfg(feature = "cohere")]
    fn provider_catalog_uses_native_metadata_for_cohere() {
        let info = get_provider_info_by_id("cohere").expect("cohere should exist");
        assert_eq!(info.name.as_ref(), "Cohere");
        assert_ne!(info.description.as_ref(), "Custom provider");
        assert!(
            info.capabilities.rerank,
            "expected cohere to support rerank"
        );
    }

    #[test]
    #[cfg(feature = "togetherai")]
    fn provider_catalog_uses_native_metadata_for_togetherai() {
        let info = get_provider_info_by_id("togetherai").expect("togetherai should exist");
        assert_eq!(info.name.as_ref(), "TogetherAI");
        assert_ne!(info.description.as_ref(), "Custom provider");
        assert!(
            info.capabilities.rerank,
            "expected togetherai to support rerank"
        );
    }
}
