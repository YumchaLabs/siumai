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

#[allow(dead_code)]
fn push_unique_model(models: &mut Vec<Cow<'static, str>>, model: Cow<'static, str>) {
    if !models
        .iter()
        .any(|existing| existing.as_ref() == model.as_ref())
    {
        models.push(model);
    }
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
            #[cfg(feature = "azure")]
            ProviderType::Azure => {
                let meta = native_metas
                    .iter()
                    .find(|m| m.id == "azure")
                    .expect("Azure metadata should be registered");
                let mut models: Vec<Cow<'static, str>> = Vec::new();
                if let Some(model) = rec.default_model.clone() {
                    models.push(Cow::Owned(model));
                }
                out.push(ProviderInfo {
                    provider_type: ptype,
                    name: Cow::Borrowed(meta.name),
                    description: Cow::Borrowed(meta.description),
                    capabilities: rec.capabilities.clone(),
                    default_base_url: Cow::Borrowed(meta.default_base_url.unwrap_or("N/A")),
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
            #[cfg(feature = "google-vertex")]
            ProviderType::Vertex => {
                use siumai_provider_google_vertex::providers::vertex::models as vertex_models;

                let meta = native_metas
                    .iter()
                    .find(|m| m.id == crate::provider::ids::VERTEX)
                    .expect("Vertex metadata should be registered");
                let mut models: Vec<Cow<'static, str>> = Vec::new();
                for model in vertex_models::ALL_CHAT
                    .iter()
                    .chain(vertex_models::ALL_EMBEDDING.iter())
                    .chain(vertex_models::ALL_IMAGE.iter())
                    .chain(vertex_models::ALL_VIDEO.iter())
                {
                    push_unique_model(&mut models, Cow::Borrowed(*model));
                }
                out.push(ProviderInfo {
                    provider_type: ptype,
                    name: Cow::Borrowed(meta.name),
                    description: Cow::Borrowed(meta.description),
                    capabilities: rec.capabilities.clone(),
                    default_base_url: Cow::Borrowed("https://aiplatform.googleapis.com/v1"),
                    supported_models: models,
                });
            }
            #[cfg(feature = "google-vertex")]
            ProviderType::AnthropicVertex => {
                use siumai_provider_google_vertex::providers::anthropic_vertex::models as anthropic_vertex_models;

                let meta = native_metas
                    .iter()
                    .find(|m| m.id == crate::provider::ids::ANTHROPIC_VERTEX)
                    .expect("Anthropic Vertex metadata should be registered");
                let mut models: Vec<Cow<'static, str>> = Vec::new();
                for model in anthropic_vertex_models::ALL_CHAT.iter() {
                    push_unique_model(&mut models, Cow::Borrowed(*model));
                }
                out.push(ProviderInfo {
                    provider_type: ptype,
                    name: Cow::Borrowed(meta.name),
                    description: Cow::Borrowed(meta.description),
                    capabilities: rec.capabilities.clone(),
                    default_base_url: Cow::Borrowed("https://aiplatform.googleapis.com/v1"),
                    supported_models: models,
                });
            }
            #[cfg(feature = "google-vertex")]
            ProviderType::VertexMaas => {
                use siumai_provider_openai_compatible::providers::openai_compatible::vertex_maas as vertex_maas_models;

                let meta = native_metas
                    .iter()
                    .find(|m| m.id == crate::provider::ids::VERTEX_MAAS)
                    .expect("Vertex MaaS metadata should be registered");
                let mut models: Vec<Cow<'static, str>> = Vec::new();
                if let Some(model) = rec.default_model.clone() {
                    push_unique_model(&mut models, Cow::Owned(model));
                }
                for model in vertex_maas_models::ALL_CHAT
                    .iter()
                    .chain(vertex_maas_models::ALL_COMPLETION.iter())
                    .chain(vertex_maas_models::ALL_EMBEDDING.iter())
                {
                    push_unique_model(&mut models, Cow::Borrowed(*model));
                }
                out.push(ProviderInfo {
                    provider_type: ptype,
                    name: Cow::Borrowed(meta.name),
                    description: Cow::Borrowed(meta.description),
                    capabilities: rec.capabilities.clone(),
                    default_base_url: Cow::Borrowed("https://aiplatform.googleapis.com/v1"),
                    supported_models: models,
                });
            }
            #[cfg(feature = "ollama")]
            ProviderType::Ollama => {
                let meta = native_metas
                    .iter()
                    .find(|m| m.id == "ollama")
                    .expect("Ollama metadata should be registered");
                let mut models: Vec<Cow<'static, str>> = Vec::new();
                for model in siumai_provider_ollama::providers::ollama::models::ALL_CHAT {
                    push_unique_model(&mut models, Cow::Borrowed(*model));
                }
                for model in siumai_provider_ollama::providers::ollama::models::ALL_EMBEDDING {
                    push_unique_model(&mut models, Cow::Borrowed(*model));
                }
                out.push(ProviderInfo {
                    provider_type: ptype,
                    name: Cow::Borrowed(meta.name),
                    description: Cow::Borrowed(meta.description),
                    capabilities: rec.capabilities.clone(),
                    default_base_url: Cow::Borrowed(
                        meta.default_base_url.unwrap_or("http://localhost:11434"),
                    ),
                    supported_models: models,
                });
            }
            ProviderType::DeepSeek => {
                #[cfg(feature = "deepseek")]
                {
                    let meta = native_metas
                        .iter()
                        .find(|m| m.id == "deepseek")
                        .expect("DeepSeek metadata should be registered");
                    let mut models: Vec<Cow<'static, str>> = Vec::new();
                    if let Some(m) = rec.default_model.clone() {
                        models.push(Cow::Owned(m));
                    }
                    for model in siumai_provider_deepseek::providers::deepseek::models::ALL_CHAT {
                        push_unique_model(&mut models, Cow::Borrowed(*model));
                    }
                    out.push(ProviderInfo {
                        provider_type: ptype,
                        name: Cow::Borrowed(meta.name),
                        description: Cow::Borrowed(meta.description),
                        capabilities: rec.capabilities.clone(),
                        default_base_url: Cow::Borrowed(
                            meta.default_base_url.unwrap_or("https://api.deepseek.com"),
                        ),
                        supported_models: models,
                    });
                    continue;
                }
                #[cfg(all(not(feature = "deepseek"), feature = "openai"))]
                {
                    if let Some(cfg) = siumai_provider_openai_compatible::providers::openai_compatible::get_provider_config("deepseek") {
                        let mut models: Vec<Cow<'static, str>> = Vec::new();
                        if let Some(m) = cfg.default_model.clone() {
                            models.push(Cow::Owned(m));
                        }
                        for model in [
                            siumai_provider_openai_compatible::providers::openai_compatible::providers::models::deepseek::CHAT,
                            siumai_provider_openai_compatible::providers::openai_compatible::providers::models::deepseek::REASONER,
                        ] {
                            if !models.iter().any(|existing| existing.as_ref() == model) {
                                models.push(Cow::Borrowed(model));
                            }
                        }
                        out.push(ProviderInfo {
                            provider_type: ptype,
                            name: Cow::Owned(cfg.name),
                            description: Cow::Borrowed(
                                "OpenAI-compatible provider with DeepSeek-specific routing",
                            ),
                            capabilities: rec.capabilities.clone(),
                            default_base_url: Cow::Owned(cfg.base_url),
                            supported_models: models,
                        });
                        continue;
                    }
                }
                #[cfg(not(feature = "deepseek"))]
                out.push(ProviderInfo {
                    provider_type: ptype,
                    name: Cow::Owned(rec.name.clone()),
                    description: Cow::Owned(rec.id.clone()),
                    capabilities: rec.capabilities.clone(),
                    default_base_url: Cow::Owned(rec.base_url.unwrap_or_default()),
                    supported_models: Vec::new(),
                });
            }
            #[cfg(feature = "deepinfra")]
            ProviderType::DeepInfra => {
                use siumai_provider_openai_compatible::providers::openai_compatible::deepinfra as deepinfra_models;

                let meta = native_metas
                    .iter()
                    .find(|m| m.id == crate::provider::ids::DEEPINFRA)
                    .expect("DeepInfra metadata should be registered");
                let mut models: Vec<Cow<'static, str>> = Vec::new();
                if let Some(model) = rec.default_model.clone() {
                    push_unique_model(&mut models, Cow::Owned(model));
                }

                for model in deepinfra_models::ALL_CHAT
                    .iter()
                    .chain(deepinfra_models::ALL_COMPLETION.iter())
                    .chain(deepinfra_models::ALL_EMBEDDING.iter())
                    .chain(deepinfra_models::ALL_IMAGE.iter())
                {
                    push_unique_model(&mut models, Cow::Borrowed(*model));
                }

                out.push(ProviderInfo {
                    provider_type: ptype,
                    name: Cow::Borrowed(meta.name),
                    description: Cow::Borrowed(meta.description),
                    capabilities: rec.capabilities.clone(),
                    default_base_url: Cow::Borrowed(
                        meta.default_base_url
                            .unwrap_or("https://api.deepinfra.com/v1"),
                    ),
                    supported_models: models,
                });
            }
            #[cfg(feature = "cohere")]
            ProviderType::Cohere => {
                use siumai_provider_cohere::providers::cohere::models as cohere_models;

                let meta = native_metas
                    .iter()
                    .find(|m| m.id == crate::provider::ids::COHERE)
                    .expect("Cohere metadata should be registered");
                let mut models: Vec<Cow<'static, str>> = Vec::new();
                if let Some(model) = rec.default_model.clone() {
                    push_unique_model(&mut models, Cow::Owned(model));
                }
                for model in cohere_models::ALL_CHAT
                    .iter()
                    .chain(cohere_models::ALL_EMBEDDING.iter())
                    .chain(cohere_models::ALL_RERANK.iter())
                {
                    push_unique_model(&mut models, Cow::Borrowed(*model));
                }

                out.push(ProviderInfo {
                    provider_type: ptype,
                    name: Cow::Borrowed(meta.name),
                    description: Cow::Borrowed(meta.description),
                    capabilities: rec.capabilities.clone(),
                    default_base_url: Cow::Borrowed(
                        meta.default_base_url.unwrap_or("https://api.cohere.com/v2"),
                    ),
                    supported_models: models,
                });
            }
            #[cfg(feature = "togetherai")]
            ProviderType::TogetherAi => {
                use siumai_provider_togetherai::providers::togetherai::models as togetherai_models;

                let meta = native_metas
                    .iter()
                    .find(|m| m.id == crate::provider::ids::TOGETHERAI)
                    .expect("TogetherAI metadata should be registered");
                let mut models: Vec<Cow<'static, str>> = Vec::new();
                if let Some(model) = rec.default_model.clone() {
                    push_unique_model(&mut models, Cow::Owned(model));
                }
                for model in [
                    siumai_provider_openai_compatible::providers::openai_compatible::default_models::get_default_embedding_model(
                        crate::provider::ids::TOGETHERAI,
                    ),
                    siumai_provider_openai_compatible::providers::openai_compatible::default_models::get_default_image_model(
                        crate::provider::ids::TOGETHERAI,
                    ),
                    siumai_provider_openai_compatible::providers::openai_compatible::default_models::get_default_speech_model(
                        crate::provider::ids::TOGETHERAI,
                    ),
                    siumai_provider_openai_compatible::providers::openai_compatible::default_models::get_default_transcription_model(
                        crate::provider::ids::TOGETHERAI,
                    ),
                    Some("Salesforce/Llama-Rank-v1"),
                ]
                .into_iter()
                .flatten()
                {
                    push_unique_model(&mut models, Cow::Borrowed(model));
                }
                for model in togetherai_models::ALL_CHAT
                    .iter()
                    .chain(togetherai_models::ALL_COMPLETION.iter())
                    .chain(togetherai_models::ALL_EMBEDDING.iter())
                    .chain(togetherai_models::ALL_IMAGE.iter())
                    .chain(togetherai_models::ALL_RERANK.iter())
                {
                    push_unique_model(&mut models, Cow::Borrowed(*model));
                }

                out.push(ProviderInfo {
                    provider_type: ptype,
                    name: Cow::Borrowed(meta.name),
                    description: Cow::Borrowed(meta.description),
                    capabilities: rec.capabilities.clone(),
                    default_base_url: Cow::Borrowed(
                        meta.default_base_url
                            .unwrap_or("https://api.together.xyz/v1"),
                    ),
                    supported_models: models,
                });
            }
            #[cfg(feature = "openai")]
            ProviderType::Mistral => {
                use siumai_provider_openai_compatible::providers::openai_compatible::mistral as mistral_models;

                let mut models: Vec<Cow<'static, str>> = Vec::new();
                if let Some(model) = rec.default_model.clone() {
                    push_unique_model(&mut models, Cow::Owned(model));
                }

                for model in mistral_models::ALL_CHAT
                    .iter()
                    .chain(mistral_models::ALL_EMBEDDING.iter())
                {
                    push_unique_model(&mut models, Cow::Borrowed(*model));
                }

                out.push(ProviderInfo {
                    provider_type: ptype,
                    name: Cow::Owned(rec.name.clone()),
                    description: Cow::Borrowed(
                        "Mistral AI provider surface via OpenAI-compatible chat and embedding endpoints",
                    ),
                    capabilities: rec.capabilities.clone(),
                    default_base_url: Cow::Owned(
                        rec.base_url
                            .clone()
                            .unwrap_or_else(|| "https://api.mistral.ai/v1".to_string()),
                    ),
                    supported_models: models,
                });
            }
            #[cfg(feature = "openai")]
            ProviderType::Fireworks => {
                use siumai_provider_openai_compatible::providers::openai_compatible::fireworks as fireworks_models;

                let mut models: Vec<Cow<'static, str>> = Vec::new();
                if let Some(model) = rec.default_model.clone() {
                    push_unique_model(&mut models, Cow::Owned(model));
                }
                push_unique_model(&mut models, Cow::Borrowed("whisper-v3"));

                for model in fireworks_models::ALL_CHAT
                    .iter()
                    .chain(fireworks_models::ALL_COMPLETION.iter())
                    .chain(fireworks_models::ALL_EMBEDDING.iter())
                    .chain(fireworks_models::ALL_IMAGE.iter())
                {
                    push_unique_model(&mut models, Cow::Borrowed(*model));
                }

                out.push(ProviderInfo {
                    provider_type: ptype,
                    name: Cow::Owned(rec.name.clone()),
                    description: Cow::Borrowed(
                        "Fireworks AI unified provider surface via OpenAI-compatible chat, completion, embedding, and transcription endpoints plus provider-owned image generation and edit workflows",
                    ),
                    capabilities: rec.capabilities.clone(),
                    default_base_url: Cow::Owned(
                        rec.base_url.clone().unwrap_or_else(|| {
                            "https://api.fireworks.ai/inference/v1".to_string()
                        }),
                    ),
                    supported_models: models,
                });
            }
            #[cfg(feature = "openai")]
            ProviderType::Perplexity => {
                use siumai_provider_openai_compatible::providers::openai_compatible::perplexity as perplexity_models;

                let mut models: Vec<Cow<'static, str>> = Vec::new();
                if let Some(model) = rec.default_model.clone() {
                    push_unique_model(&mut models, Cow::Owned(model));
                }
                for model in perplexity_models::ALL_CHAT {
                    push_unique_model(&mut models, Cow::Borrowed(*model));
                }

                out.push(ProviderInfo {
                    provider_type: ptype,
                    name: Cow::Owned(rec.name.clone()),
                    description: Cow::Borrowed(
                        "Perplexity language models on the hosted-search OpenAI-compatible chat surface",
                    ),
                    capabilities: rec.capabilities.clone(),
                    default_base_url: Cow::Owned(
                        rec.base_url
                            .clone()
                            .unwrap_or_else(|| "https://api.perplexity.ai".to_string()),
                    ),
                    supported_models: models,
                });
            }
            #[cfg(feature = "bedrock")]
            ProviderType::Bedrock => {
                let meta = native_metas
                    .iter()
                    .find(|m| m.id == crate::provider::ids::BEDROCK)
                    .expect("Bedrock metadata should be registered");
                let mut models: Vec<Cow<'static, str>> = Vec::new();
                if let Some(model) = rec.default_model.clone() {
                    models.push(Cow::Owned(model));
                }

                out.push(ProviderInfo {
                    provider_type: ptype,
                    name: Cow::Borrowed(meta.name),
                    description: Cow::Borrowed(meta.description),
                    capabilities: rec.capabilities.clone(),
                    default_base_url: Cow::Borrowed(meta.default_base_url.unwrap_or("N/A")),
                    supported_models: models,
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
                let mut models: Vec<Cow<'static, str>> = Vec::new();
                if let Some(model) = rec.default_model.clone() {
                    push_unique_model(&mut models, Cow::Owned(model));
                }
                for model in siumai_provider_minimaxi::providers::minimaxi::models::ALL_CHAT {
                    push_unique_model(&mut models, Cow::Borrowed(*model));
                }
                for model in siumai_provider_minimaxi::providers::minimaxi::models::ALL_SPEECH {
                    push_unique_model(&mut models, Cow::Borrowed(*model));
                }
                for model in siumai_provider_minimaxi::providers::minimaxi::models::ALL_VIDEO {
                    push_unique_model(&mut models, Cow::Borrowed(*model));
                }
                for model in siumai_provider_minimaxi::providers::minimaxi::models::ALL_MUSIC {
                    push_unique_model(&mut models, Cow::Borrowed(*model));
                }
                for model in siumai_provider_minimaxi::providers::minimaxi::models::ALL_IMAGE {
                    push_unique_model(&mut models, Cow::Borrowed(*model));
                }
                out.push(ProviderInfo {
                    provider_type: ptype,
                    name: Cow::Borrowed(meta.name),
                    description: Cow::Borrowed(meta.description),
                    capabilities: rec.capabilities.clone(),
                    default_base_url: Cow::Borrowed(
                        meta.default_base_url
                            .unwrap_or("https://api.minimaxi.com/v1"),
                    ),
                    supported_models: models,
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
                // (e.g. "openrouter", "siliconflow"). Keep this catalog
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
                            "moonshotai" => {
                                models.push(Cow::Borrowed(oai_models::moonshotai::KIMI_K2_0905));
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
    #[test]
    #[cfg(feature = "openai")]
    fn provider_catalog_includes_openai_compatible_providers() {
        let providers = super::get_supported_providers();
        assert!(
            providers
                .iter()
                .any(|p| p.provider_type == super::ProviderType::DeepSeek),
            "expected openai-compatible provider 'deepseek' to be listed"
        );
    }

    #[test]
    #[cfg(feature = "openai")]
    fn provider_catalog_lookup_by_id_works_for_openai_compatible() {
        let info = super::get_provider_info_by_id("deepseek").expect("deepseek should exist");
        assert_eq!(info.provider_type, super::ProviderType::DeepSeek);
        assert!(
            info.supported_models
                .iter()
                .any(|m| m.as_ref() == "deepseek-chat"),
            "expected deepseek model catalog entry to include deepseek-chat"
        );
    }

    #[test]
    #[cfg(feature = "openai")]
    fn provider_catalog_keeps_custom_openai_compatible_variants() {
        let info = super::get_provider_info_by_id("openrouter").expect("openrouter should exist");
        assert!(
            matches!(info.provider_type, super::ProviderType::Custom(id) if id == "openrouter")
        );
    }

    #[test]
    #[cfg(feature = "deepseek")]
    fn provider_catalog_uses_native_metadata_for_deepseek() {
        let info = super::get_provider_info_by_id("deepseek").expect("deepseek should exist");
        assert_eq!(info.provider_type, super::ProviderType::DeepSeek);
        assert_eq!(info.name.as_ref(), "DeepSeek");
        assert_ne!(info.description.as_ref(), "Custom provider");
        assert!(
            info.capabilities.chat && info.capabilities.streaming && info.capabilities.tools,
            "expected deepseek to expose text/tool capabilities"
        );
        assert!(
            !info.capabilities.embedding
                && !info.capabilities.image_generation
                && !info.capabilities.rerank,
            "expected deepseek non-text capabilities to remain deferred"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|m| m.as_ref() == siumai_provider_deepseek::providers::deepseek::models::CHAT),
            "expected deepseek default model to be listed"
        );
    }

    #[test]
    #[cfg(feature = "ollama")]
    fn provider_catalog_uses_native_metadata_for_ollama() {
        let info = super::get_provider_info_by_id("ollama").expect("ollama should exist");
        assert_eq!(info.provider_type, super::ProviderType::Ollama);
        assert_eq!(info.name.as_ref(), "Ollama");
        assert_ne!(info.description.as_ref(), "Custom provider");
        assert!(info.capabilities.chat && info.capabilities.embedding);
        assert!(
            info.supported_models
                .iter()
                .any(|m| { m.as_ref() == siumai_provider_ollama::providers::ollama::models::CHAT }),
            "expected curated Ollama chat default to be listed"
        );
        assert!(
            info.supported_models.iter().any(|m| {
                m.as_ref() == siumai_provider_ollama::providers::ollama::models::EMBEDDING
            }),
            "expected curated Ollama embedding default to be listed"
        );
    }

    #[test]
    #[cfg(feature = "minimaxi")]
    fn provider_catalog_uses_native_metadata_for_minimaxi() {
        let info = super::get_provider_info_by_id("minimaxi").expect("minimaxi should exist");
        assert_eq!(info.provider_type, super::ProviderType::MiniMaxi);
        assert_eq!(info.name.as_ref(), "MiniMaxi");
        assert_ne!(info.description.as_ref(), "Custom provider");
        assert!(
            info.capabilities.chat
                && info.capabilities.speech
                && info.capabilities.image_generation
                && info.capabilities.file_management,
            "expected minimaxi to expose its unified multimodal capability surface"
        );
        assert!(
            info.supported_models.iter().any(|m| {
                m.as_ref() == siumai_provider_minimaxi::providers::minimaxi::models::CHAT
            }),
            "expected curated MiniMaxi chat default to be listed"
        );
        assert!(
            info.supported_models.iter().any(|m| {
                m.as_ref() == siumai_provider_minimaxi::providers::minimaxi::models::IMAGE
            }),
            "expected curated MiniMaxi image default to be listed"
        );
    }

    #[test]
    #[cfg(feature = "deepinfra")]
    fn provider_catalog_uses_native_metadata_for_deepinfra() {
        let info = super::get_provider_info_by_id("deepinfra").expect("deepinfra should exist");
        assert_eq!(info.provider_type, super::ProviderType::DeepInfra);
        assert_eq!(info.name.as_ref(), "DeepInfra");
        assert_ne!(info.description.as_ref(), "Custom provider");
        assert!(
            info.capabilities.chat
                && info.capabilities.completion
                && info.capabilities.embedding
                && info.capabilities.image_generation,
            "expected deepinfra to expose unified text/embedding/image capabilities"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|m| m.as_ref() == "meta-llama/Llama-3.3-70B-Instruct"),
            "expected deepinfra default chat model to be listed"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|m| m.as_ref() == "BAAI/bge-base-en-v1.5"),
            "expected deepinfra default embedding model to be listed"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|m| m.as_ref() == "black-forest-labs/FLUX-1-schnell"),
            "expected deepinfra default image model to be listed"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|m| m.as_ref() == "meta-llama/Meta-Llama-3.1-405B-Instruct"),
            "expected curated deepinfra chat models to be listed"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|m| m.as_ref() == "black-forest-labs/FLUX.1-Kontext-pro"),
            "expected curated deepinfra image models to be listed"
        );
    }

    #[test]
    #[cfg(feature = "azure")]
    fn provider_catalog_uses_native_metadata_for_azure() {
        let info = super::get_provider_info_by_id("azure").expect("azure should exist");
        assert_eq!(info.provider_type, super::ProviderType::Azure);
        assert_eq!(info.name.as_ref(), "Azure OpenAI");
        assert_ne!(info.description.as_ref(), "Custom provider");
        assert!(
            info.capabilities.chat
                && info.capabilities.completion
                && info.capabilities.streaming
                && info.capabilities.tools
                && info.capabilities.embedding
                && info.capabilities.image_generation
                && info.capabilities.audio
                && info.capabilities.file_management,
            "expected azure to expose the native Azure OpenAI family capabilities"
        );
    }

    #[test]
    #[cfg(feature = "azure")]
    fn provider_catalog_lookup_by_id_maps_azure_chat_to_azure_family() {
        let info = super::get_provider_info_by_id("azure-chat").expect("azure-chat should resolve");
        assert_eq!(info.provider_type, super::ProviderType::Azure);
        assert_eq!(info.name.as_ref(), "Azure OpenAI");
    }

    #[test]
    #[cfg(feature = "openai")]
    fn provider_catalog_lookup_by_id_maps_openai_family_variants() {
        let chat =
            super::get_provider_info_by_id("openai-chat").expect("openai-chat should resolve");
        assert_eq!(chat.provider_type, super::ProviderType::OpenAi);
        assert_eq!(chat.name.as_ref(), "OpenAI");

        let responses = super::get_provider_info_by_id("openai-responses")
            .expect("openai-responses should resolve");
        assert_eq!(responses.provider_type, super::ProviderType::OpenAi);
        assert_eq!(responses.name.as_ref(), "OpenAI");
    }

    #[test]
    #[cfg(feature = "openai")]
    fn provider_catalog_lookup_by_id_maps_mistral_to_first_class_provider_type() {
        let info = super::get_provider_info_by_id("mistral").expect("mistral should resolve");
        assert_eq!(info.provider_type, super::ProviderType::Mistral);
        assert_eq!(info.name.as_ref(), "Mistral AI");
        assert!(
            info.supported_models
                .iter()
                .any(|model| model.as_ref() == "mistral-large-latest"),
            "expected mistral chat default to be listed"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|model| model.as_ref() == "mistral-small-latest"),
            "expected curated mistral reasoning-capable model to be listed"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|model| model.as_ref() == "mistral-embed"),
            "expected mistral embedding default to be listed"
        );
    }

    #[test]
    #[cfg(feature = "openai")]
    fn provider_catalog_lookup_by_id_maps_fireworks_to_first_class_provider_type() {
        let info = super::get_provider_info_by_id("fireworks").expect("fireworks should resolve");
        assert_eq!(info.provider_type, super::ProviderType::Fireworks);
        assert_eq!(info.name.as_ref(), "Fireworks AI");
        assert!(
            info.supported_models
                .iter()
                .any(|model| model.as_ref() == "accounts/fireworks/models/llama-v3p1-8b-instruct"),
            "expected fireworks chat default to be listed"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|model| model.as_ref() == "accounts/fireworks/models/deepseek-v3"),
            "expected curated fireworks chat model subset to be listed"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|model| model.as_ref() == "accounts/fireworks/models/llama-v3-8b-instruct"),
            "expected curated fireworks completion model subset to be listed"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|model| model.as_ref() == "nomic-ai/nomic-embed-text-v1.5"),
            "expected fireworks embedding default to be listed"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|model| model.as_ref() == "whisper-v3"),
            "expected fireworks transcription default to be listed"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|model| { model.as_ref() == "accounts/fireworks/models/flux-1-dev-fp8" }),
            "expected fireworks image default to be listed"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|model| { model.as_ref() == "accounts/fireworks/models/flux-kontext-pro" }),
            "expected fireworks editing image model to be listed"
        );
    }

    #[test]
    #[cfg(feature = "openai")]
    fn provider_catalog_lookup_by_id_maps_perplexity_to_first_class_provider_type() {
        let info = super::get_provider_info_by_id("perplexity").expect("perplexity should resolve");
        assert_eq!(info.provider_type, super::ProviderType::Perplexity);
        assert_eq!(info.name.as_ref(), "Perplexity");
        assert!(
            info.supported_models
                .iter()
                .any(|model| { model.as_ref() == "sonar" }),
            "expected perplexity default model to be listed"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|model| { model.as_ref() == "sonar-deep-research" }),
            "expected curated perplexity research model to be listed"
        );
    }

    #[test]
    #[cfg(feature = "google-vertex")]
    fn provider_catalog_uses_native_metadata_for_vertex() {
        let info =
            super::get_provider_info_by_id("google-vertex").expect("google-vertex should exist");
        assert_eq!(info.provider_type, super::ProviderType::Vertex);
        assert_eq!(info.name.as_ref(), "Google Vertex AI");
        assert_ne!(info.description.as_ref(), "Custom provider");
        assert!(
            info.capabilities.chat
                && info.capabilities.streaming
                && info.capabilities.embedding
                && info.capabilities.image_generation
                && info.capabilities.supports("video"),
            "expected vertex to expose unified chat/embedding/image/video capabilities"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|m| m.as_ref() == "gemini-2.5-flash"),
            "expected vertex chat model ids to be listed"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|m| m.as_ref() == "text-embedding-005"),
            "expected vertex embedding model ids to be listed"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|m| m.as_ref() == "imagen-4.0-generate-001"),
            "expected vertex image model ids to be listed"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|m| m.as_ref() == "imagen-4.0-ultra-generate-001"),
            "expected vertex ultra image model ids to be listed"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|m| m.as_ref() == "gemini-2.5-flash-image"),
            "expected vertex Gemini image model ids to be listed"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|m| m.as_ref() == "imagen-3.0-edit-001"),
            "expected vertex curated image edit model ids to be listed"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|m| m.as_ref() == "veo-3.1-generate-preview"),
            "expected vertex curated video model ids to be listed"
        );
    }

    #[test]
    #[cfg(feature = "google-vertex")]
    fn provider_catalog_uses_native_metadata_for_anthropic_vertex() {
        let info = super::get_provider_info_by_id("google-vertex-anthropic")
            .expect("google-vertex-anthropic should exist");
        assert_eq!(info.provider_type, super::ProviderType::AnthropicVertex);
        assert_eq!(info.name.as_ref(), "Anthropic on Vertex");
        assert_ne!(info.description.as_ref(), "Custom provider");
        assert!(
            info.capabilities.chat && info.capabilities.streaming && info.capabilities.tools,
            "expected anthropic-vertex to expose Anthropic-style chat capabilities"
        );
        assert!(
            !info.capabilities.completion
                && !info.capabilities.embedding
                && !info.capabilities.image_generation,
            "expected anthropic-vertex non-Anthropic capability flags to remain false"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|m| m.as_ref() == "claude-sonnet-4-6"),
            "expected anthropic-vertex curated model ids to be listed"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|m| m.as_ref() == "claude-3-5-sonnet-v2@20241022"),
            "expected anthropic-vertex curated fallback model ids to be listed"
        );
    }

    #[test]
    #[cfg(feature = "google-vertex")]
    fn provider_catalog_uses_native_metadata_for_vertex_maas() {
        let info = super::get_provider_info_by_id("vertex.maas").expect("vertex.maas should exist");
        assert_eq!(info.provider_type, super::ProviderType::VertexMaas);
        assert_eq!(info.name.as_ref(), "Google Vertex MaaS");
        assert_ne!(info.description.as_ref(), "Custom provider");
        assert!(
            info.capabilities.chat
                && info.capabilities.completion
                && info.capabilities.embedding
                && info.capabilities.tools,
            "expected vertex-maas to expose the OpenAI-compatible text/completion/embedding surface"
        );
        assert!(
            !info.capabilities.image_generation
                && !info.capabilities.rerank
                && !info.capabilities.speech
                && !info.capabilities.transcription,
            "expected vertex-maas non-text capabilities to remain deferred"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|m| m.as_ref() == "deepseek-ai/deepseek-v3.2-maas"),
            "expected vertex-maas curated model ids to be listed"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|m| m.as_ref() == "qwen/qwen3-next-80b-a3b-thinking-maas"),
            "expected vertex-maas curated model ids to include the audited thinking variant"
        );
    }

    #[test]
    #[cfg(feature = "cohere")]
    fn provider_catalog_uses_native_metadata_for_cohere() {
        let info = super::get_provider_info_by_id("cohere").expect("cohere should exist");
        assert_eq!(info.provider_type, super::ProviderType::Cohere);
        assert_eq!(info.name.as_ref(), "Cohere");
        assert_ne!(info.description.as_ref(), "Custom provider");
        assert!(
            info.capabilities.chat
                && info.capabilities.streaming
                && info.capabilities.tools
                && info.capabilities.embedding
                && info.capabilities.rerank,
            "expected cohere to expose the unified AI SDK-style provider surface"
        );
        assert!(
            !info.capabilities.completion
                && !info.capabilities.image_generation
                && !info.capabilities.speech
                && !info.capabilities.transcription
                && !info.capabilities.audio,
            "expected cohere non-chat/embed/rerank capability flags to remain deferred"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|m| m.as_ref() == "command-a-03-2025"),
            "expected cohere chat models to be listed"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|m| m.as_ref() == "embed-v4.0"),
            "expected cohere embedding models to be listed"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|m| m.as_ref() == "rerank-v3.5"),
            "expected cohere rerank models to be listed"
        );
    }

    #[test]
    #[cfg(feature = "togetherai")]
    fn provider_catalog_uses_native_metadata_for_togetherai() {
        let info = super::get_provider_info_by_id("togetherai").expect("togetherai should exist");
        assert_eq!(info.provider_type, super::ProviderType::TogetherAi);
        assert_eq!(info.name.as_ref(), "TogetherAI");
        assert_ne!(info.description.as_ref(), "Custom provider");
        assert!(
            info.capabilities.rerank
                && info.capabilities.chat
                && info.capabilities.completion
                && info.capabilities.embedding
                && info.capabilities.image_generation
                && info.capabilities.speech
                && info.capabilities.transcription
                && info.capabilities.audio,
            "expected togetherai to expose the unified AI SDK-style provider surface"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|m| m.as_ref() == "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),
            "expected togetherai chat default model to be listed"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|m| m.as_ref() == "Salesforce/Llama-Rank-v1"),
            "expected togetherai rerank default model to be listed"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|m| m.as_ref() == "Qwen/Qwen2.5-Coder-32B-Instruct"),
            "expected togetherai completion models to be listed"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|m| m.as_ref() == "togethercomputer/m2-bert-80M-8k-retrieval"),
            "expected togetherai embedding default model to be listed"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|m| m.as_ref() == "black-forest-labs/FLUX.1-schnell"),
            "expected togetherai image default model to be listed"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|m| m.as_ref() == "black-forest-labs/FLUX.1-kontext-pro"),
            "expected togetherai curated image edit models to be listed"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|m| m.as_ref() == "cartesia/sonic-2"),
            "expected togetherai speech default model to be listed"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|m| m.as_ref() == "openai/whisper-large-v3"),
            "expected togetherai transcription default model to be listed"
        );
        assert!(
            info.supported_models
                .iter()
                .any(|m| m.as_ref() == "mixedbread-ai/Mxbai-Rerank-Large-V2"),
            "expected togetherai curated rerank models to be listed"
        );
    }

    #[test]
    #[cfg(feature = "bedrock")]
    fn provider_catalog_uses_native_metadata_for_bedrock() {
        let info = super::get_provider_info_by_id("bedrock").expect("bedrock should exist");
        assert_eq!(info.provider_type, super::ProviderType::Bedrock);
        assert_eq!(info.name.as_ref(), "Amazon Bedrock");
        assert_ne!(info.description.as_ref(), "Custom provider");
        assert!(
            info.capabilities.chat
                && info.capabilities.streaming
                && info.capabilities.tools
                && info.capabilities.embedding
                && info.capabilities.image_generation
                && info.capabilities.rerank,
            "expected bedrock to expose chat/embedding/image/rerank capabilities"
        );
        assert!(
            !info.capabilities.speech && !info.capabilities.transcription,
            "expected bedrock non-audited audio capabilities to remain deferred"
        );
        assert!(
            info.supported_models.is_empty(),
            "expected bedrock catalog to avoid inventing model lists"
        );
    }
}
