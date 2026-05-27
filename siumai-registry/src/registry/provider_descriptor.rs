//! Built-in provider descriptor seam.
//!
//! This module owns the registry-facing provider facts that otherwise have to be
//! remembered by helpers, default-model lookup, and built-in factory registration.

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::LlmError;
use crate::provider::ids;
use crate::registry::entry::ProviderFactory;

#[allow(dead_code)]
fn unsupported_provider_feature(provider_name: &str, feature: &str) -> LlmError {
    LlmError::UnsupportedOperation(format!(
        "{provider_name} provider requires the '{feature}' feature to be enabled"
    ))
}

#[cfg(not(feature = "openai"))]
fn unsupported_openai_compatible_provider(provider_id: &str) -> LlmError {
    LlmError::UnsupportedOperation(format!(
        "OpenAI-compatible provider '{provider_id}' requires the 'openai' feature to be enabled"
    ))
}

/// Resolve the public compatibility default model for a provider id.
pub(crate) fn builtin_provider_default_model(provider_id: &str) -> Result<String, LlmError> {
    let normalized = crate::provider::resolver::normalize_provider_id(provider_id);
    let native_policy_id = if ids::is_openai_family(&normalized) {
        ids::OPENAI
    } else if ids::is_azure_family(&normalized) {
        ids::AZURE
    } else {
        normalized.as_str()
    };

    if let Some(policy) =
        crate::native_provider_metadata::native_provider_default_model_policy(native_policy_id)
    {
        if let Some(model) = policy.default_model() {
            return Ok(model.to_string());
        }
        if let Some(message) = policy.explicit_required_message() {
            return Err(LlmError::ConfigurationError(message.to_string()));
        }
    }

    #[cfg(feature = "openai")]
    {
        if let Some(model) =
            siumai_provider_openai_compatible::providers::openai_compatible::default_models::get_default_chat_model(
                &normalized,
            )
        {
            return Ok(model.to_string());
        }
    }

    if ids::BuiltinProviderId::parse(&normalized).is_some() {
        let _ = builtin_provider_factory(&normalized)?;
    }

    #[cfg(not(feature = "openai"))]
    {
        if ids::BuiltinProviderId::parse(&normalized).is_none() {
            return Err(unsupported_openai_compatible_provider(&normalized));
        }
    }

    Err(LlmError::ConfigurationError(format!(
        "Provider '{normalized}' requires an explicit model id"
    )))
}

/// Resolve an OpenAI-compatible provider id into a registry factory.
pub(crate) fn openai_compatible_provider_factory(
    provider_id: &str,
) -> Result<Arc<dyn ProviderFactory>, LlmError> {
    let normalized = crate::provider::resolver::normalize_provider_id(provider_id);
    #[cfg(feature = "openai")]
    {
        Ok(Arc::new(
            crate::registry::factories::OpenAICompatibleProviderFactory::new(
                normalized.to_string(),
            ),
        ) as Arc<dyn ProviderFactory>)
    }

    #[cfg(not(feature = "openai"))]
    {
        Err(unsupported_openai_compatible_provider(&normalized))
    }
}

fn insert_builtin_provider_factory(
    providers: &mut HashMap<String, Arc<dyn ProviderFactory>>,
    provider_id: &str,
) -> Result<(), LlmError> {
    providers.insert(
        provider_id.to_string(),
        builtin_provider_factory(provider_id)?,
    );
    Ok(())
}

/// Resolve a built-in provider id into its registry factory.
pub(crate) fn builtin_provider_factory(
    provider_id: &str,
) -> Result<Arc<dyn ProviderFactory>, LlmError> {
    let normalized = crate::provider::resolver::normalize_provider_id(provider_id);

    match ids::BuiltinProviderId::parse(&normalized) {
        Some(
            ids::BuiltinProviderId::OpenAi
            | ids::BuiltinProviderId::OpenAiChat
            | ids::BuiltinProviderId::OpenAiResponses,
        ) => {
            #[cfg(feature = "openai")]
            {
                Ok(Arc::new(crate::registry::factories::OpenAIProviderFactory)
                    as Arc<dyn ProviderFactory>)
            }
            #[cfg(not(feature = "openai"))]
            {
                Err(unsupported_provider_feature("OpenAI", "openai"))
            }
        }
        Some(ids::BuiltinProviderId::Anthropic) => {
            #[cfg(feature = "anthropic")]
            {
                Ok(
                    Arc::new(crate::registry::factories::AnthropicProviderFactory)
                        as Arc<dyn ProviderFactory>,
                )
            }
            #[cfg(not(feature = "anthropic"))]
            {
                Err(unsupported_provider_feature("Anthropic", "anthropic"))
            }
        }
        Some(ids::BuiltinProviderId::AnthropicVertex) => {
            #[cfg(feature = "google-vertex")]
            {
                Ok(
                    Arc::new(crate::registry::factories::AnthropicVertexProviderFactory)
                        as Arc<dyn ProviderFactory>,
                )
            }
            #[cfg(not(feature = "google-vertex"))]
            {
                Err(unsupported_provider_feature(
                    "Anthropic on Vertex",
                    "google-vertex",
                ))
            }
        }
        Some(ids::BuiltinProviderId::Gemini) => {
            #[cfg(feature = "google")]
            {
                Ok(Arc::new(crate::registry::factories::GeminiProviderFactory)
                    as Arc<dyn ProviderFactory>)
            }
            #[cfg(not(feature = "google"))]
            {
                Err(unsupported_provider_feature("Gemini", "google"))
            }
        }
        Some(ids::BuiltinProviderId::Vertex) => {
            #[cfg(feature = "google-vertex")]
            {
                Ok(
                    Arc::new(crate::registry::factories::GoogleVertexProviderFactory)
                        as Arc<dyn ProviderFactory>,
                )
            }
            #[cfg(not(feature = "google-vertex"))]
            {
                Err(unsupported_provider_feature(
                    "Google Vertex",
                    "google-vertex",
                ))
            }
        }
        Some(ids::BuiltinProviderId::VertexMaas) => {
            #[cfg(feature = "google-vertex")]
            {
                Ok(
                    Arc::new(crate::registry::factories::VertexMaasProviderFactory)
                        as Arc<dyn ProviderFactory>,
                )
            }
            #[cfg(not(feature = "google-vertex"))]
            {
                Err(unsupported_provider_feature(
                    "Google Vertex MaaS",
                    "google-vertex",
                ))
            }
        }
        Some(ids::BuiltinProviderId::GoogleVertexXai) => {
            #[cfg(feature = "google-vertex")]
            {
                Ok(
                    Arc::new(crate::registry::factories::GoogleVertexXaiProviderFactory)
                        as Arc<dyn ProviderFactory>,
                )
            }
            #[cfg(not(feature = "google-vertex"))]
            {
                Err(unsupported_provider_feature(
                    "Google Vertex xAI",
                    "google-vertex",
                ))
            }
        }
        Some(ids::BuiltinProviderId::Ollama) => {
            #[cfg(feature = "ollama")]
            {
                Ok(Arc::new(crate::registry::factories::OllamaProviderFactory)
                    as Arc<dyn ProviderFactory>)
            }
            #[cfg(not(feature = "ollama"))]
            {
                Err(unsupported_provider_feature("Ollama", "ollama"))
            }
        }
        Some(ids::BuiltinProviderId::DeepSeek) => {
            #[cfg(feature = "deepseek")]
            {
                Ok(
                    Arc::new(crate::registry::factories::DeepSeekProviderFactory)
                        as Arc<dyn ProviderFactory>,
                )
            }
            #[cfg(all(not(feature = "deepseek"), feature = "openai"))]
            {
                openai_compatible_provider_factory(ids::DEEPSEEK)
            }
            #[cfg(all(not(feature = "deepseek"), not(feature = "openai")))]
            {
                Err(unsupported_provider_feature("DeepSeek", "deepseek"))
            }
        }
        Some(ids::BuiltinProviderId::DeepInfra) => {
            #[cfg(feature = "deepinfra")]
            {
                Ok(
                    Arc::new(crate::registry::factories::DeepInfraProviderFactory)
                        as Arc<dyn ProviderFactory>,
                )
            }
            #[cfg(all(not(feature = "deepinfra"), feature = "openai"))]
            {
                openai_compatible_provider_factory(ids::DEEPINFRA)
            }
            #[cfg(all(not(feature = "deepinfra"), not(feature = "openai")))]
            {
                Err(unsupported_provider_feature("DeepInfra", "deepinfra"))
            }
        }
        Some(ids::BuiltinProviderId::Fireworks) => {
            #[cfg(feature = "openai")]
            {
                Ok(
                    Arc::new(crate::registry::factories::FireworksProviderFactory)
                        as Arc<dyn ProviderFactory>,
                )
            }
            #[cfg(not(feature = "openai"))]
            {
                Err(unsupported_openai_compatible_provider(ids::FIREWORKS))
            }
        }
        Some(ids::BuiltinProviderId::Cerebras) => {
            #[cfg(feature = "openai")]
            {
                openai_compatible_provider_factory(ids::CEREBRAS)
            }
            #[cfg(not(feature = "openai"))]
            {
                Err(unsupported_openai_compatible_provider(ids::CEREBRAS))
            }
        }
        Some(ids::BuiltinProviderId::Xai) => {
            #[cfg(feature = "xai")]
            {
                Ok(Arc::new(crate::registry::factories::XAIProviderFactory)
                    as Arc<dyn ProviderFactory>)
            }
            #[cfg(all(not(feature = "xai"), feature = "openai"))]
            {
                openai_compatible_provider_factory(ids::XAI)
            }
            #[cfg(all(not(feature = "xai"), not(feature = "openai")))]
            {
                Err(unsupported_provider_feature("xAI", "xai"))
            }
        }
        Some(ids::BuiltinProviderId::Groq) => {
            #[cfg(feature = "groq")]
            {
                Ok(Arc::new(crate::registry::factories::GroqProviderFactory)
                    as Arc<dyn ProviderFactory>)
            }
            #[cfg(all(not(feature = "groq"), feature = "openai"))]
            {
                openai_compatible_provider_factory(ids::GROQ)
            }
            #[cfg(all(not(feature = "groq"), not(feature = "openai")))]
            {
                Err(unsupported_provider_feature("Groq", "groq"))
            }
        }
        Some(ids::BuiltinProviderId::MiniMaxi) => {
            #[cfg(feature = "minimaxi")]
            {
                Ok(
                    Arc::new(crate::registry::factories::MiniMaxiProviderFactory)
                        as Arc<dyn ProviderFactory>,
                )
            }
            #[cfg(all(not(feature = "minimaxi"), feature = "openai"))]
            {
                openai_compatible_provider_factory(ids::MINIMAXI)
            }
            #[cfg(all(not(feature = "minimaxi"), not(feature = "openai")))]
            {
                Err(unsupported_provider_feature("MiniMaxi", "minimaxi"))
            }
        }
        Some(ids::BuiltinProviderId::Cohere) => {
            #[cfg(feature = "cohere")]
            {
                Ok(Arc::new(crate::registry::factories::CohereProviderFactory)
                    as Arc<dyn ProviderFactory>)
            }
            #[cfg(all(not(feature = "cohere"), feature = "openai"))]
            {
                openai_compatible_provider_factory(ids::COHERE)
            }
            #[cfg(all(not(feature = "cohere"), not(feature = "openai")))]
            {
                Err(unsupported_provider_feature("Cohere", "cohere"))
            }
        }
        Some(ids::BuiltinProviderId::TogetherAi) => {
            #[cfg(feature = "togetherai")]
            {
                Ok(
                    Arc::new(crate::registry::factories::TogetherAiProviderFactory)
                        as Arc<dyn ProviderFactory>,
                )
            }
            #[cfg(all(not(feature = "togetherai"), feature = "openai"))]
            {
                openai_compatible_provider_factory(ids::TOGETHERAI)
            }
            #[cfg(all(not(feature = "togetherai"), not(feature = "openai")))]
            {
                Err(unsupported_provider_feature("TogetherAI", "togetherai"))
            }
        }
        Some(ids::BuiltinProviderId::Deepgram) => {
            #[cfg(feature = "deepgram")]
            {
                Ok(
                    Arc::new(crate::registry::factories::DeepgramProviderFactory)
                        as Arc<dyn ProviderFactory>,
                )
            }
            #[cfg(not(feature = "deepgram"))]
            {
                Err(unsupported_provider_feature("Deepgram", "deepgram"))
            }
        }
        Some(ids::BuiltinProviderId::ElevenLabs) => {
            #[cfg(feature = "elevenlabs")]
            {
                Ok(
                    Arc::new(crate::registry::factories::ElevenLabsProviderFactory)
                        as Arc<dyn ProviderFactory>,
                )
            }
            #[cfg(not(feature = "elevenlabs"))]
            {
                Err(unsupported_provider_feature("ElevenLabs", "elevenlabs"))
            }
        }
        Some(ids::BuiltinProviderId::Bedrock) => {
            #[cfg(feature = "bedrock")]
            {
                Ok(Arc::new(crate::registry::factories::BedrockProviderFactory)
                    as Arc<dyn ProviderFactory>)
            }
            #[cfg(not(feature = "bedrock"))]
            {
                Err(unsupported_provider_feature("Amazon Bedrock", "bedrock"))
            }
        }
        Some(ids::BuiltinProviderId::Gateway) => {
            #[cfg(feature = "gateway")]
            {
                Ok(Arc::new(crate::registry::factories::GatewayProviderFactory)
                    as Arc<dyn ProviderFactory>)
            }
            #[cfg(not(feature = "gateway"))]
            {
                Err(unsupported_provider_feature("Vercel AI Gateway", "gateway"))
            }
        }
        Some(ids::BuiltinProviderId::Azure | ids::BuiltinProviderId::AzureChat) => {
            #[cfg(feature = "azure")]
            {
                azure_provider_factory_with_options(
                    &normalized,
                    siumai_provider_azure::providers::azure_openai::AzureUrlConfig::default(),
                    "azure",
                )
            }
            #[cfg(not(feature = "azure"))]
            {
                Err(unsupported_provider_feature("Azure OpenAI", "azure"))
            }
        }
        None => openai_compatible_provider_factory(&normalized),
    }
}

/// Resolve an Azure OpenAI built-in provider id into a registry factory with Azure URL options.
#[cfg(feature = "azure")]
pub(crate) fn azure_provider_factory_with_options(
    provider_id: &str,
    url_config: siumai_provider_azure::providers::azure_openai::AzureUrlConfig,
    provider_metadata_key: &'static str,
) -> Result<Arc<dyn ProviderFactory>, LlmError> {
    let normalized = crate::provider::resolver::normalize_provider_id(provider_id);
    if !ids::is_azure_family(&normalized) {
        return Err(LlmError::InvalidParameter(format!(
            "Azure provider factory options require an Azure provider id, got '{provider_id}'"
        )));
    }

    let chat_mode = match normalized.as_str() {
        ids::AZURE_CHAT => {
            siumai_provider_azure::providers::azure_openai::AzureChatMode::ChatCompletions
        }
        _ => siumai_provider_azure::providers::azure_openai::AzureChatMode::Responses,
    };

    Ok(Arc::new(
        crate::registry::factories::AzureOpenAiProviderFactory::new(chat_mode)
            .with_url_config(url_config)
            .with_provider_metadata_key(provider_metadata_key),
    ) as Arc<dyn ProviderFactory>)
}

/// Register all built-in provider factories enabled for this build.
pub(crate) fn register_enabled_builtin_provider_factories(
    providers: &mut HashMap<String, Arc<dyn ProviderFactory>>,
) {
    providers.reserve(0);

    #[cfg(feature = "openai")]
    {
        insert_builtin_provider_factory(providers, ids::OPENAI)
            .expect("OpenAI factory should be available when the openai feature is enabled");
    }

    #[cfg(feature = "azure")]
    {
        insert_builtin_provider_factory(providers, ids::AZURE)
            .expect("Azure factory should be available when the azure feature is enabled");
        insert_builtin_provider_factory(providers, ids::AZURE_CHAT)
            .expect("Azure Chat factory should be available when the azure feature is enabled");
    }

    #[cfg(feature = "anthropic")]
    {
        insert_builtin_provider_factory(providers, ids::ANTHROPIC)
            .expect("Anthropic factory should be available when the anthropic feature is enabled");
    }

    #[cfg(feature = "google")]
    {
        insert_builtin_provider_factory(providers, ids::GEMINI)
            .expect("Gemini factory should be available when the google feature is enabled");
    }

    #[cfg(feature = "google-vertex")]
    {
        insert_builtin_provider_factory(providers, ids::ANTHROPIC_VERTEX).expect(
            "Anthropic Vertex factory should be available when the google-vertex feature is enabled",
        );
        insert_builtin_provider_factory(providers, ids::VERTEX)
            .expect("Vertex factory should be available when the google-vertex feature is enabled");
        insert_builtin_provider_factory(providers, ids::VERTEX_MAAS).expect(
            "Vertex MaaS factory should be available when the google-vertex feature is enabled",
        );
        insert_builtin_provider_factory(providers, ids::GOOGLE_VERTEX_XAI).expect(
            "Google Vertex xAI factory should be available when the google-vertex feature is enabled",
        );
        insert_builtin_provider_factory(providers, ids::GOOGLE_VERTEX_ALIAS).expect(
            "Vertex alias factory should be available when the google-vertex feature is enabled",
        );
    }

    #[cfg(feature = "groq")]
    {
        insert_builtin_provider_factory(providers, ids::GROQ)
            .expect("Groq factory should be available when the groq feature is enabled");
    }

    #[cfg(feature = "xai")]
    {
        insert_builtin_provider_factory(providers, ids::XAI)
            .expect("xAI factory should be available when the xai feature is enabled");
    }

    #[cfg(feature = "ollama")]
    {
        insert_builtin_provider_factory(providers, ids::OLLAMA)
            .expect("Ollama factory should be available when the ollama feature is enabled");
    }

    #[cfg(feature = "minimaxi")]
    {
        insert_builtin_provider_factory(providers, ids::MINIMAXI)
            .expect("MiniMaxi factory should be available when the minimaxi feature is enabled");
    }

    #[cfg(feature = "cohere")]
    {
        insert_builtin_provider_factory(providers, ids::COHERE)
            .expect("Cohere factory should be available when the cohere feature is enabled");
    }

    #[cfg(feature = "togetherai")]
    {
        insert_builtin_provider_factory(providers, ids::TOGETHERAI).expect(
            "TogetherAI factory should be available when the togetherai feature is enabled",
        );
    }

    #[cfg(feature = "deepgram")]
    {
        insert_builtin_provider_factory(providers, ids::DEEPGRAM)
            .expect("Deepgram factory should be available when the deepgram feature is enabled");
    }

    #[cfg(feature = "elevenlabs")]
    {
        insert_builtin_provider_factory(providers, ids::ELEVENLABS).expect(
            "ElevenLabs factory should be available when the elevenlabs feature is enabled",
        );
    }

    #[cfg(feature = "bedrock")]
    {
        insert_builtin_provider_factory(providers, ids::BEDROCK)
            .expect("Bedrock factory should be available when the bedrock feature is enabled");
    }

    #[cfg(feature = "gateway")]
    {
        insert_builtin_provider_factory(providers, ids::GATEWAY)
            .expect("Gateway factory should be available when the gateway feature is enabled");
    }

    #[cfg(feature = "deepseek")]
    {
        insert_builtin_provider_factory(providers, ids::DEEPSEEK)
            .expect("DeepSeek factory should be available when the deepseek feature is enabled");
    }

    #[cfg(feature = "deepinfra")]
    {
        insert_builtin_provider_factory(providers, ids::DEEPINFRA)
            .expect("DeepInfra factory should be available when the deepinfra feature is enabled");
    }

    #[cfg(feature = "openai")]
    {
        let builtin =
            siumai_provider_openai_compatible::providers::openai_compatible::get_builtin_providers(
            );
        for (_id, cfg) in builtin {
            let id_str = cfg.id.clone();
            if providers.contains_key(&id_str) {
                continue;
            }
            let built_in_requires_disabled_feature = matches!(
                ids::BuiltinProviderId::parse(&id_str),
                Some(ids::BuiltinProviderId::VertexMaas | ids::BuiltinProviderId::GoogleVertexXai)
            );
            if built_in_requires_disabled_feature {
                continue;
            }
            insert_builtin_provider_factory(providers, &id_str).unwrap_or_else(|err| {
                panic!("OpenAI-compatible factory should be available for provider {id_str}: {err}")
            });
        }
    }
}
