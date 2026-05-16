//! Registry convenience helpers
//!
//! English-only comments in code as requested.

use std::collections::HashMap;
#[cfg(feature = "builtins")]
use std::sync::Arc;

#[cfg(feature = "builtins")]
use crate::error::LlmError;
#[cfg(feature = "builtins")]
use crate::execution::http::interceptor::LoggingInterceptor;
#[cfg(feature = "builtins")]
use crate::execution::middleware::samples::chain_default_and_clamp;
#[cfg(feature = "builtins")]
use crate::provider::ids;
use crate::registry::entry::{ProviderRegistryHandle, RegistryOptions, create_provider_registry};

#[cfg(feature = "builtins")]
use crate::registry::entry::ProviderFactory;

/// Resolve the public compatibility default model for a provider id.
///
/// `SiumaiBuilder` uses this when callers choose a provider but omit `.model(...)`.
/// Native provider defaults come from `native_provider_metadata`, while configured
/// OpenAI-compatible providers use their provider-owned config tables.
#[cfg(feature = "builtins")]
pub fn builtin_provider_default_model(provider_id: &str) -> Result<String, LlmError> {
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

#[cfg(feature = "builtins")]
#[allow(dead_code)]
fn unsupported_provider_feature(provider_name: &str, feature: &str) -> LlmError {
    LlmError::UnsupportedOperation(format!(
        "{provider_name} provider requires the '{feature}' feature to be enabled"
    ))
}

#[cfg(all(feature = "builtins", not(feature = "openai")))]
fn unsupported_openai_compatible_provider(provider_id: &str) -> LlmError {
    LlmError::UnsupportedOperation(format!(
        "OpenAI-compatible provider '{provider_id}' requires the 'openai' feature to be enabled"
    ))
}

#[cfg(feature = "builtins")]
/// Resolve an OpenAI-compatible provider id into a registry factory.
///
/// Unlike `builtin_provider_factory`, this helper intentionally accepts provider ids that are not
/// native Siumai families. This covers configured OpenAI-compatible vendors and advanced custom
/// compatible ids while keeping concrete factory construction inside the registry crate.
pub fn openai_compatible_provider_factory(
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

#[cfg(feature = "builtins")]
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
///
/// Custom provider registries should still implement and register `ProviderFactory` directly.
/// This helper exists so normal built-in provider construction does not require callers to depend
/// on concrete factory structs under `registry::factories`.
#[cfg(feature = "builtins")]
pub fn builtin_provider_factory(provider_id: &str) -> Result<Arc<dyn ProviderFactory>, LlmError> {
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
///
/// This is the provider-specific companion to `builtin_provider_factory` for Azure's
/// deployment-based URL mode and metadata-key selection. It keeps concrete Azure factory
/// construction inside the registry crate while still allowing advanced registry setups to choose
/// Azure URL semantics explicitly.
#[cfg(feature = "azure")]
pub fn azure_provider_factory_with_options(
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

/// Create a registry with common defaults:
/// - separator ':'
/// - language model middlewares: default params + clamp top_p
/// - LRU cache: 100 entries (default)
/// - TTL: None (no expiration)
/// - auto_middleware: true (automatically add model-specific middlewares)
/// - Built-in provider factories registered for common providers
///   (OpenAI, Azure OpenAI, Anthropic, Anthropic Vertex, Gemini, Groq, xAI, Ollama,
///   MiniMaxi, DeepSeek, DeepInfra, and all OpenAI-compatible providers)
#[cfg(feature = "builtins")]
pub fn create_registry_with_defaults() -> ProviderRegistryHandle {
    // Register built-in provider factories for the handle-level registry.
    let mut providers: HashMap<String, Arc<dyn ProviderFactory>> = HashMap::new();
    // In feature-minimal builds, the inserts below may be compiled out; keep the binding mutable
    // without triggering `unused_mut` in those configurations.
    providers.reserve(0);

    // Native provider factories
    #[cfg(feature = "openai")]
    {
        insert_builtin_provider_factory(&mut providers, ids::OPENAI)
            .expect("OpenAI factory should be available when the openai feature is enabled");
    }

    #[cfg(feature = "azure")]
    {
        insert_builtin_provider_factory(&mut providers, ids::AZURE)
            .expect("Azure factory should be available when the azure feature is enabled");
        // Variant: Azure Chat Completions (Vercel-aligned `azure.chat(...)`).
        insert_builtin_provider_factory(&mut providers, ids::AZURE_CHAT)
            .expect("Azure Chat factory should be available when the azure feature is enabled");
    }

    #[cfg(feature = "anthropic")]
    {
        insert_builtin_provider_factory(&mut providers, ids::ANTHROPIC)
            .expect("Anthropic factory should be available when the anthropic feature is enabled");
    }

    #[cfg(feature = "google")]
    {
        insert_builtin_provider_factory(&mut providers, ids::GEMINI)
            .expect("Gemini factory should be available when the google feature is enabled");
    }

    #[cfg(feature = "google-vertex")]
    {
        insert_builtin_provider_factory(&mut providers, ids::ANTHROPIC_VERTEX).expect(
            "Anthropic Vertex factory should be available when the google-vertex feature is enabled",
        );
        insert_builtin_provider_factory(&mut providers, ids::VERTEX)
            .expect("Vertex factory should be available when the google-vertex feature is enabled");
        // Alias for package naming consistency (Vercel-style `@ai-sdk/google-vertex`).
        insert_builtin_provider_factory(&mut providers, ids::GOOGLE_VERTEX_ALIAS).expect(
            "Vertex alias factory should be available when the google-vertex feature is enabled",
        );
    }

    #[cfg(feature = "groq")]
    {
        insert_builtin_provider_factory(&mut providers, ids::GROQ)
            .expect("Groq factory should be available when the groq feature is enabled");
    }

    #[cfg(feature = "xai")]
    {
        insert_builtin_provider_factory(&mut providers, ids::XAI)
            .expect("xAI factory should be available when the xai feature is enabled");
    }

    #[cfg(feature = "ollama")]
    {
        insert_builtin_provider_factory(&mut providers, ids::OLLAMA)
            .expect("Ollama factory should be available when the ollama feature is enabled");
    }

    #[cfg(feature = "minimaxi")]
    {
        insert_builtin_provider_factory(&mut providers, ids::MINIMAXI)
            .expect("MiniMaxi factory should be available when the minimaxi feature is enabled");
    }

    #[cfg(feature = "cohere")]
    {
        insert_builtin_provider_factory(&mut providers, ids::COHERE)
            .expect("Cohere factory should be available when the cohere feature is enabled");
    }

    #[cfg(feature = "togetherai")]
    {
        insert_builtin_provider_factory(&mut providers, ids::TOGETHERAI).expect(
            "TogetherAI factory should be available when the togetherai feature is enabled",
        );
    }

    #[cfg(feature = "bedrock")]
    {
        insert_builtin_provider_factory(&mut providers, ids::BEDROCK)
            .expect("Bedrock factory should be available when the bedrock feature is enabled");
    }

    // Provider-specific factories built on top of the OpenAI-compatible runtime.
    #[cfg(feature = "deepseek")]
    {
        insert_builtin_provider_factory(&mut providers, ids::DEEPSEEK)
            .expect("DeepSeek factory should be available when the deepseek feature is enabled");
    }

    #[cfg(feature = "deepinfra")]
    {
        insert_builtin_provider_factory(&mut providers, ids::DEEPINFRA)
            .expect("DeepInfra factory should be available when the deepinfra feature is enabled");
    }

    // OpenAI-compatible provider factories (SiliconFlow, OpenRouter, etc.)
    #[cfg(feature = "openai")]
    {
        let builtin =
            siumai_provider_openai_compatible::providers::openai_compatible::get_builtin_providers(
            );
        for (_id, cfg) in builtin {
            let id_str = cfg.id.clone();
            // Skip providers that already have dedicated factories registered (e.g., deepseek, groq, minimaxi).
            if providers.contains_key(&id_str) {
                continue;
            }
            insert_builtin_provider_factory(&mut providers, &id_str).unwrap_or_else(|err| {
                panic!("OpenAI-compatible factory should be available for provider {id_str}: {err}")
            });
        }
    }

    create_provider_registry(
        providers,
        Some(RegistryOptions {
            separator: ':',
            language_model_middleware: chain_default_and_clamp(),
            http_interceptors: vec![std::sync::Arc::new(LoggingInterceptor)],
            http_client: None,
            http_transport: None,
            http_config: None,
            api_key: None,
            base_url: None,
            reasoning_enabled: None,
            reasoning_budget: None,
            provider_build_overrides: HashMap::new(),
            retry_options: None,
            max_cache_entries: None, // Use default (100)
            client_ttl: None,        // No expiration
            auto_middleware: true,   // Enable automatic middleware
        }),
    )
}

/// Create an empty registry (no middlewares) with ':' separator.
/// Note: auto_middleware is still enabled by default, so model-specific middlewares
/// (like ExtractReasoningMiddleware) will still be added automatically.
pub fn create_empty_registry() -> ProviderRegistryHandle {
    create_provider_registry(
        HashMap::new(),
        Some(RegistryOptions {
            separator: ':',
            language_model_middleware: Vec::new(),
            http_interceptors: Vec::new(),
            http_client: None,
            http_transport: None,
            http_config: None,
            api_key: None,
            base_url: None,
            reasoning_enabled: None,
            reasoning_budget: None,
            provider_build_overrides: HashMap::new(),
            retry_options: None,
            max_cache_entries: None, // Use default (100)
            client_ttl: None,        // No expiration
            auto_middleware: true,   // Enable automatic middleware
        }),
    )
}

/// Create a bare registry with NO middlewares at all (including no auto middlewares).
/// This is useful for testing or when you want complete control over middleware.
pub fn create_bare_registry() -> ProviderRegistryHandle {
    create_provider_registry(
        HashMap::new(),
        Some(RegistryOptions {
            separator: ':',
            language_model_middleware: Vec::new(),
            http_interceptors: Vec::new(),
            http_client: None,
            http_transport: None,
            http_config: None,
            api_key: None,
            base_url: None,
            reasoning_enabled: None,
            reasoning_budget: None,
            provider_build_overrides: HashMap::new(),
            retry_options: None,
            max_cache_entries: None, // Use default (100)
            client_ttl: None,        // No expiration
            auto_middleware: false,  // Disable automatic middleware
        }),
    )
}

/// Compare two provider identifiers considering registry aliases.
///
/// Examples:
/// - "gemini" and "google" are treated as the same provider when the alias is registered.
/// - Case-sensitive comparison.
#[cfg(feature = "builtins")]
pub fn matches_provider_id(provider_id: &str, custom_id: &str) -> bool {
    if provider_id == custom_id {
        return true;
    }
    let guard = match crate::registry::global_registry().read() {
        Ok(g) => g,
        Err(_) => return false,
    };
    guard.is_same_provider(provider_id, custom_id)
}

#[cfg(all(test, any(feature = "azure", feature = "openai")))]
mod tests {
    use super::*;

    #[cfg(feature = "openai")]
    #[test]
    fn openai_compatible_provider_factory_uses_requested_provider_id() {
        let factory = match openai_compatible_provider_factory("openrouter") {
            Ok(factory) => factory,
            Err(err) => panic!("expected openai-compatible factory: {err:?}"),
        };

        let capabilities = factory.capabilities();
        assert_eq!(factory.provider_id().as_ref(), "openai-compatible");
        assert!(capabilities.chat);
        assert!(capabilities.embedding);
        assert!(capabilities.streaming);
    }

    #[cfg(feature = "azure")]
    #[test]
    fn azure_provider_factory_with_options_returns_azure_factory() {
        let factory = match azure_provider_factory_with_options(
            "azure",
            siumai_provider_azure::providers::azure_openai::AzureUrlConfig::default(),
            "azure",
        ) {
            Ok(factory) => factory,
            Err(err) => panic!("expected azure factory: {err:?}"),
        };

        assert_eq!(factory.provider_id().as_ref(), "azure");
    }

    #[cfg(feature = "azure")]
    #[test]
    fn azure_provider_factory_with_options_rejects_non_azure_id() {
        let err = match azure_provider_factory_with_options(
            "openai",
            siumai_provider_azure::providers::azure_openai::AzureUrlConfig::default(),
            "azure",
        ) {
            Ok(_) => panic!("expected non-azure provider id to be rejected"),
            Err(err) => err,
        };

        match err {
            LlmError::InvalidParameter(message) => {
                assert!(message.contains("require an Azure provider id"));
            }
            other => panic!("expected InvalidParameter, got {other:?}"),
        }
    }
}
