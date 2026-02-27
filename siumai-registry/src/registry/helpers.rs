//! Registry convenience helpers
//!
//! English-only comments in code as requested.

use std::collections::HashMap;
#[cfg(feature = "builtins")]
use std::sync::Arc;

#[cfg(feature = "builtins")]
use crate::execution::http::interceptor::LoggingInterceptor;
#[cfg(feature = "builtins")]
use crate::execution::middleware::samples::chain_default_and_clamp;
use crate::provider::ids;
use crate::registry::entry::{ProviderRegistryHandle, RegistryOptions, create_provider_registry};

#[cfg(feature = "builtins")]
use crate::registry::entry::ProviderFactory;

/// Create a registry with common defaults:
/// - separator ':'
/// - language model middlewares: default params + clamp top_p
/// - LRU cache: 100 entries (default)
/// - TTL: None (no expiration)
/// - auto_middleware: true (automatically add model-specific middlewares)
/// - Built-in provider factories registered for common providers
///   (OpenAI, Azure OpenAI, Anthropic, Anthropic Vertex, Gemini, Groq, xAI, Ollama,
///   MiniMaxi, and all OpenAI-compatible providers)
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
        providers.insert(
            ids::OPENAI.to_string(),
            Arc::new(crate::registry::factories::OpenAIProviderFactory) as Arc<dyn ProviderFactory>,
        );
    }

    #[cfg(feature = "azure")]
    {
        providers.insert(
            ids::AZURE.to_string(),
            Arc::new(crate::registry::factories::AzureOpenAiProviderFactory::default())
                as Arc<dyn ProviderFactory>,
        );
        // Variant: Azure Chat Completions (Vercel-aligned `azure.chat(...)`).
        providers.insert(
            ids::AZURE_CHAT.to_string(),
            Arc::new(crate::registry::factories::AzureOpenAiProviderFactory::new(
                siumai_provider_azure::providers::azure_openai::AzureChatMode::ChatCompletions,
            )) as Arc<dyn ProviderFactory>,
        );
    }

    #[cfg(feature = "anthropic")]
    {
        providers.insert(
            ids::ANTHROPIC.to_string(),
            Arc::new(crate::registry::factories::AnthropicProviderFactory)
                as Arc<dyn ProviderFactory>,
        );
    }

    #[cfg(feature = "google")]
    {
        providers.insert(
            ids::GEMINI.to_string(),
            Arc::new(crate::registry::factories::GeminiProviderFactory) as Arc<dyn ProviderFactory>,
        );
    }

    #[cfg(feature = "google-vertex")]
    {
        providers.insert(
            ids::ANTHROPIC_VERTEX.to_string(),
            Arc::new(crate::registry::factories::AnthropicVertexProviderFactory)
                as Arc<dyn ProviderFactory>,
        );
        providers.insert(
            ids::VERTEX.to_string(),
            Arc::new(crate::registry::factories::GoogleVertexProviderFactory)
                as Arc<dyn ProviderFactory>,
        );
        // Alias for package naming consistency (Vercel-style `@ai-sdk/google-vertex`).
        providers.insert(
            ids::GOOGLE_VERTEX_ALIAS.to_string(),
            Arc::new(crate::registry::factories::GoogleVertexProviderFactory)
                as Arc<dyn ProviderFactory>,
        );
    }

    #[cfg(feature = "groq")]
    {
        providers.insert(
            ids::GROQ.to_string(),
            Arc::new(crate::registry::factories::GroqProviderFactory) as Arc<dyn ProviderFactory>,
        );
    }

    #[cfg(feature = "xai")]
    {
        providers.insert(
            ids::XAI.to_string(),
            Arc::new(crate::registry::factories::XAIProviderFactory) as Arc<dyn ProviderFactory>,
        );
    }

    #[cfg(feature = "ollama")]
    {
        providers.insert(
            ids::OLLAMA.to_string(),
            Arc::new(crate::registry::factories::OllamaProviderFactory) as Arc<dyn ProviderFactory>,
        );
    }

    #[cfg(feature = "minimaxi")]
    {
        providers.insert(
            ids::MINIMAXI.to_string(),
            Arc::new(crate::registry::factories::MiniMaxiProviderFactory)
                as Arc<dyn ProviderFactory>,
        );
    }

    // OpenAI-compatible provider factories (DeepSeek, SiliconFlow, OpenRouter, etc.)
    #[cfg(feature = "openai")]
    {
        let builtin =
            siumai_provider_openai_compatible::providers::openai_compatible::get_builtin_providers(
            );
        for (_id, cfg) in builtin {
            let id_str = cfg.id.clone();
            // Skip providers that already have native factories registered (e.g., groq, minimaxi).
            if providers.contains_key(&id_str) {
                continue;
            }
            providers.insert(
                id_str.clone(),
                Arc::new(crate::registry::factories::OpenAICompatibleProviderFactory::new(id_str))
                    as Arc<dyn ProviderFactory>,
            );
        }
    }

    create_provider_registry(
        providers,
        Some(RegistryOptions {
            separator: ':',
            language_model_middleware: chain_default_and_clamp(),
            http_interceptors: vec![std::sync::Arc::new(LoggingInterceptor)],
            http_config: None,
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
            http_config: None,
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
            http_config: None,
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
