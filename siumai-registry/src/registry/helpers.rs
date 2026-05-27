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
use crate::registry::entry::{ProviderRegistryHandle, RegistryOptions, create_provider_registry};

#[cfg(feature = "builtins")]
use crate::registry::entry::ProviderFactory;

/// Resolve the public compatibility default model for a provider id.
///
/// `SiumaiBuilder` uses this when callers choose a provider but omit `.model(...)`.
/// Native provider defaults and OpenAI-compatible fallback tables are owned by the
/// built-in provider descriptor seam.
#[cfg(feature = "builtins")]
pub fn builtin_provider_default_model(provider_id: &str) -> Result<String, LlmError> {
    crate::registry::provider_descriptor::builtin_provider_default_model(provider_id)
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
    crate::registry::provider_descriptor::openai_compatible_provider_factory(provider_id)
}

/// Resolve a built-in provider id into its registry factory.
///
/// Custom provider registries should still implement and register `ProviderFactory` directly.
/// This helper exists so normal built-in provider construction does not require callers to depend
/// on concrete factory structs under `registry::factories`.
#[cfg(feature = "builtins")]
pub fn builtin_provider_factory(provider_id: &str) -> Result<Arc<dyn ProviderFactory>, LlmError> {
    crate::registry::provider_descriptor::builtin_provider_factory(provider_id)
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
    crate::registry::provider_descriptor::azure_provider_factory_with_options(
        provider_id,
        url_config,
        provider_metadata_key,
    )
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
    let mut providers: HashMap<String, Arc<dyn ProviderFactory>> = HashMap::new();
    crate::registry::provider_descriptor::register_enabled_builtin_provider_factories(
        &mut providers,
    );

    create_provider_registry(
        providers,
        Some(RegistryOptions {
            language_model_middleware: chain_default_and_clamp(),
            http_interceptors: vec![std::sync::Arc::new(LoggingInterceptor)],
            ..Default::default()
        }),
    )
}

/// Create an empty registry (no middlewares) with ':' separator.
/// Note: auto_middleware is still enabled by default, so model-specific middlewares
/// (like ExtractReasoningMiddleware) will still be added automatically.
pub fn create_empty_registry() -> ProviderRegistryHandle {
    create_provider_registry(HashMap::new(), None)
}

/// Create a bare registry with NO middlewares at all (including no auto middlewares).
/// This is useful for testing or when you want complete control over middleware.
pub fn create_bare_registry() -> ProviderRegistryHandle {
    create_provider_registry(
        HashMap::new(),
        Some(RegistryOptions {
            auto_middleware: false,
            ..Default::default()
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
