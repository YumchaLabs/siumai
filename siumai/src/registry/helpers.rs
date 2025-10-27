//! Registry convenience helpers
//!
//! English-only comments in code as requested.

use std::collections::HashMap;

use crate::execution::http::interceptor::LoggingInterceptor;
use crate::execution::middleware::samples::chain_default_and_clamp;
use crate::registry::entry::{ProviderRegistryHandle, RegistryOptions, create_provider_registry};

/// Create a registry with common defaults:
/// - separator ':'
/// - language model middlewares: default params + clamp top_p
/// - LRU cache: 100 entries (default)
/// - TTL: None (no expiration)
/// - auto_middleware: true (automatically add model-specific middlewares)
pub fn create_registry_with_defaults() -> ProviderRegistryHandle {
    create_provider_registry(
        HashMap::new(),
        Some(RegistryOptions {
            separator: ':',
            language_model_middleware: chain_default_and_clamp(),
            http_interceptors: vec![std::sync::Arc::new(LoggingInterceptor)],
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
