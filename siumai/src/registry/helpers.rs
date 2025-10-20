//! Registry convenience helpers
//!
//! English-only comments in code as requested.

use std::collections::HashMap;

use crate::middleware::samples::chain_default_and_clamp;
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
            max_cache_entries: None, // Use default (100)
            client_ttl: None,        // No expiration
            auto_middleware: false,  // Disable automatic middleware
        }),
    )
}
