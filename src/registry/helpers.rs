//! Registry convenience helpers
//!
//! English-only comments in code as requested.

use std::collections::HashMap;

use crate::middleware::samples::chain_default_and_clamp;
use crate::registry::entry::{ProviderRegistryHandle, RegistryOptions, create_provider_registry};

/// Create a registry with common defaults:
/// - separator ':'
/// - language model middlewares: default params + clamp top_p
pub fn create_registry_with_defaults() -> ProviderRegistryHandle {
    create_provider_registry(
        HashMap::new(),
        Some(RegistryOptions {
            separator: ':',
            language_model_middleware: chain_default_and_clamp(),
        }),
    )
}

/// Create an empty registry (no middlewares) with ':' separator.
pub fn create_empty_registry() -> ProviderRegistryHandle {
    create_provider_registry(
        HashMap::new(),
        Some(RegistryOptions {
            separator: ':',
            language_model_middleware: Vec::new(),
        }),
    )
}
