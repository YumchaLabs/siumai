//! Provider registries and handles.
//!
//! This module intentionally separates two concerns:
//! - **Provider-agnostic registry handle**: `entry::*` provides the Vercel-style
//!   `"provider:model"` access pattern (with caching, middleware injection, etc.).
//! - **Built-in provider catalog + wiring** (optional): when the `builtins` feature
//!   is enabled, `siumai-registry` also includes a small provider catalog for
//!   aliasing / OpenAI-compatible adapter wiring and a `global()` handle with
//!   built-in factories pre-registered.
//!
//! This keeps `siumai-registry` usable as an integration point for external
//! provider crates without pulling in built-in provider implementations by default.

pub mod builder;
pub mod entry;

// Built-in provider factory implementations (feature-gated; depend on provider crates).
#[cfg(feature = "builtins")]
pub mod factories;
#[cfg(feature = "builtins")]
pub mod factory;

pub mod helpers;

// -----------------------------------------------------------------------------
// Built-in provider catalog (optional)
// -----------------------------------------------------------------------------

#[cfg(feature = "builtins")]
mod builtins {
    use std::collections::HashMap;
    use std::sync::{OnceLock, RwLock};

    #[cfg(feature = "openai")]
    use std::sync::Arc;

    use crate::traits::ProviderCapabilities;

    #[cfg(feature = "openai")]
    use crate::error::LlmError;

    #[cfg(feature = "openai")]
    use siumai_provider_openai_compatible::providers::openai_compatible::{
        ConfigurableAdapter, ProviderAdapter, ProviderConfig,
    };

    /// Unified provider record maintained by the built-in catalog.
    #[derive(Debug, Clone)]
    pub struct ProviderRecord {
        pub id: String,
        pub name: String,
        pub base_url: Option<String>,
        pub capabilities: ProviderCapabilities,
        #[cfg(feature = "openai")]
        pub adapter: Option<Arc<dyn ProviderAdapter>>, // for OpenAI-compatible
        pub aliases: Vec<String>,
        /// Optional model id prefixes that hint routing for this provider.
        pub model_prefixes: Vec<String>,
        /// Default model for this provider (optional).
        pub default_model: Option<String>,
    }

    impl ProviderRecord {
        pub fn with_alias(mut self, alias: impl Into<String>) -> Self {
            self.aliases.push(alias.into());
            self
        }

        pub fn with_model_prefix(mut self, prefix: impl Into<String>) -> Self {
            self.model_prefixes.push(prefix.into());
            self
        }

        pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
            self.default_model = Some(model.into());
            self
        }
    }

    /// Built-in provider catalog (metadata + OpenAI-compatible adapters).
    ///
    /// This is *not* the main "provider:model" registry handle. It exists to
    /// support built-in providers and OpenAI-compatible wiring for the default
    /// `global()` handle and documentation utilities.
    #[derive(Default)]
    pub struct ProviderRegistry {
        by_id: HashMap<String, ProviderRecord>,
        /// alias -> canonical id, and id -> id for O(1) canonical lookups.
        alias_index: HashMap<String, String>,
    }

    impl ProviderRegistry {
        /// Create a new empty catalog.
        pub fn new() -> Self {
            Self::default()
        }

        /// Create a catalog with all built-in providers pre-registered.
        pub fn with_builtin_providers() -> Self {
            let mut registry = Self::new();
            registry.register_builtin_providers();
            registry
        }

        fn register_builtin_providers(&mut self) {
            self.register_native_providers();

            // Register OpenAI-compatible providers.
            #[cfg(feature = "openai")]
            self.register_openai_compatible_providers();
        }

        fn register_native_providers(&mut self) {
            // Register native providers from the shared metadata table so that
            // names, base URLs, and capabilities stay consistent across the
            // registry and documentation helpers.
            let metas = crate::native_provider_metadata::native_providers_metadata();
            for meta in metas {
                // Metadata-only provider ids (not backed by built-in factories yet).
                // Keep them out of the default built-in catalog to avoid suggesting
                // they are buildable.
                if matches!(
                    meta.id,
                    crate::provider::ids::COHERE
                        | crate::provider::ids::TOGETHERAI
                        | crate::provider::ids::BEDROCK
                ) {
                    continue;
                }
                self.register_native(
                    meta.id,
                    meta.name,
                    meta.default_base_url.map(|url| url.to_string()),
                    meta.capabilities.clone(),
                );
            }

            // Anthropic on Vertex AI (native wrapper around Anthropic via Vertex).
            #[cfg(feature = "google-vertex")]
            {
                if let Some(rec) = self
                    .resolve(crate::provider::ids::ANTHROPIC_VERTEX)
                    .cloned()
                {
                    let rec = rec
                        .with_alias("google-vertex-anthropic")
                        .with_model_prefix("claude");
                    self.register(rec);
                }
            }

            // Google Gemini alias.
            #[cfg(feature = "google")]
            {
                self.add_alias(crate::provider::ids::GEMINI, "google");
            }

            // Google Vertex alias (package naming consistency with `@ai-sdk/google-vertex`).
            #[cfg(feature = "google-vertex")]
            {
                self.add_alias(
                    crate::provider::ids::VERTEX,
                    crate::provider::ids::GOOGLE_VERTEX_ALIAS,
                );
            }
        }

        /// Register all OpenAI-compatible providers from config.
        #[cfg(feature = "openai")]
        fn register_openai_compatible_providers(&mut self) {
            let builtin_providers =
                siumai_provider_openai_compatible::providers::openai_compatible::get_builtin_providers();

            for (_id, config) in builtin_providers {
                let _ = self.register_openai_compatible_from_config(config);
            }
        }

        pub fn register(&mut self, record: ProviderRecord) {
            let id = record.id.clone();
            let aliases = record.aliases.clone();
            self.by_id.insert(id.clone(), record);

            self.alias_index.insert(id.clone(), id.clone());
            for a in aliases {
                self.alias_index.insert(a, id.clone());
            }
        }

        #[cfg(feature = "openai")]
        pub fn register_openai_compatible_from_config(
            &mut self,
            config: ProviderConfig,
        ) -> Result<(), LlmError> {
            let adapter = Arc::new(ConfigurableAdapter::new(config.clone()));
            let capabilities = adapter.capabilities();

            let record = ProviderRecord {
                id: config.id.clone(),
                name: config.name,
                base_url: Some(config.base_url),
                capabilities,
                adapter: Some(adapter),
                aliases: vec![],
                model_prefixes: vec![],
                default_model: config.default_model,
            };

            self.register(record);
            Ok(())
        }

        #[cfg(feature = "openai")]
        pub fn register_openai_compatible(&mut self, provider_id: &str) -> Result<(), LlmError> {
            let config =
                siumai_provider_openai_compatible::providers::openai_compatible::get_provider_config(
                    provider_id,
                )
            .ok_or_else(|| {
                LlmError::ConfigurationError(format!(
                    "Unknown OpenAI-compatible provider: {}",
                    provider_id
                ))
            })?;

            self.register_openai_compatible_from_config(config)
        }

        /// Register an OpenAI-compatible vendor preset (OpenAI-like provider).
        ///
        /// This is an alias of `register_openai_compatible`, provided to keep the mental model:
        /// "OpenAI is the protocol family, vendors are presets/config."
        #[cfg(feature = "openai")]
        pub fn register_openai_vendor(&mut self, vendor_id: &str) -> Result<(), LlmError> {
            self.register_openai_compatible(vendor_id)
        }

        /// Register an OpenAI-compatible vendor preset from config (OpenAI-like provider).
        #[cfg(feature = "openai")]
        pub fn register_openai_vendor_from_config(
            &mut self,
            config: ProviderConfig,
        ) -> Result<(), LlmError> {
            self.register_openai_compatible_from_config(config)
        }

        pub fn register_native(
            &mut self,
            id: &str,
            name: &str,
            base_url: Option<String>,
            capabilities: ProviderCapabilities,
        ) {
            let record = ProviderRecord {
                id: id.to_string(),
                name: name.to_string(),
                base_url,
                capabilities,
                #[cfg(feature = "openai")]
                adapter: None,
                aliases: vec![],
                model_prefixes: vec![],
                default_model: None,
            };
            self.register(record);
        }

        pub fn add_alias(&mut self, id: &str, alias: &str) {
            if let Some(rec) = self.by_id.get_mut(id) {
                rec.aliases.push(alias.to_string());
                self.alias_index.insert(alias.to_string(), rec.id.clone());
            }
        }

        #[cfg(feature = "openai")]
        pub fn register_custom_provider(&mut self, config: ProviderConfig) {
            let _ = self.register_openai_compatible_from_config(config);
        }

        pub fn resolve(&self, id_or_alias: &str) -> Option<&ProviderRecord> {
            if let Some(canon) = self.alias_index.get(id_or_alias) {
                return self.by_id.get(canon);
            }
            self.by_id.get(id_or_alias)
        }

        pub fn resolve_for_model(&self, model_id: &str) -> Option<&ProviderRecord> {
            self.by_id
                .values()
                .find(|rec| rec.model_prefixes.iter().any(|p| model_id.starts_with(p)))
        }

        #[cfg(feature = "openai")]
        pub fn get_adapter(&self, provider_id: &str) -> Result<Arc<dyn ProviderAdapter>, LlmError> {
            let record = self.resolve(provider_id).ok_or_else(|| {
                LlmError::ConfigurationError(format!("Unknown provider: {}", provider_id))
            })?;

            record.adapter.clone().ok_or_else(|| {
                LlmError::ConfigurationError(format!(
                    "Provider {} does not have an adapter (not OpenAI-compatible)",
                    provider_id
                ))
            })
        }

        pub fn list(&self) -> Vec<&str> {
            self.by_id.keys().map(|s| s.as_str()).collect()
        }

        pub fn list_providers(&self) -> Vec<&ProviderRecord> {
            self.by_id.values().collect()
        }

        pub fn canonical_id<'a>(&'a self, id_or_alias: &str) -> Option<&'a str> {
            self.alias_index
                .get(id_or_alias)
                .map(|s| s.as_str())
                .or_else(|| self.by_id.get(id_or_alias).map(|r| r.id.as_str()))
        }

        pub fn is_same_provider(&self, a: &str, b: &str) -> bool {
            match (self.canonical_id(a), self.canonical_id(b)) {
                (Some(x), Some(y)) => x == y,
                _ => false,
            }
        }
    }

    // Global provider catalog instance.
    static GLOBAL_REGISTRY: OnceLock<RwLock<ProviderRegistry>> = OnceLock::new();

    /// Get the global provider catalog (initialized with built-in providers).
    pub fn global_registry() -> &'static RwLock<ProviderRegistry> {
        GLOBAL_REGISTRY.get_or_init(|| RwLock::new(ProviderRegistry::with_builtin_providers()))
    }

    /// Get the global registry handle (recommended for most use cases).
    pub fn global() -> &'static super::entry::ProviderRegistryHandle {
        static GLOBAL_HANDLE: OnceLock<super::entry::ProviderRegistryHandle> = OnceLock::new();
        GLOBAL_HANDLE.get_or_init(super::helpers::create_registry_with_defaults)
    }

    /// Convenience function to get an adapter for an OpenAI-compatible provider.
    #[cfg(feature = "openai")]
    pub fn get_provider_adapter(provider_id: &str) -> Result<Arc<dyn ProviderAdapter>, LlmError> {
        global_registry()
            .read()
            .map_err(|_| {
                LlmError::ConfigurationError("Failed to lock provider registry".to_string())
            })?
            .get_adapter(provider_id)
    }

    /// Convenience function to get an adapter for an OpenAI vendor preset (OpenAI-like provider).
    #[cfg(feature = "openai")]
    pub fn get_openai_vendor_adapter(
        provider_id: &str,
    ) -> Result<Arc<dyn ProviderAdapter>, LlmError> {
        get_provider_adapter(provider_id)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_registry_resolve_by_id() {
            let _registry = ProviderRegistry::with_builtin_providers();

            #[cfg(feature = "openai")]
            assert!(_registry.resolve("openai").is_some());

            #[cfg(feature = "anthropic")]
            assert!(_registry.resolve("anthropic").is_some());

            #[cfg(feature = "google")]
            assert!(_registry.resolve("gemini").is_some());
        }

        #[test]
        #[cfg(feature = "cohere")]
        fn test_registry_does_not_register_metadata_only_cohere() {
            let registry = ProviderRegistry::with_builtin_providers();
            assert!(registry.resolve(crate::provider::ids::COHERE).is_none());
        }

        #[test]
        #[cfg(feature = "togetherai")]
        fn test_registry_does_not_register_metadata_only_togetherai() {
            let registry = ProviderRegistry::with_builtin_providers();
            assert!(registry.resolve(crate::provider::ids::TOGETHERAI).is_none());
        }

        #[test]
        #[cfg(feature = "bedrock")]
        fn test_registry_does_not_register_metadata_only_bedrock() {
            let registry = ProviderRegistry::with_builtin_providers();
            assert!(registry.resolve(crate::provider::ids::BEDROCK).is_none());
        }

        #[test]
        #[cfg(feature = "google")]
        fn test_registry_resolve_gemini_by_google_alias() {
            let registry = ProviderRegistry::with_builtin_providers();

            let gemini_by_id = registry.resolve("gemini");
            assert!(gemini_by_id.is_some());

            let gemini_by_alias = registry.resolve("google");
            assert!(gemini_by_alias.is_some());

            assert_eq!(gemini_by_id.unwrap().id, gemini_by_alias.unwrap().id);
            assert_eq!(gemini_by_id.unwrap().id, "gemini");
        }

        #[test]
        fn test_registry_resolve_unknown_provider() {
            let registry = ProviderRegistry::with_builtin_providers();
            assert!(registry.resolve("unknown_provider").is_none());
        }
    }
}

#[cfg(feature = "builtins")]
pub use builtins::{ProviderRecord, ProviderRegistry, global, global_registry};

#[cfg(all(feature = "builtins", feature = "openai"))]
pub use builtins::get_provider_adapter;

#[cfg(all(feature = "builtins", feature = "openai"))]
pub use builtins::get_openai_vendor_adapter;

// -----------------------------------------------------------------------------
// Public re-exports (provider-agnostic)
// -----------------------------------------------------------------------------

pub use entry::{
    BuildContext, EmbeddingModelHandle, ImageModelHandle, LanguageModelHandle, ProviderFactory,
    ProviderRegistryHandle, RegistryOptions, RerankingModelHandle, SpeechModelHandle,
    TranscriptionModelHandle, create_provider_registry,
};

pub use helpers::{create_bare_registry, create_empty_registry};

#[cfg(feature = "builtins")]
pub use helpers::create_registry_with_defaults;
