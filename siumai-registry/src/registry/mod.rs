//! Unified Provider Registry
//!
//! A configuration-driven registry that unifies provider lookup and adapter wiring.
//! This registry combines native providers and OpenAI-compatible providers into a
//! single, unified interface inspired by Cherry Studio's design.

use std::collections::HashMap;
#[cfg(feature = "openai")]
use std::sync::Arc;
use std::sync::{OnceLock, RwLock};

#[cfg(feature = "openai")]
use crate::error::LlmError;
use crate::traits::ProviderCapabilities;

#[cfg(feature = "openai")]
use crate::providers::openai_compatible::adapter::ProviderAdapter;
#[cfg(feature = "openai")]
use crate::providers::openai_compatible::{ConfigurableAdapter, ProviderConfig};
pub mod builder;

/// Unified provider record maintained by the registry
#[derive(Debug, Clone)]
pub struct ProviderRecord {
    pub id: String,
    pub name: String,
    pub base_url: Option<String>,
    pub capabilities: ProviderCapabilities,
    #[cfg(feature = "openai")]
    pub adapter: Option<Arc<dyn ProviderAdapter>>, // for OpenAI-compatible
    pub aliases: Vec<String>,
    /// Optional model id prefixes that hint routing for this provider
    pub model_prefixes: Vec<String>,
    /// Default model for this provider (optional)
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

/// Unified Provider Registry
///
/// This registry manages both native providers (OpenAI, Anthropic, Gemini, etc.)
/// and OpenAI-compatible providers (DeepSeek, SiliconFlow, OpenRouter, etc.)
/// in a single, unified interface.
#[derive(Default)]
pub struct ProviderRegistry {
    by_id: HashMap<String, ProviderRecord>,
    /// alias -> canonical id, and id -> id for O(1) canonical lookups
    alias_index: HashMap<String, String>,
}

impl ProviderRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a registry with all built-in providers pre-registered
    ///
    /// This includes:
    /// - Native providers: OpenAI, Anthropic, Gemini, Groq, xAI, Ollama
    /// - OpenAI-compatible providers: DeepSeek, SiliconFlow, OpenRouter, Together, etc.
    pub fn with_builtin_providers() -> Self {
        let mut registry = Self::new();
        registry.register_builtin_providers();
        registry
    }

    /// Register all built-in providers
    fn register_builtin_providers(&mut self) {
        // Register native providers
        self.register_native_providers();

        // Register OpenAI-compatible providers
        #[cfg(feature = "openai")]
        self.register_openai_compatible_providers();
    }

    /// Register native providers (OpenAI, Anthropic, Gemini, etc.)
    fn register_native_providers(&mut self) {
        // Register native providers from the shared metadata table so that
        // names, base URLs, and capabilities stay consistent across the
        // registry and documentation helpers.
        let metas = crate::providers::metadata::native_providers_metadata();
        for meta in metas {
            self.register_native(
                meta.id,
                meta.name,
                meta.default_base_url.map(|url| url.to_string()),
                meta.capabilities.clone(),
            );
        }

        // Anthropic on Vertex AI (native wrapper around Anthropic via Vertex)
        #[cfg(feature = "anthropic")]
        {
            // Add common alias and model prefix to ease lookup/routing.
            if let Some(rec) = self.resolve("anthropic-vertex").cloned() {
                let rec = rec
                    .with_alias("google-vertex-anthropic")
                    .with_model_prefix("claude");
                self.register(rec);
            }
        }

        // Google Gemini alias
        #[cfg(feature = "google")]
        {
            // Add "google" as an alias for "gemini"
            self.add_alias("gemini", "google");
        }
    }

    /// Register all OpenAI-compatible providers from config
    #[cfg(feature = "openai")]
    fn register_openai_compatible_providers(&mut self) {
        let builtin_providers =
            crate::providers::openai_compatible::config::get_builtin_providers();

        for (_id, config) in builtin_providers {
            let _ = self.register_openai_compatible_from_config(config);
        }
    }

    /// Register a prebuilt record
    pub fn register(&mut self, record: ProviderRecord) {
        // Insert record
        let id = record.id.clone();
        let aliases = record.aliases.clone();
        self.by_id.insert(id.clone(), record);

        // Update alias index: id -> id
        self.alias_index.insert(id.clone(), id.clone());
        // Update alias index: alias -> id
        for a in aliases {
            self.alias_index.insert(a, id.clone());
        }
    }

    /// Register an OpenAI-compatible provider from configuration
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

    /// Register an OpenAI-compatible provider by id using the built-in config
    #[cfg(feature = "openai")]
    pub fn register_openai_compatible(&mut self, provider_id: &str) -> Result<(), LlmError> {
        let config = crate::providers::openai_compatible::config::get_provider_config(provider_id)
            .ok_or_else(|| {
                LlmError::ConfigurationError(format!(
                    "Unknown OpenAI-compatible provider: {}",
                    provider_id
                ))
            })?;

        self.register_openai_compatible_from_config(config)
    }

    /// Register a native (non OpenAI-compatible) provider record
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

    /// Add an alias for an existing provider and update indexes
    pub fn add_alias(&mut self, id: &str, alias: &str) {
        if let Some(rec) = self.by_id.get_mut(id) {
            rec.aliases.push(alias.to_string());
            self.alias_index.insert(alias.to_string(), rec.id.clone());
        }
    }

    /// Register a custom OpenAI-compatible provider
    #[cfg(feature = "openai")]
    pub fn register_custom_provider(&mut self, config: ProviderConfig) {
        let _ = self.register_openai_compatible_from_config(config);
    }

    /// Resolve a provider record by id or alias (O(1) with alias index)
    pub fn resolve(&self, id_or_alias: &str) -> Option<&ProviderRecord> {
        if let Some(canon) = self.alias_index.get(id_or_alias) {
            return self.by_id.get(canon);
        }
        self.by_id.get(id_or_alias)
    }

    /// Resolve by model id prefix (best-effort). Returns the first match.
    pub fn resolve_for_model(&self, model_id: &str) -> Option<&ProviderRecord> {
        self.by_id
            .values()
            .find(|rec| rec.model_prefixes.iter().any(|p| model_id.starts_with(p)))
    }

    /// Get an adapter for an OpenAI-compatible provider
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

    /// List all registered provider IDs
    pub fn list(&self) -> Vec<&str> {
        self.by_id.keys().map(|s| s.as_str()).collect()
    }

    /// List all registered providers with their details
    pub fn list_providers(&self) -> Vec<&ProviderRecord> {
        self.by_id.values().collect()
    }

    /// Return canonical id for given id or alias; None if unknown
    pub fn canonical_id<'a>(&'a self, id_or_alias: &str) -> Option<&'a str> {
        self.alias_index
            .get(id_or_alias)
            .map(|s| s.as_str())
            .or_else(|| self.by_id.get(id_or_alias).map(|r| r.id.as_str()))
    }

    /// Compare two identifiers for same provider using canonical ids
    pub fn is_same_provider(&self, a: &str, b: &str) -> bool {
        match (self.canonical_id(a), self.canonical_id(b)) {
            (Some(x), Some(y)) => x == y,
            _ => false,
        }
    }
}

// Global provider registry instance
static GLOBAL_REGISTRY: OnceLock<RwLock<ProviderRegistry>> = OnceLock::new();

/// Get the global registry instance (initialized with built-in providers)
pub fn global_registry() -> &'static RwLock<ProviderRegistry> {
    GLOBAL_REGISTRY.get_or_init(|| RwLock::new(ProviderRegistry::with_builtin_providers()))
}

/// Get the global registry handle (recommended for most use cases)
///
/// This returns a `ProviderRegistryHandle` that provides unified access to all providers
/// via the `provider:model` format, similar to Vercel AI SDK.
///
/// # Features
/// - Unified access: `registry.language_model("openai:gpt-4")`
/// - LRU cache with TTL for performance
/// - Automatic middleware injection
/// - Support for all provider types
///
/// # Example
/// ```rust,no_run
/// use siumai::prelude::*;
/// use siumai::registry;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let reg = registry::global();
/// let model = reg.language_model("openai:gpt-4")?;
/// let resp = model.chat(vec![user!("Hello!")]).await?;
/// # Ok(())
/// # }
/// ```
pub fn global() -> &'static entry::ProviderRegistryHandle {
    static GLOBAL_HANDLE: OnceLock<entry::ProviderRegistryHandle> = OnceLock::new();
    GLOBAL_HANDLE.get_or_init(helpers::create_registry_with_defaults)
}

/// Convenience function to get an adapter for an OpenAI-compatible provider
#[cfg(feature = "openai")]
pub fn get_provider_adapter(provider_id: &str) -> Result<Arc<dyn ProviderAdapter>, LlmError> {
    global_registry()
        .read()
        .map_err(|_| LlmError::ConfigurationError("Failed to lock provider registry".to_string()))?
        .get_adapter(provider_id)
}

// Re-export construction helpers for convenience
pub mod factory;

/// Provider registry handle - unified access to all providers
///
/// This module provides a Vercel AI SDK-aligned registry system with:
/// - Unified `provider:model` access pattern
/// - LRU cache with TTL for performance
/// - Automatic middleware injection
/// - Factory-based provider creation
pub mod entry;
/// Provider factory implementations
pub mod factories;
/// Convenience helpers to bootstrap registries with common defaults
pub mod helpers;

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
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
    #[cfg(feature = "google")]
    fn test_registry_resolve_gemini_by_google_alias() {
        let registry = ProviderRegistry::with_builtin_providers();

        // Resolve by primary ID
        let gemini_by_id = registry.resolve("gemini");
        assert!(gemini_by_id.is_some());

        // Resolve by alias
        let gemini_by_alias = registry.resolve("google");
        assert!(gemini_by_alias.is_some());

        // Both should resolve to the same provider
        assert_eq!(gemini_by_id.unwrap().id, gemini_by_alias.unwrap().id);
        assert_eq!(gemini_by_id.unwrap().id, "gemini");
    }

    #[test]
    fn test_registry_resolve_unknown_provider() {
        let registry = ProviderRegistry::with_builtin_providers();
        assert!(registry.resolve("unknown_provider").is_none());
    }
}

// Re-export commonly used items for convenience
pub use entry::{
    BuildContext, EmbeddingModelHandle, ImageModelHandle, LanguageModelHandle, ProviderFactory,
    ProviderRegistryHandle, RegistryOptions, RerankingModelHandle, SpeechModelHandle,
    TranscriptionModelHandle, create_provider_registry,
};
pub use helpers::{create_empty_registry, create_registry_with_defaults};

// External config loaders were removed to keep the core deterministic and simple.
