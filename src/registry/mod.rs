//! ProviderRegistry v2 (skeleton)
//!
//! A configuration-driven registry that unifies provider lookup and adapter wiring.
//! This is a lightweight skeleton inspired by Cherry Studio's registry, intended
//! to gradually replace scattered builder-time branching in SiumaiBuilder.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::error::LlmError;
use crate::traits::ProviderCapabilities;

#[cfg(feature = "openai")]
use crate::providers::openai_compatible::adapter::ProviderAdapter;
#[cfg(feature = "openai")]
use crate::providers::openai_compatible::registry as compat_registry;

/// Unified provider record maintained by the registry
#[derive(Debug, Clone)]
pub struct ProviderRecord {
    pub id: String,
    pub name: String,
    pub base_url: Option<String>,
    pub capabilities: ProviderCapabilities,
    #[cfg(feature = "openai")]
    pub adapter: Option<std::sync::Arc<dyn ProviderAdapter>>, // for OpenAI-compatible
    pub aliases: Vec<String>,
    /// Optional model id prefixes that hint routing for this provider
    pub model_prefixes: Vec<String>,
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
}

/// Provider registry v2
#[derive(Default)]
pub struct ProviderRegistryV2 {
    by_id: HashMap<String, ProviderRecord>,
}

impl ProviderRegistryV2 {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a prebuilt record
    pub fn register(&mut self, record: ProviderRecord) {
        self.by_id.insert(record.id.clone(), record);
    }

    /// Register an OpenAI-compatible provider by id using the built-in compat registry
    #[cfg(feature = "openai")]
    pub fn register_openai_compatible(&mut self, provider_id: &str) -> Result<(), LlmError> {
        // Pull config from compat registry
        let config = crate::providers::openai_compatible::config::get_provider_config(provider_id)
            .ok_or_else(|| {
                LlmError::ConfigurationError(format!(
                    "Unknown OpenAI-compatible provider: {}",
                    provider_id
                ))
            })?;

        let adapter = compat_registry::get_provider_adapter(provider_id)?;
        // Map from compat adapter into our ProviderCapabilities
        let capabilities = adapter.capabilities();

        let record = ProviderRecord {
            id: provider_id.to_string(),
            name: config.name,
            base_url: Some(config.base_url),
            capabilities,
            adapter: Some(adapter),
            aliases: vec![],
            model_prefixes: vec![],
        };

        self.register(record);
        Ok(())
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
        };
        self.register(record);
    }

    /// Resolve a provider record by id or alias (simple implementation)
    pub fn resolve(&self, id_or_alias: &str) -> Option<&ProviderRecord> {
        if let Some(rec) = self.by_id.get(id_or_alias) {
            return Some(rec);
        }
        // naive alias search
        self.by_id
            .values()
            .find(|rec| rec.aliases.iter().any(|a| a == id_or_alias))
    }

    /// Resolve by model id prefix (best-effort). Returns the first match.
    pub fn resolve_for_model(&self, model_id: &str) -> Option<&ProviderRecord> {
        self.by_id
            .values()
            .find(|rec| rec.model_prefixes.iter().any(|p| model_id.starts_with(p)))
    }

    /// List registered providers
    pub fn list(&self) -> Vec<&str> {
        self.by_id.keys().map(|s| s.as_str()).collect()
    }
}

static GLOBAL: OnceLock<Mutex<ProviderRegistryV2>> = OnceLock::new();

/// Get global registry instance
pub fn global_registry() -> &'static Mutex<ProviderRegistryV2> {
    GLOBAL.get_or_init(|| Mutex::new(ProviderRegistryV2::new()))
}

// Re-export construction helpers for convenience
#[cfg(feature = "openai")]
pub mod factory;

/// Experimental registry entry (Iteration A): minimal handle + options
pub mod entry;
/// Provider factory implementations
pub mod factories;
/// Convenience helpers to bootstrap registries with common defaults
pub mod helpers;

// External config loaders were removed to keep the core deterministic and simple.
