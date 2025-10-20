//! Unified Provider Registry
//!
//! A configuration-driven registry that unifies provider lookup and adapter wiring.
//! This registry combines native providers and OpenAI-compatible providers into a
//! single, unified interface inspired by Cherry Studio's design.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::error::LlmError;
use crate::traits::ProviderCapabilities;

#[cfg(feature = "openai")]
use crate::providers::openai_compatible::adapter::ProviderAdapter;
#[cfg(feature = "openai")]
use crate::providers::openai_compatible::registry::{ConfigurableAdapter, ProviderConfig};

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
        // OpenAI
        #[cfg(feature = "openai")]
        self.register_native(
            "openai",
            "OpenAI",
            Some("https://api.openai.com/v1".to_string()),
            ProviderCapabilities::new()
                .with_chat()
                .with_streaming()
                .with_tools()
                .with_vision()
                .with_embedding()
                .with_custom_feature("image_generation", true)
                .with_custom_feature("audio", true)
                .with_custom_feature("files", true)
                .with_custom_feature("rerank", true),
        );

        // Anthropic
        #[cfg(feature = "anthropic")]
        self.register_native(
            "anthropic",
            "Anthropic",
            Some("https://api.anthropic.com".to_string()),
            ProviderCapabilities::new()
                .with_chat()
                .with_streaming()
                .with_tools()
                .with_vision()
                .with_custom_feature("thinking", true),
        );

        // Google Gemini
        #[cfg(feature = "google")]
        self.register_native(
            "gemini",
            "Google Gemini",
            Some("https://generativelanguage.googleapis.com".to_string()),
            ProviderCapabilities::new()
                .with_chat()
                .with_streaming()
                .with_tools()
                .with_vision()
                .with_embedding()
                .with_custom_feature("thinking", true),
        );

        // Groq
        #[cfg(feature = "groq")]
        self.register_native(
            "groq",
            "Groq",
            Some("https://api.groq.com/openai/v1".to_string()),
            ProviderCapabilities::new()
                .with_chat()
                .with_streaming()
                .with_tools(),
        );

        // xAI
        #[cfg(feature = "xai")]
        self.register_native(
            "xai",
            "xAI",
            Some("https://api.x.ai/v1".to_string()),
            ProviderCapabilities::new()
                .with_chat()
                .with_streaming()
                .with_tools()
                .with_custom_feature("thinking", true),
        );

        // Ollama
        #[cfg(feature = "ollama")]
        self.register_native(
            "ollama",
            "Ollama",
            Some("http://localhost:11434".to_string()),
            ProviderCapabilities::new()
                .with_chat()
                .with_streaming()
                .with_tools()
                .with_embedding(),
        );
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
        self.by_id.insert(record.id.clone(), record);
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

    /// Register a custom OpenAI-compatible provider
    #[cfg(feature = "openai")]
    pub fn register_custom_provider(&mut self, config: ProviderConfig) {
        let _ = self.register_openai_compatible_from_config(config);
    }

    /// Resolve a provider record by id or alias
    pub fn resolve(&self, id_or_alias: &str) -> Option<&ProviderRecord> {
        if let Some(rec) = self.by_id.get(id_or_alias) {
            return Some(rec);
        }
        // Search by alias
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
}

// Global provider registry instance
static GLOBAL_REGISTRY: OnceLock<Mutex<ProviderRegistry>> = OnceLock::new();

/// Get the global registry instance (initialized with built-in providers)
pub fn global_registry() -> &'static Mutex<ProviderRegistry> {
    GLOBAL_REGISTRY.get_or_init(|| Mutex::new(ProviderRegistry::with_builtin_providers()))
}

/// Convenience function to get an adapter for an OpenAI-compatible provider
#[cfg(feature = "openai")]
pub fn get_provider_adapter(provider_id: &str) -> Result<Arc<dyn ProviderAdapter>, LlmError> {
    global_registry()
        .lock()
        .map_err(|_| LlmError::ConfigurationError("Failed to lock provider registry".to_string()))?
        .get_adapter(provider_id)
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

// Re-export commonly used items for convenience
pub use entry::{
    ProviderFactory, ProviderRegistryHandle, RegistryOptions, create_provider_registry,
};
pub use helpers::{create_empty_registry, create_registry_with_defaults};

// External config loaders were removed to keep the core deterministic and simple.
