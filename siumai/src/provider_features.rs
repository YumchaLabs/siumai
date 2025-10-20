//! Provider-Specific Features
//!
//! This module provides a unified interface for managing provider-specific
//! features and capabilities across different AI providers.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::LlmError;

/// Provider-specific feature configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderFeatures {
    /// Provider name
    pub provider: String,
    /// Feature configurations
    pub features: HashMap<String, FeatureConfig>,
}

impl ProviderFeatures {
    /// Create a new provider features configuration
    pub fn new<S: Into<String>>(provider: S) -> Self {
        Self {
            provider: provider.into(),
            features: HashMap::new(),
        }
    }

    /// Add a feature configuration
    pub fn with_feature<S: Into<String>>(mut self, name: S, config: FeatureConfig) -> Self {
        self.features.insert(name.into(), config);
        self
    }

    /// Enable a simple feature
    pub fn enable_feature<S: Into<String>>(mut self, name: S) -> Self {
        self.features
            .insert(name.into(), FeatureConfig::Boolean(true));
        self
    }

    /// Disable a feature
    pub fn disable_feature<S: Into<String>>(mut self, name: S) -> Self {
        self.features
            .insert(name.into(), FeatureConfig::Boolean(false));
        self
    }

    /// Get a feature configuration
    pub fn get_feature(&self, name: &str) -> Option<&FeatureConfig> {
        self.features.get(name)
    }

    /// Check if a feature is enabled
    pub fn is_feature_enabled(&self, name: &str) -> bool {
        match self.get_feature(name) {
            Some(FeatureConfig::Boolean(enabled)) => *enabled,
            Some(FeatureConfig::Object(obj)) => obj
                .get("enabled")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            _ => false,
        }
    }

    /// Convert to request parameters
    pub fn to_request_params(&self) -> HashMap<String, serde_json::Value> {
        let mut params = HashMap::new();

        for (name, config) in &self.features {
            match config {
                FeatureConfig::Boolean(enabled) => {
                    if *enabled {
                        params.insert(name.clone(), serde_json::Value::Bool(true));
                    }
                }
                FeatureConfig::String(value) => {
                    params.insert(name.clone(), serde_json::Value::String(value.clone()));
                }
                FeatureConfig::Number(value) => {
                    if let Some(num) = serde_json::Number::from_f64(*value) {
                        params.insert(name.clone(), serde_json::Value::Number(num));
                    }
                }
                FeatureConfig::Object(obj) => {
                    params.insert(name.clone(), obj.clone());
                }
            }
        }

        params
    }
}

/// Feature configuration types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureConfig {
    /// Boolean feature (enabled/disabled)
    Boolean(bool),
    /// String configuration
    String(String),
    /// Numeric configuration
    Number(f64),
    /// Complex object configuration
    Object(serde_json::Value),
}

impl FeatureConfig {
    /// Create a boolean feature config
    pub const fn boolean(enabled: bool) -> Self {
        Self::Boolean(enabled)
    }

    /// Create a string feature config
    pub fn string<S: Into<String>>(value: S) -> Self {
        Self::String(value.into())
    }

    /// Create a numeric feature config
    pub const fn number(value: f64) -> Self {
        Self::Number(value)
    }

    /// Create an object feature config
    pub const fn object(value: serde_json::Value) -> Self {
        Self::Object(value)
    }
}

/// Provider-specific feature registry
pub struct ProviderFeatureRegistry {
    /// Registered features by provider
    features: HashMap<String, HashMap<String, FeatureDefinition>>,
}

impl ProviderFeatureRegistry {
    /// Create a new feature registry
    pub fn new() -> Self {
        let mut registry = Self {
            features: HashMap::new(),
        };
        registry.register_default_features();
        registry
    }

    /// Register a feature for a provider
    pub fn register_feature<S: Into<String>>(
        &mut self,
        provider: S,
        name: S,
        definition: FeatureDefinition,
    ) {
        let provider_key = provider.into();
        let feature_name = name.into();

        self.features
            .entry(provider_key)
            .or_default()
            .insert(feature_name, definition);
    }

    /// Get feature definition
    pub fn get_feature_definition(&self, provider: &str, name: &str) -> Option<&FeatureDefinition> {
        self.features
            .get(provider)
            .and_then(|provider_features| provider_features.get(name))
    }

    /// Get all features for a provider
    pub fn get_provider_features(
        &self,
        provider: &str,
    ) -> Option<&HashMap<String, FeatureDefinition>> {
        self.features.get(provider)
    }

    /// Validate feature configuration
    pub fn validate_feature_config(
        &self,
        provider: &str,
        name: &str,
        config: &FeatureConfig,
    ) -> Result<(), LlmError> {
        if let Some(definition) = self.get_feature_definition(provider, name) {
            definition.validate(config)
        } else {
            Err(LlmError::InvalidParameter(format!(
                "Unknown feature '{name}' for provider '{provider}'"
            )))
        }
    }

    /// Register default features for all providers
    fn register_default_features(&mut self) {
        // OpenAI features
        self.register_feature(
            "openai",
            "structured_output",
            FeatureDefinition::new("Structured Output")
                .with_description("Enable structured JSON output with schema validation")
                .with_config_type(FeatureConfigType::Object),
        );

        self.register_feature(
            "openai",
            "web_search",
            FeatureDefinition::new("Web Search")
                .with_description("Enable built-in web search via Responses API")
                .with_config_type(FeatureConfigType::Boolean),
        );

        self.register_feature(
            "openai",
            "file_search",
            FeatureDefinition::new("File Search")
                .with_description("Enable file search with vector stores")
                .with_config_type(FeatureConfigType::Object),
        );

        // Anthropic features
        self.register_feature(
            "anthropic",
            "prompt_caching",
            FeatureDefinition::new("Prompt Caching")
                .with_description("Enable prompt caching for cost reduction")
                .with_config_type(FeatureConfigType::Object),
        );

        self.register_feature(
            "anthropic",
            "thinking_mode",
            FeatureDefinition::new("Thinking Mode")
                .with_description("Enable access to Claude's reasoning process")
                .with_config_type(FeatureConfigType::Object),
        );

        // Gemini features
        self.register_feature(
            "gemini",
            "code_execution",
            FeatureDefinition::new("Code Execution")
                .with_description("Enable Python code execution")
                .with_config_type(FeatureConfigType::Object),
        );

        self.register_feature(
            "gemini",
            "search_grounding",
            FeatureDefinition::new("Search Grounding")
                .with_description("Enable search-augmented generation")
                .with_config_type(FeatureConfigType::Boolean),
        );
    }
}

impl Default for ProviderFeatureRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Feature definition
#[derive(Debug, Clone)]
pub struct FeatureDefinition {
    /// Feature name
    pub name: String,
    /// Feature description
    pub description: String,
    /// Expected configuration type
    pub config_type: FeatureConfigType,
    /// Whether the feature is experimental
    pub experimental: bool,
    /// Required API version
    pub min_api_version: Option<String>,
}

impl FeatureDefinition {
    /// Create a new feature definition
    pub fn new<S: Into<String>>(name: S) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            config_type: FeatureConfigType::Boolean,
            experimental: false,
            min_api_version: None,
        }
    }

    /// Set description
    pub fn with_description<S: Into<String>>(mut self, description: S) -> Self {
        self.description = description.into();
        self
    }

    /// Set configuration type
    pub const fn with_config_type(mut self, config_type: FeatureConfigType) -> Self {
        self.config_type = config_type;
        self
    }

    /// Mark as experimental
    pub const fn experimental(mut self) -> Self {
        self.experimental = true;
        self
    }

    /// Set minimum API version
    pub fn with_min_api_version<S: Into<String>>(mut self, version: S) -> Self {
        self.min_api_version = Some(version.into());
        self
    }

    /// Validate feature configuration
    pub fn validate(&self, config: &FeatureConfig) -> Result<(), LlmError> {
        match (&self.config_type, config) {
            (FeatureConfigType::Boolean, FeatureConfig::Boolean(_)) => Ok(()),
            (FeatureConfigType::String, FeatureConfig::String(_)) => Ok(()),
            (FeatureConfigType::Number, FeatureConfig::Number(_)) => Ok(()),
            (FeatureConfigType::Object, FeatureConfig::Object(_)) => Ok(()),
            _ => Err(LlmError::InvalidParameter(format!(
                "Invalid configuration type for feature '{}'. Expected {:?}, got {:?}",
                self.name, self.config_type, config
            ))),
        }
    }
}

/// Feature configuration type
#[derive(Debug, Clone)]
pub enum FeatureConfigType {
    /// Boolean configuration
    Boolean,
    /// String configuration
    String,
    /// Numeric configuration
    Number,
    /// Object configuration
    Object,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_features() {
        let features = ProviderFeatures::new("openai")
            .enable_feature("structured_output")
            .with_feature("web_search", FeatureConfig::boolean(true))
            .with_feature("temperature", FeatureConfig::number(0.7));

        assert!(features.is_feature_enabled("structured_output"));
        assert!(features.is_feature_enabled("web_search"));
        assert!(!features.is_feature_enabled("nonexistent"));

        let params = features.to_request_params();
        assert_eq!(
            params.get("structured_output"),
            Some(&serde_json::Value::Bool(true))
        );
        assert_eq!(
            params.get("temperature"),
            Some(&serde_json::Value::Number(
                serde_json::Number::from_f64(0.7).unwrap()
            ))
        );
    }

    #[test]
    fn test_feature_registry() {
        let registry = ProviderFeatureRegistry::new();

        let openai_features = registry.get_provider_features("openai").unwrap();
        assert!(openai_features.contains_key("structured_output"));
        assert!(openai_features.contains_key("web_search"));

        let anthropic_features = registry.get_provider_features("anthropic").unwrap();
        assert!(anthropic_features.contains_key("prompt_caching"));
        assert!(anthropic_features.contains_key("thinking_mode"));
    }

    #[test]
    fn test_feature_validation() {
        let registry = ProviderFeatureRegistry::new();

        // Valid configuration
        let config = FeatureConfig::boolean(true);
        assert!(
            registry
                .validate_feature_config("openai", "web_search", &config)
                .is_ok()
        );

        // Invalid configuration type
        let config = FeatureConfig::string("invalid");
        assert!(
            registry
                .validate_feature_config("openai", "web_search", &config)
                .is_err()
        );

        // Unknown feature
        let config = FeatureConfig::boolean(true);
        assert!(
            registry
                .validate_feature_config("openai", "unknown_feature", &config)
                .is_err()
        );
    }
}
