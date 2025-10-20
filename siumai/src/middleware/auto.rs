//! Automatic middleware configuration based on provider and model.
//!
//! This module provides automatic middleware addition based on the provider
//! and model being used, similar to Cherry Studio's approach.

use super::{
    builder::MiddlewareBuilder, language_model::LanguageModelMiddleware,
    presets::ExtractReasoningMiddleware,
};
use std::sync::Arc;

/// Configuration for automatic middleware selection.
#[derive(Debug, Clone)]
pub struct MiddlewareConfig {
    /// Provider ID (e.g., "openai", "anthropic", "gemini")
    pub provider_id: String,
    /// Model ID (e.g., "gpt-4", "claude-3", "gemini-2.5-pro")
    pub model_id: String,
    /// Whether to enable reasoning/thinking extraction
    pub enable_reasoning: bool,
    /// Whether the output is streamed
    pub stream_output: bool,
}

impl MiddlewareConfig {
    /// Create a new middleware configuration.
    pub fn new(provider_id: impl Into<String>, model_id: impl Into<String>) -> Self {
        Self {
            provider_id: provider_id.into(),
            model_id: model_id.into(),
            enable_reasoning: true, // Default: enabled
            stream_output: true,    // Default: streaming
        }
    }

    /// Set whether to enable reasoning extraction.
    pub fn with_enable_reasoning(mut self, enable: bool) -> Self {
        self.enable_reasoning = enable;
        self
    }

    /// Set whether the output is streamed.
    pub fn with_stream_output(mut self, stream: bool) -> Self {
        self.stream_output = stream;
        self
    }
}

/// Build automatic middlewares based on configuration.
///
/// This function automatically adds appropriate middlewares based on the
/// provider and model being used.
///
/// # Arguments
///
/// * `config` - The middleware configuration
///
/// # Returns
///
/// A `MiddlewareBuilder` with automatically added middlewares.
///
/// # Example
///
/// ```rust,ignore
/// use siumai::middleware::auto::{MiddlewareConfig, build_auto_middlewares};
///
/// let config = MiddlewareConfig::new("openai", "o1-preview");
/// let builder = build_auto_middlewares(&config);
/// let middlewares = builder.build();
/// ```
pub fn build_auto_middlewares(config: &MiddlewareConfig) -> MiddlewareBuilder {
    let mut builder = MiddlewareBuilder::new();

    // 1. Add provider-specific middlewares
    add_provider_specific_middlewares(&mut builder, config);

    // 2. Add model-specific middlewares
    add_model_specific_middlewares(&mut builder, config);

    // 3. Add feature-specific middlewares
    // (e.g., simulate streaming for non-streaming output)
    // This can be added in the future

    builder
}

/// Build automatic middlewares and return as a vector.
///
/// This is a convenience function that builds the middleware chain and
/// returns it as a vector, ready to be used with `with_model_middlewares()`.
///
/// # Arguments
///
/// * `provider_id` - The provider identifier (e.g., "openai", "anthropic")
/// * `model_id` - The model identifier (e.g., "gpt-4", "o1-preview")
///
/// # Returns
///
/// A vector of middlewares ready to be installed on a client.
///
/// # Example
///
/// ```rust,ignore
/// use siumai::middleware::auto::build_auto_middlewares_vec;
///
/// let middlewares = build_auto_middlewares_vec("openai", "o1-preview");
/// let client = client.with_model_middlewares(middlewares);
/// ```
pub fn build_auto_middlewares_vec(
    provider_id: &str,
    model_id: &str,
) -> Vec<Arc<dyn LanguageModelMiddleware>> {
    let config = MiddlewareConfig::new(provider_id, model_id);
    build_auto_middlewares(&config).build()
}

/// Add provider-specific middlewares.
fn add_provider_specific_middlewares(builder: &mut MiddlewareBuilder, config: &MiddlewareConfig) {
    let provider_lower = config.provider_id.to_lowercase();

    match provider_lower.as_str() {
        "openai" | "azure-openai" | "openai-compatible" => {
            // OpenAI and compatible providers
            if config.enable_reasoning {
                builder.add(
                    "extract-reasoning",
                    Arc::new(ExtractReasoningMiddleware::for_model(&config.model_id)),
                );
            }
        }
        "anthropic" => {
            // Anthropic-specific middlewares
            // Anthropic uses JSON fields for thinking, which is already handled
            // by the provider transformer, but we still add the middleware for
            // fallback tag extraction
            if config.enable_reasoning {
                builder.add(
                    "extract-reasoning",
                    Arc::new(ExtractReasoningMiddleware::for_model(&config.model_id)),
                );
            }
        }
        "gemini" | "google" => {
            // Gemini-specific middlewares
            if config.enable_reasoning {
                builder.add(
                    "extract-reasoning",
                    Arc::new(ExtractReasoningMiddleware::for_model(&config.model_id)),
                );
            }
        }
        "xai" => {
            // xAI-specific middlewares
            if config.enable_reasoning {
                builder.add(
                    "extract-reasoning",
                    Arc::new(ExtractReasoningMiddleware::for_model(&config.model_id)),
                );
            }
        }
        _ => {
            // Default: add reasoning extraction for all providers
            if config.enable_reasoning {
                builder.add(
                    "extract-reasoning",
                    Arc::new(ExtractReasoningMiddleware::for_model(&config.model_id)),
                );
            }
        }
    }
}

/// Add model-specific middlewares.
fn add_model_specific_middlewares(_builder: &mut MiddlewareBuilder, _config: &MiddlewareConfig) {
    // This can be used to add middlewares based on specific model characteristics
    // For example:
    // - Image generation models
    // - Multimodal models
    // - Models with specific capabilities

    // Currently, most model-specific logic is handled in provider-specific section
    // This is a placeholder for future extensions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_auto_middlewares_openai() {
        let config = MiddlewareConfig::new("openai", "o1-preview");
        let builder = build_auto_middlewares(&config);

        assert!(builder.has("extract-reasoning"));
        assert_eq!(builder.len(), 1);
    }

    #[test]
    fn test_build_auto_middlewares_gemini() {
        let config = MiddlewareConfig::new("gemini", "gemini-2.5-pro");
        let builder = build_auto_middlewares(&config);

        assert!(builder.has("extract-reasoning"));
        assert_eq!(builder.len(), 1);
    }

    #[test]
    fn test_build_auto_middlewares_disabled_reasoning() {
        let config = MiddlewareConfig::new("openai", "gpt-4").with_enable_reasoning(false);
        let builder = build_auto_middlewares(&config);

        assert!(!builder.has("extract-reasoning"));
        assert_eq!(builder.len(), 0);
    }

    #[test]
    fn test_middleware_config_builder() {
        let config = MiddlewareConfig::new("openai", "gpt-4")
            .with_enable_reasoning(false)
            .with_stream_output(false);

        assert_eq!(config.provider_id, "openai");
        assert_eq!(config.model_id, "gpt-4");
        assert!(!config.enable_reasoning);
        assert!(!config.stream_output);
    }
}
