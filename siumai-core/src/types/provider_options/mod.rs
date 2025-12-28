//! Type-safe provider-specific options
//!
//! This module provides strongly-typed options for provider-specific features,
//! replacing the weakly-typed `HashMap<String, Value>` approach.
//!
//! # User Extensibility
//!
//! Users can extend the system with custom provider features by implementing
//! the `CustomProviderOptions` trait:
//!
//! ```rust,ignore
//! use siumai::types::{CustomProviderOptions, ChatRequest, ProviderOptions};
//!
//! #[derive(Debug, Clone)]
//! struct MyCustomFeature {
//!     pub custom_param: String,
//! }
//!
//! impl CustomProviderOptions for MyCustomFeature {
//!     fn provider_id(&self) -> &str {
//!         "my-provider"
//!     }
//!
//!     fn to_json(&self) -> Result<serde_json::Value, crate::error::LlmError> {
//!         Ok(serde_json::json!({
//!             "custom_param": self.custom_param
//!         }))
//!     }
//! }
//!
//! // Usage
//! let feature = MyCustomFeature { custom_param: "value".to_string() };
//! let req = ChatRequest::new(messages)
//!     .with_provider_options(ProviderOptions::Custom {
//!         provider_id: "my-provider".to_string(),
//!         options: feature.to_json()?.as_object().unwrap().clone().into_iter().collect(),
//!     });
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Provider-specific modules
pub mod custom;
#[cfg(feature = "groq")]
pub mod groq;
#[cfg(feature = "ollama")]
pub mod ollama;
#[cfg(feature = "xai")]
pub mod xai;

// Re-exports
pub use custom::CustomProviderOptions;
#[cfg(feature = "groq")]
pub use groq::*;
#[cfg(feature = "ollama")]
pub use ollama::*;
#[cfg(feature = "xai")]
pub use xai::*;

/// Type-safe provider-specific options
///
/// This enum provides compile-time type safety for provider-specific features,
/// replacing the previous `ProviderParams` HashMap approach.
///
/// # Example
///
/// ```rust,ignore
/// use siumai::types::{ChatRequest, ProviderOptions, XaiOptions, XaiSearchParameters, SearchMode};
///
/// let req = ChatRequest::new(messages)
///     .with_xai_options(
///         XaiOptions::new()
///             .with_search(XaiSearchParameters {
///                 mode: SearchMode::On,
///                 return_citations: Some(true),
///                 ..Default::default()
///             })
///     );
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(tag = "provider", content = "options")]
pub enum ProviderOptions {
    /// No provider-specific options
    #[default]
    None,
    /// OpenAI-specific options (provider-owned JSON payload)
    ///
    /// This is intentionally kept as a JSON payload to avoid making `siumai-core`
    /// own provider-specific typed option structs.
    #[serde(rename = "openai")]
    #[cfg(feature = "openai")]
    OpenAi(serde_json::Value),
    /// Anthropic-specific options (provider-owned JSON payload)
    ///
    /// This is intentionally kept as a JSON payload to avoid making `siumai-core`
    /// own provider-specific typed option structs.
    #[serde(rename = "anthropic")]
    #[cfg(feature = "anthropic")]
    Anthropic(serde_json::Value),
    /// xAI (Grok) specific options
    #[serde(rename = "xai")]
    #[cfg(feature = "xai")]
    Xai(XaiOptions),
    /// Google Gemini specific options (provider-owned JSON payload)
    ///
    /// This is intentionally kept as a JSON payload to avoid making `siumai-core`
    /// own provider-specific typed option structs.
    #[serde(rename = "gemini")]
    #[cfg(feature = "google")]
    Gemini(serde_json::Value),
    /// Groq-specific options
    #[serde(rename = "groq")]
    #[cfg(feature = "groq")]
    Groq(GroqOptions),
    /// Ollama-specific options
    #[serde(rename = "ollama")]
    #[cfg(feature = "ollama")]
    Ollama(OllamaOptions),
    /// Custom provider options (for user extensions)
    #[serde(rename = "custom")]
    Custom {
        provider_id: String,
        options: HashMap<String, serde_json::Value>,
    },
}

impl ProviderOptions {
    /// Get the provider ID this options is for
    pub fn provider_id(&self) -> Option<&str> {
        match self {
            Self::None => None,
            #[cfg(feature = "openai")]
            Self::OpenAi(_) => Some("openai"),
            #[cfg(feature = "anthropic")]
            Self::Anthropic(_) => Some("anthropic"),
            #[cfg(feature = "xai")]
            Self::Xai(_) => Some("xai"),
            #[cfg(feature = "google")]
            Self::Gemini(_) => Some("gemini"),
            #[cfg(feature = "groq")]
            Self::Groq(_) => Some("groq"),
            #[cfg(feature = "ollama")]
            Self::Ollama(_) => Some("ollama"),
            Self::Custom { provider_id, .. } => Some(provider_id),
        }
    }

    /// Check if options match the given provider
    pub fn is_for_provider(&self, provider_id: &str) -> bool {
        self.provider_id() == Some(provider_id)
    }

    /// Check if this is None
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    /// Create Custom variant from a CustomProviderOptions implementation
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let custom_feature = MyCustomFeature { ... };
    /// let options = ProviderOptions::from_custom(custom_feature)?;
    /// ```
    pub fn from_custom<T: CustomProviderOptions>(
        custom: T,
    ) -> Result<Self, crate::error::LlmError> {
        let provider_id = custom.provider_id().to_string();
        let json = custom.to_json()?;

        // Convert JSON object to HashMap
        let options = if let serde_json::Value::Object(map) = json {
            map.into_iter().collect()
        } else {
            HashMap::new()
        };

        Ok(Self::Custom {
            provider_id,
            options,
        })
    }

    /// Convert this typed options into an open `ProviderOptionsMap` entry.
    ///
    /// This is a compatibility bridge during the refactor: typed options are still supported,
    /// but providers should gradually migrate to reading the open `providerOptions` map.
    pub fn to_provider_options_map_entry(&self) -> Option<(String, serde_json::Value)> {
        match self {
            Self::None => None,
            #[cfg(feature = "openai")]
            Self::OpenAi(value) => Some(("openai".to_string(), value.clone())),
            #[cfg(feature = "anthropic")]
            Self::Anthropic(value) => Some(("anthropic".to_string(), value.clone())),
            #[cfg(feature = "xai")]
            Self::Xai(opts) => serde_json::to_value(opts).ok().map(|v| ("xai".to_string(), v)),
            #[cfg(feature = "google")]
            Self::Gemini(value) => Some(("gemini".to_string(), value.clone())),
            #[cfg(feature = "groq")]
            Self::Groq(opts) => serde_json::to_value(opts)
                .ok()
                .map(|v| ("groq".to_string(), v)),
            #[cfg(feature = "ollama")]
            Self::Ollama(opts) => serde_json::to_value(opts)
                .ok()
                .map(|v| ("ollama".to_string(), v)),
            Self::Custom {
                provider_id,
                options,
            } => {
                let mut map = serde_json::Map::new();
                for (k, v) in options {
                    map.insert(k.clone(), v.clone());
                }
                Some((provider_id.clone(), serde_json::Value::Object(map)))
            }
        }
    }
}
