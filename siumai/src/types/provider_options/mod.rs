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
pub mod anthropic;
pub mod custom;
pub mod gemini;
pub mod groq;
pub mod ollama;
pub mod openai;
pub mod xai;

// Re-exports
pub use anthropic::*;
pub use custom::CustomProviderOptions;
pub use gemini::*;
pub use groq::*;
pub use ollama::*;
pub use openai::*;
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
    /// OpenAI-specific options
    #[serde(rename = "openai")]
    OpenAi(Box<OpenAiOptions>),
    /// Anthropic-specific options
    #[serde(rename = "anthropic")]
    Anthropic(AnthropicOptions),
    /// xAI (Grok) specific options
    #[serde(rename = "xai")]
    Xai(XaiOptions),
    /// Google Gemini specific options
    #[serde(rename = "gemini")]
    Gemini(GeminiOptions),
    /// Groq-specific options
    #[serde(rename = "groq")]
    Groq(GroqOptions),
    /// Ollama-specific options
    #[serde(rename = "ollama")]
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
            Self::OpenAi(_) => Some("openai"),
            Self::Anthropic(_) => Some("anthropic"),
            Self::Xai(_) => Some("xai"),
            Self::Gemini(_) => Some("gemini"),
            Self::Groq(_) => Some("groq"),
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
}
