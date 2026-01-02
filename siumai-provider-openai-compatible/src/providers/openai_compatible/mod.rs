//! OpenAI-Compatible Provider Interface
//!
//! This module provides model constants for OpenAI-compatible providers.
//! These providers now use the OpenAI client directly with custom base URLs.
//!
//! # Usage
//! ```rust,no_run
//! use siumai::prelude::*;
//! use siumai::providers::openai_compatible::{deepseek, openrouter};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Recommended: enter from `openai()` and switch to a vendor preset
//!     // (keeps the mental model "OpenAI-like protocol family").
//!     let siliconflow = LlmBuilder::new()
//!         .openai()
//!         .compatible("siliconflow")
//!         .api_key("your-api-key")
//!         .model(deepseek::CHAT) // or a vendor-specific model id
//!         .build()
//!         .await?;
//!
//!     // DeepSeek using OpenAI client with DeepSeek endpoint
//!     let deepseek = LlmBuilder::new()
//!         .deepseek()
//!         .api_key("your-api-key")
//!         .model(deepseek::REASONER)  // Using model constant
//!         .build()
//!         .await?;
//!
//!     // OpenRouter using OpenAI client with OpenRouter endpoint
//!     let openrouter = LlmBuilder::new()
//!         .openrouter()
//!         .api_key("your-api-key")
//!         .model(openrouter::openai::GPT_4)  // Using model constant
//!         .build()
//!         .await?;
//!
//!     // Other providers using OpenAI client with custom base URL
//!     let groq = LlmBuilder::new()
//!         .openai()
//!         .base_url("https://api.groq.com/openai/v1")
//!         .api_key("your-api-key")
//!         .model("llama-3.1-70b-versatile")
//!         .build()
//!         .await?;
//!
//!     Ok(())
//! }
//! ```

pub mod providers;

// New adapter system modules
pub mod builder;
pub mod config;
pub mod default_models;
pub mod openai_client;
pub mod spec;
// Macro list for generating builder methods across modules
pub mod builder_list;

// Protocol (standard) modules live under `standards::openai::compat`.
// Keep these module paths for backward compatibility.
pub mod adapter {
    pub use crate::standards::openai::compat::adapter::*;
}
pub mod openai_config {
    pub use crate::standards::openai::compat::openai_config::*;
}
pub mod streaming {
    pub use crate::standards::openai::compat::streaming::*;
}
pub mod transformers {
    pub use crate::standards::openai::compat::transformers::*;
}
pub mod types {
    pub use crate::standards::openai::compat::types::*;
}

// Backward compatible path: `providers::openai_compatible::registry::*`
pub mod registry {
    pub use crate::standards::openai::compat::provider_registry::*;
}

// Re-export model constants for easy access
pub use providers::models::{deepseek, groq, moonshot, openrouter, siliconflow, xai};

// Re-export new adapter system
pub use crate::standards::openai::compat::provider_registry::{
    ConfigurableAdapter, ProviderConfig,
};
pub use adapter::{ProviderAdapter, ProviderCompatibility};
pub use builder::OpenAiCompatibleBuilder;
pub use config::{
    get_builtin_providers, get_provider_config, list_provider_ids, provider_supports_capability,
};
pub use openai_client::OpenAiCompatibleClient;
pub use openai_config::OpenAiCompatibleConfig;
pub use types::{FieldMappings, ModelConfig, RequestType};

// Test modules
#[cfg(test)]
mod tests {
    pub mod base_url_tests;
}
