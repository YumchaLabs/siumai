//! OpenAI-Compatible Provider Interface
//!
//! This module provides model constants for OpenAI-compatible providers.
//! These providers use a dedicated OpenAI-compatible client (`OpenAiCompatibleClient`) that
//! applies provider adapters (field mappings, reasoning extraction, etc.).
//!
//! # Usage
//! ```rust,no_run
//! use siumai_provider_openai_compatible::providers::openai_compatible::{deepseek, OpenAiCompatibleClient};
//! use siumai_provider_openai_compatible::types::ChatRequest;
//! use siumai_provider_openai_compatible::{text, user};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Config-first construction (recommended):
//!     // Reads `DEEPSEEK_API_KEY` by default for provider_id = "deepseek".
//!     let client = OpenAiCompatibleClient::from_builtin_env("deepseek", Some(deepseek::CHAT)).await?;
//!
//!     // Invocation goes through the stable model-family APIs:
//!     let req = ChatRequest::new(vec![user!("hi")]);
//!     let _resp = text::generate(&client, req, text::GenerateOptions::default()).await?;
//!
//!     Ok(())
//! }
//! ```

pub mod providers;

// New adapter system modules
pub mod builder;
pub mod config;
pub mod default_models;
pub mod middleware;
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
pub use middleware::OpenAiCompatibleToolWarningsMiddleware;
pub use openai_client::OpenAiCompatibleClient;
pub use openai_config::OpenAiCompatibleConfig;
pub use types::{FieldMappings, ModelConfig, RequestType};

// Test modules
#[cfg(test)]
mod tests {
    pub mod base_url_tests;
}
