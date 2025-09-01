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
pub mod adapter;
pub mod builder;
pub mod default_models;
pub mod openai_client;
pub mod openai_config;
pub mod streaming;
pub mod types;

// Re-export model constants for easy access
pub use providers::models::{deepseek, groq, openrouter, siliconflow, xai};

// Re-export new adapter system
pub use adapter::ProviderAdapter;
pub use builder::OpenAiCompatibleBuilder;
pub use openai_client::OpenAiCompatibleClient;
pub use openai_config::OpenAiCompatibleConfig;
pub use types::{FieldMappings, ModelConfig, RequestType};
