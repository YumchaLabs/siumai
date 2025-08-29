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

pub mod config;
pub mod images;
pub mod providers;

// Re-export model constants for easy access
pub use providers::models::{deepseek, groq, openrouter, siliconflow, xai};
