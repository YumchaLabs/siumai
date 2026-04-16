//! `Groq` Provider Module
//!
//! Thin wrapper around the OpenAI-compatible vendor implementation.
//!
//! # Architecture
//! - `models.rs` - Built-in model catalog (fallback)
//! - `builder.rs` - Builder that delegates to `openai().compatible("groq")`
//!
//! # Example Usage
//! ```rust,no_run
//! use siumai::models;
//! use siumai::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = LlmBuilder::new()
//!         .groq()
//!         .api_key("your-api-key")
//!         .model(models::groq::LLAMA_3_3_70B_VERSATILE)
//!         .build()
//!         .await?;
//!
//!     // Use chat capability
//!     let messages = vec![user!("Hello, world!")];
//!     let response = client.chat(messages).await?;
//!
//!     // Use audio capability (if available)
//!     // let audio_data = client.speech("Hello, world!").await?;
//!
//!     Ok(())
//! }
//! ```

pub mod builder;
mod client;
pub mod config;
pub mod ext;
mod middleware;
pub mod models;
pub mod spec;
mod transformers;
mod utils;
#[allow(deprecated)]
pub use crate::provider_options::GroqProviderOptions;
pub use crate::provider_options::{
    GroqLanguageModelOptions, GroqOptions, GroqReasoningEffort, GroqReasoningFormat,
    GroqServiceTier, GroqTranscriptionModelOptions,
};
pub use builder::GroqBuilder;
pub use client::GroqClient;
pub use config::GroqConfig;
pub use ext::GroqChatRequestExt;
pub use spec::GroqSpec;
