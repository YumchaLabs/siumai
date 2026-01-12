//! `DeepSeek` Provider Module
//!
//! Thin wrapper around the OpenAI-compatible vendor implementation.
//!
//! # Architecture
//! - `builder.rs` - Builder that delegates to `openai().compatible("deepseek")`
//! - `models.rs`  - Model constants (re-exported from OpenAI-compatible catalog)
//!
//! # Example Usage
//! ```rust,no_run
//! use siumai::prelude::*;
//! use siumai::models;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = Provider::deepseek()
//!         .api_key("your-api-key")
//!         .model(models::openai_compatible::deepseek::CHAT)
//!         .build()
//!         .await?;
//!
//!     let messages = vec![user!("Hello, DeepSeek!")];
//!     let response = client.chat(messages).await?;
//!     println!("{}", response.content_text().unwrap_or_default());
//!     Ok(())
//! }
//! ```

pub mod builder;
pub mod models;

pub use builder::DeepSeekBuilder;

pub type DeepSeekClient =
    siumai_provider_openai_compatible::providers::openai_compatible::OpenAiCompatibleClient;
