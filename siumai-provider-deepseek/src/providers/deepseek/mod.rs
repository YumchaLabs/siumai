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
mod client;
pub mod config;
pub mod ext;
pub mod models;
pub mod spec;

pub use builder::DeepSeekBuilder;
pub use client::DeepSeekClient;
pub use config::DeepSeekConfig;

// Provider-owned typed options live at the crate root; re-export them under the provider path.
pub use crate::provider_options::DeepSeekOptions;
pub use spec::DeepSeekSpec;
