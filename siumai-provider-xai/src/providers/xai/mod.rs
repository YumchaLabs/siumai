//! `xAI` Provider Module
//!
//! Thin wrapper around the OpenAI-compatible vendor implementation with provider-owned entry types.
//!
//! # Architecture
//! - `builder.rs` - Builder that delegates to `openai().compatible("xai")`
//! - `config.rs`  - Provider-owned config-first surface
//! - `client.rs`  - Provider-owned client wrapper
//! - `models.rs`  - Built-in model catalog (fallback)
//!
//! # Example Usage
//! ```rust,no_run
//! use siumai::models;
//! use siumai::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = Provider::xai()
//!         .api_key("your-api-key")
//!         .model(models::xai::GROK_3_LATEST)
//!         .build()
//!         .await?;
//!
//!     let messages = vec![user!("Hello, world!")];
//!     let response = client.chat(messages).await?;
//!
//!     Ok(())
//! }
//! ```

mod audio;
pub mod builder;
mod client;
pub mod config;
/// xAI extension APIs (non-unified surface)
pub mod ext;
pub mod models;

pub use builder::XaiBuilder;
pub use client::XaiClient;
pub use config::XaiConfig;

// Provider-owned typed options live at the crate root; re-export them under the provider path.
pub use crate::provider_options::{
    SearchMode, SearchSource, SearchSourceType, XaiOptions, XaiSearchParameters, XaiTtsOptions,
};
