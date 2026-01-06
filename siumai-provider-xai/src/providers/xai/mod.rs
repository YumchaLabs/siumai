//! `xAI` Provider Module
//!
//! Thin wrapper around the OpenAI-compatible vendor implementation.
//!
//! # Architecture
//! - `models.rs` - Built-in model catalog (fallback)
//! - `builder.rs` - Builder that delegates to `openai().compatible("xai")`
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
//!     // Use chat capability
//!     let messages = vec![user!("Hello, world!")];
//!     let response = client.chat(messages).await?;
//!
//!     Ok(())
//! }
//! ```

pub mod builder;
/// xAI extension APIs (non-unified surface)
pub mod ext;
pub mod models;

pub use builder::XaiBuilder;

pub type XaiClient = siumai_protocol_openai::providers::openai_compatible::OpenAiCompatibleClient;

// Provider-owned typed options live at the crate root; re-export them under the provider path.
pub use crate::provider_options::{
    SearchMode, SearchSource, SearchSourceType, XaiOptions, XaiSearchParameters,
};
