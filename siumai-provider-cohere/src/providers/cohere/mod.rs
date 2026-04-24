//! `Cohere` provider module.
//!
//! This module exposes a provider-owned config/client/builder surface for the
//! native Cohere chat + embedding + rerank integration.

pub mod builder;
pub mod client;
pub mod config;
pub mod ext;
pub mod models;
pub mod settings;

pub use builder::CohereBuilder;
pub use client::CohereClient;
pub use config::CohereConfig;
pub use ext::{CohereChatRequestExt, CohereEmbeddingRequestExt, CohereRerankRequestExt};
pub use settings::CohereProviderSettings;
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
