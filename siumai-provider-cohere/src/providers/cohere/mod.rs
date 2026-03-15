//! `Cohere` provider module.
//!
//! This module exposes a provider-owned config/client/builder surface for the
//! rerank-only Cohere integration.

pub mod builder;
pub mod client;
pub mod config;
pub mod ext;

pub use builder::CohereBuilder;
pub use client::CohereClient;
pub use config::CohereConfig;
pub use ext::CohereRerankRequestExt;
