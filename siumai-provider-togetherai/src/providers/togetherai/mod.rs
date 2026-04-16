//! `TogetherAI` provider module.
//!
//! This module exposes a provider-owned config/client/builder surface for the
//! TogetherAI native surfaces that currently back reranking plus provider-owned
//! request option helpers used by the unified registry facade.

pub mod builder;
pub mod client;
pub mod config;
pub mod ext;
pub mod models;

pub use builder::TogetherAiBuilder;
pub use client::TogetherAiClient;
pub use config::TogetherAiConfig;
pub use ext::{TogetherAiImageRequestExt, TogetherAiRerankRequestExt};
