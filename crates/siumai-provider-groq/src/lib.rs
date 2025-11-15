//! Siumai Groq Provider (extracted)
//!
//! This crate will gradually host the Groq provider implementation.
//! For now it exposes a small set of stateless helpers and a
//! core-level ProviderSpec that can be consumed by the aggregator
//! crate via feature gates.

pub const VERSION: &str = "0.0.1-scaffolding";

#[derive(Debug, Clone, Default)]
pub struct GroqProviderMarker;

/// HTTP helpers for Groq API.
pub mod headers;
/// Core-level provider spec 实现.
pub mod spec;

pub use headers::build_groq_json_headers;
pub use spec::GroqCoreSpec;
