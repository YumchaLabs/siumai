//! Siumai xAI Provider (extracted)
//!
//! This crate will gradually host the xAI provider implementation.
//! For now it exposes a small set of stateless helpers and a
//! core-level ProviderSpec that can be consumed by the aggregator
//! crate via feature gates.

pub const VERSION: &str = "0.0.1-scaffolding";

#[derive(Debug, Clone, Default)]
pub struct XaiProviderMarker;

/// HTTP helpers for xAI API.
pub mod headers;
/// Core-level provider spec 实现.
pub mod spec;

pub use headers::build_xai_json_headers;
pub use spec::XaiCoreSpec;
