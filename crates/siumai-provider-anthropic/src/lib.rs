//! Siumai Anthropic Provider (extracted)
//!
//! This crate will gradually host the Anthropic provider implementation.
//! For now it exposes a small set of stateless helpers that can be
//! consumed by the aggregator crate via feature gates.

pub const VERSION: &str = "0.0.1-scaffolding";

#[derive(Debug, Clone, Default)]
pub struct AnthropicProviderMarker;

/// Provider-level constants (endpoints, paths).
pub mod constants;
/// Error mapping helpers.
pub mod error;
/// HTTP helpers for Anthropic API.
pub mod headers;
/// Core-level provider spec implementation.
pub mod spec;

pub use constants::ANTHROPIC_V1_ENDPOINT;
pub use error::map_anthropic_error;
pub use headers::build_anthropic_json_headers;
pub use spec::AnthropicCoreSpec;
