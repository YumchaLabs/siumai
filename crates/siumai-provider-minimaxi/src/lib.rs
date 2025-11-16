//! Siumai MiniMaxi Provider (extracted)
//!
//! This crate will gradually host the MiniMaxi provider implementation.
//! For now it exposes a small set of stateless helpers that can be
//! consumed by the aggregator crate via feature gates.

pub const VERSION: &str = "0.0.1-scaffolding";

#[derive(Debug, Clone, Default)]
pub struct MiniMaxiProviderMarker;

/// Provider-level constants (endpoints, paths).
pub mod constants;
/// Error mapping helpers.
pub mod error;
/// HTTP helpers shared between Anthropic-compatible and OpenAI-compatible APIs.
pub mod headers;
/// Core-level provider spec implementation.
pub mod spec;

pub use constants::{ANTHROPIC_BASE_URL, OPENAI_BASE_URL};
pub use error::map_minimaxi_error;
pub use headers::{build_anthropic_headers, build_openai_auth_headers};
pub use spec::MinimaxiCoreSpec;
