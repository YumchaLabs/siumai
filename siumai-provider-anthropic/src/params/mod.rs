//! Provider-owned parameter types (legacy).
//!
//! These types are kept for backward compatibility with older siumai APIs that exposed
//! provider-specific parameter structs. New code should prefer:
//! - `CommonParams` for cross-provider parameters
//! - `provider_options_map` / typed provider options (`siumai::provider_ext::anthropic::*`)

pub mod anthropic;

pub use anthropic::*;
