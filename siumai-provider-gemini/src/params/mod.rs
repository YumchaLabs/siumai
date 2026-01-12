//! Provider-owned parameter types (legacy).
//!
//! These types are kept for backward compatibility with older siumai APIs that exposed
//! provider-specific parameter structs. New code should prefer:
//! - `CommonParams` for cross-provider parameters
//! - typed provider options (`siumai::provider_ext::gemini::*`)

pub mod gemini;

pub use gemini::*;
