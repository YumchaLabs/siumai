//! Provider-owned parameter types (legacy).
//!
//! These types are kept for backward compatibility with older siumai APIs that exposed
//! OpenAI-specific parameter structs. New code should prefer:
//! - `CommonParams` for cross-provider parameters
//! - typed provider options (`siumai::provider_ext::openai::*`)

pub mod openai;

pub use openai::*;
