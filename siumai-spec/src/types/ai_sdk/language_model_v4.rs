//! AI SDK V4 prompt and generated content projection types.
//!
//! The public `ai_sdk` surface re-exports these types, while the physical ownership stays split
//! between shared V4 data, prompt-side projections, and response-side generated content.

mod content;
mod prompt;
mod shared;

pub use content::*;
pub use prompt::*;
pub use shared::*;

pub(super) use shared::{
    deserialize_optional_language_model_v4_provider_metadata,
    serialize_optional_language_model_v4_provider_metadata,
};
