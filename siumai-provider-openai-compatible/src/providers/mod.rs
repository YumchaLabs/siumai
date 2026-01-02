//! Provider implementations owned by this crate.
//!
//! This crate intentionally does **not** ship the native OpenAI provider.
#[cfg(feature = "openai-standard")]
pub mod openai_compatible;
