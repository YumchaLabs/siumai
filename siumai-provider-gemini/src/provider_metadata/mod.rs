//! Typed response metadata re-exported from the Gemini protocol crate.
//!
//! The protocol crate owns Gemini response metadata shapes because they are part of the
//! GenerateContent protocol mapping. The provider crate keeps this module as the stable
//! provider-facing import path.

#[cfg(feature = "google")]
pub mod gemini;
