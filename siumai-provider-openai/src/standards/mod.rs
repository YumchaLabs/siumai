//! Protocol standards owned by this crate.
#![deny(unsafe_code)]

#[cfg(any(feature = "openai-standard", feature = "openai"))]
pub use siumai_provider_openai_compatible::standards::openai;
