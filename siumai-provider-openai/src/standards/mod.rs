//! Protocol standards re-exported by this crate for compatibility.
#![deny(unsafe_code)]

#[cfg(any(feature = "openai-standard", feature = "openai"))]
pub use siumai_provider_openai_compatible::standards::openai;
