//! Protocol standards re-exported by this crate for compatibility.
#![deny(unsafe_code)]

#[cfg(any(feature = "anthropic", feature = "anthropic-standard"))]
pub use siumai_provider_anthropic_compatible::standards::anthropic;
