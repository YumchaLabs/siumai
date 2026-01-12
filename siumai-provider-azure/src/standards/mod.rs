//! Protocol standards re-exported by this crate for compatibility.
#![deny(unsafe_code)]

#[cfg(any(feature = "azure-standard", feature = "azure"))]
pub use siumai_protocol_openai::standards::openai;
