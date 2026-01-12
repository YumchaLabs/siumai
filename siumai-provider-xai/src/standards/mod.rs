#![deny(unsafe_code)]

#[cfg(feature = "xai")]
pub use siumai_protocol_openai::standards::openai;

#[cfg(feature = "xai")]
pub mod xai;
