//! siumai-provider-openai-compatible
//!
//! Legacy crate name for the OpenAI(-like) protocol family.
//!
//! Downstream code should prefer `siumai-protocol-openai`. This crate remains as a compatibility
//! alias and re-exports the full public surface from `siumai-protocol-openai`.
#![deny(unsafe_code)]

pub use siumai_protocol_openai::*;
