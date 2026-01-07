//! siumai-provider-anthropic-compatible
//!
//! Legacy crate name for the Anthropic Messages protocol mapping.
//!
//! Downstream code should prefer `siumai-protocol-anthropic`. This crate remains as a compatibility
//! alias and re-exports the full public surface from `siumai-protocol-anthropic`.
#![deny(unsafe_code)]

pub use siumai_protocol_anthropic::*;
