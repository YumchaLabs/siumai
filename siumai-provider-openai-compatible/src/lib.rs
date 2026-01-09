//! siumai-provider-openai-compatible
//!
//! Legacy crate name for the OpenAI(-like) protocol family.
//!
//! Downstream code should prefer `siumai-protocol-openai`. This crate remains as a compatibility
//! alias and re-exports the full public surface from `siumai-protocol-openai`.
//!
//! In the workspace split (beta.5+), this crate also hosts the OpenAI-compatible provider
//! implementation (vendor adapters + routing), aligned with the Vercel AI SDK package layout.
#![deny(unsafe_code)]

pub use siumai_protocol_openai::*;

/// OpenAI-compatible providers (vendor presets, adapter registry, client).
pub mod providers;
