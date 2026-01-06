//! siumai-protocol-anthropic
//!
//! Anthropic Messages protocol standard mapping for siumai.
//!
//! This crate is currently a thin compatibility layer over the legacy crate name
//! `siumai-provider-anthropic-compatible`. Downstream code should prefer importing
//! from this crate going forward.
#![deny(unsafe_code)]

pub use siumai_provider_anthropic_compatible::*;
