//! siumai-protocol-openai
//!
//! OpenAI(-like) protocol standard mapping for siumai.
//!
//! This crate is currently a thin compatibility layer over the legacy crate name
//! `siumai-provider-openai-compatible`. Downstream code should prefer importing
//! from this crate going forward.
#![deny(unsafe_code)]

pub use siumai_provider_openai_compatible::*;
