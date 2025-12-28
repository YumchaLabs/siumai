//! Provider-owned typed options.
//!
//! This module hosts typed provider options that were historically defined in `siumai-core`.
//! During the refactor, these types are intentionally owned by the provider crate to reduce
//! coupling in the provider-agnostic core.

pub mod anthropic;

