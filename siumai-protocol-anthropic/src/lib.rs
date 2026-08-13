//! siumai-protocol-anthropic
//!
//! Anthropic Messages protocol mapping for siumai.
//!
//! This crate owns the vendor-agnostic protocol layer for Anthropic Messages:
//! - Messages API request/response mapping
//! - Streaming event conversion
//! - Prompt caching + thinking helpers (protocol-level)
//!
//! Provider crates should depend on this crate and keep vendor-specific quirks behind
//! provider-owned presets/wrappers.
#![deny(unsafe_code)]

/// Canonical-core codec for the Anthropic Messages wire protocol.
pub mod messages;
