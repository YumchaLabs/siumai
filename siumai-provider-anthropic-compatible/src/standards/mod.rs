//! Protocol standards owned by this crate.
#![deny(unsafe_code)]

#[cfg(any(feature = "anthropic", feature = "anthropic-standard"))]
pub mod anthropic;

