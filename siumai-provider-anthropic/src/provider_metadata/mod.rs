//! Typed response metadata re-exported from the Anthropic protocol crate.
//!
//! The protocol crate owns Anthropic Messages response metadata shapes. This provider crate keeps
//! this module as the stable provider-facing import path.

pub mod anthropic {
    pub use siumai_protocol_anthropic::provider_metadata::anthropic::*;
}
