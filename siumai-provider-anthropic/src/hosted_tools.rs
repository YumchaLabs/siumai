//! Anthropic provider-hosted tool constructors.
//!
//! The protocol crate owns the Anthropic tool-id shapes; this provider crate re-exports them from
//! the provider package surface.

pub mod anthropic {
    pub use siumai_protocol_anthropic::hosted_tools::anthropic::*;
}
