//! OpenAI provider-hosted tool constructors.
//!
//! The protocol crate owns the OpenAI tool-id shapes; this provider crate re-exports them from the
//! provider package surface.

pub mod openai {
    pub use siumai_protocol_openai::hosted_tools::openai::*;
}
