//! Typed response metadata re-exported from the OpenAI protocol crate.
//!
//! The protocol crate owns OpenAI/OpenAI-compatible response metadata shapes. This provider crate
//! keeps this module as the stable provider-facing import path.

pub mod openai {
    pub use siumai_protocol_openai::provider_metadata::openai::*;
}
