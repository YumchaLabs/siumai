//! Provider wire-format encoders.
//!
//! This module is intended for gateway/proxy use-cases where a unified response
//! (`ChatResponse`) needs to be re-serialized into a provider-native wire format.
//!
//! English-only comments in code as requested.

pub mod response_json;

pub use response_json::*;
