//! OpenAI(-compatible) protocol helpers.
//!
//! This module is protocol-level (wire format + compatibility adapters) and is
//! shared across provider implementations that speak OpenAI-like APIs.
#![deny(unsafe_code)]

pub mod compat;
pub mod types;
pub mod utils;
