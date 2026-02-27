//! siumai-core
//!
//! Provider-agnostic runtime, types, and shared execution primitives.
#![deny(unsafe_code)]

pub mod auth;
pub mod builder;
pub mod client;
pub mod core;
pub mod custom_provider;
pub mod defaults;
pub mod encoding;
pub mod error;
pub mod execution;
pub mod hosted_tools;
pub mod observability;
pub mod params;
pub mod retry;
pub mod retry_api;
pub mod standards;
pub mod streaming;
pub mod tools;
pub mod traits;
pub mod types;
pub mod utils;

pub use error::LlmError;
