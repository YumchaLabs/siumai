//! siumai-core
//!
//! Provider-agnostic runtime, types, and shared execution primitives.
#![deny(unsafe_code)]

pub mod auth;
pub mod bridge;
pub mod builder;
pub mod client;
pub mod core;
pub mod custom_provider;
pub mod defaults;
pub mod embedding;
pub mod encoding;
pub mod error;
pub mod execution;
pub mod hosted_tools;
pub mod image;
pub mod observability;
pub mod params;
pub mod rerank;
pub mod retry;
pub mod retry_api;
pub mod speech;
pub mod standards;
pub mod streaming;
pub mod structured_output;
pub mod text;
pub mod tooling;
pub mod tools;
pub mod traits;
pub mod transcription;
pub mod types;
pub mod utils;

pub use error::LlmError;
