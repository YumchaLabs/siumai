//! Transformers Layer
//!
//! Stable, provider-agnostic traits for transforming requests, responses,
//! and streaming chunks across providers. Forms the core of the unified
//! execution pipeline together with the Executors layer.

pub mod audio;
pub mod files;
pub mod request;
pub mod response;
pub mod stream;
