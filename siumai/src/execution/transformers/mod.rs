//! Transformers Layer
//!
//! Stable, provider-agnostic traits for transforming requests, responses,
//! and streaming chunks across providers. Forms the core of the unified
//! execution pipeline together with the Executors layer.

pub mod audio;
pub mod files;
pub mod hook_builder;
mod json_path; // internal JSON path utils used by request transformer
pub mod request;
pub mod rerank_request;
pub mod rerank_response;
pub mod response;
pub mod stream;
