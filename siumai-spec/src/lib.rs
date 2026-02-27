//! siumai-spec
//!
//! Provider-agnostic specs and data types for siumai.
//!
//! This crate intentionally contains only *spec-level* types (requests, responses,
//! messages, tool schemas, and lightweight configuration structs). Runtime
//! execution, HTTP, retries, streaming executors, and provider implementations
//! live in other crates (e.g. `siumai-core`, `siumai-provider-*`).
#![deny(unsafe_code)]

pub mod error;
pub mod observability;
pub mod tools;
pub mod types;
