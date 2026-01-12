//! OpenAI-compatible protocol compatibility layer.
//!
//! This module intentionally contains provider-agnostic building blocks that are
//! reused by multiple OpenAI-like providers.
#![deny(unsafe_code)]

pub mod adapter;
pub mod openai_config;
pub mod provider_registry;
pub mod streaming;
pub mod transformers;
pub mod types;

#[cfg(test)]
mod streaming_tests;

#[cfg(test)]
mod transformers_tests;
