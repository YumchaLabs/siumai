//! siumai-provider-openai
//!
//! Rust-first configured OpenAI provider.
#![deny(unsafe_code)]

/// Rust-first configured OpenAI runtime and explicit API-mode models.
pub mod configured;

pub use configured::*;
