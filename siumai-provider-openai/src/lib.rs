//! siumai-provider-openai
//!
//! Rust-first configured OpenAI provider.
#![deny(unsafe_code)]

/// Rust-first configured OpenAI runtime and explicit API-mode models.
pub mod configured;

pub use configured::*;

/// Typed prompt-cache controls shared by Chat Completions and Responses.
pub mod prompt_cache {
    pub use crate::configured::{
        OpenAiAnnotationError, OpenAiContentOptions, OpenAiPromptCacheMarker,
        OpenAiPromptCacheMode, OpenAiPromptCacheOptions, OpenAiPromptCacheRetention,
        OpenAiPromptCacheTtl,
    };
}
