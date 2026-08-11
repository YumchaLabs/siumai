//! siumai-provider-openai
//!
//! Rust-first configured OpenAI provider with complementary portable and
//! provider-owned surfaces.
//!
//! [`OpenAiProvider`] is configured once and creates lightweight model handles:
//!
//! - [`OpenAiProvider::language_model`] and [`OpenAiProvider::responses`] use
//!   Responses for the portable language contract;
//! - [`OpenAiProvider::chat_completions`] keeps the explicit Chat Completions
//!   protocol path;
//! - embedding, image, buffered speech, and final-result transcription use the
//!   corresponding portable model-family traits.
//!
//! Product-specific lifecycles remain owned by this crate. Use
//! [`OpenAiProvider::responses_resource`], [`OpenAiProvider::conversations`],
//! [`OpenAiProvider::files`], and [`OpenAiProvider::vector_stores`] directly.
//! Skills remain an experimental provider resource exposed through
//! [`experimental::skills::OpenAiSkillsProviderExt`]. Hosted/server tools are
//! provider-executed wire data and never become caller-owned portable tool
//! calls.
//!
//! The optional `openai-realtime` and `openai-responses-websocket` features are
//! independent provider-native session surfaces. Realtime is not a Responses
//! transport alias, and Responses WebSocket is not a provider-neutral session
//! family.
#![deny(unsafe_code)]

/// Rust-first configured OpenAI runtime and explicit API-mode models.
pub mod configured;

pub use configured::*;

/// Typed prompt-cache controls shared by Chat Completions and Responses.
pub mod prompt_cache {
    pub use crate::configured::{
        OpenAiContentOptions, OpenAiPromptCacheMode, OpenAiPromptCacheOptions,
        OpenAiPromptCacheRetention, OpenAiPromptCacheTtl,
    };
}
