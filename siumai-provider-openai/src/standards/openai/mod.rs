//! OpenAI API Standard
//!
//! This module implements the OpenAI API format, which has become a de facto standard
//! in the LLM industry. Many providers implement OpenAI-compatible APIs.
//!
//! ## Supported Providers
//!
//! - OpenAI (native)
//! - DeepSeek
//! - SiliconFlow
//! - Together
//! - OpenRouter
//! - Groq
//! - xAI
//! - Many others
//!
//! ## Capabilities
//!
//! - Chat Completions API
//! - Embeddings API
//! - Image Generation API
//! - Audio API
//! - Moderation API
//! - Rerank API (extension)

pub mod audio;
pub mod chat;
pub mod compat;
pub mod embedding;
pub mod errors;
pub mod files;
pub mod headers;
pub mod image;
pub mod rerank;
pub mod responses_sse;
pub mod transformers;
pub mod types;
pub mod utils;

#[cfg(test)]
mod chat_adapter_sse_tests;

// Re-export main types
pub use audio::{
    OpenAiAudioDefaults, OpenAiAudioTransformer, OpenAiAudioTransformerWithProviderId,
};
pub use chat::{OpenAiChatAdapter, OpenAiChatStandard};
pub use embedding::{OpenAiEmbeddingAdapter, OpenAiEmbeddingStandard};
pub use files::{OpenAiFilesTransformer, OpenAiFilesTransformerWithProviderId};
pub use image::{OpenAiImageAdapter, OpenAiImageStandard};
pub use rerank::{OpenAiRerankAdapter, OpenAiRerankStandard};
