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
//! - Audio API (OpenAI-style)
//! - Moderation API
//! - Rerank API (extension)
pub mod audio;
pub mod chat;
pub mod compat;
pub mod embedding;
pub mod errors;
pub mod headers;
pub mod image;
pub mod rerank;
pub mod transformers;
pub mod types;
pub mod utils;

// Re-export main types
pub use audio::{
    OpenAiAudioDefaults, OpenAiAudioTransformer, OpenAiAudioTransformerWithProviderId,
};
pub use chat::{OpenAiChatAdapter, OpenAiChatStandard};
pub use embedding::{OpenAiEmbeddingAdapter, OpenAiEmbeddingStandard};
pub use image::{OpenAiImageAdapter, OpenAiImageStandard};
pub use rerank::{OpenAiRerankAdapter, OpenAiRerankStandard};
