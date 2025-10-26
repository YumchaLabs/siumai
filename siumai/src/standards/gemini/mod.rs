//! Google Gemini API Standard
//!
//! This module implements the Google Gemini API format with a standard + adapter
//! approach similar to OpenAI and Anthropic standards.
//!
//! ## Supported Providers
//!
//! - Google Gemini (native)
//! - Potentially other providers that adopt Gemini format via adapters
//!
//! ## Capabilities
//!
//! - Generate Content API (Chat)
//! - Embeddings API
//! - Image generation (via generateContent)
//! - Safety Settings / Grounding (adapter hooks)

pub mod chat;
pub mod embedding;
pub mod image;

// Re-export main types
pub use chat::{GeminiChatAdapter, GeminiChatSpec, GeminiChatStandard};
pub use embedding::{GeminiEmbeddingAdapter, GeminiEmbeddingSpec, GeminiEmbeddingStandard};
pub use image::{GeminiImageAdapter, GeminiImageSpec, GeminiImageStandard};
