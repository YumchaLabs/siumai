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
pub mod convert;
pub mod embedding;
pub mod headers;
pub mod image;
mod sources;
pub mod streaming;
pub mod transformers;
pub mod types;

pub(super) fn normalize_gemini_model_id(model: &str) -> String {
    let trimmed = model.trim().trim_matches('/');
    if trimmed.is_empty() {
        return String::new();
    }

    // Accept a variety of resource-style names and normalize to the bare model id:
    // - "gemini-2.0-flash"
    // - "models/gemini-2.0-flash"
    // - "publishers/google/models/gemini-2.0-flash"
    // - "projects/.../publishers/google/models/gemini-2.0-flash"
    if let Some(pos) = trimmed.rfind("/models/") {
        return trimmed[(pos + "/models/".len())..].to_string();
    }
    if let Some(rest) = trimmed.strip_prefix("models/") {
        return rest.to_string();
    }

    trimmed.to_string()
}

// Re-export main types
pub use chat::{GeminiChatAdapter, GeminiChatSpec, GeminiChatStandard};
pub use embedding::{GeminiEmbeddingAdapter, GeminiEmbeddingSpec, GeminiEmbeddingStandard};
pub use image::{GeminiImageAdapter, GeminiImageSpec, GeminiImageStandard};
