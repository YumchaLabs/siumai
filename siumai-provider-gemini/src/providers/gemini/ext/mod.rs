//! Gemini provider extension APIs (non-unified surface)
//!
//! These APIs are intentionally *not* part of the Vercel-aligned unified model families.
//! Use them when you need Gemini-specific endpoints/resources beyond the unified surface.

pub mod code_execution;
pub mod embedding_options;
pub mod file_search_stores;
pub mod hosted_tools;
pub mod image_options;
pub mod request_options;
pub mod tools;
pub mod video_options;

pub use embedding_options::GoogleEmbeddingRequestExt;
pub use image_options::{GeminiImageRequestExt, GoogleImageRequestExt};
pub use request_options::{GeminiChatRequestExt, GoogleChatRequestExt};
pub use video_options::GoogleVideoRequestExt;
