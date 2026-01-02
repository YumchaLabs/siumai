//! Gemini provider extension APIs (non-unified surface)
//!
//! These APIs are intentionally *not* part of the Vercel-aligned unified model families.
//! Use them when you need Gemini-specific endpoints/resources beyond the unified surface.

pub mod code_execution;
pub mod file_search_stores;
pub mod hosted_tools;
pub mod request_options;
pub mod tools;

pub use request_options::GeminiChatRequestExt;
