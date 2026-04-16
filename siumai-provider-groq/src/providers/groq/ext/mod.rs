//! Groq provider extension APIs (non-unified surface)
//!
//! These APIs are intentionally *not* part of the Vercel-aligned unified model families.
//! Use them when you need Groq-specific escape hatches.

pub mod audio_options;
pub mod request_options;

pub use audio_options::{GroqSttOptions, GroqTtsOptions};
pub use request_options::GroqChatRequestExt;
