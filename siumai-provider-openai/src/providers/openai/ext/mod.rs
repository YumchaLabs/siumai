//! OpenAI provider extension APIs (non-unified surface)
//!
//! These APIs are intentionally *not* part of the Vercel-aligned unified model families.
//! Use them when you need OpenAI-specific endpoints/resources beyond the unified surface.

pub mod audio_options;
pub mod hosted_tools;
pub mod moderation;
pub mod request_options;
pub mod responses;
pub mod speech_streaming;
pub mod transcription_streaming;

pub use request_options::OpenAiChatRequestExt;
