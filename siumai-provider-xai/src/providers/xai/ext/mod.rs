//! xAI provider extension APIs (non-unified surface)
//!
//! These APIs are intentionally *not* part of the Vercel-aligned unified model families.
//! Use them when you need xAI-specific escape hatches.

pub mod request_options;

pub use request_options::XaiChatRequestExt;
