//! Explicit compatibility surfaces.
//!
//! New model-family code should use the family traits/modules directly. The modules here keep
//! historical broad or dynamic-dispatch contracts available while making their migration role
//! visible in import paths.

pub mod client;

/// Legacy chat content payloads.
///
/// Prefer prompt/model-message parts for request input and generated-output parts for response
/// output. These re-exports exist for code that intentionally needs legacy `ContentPart` /
/// `MessageContent` serde-compatible carriers.
pub mod content {
    pub use crate::types::compat::content::*;
}
