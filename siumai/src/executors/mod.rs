//! Executors Layer
//!
//! Stable HTTP orchestration that wires transformers with provider endpoints
//! to perform capability-specific operations (chat, embedding, image, audio, files).

pub mod audio;
pub mod chat;
pub mod embedding;
pub mod files;
pub mod image;

// Shared type aliases to simplify complex executor hook types
use std::sync::Arc;

/// Hook to mutate JSON request bodies before sending.
/// Useful for provider-specific tweaks or user-provided interceptors.
pub type BeforeSendHook = Arc<
    dyn Fn(&serde_json::Value) -> Result<serde_json::Value, crate::error::LlmError> + Send + Sync,
>;
