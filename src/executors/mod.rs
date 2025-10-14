//! Executors Layer
//!
//! Stable HTTP orchestration that wires transformers with provider endpoints
//! to perform capability-specific operations (chat, embedding, image, audio, files).

pub mod audio;
pub mod chat;
pub mod embedding;
pub mod files;
pub mod image;
