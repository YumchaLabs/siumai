//! Executors layer (Phase 0 scaffolding)
//!
//! Executors orchestrate transformers and HTTP to perform capability-specific
//! operations (chat/embedding/image/audio). Introduced behind `new-exec`.

pub mod audio;
pub mod chat;
pub mod embedding;
pub mod files;
pub mod image;
