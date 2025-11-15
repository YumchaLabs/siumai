//! Siumai Gemini Standard
//!
//! Hosts Gemini standard request/response/streaming transformers and adapters.

/// Version of the Gemini standard crate.
pub const VERSION: &str = "0.0.1";

/// Gemini standard modules.
pub mod gemini;

// Re-export main chat/embedding/image standards and adapters
pub use gemini::chat::{GeminiChatAdapter, GeminiChatStandard, GeminiDefaultChatAdapter};
pub use gemini::embedding::GeminiEmbeddingStandard;
pub use gemini::image::GeminiImageStandard;
