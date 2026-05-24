//! Protocol mapping facade (stable imports for protocol standards).
//!
//! This module exists to decouple downstream code from internal crate names. Over time we may
//! rename protocol crates, but `siumai::protocol::*` should remain stable.

/// OpenAI-like protocol standard mapping (Chat/Embedding/Image/Rerank).
///
/// Backed by `siumai-protocol-openai` (preferred; wraps the legacy `*-compatible` crate name).
#[cfg(any(feature = "openai", feature = "protocol-openai"))]
pub mod openai {
    pub use siumai_protocol_openai::standards::openai::*;
}

/// Anthropic Messages protocol standard mapping (Chat + streaming).
///
/// Backed by `siumai-protocol-anthropic` (preferred; wraps the legacy `*-compatible` crate name).
#[cfg(any(feature = "anthropic", feature = "protocol-anthropic"))]
pub mod anthropic {
    pub use siumai_protocol_anthropic::standards::anthropic::*;
}

/// Google Gemini protocol standard mapping (GenerateContent + streaming).
///
/// Backed by `siumai-protocol-gemini`.
#[cfg(any(feature = "google", feature = "protocol-gemini"))]
pub mod gemini {
    pub use siumai_protocol_gemini::standards::gemini::*;
}
