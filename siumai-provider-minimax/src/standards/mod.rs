//! Protocol standards re-exported for MiniMax's internal use.
//!
//! MiniMax uses:
//! - Anthropic-style chat mapping
//! - OpenAI-style image/audio endpoints

#[cfg(feature = "minimax")]
pub use siumai_protocol_anthropic::standards::anthropic;

#[cfg(feature = "minimax")]
pub use siumai_protocol_openai::standards::openai;
