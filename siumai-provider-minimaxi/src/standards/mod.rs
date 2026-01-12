//! Protocol standards re-exported for MiniMaxi's internal use.
//!
//! MiniMaxi uses:
//! - Anthropic-style chat mapping
//! - OpenAI-style image/audio endpoints

#[cfg(feature = "minimaxi")]
pub use siumai_protocol_anthropic::standards::anthropic;

#[cfg(feature = "minimaxi")]
pub use siumai_protocol_openai::standards::openai;
