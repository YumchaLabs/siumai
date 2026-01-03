//! Protocol standards re-exported for MiniMaxi's internal use.
//!
//! MiniMaxi uses:
//! - Anthropic-style chat mapping
//! - OpenAI-style image/audio endpoints

#[cfg(feature = "minimaxi")]
pub use siumai_provider_anthropic_compatible::standards::anthropic;

#[cfg(feature = "minimaxi")]
pub use siumai_provider_openai_compatible::standards::openai;
