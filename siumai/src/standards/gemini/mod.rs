//! Google Gemini API Standard
//!
//! This module implements the Google Gemini API format.
//! Note: This is provider-specific and not widely adopted by other providers.
//!
//! ## Supported Providers
//!
//! - Google Gemini (native)
//!
//! ## Capabilities
//!
//! - Generate Content API (Chat)
//! - Embeddings API
//! - Safety Settings (Gemini-specific)
//! - Grounding (Gemini-specific)

pub mod chat;

// Re-export main types
pub use chat::GeminiChatStandard;
