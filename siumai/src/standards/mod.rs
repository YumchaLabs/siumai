//! API Standards Layer
//!
//! This module contains implementations of different LLM API standards (formats).
//! Each standard represents a specific API format that can be implemented by multiple providers.
//!
//! ## Design Philosophy
//!
//! - **Standards are independent of providers**: OpenAI Chat API is a standard that can be
//!   implemented by OpenAI, DeepSeek, SiliconFlow, Together, OpenRouter, etc.
//! - **Providers can support multiple standards**: DeepSeek supports both OpenAI and Anthropic formats
//! - **Standards are composable**: Each standard provides transformers and executors that can be
//!   configured with provider-specific adapters
//!
//! ## Available Standards
//!
//! - `openai`: OpenAI API format (Chat, Embedding, Image, Audio, etc.)
//! - `anthropic`: Anthropic Messages API format
//! - `gemini`: Google Gemini API format (provider-specific, not widely adopted)
//!
//! ## Example
//!
//! ```rust,ignore
//! use siumai::standards::openai::chat::OpenAiChatStandard;
//!
//! // Create a standard OpenAI Chat implementation
//! let standard = OpenAiChatStandard::new();
//!
//! // Or with a provider-specific adapter
//! let standard = OpenAiChatStandard::with_adapter(
//!     Arc::new(DeepSeekOpenAiAdapter)
//! );
//! ```

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "anthropic")]
pub mod anthropic;

#[cfg(feature = "google")]
pub mod gemini;
