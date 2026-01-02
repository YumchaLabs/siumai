//! OpenAI family provider implementations.

#[cfg(feature = "openai")]
pub mod openai;

// OpenAI-compatible vendors reuse the OpenAI-like protocol standard, but do not require
// the native OpenAI provider implementation.
#[cfg(feature = "openai-standard")]
pub mod openai_compatible;
