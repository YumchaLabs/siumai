//! Standards aggregation for `siumai-providers`.
//!
//! Historically, protocol standards lived in `siumai-core`. During the fearless refactor,
//! OpenAI-like standards are extracted into `siumai-provider-openai` to reduce coupling
//! in the provider-agnostic core.
#![deny(unsafe_code)]

#[cfg(any(
    feature = "openai",
    feature = "groq",
    feature = "xai"
))]
pub use siumai_provider_openai::standards::openai;

#[cfg(feature = "anthropic")]
pub mod anthropic;

#[cfg(feature = "google")]
pub mod gemini;

#[cfg(feature = "ollama")]
pub mod ollama;
