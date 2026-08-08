//! Feature-gated, curated provider namespaces.
//!
//! The facade exports configured providers, concrete stable-family models,
//! typed provider options, and explicitly named resources. Protocol codecs and
//! provider implementation internals remain available from their owning crates.

#[cfg(feature = "alibaba")]
pub mod alibaba;
#[cfg(feature = "anthropic")]
pub mod anthropic;
#[cfg(feature = "cohere")]
pub mod cohere;
#[cfg(feature = "deepgram")]
pub mod deepgram;
#[cfg(feature = "deepseek")]
pub mod deepseek;
#[cfg(feature = "elevenlabs")]
pub mod elevenlabs;
#[cfg(feature = "google")]
pub mod google;
#[cfg(feature = "google-vertex-anthropic")]
pub mod google_vertex_anthropic;
#[cfg(feature = "groq")]
pub mod groq;
#[cfg(feature = "minimax")]
pub mod minimax;
#[cfg(feature = "moonshotai")]
pub mod moonshotai;
#[cfg(feature = "openai")]
pub mod openai;
#[cfg(feature = "openai-compatible")]
pub mod openai_compatible;
#[cfg(feature = "volcengine")]
pub mod volcengine;
#[cfg(feature = "xai")]
pub mod xai;
