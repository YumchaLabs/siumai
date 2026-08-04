//! Feature-gated, curated provider namespaces.
//!
//! The facade exports configured providers, concrete stable-family models,
//! typed provider options, and explicitly named resources. Protocol codecs and
//! provider implementation internals remain available from their owning crates.

#[cfg(feature = "cohere")]
pub mod cohere;
#[cfg(feature = "deepgram")]
pub mod deepgram;
#[cfg(feature = "elevenlabs")]
pub mod elevenlabs;
#[cfg(feature = "google")]
pub mod google;
#[cfg(feature = "openai")]
pub mod openai;
#[cfg(feature = "openai-compatible")]
pub mod openai_compatible;

#[doc(hidden)]
pub mod legacy;
