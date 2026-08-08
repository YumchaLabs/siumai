//! siumai-protocol-openai
//!
//! OpenAI Chat Completions, Responses, and Realtime wire codecs.
#![deny(unsafe_code)]

/// Canonical-core Chat Completions codec used by configured providers.
pub mod chat_completions;

/// Shared OpenAI-family error-envelope decoding and classification.
pub mod openai_error;

/// Native Responses codec and lifecycle decoder.
#[cfg(feature = "openai-responses")]
pub mod responses;

/// Experimental Realtime and Realtime Translation wire codecs.
#[cfg(feature = "openai-realtime")]
pub mod realtime;
