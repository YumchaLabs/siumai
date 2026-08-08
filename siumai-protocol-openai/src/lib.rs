//! siumai-protocol-openai
//!
//! OpenAI Chat Completions, Responses, and Realtime wire codecs.
#![deny(unsafe_code)]

/// Canonical-core Chat Completions codec used by configured providers.
pub mod chat_completions;

/// OpenAI text embedding request and response codecs.
pub mod embedding;

/// OpenAI non-streaming image-generation codecs.
pub mod image;

/// Shared OpenAI-family error-envelope decoding and classification.
pub mod openai_error;

/// Provider-native resource JSON contracts.
pub mod resources;

/// OpenAI buffered text-to-speech codecs.
pub mod speech;

/// OpenAI final-result audio transcription codecs.
pub mod transcription;

/// Native Responses codec and lifecycle decoder.
#[cfg(feature = "openai-responses")]
pub mod responses;

/// Experimental Realtime and Realtime Translation wire codecs.
#[cfg(feature = "openai-realtime")]
pub mod realtime;
