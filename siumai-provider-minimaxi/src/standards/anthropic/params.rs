//! Anthropic standard parameters (minimal).
//!
//! MiniMaxi currently uses Anthropic protocol mapping mainly for request/response
//! transformation and streaming event conversion. The streaming converter keeps a
//! config slot for forward-compatibility; at the moment, no knobs are read.

/// Placeholder config for Anthropic streaming conversion.
#[derive(Debug, Clone, Default)]
pub struct AnthropicParams;

