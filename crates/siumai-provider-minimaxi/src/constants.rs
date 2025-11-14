//! MiniMaxi provider constants.
//!
//! These mirror the defaults used in the aggregator crate so that
//! provider-specific configuration can live close to the provider
//! implementation.

/// Default base URL for MiniMaxi Anthropic-compatible chat API.
pub const ANTHROPIC_BASE_URL: &str = "https://api.minimaxi.com/anthropic";

/// Default base URL for MiniMaxi OpenAI-compatible APIs (audio/image/video/music).
pub const OPENAI_BASE_URL: &str = "https://api.minimaxi.com/v1";
