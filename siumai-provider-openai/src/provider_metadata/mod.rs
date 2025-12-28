//! Provider-specific metadata (provider-owned).
//!
//! The unified response surface stores metadata as a nested map:
//! `{ "provider_id": { "key": value, ... }, ... }`.
//!
//! Provider crates may define typed views over those maps for ergonomics.

pub mod openai;

