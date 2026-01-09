//! OpenAI-compatible providers (vendor presets, adapter registry, routing).
//!
//! Note: The implementation lives in `siumai-provider-openai-compatible`. This module only
//! re-exports it to keep historical module paths compiling during the fearless refactor.

pub use siumai_provider_openai_compatible::providers::openai_compatible::*;
