//! siumai-protocol-openai
//!
//! OpenAI(-like) protocol family mapping for siumai.
//!
//! This crate owns the vendor-agnostic protocol layer for the OpenAI family:
//! - OpenAI-like protocol mapping (Chat/Embedding/Image/Rerank)
//! - OpenAI-compatible protocol building blocks (types + transformers)
//! - OpenAI Responses API mapping (feature-gated)
//!
//! Provider crates (e.g. `siumai-provider-openai`, `siumai-provider-xai`,
//! `siumai-provider-openai-compatible`) should depend on this crate, and keep provider-specific
//! quirks behind provider-owned presets/wrappers.
#![deny(unsafe_code)]

/// Canonical-core Chat Completions codec used by configured providers.
pub mod chat_completions;

// Keep provider-agnostic core modules available only to this crate's implementation.
// Protocol crates must not publicly mirror `siumai-core`; downstream code should import
// shared core types from `siumai-core` or the top-level `siumai` facade.
#[allow(unused_imports)]
pub(crate) use siumai_provider_utils as provider_utils;

#[allow(unused_imports)]
pub(crate) use siumai_core::{
    LlmError, auth, client, core, defaults, encoding, error, execution, observability, retry,
    retry_api, streaming, traits, types, utils,
};

pub mod hosted_tools;

/// Provider-defined tool catalog owned by the OpenAI protocol family.
pub mod tool_catalog;

/// Protocol-owned typed metadata views.
pub mod provider_metadata;

pub mod standards;
