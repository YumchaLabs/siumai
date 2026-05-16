//! siumai-provider-openai-compatible
//!
//! Legacy crate name for the OpenAI(-like) protocol family.
//!
//! Downstream code should prefer `siumai-protocol-openai`. This crate remains as a compatibility
//! alias and re-exports the full public surface from `siumai-protocol-openai`.
//!
//! In the workspace split (beta.5+), this crate also hosts the OpenAI-compatible provider
//! implementation (vendor adapters + routing), aligned with the Vercel AI SDK package layout.
#![deny(unsafe_code)]

mod macros;

pub use siumai_protocol_openai::*;

// Internal core aliases used by the compatibility provider implementation.
// They are intentionally not part of this legacy crate's public compatibility surface.
#[allow(unused_imports)]
pub(crate) use siumai_core::{
    LlmError, auth, client, core, defaults, encoding, error, execution, observability, retry,
    retry_api, streaming, tools, traits, types, utils,
};

pub(crate) mod builder {
    #[allow(unused_imports)]
    pub(crate) use siumai_core::builder::*;
}

/// Provider-owned typed option structs for OpenAI-compatible vendors.
pub mod provider_options;

/// OpenAI-compatible providers (vendor presets, adapter registry, client).
pub mod providers;
