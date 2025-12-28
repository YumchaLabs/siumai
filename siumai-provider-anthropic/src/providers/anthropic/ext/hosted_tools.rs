//! Anthropic provider-hosted tools (non-unified surface).
//!
//! This module re-exports the provider-defined tool factories under the Anthropic
//! provider namespace, similar to Vercel AI SDK provider packages.
//!
//! These tools are executed by Anthropic servers and should be used via
//! `Tool::ProviderDefined` in chat requests.

pub use crate::hosted_tools::anthropic::*;
