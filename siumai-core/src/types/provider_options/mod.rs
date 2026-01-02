//! Provider option helpers.
//!
//! In the refactored architecture, Siumai uses the open `ProviderOptionsMap` (provider-id keyed
//! JSON map) as the only transport for provider-specific options.
//!
//! Typed option structs are owned by provider crates and are typically attached via:
//! - `ChatRequest::with_provider_option(provider_id, serde_json::to_value(opts)?)`
//! - Provider-owned extension traits under `siumai::provider_ext::<provider>::*` (preferred)

// Provider helper modules
pub mod custom;

// Re-export user extensibility trait.
pub use custom::CustomProviderOptions;
