//! Provider-owned Google Vertex auth helpers.
//!
//! The generic token provider contract remains in `siumai-core`; Vertex-specific URL helpers live
//! here so core does not own provider URL construction.

pub use siumai_core::auth::{StaticTokenProvider, TokenProvider};

#[cfg(feature = "gcp")]
pub mod adc;
#[cfg(feature = "gcp")]
pub mod service_account;

pub mod vertex;
