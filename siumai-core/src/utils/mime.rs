//! Compatibility re-export for MIME helpers.
//!
//! The implementation moved to `siumai-provider-utils` in FCAB-090. Keep this module as a
//! migration alias while provider/protocol crates move to the provider-utils seam.

pub use siumai_provider_utils::mime::*;
