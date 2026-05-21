//! Compatibility re-export for provider builder helpers.
//!
//! The implementation moved to `siumai-provider-utils` in FCAB-090. Keep this module as a
//! migration alias while downstream callers move to the provider-utils seam.

pub use siumai_provider_utils::builder_helpers::*;
