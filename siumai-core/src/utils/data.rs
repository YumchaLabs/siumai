//! Compatibility re-export for `data` provider utility helpers.
//!
//! The implementation moved to `siumai-provider-utils` in FCAB-100. Keep this module as a
//! migration alias while callers move to the provider-utils seam.

pub use siumai_provider_utils::data::*;
