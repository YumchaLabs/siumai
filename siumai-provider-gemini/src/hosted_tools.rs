//! Google/Gemini provider-hosted tool constructors.
//!
//! The protocol crate owns the Google tool-id shapes; this provider crate re-exports them from the
//! provider package surface.

pub mod google {
    pub use siumai_protocol_gemini::hosted_tools::google::*;
}
