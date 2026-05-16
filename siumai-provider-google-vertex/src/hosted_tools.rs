//! Google Vertex provider-hosted tool constructors.
//!
//! Vertex reuses Google/Gemini hosted tool-id shapes and, for Anthropic-on-Vertex, the Anthropic
//! hosted tool-id shapes.

pub mod google {
    pub use siumai_protocol_gemini::hosted_tools::google::*;
}

#[cfg(feature = "google-vertex")]
pub mod anthropic {
    pub use siumai_protocol_anthropic::hosted_tools::anthropic::*;
}
