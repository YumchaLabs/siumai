//! Provider-defined tool factories (Vercel-aligned).
//!
//! This facade keeps the historical `siumai::tools::*` import path while delegating canonical
//! provider catalogs to protocol/provider-owned crates.

use siumai_core::types::Tool;

#[cfg(any(feature = "openai", feature = "protocol-openai"))]
pub mod openai {
    pub use siumai_protocol_openai::tool_catalog::openai::*;
}

#[cfg(any(feature = "anthropic", feature = "protocol-anthropic"))]
pub mod anthropic {
    pub use siumai_protocol_anthropic::tool_catalog::anthropic::*;
}

#[cfg(any(
    feature = "google",
    feature = "google-vertex",
    feature = "protocol-gemini"
))]
pub mod google {
    pub use siumai_protocol_gemini::tool_catalog::google::*;
}

#[cfg(feature = "groq")]
pub mod groq {
    pub use siumai_provider_groq::tools::groq::*;
}

#[cfg(feature = "xai")]
pub mod xai {
    pub use siumai_provider_xai::tools::xai::*;
}

/// Create a provider-defined tool from a known provider-owned tool id.
///
/// This is a facade-only compatibility helper. The canonical catalogs live in the relevant
/// protocol/provider crates.
pub fn provider_defined_tool(id: &str) -> Option<Tool> {
    #[cfg(any(feature = "openai", feature = "protocol-openai"))]
    if let Some(tool) = openai::provider_defined_tool(id) {
        return Some(tool);
    }

    #[cfg(any(feature = "anthropic", feature = "protocol-anthropic"))]
    if let Some(tool) = anthropic::provider_defined_tool(id) {
        return Some(tool);
    }

    #[cfg(any(
        feature = "google",
        feature = "google-vertex",
        feature = "protocol-gemini"
    ))]
    if let Some(tool) = google::provider_defined_tool(id) {
        return Some(tool);
    }

    #[cfg(feature = "groq")]
    if let Some(tool) = groq::provider_defined_tool(id) {
        return Some(tool);
    }

    #[cfg(feature = "xai")]
    if let Some(tool) = xai::provider_defined_tool(id) {
        return Some(tool);
    }

    None
}
