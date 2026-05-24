//! Hosted tools are part of the stable unified experience (Vercel-aligned).

#[cfg(any(feature = "openai", feature = "protocol-openai"))]
pub mod openai {
    pub use siumai_protocol_openai::hosted_tools::openai::*;
}

#[cfg(any(feature = "anthropic", feature = "protocol-anthropic"))]
pub mod anthropic {
    pub use siumai_protocol_anthropic::hosted_tools::anthropic::*;
}

#[cfg(any(
    feature = "google",
    feature = "google-vertex",
    feature = "protocol-gemini"
))]
pub mod google {
    pub use siumai_protocol_gemini::hosted_tools::google::*;
}
