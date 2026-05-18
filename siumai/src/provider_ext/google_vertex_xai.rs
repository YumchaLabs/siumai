/// Lower-level Google Vertex xAI text-family compat client/config aliases.
///
/// These map to the shared OpenAI-compatible runtime used by the audited
/// `@ai-sdk/google-vertex/xai` chat/language-model lane. For unified construction, use
/// [`google_vertex_xai()`], [`create_google_vertex_xai()`],
/// [`crate::compat::Provider::google_vertex_xai()`], or
/// [`SiumaiBuilder::google_vertex_xai()`].
pub use siumai_provider_openai_compatible::providers::openai_compatible::{
    GOOGLE_VERTEX_XAI_VERSION as VERSION, GoogleVertexXaiClient, GoogleVertexXaiConfig,
    GoogleVertexXaiModelId, GoogleVertexXaiProviderSettings,
};
use siumai_registry::provider::SiumaiBuilder;

/// Curated Google Vertex xAI model constants aligned with the AI SDK package surface.
pub mod models {
    pub use siumai_provider_openai_compatible::providers::openai_compatible::google_vertex_xai::{
        self, chat,
    };
}

/// Create the unified Google Vertex xAI provider builder.
pub fn google_vertex_xai() -> SiumaiBuilder {
    SiumaiBuilder::new().google_vertex_xai()
}

/// Create the unified Google Vertex xAI provider builder.
///
/// This is the Rust package-surface analogue of AI SDK `createGoogleVertexXai()`.
pub fn create_google_vertex_xai() -> SiumaiBuilder {
    google_vertex_xai()
}

/// Alias for [`google_vertex_xai()`].
pub fn vertex_xai() -> SiumaiBuilder {
    google_vertex_xai()
}

pub use models::{chat, google_vertex_xai as model_sets};
