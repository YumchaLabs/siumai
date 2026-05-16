/// Lower-level Vertex MaaS text-family compat client/config aliases.
///
/// These map to the shared OpenAI-compatible runtime used by the audited Vertex MaaS
/// chat, completion, and embedding lanes. For the unified AI SDK-style provider surface,
/// use [`vertex_maas()`], [`create_vertex_maas()`], [`crate::compat::Provider::vertex_maas()`],
/// or [`SiumaiBuilder::vertex_maas()`].
pub use siumai_provider_openai_compatible::providers::openai_compatible::{
    GOOGLE_VERTEX_MAAS_VERSION as VERSION, GoogleVertexMaasClient, GoogleVertexMaasConfig,
    GoogleVertexMaasModelId, GoogleVertexMaasProviderSettings,
};
use siumai_registry::provider::SiumaiBuilder;

/// Curated Vertex MaaS model constants aligned with the audited AI SDK package subset.
pub mod models {
    pub use siumai_provider_openai_compatible::providers::openai_compatible::vertex_maas::{
        self, chat, completion, embedding,
    };
}

/// Create the unified Google Vertex MaaS provider builder.
pub fn vertex_maas() -> SiumaiBuilder {
    SiumaiBuilder::new().vertex_maas()
}

/// Create the unified Google Vertex MaaS provider builder.
///
/// This is the Rust package-surface analogue of AI SDK `createVertexMaas()`.
pub fn create_vertex_maas() -> SiumaiBuilder {
    vertex_maas()
}

pub use models::{chat, completion, embedding, vertex_maas as model_sets};
