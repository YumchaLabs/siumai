pub mod builder;
pub mod client;
pub mod context;
pub mod ext;
pub mod models;
mod settings;
mod video;

/// Package version aligned with the provider crate release.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[allow(deprecated)]
pub use crate::provider_options::vertex::GoogleVertexImageProviderOptions;
#[allow(deprecated)]
pub use crate::provider_options::vertex::GoogleVertexVideoProviderOptions;
pub use crate::provider_options::vertex::{
    GoogleVertexEmbeddingModelOptions, GoogleVertexImageModelOptions, GoogleVertexReferenceImage,
    GoogleVertexVideoModelId, GoogleVertexVideoModelOptions,
};
pub use builder::GoogleVertexBuilder;
pub use client::{GoogleVertexClient, GoogleVertexConfig};
pub use ext::{VertexEmbeddingRequestExt, VertexImagenRequestExt, VertexVideoRequestExt};
pub use settings::GoogleVertexProviderSettings;
pub use siumai_protocol_gemini::standards::gemini::types::SharedIdGenerator;
