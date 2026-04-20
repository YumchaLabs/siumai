pub mod builder;
pub mod client;
pub mod context;
pub mod ext;
pub mod models;
mod video;

#[allow(deprecated)]
pub use crate::provider_options::vertex::GoogleVertexImageProviderOptions;
#[allow(deprecated)]
pub use crate::provider_options::vertex::GoogleVertexVideoProviderOptions;
pub use crate::provider_options::vertex::{
    GoogleVertexEmbeddingModelOptions, GoogleVertexImageModelOptions,
    GoogleVertexReferenceImage, GoogleVertexVideoModelId, GoogleVertexVideoModelOptions,
};
pub use builder::GoogleVertexBuilder;
pub use client::{GoogleVertexClient, GoogleVertexConfig};
pub use ext::{VertexEmbeddingRequestExt, VertexImagenRequestExt, VertexVideoRequestExt};
