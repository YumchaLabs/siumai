pub mod builder;
pub mod client;
pub mod context;
pub mod ext;
pub mod models;

#[allow(deprecated)]
pub use crate::provider_options::vertex::GoogleVertexImageProviderOptions;
pub use crate::provider_options::vertex::{
    GoogleVertexEmbeddingModelOptions, GoogleVertexImageModelOptions,
};
pub use builder::GoogleVertexBuilder;
pub use client::{GoogleVertexClient, GoogleVertexConfig};
pub use ext::{VertexEmbeddingRequestExt, VertexImagenRequestExt};
