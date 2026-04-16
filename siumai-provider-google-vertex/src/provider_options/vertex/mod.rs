pub mod embedding;
pub mod imagen;

pub use embedding::VertexEmbeddingOptions;
pub use imagen::{
    VertexImagenEditOptions, VertexImagenInlineImage, VertexImagenMaskImageConfig,
    VertexImagenOptions, VertexImagenReferenceImage,
};

/// AI SDK-style alias for Google Vertex embedding model options.
pub type GoogleVertexEmbeddingModelOptions = VertexEmbeddingOptions;

/// AI SDK-style alias for Google Vertex image model options.
pub type GoogleVertexImageModelOptions = VertexImagenOptions;

/// Deprecated AI SDK compatibility alias for Google Vertex image options.
#[deprecated(note = "Use `GoogleVertexImageModelOptions` instead.")]
pub type GoogleVertexImageProviderOptions = GoogleVertexImageModelOptions;
