pub use siumai_provider_google_vertex::providers::vertex::{
    GoogleVertexBuilder, GoogleVertexClient, GoogleVertexConfig, GoogleVertexProviderSettings,
    SharedIdGenerator, VERSION,
};

/// Create the Google Vertex provider builder.
pub fn google_vertex() -> GoogleVertexBuilder {
    crate::compat::Provider::google_vertex()
}

/// Create the Google Vertex provider builder.
///
/// This is the Rust package-surface analogue of AI SDK `createGoogleVertex()`.
pub fn create_google_vertex() -> GoogleVertexBuilder {
    google_vertex()
}

/// Create the Google Vertex provider builder.
///
/// Deprecated package alias of `google_vertex()`, mirroring AI SDK `vertex`.
pub fn vertex() -> GoogleVertexBuilder {
    google_vertex()
}

/// Create the Google Vertex provider builder.
///
/// Deprecated package alias of `create_google_vertex()`, mirroring AI SDK `createVertex`.
pub fn create_vertex() -> GoogleVertexBuilder {
    create_google_vertex()
}

/// Curated Google Vertex model constants aligned with the audited public subset.
pub mod models {
    pub use siumai_provider_google_vertex::providers::vertex::models::{
        self as model_sets, chat, embedding, image, video,
    };
}

/// Provider tool factories that return `Tool` directly (Vercel-aligned `googleVertexTools`).
pub mod tools {
    use siumai_core::types::Tool;

    pub use crate::tools::google::{
        code_execution, enterprise_web_search, google_maps, url_context,
    };

    pub fn google_search() -> Tool {
        crate::hosted_tools::google::google_search().build()
    }

    pub fn file_search(file_search_store_names: Vec<String>) -> Tool {
        crate::hosted_tools::google::file_search()
            .with_file_search_store_names(file_search_store_names)
            .build()
    }

    pub fn vertex_rag_store(rag_corpus: impl Into<String>) -> Tool {
        crate::hosted_tools::google::vertex_rag_store(rag_corpus).build()
    }
}

/// Provider-executed tool builders (typed args).
pub mod hosted_tools {
    pub use crate::hosted_tools::google::{
        FileSearchConfig, GoogleSearchConfig, VertexRagStoreConfig, code_execution,
        enterprise_web_search, file_search, google_maps, google_search, url_context,
        vertex_rag_store,
    };
}

/// Typed provider options (`provider_options_map["vertex"]`).
pub mod options {
    #[allow(deprecated)]
    pub use siumai_provider_google_vertex::provider_options::vertex::GoogleVertexImageProviderOptions;
    #[allow(deprecated)]
    pub use siumai_provider_google_vertex::provider_options::vertex::GoogleVertexVideoProviderOptions;
    pub use siumai_provider_google_vertex::provider_options::vertex::{
        GoogleVertexEmbeddingModelOptions, GoogleVertexImageModelOptions,
        GoogleVertexReferenceImage, GoogleVertexVideoModelId, GoogleVertexVideoModelOptions,
        VertexEmbeddingOptions, VertexImagenEditMode, VertexImagenEditOptions,
        VertexImagenInlineImage, VertexImagenMaskImageConfig, VertexImagenMaskMode,
        VertexImagenOptions, VertexImagenReferenceImage, VertexImagenSafetySetting,
        VertexImagenSampleImageSize, VertexPersonGeneration,
    };
    pub use siumai_provider_google_vertex::providers::vertex::{
        VertexEmbeddingRequestExt, VertexImagenRequestExt, VertexVideoRequestExt,
    };
}

pub use models::{chat, embedding, image, model_sets, video};
#[allow(deprecated)]
pub use options::GoogleVertexImageProviderOptions;
#[allow(deprecated)]
pub use options::GoogleVertexVideoProviderOptions;
pub use options::{
    GoogleVertexEmbeddingModelOptions, GoogleVertexImageModelOptions, GoogleVertexReferenceImage,
    GoogleVertexVideoModelId, GoogleVertexVideoModelOptions, VertexEmbeddingOptions,
    VertexEmbeddingRequestExt, VertexImagenEditMode, VertexImagenEditOptions,
    VertexImagenInlineImage, VertexImagenMaskImageConfig, VertexImagenMaskMode,
    VertexImagenOptions, VertexImagenReferenceImage, VertexImagenRequestExt,
    VertexImagenSafetySetting, VertexImagenSampleImageSize, VertexPersonGeneration,
    VertexVideoRequestExt,
};

/// Typed response metadata helpers (`ChatResponse.provider_metadata["vertex"]`).
pub mod metadata {
    pub use siumai_provider_google_vertex::provider_metadata::vertex::{
        VertexChatResponseExt, VertexContentPartExt, VertexGroundingMetadata, VertexLogprobsResult,
        VertexMetadata, VertexPromptFeedback, VertexSafetyRating, VertexSource,
        VertexUrlContextMetadata, VertexUsageMetadata,
    };
}
pub use metadata::{
    VertexChatResponseExt, VertexContentPartExt, VertexGroundingMetadata, VertexLogprobsResult,
    VertexMetadata, VertexPromptFeedback, VertexSafetyRating, VertexSource,
    VertexUrlContextMetadata, VertexUsageMetadata,
};
