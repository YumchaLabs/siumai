pub use siumai_provider_ollama::providers::ollama::{OllamaBuilder, OllamaClient, OllamaConfig};

/// Curated Ollama model constants for the public provider surface.
pub mod models {
    pub use siumai_provider_ollama::providers::ollama::models::{
        self as model_sets, chat, embedding,
    };
}

/// Typed response metadata helpers (`ChatResponse.provider_metadata["ollama"]`).
pub mod metadata {
    pub use siumai_provider_ollama::provider_metadata::ollama::{
        OllamaChatResponseExt, OllamaMetadata,
    };
}

pub use metadata::{OllamaChatResponseExt, OllamaMetadata};

/// Typed provider options (`provider_options_map["ollama"]`).
pub mod options {
    pub use siumai_provider_ollama::provider_options::OllamaOptions;
    pub use siumai_provider_ollama::providers::ollama::ext::OllamaChatRequestExt;
    pub use siumai_provider_ollama::providers::ollama::types::{
        OllamaEmbeddingOptions, OllamaEmbeddingRequestExt,
    };
}

// Provider-owned typed options (kept out of `siumai-core`).
pub use models::{chat, embedding, model_sets};
pub use options::{
    OllamaChatRequestExt, OllamaEmbeddingOptions, OllamaEmbeddingRequestExt, OllamaOptions,
};

/// Non-unified Ollama extension APIs (escape hatches).
pub mod ext {
    pub use siumai_provider_ollama::providers::ollama::ext::request_options;
}

/// Provider-owned Ollama default parameter struct.
pub use siumai_provider_ollama::providers::ollama::config::OllamaParams;
