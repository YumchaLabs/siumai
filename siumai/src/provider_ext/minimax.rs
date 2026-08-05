pub use siumai_provider_minimax::providers::minimax::{MinimaxBuilder, MinimaxClient};

/// Curated MiniMax model constants for the public provider surface.
pub mod models {
    pub use siumai_provider_minimax::providers::minimax::models::{
        self as model_sets, chat, image, music, speech, video,
    };
}

/// Typed response metadata helpers (`ChatResponse.provider_metadata["minimax"]`).
pub mod metadata {
    pub use siumai_provider_minimax::provider_metadata::minimax::{
        MinimaxChatResponseExt, MinimaxCitation, MinimaxCitationsBlock, MinimaxContentPartExt,
        MinimaxMetadata, MinimaxServerToolUse, MinimaxSource, MinimaxToolCallMetadata,
        MinimaxToolCaller,
    };
}
pub use metadata::{
    MinimaxChatResponseExt, MinimaxCitation, MinimaxCitationsBlock, MinimaxContentPartExt,
    MinimaxMetadata, MinimaxServerToolUse, MinimaxSource, MinimaxToolCallMetadata,
    MinimaxToolCaller,
};

/// Typed provider options (`provider_options_map["minimax"]`).
pub mod options {
    pub use siumai_provider_minimax::provider_options::{
        MinimaxOptions, MinimaxServiceTier, MinimaxThinking, MinimaxTtsOptions,
        MinimaxVideoOptions,
    };
    pub use siumai_provider_minimax::providers::minimax::ext::tts::MinimaxTtsRequestBuilder;
    pub use siumai_provider_minimax::providers::minimax::ext::tts_options::MinimaxTtsRequestExt;
    pub use siumai_provider_minimax::providers::minimax::ext::{
        MinimaxChatRequestExt, MinimaxVideoRequestExt,
    };
}

// Provider-owned typed options (kept out of `siumai-core`).
pub use models::{chat, image, model_sets, music, speech, video};
pub use options::{
    MinimaxChatRequestExt, MinimaxOptions, MinimaxServiceTier, MinimaxThinking, MinimaxTtsOptions,
    MinimaxTtsRequestBuilder, MinimaxTtsRequestExt, MinimaxVideoOptions, MinimaxVideoRequestExt,
};

/// Non-unified MiniMax extension APIs (escape hatches).
pub mod ext {
    pub use siumai_provider_minimax::providers::minimax::ext::{
        music, thinking, video,
    };
}

/// Provider-specific resources not covered by the unified families.
pub mod resources {
    /// MiniMax file management API client (extension resource).
    pub use siumai_provider_minimax::providers::minimax::files::MinimaxFiles;
}

/// MiniMax low-level config (for advanced use; prefer the builder for most cases).
pub use siumai_provider_minimax::providers::minimax::config::MinimaxConfig;
