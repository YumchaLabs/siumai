pub use siumai_provider_minimaxi::providers::minimaxi::{MinimaxiBuilder, MinimaxiClient};

/// Curated MiniMaxi model constants for the public provider surface.
pub mod models {
    pub use siumai_provider_minimaxi::providers::minimaxi::models::{
        self as model_sets, chat, image, music, speech, video,
    };
}

/// Typed response metadata helpers (`ChatResponse.provider_metadata["minimaxi"]`).
pub mod metadata {
    pub use siumai_provider_minimaxi::provider_metadata::minimaxi::{
        MinimaxiChatResponseExt, MinimaxiCitation, MinimaxiCitationsBlock, MinimaxiContentPartExt,
        MinimaxiMetadata, MinimaxiServerToolUse, MinimaxiSource, MinimaxiToolCallMetadata,
        MinimaxiToolCaller,
    };
}
pub use metadata::{
    MinimaxiChatResponseExt, MinimaxiCitation, MinimaxiCitationsBlock, MinimaxiContentPartExt,
    MinimaxiMetadata, MinimaxiServerToolUse, MinimaxiSource, MinimaxiToolCallMetadata,
    MinimaxiToolCaller,
};

/// Typed provider options (`provider_options_map["minimaxi"]`).
pub mod options {
    pub use siumai_provider_minimaxi::provider_options::{
        MinimaxiOptions, MinimaxiResponseFormat, MinimaxiThinkingModeConfig, MinimaxiTtsOptions,
        MinimaxiVideoOptions,
    };
    pub use siumai_provider_minimaxi::providers::minimaxi::ext::tts::MinimaxiTtsRequestBuilder;
    pub use siumai_provider_minimaxi::providers::minimaxi::ext::tts_options::MinimaxiTtsRequestExt;
    pub use siumai_provider_minimaxi::providers::minimaxi::ext::{
        MinimaxiChatRequestExt, MinimaxiVideoRequestExt,
    };
}

// Provider-owned typed options (kept out of `siumai-core`).
pub use models::{chat, image, model_sets, music, speech, video};
pub use options::{
    MinimaxiChatRequestExt, MinimaxiOptions, MinimaxiResponseFormat, MinimaxiThinkingModeConfig,
    MinimaxiTtsOptions, MinimaxiTtsRequestBuilder, MinimaxiTtsRequestExt, MinimaxiVideoOptions,
    MinimaxiVideoRequestExt,
};

/// Non-unified MiniMaxi extension APIs (escape hatches).
pub mod ext {
    pub use siumai_provider_minimaxi::providers::minimaxi::ext::{
        music, structured_output, thinking, video,
    };
}

/// Provider-specific resources not covered by the unified families.
pub mod resources {
    /// MiniMaxi file management API client (extension resource).
    pub use siumai_provider_minimaxi::providers::minimaxi::files::MinimaxiFiles;
}

/// MiniMaxi low-level config (for advanced use; prefer the builder for most cases).
pub use siumai_provider_minimaxi::providers::minimaxi::config::MinimaxiConfig;
