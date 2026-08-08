//! Curated Anthropic-on-Vertex provider facade.

pub use siumai_provider_google_vertex::{
    GOOGLE_VERTEX_ANTHROPIC_REPLAY_AUDIENCE, GoogleVertexAnthropicConfigError,
    GoogleVertexAnthropicEndpointError, GoogleVertexAnthropicLanguageModel,
    GoogleVertexAnthropicProfileError, GoogleVertexAnthropicProvider,
    GoogleVertexAnthropicProviderBuilder, GoogleVertexCredential, GoogleVertexCredentialError,
    GoogleVertexTokenSource,
};

pub mod models {
    pub use siumai_provider_google_vertex::anthropic_vertex::{
        CLAUDE_FABLE_5, CLAUDE_HAIKU_4_5_20251001, CLAUDE_OPUS_4_5_20251101, CLAUDE_OPUS_4_6,
        CLAUDE_OPUS_4_7, CLAUDE_OPUS_4_8, CLAUDE_OPUS_5, CLAUDE_SONNET_4_5_20250929,
        CLAUDE_SONNET_4_6, CLAUDE_SONNET_5, current_models,
    };
}

pub mod options {
    pub use siumai_provider_google_vertex::anthropic_vertex::{
        GoogleVertexAnthropicMessagesOptions, MessagesMetadata, OutputEffort, ThinkingConfig,
        ThinkingDisplay,
    };
}

pub mod annotations {
    pub use siumai_provider_google_vertex::{
        GoogleVertexAnthropicAnnotationResolver, GoogleVertexAnthropicCacheTtl,
        GoogleVertexAnthropicContentCache, GoogleVertexAnthropicMessageCache,
        GoogleVertexAnthropicTool, GoogleVertexAnthropicToolOptions,
        GoogleVertexAnthropicToolSpecError,
    };
}

pub mod tools {
    pub use siumai_provider_google_vertex::anthropic_vertex::{
        ComputerToolOptions, TextEditorToolOptions, WebSearchToolOptions,
    };
}
