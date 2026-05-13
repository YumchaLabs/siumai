pub use siumai_provider_google_vertex::providers::anthropic_vertex::{
    GoogleVertexAnthropicMessagesModelId, GoogleVertexAnthropicProviderSettings,
    VertexAnthropicBuilder, VertexAnthropicClient, VertexAnthropicConfig,
};

/// Create the Anthropic-on-Vertex provider builder.
pub fn vertex_anthropic() -> VertexAnthropicBuilder {
    crate::Provider::vertex_anthropic()
}

/// Create the Anthropic-on-Vertex provider builder.
///
/// This is the Rust package-surface analogue of AI SDK `createVertexAnthropic()`.
pub fn create_vertex_anthropic() -> VertexAnthropicBuilder {
    vertex_anthropic()
}

/// Curated Anthropic-on-Vertex model constants aligned with the audited public subset.
pub mod models {
    pub use siumai_provider_google_vertex::providers::anthropic_vertex::models::{
        self as model_sets, chat,
    };
}

/// Provider tool factories that return `Tool` directly (Vertex Anthropic supported subset).
pub mod tools {
    pub use siumai_provider_google_vertex::providers::anthropic_vertex::tools::*;
}

/// Provider-executed tool builders and typed helper inputs.
pub mod hosted_tools {
    pub use siumai_provider_google_vertex::providers::anthropic_vertex::hosted_tools::*;
}

/// Compatibility alias for older imports.
pub mod provider_tools {
    pub use siumai_provider_google_vertex::providers::anthropic_vertex::provider_tools::*;
}

/// Typed response metadata helpers (`ChatResponse.provider_metadata["anthropic"]`).
pub mod metadata {
    pub use siumai_provider_google_vertex::providers::anthropic_vertex::{
        AnthropicChatResponseExt, AnthropicCitation, AnthropicCitationsBlock,
        AnthropicContentPartExt, AnthropicMessageContainerMetadata, AnthropicMessageContainerSkill,
        AnthropicMessageMetadata, AnthropicMetadata, AnthropicServerToolUse, AnthropicSource,
        AnthropicToolCallMetadata, AnthropicToolCaller, AnthropicUsageIteration,
    };
}

pub use metadata::{
    AnthropicChatResponseExt, AnthropicCitation, AnthropicCitationsBlock, AnthropicContentPartExt,
    AnthropicMessageContainerMetadata, AnthropicMessageContainerSkill, AnthropicMessageMetadata,
    AnthropicMetadata, AnthropicServerToolUse, AnthropicSource, AnthropicToolCallMetadata,
    AnthropicToolCaller, AnthropicUsageIteration,
};

/// Typed provider options (`provider_options_map["anthropic"]` on the Vertex wrapper path).
pub mod options {
    pub use siumai_provider_google_vertex::providers::anthropic_vertex::{
        VertexAnthropicChatRequestExt, VertexAnthropicOptions, VertexAnthropicStructuredOutputMode,
        VertexAnthropicThinkingMode,
    };
}

pub use models::{chat, model_sets};
pub use options::{
    VertexAnthropicChatRequestExt, VertexAnthropicOptions, VertexAnthropicStructuredOutputMode,
    VertexAnthropicThinkingMode,
};

/// Non-unified Anthropic-on-Vertex extension APIs (escape hatches).
pub mod ext {
    pub use siumai_provider_google_vertex::providers::anthropic_vertex::ext::*;
}
