pub mod builder;
pub mod client;
pub mod context;
pub mod ext;
pub mod models;
mod settings;
pub mod spec;

/// Provider tool factories that return `Tool` directly (Vertex Anthropic supported subset).
pub mod tools {
    use crate::types::Tool;

    pub use crate::hosted_tools::anthropic::{
        tool_search_bm25_20251119, tool_search_regex_20251119,
    };
    pub use siumai_core::tools::anthropic::{
        bash_20241022, bash_20250124, computer_20241022, text_editor_20241022,
        text_editor_20250124, text_editor_20250429, text_editor_20250728,
    };

    pub fn web_search_20250305() -> Tool {
        crate::hosted_tools::anthropic::web_search_20250305().build()
    }
}

/// Provider-executed tool builders and typed helper inputs.
pub mod hosted_tools {
    pub use crate::hosted_tools::anthropic::{
        UserLocation, WebSearch20250305Config, tool_search_bm25_20251119,
        tool_search_regex_20251119, web_search_20250305,
    };
}

/// Compatibility alias for older imports.
pub mod provider_tools {
    pub use super::tools::*;
}

pub use crate::provider_options::anthropic_vertex::{
    VertexAnthropicOptions, VertexAnthropicStructuredOutputMode, VertexAnthropicThinkingMode,
};
pub use builder::VertexAnthropicBuilder;
pub use client::{VertexAnthropicClient, VertexAnthropicConfig};
pub use ext::VertexAnthropicChatRequestExt;
pub use models::GoogleVertexAnthropicMessagesModelId;
pub use settings::GoogleVertexAnthropicProviderSettings;
pub use siumai_protocol_anthropic::provider_metadata::anthropic::{
    AnthropicChatResponseExt, AnthropicCitation, AnthropicCitationsBlock, AnthropicContentPartExt,
    AnthropicMessageContainerMetadata, AnthropicMessageContainerSkill, AnthropicMessageMetadata,
    AnthropicMetadata, AnthropicServerToolUse, AnthropicSource, AnthropicToolCallMetadata,
    AnthropicToolCaller, AnthropicUsageIteration,
};
