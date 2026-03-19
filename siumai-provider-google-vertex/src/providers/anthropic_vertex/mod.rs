pub mod builder;
pub mod client;
pub mod context;
pub mod ext;
pub mod spec;

pub use crate::provider_options::anthropic_vertex::{
    VertexAnthropicOptions, VertexAnthropicStructuredOutputMode, VertexAnthropicThinkingMode,
};
pub use builder::VertexAnthropicBuilder;
pub use client::{VertexAnthropicClient, VertexAnthropicConfig};
pub use ext::VertexAnthropicChatRequestExt;
pub use siumai_protocol_anthropic::provider_metadata::anthropic::{
    AnthropicChatResponseExt, AnthropicCitation, AnthropicCitationsBlock, AnthropicContentPartExt,
    AnthropicMetadata, AnthropicServerToolUse, AnthropicSource, AnthropicToolCallMetadata,
    AnthropicToolCaller,
};
