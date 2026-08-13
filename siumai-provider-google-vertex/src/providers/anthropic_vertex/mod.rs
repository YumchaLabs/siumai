//! Anthropic Messages-compatible models served through Google Vertex AI.
//!
//! This module owns Google identity, Vertex endpoint construction, Google bearer
//! authentication, provider-specific options and annotations, verified model
//! advisories, and the Vertex wire projection. Anthropic Messages encoding,
//! decoding, streaming, and shared execution remain in the protocol and
//! compatible-engine crates.

mod annotations;
mod auth;
mod endpoint;
mod models;
mod options;
mod profile;
mod projection;
mod provider;
mod request_policy;

pub use annotations::{
    GoogleVertexAnthropicAnnotationResolver, GoogleVertexAnthropicCacheTtl,
    GoogleVertexAnthropicContentCache, GoogleVertexAnthropicMessageCache,
    GoogleVertexAnthropicTool, GoogleVertexAnthropicToolOptions,
    GoogleVertexAnthropicToolSpecError,
};
pub use auth::{GoogleVertexCredential, GoogleVertexCredentialError, GoogleVertexTokenSource};
pub use endpoint::GoogleVertexAnthropicEndpointError;
pub use models::{
    CLAUDE_FABLE_5, CLAUDE_HAIKU_4_5_20251001, CLAUDE_OPUS_4_5_20251101, CLAUDE_OPUS_4_6,
    CLAUDE_OPUS_4_7, CLAUDE_OPUS_4_8, CLAUDE_OPUS_5, CLAUDE_SONNET_4_5_20250929, CLAUDE_SONNET_4_6,
    CLAUDE_SONNET_5, current_models,
};
pub use options::GoogleVertexAnthropicMessagesOptions;
pub use profile::GoogleVertexAnthropicProfileError;
pub use provider::{
    GOOGLE_VERTEX_ANTHROPIC_REPLAY_AUDIENCE, GoogleVertexAnthropicConfigError,
    GoogleVertexAnthropicLanguageModel, GoogleVertexAnthropicProvider,
    GoogleVertexAnthropicProviderBuilder,
};
pub use siumai_protocol_anthropic::messages::{
    ComputerToolOptions, MessagesMetadata, OutputEffort, TextEditorToolOptions, ThinkingConfig,
    ThinkingDisplay, WebSearchToolOptions,
};

#[cfg(test)]
mod tests;
