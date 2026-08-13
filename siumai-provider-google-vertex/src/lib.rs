//! siumai-provider-google-vertex
//!
//! Anthropic Messages provider served through Google Vertex AI.
#![deny(unsafe_code)]

mod providers;

pub use providers::anthropic_vertex;
pub use providers::anthropic_vertex::{
    GOOGLE_VERTEX_ANTHROPIC_REPLAY_AUDIENCE, GoogleVertexAnthropicAnnotationResolver,
    GoogleVertexAnthropicCacheTtl, GoogleVertexAnthropicConfigError,
    GoogleVertexAnthropicContentCache, GoogleVertexAnthropicEndpointError,
    GoogleVertexAnthropicLanguageModel, GoogleVertexAnthropicMessageCache,
    GoogleVertexAnthropicMessagesOptions, GoogleVertexAnthropicProfileError,
    GoogleVertexAnthropicProvider, GoogleVertexAnthropicProviderBuilder, GoogleVertexAnthropicTool,
    GoogleVertexAnthropicToolOptions, GoogleVertexAnthropicToolSpecError, GoogleVertexCredential,
    GoogleVertexCredentialError, GoogleVertexTokenSource,
};
