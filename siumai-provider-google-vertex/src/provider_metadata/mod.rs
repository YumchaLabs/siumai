//! Provider-owned typed response metadata.

pub mod vertex;

pub use vertex::{
    VertexChatResponseExt, VertexContentPartExt, VertexGroundingMetadata, VertexLogprobsResult,
    VertexMetadata, VertexPromptFeedback, VertexSafetyRating, VertexSource,
    VertexUrlContextMetadata, VertexUsageMetadata,
};
