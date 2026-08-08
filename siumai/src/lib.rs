//! The ergonomic Siumai facade.
//!
//! The facade exposes six stable provider-neutral model families, optional
//! immutable routing, and curated provider APIs under explicit namespaces.
//! Native provider extensions remain provider-owned and are not flattened into
//! a least-common-denominator client.
//!
//! The root README is included below so its maintained Rust examples are
//! compiled by the facade doctest lane.

#![doc = include_str!("../../README.md")]
#![deny(unsafe_code)]

pub mod families;
pub mod prelude;
pub mod providers;
#[cfg(feature = "registry")]
pub mod registry;
#[cfg(feature = "runtime")]
pub mod runtime;

pub use siumai_core as core;
pub use siumai_core::{
    AssistantHistoryOmission, AssistantHistoryOmissionKind, AssistantHistoryProjection,
    CallOptions, Cancellation, Citation, ContentPart, EmbeddingLimits, EmbeddingModel,
    EmbeddingModelProvider, EmbeddingRequest, EmbeddingResponse, Error, ErrorKind, FinishReason,
    GenerationConfig, GenerationConfigError, ImageArtifact, ImageLimits, ImageModel,
    ImageModelProvider, ImageRequest, ImageResponse, ImageSize, InvalidId, InvalidToolCall,
    InvalidToolInput, InvalidToolSpec, LanguageIncompleteReason, LanguageModel,
    LanguageModelProvider, LanguageRequest, LanguageRequestError, LanguageResponse,
    LanguageResponseError, LanguageResponseStatus, LanguageStream, LanguageStreamEvent, MediaData,
    MediaPart, Message, MessageRole, MessageValidationError, Model, ModelDescriptor, ModelFamily,
    ModelId, ModelLookupError, OpaqueProviderItem, PartialStructuredOutput, Provider, ProviderId,
    ProviderOptionError, ProviderOptions, ProviderProvenanceError, ReplayAudience, ReplayDomain,
    ReplayDomainId, RerankCandidate, RerankLimits, RerankModel, RerankModelProvider, RerankRequest,
    RerankResponse, RerankResult, ResponseDiagnostics, ResponseMetadata, SpeechLimits, SpeechModel,
    SpeechModelProvider, SpeechRequest, SpeechResponse, StreamTerminal, StructuredOutputSpec,
    ToolCall, ToolCallParts, ToolChoice, ToolInput, ToolOutcome, ToolResult, ToolSpec,
    TranscriptSegment, TranscriptionLimits, TranscriptionModel, TranscriptionModelProvider,
    TranscriptionRequest, TranscriptionResponse, TypedProviderOptions, Usage, UsageValue, Warning,
    WarningKind,
};

#[cfg(feature = "runtime")]
pub use runtime::{
    ModelTarget, Runtime, RuntimeBuilder, RuntimeConfigError, StepOptions, generate, stream,
};

#[cfg(doctest)]
#[doc = include_str!("../../docs/migration/siumai-next.md")]
mod migration_guide_doctests {}
