//! The ergonomic Siumai facade.
//!
//! The facade exposes six stable provider-neutral model families, optional
//! immutable routing, and curated provider APIs under explicit namespaces.
//! Native provider extensions remain provider-owned and are not flattened into
//! a least-common-denominator client.

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
    CallOptions, Cancellation, Citation, ContentPart, EmbeddingLimits, EmbeddingModel,
    EmbeddingModelProvider, EmbeddingRequest, EmbeddingResponse, Error, ErrorKind, FinishReason,
    GenerationConfig, GenerationConfigError, ImageArtifact, ImageLimits, ImageModel,
    ImageModelProvider, ImageRequest, ImageResponse, ImageSize, InvalidId, InvalidToolSpec,
    LanguageIncompleteReason, LanguageModel, LanguageModelProvider, LanguageRequest,
    LanguageRequestError, LanguageResponse, LanguageResponseError, LanguageResponseStatus,
    LanguageStream, LanguageStreamEvent, MediaData, MediaPart, Message, MessageRole, Model,
    ModelDescriptor, ModelFamily, ModelId, ModelLookupError, OpaqueProviderItem,
    PartialStructuredOutput, Provider, ProviderId, ProviderOptionError, ProviderOptions,
    RerankCandidate, RerankLimits, RerankModel, RerankModelProvider, RerankRequest, RerankResponse,
    RerankResult, ResponseMetadata, SpeechLimits, SpeechModel, SpeechModelProvider, SpeechRequest,
    SpeechResponse, StreamTerminal, StructuredOutputSpec, ToolCall, ToolChoice, ToolOutcome,
    ToolResult, ToolSpec, TranscriptSegment, TranscriptionLimits, TranscriptionModel,
    TranscriptionModelProvider, TranscriptionRequest, TranscriptionResponse, TypedProviderOptions,
    Usage, UsageValue, Warning, WarningKind,
};

#[cfg(feature = "runtime")]
pub use runtime::{
    ModelTarget, Runtime, RuntimeBuilder, RuntimeConfigError, StepOptions, generate, stream,
};
