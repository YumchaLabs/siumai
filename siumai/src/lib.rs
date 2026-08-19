//! The ergonomic Siumai facade.
//!
//! The facade exposes six stable provider-neutral model families, optional
//! immutable routing, and curated provider APIs under explicit namespaces.
//! Native provider extensions remain provider-owned and are not flattened into
//! a least-common-denominator client.
//!
//! The facade README is included below so the published crate remains
//! self-contained and its maintained Rust examples stay in the doctest lane.

#![doc = include_str!("../README.md")]
#![deny(unsafe_code)]

mod call;
pub mod embedding;
pub mod image;
pub mod language;
pub mod prelude;
pub mod providers;
#[cfg(feature = "registry")]
pub mod registry;
pub mod rerank;
#[cfg(feature = "runtime")]
pub mod runtime;
mod siumai;
pub mod speech;
pub mod transcription;
#[cfg(feature = "transport")]
pub mod transport;

#[cfg(feature = "anthropic")]
pub use self::siumai::{AnthropicCredentialStage, AnthropicProviderStage};
pub use self::siumai::{
    EmbeddingClient, ImageClient, LanguageClient, RerankClient, Siumai, SiumaiBuilder,
    SpeechClient, TranscriptionClient,
};
#[cfg(feature = "google")]
pub use self::siumai::{GeminiCredentialStage, GeminiProviderStage};
#[cfg(feature = "openai-compatible")]
pub use self::siumai::{
    OpenAiCompatibleCredentialStage, OpenAiCompatibleProfileStage, OpenAiCompatibleProviderStage,
};
#[cfg(feature = "openai")]
pub use self::siumai::{OpenAiCredentialStage, OpenAiProviderStage};
pub use embedding::EmbeddingCall;
pub use image::ImageCall;
pub use language::LanguageCall;
pub use rerank::RerankCall;
pub use siumai_core as core;
pub use siumai_core::{
    AssistantHistoryOmission, AssistantHistoryOmissionKind, AssistantHistoryProjection,
    CallOptions, CallOptionsError, Cancellation, Citation, ContentPart, EmbeddingLimits,
    EmbeddingModel, EmbeddingModelProvider, EmbeddingRequest, EmbeddingResponse, Error, ErrorKind,
    GenerationConfig, GenerationConfigError, ImageArtifact, ImageLimits, ImageModel,
    ImageModelProvider, ImageRequest, ImageResponse, ImageSize, InvalidId, InvalidToolCall,
    InvalidToolInput, InvalidToolSpec, LanguageCallError, LanguageCompletionReason,
    LanguageIncompleteReason, LanguageInput, LanguageModel, LanguageModelProvider, LanguageRequest,
    LanguageRequestError, LanguageResponse, LanguageResponseError, LanguageStream,
    LanguageStreamEvent, LanguageTermination, MediaData, MediaPart, Message, MessagePart,
    MessageRole, MessageValidationError, Model, ModelDescriptor, ModelFamily, ModelId,
    ModelLookupError, OpaqueProviderItem, PartialLanguageOutput, PartialLanguageOutputBudget,
    PartialLanguageOutputError, PartialLanguageOutputPart, PartialStructuredOutput, Provider,
    ProviderId, ProviderOptionError, ProviderOptions, ProviderProvenanceError, ReplayAudience,
    ReplayDomain, ReplayDomainId, RerankCandidate, RerankLimits, RerankModel, RerankModelProvider,
    RerankRequest, RerankResponse, RerankResult, ResponseDiagnostics, ResponseMetadata,
    RetryIntent, SpeechLimits, SpeechModel, SpeechModelProvider, SpeechRequest, SpeechResponse,
    StreamTerminal, StructuredOutputSpec, ToolCall, ToolCallParts, ToolChoice, ToolInput,
    ToolOutcome, ToolResult, ToolSpec, TranscriptSegment, TranscriptionLimits, TranscriptionModel,
    TranscriptionModelProvider, TranscriptionRequest, TranscriptionResponse, TypedProviderOptions,
    Usage, UsageUpdate, UsageUpdateKind, UsageValue, Warning, WarningKind,
};
pub use speech::SpeechCall;
pub use transcription::TranscriptionCall;

#[cfg(feature = "runtime")]
pub use runtime::{
    BudgetError, ModelTarget, RunBudget, RunBudgetBuilder, RunTimeouts, Runtime, RuntimeBuilder,
    RuntimeConfigError, StepOptions,
};
