//! siumai-core
//!
//! Provider-agnostic runtime, types, and shared execution primitives.
#![deny(unsafe_code)]

pub mod auth;
pub mod builder;
pub mod client;
pub mod compat;
pub mod completion;
pub mod core;
pub mod custom_provider;
pub mod defaults;
pub mod embedding;
pub mod encoding;
pub mod error;
pub mod execution;
pub mod experimental;
pub mod image;
pub mod language;
pub mod model;
pub mod observability;
pub mod options;
pub mod params;
pub mod profile;
pub mod provider;
pub mod rerank;
pub mod retry;
pub mod retry_api;
pub mod speech;
pub mod standards;
pub mod stream;
pub mod streaming;
pub mod structured_output;
pub mod text;
pub mod tool;
pub mod tooling;
pub mod traits;
pub mod transcription;
pub mod types;
pub mod ui;
pub mod usage;
pub mod utils;
pub mod video;

pub use error::{
    DiagnosticHeaderError, DiagnosticTextError, Error, ErrorContext, ErrorDetail, ErrorKind,
    LlmError, LlmErrorExt, PublicDiagnosticText, ResourceKind, ResponseDiagnostics,
    SafeResponseHeaders, SensitiveErrorSource, SensitiveResponse,
};
pub use language::{
    Citation, ContentPart, DEFAULT_OPAQUE_COLLECTION_BYTE_LIMIT, DEFAULT_OPAQUE_ITEM_COUNT_LIMIT,
    DEFAULT_OPAQUE_ITEM_LIMIT, FinishReason, GenerationConfig, GenerationConfigError,
    LanguageIncompleteReason, LanguageRequest, LanguageRequestError, LanguageResponse,
    LanguageResponseError, LanguageResponseStatus, MediaData, MediaPart, Message, MessageRole,
    OpaqueProviderBudget, OpaqueProviderItem, OpaqueProviderItemBuilder, OpaqueProviderItemError,
    PartialStructuredOutput, ProviderItemRelation, ProviderProvenance, StructuredOutputSpec,
    ToolChoice, Warning, WarningKind,
};
pub use model::{
    EmbeddingLimits, EmbeddingModel, EmbeddingRequest, EmbeddingResponse, ImageArtifact,
    ImageLimits, ImageModel, ImageRequest, ImageResponse, ImageSize, LanguageModel, Model,
    ModelDescriptor, ModelFamily, RerankCandidate, RerankLimits, RerankModel, RerankRequest,
    RerankResponse, RerankResult, ResponseMetadata, SpeechLimits, SpeechModel, SpeechRequest,
    SpeechResponse, TranscriptSegment, TranscriptionLimits, TranscriptionModel,
    TranscriptionRequest, TranscriptionResponse,
};
pub use options::{
    CallOptions, Cancellation, ProviderOptionError, ProviderOptionLayers, ProviderOptionMerger,
    ProviderOptionOrigin, ProviderOptions, RetryIntent, TypedProviderOptions,
};
pub use profile::{
    ApiStability, AvailabilityScope, CatalogError, GenericSupportClaim, ModelCatalog,
    ModelLifecycle, ModelProfile, OfficialSource, ProfileError, ProviderProfile, SupportFidelity,
    SupportScope, VerificationDate, VerificationEvidence, VerifiedFidelity, VerifiedSupportClaim,
};
pub use provider::{
    ApiModeId, EmbeddingModelProvider, ImageModelProvider, InvalidId, LanguageModelProvider,
    ModelAdvisory, ModelFactory, ModelId, ModelLookupError, ModelOperation, ModelPolicy,
    ModelPolicyContext, ModelPolicyDecision, PlatformId, ProfileId, ProtocolContractId, ProtocolId,
    Provider, ProviderId, ProviderRegistration, ProviderScope, RerankModelProvider, RouteId,
    SpeechModelProvider, SupportState, TranscriptionModelProvider, UnsupportedReason,
};
pub use stream::{
    DecoderLifecycle, LanguageStream, LanguageStreamDecoder, LanguageStreamEvent,
    StreamContractError, StreamLifecycle, StreamTerminal,
};
pub use tool::{
    ExecutionOwner, InvalidToolSpec, ToolBindingIdentity, ToolCall, ToolOutcome, ToolResult,
    ToolSpec,
};
pub use usage::{Usage, UsageValue};
