//! siumai-core
//!
//! Provider-neutral model-family contracts for Siumai.
#![deny(unsafe_code)]

pub mod annotations;
pub mod error;
pub mod experimental;
pub mod language;
pub mod model;
pub mod options;
pub mod profile;
pub mod provider;
pub mod stream;
pub mod tool;
pub mod usage;

pub use annotations::{
    ContentAnnotationTarget, ContentAnnotations, DEFAULT_PROVIDER_ANNOTATION_COLLECTION_BYTE_LIMIT,
    DEFAULT_PROVIDER_ANNOTATION_ENTRY_BYTE_LIMIT, DEFAULT_PROVIDER_ANNOTATION_ENTRY_COUNT_LIMIT,
    DEFAULT_PROVIDER_ANNOTATION_NAMESPACE_LIMIT, MessageAnnotationTarget, MessageAnnotations,
    ProviderAnnotation, ProviderAnnotationBudget, ProviderAnnotationError, ProviderAnnotationKind,
    ProviderAnnotationTarget, ProviderAnnotations, ToolAnnotationTarget, ToolAnnotations,
    TypedProviderAnnotation,
};
pub use error::{
    DiagnosticTextError, Error, ErrorContext, ErrorDetail, ErrorKind, MAX_RETRY_AFTER_HINT,
    PublicDiagnosticText, ResourceKind, ResponseDiagnostics, SensitiveErrorSource,
    SensitiveResponse,
};
pub use language::{
    AssistantHistoryOmission, AssistantHistoryOmissionKind, AssistantHistoryProjection, Citation,
    ContentPart, DEFAULT_OPAQUE_COLLECTION_BYTE_LIMIT, DEFAULT_OPAQUE_ITEM_COUNT_LIMIT,
    DEFAULT_OPAQUE_ITEM_LIMIT, FinishReason, GenerationConfig, GenerationConfigError,
    LanguageIncompleteReason, LanguageRequest, LanguageRequestBudget, LanguageRequestError,
    LanguageResponse, LanguageResponseError, LanguageResponseStatus, MediaData, MediaPart, Message,
    MessagePart, MessageRole, MessageValidationError, OpaqueProviderBudget, OpaqueProviderItem,
    OpaqueProviderItemBuilder, OpaqueProviderItemError, PartialStructuredOutput,
    ProviderItemRelation, ProviderProvenance, ProviderProvenanceError, StructuredOutputSpec,
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
    CallOptions, Cancellation, MAX_PROVIDER_OPTION_ENTRIES, MAX_PROVIDER_OPTION_TARGETS,
    MAX_PROVIDER_OPTION_TOTAL_BYTES, ProviderOptionBindingRequirement, ProviderOptionContext,
    ProviderOptionError, ProviderOptionLayers, ProviderOptionMerger, ProviderOptionOrigin,
    ProviderOptionSelection, ProviderOptionTarget, ProviderOptions, RetryIntent,
    TypedProviderOptions,
};
pub use profile::{
    ApiStability, CatalogError, GenericSupportClaim, ModelCatalog, ModelLifecycle, ModelProfile,
    NativeSupportScope, NativeSurfaceBinding, NativeSurfaceKind, NativeVerificationEvidence,
    OfficialSource, ProfileError, ProviderProfile, ProviderSupportManifest, SupportFidelity,
    SupportManifestError, SupportScope, UpstreamLifecycle, UpstreamMaturity, UpstreamSupportStatus,
    VerificationDate, VerificationEvidence, VerifiedFidelity, VerifiedNativeSupportClaim,
    VerifiedSupportClaim,
};
pub use provider::{
    ApiModeId, EmbeddingModelProvider, ImageModelProvider, InvalidId, LanguageModelProvider,
    ModelFactory, ModelId, ModelLookupError, ModelOperation, NativeSurfaceId, PlatformId,
    ProfileId, ProtocolContractId, ProtocolId, Provider, ProviderId, ProviderInstanceId,
    ProviderRegistration, ProviderRegistrationError, ProviderScope, ReplayAudience, ReplayDomain,
    ReplayDomainId, RerankModelProvider, RouteId, SpeechModelProvider, TranscriptionModelProvider,
};
pub use stream::{
    DecoderLifecycle, LanguageStream, LanguageStreamDecoder, LanguageStreamEvent,
    StreamContractError, StreamLifecycle, StreamTerminal,
};
pub use tool::{
    DEFAULT_TOOL_INPUT_BYTE_LIMIT, ExecutionOwner, InvalidToolCall, InvalidToolInput,
    InvalidToolSpec, ToolBindingIdentity, ToolCall, ToolCallParts, ToolInput, ToolOutcome,
    ToolResult, ToolSpec, ToolSpecParts,
};
pub use usage::{Usage, UsageValue};
