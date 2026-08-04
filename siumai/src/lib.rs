//! The ergonomic Siumai facade.
//!
//! The facade exposes six stable provider-neutral model families, optional
//! immutable routing, and provider crates under explicit namespaces. Native
//! provider extensions remain available through their provider namespace and
//! are not flattened into a least-common-denominator client.

#![deny(unsafe_code)]

pub mod families;
pub mod prelude;
pub mod providers;

pub use siumai_core as core;
pub use siumai_core::{
    ApiModeId, ApiStability, AvailabilityScope, CallOptions, Cancellation, Citation, ContentPart,
    DecoderLifecycle, DiagnosticHeaderError, DiagnosticTextError, EmbeddingLimits, EmbeddingModel,
    EmbeddingModelProvider, EmbeddingRequest, EmbeddingResponse, Error, ErrorContext, ErrorDetail,
    ErrorKind, ExecutionOwner, FinishReason, GenerationConfig, GenerationConfigError,
    GenericSupportClaim, ImageArtifact, ImageLimits, ImageModel, ImageModelProvider, ImageRequest,
    ImageResponse, ImageSize, InvalidId, InvalidToolSpec, LanguageModel, LanguageModelProvider,
    LanguageRequest, LanguageRequestError, LanguageResponse, LanguageStream, LanguageStreamDecoder,
    LanguageStreamEvent, MediaData, MediaPart, Message, MessageRole, Model, ModelAdvisory,
    ModelCatalog, ModelDescriptor, ModelFamily, ModelId, ModelLifecycle, ModelLookupError,
    ModelOperation, ModelPolicy, ModelPolicyContext, ModelPolicyDecision, ModelProfile,
    OfficialSource, OpaqueProviderItem, OpaqueProviderItemError, PartialStructuredOutput,
    PlatformId, ProfileError, ProfileId, ProtocolContractId, ProtocolId, Provider, ProviderId,
    ProviderOptionError, ProviderOptionLayers, ProviderOptionMerger, ProviderOptionOrigin,
    ProviderOptions, ProviderProfile, ProviderProvenance, ProviderRegistration, ProviderScope,
    PublicDiagnosticText, RerankCandidate, RerankLimits, RerankModel, RerankModelProvider,
    RerankRequest, RerankResponse, RerankResult, ResourceKind, ResponseDiagnostics,
    ResponseMetadata, RetryIntent, RouteId, SafeResponseHeaders, SensitiveErrorSource,
    SensitiveResponse, SpeechLimits, SpeechModel, SpeechModelProvider, SpeechRequest,
    SpeechResponse, StreamContractError, StreamLifecycle, StreamTerminal, StructuredOutputSpec,
    SupportFidelity, SupportScope, SupportState, ToolBindingIdentity, ToolCall, ToolChoice,
    ToolOutcome, ToolResult, ToolSpec, TranscriptSegment, TranscriptionLimits, TranscriptionModel,
    TranscriptionModelProvider, TranscriptionRequest, TranscriptionResponse, TypedProviderOptions,
    UnsupportedReason, Usage, UsageValue, VerificationDate, VerificationEvidence, VerifiedFidelity,
    VerifiedSupportClaim, Warning, WarningKind,
};

#[cfg(feature = "registry")]
pub use siumai_registry as registry;
