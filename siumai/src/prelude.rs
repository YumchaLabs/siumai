//! Common imports for the stable Siumai surface.

pub use crate::families::{embedding, image, rerank, speech, transcription};
pub use crate::{
    CallOptions, Cancellation, ContentPart, EmbeddingModel, EmbeddingRequest, EmbeddingResponse,
    Error, ImageModel, ImageRequest, ImageResponse, LanguageCallError, LanguageCompletionReason,
    LanguageIncompleteReason, LanguageInput, LanguageModel, LanguageRequest, LanguageResponse,
    LanguageStream, LanguageStreamEvent, LanguageTermination, Message, MessagePart, MessageRole,
    Model, ModelDescriptor, ModelFamily, ModelId, PartialLanguageOutput, Provider, ProviderId,
    RerankCandidate, RerankModel, RerankRequest, RerankResponse, RetryIntent, SpeechModel,
    SpeechRequest, SpeechResponse, StreamTerminal, ToolChoice, ToolSpec, TranscriptionModel,
    TranscriptionRequest, TranscriptionResponse, Usage, UsageUpdate, UsageUpdateKind,
};
pub use crate::{LanguageCall, language};

#[cfg(feature = "runtime")]
pub use crate::{RunBudget, RunTimeouts, Runtime, StepOptions, generate, stream};

#[cfg(feature = "registry")]
pub use crate::registry::{
    ModelReference, RegisterProviderError, Registry, RegistryBuildError, RegistryBuilder,
    RegistryBuilderExt, RegistryModelContext, RegistryResolveError, RegistrySnapshot, RouteId,
};
