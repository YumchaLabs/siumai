//! Common imports for the stable Siumai surface.

pub use crate::{
    CallOptions, Cancellation, ContentPart, EmbeddingCall, EmbeddingModel, EmbeddingRequest,
    EmbeddingResponse, Error, ImageCall, ImageModel, ImageRequest, ImageResponse, LanguageCall,
    LanguageCallError, LanguageCompletionReason, LanguageIncompleteReason, LanguageInput,
    LanguageModel, LanguageRequest, LanguageResponse, LanguageStream, LanguageStreamEvent,
    LanguageTermination, Message, MessagePart, MessageRole, Model, ModelDescriptor, ModelFamily,
    ModelId, PartialLanguageOutput, Provider, ProviderId, RerankCall, RerankCandidate,
    RerankClient, RerankModel, RerankRequest, RerankResponse, RetryIntent, Siumai, SiumaiBuilder,
    SpeechCall, SpeechClient, SpeechModel, SpeechRequest, SpeechResponse, StreamTerminal,
    ToolChoice, ToolSpec, TranscriptionCall, TranscriptionClient, TranscriptionModel,
    TranscriptionRequest, TranscriptionResponse, Usage, UsageUpdate, UsageUpdateKind,
};
pub use crate::{EmbeddingClient, ImageClient, LanguageClient};
pub use crate::{embedding, image, language, rerank, speech, transcription};

#[cfg(feature = "runtime")]
pub use crate::{RunBudget, RunTimeouts, Runtime, StepOptions};

#[cfg(feature = "registry")]
pub use crate::registry::{
    ModelReference, RegisterProviderError, Registry, RegistryBuildError, RegistryBuilder,
    RegistryBuilderExt, RegistryModelContext, RegistryResolveError, RegistrySnapshot, RouteId,
};
