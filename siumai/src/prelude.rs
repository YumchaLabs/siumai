//! Common imports for the stable Siumai surface.

pub use crate::families::{embedding, image, language, rerank, speech, transcription};
pub use crate::{
    CallOptions, Cancellation, ContentPart, EmbeddingModel, EmbeddingRequest, EmbeddingResponse,
    Error, ImageModel, ImageRequest, ImageResponse, LanguageModel, LanguageRequest,
    LanguageResponse, LanguageStream, LanguageStreamEvent, Message, MessageRole, Model,
    ModelDescriptor, ModelFamily, ModelId, Provider, ProviderId, RerankCandidate, RerankModel,
    RerankRequest, RerankResponse, SpeechModel, SpeechRequest, SpeechResponse, StreamTerminal,
    ToolChoice, ToolSpec, TranscriptionModel, TranscriptionRequest, TranscriptionResponse, Usage,
};

#[cfg(feature = "registry")]
pub use crate::registry::{
    ModelReference, Registry, RegistryBuildError, RegistryBuilder, RegistryBuilderExt,
    RegistryMiddleware, RegistryModelContext, RegistryResolveError, RegistrySnapshot, RouteId,
};
