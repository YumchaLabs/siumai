//! Gemini API types (protocol layer; split by concern)

mod config;
mod content;
mod file_search;
mod generation;

pub use config::{
    FunctionCallingConfig, FunctionCallingMode, GeminiConfig, GeminiEmbeddingOptions,
    GeminiEmbeddingRequestExt, ToolConfig,
};
pub use content::{
    Blob,
    Candidate,
    CitationMetadata,
    CitationSource,
    CodeExecution,
    CodeExecutionOutcome,
    CodeExecutionResult,
    CodeLanguage,
    Content,
    CreateFileRequest,
    CreateFileResponse,
    DownloadFileResponse,
    DynamicRetrievalConfig,
    DynamicRetrievalMode,
    EnterpriseWebSearch,
    ExecutableCode,
    FileData,
    FileSearch,
    FinishReason,
    FunctionCall,
    FunctionDeclaration,
    FunctionResponse,
    // File management types
    GeminiFile,
    GeminiFileState,
    GeminiStatus,
    GeminiTool,
    GoogleMaps,
    GoogleSearch,
    GoogleSearchRetrieval,
    // Grounding types
    GroundingChunk,
    GroundingMetadata,
    GroundingSupport,
    HarmCategory,
    HarmProbability,
    ListFilesResponse,
    MapsGroundingChunk,
    Part,
    Retrieval,
    RetrievalMetadata,
    RetrievedContextChunk,
    SafetyRating,
    SafetySetting,
    SearchEntryPoint,
    UrlContext,
    UrlContextMetadata,
    UrlMetadata,
    UrlRetrievalStatus,
    VertexRagResources,
    VertexRagStore,
    VideoFileMetadata,
    WebGroundingChunk,
};
pub use generation::{
    BlockReason, GenerateContentRequest, GenerateContentResponse, GenerationConfig, PromptFeedback,
    ThinkingConfig, UsageMetadata,
};

// Provider-specific File Search types
pub use file_search::{
    ChunkingConfig, FileSearchOperation, FileSearchStore, FileSearchStoresList,
    FileSearchUploadConfig, WhiteSpaceChunkingConfig,
};
