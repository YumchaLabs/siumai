//! Gemini API types (split by concern)

mod config;
mod content;
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
    CodeExecutionResult,
    CodeLanguage,
    Content,
    CreateFileRequest,
    CreateFileResponse,
    DownloadFileResponse,
    DynamicRetrievalConfig,
    DynamicRetrievalMode,
    ExecutableCode,
    FileData,
    FinishReason,
    FunctionCall,
    FunctionDeclaration,
    FunctionResponse,
    // File management types
    GeminiFile,
    GeminiFileState,
    GeminiStatus,
    GeminiTool,
    GoogleSearch,
    GoogleSearchRetrieval,
    // Grounding types
    GroundingChunk,
    GroundingMetadata,
    GroundingSupport,
    HarmCategory,
    HarmProbability,
    ListFilesResponse,
    Part,
    RetrievalMetadata,
    RetrievedContextChunk,
    SafetyRating,
    SafetySetting,
    SearchEntryPoint,
    UrlContext,
    UrlContextMetadata,
    UrlMetadata,
    UrlRetrievalStatus,
    VideoFileMetadata,
    WebGroundingChunk,
};
pub use generation::{
    BlockReason, GenerateContentRequest, GenerateContentResponse, GenerationConfig, PromptFeedback,
    ThinkingConfig, UsageMetadata,
};
