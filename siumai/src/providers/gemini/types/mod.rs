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
    HarmCategory,
    HarmProbability,
    ListFilesResponse,
    Part,
    SafetyRating,
    SafetySetting,
    VideoFileMetadata,
};
pub use generation::{
    BlockReason, GenerateContentRequest, GenerateContentResponse, GenerationConfig, PromptFeedback,
    ThinkingConfig, UsageMetadata,
};
