//! Curated xAI provider facade.

pub use siumai_provider_xai::{
    CredentialSourceError, DynamicCredentialSource, XaiConfigError, XaiCredential, XaiDeletedFile,
    XaiFile, XaiFileContent, XaiFileId, XaiFileList, XaiFileListOptions, XaiFileOrder,
    XaiFileUpload, XaiFiles, XaiImageModel, XaiLanguageApi, XaiLanguageModel, XaiProvider,
    XaiProviderBuilder, XaiSpeechModel, XaiTranscriptionModel, XaiVideoArtifact,
    XaiVideoAspectRatio, XaiVideoCreateRequest, XaiVideoImage, XaiVideoJob, XaiVideoJobId,
    XaiVideoJobState, XaiVideoJobs, XaiVideoResolution,
};

pub mod models {
    pub use siumai_provider_xai::providers::xai::models::{
        OFFICIAL_SOURCE, VERIFIED_ON, code, hints, image, language, recommended, speech,
        transcription, video,
    };
}

pub mod options {
    pub use siumai_provider_xai::{
        NewsSearchSource, RssSearchSource, SearchMode, SearchSource, WebSearchSource,
        XSearchSource, XaiChatOptions, XaiChatReasoningEffort, XaiReasoningSummary,
        XaiResponseInclude, XaiResponsesOptions, XaiResponsesReasoningEffort, XaiSearchParameters,
    };
}

pub mod tools {
    pub use siumai_provider_xai::tools::{
        XaiFileSearchTool, XaiMcpTool, XaiResponsesTool, XaiWebSearchTool, XaiXSearchTool,
        code_execution, file_search, mcp, view_image, view_x_video, web_search, web_search_with,
        x_search, x_search_with,
    };
}
