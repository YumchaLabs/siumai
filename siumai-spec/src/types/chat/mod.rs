//! Chat-related types and message handling

mod content;
mod message;
mod metadata;
mod request;
mod response;
mod response_format;
mod ui;

// Re-export all public types
pub use content::{
    ContentPart, FilePartSource, ImageDetail, MediaSource, MessageContent, ProviderReference,
    SourcePart, ToolResultContentPart, ToolResultFileId, ToolResultOutput,
};
pub use message::{ChatMessage, ChatMessageBuilder, MessageRole};
pub use metadata::{CacheControl, MessageMetadata, ToolCallInfo, ToolResultInfo};
pub use request::{ChatRequest, ChatRequestBuilder};
pub use response::{AudioOutput, ChatResponse};
pub use response_format::ResponseFormat;
pub use ui::{
    UiCustomPart, UiDataPart, UiFilePart, UiMessage, UiMessagePart, UiMessageRole, UiPartState,
    UiProviderMetadata, UiReasoningFilePart, UiReasoningPart, UiSourceDocumentPart,
    UiSourceUrlPart, UiTextPart, UiToolApproval, UiToolApprovalDecision, UiToolApprovalRequest,
    UiToolApprovedApproval, UiToolDeniedApproval, UiToolInvocation, UiToolInvocationState,
    UiToolKind, UiToolPart, UiToolPartState,
};
