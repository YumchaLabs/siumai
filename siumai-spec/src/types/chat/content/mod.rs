//! Content types for chat messages

mod media;
mod message_content;
mod part;
mod tool_result;

pub use media::{FilePartSource, ImageDetail, MediaSource, ProviderReference};
pub use message_content::MessageContent;
pub use part::{ContentPart, SourcePart};
pub use tool_result::{ToolResultContentPart, ToolResultFileId, ToolResultOutput};

/// Explicit compatibility namespace for the legacy chat content carrier.
///
/// `ContentPart` and `MessageContent` are still the serde-facing payloads used by legacy
/// `ChatMessage` / `ChatResponse`, but new request code should prefer `ModelMessage` prompt parts
/// and new response code should prefer generated-output content parts. Import from this module when
/// you intentionally need the legacy compatibility carrier.
pub mod compat {
    pub use super::{
        ContentPart, FilePartSource, ImageDetail, MediaSource, MessageContent, ProviderReference,
        SourcePart, ToolResultContentPart, ToolResultFileId, ToolResultOutput,
    };
}
