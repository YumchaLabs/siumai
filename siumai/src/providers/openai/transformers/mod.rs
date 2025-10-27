//! OpenAI native transformers module (split by concern)

pub mod audio;
pub mod files;
pub mod request;
pub mod response;
pub mod stream;

// Re-export public types to preserve existing import paths
pub use audio::OpenAiAudioTransformer;
pub use files::OpenAiFilesTransformer;
pub use request::{OpenAiRequestTransformer, OpenAiResponsesRequestTransformer};
pub use response::{
    OpenAiResponseTransformer, OpenAiResponsesResponseTransformer,
    extract_thinking_from_multiple_fields,
};
pub use stream::{OpenAiResponsesStreamChunkTransformer, OpenAiStreamChunkTransformer};
