//! OpenAI-compatible protocol transformers
//!
//! These transformers implement the OpenAI(-compatible) wire format mapping and
//! are intended to be reused by multiple providers.

pub mod request;
pub mod response;
pub mod stream;

pub use request::OpenAiRequestTransformer;
#[cfg(feature = "openai-responses")]
pub use request::OpenAiResponsesRequestTransformer;
pub use response::{OpenAiResponseTransformer, extract_thinking_from_multiple_fields};
#[cfg(feature = "openai-responses")]
pub use response::OpenAiResponsesResponseTransformer;
pub use stream::OpenAiStreamChunkTransformer;
#[cfg(feature = "openai-responses")]
pub use stream::OpenAiResponsesStreamChunkTransformer;
