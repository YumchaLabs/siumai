//! OpenAI-compatible Chat Completions wire codec for the canonical core.

mod dialect;
mod reasoning;
mod request;
mod response;
mod sse;
mod stream;
mod wire;

pub use dialect::{ChatCompletionsDialect, DialectError, MaxOutputTokensField, WireFieldName};
pub use reasoning::REASONING_DETAILS_OPAQUE_KIND;
pub use request::{
    CHAT_COMPLETIONS_TARGET, ChatPromptCacheBlock, ChatRequestEncodingOptions, encode_request,
    encode_request_with_options, is_protected_option_field,
};
pub use response::decode_response;
pub use sse::ChatCompletionsSseEncoder;
pub use stream::ChatCompletionsStreamDecoder;

/// Stable protocol identity used by configured-provider scopes.
pub const PROTOCOL_ID: &str = "openai";

/// Stable API-mode identity for the Chat Completions operation.
pub const API_MODE_ID: &str = "chat-completions";
