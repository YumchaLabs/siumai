//! Stream bridge implementation.

mod inspect;
mod serialize;

#[cfg(test)]
mod tests;

pub use inspect::inspect_chat_stream_bridge;
pub use serialize::bridge_chat_stream_to_bytes;

#[cfg(feature = "anthropic")]
pub use serialize::bridge_chat_stream_to_anthropic_messages_sse;
#[cfg(feature = "google")]
pub use serialize::bridge_chat_stream_to_gemini_generate_content_sse;
#[cfg(feature = "openai")]
pub use serialize::{
    bridge_chat_stream_to_openai_chat_completions_sse, bridge_chat_stream_to_openai_responses_sse,
};
