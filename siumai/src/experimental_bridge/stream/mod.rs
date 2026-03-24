//! Stream bridge implementation.

mod inspect;
mod profile;
mod serialize;

#[cfg(all(test, any(feature = "openai", feature = "anthropic")))]
mod tests;

pub use inspect::inspect_chat_stream_bridge;
pub use serialize::{
    bridge_chat_stream_to_bytes, bridge_chat_stream_to_bytes_with_options,
    transform_chat_stream_with_bridge_options,
};

#[cfg(feature = "anthropic")]
pub use serialize::{
    bridge_chat_stream_to_anthropic_messages_sse,
    bridge_chat_stream_to_anthropic_messages_sse_with_options,
};
#[cfg(any(feature = "google", feature = "google-vertex"))]
pub use serialize::{
    bridge_chat_stream_to_gemini_generate_content_sse,
    bridge_chat_stream_to_gemini_generate_content_sse_with_options,
};
#[cfg(feature = "openai")]
pub use serialize::{
    bridge_chat_stream_to_openai_chat_completions_sse,
    bridge_chat_stream_to_openai_chat_completions_sse_with_options,
    bridge_chat_stream_to_openai_responses_sse,
    bridge_chat_stream_to_openai_responses_sse_with_options,
};
