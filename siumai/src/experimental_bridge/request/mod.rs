//! Request bridge implementation.

mod inspect;
pub mod pairs;
mod primitives;
mod serialize;

#[cfg(test)]
mod tests;

pub use inspect::inspect_chat_request_bridge;
pub use serialize::bridge_chat_request_to_json;

#[cfg(feature = "anthropic")]
pub use serialize::bridge_chat_request_to_anthropic_messages_json;
#[cfg(feature = "google")]
pub use serialize::bridge_chat_request_to_gemini_generate_content_json;
#[cfg(feature = "openai")]
pub use serialize::{
    bridge_chat_request_to_openai_chat_completions_json,
    bridge_chat_request_to_openai_responses_json,
};
