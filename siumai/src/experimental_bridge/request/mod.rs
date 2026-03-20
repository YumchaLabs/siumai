//! Request bridge implementation.

mod inspect;
pub mod pairs;
mod primitives;
mod serialize;

#[cfg(test)]
mod tests;

pub use inspect::inspect_chat_request_bridge;
pub use serialize::{bridge_chat_request_to_json, bridge_chat_request_to_json_with_options};

#[cfg(feature = "anthropic")]
pub use serialize::{
    bridge_chat_request_to_anthropic_messages_json,
    bridge_chat_request_to_anthropic_messages_json_with_options,
};
#[cfg(feature = "google")]
pub use serialize::{
    bridge_chat_request_to_gemini_generate_content_json,
    bridge_chat_request_to_gemini_generate_content_json_with_options,
};
#[cfg(feature = "openai")]
pub use serialize::{
    bridge_chat_request_to_openai_chat_completions_json,
    bridge_chat_request_to_openai_chat_completions_json_with_options,
    bridge_chat_request_to_openai_responses_json,
    bridge_chat_request_to_openai_responses_json_with_options,
};
