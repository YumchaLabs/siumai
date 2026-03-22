//! Response bridge implementation.

mod inspect;
mod serialize;
mod target_caps;

#[cfg(test)]
mod tests;

pub use inspect::inspect_chat_response_bridge;
pub use serialize::{
    bridge_chat_response_to_json_bytes, bridge_chat_response_to_json_bytes_with_options,
    bridge_chat_response_to_json_value, bridge_chat_response_to_json_value_with_options,
};

#[cfg(feature = "anthropic")]
pub use serialize::{
    bridge_chat_response_to_anthropic_messages_json_bytes,
    bridge_chat_response_to_anthropic_messages_json_bytes_with_options,
    bridge_chat_response_to_anthropic_messages_json_value,
    bridge_chat_response_to_anthropic_messages_json_value_with_options,
};
#[cfg(any(feature = "google", feature = "google-vertex"))]
pub use serialize::{
    bridge_chat_response_to_gemini_generate_content_json_bytes,
    bridge_chat_response_to_gemini_generate_content_json_bytes_with_options,
    bridge_chat_response_to_gemini_generate_content_json_value,
    bridge_chat_response_to_gemini_generate_content_json_value_with_options,
};
#[cfg(feature = "openai")]
pub use serialize::{
    bridge_chat_response_to_openai_chat_completions_json_bytes,
    bridge_chat_response_to_openai_chat_completions_json_bytes_with_options,
    bridge_chat_response_to_openai_chat_completions_json_value,
    bridge_chat_response_to_openai_chat_completions_json_value_with_options,
    bridge_chat_response_to_openai_responses_json_bytes,
    bridge_chat_response_to_openai_responses_json_bytes_with_options,
    bridge_chat_response_to_openai_responses_json_value,
    bridge_chat_response_to_openai_responses_json_value_with_options,
};
