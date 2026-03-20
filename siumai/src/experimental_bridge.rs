//! Experimental protocol bridge helpers.
//!
//! The current focus is explicit request bridging with reusable internal
//! structure:
//!
//! - planner
//! - request primitives
//! - request pair modules
//! - request inspection
//! - request serialization
//! - response inspection / serialization
//! - stream inspection / serialization

mod customize;
pub mod planner;
pub mod request;
pub mod response;
pub mod stream;

pub use planner::{RequestBridgePath, RequestBridgePlan, plan_chat_request_bridge};
pub use request::{
    bridge_chat_request_to_json, bridge_chat_request_to_json_with_options,
    inspect_chat_request_bridge,
};
pub use response::{
    bridge_chat_response_to_json_bytes, bridge_chat_response_to_json_bytes_with_options,
    bridge_chat_response_to_json_value, bridge_chat_response_to_json_value_with_options,
    inspect_chat_response_bridge,
};
pub use stream::{
    bridge_chat_stream_to_bytes, bridge_chat_stream_to_bytes_with_options,
    inspect_chat_stream_bridge, transform_chat_stream_with_bridge_options,
};

#[cfg(feature = "anthropic")]
pub use request::{
    bridge_anthropic_messages_json_to_chat_request, bridge_chat_request_to_anthropic_messages_json,
    bridge_chat_request_to_anthropic_messages_json_with_options,
};
#[cfg(feature = "google")]
pub use request::{
    bridge_chat_request_to_gemini_generate_content_json,
    bridge_chat_request_to_gemini_generate_content_json_with_options,
};
#[cfg(feature = "openai")]
pub use request::{
    bridge_chat_request_to_openai_chat_completions_json,
    bridge_chat_request_to_openai_chat_completions_json_with_options,
    bridge_chat_request_to_openai_responses_json,
    bridge_chat_request_to_openai_responses_json_with_options,
    bridge_openai_chat_completions_json_to_chat_request,
    bridge_openai_responses_json_to_chat_request,
};
#[cfg(feature = "anthropic")]
pub use response::{
    bridge_chat_response_to_anthropic_messages_json_bytes,
    bridge_chat_response_to_anthropic_messages_json_bytes_with_options,
    bridge_chat_response_to_anthropic_messages_json_value,
    bridge_chat_response_to_anthropic_messages_json_value_with_options,
};
#[cfg(feature = "google")]
pub use response::{
    bridge_chat_response_to_gemini_generate_content_json_bytes,
    bridge_chat_response_to_gemini_generate_content_json_bytes_with_options,
    bridge_chat_response_to_gemini_generate_content_json_value,
    bridge_chat_response_to_gemini_generate_content_json_value_with_options,
};
#[cfg(feature = "openai")]
pub use response::{
    bridge_chat_response_to_openai_chat_completions_json_bytes,
    bridge_chat_response_to_openai_chat_completions_json_bytes_with_options,
    bridge_chat_response_to_openai_chat_completions_json_value,
    bridge_chat_response_to_openai_chat_completions_json_value_with_options,
    bridge_chat_response_to_openai_responses_json_bytes,
    bridge_chat_response_to_openai_responses_json_bytes_with_options,
    bridge_chat_response_to_openai_responses_json_value,
    bridge_chat_response_to_openai_responses_json_value_with_options,
};
#[cfg(feature = "anthropic")]
pub use stream::{
    bridge_chat_stream_to_anthropic_messages_sse,
    bridge_chat_stream_to_anthropic_messages_sse_with_options,
};
#[cfg(feature = "google")]
pub use stream::{
    bridge_chat_stream_to_gemini_generate_content_sse,
    bridge_chat_stream_to_gemini_generate_content_sse_with_options,
};
#[cfg(feature = "openai")]
pub use stream::{
    bridge_chat_stream_to_openai_chat_completions_sse,
    bridge_chat_stream_to_openai_chat_completions_sse_with_options,
    bridge_chat_stream_to_openai_responses_sse,
    bridge_chat_stream_to_openai_responses_sse_with_options,
};
