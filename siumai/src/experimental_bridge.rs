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

pub mod planner;
pub mod request;

pub use planner::{RequestBridgePath, RequestBridgePlan, plan_chat_request_bridge};
pub use request::{bridge_chat_request_to_json, inspect_chat_request_bridge};

#[cfg(feature = "anthropic")]
pub use request::bridge_chat_request_to_anthropic_messages_json;
#[cfg(feature = "google")]
pub use request::bridge_chat_request_to_gemini_generate_content_json;
#[cfg(feature = "openai")]
pub use request::{
    bridge_chat_request_to_openai_chat_completions_json,
    bridge_chat_request_to_openai_responses_json,
};
