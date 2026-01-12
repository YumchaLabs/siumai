//! Anthropic Utility Functions
//!
//! Common utility functions for Anthropic Claude API interactions.

use super::server_tools;
use super::types::*;
use crate::error::LlmError;
use crate::execution::http::headers::HttpHeaderBuilder;
use crate::types::*;
use base64::Engine;
use reqwest::header::HeaderMap;

mod content;
mod errors;
mod finish;
mod headers;
mod messages;
mod models;
mod parse;
mod provider_metadata;
mod tool_choice;
mod tools;

pub use content::convert_message_content;
pub use errors::map_anthropic_error;
pub use finish::parse_finish_reason;
pub use headers::build_headers;
pub use messages::convert_messages;
pub use models::get_default_models;
pub use parse::{
    create_usage_from_response, extract_thinking_content, parse_response_content,
    parse_response_content_and_tools,
};
pub use provider_metadata::{
    map_container_provider_metadata, map_context_management_provider_metadata,
};
pub use tool_choice::convert_tool_choice;
pub use tools::convert_tools_to_anthropic_format;
