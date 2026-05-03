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
pub(crate) use finish::{raw_anthropic_stop_reason, replay_anthropic_stop_reason};
pub use headers::build_headers;
pub use messages::convert_messages;
pub use models::get_default_models;
pub(crate) use parse::{
    ParsedAnthropicResponseContent, parse_response_content_and_tools_with_context_and_params,
};
pub use parse::{
    create_usage_from_json_value, create_usage_from_response, extract_thinking_content,
    parse_response_content, parse_response_content_and_tools,
};
pub use provider_metadata::{
    map_container_provider_metadata, map_context_management_provider_metadata,
    map_usage_iterations_provider_metadata, raw_container_from_provider_metadata,
    raw_context_management_from_provider_metadata,
};
pub use tool_choice::convert_tool_choice;
pub use tools::convert_tools_to_anthropic_format;

pub(crate) fn resolve_anthropic_provider_reference(
    provider_reference: &ProviderReference,
) -> Result<&str, LlmError> {
    provider_reference.get("anthropic").ok_or_else(|| {
        let available = provider_reference.available_providers();
        let available = if available.is_empty() {
            "none".to_string()
        } else {
            available.join(", ")
        };
        LlmError::InvalidParameter(format!(
            "No provider reference found for provider 'anthropic'. Available providers: {available}"
        ))
    })
}
