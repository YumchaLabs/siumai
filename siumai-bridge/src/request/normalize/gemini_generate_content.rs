//! Compatibility shim for Gemini GenerateContent request normalization.
//!
//! FCAB-110 moved the protocol-specific JSON -> `ChatRequest` parser into
//! `siumai-protocol-gemini::standards::gemini::request_bridge`. The bridge crate keeps public
//! wrappers, reports, loss policy, hooks, and dispatch here.

use serde_json::Value;
use siumai_core::LlmError;
use siumai_core::types::ChatRequest;

pub(super) fn parse_json_to_chat_request(value: &Value) -> Result<ChatRequest, LlmError> {
    siumai_protocol_gemini::standards::gemini::request_bridge::parse_json_to_chat_request(value)
}
