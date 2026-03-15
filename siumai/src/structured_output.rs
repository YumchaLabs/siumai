//! Structured output helpers.
//!
//! This module provides provider-agnostic JSON extraction utilities for responses produced
//! with `ChatRequest.response_format`.

use siumai_core::error::LlmError;
use siumai_core::streaming::ChatStream;
use siumai_core::types::ChatResponse;

/// Extract a `serde_json::Value` from a model output string.
pub fn extract_json_value(text: &str) -> Result<serde_json::Value, LlmError> {
    siumai_core::structured_output::extract_json_value(text)
}

/// Extract a `serde_json::Value` from a unified chat response.
pub fn extract_json_value_from_response(
    response: &ChatResponse,
) -> Result<serde_json::Value, LlmError> {
    siumai_core::structured_output::extract_json_value_from_response(response)
}

/// Extract a `serde_json::Value` from a streaming response.
pub async fn extract_json_value_from_stream(
    stream: ChatStream,
) -> Result<serde_json::Value, LlmError> {
    siumai_core::structured_output::extract_json_value_from_stream(stream).await
}

/// Extract and deserialize structured output into a typed value.
pub fn extract_json<T: serde::de::DeserializeOwned>(text: &str) -> Result<T, LlmError> {
    siumai_core::structured_output::extract_json(text)
}

/// Extract and deserialize structured output from a stream into a typed value.
pub async fn extract_json_from_stream<T: serde::de::DeserializeOwned>(
    stream: ChatStream,
) -> Result<T, LlmError> {
    siumai_core::structured_output::extract_json_from_stream(stream).await
}

/// Extract and deserialize structured output from a response into a typed value.
pub fn extract_json_from_response<T: serde::de::DeserializeOwned>(
    response: &ChatResponse,
) -> Result<T, LlmError> {
    siumai_core::structured_output::extract_json_from_response(response)
}
