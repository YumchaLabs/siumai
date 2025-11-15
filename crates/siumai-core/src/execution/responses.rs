//! Core Responses API abstractions (provider-agnostic)
//!
//! This module defines minimal types and traits for modelling
//! OpenAI-style Responses API at the `siumai-core` layer. The goal
//! is to decouple request/response shaping for Responses API from
//! any particular aggregator or provider implementation.
//!
//! The initial version intentionally focuses on the request/response
//! shape and leaves streaming to the existing `ChatStreamEventCore`
//! model. Providers and standards crates can evolve these types
//! gradually without impacting the main chat abstractions.

use crate::error::LlmError;
use crate::types::FinishReasonCore;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Minimal core representation of a Responses API request.
///
/// This mirrors the top-level shape of OpenAI's Responses API:
/// - `model`: target model id
/// - `input`: array of JSON items (messages, tool calls, etc.)
/// - `extra`: provider/standard-specific fields (include/background/metadata/...)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResponsesInput {
    /// Target model identifier
    pub model: String,
    /// Input items (messages, tool-call outputs, etc.)
    ///
    /// Standards are responsible for constructing these from
    /// aggregator-level messages and tools.
    pub input: Vec<serde_json::Value>,
    /// Provider/standard-specific configuration (flattened).
    ///
    /// This is a generic extension point for flags such as:
    /// - `include`, `background`, `metadata`
    /// - `reasoning_effort`, `service_tier`
    /// - Any future Responses API toggles.
    #[serde(default)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Minimal usage information for Responses API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Minimal core representation of a Responses API result.
///
/// The concrete shape of `output` is standard/provider-specific; this
/// struct simply provides a container for the primary payload and
/// optional usage/metadata.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResponsesResult {
    /// Primary output payload (text/items/json), standard-defined.
    pub output: serde_json::Value,
    /// Optional token usage statistics.
    pub usage: Option<ResponsesUsage>,
    /// Optional normalized finish reason for the response.
    pub finish_reason: Option<FinishReasonCore>,
    /// Provider/standard-specific metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Request transformer for Responses API.
///
/// Standards crates implement this trait to map from a core
/// `ResponsesInput` into provider-native JSON bodies.
pub trait ResponsesRequestTransformer: Send + Sync {
    fn provider_id(&self) -> &str;
    fn transform_responses(&self, req: &ResponsesInput) -> Result<serde_json::Value, LlmError>;
}

/// Response transformer for Responses API.
///
/// Standards crates implement this trait to map provider-native JSON
/// back into the minimal `ResponsesResult` shape.
pub trait ResponsesResponseTransformer: Send + Sync {
    fn provider_id(&self) -> &str;
    fn transform_responses_response(
        &self,
        raw: &serde_json::Value,
    ) -> Result<ResponsesResult, LlmError>;
}
