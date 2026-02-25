//! Responses API configuration types

use serde::{Deserialize, Serialize};

use super::enums::{PromptCacheRetention, TextVerbosity, Truncation};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesLogprobs {
    Bool(bool),
    Top(u32),
}

/// Responses API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesApiConfig {
    /// Whether to use Responses API endpoint
    pub enabled: bool,
    /// The ID of the OpenAI conversation to continue.
    pub conversation: Option<String>,
    /// Previous response ID for continuation
    pub previous_response_id: Option<String>,
    /// Whether to generate output. Set to `false` for connection warm-up in WebSocket mode.
    pub generate: Option<bool>,
    /// Prompt cache key (Responses API)
    pub prompt_cache_key: Option<String>,
    /// Prompt cache retention policy (Responses API).
    pub prompt_cache_retention: Option<PromptCacheRetention>,
    /// Response format schema
    pub response_format: Option<serde_json::Value>,
    /// Whether to use strict JSON schema validation for `responseFormat` mappings.
    pub strict_json_schema: Option<bool>,
    /// Whether to run the model response in the background
    pub background: Option<bool>,
    /// Specify additional output data to include in the model response
    pub include: Option<Vec<String>>,
    /// Inserts a system (or developer) message as the first item in the model's context
    pub instructions: Option<String>,
    /// The maximum number of total calls to built-in tools that can be processed in a response
    pub max_tool_calls: Option<u32>,
    /// Logprobs request option (Vercel-aligned); mapped into `top_logprobs` + `include`.
    pub logprobs: Option<ResponsesLogprobs>,
    /// Reasoning summary mode (Responses API; reasoning models only).
    pub reasoning_summary: Option<String>,
    /// The identifier for safety monitoring and tracking.
    pub safety_identifier: Option<String>,
    /// Whether to store the generated model response for later retrieval via API
    pub store: Option<bool>,
    /// The truncation strategy to use for the model response
    pub truncation: Option<Truncation>,
    /// Text verbosity level for the response
    pub text_verbosity: Option<TextVerbosity>,
    /// Metadata to store with the generation.
    pub metadata: Option<serde_json::Value>,
    /// Whether to allow the model to run tool calls in parallel
    pub parallel_tool_calls: Option<bool>,
    /// A unique identifier representing your end-user.
    pub user: Option<String>,
}

impl Default for ResponsesApiConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            conversation: None,
            previous_response_id: None,
            generate: None,
            prompt_cache_key: None,
            prompt_cache_retention: None,
            response_format: None,
            strict_json_schema: None,
            background: None,
            include: None,
            instructions: None,
            max_tool_calls: None,
            logprobs: None,
            reasoning_summary: None,
            safety_identifier: None,
            store: None,
            truncation: None,
            text_verbosity: None,
            metadata: None,
            parallel_tool_calls: None,
            user: None,
        }
    }
}

impl ResponsesApiConfig {
    /// Create new Responses API config (enabled by default)
    pub fn new() -> Self {
        Self::default()
    }

    /// Set conversation ID for continuation
    pub fn with_conversation(mut self, conversation: impl Into<String>) -> Self {
        self.conversation = Some(conversation.into());
        self
    }

    /// Set previous response ID for continuation
    pub fn with_previous_response(mut self, response_id: String) -> Self {
        self.previous_response_id = Some(response_id);
        self
    }

    /// Set whether to generate output. Use `false` to warm up WebSocket connections.
    pub fn with_generate(mut self, generate: bool) -> Self {
        self.generate = Some(generate);
        self
    }

    /// Set prompt cache key
    pub fn with_prompt_cache_key(mut self, key: impl Into<String>) -> Self {
        self.prompt_cache_key = Some(key.into());
        self
    }

    /// Set prompt cache retention policy
    pub fn with_prompt_cache_retention(mut self, retention: PromptCacheRetention) -> Self {
        self.prompt_cache_retention = Some(retention);
        self
    }

    /// Set response format schema
    pub fn with_response_format(mut self, format: serde_json::Value) -> Self {
        self.response_format = Some(format);
        self
    }

    /// Set strict JSON schema behavior
    pub fn with_strict_json_schema(mut self, strict: bool) -> Self {
        self.strict_json_schema = Some(strict);
        self
    }

    /// Set whether to run the model response in the background
    pub fn with_background(mut self, background: bool) -> Self {
        self.background = Some(background);
        self
    }

    /// Set additional output data to include in the model response
    pub fn with_include(mut self, include: Vec<String>) -> Self {
        self.include = Some(include);
        self
    }

    /// Set system/developer instructions
    pub fn with_instructions(mut self, instructions: String) -> Self {
        self.instructions = Some(instructions);
        self
    }

    /// Set maximum number of tool calls
    pub fn with_max_tool_calls(mut self, max_tool_calls: u32) -> Self {
        self.max_tool_calls = Some(max_tool_calls);
        self
    }

    /// Enable logprobs (Vercel-aligned)
    pub fn with_logprobs(mut self, logprobs: ResponsesLogprobs) -> Self {
        self.logprobs = Some(logprobs);
        self
    }

    /// Set reasoning summary mode
    pub fn with_reasoning_summary(mut self, summary: impl Into<String>) -> Self {
        self.reasoning_summary = Some(summary.into());
        self
    }

    /// Set safety identifier
    pub fn with_safety_identifier(mut self, id: impl Into<String>) -> Self {
        self.safety_identifier = Some(id.into());
        self
    }

    /// Set whether to store the response
    pub fn with_store(mut self, store: bool) -> Self {
        self.store = Some(store);
        self
    }

    /// Set truncation strategy
    pub fn with_truncation(mut self, truncation: Truncation) -> Self {
        self.truncation = Some(truncation);
        self
    }

    /// Set text verbosity level
    pub fn with_text_verbosity(mut self, verbosity: TextVerbosity) -> Self {
        self.text_verbosity = Some(verbosity);
        self
    }

    /// Set metadata
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Set parallel tool calls
    pub fn with_parallel_tool_calls(mut self, parallel: bool) -> Self {
        self.parallel_tool_calls = Some(parallel);
        self
    }

    /// Set end-user identifier
    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }
}
