//! OpenAI-specific Provider Options
//!
//! This module contains types for OpenAI-specific features including:
//! - Responses API configuration
//! - Built-in tools (web search, file search, computer use)
//! - Reasoning effort settings
//! - Service tier preferences

use serde::{Deserialize, Serialize};

// Re-export OpenAiBuiltInTool from tools module to avoid duplication
pub use crate::types::tools::OpenAiBuiltInTool;

/// OpenAI-specific options
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpenAiOptions {
    /// Responses API configuration
    pub responses_api: Option<ResponsesApiConfig>,
    /// Built-in tools (web search, file search, computer use)
    pub built_in_tools: Vec<OpenAiBuiltInTool>,
    /// Reasoning effort (for o1/o3 models)
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Service tier preference
    pub service_tier: Option<ServiceTier>,
}

impl OpenAiOptions {
    /// Create new OpenAI options
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable Responses API with configuration
    pub fn with_responses_api(mut self, config: ResponsesApiConfig) -> Self {
        self.responses_api = Some(config);
        self
    }

    /// Add a built-in tool
    pub fn with_built_in_tool(mut self, tool: OpenAiBuiltInTool) -> Self {
        self.built_in_tools.push(tool);
        self
    }

    /// Enable web search (shorthand for built-in tool)
    pub fn with_web_search(mut self) -> Self {
        self.built_in_tools.push(OpenAiBuiltInTool::WebSearch);
        self
    }

    /// Enable file search with vector store IDs
    pub fn with_file_search(mut self, vector_store_ids: Vec<String>) -> Self {
        self.built_in_tools.push(OpenAiBuiltInTool::FileSearch {
            vector_store_ids: Some(vector_store_ids),
        });
        self
    }

    /// Enable computer use with display settings
    pub fn with_computer_use(
        mut self,
        display_width: u32,
        display_height: u32,
        environment: String,
    ) -> Self {
        self.built_in_tools.push(OpenAiBuiltInTool::ComputerUse {
            display_width,
            display_height,
            environment,
        });
        self
    }

    /// Set reasoning effort
    pub fn with_reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }

    /// Set service tier
    pub fn with_service_tier(mut self, tier: ServiceTier) -> Self {
        self.service_tier = Some(tier);
        self
    }
}

/// Responses API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesApiConfig {
    /// Whether to use Responses API endpoint
    pub enabled: bool,
    /// Previous response ID for continuation
    pub previous_response_id: Option<String>,
    /// Response format schema
    pub response_format: Option<serde_json::Value>,
    /// Whether to run the model response in the background
    pub background: Option<bool>,
    /// Specify additional output data to include in the model response
    /// Supported values:
    /// - `file_search_call.results`: Include the search results of the file search tool call
    /// - `message.input_image.image_url`: Include image URLs from the input message
    /// - `computer_call_output.output.image_url`: Include image URLs from the computer call output
    /// - `reasoning.encrypted_content`: Include an encrypted version of reasoning tokens
    pub include: Option<Vec<String>>,
    /// Inserts a system (or developer) message as the first item in the model's context
    pub instructions: Option<String>,
    /// The maximum number of total calls to built-in tools that can be processed in a response
    pub max_tool_calls: Option<u32>,
    /// Whether to store the generated model response for later retrieval via API
    pub store: Option<bool>,
    /// The truncation strategy to use for the model response
    pub truncation: Option<Truncation>,
    /// Text verbosity level for the response
    pub text_verbosity: Option<TextVerbosity>,
    /// Set of key-value pairs that can be attached to an object
    pub metadata: Option<std::collections::HashMap<String, String>>,
    /// Whether to allow the model to run tool calls in parallel
    pub parallel_tool_calls: Option<bool>,
}

impl Default for ResponsesApiConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            previous_response_id: None,
            response_format: None,
            background: None,
            include: None,
            instructions: None,
            max_tool_calls: None,
            store: None,
            truncation: None,
            text_verbosity: None,
            metadata: None,
            parallel_tool_calls: None,
        }
    }
}

impl ResponsesApiConfig {
    /// Create new Responses API config (enabled by default)
    pub fn new() -> Self {
        Self::default()
    }

    /// Set previous response ID for continuation
    pub fn with_previous_response(mut self, response_id: String) -> Self {
        self.previous_response_id = Some(response_id);
        self
    }

    /// Set response format schema
    pub fn with_response_format(mut self, format: serde_json::Value) -> Self {
        self.response_format = Some(format);
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
    pub fn with_metadata(mut self, metadata: std::collections::HashMap<String, String>) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Set whether to allow parallel tool calls
    pub fn with_parallel_tool_calls(mut self, parallel: bool) -> Self {
        self.parallel_tool_calls = Some(parallel);
        self
    }
}

/// Truncation strategy for Responses API
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Truncation {
    /// Drop items in the middle to fit context window
    Auto,
    /// Error if exceeding context window
    Disabled,
}

/// Text verbosity level for Responses API
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TextVerbosity {
    Low,
    Medium,
    High,
}

/// Reasoning effort for o1/o3 models
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}

/// Service tier preference
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ServiceTier {
    Auto,
    Default,
}
