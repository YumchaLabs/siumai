//! Alibaba/Qwen options for the explicit configured provider runtime.
//!
//! These types intentionally separate Chat Completions fields from Responses fields. The two
//! endpoints share an OpenAI-shaped transport, but their provider tools, cache semantics, and
//! source/citation behavior are not interchangeable.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use siumai_anthropic_compatible::MessagesCallOptions;
use siumai_core::{ModelFamily, ProviderOptionError, ProviderOptions, TypedProviderOptions};
use siumai_protocol_anthropic::messages::{API_MODE_ID as MESSAGES_API_MODE_ID, ThinkingConfig};
use siumai_protocol_openai::chat_completions::API_MODE_ID as CHAT_API_MODE_ID;
use siumai_protocol_openai::responses::API_MODE_ID as RESPONSES_API_MODE_ID;

/// Thinking modes verified for Alibaba's Anthropic-compatible Messages API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
#[non_exhaustive]
pub enum AlibabaMessagesThinking {
    Disabled,
    Enabled { budget_tokens: u64 },
}

impl AlibabaMessagesThinking {
    pub const fn enabled(budget_tokens: u64) -> Self {
        Self::Enabled { budget_tokens }
    }

    const fn protocol(self) -> ThinkingConfig {
        match self {
            Self::Disabled => ThinkingConfig::Disabled,
            Self::Enabled { budget_tokens } => ThinkingConfig::enabled(budget_tokens),
        }
    }
}

/// Typed options for Alibaba's Anthropic-compatible Messages mode.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "snake_case")]
pub struct AlibabaMessagesOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<AlibabaMessagesThinking>,
}

impl AlibabaMessagesOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub const fn with_thinking(mut self, thinking: AlibabaMessagesThinking) -> Self {
        self.thinking = Some(thinking);
        self
    }

    pub const fn thinking(&self) -> Option<AlibabaMessagesThinking> {
        self.thinking
    }

    pub fn provider_options(&self) -> Result<ProviderOptions, ProviderOptionError> {
        ProviderOptions::typed(self)
    }

    pub(crate) fn to_engine(&self) -> MessagesCallOptions {
        match self.thinking {
            Some(thinking) => MessagesCallOptions::new().with_thinking(thinking.protocol()),
            None => MessagesCallOptions::new(),
        }
    }
}

impl TypedProviderOptions for AlibabaMessagesOptions {
    const NAMESPACE: &'static str = "alibaba";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
    const API_MODE: Option<&'static str> = Some(MESSAGES_API_MODE_ID);

    fn validate(&self) -> Result<(), ProviderOptionError> {
        if matches!(
            self.thinking,
            Some(AlibabaMessagesThinking::Enabled {
                budget_tokens: 0..=1_023
            })
        ) {
            return Err(ProviderOptionError::Rejected {
                path: "thinking.budget_tokens".to_string(),
                reason: "thinking budget must be at least 1024 tokens".to_string(),
            });
        }
        Ok(())
    }
}

/// One Chat Completions content block that should terminate an explicit prompt-cache prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AlibabaPromptCacheBreakpoint {
    pub message_index: usize,
    pub content_index: usize,
}

impl AlibabaPromptCacheBreakpoint {
    pub const fn new(message_index: usize, content_index: usize) -> Self {
        Self {
            message_index,
            content_index,
        }
    }
}

/// Qwen reasoning effort accepted by the Responses adapter.
///
/// Alibaba uses the nested wire shape `reasoning: { "effort": "..." }`.
/// The provider option keeps that detail out of the call site while retaining
/// the exact values currently documented by Alibaba.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AlibabaReasoningEffort {
    None,
    Minimal,
    Medium,
    High,
}

impl AlibabaReasoningEffort {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Minimal => "minimal",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

/// Alibaba Chat Completions web-search strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlibabaSearchStrategy {
    Agent,
}

/// Search controls for Alibaba's compatible Chat endpoint.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AlibabaSearchOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_strategy: Option<AlibabaSearchStrategy>,
}

impl AlibabaSearchOptions {
    pub const fn agent() -> Self {
        Self {
            search_strategy: Some(AlibabaSearchStrategy::Agent),
        }
    }
}

/// Provider-owned Chat Completions options for Qwen.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AlibabaChatOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_thinking: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_search: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_options: Option<AlibabaSearchOptions>,
    /// Explicit Alibaba prompt-cache breakpoints, limited to four per request.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prompt_cache_breakpoints: Vec<AlibabaPromptCacheBreakpoint>,
}

impl AlibabaChatOptions {
    pub const fn new() -> Self {
        Self {
            enable_thinking: None,
            thinking_budget: None,
            parallel_tool_calls: None,
            enable_search: None,
            search_options: None,
            prompt_cache_breakpoints: Vec::new(),
        }
    }

    pub const fn with_enable_thinking(mut self, enabled: bool) -> Self {
        self.enable_thinking = Some(enabled);
        self
    }

    pub const fn with_thinking_budget(mut self, budget: u32) -> Self {
        self.thinking_budget = Some(budget);
        self
    }

    pub const fn with_parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.parallel_tool_calls = Some(enabled);
        self
    }

    pub const fn with_enable_search(mut self, enabled: bool) -> Self {
        self.enable_search = Some(enabled);
        self
    }

    pub const fn with_search_options(mut self, options: AlibabaSearchOptions) -> Self {
        self.search_options = Some(options);
        self
    }

    pub fn with_prompt_cache_breakpoint(
        mut self,
        breakpoint: AlibabaPromptCacheBreakpoint,
    ) -> Self {
        self.prompt_cache_breakpoints.push(breakpoint);
        self
    }
}

impl TypedProviderOptions for AlibabaChatOptions {
    const NAMESPACE: &'static str = "alibaba";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
    const API_MODE: Option<&'static str> = Some(CHAT_API_MODE_ID);

    fn validate(&self) -> Result<(), ProviderOptionError> {
        if self.thinking_budget == Some(0) {
            return Err(ProviderOptionError::Rejected {
                path: "thinking_budget".to_string(),
                reason: "thinking budget must be greater than zero".to_string(),
            });
        }
        if self.search_options.is_some() && self.enable_search != Some(true) {
            return Err(ProviderOptionError::Rejected {
                path: "search_options".to_string(),
                reason: "search_options requires enable_search=true".to_string(),
            });
        }
        if self.prompt_cache_breakpoints.len() > 4 {
            return Err(ProviderOptionError::Rejected {
                path: "prompt_cache_breakpoints".to_string(),
                reason: "Alibaba accepts at most four explicit prompt-cache breakpoints"
                    .to_string(),
            });
        }
        Ok(())
    }
}

/// Alibaba Responses built-in tools.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AlibabaResponsesTool {
    WebSearch,
    WebExtractor,
    CodeInterpreter,
    WebSearchImage,
    ImageSearch,
    FileSearch {
        vector_store_ids: Vec<String>,
    },
    Mcp {
        server_protocol: String,
        server_label: String,
        server_url: String,
    },
}

impl AlibabaResponsesTool {
    pub fn web_search() -> Self {
        Self::WebSearch
    }

    pub fn file_search(vector_store_id: impl Into<String>) -> Self {
        Self::FileSearch {
            vector_store_ids: vec![vector_store_id.into()],
        }
    }

    pub fn mcp(
        server_protocol: impl Into<String>,
        server_label: impl Into<String>,
        server_url: impl Into<String>,
    ) -> Self {
        // Authenticated MCP headers are deliberately not modeled as request options.
        // This profile currently supports only MCP servers that need no custom credentials.
        Self::Mcp {
            server_protocol: server_protocol.into(),
            server_label: server_label.into(),
            server_url: server_url.into(),
        }
    }

    pub fn as_value(&self) -> Result<Value, ProviderOptionError> {
        serde_json::to_value(self)
            .map_err(|error| ProviderOptionError::Serialization(error.to_string()))
    }

    fn validate(&self, index: usize) -> Result<(), ProviderOptionError> {
        match self {
            Self::FileSearch { vector_store_ids } => {
                if vector_store_ids.len() != 1
                    || vector_store_ids
                        .first()
                        .is_none_or(|value| value.trim().is_empty())
                {
                    return Err(ProviderOptionError::Rejected {
                        path: format!("native_tools[{index}].vector_store_ids"),
                        reason:
                            "Alibaba file_search requires exactly one non-empty vector store id"
                                .to_string(),
                    });
                }
            }
            Self::Mcp {
                server_protocol,
                server_label,
                server_url,
            } => {
                if server_protocol.trim().is_empty()
                    || server_label.trim().is_empty()
                    || server_url.trim().is_empty()
                {
                    return Err(ProviderOptionError::Rejected {
                        path: format!("native_tools[{index}]"),
                        reason: "Alibaba MCP requires protocol, label, and URL".to_string(),
                    });
                }
            }
            Self::WebSearch
            | Self::WebExtractor
            | Self::CodeInterpreter
            | Self::WebSearchImage
            | Self::ImageSearch => {}
        }
        Ok(())
    }
}

/// Provider-owned Responses options for Qwen.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AlibabaResponsesOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<AlibabaReasoningEffort>,
    /// Enables the native `x-dashscope-session-cache: enable` wire header.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_cache: Option<bool>,
    /// Provider-native Responses tools, extracted by the configured codec policy.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub native_tools: Vec<AlibabaResponsesTool>,
}

impl AlibabaResponsesOptions {
    pub const fn new() -> Self {
        Self {
            previous_response_id: None,
            conversation: None,
            store: None,
            reasoning_effort: None,
            session_cache: None,
            native_tools: Vec::new(),
        }
    }

    pub fn with_previous_response_id(mut self, value: impl Into<String>) -> Self {
        self.previous_response_id = Some(value.into());
        self
    }

    pub fn with_conversation(mut self, value: Value) -> Self {
        self.conversation = Some(value);
        self
    }

    pub const fn with_store(mut self, enabled: bool) -> Self {
        self.store = Some(enabled);
        self
    }

    pub const fn with_reasoning_effort(mut self, value: AlibabaReasoningEffort) -> Self {
        self.reasoning_effort = Some(value);
        self
    }

    pub const fn with_session_cache(mut self, enabled: bool) -> Self {
        self.session_cache = Some(enabled);
        self
    }

    pub fn with_native_tool(mut self, tool: AlibabaResponsesTool) -> Self {
        self.native_tools.push(tool);
        self
    }
}

impl TypedProviderOptions for AlibabaResponsesOptions {
    const NAMESPACE: &'static str = "alibaba";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
    const API_MODE: Option<&'static str> = Some(RESPONSES_API_MODE_ID);

    fn validate(&self) -> Result<(), ProviderOptionError> {
        if self
            .previous_response_id
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(ProviderOptionError::Rejected {
                path: "previous_response_id".to_string(),
                reason: "response id must not be empty".to_string(),
            });
        }
        for (index, tool) in self.native_tools.iter().enumerate() {
            tool.validate(index)?;
        }
        Ok(())
    }
}

/// Explicit Responses-session cache wire header used by Alibaba Model Studio.
pub const ALIBABA_SESSION_CACHE_HEADER: &str = "x-dashscope-session-cache";
