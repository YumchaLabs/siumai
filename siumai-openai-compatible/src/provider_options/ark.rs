//! Volcengine ARK / Doubao provider options.
//!
//! ARK's Chat and Responses endpoints are OpenAI-shaped but not fully OpenAI-compatible. These
//! typed options preserve ARK-only thinking, caching, and built-in-tool semantics without putting
//! them into the provider-agnostic core request types.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use siumai_core::{ModelFamily, ProviderOptionError, TypedProviderOptions};
use siumai_protocol_openai::chat_completions::API_MODE_ID as CHAT_API_MODE_ID;
use siumai_protocol_openai::responses::API_MODE_ID as RESPONSES_API_MODE_ID;

/// ARK Chat thinking mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ArkThinkingType {
    Enabled,
    Disabled,
    Auto,
}

/// ARK Chat thinking object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArkThinking {
    #[serde(rename = "type")]
    pub r#type: ArkThinkingType,
}

impl ArkThinking {
    pub const fn enabled() -> Self {
        Self {
            r#type: ArkThinkingType::Enabled,
        }
    }

    pub const fn disabled() -> Self {
        Self {
            r#type: ArkThinkingType::Disabled,
        }
    }

    pub const fn auto() -> Self {
        Self {
            r#type: ArkThinkingType::Auto,
        }
    }
}

/// Provider-owned ARK Chat options.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ArkChatOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ArkThinking>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,
}

impl ArkChatOptions {
    pub const fn new() -> Self {
        Self {
            thinking: None,
            parallel_tool_calls: None,
            service_tier: None,
            logprobs: None,
            top_logprobs: None,
        }
    }

    pub const fn with_thinking(mut self, value: ArkThinking) -> Self {
        self.thinking = Some(value);
        self
    }

    pub const fn with_parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.parallel_tool_calls = Some(enabled);
        self
    }

    pub fn with_service_tier(mut self, value: impl Into<String>) -> Self {
        self.service_tier = Some(value.into());
        self
    }

    pub const fn with_logprobs(mut self, enabled: bool) -> Self {
        self.logprobs = Some(enabled);
        self
    }

    pub const fn with_top_logprobs(mut self, value: u8) -> Self {
        self.top_logprobs = Some(value);
        self
    }
}

impl TypedProviderOptions for ArkChatOptions {
    const NAMESPACE: &'static str = "volcengine";
    const MODEL_FAMILY: ModelFamily = ModelFamily::Language;
    const API_MODE: Option<&'static str> = Some(CHAT_API_MODE_ID);

    fn validate(&self) -> Result<(), ProviderOptionError> {
        if self.top_logprobs.is_some_and(|value| value > 20) {
            return Err(ProviderOptionError::Rejected {
                path: "top_logprobs".to_string(),
                reason: "ARK top_logprobs must not exceed 20".to_string(),
            });
        }
        Ok(())
    }
}

/// ARK Responses caching declaration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ArkCaching {
    pub r#type: ArkCachingType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ArkCachingType {
    Enabled,
    Disabled,
}

impl ArkCaching {
    pub const fn enabled() -> Self {
        Self {
            r#type: ArkCachingType::Enabled,
            prefix: None,
        }
    }

    pub const fn disabled() -> Self {
        Self {
            r#type: ArkCachingType::Disabled,
            prefix: None,
        }
    }

    pub const fn with_prefix(mut self, enabled: bool) -> Self {
        self.prefix = Some(enabled);
        self
    }
}

/// ARK Responses built-in tools.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ArkResponsesTool {
    WebSearch {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        sources: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        max_keyword: Option<u8>,
        #[serde(skip_serializing_if = "Option::is_none")]
        limit: Option<u8>,
        #[serde(skip_serializing_if = "Option::is_none")]
        user_location: Option<Value>,
    },
    ImageProcess,
    KnowledgeSearch {
        knowledge_resource_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        limit: Option<u8>,
        #[serde(skip_serializing_if = "Option::is_none")]
        dense_weight: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        ranking_options: Option<Value>,
    },
    Mcp {
        server_label: String,
        server_url: String,
    },
}

impl ArkResponsesTool {
    pub fn web_search() -> Self {
        Self::WebSearch {
            sources: Vec::new(),
            max_keyword: None,
            limit: None,
            user_location: None,
        }
    }

    pub fn knowledge_search(resource_id: impl Into<String>) -> Self {
        Self::KnowledgeSearch {
            knowledge_resource_id: resource_id.into(),
            limit: None,
            dense_weight: None,
            ranking_options: None,
        }
    }

    pub fn mcp(label: impl Into<String>, url: impl Into<String>) -> Self {
        // Authenticated MCP headers are deliberately not modeled as request options.
        // This profile currently supports only MCP servers that need no custom credentials.
        Self::Mcp {
            server_label: label.into(),
            server_url: url.into(),
        }
    }

    pub fn as_value(&self) -> Result<Value, ProviderOptionError> {
        serde_json::to_value(self)
            .map_err(|error| ProviderOptionError::Serialization(error.to_string()))
    }

    fn validate(&self, index: usize) -> Result<(), ProviderOptionError> {
        match self {
            Self::WebSearch {
                max_keyword, limit, ..
            } => {
                if max_keyword.is_some_and(|value| !(1..=50).contains(&value))
                    || limit.is_some_and(|value| !(1..=50).contains(&value))
                {
                    return Err(ProviderOptionError::Rejected {
                        path: format!("native_tools[{index}]"),
                        reason: "ARK web_search limits are outside the documented range"
                            .to_string(),
                    });
                }
            }
            Self::KnowledgeSearch {
                knowledge_resource_id,
                dense_weight,
                ..
            } => {
                if knowledge_resource_id.trim().is_empty()
                    || dense_weight.is_some_and(|value| !(0.0..=1.0).contains(&value))
                {
                    return Err(ProviderOptionError::Rejected {
                        path: format!("native_tools[{index}]"),
                        reason: "ARK knowledge_search resource and dense_weight are invalid"
                            .to_string(),
                    });
                }
            }
            Self::Mcp {
                server_label,
                server_url,
            } => {
                if server_label.trim().is_empty() || server_url.trim().is_empty() {
                    return Err(ProviderOptionError::Rejected {
                        path: format!("native_tools[{index}]"),
                        reason: "ARK MCP requires a label and URL".to_string(),
                    });
                }
            }
            Self::ImageProcess => {}
        }
        Ok(())
    }
}

/// Provider-owned ARK Responses options.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ArkResponsesOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub caching: Option<ArkCaching>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expire_at: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tool_calls: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_management: Option<Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub native_tools: Vec<ArkResponsesTool>,
}

impl ArkResponsesOptions {
    pub const fn new() -> Self {
        Self {
            previous_response_id: None,
            store: None,
            caching: None,
            expire_at: None,
            max_tool_calls: None,
            context_management: None,
            native_tools: Vec::new(),
        }
    }

    pub fn with_previous_response_id(mut self, value: impl Into<String>) -> Self {
        self.previous_response_id = Some(value.into());
        self
    }

    pub const fn with_store(mut self, enabled: bool) -> Self {
        self.store = Some(enabled);
        self
    }

    pub fn with_caching(mut self, value: ArkCaching) -> Self {
        self.caching = Some(value);
        self
    }

    pub const fn with_expire_at(mut self, unix_timestamp_seconds: i64) -> Self {
        self.expire_at = Some(unix_timestamp_seconds);
        self
    }

    pub const fn with_max_tool_calls(mut self, maximum: i64) -> Self {
        self.max_tool_calls = Some(maximum);
        self
    }

    pub fn with_context_management(mut self, value: Value) -> Self {
        self.context_management = Some(value);
        self
    }

    pub fn with_native_tool(mut self, value: ArkResponsesTool) -> Self {
        self.native_tools.push(value);
        self
    }
}

impl TypedProviderOptions for ArkResponsesOptions {
    const NAMESPACE: &'static str = "volcengine";
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
        if self.expire_at.is_some_and(|value| value <= 0) {
            return Err(ProviderOptionError::Rejected {
                path: "expire_at".to_string(),
                reason: "ARK expiration timestamp must be greater than zero".to_string(),
            });
        }
        if self.max_tool_calls.is_some_and(|value| value <= 0) {
            return Err(ProviderOptionError::Rejected {
                path: "max_tool_calls".to_string(),
                reason: "ARK max_tool_calls must be greater than zero".to_string(),
            });
        }
        if let Some(caching) = &self.caching
            && caching.r#type == ArkCachingType::Disabled
            && caching.prefix.is_some()
        {
            return Err(ProviderOptionError::Rejected {
                path: "caching.prefix".to_string(),
                reason: "ARK disabled caching cannot declare a prefix".to_string(),
            });
        }
        for (index, tool) in self.native_tools.iter().enumerate() {
            tool.validate(index)?;
        }
        Ok(())
    }
}

pub const ARK_BETA_IMAGE_PROCESS_HEADER: &str = "ark-beta-image-process";
pub const ARK_BETA_KNOWLEDGE_SEARCH_HEADER: &str = "ark-beta-knowledge-search";
