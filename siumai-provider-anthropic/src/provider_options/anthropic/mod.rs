//! Anthropic provider options.
//!
//! These types are carried via the open `providerOptions` JSON map (`provider_id = "anthropic"`),
//! and should be carried via `providerOptions["anthropic"]`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicEffort {
    Low,
    Medium,
    High,
    Max,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicSpeed {
    Fast,
    Standard,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicRequestCacheControlType {
    Ephemeral,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AnthropicRequestCacheControlTtl {
    #[serde(rename = "5m")]
    FiveMinutes,
    #[serde(rename = "1h")]
    OneHour,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnthropicRequestCacheControl {
    #[serde(rename = "type")]
    pub kind: AnthropicRequestCacheControlType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<AnthropicRequestCacheControlTtl>,
}

impl AnthropicRequestCacheControl {
    pub fn ephemeral() -> Self {
        Self {
            kind: AnthropicRequestCacheControlType::Ephemeral,
            ttl: None,
        }
    }

    pub fn with_ttl(mut self, ttl: AnthropicRequestCacheControlTtl) -> Self {
        self.ttl = Some(ttl);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AnthropicStructuredOutputMode {
    Auto,
    OutputFormat,
    JsonTool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicThinkingConfig {
    Adaptive,
    Enabled {
        #[serde(skip_serializing_if = "Option::is_none", alias = "budgetTokens")]
        budget_tokens: Option<u32>,
    },
    Disabled,
}

impl AnthropicThinkingConfig {
    pub fn adaptive() -> Self {
        Self::Adaptive
    }

    pub fn enabled(budget_tokens: Option<u32>) -> Self {
        Self::Enabled { budget_tokens }
    }

    pub fn disabled() -> Self {
        Self::Disabled
    }
}

/// Anthropic-specific options
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnthropicOptions {
    /// Prompt caching configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_caching: Option<PromptCachingConfig>,
    /// Thinking mode (legacy typed surface)
    #[serde(skip_serializing_if = "Option::is_none", alias = "thinkingMode")]
    pub thinking_mode: Option<ThinkingModeConfig>,
    /// Thinking configuration aligned with AI SDK `thinking`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<AnthropicThinkingConfig>,
    /// Whether prior-turn reasoning blocks should be forwarded.
    #[serde(skip_serializing_if = "Option::is_none", alias = "sendReasoning")]
    pub send_reasoning: Option<bool>,
    /// Structured output configuration (JSON object or JSON schema)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<AnthropicResponseFormat>,
    /// Structured output routing strategy (`outputFormat` vs reserved `json` tool fallback).
    #[serde(
        skip_serializing_if = "Option::is_none",
        alias = "structuredOutputMode"
    )]
    pub structured_output_mode: Option<AnthropicStructuredOutputMode>,
    /// Whether to force at most one tool call per response.
    #[serde(
        skip_serializing_if = "Option::is_none",
        alias = "disableParallelToolUse"
    )]
    pub disable_parallel_tool_use: Option<bool>,
    /// Request-level cache control.
    #[serde(skip_serializing_if = "Option::is_none", alias = "cacheControl")]
    pub cache_control: Option<AnthropicRequestCacheControl>,
    /// Request-level metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<AnthropicRequestMetadata>,
    /// MCP servers to include in the request.
    #[serde(skip_serializing_if = "Option::is_none", alias = "mcpServers")]
    pub mcp_servers: Option<Vec<AnthropicMcpServer>>,
    /// Container configuration (agent skills, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container: Option<AnthropicContainerConfig>,
    /// Context management configuration (Vercel `contextManagement` -> API `context_management`).
    #[serde(skip_serializing_if = "Option::is_none", alias = "contextManagement")]
    pub context_management: Option<AnthropicContextManagementConfig>,
    /// Fine-grained tool streaming toggle (Vercel `toolStreaming`).
    ///
    /// Vercel default is `true` when streaming; when enabled we add the
    /// `fine-grained-tool-streaming-2025-05-14` beta header token.
    #[serde(skip_serializing_if = "Option::is_none", alias = "toolStreaming")]
    pub tool_streaming: Option<bool>,
    /// Output effort (Vercel `effort`), sent as `output_config: { effort }`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<AnthropicEffort>,
    /// Fast/standard generation mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<AnthropicSpeed>,
    /// Additional beta headers requested at the provider-options layer.
    #[serde(skip_serializing_if = "Option::is_none", alias = "anthropicBeta")]
    pub anthropic_beta: Option<Vec<String>>,
}

impl AnthropicOptions {
    /// Create new Anthropic options
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable prompt caching
    pub fn with_prompt_caching(mut self, config: PromptCachingConfig) -> Self {
        self.prompt_caching = Some(config);
        self
    }

    /// Enable legacy thinking mode
    pub fn with_thinking_mode(mut self, config: ThinkingModeConfig) -> Self {
        self.thinking_mode = Some(config);
        self
    }

    /// Configure AI SDK-style thinking behavior.
    pub fn with_thinking(mut self, config: AnthropicThinkingConfig) -> Self {
        self.thinking = Some(config);
        self
    }

    /// Configure whether prior-turn reasoning blocks are forwarded.
    pub fn with_send_reasoning(mut self, send_reasoning: bool) -> Self {
        self.send_reasoning = Some(send_reasoning);
        self
    }

    /// Configure structured output as a plain JSON object
    pub fn with_json_object(mut self) -> Self {
        self.response_format = Some(AnthropicResponseFormat::JsonObject);
        self
    }

    /// Configure structured output using a JSON schema
    pub fn with_json_schema(
        mut self,
        name: impl Into<String>,
        schema: serde_json::Value,
        strict: bool,
    ) -> Self {
        self.response_format = Some(AnthropicResponseFormat::JsonSchema {
            name: name.into(),
            schema,
            strict,
        });
        self
    }

    /// Configure the structured output routing mode.
    pub fn with_structured_output_mode(mut self, mode: AnthropicStructuredOutputMode) -> Self {
        self.structured_output_mode = Some(mode);
        self
    }

    /// Configure whether parallel tool use is disabled.
    pub fn with_disable_parallel_tool_use(mut self, disabled: bool) -> Self {
        self.disable_parallel_tool_use = Some(disabled);
        self
    }

    /// Configure request-level cache control.
    pub fn with_cache_control(mut self, cache_control: AnthropicRequestCacheControl) -> Self {
        self.cache_control = Some(cache_control);
        self
    }

    /// Configure request-level metadata.
    pub fn with_metadata(mut self, metadata: AnthropicRequestMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Configure MCP servers.
    pub fn with_mcp_servers(mut self, mcp_servers: Vec<AnthropicMcpServer>) -> Self {
        self.mcp_servers = Some(mcp_servers);
        self
    }

    /// Configure container features (e.g., agent skills).
    pub fn with_container(mut self, container: AnthropicContainerConfig) -> Self {
        self.container = Some(container);
        self
    }

    pub fn with_context_management(
        mut self,
        context_management: AnthropicContextManagementConfig,
    ) -> Self {
        self.context_management = Some(context_management);
        self
    }

    pub fn with_tool_streaming(mut self, enabled: bool) -> Self {
        self.tool_streaming = Some(enabled);
        self
    }

    pub fn with_effort(mut self, effort: AnthropicEffort) -> Self {
        self.effort = Some(effort);
        self
    }

    pub fn with_speed(mut self, speed: AnthropicSpeed) -> Self {
        self.speed = Some(speed);
        self
    }

    pub fn with_anthropic_betas(mut self, betas: Vec<String>) -> Self {
        self.anthropic_beta = Some(betas);
        self
    }

    pub fn with_anthropic_beta(mut self, beta: impl Into<String>) -> Self {
        self.anthropic_beta
            .get_or_insert_with(Vec::new)
            .push(beta.into());
        self
    }
}

/// AI SDK-style alias for Anthropic language-model options.
pub type AnthropicLanguageModelOptions = AnthropicOptions;

/// Deprecated AI SDK compatibility alias for Anthropic language-model options.
#[deprecated(note = "Use `AnthropicLanguageModelOptions` instead.")]
pub type AnthropicProviderOptions = AnthropicLanguageModelOptions;

/// Programmatic callers allowed to invoke an Anthropic function tool.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AnthropicToolAllowedCaller {
    Direct,
    #[serde(rename = "code_execution_20250825")]
    CodeExecution20250825,
    #[serde(rename = "code_execution_20260120")]
    CodeExecution20260120,
}

/// Anthropic function-tool provider options carried via `providerOptions.anthropic`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnthropicToolOptions {
    /// Delay loading tool definitions until the provider decides they are needed.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "deferLoading",
        alias = "defer_loading"
    )]
    pub defer_loading: Option<bool>,

    /// Restrict which callers may invoke the tool programmatically.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "allowedCallers",
        alias = "allowed_callers"
    )]
    pub allowed_callers: Option<Vec<AnthropicToolAllowedCaller>>,

    /// Enable eager tool-input streaming for supported Anthropic function tools.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "eagerInputStreaming",
        alias = "eager_input_streaming"
    )]
    pub eager_input_streaming: Option<bool>,
}

impl AnthropicToolOptions {
    /// Create empty Anthropic function-tool options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure `deferLoading`.
    pub fn with_defer_loading(mut self, defer_loading: bool) -> Self {
        self.defer_loading = Some(defer_loading);
        self
    }

    /// Configure `allowedCallers`.
    pub fn with_allowed_callers<I>(mut self, allowed_callers: I) -> Self
    where
        I: IntoIterator<Item = AnthropicToolAllowedCaller>,
    {
        self.allowed_callers = Some(allowed_callers.into_iter().collect());
        self
    }

    /// Configure `eagerInputStreaming`.
    pub fn with_eager_input_streaming(mut self, eager_input_streaming: bool) -> Self {
        self.eager_input_streaming = Some(eager_input_streaming);
        self
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnthropicRequestMetadata {
    #[serde(skip_serializing_if = "Option::is_none", alias = "userId")]
    pub user_id: Option<String>,
}

impl AnthropicRequestMetadata {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_user_id(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicMcpServerType {
    Url,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnthropicMcpToolConfiguration {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "allowedTools")]
    pub allowed_tools: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnthropicMcpServer {
    #[serde(rename = "type")]
    pub kind: AnthropicMcpServerType,
    pub name: String,
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none", alias = "authorizationToken")]
    pub authorization_token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", alias = "toolConfiguration")]
    pub tool_configuration: Option<AnthropicMcpToolConfiguration>,
}

impl AnthropicMcpServer {
    pub fn url(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            kind: AnthropicMcpServerType::Url,
            name: name.into(),
            url: url.into(),
            authorization_token: None,
            tool_configuration: None,
        }
    }

    pub fn with_authorization_token(mut self, token: impl Into<String>) -> Self {
        self.authorization_token = Some(token.into());
        self
    }

    pub fn with_tool_configuration(
        mut self,
        tool_configuration: AnthropicMcpToolConfiguration,
    ) -> Self {
        self.tool_configuration = Some(tool_configuration);
        self
    }
}

/// Anthropic container configuration.
///
/// This is used for enabling agent skills and related features that are scoped to a container.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnthropicContainerConfig {
    /// Optional container id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Optional skills list.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skills: Option<Vec<AnthropicContainerSkill>>,
}

/// Container skill entry.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicContainerSkillType {
    Anthropic,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnthropicContainerSkill {
    /// Skill provider type (e.g., "anthropic", "custom").
    #[serde(rename = "type")]
    pub skill_type: AnthropicContainerSkillType,
    /// Skill id (snake_case for the HTTP API).
    #[serde(alias = "skillId")]
    pub skill_id: String,
    /// Skill version (e.g., "latest").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

impl AnthropicContainerSkill {
    pub fn anthropic(skill_id: impl Into<String>) -> Self {
        Self {
            skill_type: AnthropicContainerSkillType::Anthropic,
            skill_id: skill_id.into(),
            version: None,
        }
    }

    pub fn custom(skill_id: impl Into<String>) -> Self {
        Self {
            skill_type: AnthropicContainerSkillType::Custom,
            skill_id: skill_id.into(),
            version: None,
        }
    }

    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }
}

/// Request-side Anthropic context management configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnthropicContextManagementConfig {
    #[serde(default)]
    pub edits: Vec<AnthropicContextManagementEdit>,
}

impl AnthropicContextManagementConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_edit(mut self, edit: AnthropicContextManagementEdit) -> Self {
        self.edits.push(edit);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum AnthropicContextManagementEdit {
    #[serde(rename = "clear_tool_uses_20250919")]
    ClearToolUses20250919 {
        #[serde(skip_serializing_if = "Option::is_none")]
        trigger: Option<AnthropicContextManagementTrigger>,
        #[serde(skip_serializing_if = "Option::is_none")]
        keep: Option<AnthropicContextManagementToolUsesKeep>,
        #[serde(skip_serializing_if = "Option::is_none", alias = "clearAtLeast")]
        clear_at_least: Option<AnthropicContextManagementInputTokensValue>,
        #[serde(skip_serializing_if = "Option::is_none", alias = "clearToolInputs")]
        clear_tool_inputs: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none", alias = "excludeTools")]
        exclude_tools: Option<Vec<String>>,
    },
    #[serde(rename = "clear_thinking_20251015")]
    ClearThinking20251015 {
        #[serde(skip_serializing_if = "Option::is_none")]
        keep: Option<AnthropicContextManagementThinkingKeep>,
    },
    #[serde(rename = "compact_20260112")]
    Compact20260112 {
        #[serde(skip_serializing_if = "Option::is_none")]
        trigger: Option<AnthropicContextManagementInputTokensValue>,
        #[serde(
            skip_serializing_if = "Option::is_none",
            alias = "pauseAfterCompaction"
        )]
        pause_after_compaction: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        instructions: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum AnthropicContextManagementTrigger {
    #[serde(rename = "input_tokens")]
    InputTokens { value: u32 },
    #[serde(rename = "tool_uses")]
    ToolUses { value: u32 },
}

impl AnthropicContextManagementTrigger {
    pub fn input_tokens(value: u32) -> Self {
        Self::InputTokens { value }
    }

    pub fn tool_uses(value: u32) -> Self {
        Self::ToolUses { value }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum AnthropicContextManagementToolUsesKeep {
    #[serde(rename = "tool_uses")]
    ToolUses { value: u32 },
}

impl AnthropicContextManagementToolUsesKeep {
    pub fn tool_uses(value: u32) -> Self {
        Self::ToolUses { value }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum AnthropicContextManagementInputTokensValue {
    #[serde(rename = "input_tokens")]
    InputTokens { value: u32 },
}

impl AnthropicContextManagementInputTokensValue {
    pub fn input_tokens(value: u32) -> Self {
        Self::InputTokens { value }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum AnthropicContextManagementThinkingKeep {
    All(AnthropicContextManagementAllKeep),
    ThinkingTurns(AnthropicContextManagementThinkingTurnsKeep),
}

impl AnthropicContextManagementThinkingKeep {
    pub fn all() -> Self {
        Self::All(AnthropicContextManagementAllKeep::All)
    }

    pub fn thinking_turns(value: u32) -> Self {
        Self::ThinkingTurns(AnthropicContextManagementThinkingTurnsKeep {
            kind: AnthropicContextManagementThinkingTurnsKeepKind::ThinkingTurns,
            value,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicContextManagementAllKeep {
    All,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnthropicContextManagementThinkingTurnsKeep {
    #[serde(rename = "type")]
    pub kind: AnthropicContextManagementThinkingTurnsKeepKind,
    pub value: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AnthropicContextManagementThinkingTurnsKeepKind {
    ThinkingTurns,
}

/// Prompt caching configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PromptCachingConfig {
    /// Whether prompt caching is enabled
    pub enabled: bool,
    /// Cache control markers
    pub cache_control: Vec<AnthropicCacheControl>,
}

impl Default for PromptCachingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cache_control: vec![],
        }
    }
}

/// Anthropic cache control marker
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnthropicCacheControl {
    /// Cache type
    pub cache_type: AnthropicCacheType,
    /// Message index to apply cache control to
    pub message_index: usize,
}

/// Anthropic cache type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicCacheType {
    /// Ephemeral cache (5 minutes TTL)
    Ephemeral,
}

/// Legacy thinking mode configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ThinkingModeConfig {
    /// Whether thinking mode is enabled
    pub enabled: bool,
    /// Thinking budget (tokens)
    pub thinking_budget: Option<u32>,
}

impl Default for ThinkingModeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            thinking_budget: None,
        }
    }
}

impl ThinkingModeConfig {
    pub fn enabled(thinking_budget: Option<u32>) -> Self {
        Self {
            enabled: true,
            thinking_budget,
        }
    }
}

/// Anthropic structured output configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AnthropicResponseFormat {
    /// Plain JSON object output
    JsonObject,
    /// JSON schema output with name and strict flag
    JsonSchema {
        name: String,
        schema: serde_json::Value,
        strict: bool,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structured_output_mode_serializes_to_provider_options_shape() {
        let value = serde_json::to_value(
            AnthropicOptions::new()
                .with_structured_output_mode(AnthropicStructuredOutputMode::JsonTool),
        )
        .expect("serialize anthropic options");

        assert_eq!(
            value.get("structured_output_mode"),
            Some(&serde_json::json!("jsonTool"))
        );
    }

    #[test]
    fn options_deserialize_from_camel_case_shape() {
        let options: AnthropicOptions = serde_json::from_value(serde_json::json!({
            "thinking": {
                "type": "enabled",
                "budgetTokens": 2048
            },
            "sendReasoning": false,
            "structuredOutputMode": "outputFormat",
            "disableParallelToolUse": true,
            "cacheControl": {
                "type": "ephemeral",
                "ttl": "1h"
            },
            "metadata": {
                "userId": "user-1"
            },
            "mcpServers": [
                {
                    "type": "url",
                    "name": "echo",
                    "url": "https://example.com/mcp",
                    "authorizationToken": "tok",
                    "toolConfiguration": {
                        "enabled": true,
                        "allowedTools": ["ping"]
                    }
                }
            ],
            "container": {
                "id": "container-1",
                "skills": [
                    {
                        "type": "anthropic",
                        "skillId": "pptx",
                        "version": "latest"
                    }
                ]
            },
            "contextManagement": {
                "edits": [
                    {
                        "type": "compact_20260112",
                        "pauseAfterCompaction": true,
                        "instructions": "Summarize before compaction."
                    }
                ]
            },
            "toolStreaming": false,
            "effort": "max",
            "speed": "fast",
            "anthropicBeta": ["beta-1"]
        }))
        .expect("deserialize anthropic options");

        assert!(matches!(
            options.thinking,
            Some(AnthropicThinkingConfig::Enabled {
                budget_tokens: Some(2048)
            })
        ));
        assert_eq!(options.send_reasoning, Some(false));
        assert_eq!(
            options.structured_output_mode,
            Some(AnthropicStructuredOutputMode::OutputFormat)
        );
        assert_eq!(options.disable_parallel_tool_use, Some(true));
        assert!(matches!(
            options.cache_control.as_ref(),
            Some(AnthropicRequestCacheControl {
                kind: AnthropicRequestCacheControlType::Ephemeral,
                ttl: Some(AnthropicRequestCacheControlTtl::OneHour)
            })
        ));
        assert_eq!(
            options
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.user_id.as_deref()),
            Some("user-1")
        );
        assert_eq!(
            options
                .mcp_servers
                .as_ref()
                .and_then(|servers| servers.first())
                .map(|server| server.authorization_token.as_deref()),
            Some(Some("tok"))
        );
        assert_eq!(
            options
                .container
                .as_ref()
                .and_then(|container| container.id.as_deref()),
            Some("container-1")
        );
        assert!(matches!(
            options
                .container
                .as_ref()
                .and_then(|container| container.skills.as_ref())
                .and_then(|skills| skills.first()),
            Some(AnthropicContainerSkill {
                skill_type: AnthropicContainerSkillType::Anthropic,
                skill_id,
                version: Some(version),
            }) if skill_id == "pptx" && version == "latest"
        ));
        assert!(matches!(
            options
                .context_management
                .as_ref()
                .and_then(|cm| cm.edits.first()),
            Some(AnthropicContextManagementEdit::Compact20260112 {
                pause_after_compaction: Some(true),
                instructions: Some(_),
                ..
            })
        ));
        assert_eq!(options.tool_streaming, Some(false));
        assert_eq!(options.effort, Some(AnthropicEffort::Max));
        assert_eq!(options.speed, Some(AnthropicSpeed::Fast));
        assert_eq!(options.anthropic_beta, Some(vec!["beta-1".to_string()]));
    }

    #[test]
    fn context_management_serializes_to_snake_case_request_shape() {
        let value = serde_json::to_value(AnthropicOptions::new().with_context_management(
            AnthropicContextManagementConfig::new().with_edit(
                AnthropicContextManagementEdit::Compact20260112 {
                    trigger: Some(AnthropicContextManagementInputTokensValue::input_tokens(42)),
                    pause_after_compaction: Some(true),
                    instructions: Some("Keep key decisions.".to_string()),
                },
            ),
        ))
        .expect("serialize anthropic options");

        assert_eq!(
            value.get("context_management"),
            Some(&serde_json::json!({
                "edits": [{
                    "type": "compact_20260112",
                    "trigger": {
                        "type": "input_tokens",
                        "value": 42
                    },
                    "pause_after_compaction": true,
                    "instructions": "Keep key decisions."
                }]
            }))
        );
    }

    #[test]
    fn options_serialization_omits_unset_fields() {
        let value = serde_json::to_value(
            AnthropicOptions::new()
                .with_speed(AnthropicSpeed::Fast)
                .with_anthropic_beta("beta-1"),
        )
        .expect("serialize anthropic options");

        let obj = value.as_object().expect("anthropic options object");
        assert_eq!(obj.get("speed"), Some(&serde_json::json!("fast")));
        assert_eq!(
            obj.get("anthropic_beta"),
            Some(&serde_json::json!(["beta-1"]))
        );
        assert!(!obj.contains_key("thinking_mode"));
        assert!(!obj.contains_key("thinking"));
        assert!(!obj.contains_key("context_management"));
        assert!(!obj.contains_key("tool_streaming"));
    }

    #[test]
    #[allow(deprecated)]
    fn options_aliases_keep_the_same_serialized_shape() {
        let canonical = AnthropicLanguageModelOptions::new()
            .with_speed(AnthropicSpeed::Standard)
            .with_send_reasoning(true);
        let deprecated = AnthropicProviderOptions::new()
            .with_speed(AnthropicSpeed::Standard)
            .with_send_reasoning(true);

        assert_eq!(
            serde_json::to_value(canonical).expect("serialize canonical alias"),
            serde_json::to_value(deprecated).expect("serialize deprecated alias")
        );
    }

    #[test]
    fn tool_options_serialize_camel_case_shape() {
        let value = serde_json::to_value(
            AnthropicToolOptions::new()
                .with_defer_loading(true)
                .with_allowed_callers([
                    AnthropicToolAllowedCaller::Direct,
                    AnthropicToolAllowedCaller::CodeExecution20260120,
                ])
                .with_eager_input_streaming(true),
        )
        .expect("serialize anthropic tool options");

        assert_eq!(
            value,
            serde_json::json!({
                "deferLoading": true,
                "allowedCallers": ["direct", "code_execution_20260120"],
                "eagerInputStreaming": true
            })
        );
    }

    #[test]
    fn tool_options_deserialize_snake_case_aliases() {
        let options: AnthropicToolOptions = serde_json::from_value(serde_json::json!({
            "defer_loading": true,
            "allowed_callers": ["direct", "code_execution_20250825"],
            "eager_input_streaming": true
        }))
        .expect("deserialize anthropic tool options");

        assert_eq!(options.defer_loading, Some(true));
        assert_eq!(
            options.allowed_callers,
            Some(vec![
                AnthropicToolAllowedCaller::Direct,
                AnthropicToolAllowedCaller::CodeExecution20250825,
            ])
        );
        assert_eq!(options.eager_input_streaming, Some(true));
    }
}
