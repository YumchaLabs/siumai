//! Canonical Anthropic Messages wire codec.
//!
//! This module maps provider-neutral language requests, responses, and stream
//! events to the Anthropic Messages protocol. It intentionally owns no
//! credentials, endpoints, HTTP execution, retries, model catalog, or provider
//! construction.

mod annotations;
mod error;
mod native_content;
mod options;
mod request;
mod response;
mod rules;
mod stream;
mod wire;

pub use annotations::{
    AnthropicToolReference, CacheControl, CacheTtl, ContentNodeOptions, MessageNodeOptions,
    MessagesAnnotationResolver, MessagesFileBlock, MessagesFileReference,
    MidConversationToolChange, NoMessagesAnnotations, ToolNodeOptions,
};
pub use error::MessagesCodecError;
pub use native_content::{
    AnthropicHostedToolBlockRef, AnthropicHostedToolResultKind, AnthropicHostedToolResultRef,
    AnthropicMcpToolUseRef, AnthropicOpaqueContentExt, AnthropicServerToolUseRef,
};
pub use options::{
    AdvisorToolOptions, AnthropicTool, ClearThinkingEdit, ClearThinkingKeep, ClearToolInputs,
    ClearToolUsesEdit, CompactionEdit, ComputerToolOptions, ContainerSkill, ContainerSkillType,
    ContextManagement, ContextManagementEdit, ContextManagementTrigger, FallbackOutputConfig,
    InferenceGeo, InferenceSpeed, McpAuthorizationToken, McpServer, McpToolConfig,
    McpToolsetOptions, MessagesAssignedServiceTier, MessagesContainer, MessagesMetadata,
    MessagesRequestOptions, MessagesServiceTierPreference, MessagesTokenCountOptions, OutputEffort,
    ResponseInclusion, ServerFallback, ServerFallbacks, TextEditorToolOptions, ThinkingConfig,
    ThinkingDisplay, TokenTaskBudget, ToolCaller, UserLocation, WebFetchToolOptions,
    WebSearchToolOptions,
};
pub use request::{
    anthropic_tool_anchor_schema, encode_count_tokens_request,
    encode_count_tokens_request_for_scope_with_resolver_and_rules,
    encode_count_tokens_request_with_resolver_and_rules, encode_request, encode_request_for_scope,
    encode_request_for_scope_with_resolver, encode_request_for_scope_with_resolver_and_rules,
    encode_request_with_resolver, encode_request_with_resolver_and_rules,
    encode_request_with_rules, is_protected_option_field,
};
pub use response::decode_response;
pub use rules::{
    CacheControlWireStyle, MessagesEncodingRuleError, MessagesEncodingRules,
    MidConversationSystemEncoding, TemperatureEncodingRule,
};
pub use stream::MessagesStreamDecoder;

/// Stable protocol identity used for native replay provenance.
pub const PROTOCOL_ID: &str = "anthropic-messages";

/// Stable API-mode identity for the Messages operation.
pub const API_MODE_ID: &str = "messages";

/// Anthropic Messages request target relative to a configured API base.
pub const MESSAGES_TARGET: &str = "messages";

/// Anthropic token-count target relative to a configured API base.
pub const MESSAGES_COUNT_TOKENS_TARGET: &str = "messages/count_tokens";

/// Opaque-item kind used for exact Anthropic content-block replay.
pub const OPAQUE_CONTENT_BLOCK_KIND: &str = "anthropic.messages.content-block";

#[cfg(test)]
mod request_contract_tests;
#[cfg(test)]
mod response_contract_tests;
#[cfg(test)]
mod tests;
