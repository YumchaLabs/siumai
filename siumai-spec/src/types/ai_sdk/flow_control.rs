use crate::types::{
    AssistantContent, AssistantContentPart, ModelMessage, Tool, ToolContentPart, UserContent,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{GenerateTextStepResult, JSONValue};

/// Passive representation of AI SDK `StopCondition`.
///
/// The TypeScript surface accepts predicates/functions. Rust exposes the built-in
/// conditions as symbolic data and evaluates only those built-ins. `Custom` is a
/// transport lane for application-owned metadata and never evaluates to `true`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum StopCondition {
    /// Equivalent to AI SDK `isStepCount(stepCount)`.
    StepCount {
        /// Number of completed steps required for the condition to match.
        #[serde(rename = "stepCount", alias = "step_count", alias = "maxSteps")]
        step_count: usize,
    },
    /// Equivalent to AI SDK `isLoopFinished()`, which never stops by itself.
    LoopFinished,
    /// Equivalent to AI SDK `hasToolCall(...toolNames)`.
    ToolCall {
        /// Tool names that should stop the loop when present in the latest step.
        #[serde(rename = "toolNames", alias = "tool_names")]
        tool_names: Vec<String>,
    },
    /// Application-owned symbolic condition metadata.
    Custom {
        /// Opaque condition payload.
        value: JSONValue,
    },
}

impl StopCondition {
    /// Create a step-count stop condition.
    pub const fn is_step_count(step_count: usize) -> Self {
        Self::StepCount { step_count }
    }

    /// Create a condition that never stops the loop by itself.
    pub const fn is_loop_finished() -> Self {
        Self::LoopFinished
    }

    /// Create a condition that matches tool calls in the latest step.
    pub fn has_tool_call(tool_names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self::ToolCall {
            tool_names: tool_names.into_iter().map(Into::into).collect(),
        }
    }

    /// Create an application-owned custom condition.
    pub fn custom(value: impl Into<JSONValue>) -> Self {
        Self::Custom {
            value: value.into(),
        }
    }

    /// Evaluate the built-in stop condition against completed steps.
    pub fn is_met<NAME, INPUT, OUTPUT>(
        &self,
        steps: &[GenerateTextStepResult<NAME, INPUT, OUTPUT>],
    ) -> bool
    where
        NAME: AsRef<str>,
    {
        match self {
            Self::StepCount { step_count } => steps.len() == *step_count,
            Self::LoopFinished | Self::Custom { .. } => false,
            Self::ToolCall { tool_names } => {
                let Some(step) = steps.last() else {
                    return false;
                };
                step.tool_calls.iter().any(|tool_call| {
                    tool_names
                        .iter()
                        .any(|tool_name| tool_name == tool_call.tool_name.as_ref())
                })
            }
        }
    }
}

/// Create a step-count stop condition.
pub const fn is_step_count(step_count: usize) -> StopCondition {
    StopCondition::is_step_count(step_count)
}

/// Deprecated AI SDK `stepCountIs` helper alias.
#[deprecated(note = "Use is_step_count instead.")]
pub const fn step_count_is(step_count: usize) -> StopCondition {
    is_step_count(step_count)
}

/// Create a condition that never stops the loop by itself.
pub const fn is_loop_finished() -> StopCondition {
    StopCondition::is_loop_finished()
}

/// Create a condition that matches tool calls in the latest step.
pub fn has_tool_call(tool_names: impl IntoIterator<Item = impl Into<String>>) -> StopCondition {
    StopCondition::has_tool_call(tool_names)
}

/// Evaluate built-in stop conditions, returning true when any condition matches.
pub fn is_stop_condition_met<NAME, INPUT, OUTPUT>(
    stop_conditions: &[StopCondition],
    steps: &[GenerateTextStepResult<NAME, INPUT, OUTPUT>],
) -> bool
where
    NAME: AsRef<str>,
{
    stop_conditions
        .iter()
        .any(|condition| condition.is_met(steps))
}

fn tool_name(tool: &Tool) -> &str {
    match tool {
        Tool::Function { function } => function.name.as_str(),
        Tool::ProviderDefined(tool) => tool.name.as_str(),
    }
}

/// Filter tools to the active tool names, matching AI SDK `filterActiveTools`.
pub fn filter_active_tools<N>(
    tools: Option<&[Tool]>,
    active_tools: Option<&[N]>,
) -> Option<Vec<Tool>>
where
    N: AsRef<str>,
{
    let tools = tools?;
    let Some(active_tools) = active_tools else {
        return Some(tools.to_vec());
    };

    Some(
        tools
            .iter()
            .filter(|tool| {
                active_tools
                    .iter()
                    .any(|active_tool| active_tool.as_ref() == tool_name(tool))
            })
            .cloned()
            .collect(),
    )
}

/// AI SDK `experimental_filterActiveTools` helper alias using Rust naming.
pub fn experimental_filter_active_tools<N>(
    tools: Option<&[Tool]>,
    active_tools: Option<&[N]>,
) -> Option<Vec<Tool>>
where
    N: AsRef<str>,
{
    filter_active_tools(tools, active_tools)
}

/// Reasoning pruning strategy for AI SDK `pruneMessages`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum PruneReasoningMode {
    /// Remove reasoning from all assistant messages.
    All,
    /// Remove reasoning from all assistant messages except the last message.
    BeforeLastMessage,
    /// Keep reasoning parts.
    #[default]
    None,
}

/// Empty-message handling for AI SDK `pruneMessages`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum PruneEmptyMessagesMode {
    /// Keep messages even when their content is empty after pruning.
    Keep,
    /// Remove messages whose content is empty after pruning.
    #[default]
    Remove,
}

/// Tool pruning scope for AI SDK `pruneMessages`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum PruneToolCallMode {
    /// Prune matching tool parts from all messages.
    All,
    /// Prune matching tool parts from all messages except the last message.
    BeforeLastMessage,
    /// Prune matching tool parts before the last `count` messages.
    BeforeLastMessages {
        /// Number of trailing messages to keep.
        count: usize,
    },
}

/// One tool-call pruning rule for AI SDK `pruneMessages`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct PruneToolCallRule {
    /// Scope that decides which messages are eligible for pruning.
    pub mode: PruneToolCallMode,
    /// Optional tool-name allowlist. Parts for tools outside this list are kept.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<String>>,
}

impl PruneToolCallRule {
    /// Prune matching tool parts from all messages.
    pub const fn all() -> Self {
        Self {
            mode: PruneToolCallMode::All,
            tools: None,
        }
    }

    /// Prune matching tool parts before the last message.
    pub const fn before_last_message() -> Self {
        Self {
            mode: PruneToolCallMode::BeforeLastMessage,
            tools: None,
        }
    }

    /// Prune matching tool parts before the last `count` messages.
    pub const fn before_last_messages(count: usize) -> Self {
        Self {
            mode: PruneToolCallMode::BeforeLastMessages { count },
            tools: None,
        }
    }

    /// Limit pruning to specific tool names.
    pub fn with_tools(mut self, tools: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tools = Some(tools.into_iter().map(Into::into).collect());
        self
    }

    fn keep_last_messages_count(&self) -> Option<usize> {
        match self.mode {
            PruneToolCallMode::All => None,
            PruneToolCallMode::BeforeLastMessage => Some(1),
            PruneToolCallMode::BeforeLastMessages { count } => Some(count),
        }
    }
}

/// Options for AI SDK `pruneMessages`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct PruneMessagesOptions {
    /// How to remove reasoning content from assistant messages.
    #[serde(default)]
    pub reasoning: PruneReasoningMode,
    /// Tool-call/result/approval pruning rules.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<PruneToolCallRule>,
    /// Whether to keep or remove messages that become empty.
    #[serde(default)]
    pub empty_messages: PruneEmptyMessagesMode,
}

impl PruneMessagesOptions {
    /// Create options with AI SDK defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set reasoning pruning mode.
    pub const fn with_reasoning(mut self, reasoning: PruneReasoningMode) -> Self {
        self.reasoning = reasoning;
        self
    }

    /// Set tool pruning rules.
    pub fn with_tool_calls(mut self, tool_calls: Vec<PruneToolCallRule>) -> Self {
        self.tool_calls = tool_calls;
        self
    }

    /// Set empty-message handling.
    pub const fn with_empty_messages(mut self, empty_messages: PruneEmptyMessagesMode) -> Self {
        self.empty_messages = empty_messages;
        self
    }
}

fn is_tool_name_outside_rule(tools: Option<&Vec<String>>, tool_name: Option<&str>) -> bool {
    match tools {
        Some(tools) => match tool_name {
            Some(tool_name) => !tools.iter().any(|tool| tool == tool_name),
            None => true,
        },
        None => false,
    }
}

fn should_keep_tool_part(
    rule: &PruneToolCallRule,
    id: &str,
    tool_name: Option<&str>,
    kept_ids: &std::collections::HashSet<String>,
) -> bool {
    kept_ids.contains(id) || is_tool_name_outside_rule(rule.tools.as_ref(), tool_name)
}

fn collect_kept_tool_part_ids(
    message: &ModelMessage,
    kept_tool_call_ids: &mut std::collections::HashSet<String>,
    kept_approval_ids: &mut std::collections::HashSet<String>,
) {
    match message {
        ModelMessage::Assistant(message) => {
            let AssistantContent::Parts(parts) = &message.content else {
                return;
            };
            for part in parts {
                match part {
                    AssistantContentPart::ToolCall(part) => {
                        kept_tool_call_ids.insert(part.tool_call_id.clone());
                    }
                    AssistantContentPart::ToolResult(part) => {
                        kept_tool_call_ids.insert(part.tool_call_id.clone());
                    }
                    AssistantContentPart::ToolApprovalRequest(part) => {
                        kept_approval_ids.insert(part.approval_id.clone());
                    }
                    _ => {}
                }
            }
        }
        ModelMessage::Tool(message) => {
            for part in &message.content {
                match part {
                    ToolContentPart::ToolResult(part) => {
                        kept_tool_call_ids.insert(part.tool_call_id.clone());
                    }
                    ToolContentPart::ToolApprovalResponse(part) => {
                        kept_approval_ids.insert(part.approval_id.clone());
                    }
                }
            }
        }
        _ => {}
    }
}

fn prune_assistant_tool_parts(
    parts: Vec<AssistantContentPart>,
    rule: &PruneToolCallRule,
    kept_tool_call_ids: &std::collections::HashSet<String>,
    kept_approval_ids: &std::collections::HashSet<String>,
) -> Vec<AssistantContentPart> {
    let mut tool_call_id_to_tool_name = HashMap::<String, String>::new();
    let mut approval_id_to_tool_name = HashMap::<String, String>::new();

    parts
        .into_iter()
        .filter(|part| match part {
            AssistantContentPart::ToolCall(part) => {
                tool_call_id_to_tool_name.insert(part.tool_call_id.clone(), part.tool_name.clone());
                should_keep_tool_part(
                    rule,
                    &part.tool_call_id,
                    Some(part.tool_name.as_str()),
                    kept_tool_call_ids,
                )
            }
            AssistantContentPart::ToolResult(part) => should_keep_tool_part(
                rule,
                &part.tool_call_id,
                Some(part.tool_name.as_str()),
                kept_tool_call_ids,
            ),
            AssistantContentPart::ToolApprovalRequest(part) => {
                let tool_name = tool_call_id_to_tool_name
                    .get(&part.tool_call_id)
                    .map(String::as_str);
                if let Some(tool_name) = tool_name {
                    approval_id_to_tool_name
                        .insert(part.approval_id.clone(), tool_name.to_string());
                }
                should_keep_tool_part(rule, &part.approval_id, tool_name, kept_approval_ids)
            }
            _ => true,
        })
        .collect()
}

fn prune_tool_message_parts(
    parts: Vec<ToolContentPart>,
    rule: &PruneToolCallRule,
    kept_tool_call_ids: &std::collections::HashSet<String>,
    kept_approval_ids: &std::collections::HashSet<String>,
) -> Vec<ToolContentPart> {
    let approval_id_to_tool_name = HashMap::<String, String>::new();

    parts
        .into_iter()
        .filter(|part| match part {
            ToolContentPart::ToolResult(part) => should_keep_tool_part(
                rule,
                &part.tool_call_id,
                Some(part.tool_name.as_str()),
                kept_tool_call_ids,
            ),
            ToolContentPart::ToolApprovalResponse(part) => {
                let tool_name = approval_id_to_tool_name
                    .get(&part.approval_id)
                    .map(String::as_str);
                should_keep_tool_part(rule, &part.approval_id, tool_name, kept_approval_ids)
            }
        })
        .collect()
}

fn model_message_is_empty(message: &ModelMessage) -> bool {
    match message {
        ModelMessage::System(message) => message.content.is_empty(),
        ModelMessage::User(message) => match &message.content {
            UserContent::Text(text) => text.is_empty(),
            UserContent::Parts(parts) => parts.is_empty(),
        },
        ModelMessage::Assistant(message) => match &message.content {
            AssistantContent::Text(text) => text.is_empty(),
            AssistantContent::Parts(parts) => parts.is_empty(),
        },
        ModelMessage::Tool(message) => message.content.is_empty(),
    }
}

/// Prune AI SDK-style model messages.
///
/// This mirrors the pure data behavior of `generate-text/prune-messages.ts`: reasoning parts can
/// be removed from assistant messages, tool call/result/approval parts can be pruned by recency
/// and tool name, and messages that become empty are removed by default.
pub fn prune_messages(
    mut messages: Vec<ModelMessage>,
    options: PruneMessagesOptions,
) -> Vec<ModelMessage> {
    if matches!(
        options.reasoning,
        PruneReasoningMode::All | PruneReasoningMode::BeforeLastMessage
    ) {
        let last_index = messages.len().saturating_sub(1);
        messages = messages
            .into_iter()
            .enumerate()
            .map(|(index, message)| {
                let ModelMessage::Assistant(mut assistant) = message else {
                    return message;
                };
                if options.reasoning == PruneReasoningMode::BeforeLastMessage && index == last_index
                {
                    return ModelMessage::Assistant(assistant);
                }
                let AssistantContent::Parts(parts) = assistant.content else {
                    return ModelMessage::Assistant(assistant);
                };

                assistant.content = AssistantContent::Parts(
                    parts
                        .into_iter()
                        .filter(|part| !matches!(part, AssistantContentPart::Reasoning(_)))
                        .collect(),
                );
                ModelMessage::Assistant(assistant)
            })
            .collect();
    }

    for rule in &options.tool_calls {
        let keep_last_messages_count = rule.keep_last_messages_count();
        let mut kept_tool_call_ids = std::collections::HashSet::new();
        let mut kept_approval_ids = std::collections::HashSet::new();

        if let Some(count) = keep_last_messages_count {
            for message in messages.iter().rev().take(count) {
                collect_kept_tool_part_ids(
                    message,
                    &mut kept_tool_call_ids,
                    &mut kept_approval_ids,
                );
            }
        }

        let prune_before_index = keep_last_messages_count
            .map(|count| messages.len().saturating_sub(count))
            .unwrap_or(messages.len());

        messages = messages
            .into_iter()
            .enumerate()
            .map(|(index, message)| {
                if index >= prune_before_index {
                    return message;
                }

                match message {
                    ModelMessage::Assistant(mut assistant) => {
                        let AssistantContent::Parts(parts) = assistant.content else {
                            return ModelMessage::Assistant(assistant);
                        };
                        assistant.content = AssistantContent::Parts(prune_assistant_tool_parts(
                            parts,
                            rule,
                            &kept_tool_call_ids,
                            &kept_approval_ids,
                        ));
                        ModelMessage::Assistant(assistant)
                    }
                    ModelMessage::Tool(mut tool) => {
                        tool.content = prune_tool_message_parts(
                            tool.content,
                            rule,
                            &kept_tool_call_ids,
                            &kept_approval_ids,
                        );
                        ModelMessage::Tool(tool)
                    }
                    other => other,
                }
            })
            .collect();
    }

    if options.empty_messages == PruneEmptyMessagesMode::Remove {
        messages.retain(|message| !model_message_is_empty(message));
    }

    messages
}
