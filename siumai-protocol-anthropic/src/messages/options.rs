use std::collections::BTreeMap;
use std::fmt;

use secrecy::{ExposeSecret, SecretString};
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use siumai_core::{LanguageRequest, ModelId, StructuredOutputSpec, ToolChoice};
use url::Url;

use super::MessagesCodecError;
use super::annotations::{CacheControl, ToolNodeOptions};
use super::request::is_protected_option_field;

const MAX_EXTRA_DEPTH: usize = 16;
const MAX_EXTRA_FIELDS: usize = 1_024;
const MAX_EXTRA_BYTES: usize = 256 * 1024;
const MAX_FALLBACKS: usize = 3;
const MAX_TOOL_LIST: usize = 256;
const MAX_DOMAIN_LIST: usize = 256;
const MAX_DOMAIN_BYTES: usize = 253;
const MAX_TOOL_CONFIGS: usize = 256;
const MAX_TOOL_NAME_BYTES: usize = 128;
const MAX_ALLOWED_CALLERS: usize = 4;
const MAX_CONTAINER_ID_BYTES: usize = 256;
const MAX_CONTAINER_SKILLS: usize = 8;
const MAX_SKILL_ID_BYTES: usize = 256;
const MAX_SKILL_VERSION_BYTES: usize = 128;
const MAX_CONTEXT_EDITS: usize = 16;
const MAX_CONTEXT_TOOL_NAMES: usize = 256;
const MAX_CONTEXT_INSTRUCTIONS_BYTES: usize = 16 * 1024;
const MIN_COMPACTION_TRIGGER_INPUT_TOKENS: u64 = 50_000;
const MAX_MCP_SERVERS: usize = 20;
const MAX_MCP_SERVER_NAME_BYTES: usize = 128;
const MAX_MCP_SERVER_URL_BYTES: usize = 2_048;
const MAX_MCP_AUTHORIZATION_TOKEN_BYTES: usize = 16 * 1024;

/// Anthropic extended-thinking display mode for one Messages request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum ThinkingDisplay {
    /// Return summarized thinking blocks in the response.
    Summarized,
    /// Omit thinking text while retaining a replayable signature.
    Omitted,
}

impl ThinkingDisplay {
    pub(crate) const fn as_wire_str(self) -> &'static str {
        match self {
            Self::Summarized => "summarized",
            Self::Omitted => "omitted",
        }
    }
}

/// Anthropic extended-thinking mode for one Messages request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
#[non_exhaustive]
pub enum ThinkingConfig {
    Disabled,
    Enabled {
        budget_tokens: u64,
        display: Option<ThinkingDisplay>,
    },
    Adaptive {
        display: Option<ThinkingDisplay>,
    },
}

impl ThinkingConfig {
    pub const fn enabled(budget_tokens: u64) -> Self {
        Self::Enabled {
            budget_tokens,
            display: None,
        }
    }

    pub const fn adaptive() -> Self {
        Self::Adaptive { display: None }
    }

    pub const fn with_display(self, display: ThinkingDisplay) -> Self {
        match self {
            Self::Disabled => Self::Disabled,
            Self::Enabled { budget_tokens, .. } => Self::Enabled {
                budget_tokens,
                display: Some(display),
            },
            Self::Adaptive { .. } => Self::Adaptive {
                display: Some(display),
            },
        }
    }

    pub const fn display(self) -> Option<ThinkingDisplay> {
        match self {
            Self::Disabled => None,
            Self::Enabled { display, .. } | Self::Adaptive { display } => display,
        }
    }

    pub const fn budget_tokens(self) -> Option<u64> {
        match self {
            Self::Enabled { budget_tokens, .. } => Some(budget_tokens),
            Self::Disabled | Self::Adaptive { .. } => None,
        }
    }
}

/// Effort level accepted by Anthropic's `output_config.effort` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum OutputEffort {
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

impl OutputEffort {
    pub(crate) const fn as_wire_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
        }
    }
}

/// Request-side preference accepted by the Messages `service_tier` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum MessagesServiceTierPreference {
    Auto,
    StandardOnly,
}

impl MessagesServiceTierPreference {
    pub(crate) const fn as_wire_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::StandardOnly => "standard_only",
        }
    }
}

/// Response-side service tier assigned by Anthropic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum MessagesAssignedServiceTier {
    Standard,
    Priority,
    Batch,
}

impl MessagesAssignedServiceTier {
    pub(crate) const fn as_wire_str(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Priority => "priority",
            Self::Batch => "batch",
        }
    }

    pub(crate) fn from_wire_str(value: &str) -> Option<Self> {
        match value {
            "standard" => Some(Self::Standard),
            "priority" => Some(Self::Priority),
            "batch" => Some(Self::Batch),
            _ => None,
        }
    }
}

/// Inference speed override for a Messages request or fallback attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum InferenceSpeed {
    Standard,
    Fast,
}

impl InferenceSpeed {
    pub(crate) const fn as_wire_str(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Fast => "fast",
        }
    }
}

/// Request-side inference geography accepted by the current Messages API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum InferenceGeo {
    Global,
    Us,
}

impl InferenceGeo {
    pub const fn global() -> Self {
        Self::Global
    }

    pub const fn us() -> Self {
        Self::Us
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::Us => "us",
        }
    }
}

/// Token budget carried by `output_config.task_budget`.
///
/// This open protocol carrier requires a positive total and, when present, a
/// remaining budget no greater than that total. Model-specific minima belong
/// to the provider policy that selects the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenTaskBudget {
    total: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    remaining: Option<u64>,
}

impl TokenTaskBudget {
    pub fn new(total: u64) -> Result<Self, MessagesCodecError> {
        let value = Self {
            total,
            remaining: None,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn with_remaining(mut self, remaining: u64) -> Result<Self, MessagesCodecError> {
        self.remaining = Some(remaining);
        self.validate()?;
        Ok(self)
    }

    pub const fn total(self) -> u64 {
        self.total
    }

    pub const fn remaining(self) -> Option<u64> {
        self.remaining
    }

    pub(crate) fn validate(&self) -> Result<(), MessagesCodecError> {
        if self.total == 0 {
            return Err(MessagesCodecError::InvalidOption {
                field: "output_config.task_budget.total",
                reason: "must be greater than zero",
            });
        }
        if self
            .remaining
            .is_some_and(|remaining| remaining > self.total)
        {
            return Err(MessagesCodecError::InvalidOption {
                field: "output_config.task_budget.remaining",
                reason: "must not exceed the total token budget",
            });
        }
        Ok(())
    }
}

/// Source namespace for a container skill.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum ContainerSkillType {
    Anthropic,
    Custom,
}

/// One skill mounted into an Anthropic code-execution container.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContainerSkill {
    #[serde(rename = "type")]
    kind: ContainerSkillType,
    skill_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    version: Option<String>,
}

impl ContainerSkill {
    pub fn anthropic(skill_id: impl Into<String>) -> Result<Self, MessagesCodecError> {
        Self::new(ContainerSkillType::Anthropic, skill_id)
    }

    pub fn custom(skill_id: impl Into<String>) -> Result<Self, MessagesCodecError> {
        Self::new(ContainerSkillType::Custom, skill_id)
    }

    pub fn new(
        kind: ContainerSkillType,
        skill_id: impl Into<String>,
    ) -> Result<Self, MessagesCodecError> {
        let value = Self {
            kind,
            skill_id: skill_id.into(),
            version: None,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn with_version(mut self, version: impl Into<String>) -> Result<Self, MessagesCodecError> {
        self.version = Some(version.into());
        self.validate()?;
        Ok(self)
    }

    pub const fn kind(&self) -> ContainerSkillType {
        self.kind
    }

    pub fn skill_id(&self) -> &str {
        &self.skill_id
    }

    pub fn version(&self) -> Option<&str> {
        self.version.as_deref()
    }

    pub(crate) fn validate(&self) -> Result<(), MessagesCodecError> {
        validate_printable_string(
            &self.skill_id,
            "container.skills[].skill_id",
            MAX_SKILL_ID_BYTES,
            "must be a non-empty printable value of at most 256 bytes",
        )?;
        if let Some(version) = &self.version {
            validate_printable_string(
                version,
                "container.skills[].version",
                MAX_SKILL_VERSION_BYTES,
                "must be a non-empty printable value of at most 128 bytes",
            )?;
        }
        Ok(())
    }
}

/// Container reuse or construction settings for one Messages request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
#[non_exhaustive]
pub enum MessagesContainer {
    Existing(String),
    Configured {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        skills: Vec<ContainerSkill>,
    },
}

impl MessagesContainer {
    pub fn existing(id: impl Into<String>) -> Result<Self, MessagesCodecError> {
        let value = Self::Existing(id.into());
        value.validate()?;
        Ok(value)
    }

    pub const fn configured() -> Self {
        Self::Configured {
            id: None,
            skills: Vec::new(),
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Result<Self, MessagesCodecError> {
        let id = id.into();
        match &mut self {
            Self::Existing(existing) => *existing = id,
            Self::Configured {
                id: configured_id, ..
            } => *configured_id = Some(id),
        }
        self.validate()?;
        Ok(self)
    }

    pub fn with_skill(mut self, skill: ContainerSkill) -> Result<Self, MessagesCodecError> {
        self = match self {
            Self::Existing(id) => Self::Configured {
                id: Some(id),
                skills: vec![skill],
            },
            Self::Configured { id, mut skills } => {
                skills.push(skill);
                Self::Configured { id, skills }
            }
        };
        self.validate()?;
        Ok(self)
    }

    pub fn skills(&self) -> &[ContainerSkill] {
        match self {
            Self::Existing(_) => &[],
            Self::Configured { skills, .. } => skills,
        }
    }

    pub fn has_skills(&self) -> bool {
        !self.skills().is_empty()
    }

    pub(crate) fn validate(&self) -> Result<(), MessagesCodecError> {
        match self {
            Self::Existing(id) => validate_container_id(id),
            Self::Configured { id, skills } => {
                if let Some(id) = id {
                    validate_container_id(id)?;
                }
                if skills.len() > MAX_CONTAINER_SKILLS {
                    return Err(MessagesCodecError::InvalidOption {
                        field: "container.skills",
                        reason: "must contain at most eight skills",
                    });
                }
                for skill in skills {
                    skill.validate()?;
                }
                Ok(())
            }
        }
    }
}

/// Selection policy for tool inputs cleared by context management.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
#[non_exhaustive]
pub enum ClearToolInputs {
    All(bool),
    Selected(Vec<String>),
}

/// Threshold that activates one context-management edit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ContextManagementTrigger {
    InputTokens(u64),
    ToolUses(u64),
}

/// Retention policy used by the clear-thinking edit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ClearThinkingKeep {
    All,
    ThinkingTurns(u64),
}

/// Configuration for the `clear_tool_uses_20250919` edit.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClearToolUsesEdit {
    clear_at_least_input_tokens: Option<u64>,
    clear_tool_inputs: Option<ClearToolInputs>,
    exclude_tools: Vec<String>,
    keep_tool_uses: Option<u64>,
    trigger: Option<ContextManagementTrigger>,
}

impl ClearToolUsesEdit {
    pub const fn new() -> Self {
        Self {
            clear_at_least_input_tokens: None,
            clear_tool_inputs: None,
            exclude_tools: Vec::new(),
            keep_tool_uses: None,
            trigger: None,
        }
    }

    pub fn with_clear_at_least_input_tokens(mut self, value: u64) -> Self {
        self.clear_at_least_input_tokens = Some(value);
        self
    }

    pub fn with_clear_tool_inputs(mut self, value: ClearToolInputs) -> Self {
        self.clear_tool_inputs = Some(value);
        self
    }

    pub fn with_excluded_tool(mut self, name: impl Into<String>) -> Self {
        self.exclude_tools.push(name.into());
        self
    }

    pub fn with_keep_tool_uses(mut self, value: u64) -> Self {
        self.keep_tool_uses = Some(value);
        self
    }

    pub fn with_trigger(mut self, value: ContextManagementTrigger) -> Self {
        self.trigger = Some(value);
        self
    }

    pub(crate) fn validate(&self) -> Result<(), MessagesCodecError> {
        validate_positive_optional(
            self.clear_at_least_input_tokens,
            "context_management.edits[].clear_at_least.value",
        )?;
        validate_positive_optional(self.keep_tool_uses, "context_management.edits[].keep.value")?;
        validate_trigger(self.trigger)?;
        validate_context_tool_names(
            &self.exclude_tools,
            "context_management.edits[].exclude_tools",
        )?;
        if let Some(ClearToolInputs::Selected(names)) = &self.clear_tool_inputs {
            validate_context_tool_names(names, "context_management.edits[].clear_tool_inputs")?;
        }
        Ok(())
    }
}

/// Configuration for the `clear_thinking_20251015` edit.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClearThinkingEdit {
    keep: Option<ClearThinkingKeep>,
}

impl ClearThinkingEdit {
    pub const fn new() -> Self {
        Self { keep: None }
    }

    pub const fn with_keep(mut self, value: ClearThinkingKeep) -> Self {
        self.keep = Some(value);
        self
    }

    pub(crate) fn validate(&self) -> Result<(), MessagesCodecError> {
        if matches!(self.keep, Some(ClearThinkingKeep::ThinkingTurns(0))) {
            return Err(MessagesCodecError::InvalidOption {
                field: "context_management.edits[].keep.value",
                reason: "must be greater than zero",
            });
        }
        Ok(())
    }
}

/// Configuration for the `compact_20260112` edit.
///
/// Its optional input-token trigger follows the versioned edit's current
/// 50,000-token minimum.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompactionEdit {
    instructions: Option<String>,
    pause_after_compaction: Option<bool>,
    trigger_input_tokens: Option<u64>,
}

impl CompactionEdit {
    pub const fn new() -> Self {
        Self {
            instructions: None,
            pause_after_compaction: None,
            trigger_input_tokens: None,
        }
    }

    pub fn with_instructions(mut self, value: impl Into<String>) -> Self {
        self.instructions = Some(value.into());
        self
    }

    pub const fn with_pause_after_compaction(mut self, value: bool) -> Self {
        self.pause_after_compaction = Some(value);
        self
    }

    pub const fn with_trigger_input_tokens(mut self, value: u64) -> Self {
        self.trigger_input_tokens = Some(value);
        self
    }

    pub(crate) fn validate(&self) -> Result<(), MessagesCodecError> {
        if let Some(instructions) = &self.instructions {
            validate_printable_string(
                instructions,
                "context_management.edits[].instructions",
                MAX_CONTEXT_INSTRUCTIONS_BYTES,
                "must be a non-empty printable value of at most 16 KiB",
            )?;
        }
        if self
            .trigger_input_tokens
            .is_some_and(|value| value < MIN_COMPACTION_TRIGGER_INPUT_TOKENS)
        {
            return Err(MessagesCodecError::InvalidOption {
                field: "context_management.edits[].trigger.value",
                reason: "compact_20260112 requires at least 50000 input tokens",
            });
        }
        Ok(())
    }
}

/// One versioned Anthropic context-management edit.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ContextManagementEdit {
    ClearToolUses(ClearToolUsesEdit),
    ClearThinking(ClearThinkingEdit),
    Compact(CompactionEdit),
}

impl From<ClearToolUsesEdit> for ContextManagementEdit {
    fn from(value: ClearToolUsesEdit) -> Self {
        Self::ClearToolUses(value)
    }
}

impl From<ClearThinkingEdit> for ContextManagementEdit {
    fn from(value: ClearThinkingEdit) -> Self {
        Self::ClearThinking(value)
    }
}

impl From<CompactionEdit> for ContextManagementEdit {
    fn from(value: CompactionEdit) -> Self {
        Self::Compact(value)
    }
}

impl ContextManagementEdit {
    pub(crate) fn validate(&self) -> Result<(), MessagesCodecError> {
        match self {
            Self::ClearToolUses(edit) => edit.validate(),
            Self::ClearThinking(edit) => edit.validate(),
            Self::Compact(edit) => edit.validate(),
        }
    }
}

/// Ordered context-management edits for one Messages request.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ContextManagement {
    edits: Vec<ContextManagementEdit>,
}

impl ContextManagement {
    pub const fn new() -> Self {
        Self { edits: Vec::new() }
    }

    pub fn with_edit(mut self, edit: impl Into<ContextManagementEdit>) -> Self {
        self.edits.push(edit.into());
        self
    }

    pub fn edits(&self) -> &[ContextManagementEdit] {
        &self.edits
    }

    pub(crate) fn validate(&self) -> Result<(), MessagesCodecError> {
        if self.edits.is_empty() || self.edits.len() > MAX_CONTEXT_EDITS {
            return Err(MessagesCodecError::InvalidOption {
                field: "context_management.edits",
                reason: "must contain between one and 16 edits",
            });
        }
        for edit in &self.edits {
            edit.validate()?;
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ContextManagementWire {
    edits: Vec<ContextManagementEditWire>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
enum ContextManagementEditWire {
    #[serde(rename = "clear_tool_uses_20250919")]
    ClearToolUses {
        #[serde(skip_serializing_if = "Option::is_none")]
        clear_at_least: Option<ContextThresholdWire>,
        #[serde(skip_serializing_if = "Option::is_none")]
        clear_tool_inputs: Option<ClearToolInputs>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        exclude_tools: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        keep: Option<ContextThresholdWire>,
        #[serde(skip_serializing_if = "Option::is_none")]
        trigger: Option<ContextThresholdWire>,
    },
    #[serde(rename = "clear_thinking_20251015")]
    ClearThinking {
        #[serde(skip_serializing_if = "Option::is_none")]
        keep: Option<ClearThinkingKeepWire>,
    },
    #[serde(rename = "compact_20260112")]
    Compact {
        #[serde(skip_serializing_if = "Option::is_none")]
        instructions: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pause_after_compaction: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        trigger: Option<ContextThresholdWire>,
    },
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ContextThresholdWire {
    #[serde(rename = "type")]
    kind: String,
    value: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum ClearThinkingKeepWire {
    All(String),
    Threshold(ContextThresholdWire),
}

impl Serialize for ContextManagement {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        ContextManagementWire::from(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ContextManagement {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ContextManagementWire::deserialize(deserializer)?;
        Self::try_from(wire).map_err(D::Error::custom)
    }
}

impl From<&ContextManagement> for ContextManagementWire {
    fn from(value: &ContextManagement) -> Self {
        Self {
            edits: value
                .edits
                .iter()
                .map(ContextManagementEditWire::from)
                .collect(),
        }
    }
}

impl TryFrom<ContextManagementWire> for ContextManagement {
    type Error = &'static str;

    fn try_from(value: ContextManagementWire) -> Result<Self, Self::Error> {
        let edits = value
            .edits
            .into_iter()
            .map(ContextManagementEdit::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { edits })
    }
}

impl From<&ContextManagementEdit> for ContextManagementEditWire {
    fn from(value: &ContextManagementEdit) -> Self {
        match value {
            ContextManagementEdit::ClearToolUses(edit) => Self::ClearToolUses {
                clear_at_least: edit
                    .clear_at_least_input_tokens
                    .map(|value| ContextThresholdWire::new("input_tokens", value)),
                clear_tool_inputs: edit.clear_tool_inputs.clone(),
                exclude_tools: edit.exclude_tools.clone(),
                keep: edit
                    .keep_tool_uses
                    .map(|value| ContextThresholdWire::new("tool_uses", value)),
                trigger: edit.trigger.map(ContextThresholdWire::from),
            },
            ContextManagementEdit::ClearThinking(edit) => Self::ClearThinking {
                keep: edit.keep.map(ClearThinkingKeepWire::from),
            },
            ContextManagementEdit::Compact(edit) => Self::Compact {
                instructions: edit.instructions.clone(),
                pause_after_compaction: edit.pause_after_compaction,
                trigger: edit
                    .trigger_input_tokens
                    .map(|value| ContextThresholdWire::new("input_tokens", value)),
            },
        }
    }
}

impl TryFrom<ContextManagementEditWire> for ContextManagementEdit {
    type Error = &'static str;

    fn try_from(value: ContextManagementEditWire) -> Result<Self, Self::Error> {
        match value {
            ContextManagementEditWire::ClearToolUses {
                clear_at_least,
                clear_tool_inputs,
                exclude_tools,
                keep,
                trigger,
            } => Ok(Self::ClearToolUses(ClearToolUsesEdit {
                clear_at_least_input_tokens: clear_at_least
                    .map(|value| value.require("input_tokens"))
                    .transpose()?,
                clear_tool_inputs,
                exclude_tools,
                keep_tool_uses: keep.map(|value| value.require("tool_uses")).transpose()?,
                trigger: trigger
                    .map(ContextManagementTrigger::try_from)
                    .transpose()?,
            })),
            ContextManagementEditWire::ClearThinking { keep } => {
                Ok(Self::ClearThinking(ClearThinkingEdit {
                    keep: keep.map(ClearThinkingKeep::try_from).transpose()?,
                }))
            }
            ContextManagementEditWire::Compact {
                instructions,
                pause_after_compaction,
                trigger,
            } => Ok(Self::Compact(CompactionEdit {
                instructions,
                pause_after_compaction,
                trigger_input_tokens: trigger
                    .map(|value| value.require("input_tokens"))
                    .transpose()?,
            })),
        }
    }
}

impl ContextThresholdWire {
    fn new(kind: &str, value: u64) -> Self {
        Self {
            kind: kind.to_string(),
            value,
        }
    }

    fn require(self, expected: &'static str) -> Result<u64, &'static str> {
        if self.kind == expected {
            Ok(self.value)
        } else {
            Err("context-management threshold has an invalid type")
        }
    }
}

impl From<ContextManagementTrigger> for ContextThresholdWire {
    fn from(value: ContextManagementTrigger) -> Self {
        match value {
            ContextManagementTrigger::InputTokens(value) => Self::new("input_tokens", value),
            ContextManagementTrigger::ToolUses(value) => Self::new("tool_uses", value),
        }
    }
}

impl TryFrom<ContextThresholdWire> for ContextManagementTrigger {
    type Error = &'static str;

    fn try_from(value: ContextThresholdWire) -> Result<Self, Self::Error> {
        match value.kind.as_str() {
            "input_tokens" => Ok(Self::InputTokens(value.value)),
            "tool_uses" => Ok(Self::ToolUses(value.value)),
            _ => Err("context-management trigger has an invalid type"),
        }
    }
}

impl From<ClearThinkingKeep> for ClearThinkingKeepWire {
    fn from(value: ClearThinkingKeep) -> Self {
        match value {
            ClearThinkingKeep::All => Self::All("all".to_string()),
            ClearThinkingKeep::ThinkingTurns(value) => {
                Self::Threshold(ContextThresholdWire::new("thinking_turns", value))
            }
        }
    }
}

impl TryFrom<ClearThinkingKeepWire> for ClearThinkingKeep {
    type Error = &'static str;

    fn try_from(value: ClearThinkingKeepWire) -> Result<Self, Self::Error> {
        match value {
            ClearThinkingKeepWire::All(value) if value == "all" => Ok(Self::All),
            ClearThinkingKeepWire::All(_) => Err("clear-thinking keep must be all"),
            ClearThinkingKeepWire::Threshold(value) => {
                value.require("thinking_turns").map(Self::ThinkingTurns)
            }
        }
    }
}

/// Secret bearer value used by one remote MCP server.
#[derive(Clone)]
pub struct McpAuthorizationToken(SecretString);

impl McpAuthorizationToken {
    pub fn new(value: impl Into<String>) -> Result<Self, MessagesCodecError> {
        let value = value.into();
        validate_printable_string(
            &value,
            "mcp_servers[].authorization_token",
            MAX_MCP_AUTHORIZATION_TOKEN_BYTES,
            "must be a non-empty printable value of at most 16 KiB",
        )?;
        Ok(Self(SecretString::from(value)))
    }

    pub(crate) fn expose_secret(&self) -> &str {
        self.0.expose_secret()
    }

    fn validate(&self) -> Result<(), MessagesCodecError> {
        validate_printable_string(
            self.0.expose_secret(),
            "mcp_servers[].authorization_token",
            MAX_MCP_AUTHORIZATION_TOKEN_BYTES,
            "must be a non-empty printable value of at most 16 KiB",
        )
    }
}

impl PartialEq for McpAuthorizationToken {
    fn eq(&self, other: &Self) -> bool {
        self.0.expose_secret() == other.0.expose_secret()
    }
}

impl Eq for McpAuthorizationToken {}

impl fmt::Debug for McpAuthorizationToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("McpAuthorizationToken([REDACTED])")
    }
}

/// One remote MCP server available to a Messages request.
#[derive(Clone, PartialEq, Eq)]
pub struct McpServer {
    name: String,
    url: String,
    authorization_token: Option<McpAuthorizationToken>,
}

impl fmt::Debug for McpServer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("McpServer")
            .field("name", &self.name)
            .field("has_authorization_token", &self.has_authorization_token())
            .finish()
    }
}

impl McpServer {
    pub fn new(
        name: impl Into<String>,
        url: impl Into<String>,
    ) -> Result<Self, MessagesCodecError> {
        let value = Self {
            name: name.into(),
            url: url.into(),
            authorization_token: None,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn with_authorization_token(mut self, token: McpAuthorizationToken) -> Self {
        self.authorization_token = Some(token);
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub const fn has_authorization_token(&self) -> bool {
        self.authorization_token.is_some()
    }

    pub(crate) fn validate(&self) -> Result<(), MessagesCodecError> {
        validate_printable_string(
            &self.name,
            "mcp_servers[].name",
            MAX_MCP_SERVER_NAME_BYTES,
            "must be a non-empty printable value of at most 128 bytes",
        )?;
        validate_printable_string(
            &self.url,
            "mcp_servers[].url",
            MAX_MCP_SERVER_URL_BYTES,
            "must be a non-empty printable value of at most 2048 bytes",
        )?;
        let parsed = Url::parse(&self.url).map_err(|_| MessagesCodecError::InvalidOption {
            field: "mcp_servers[].url",
            reason: "must be an absolute HTTPS URL",
        })?;
        if parsed.scheme() != "https" || parsed.host_str().is_none() {
            return Err(MessagesCodecError::InvalidOption {
                field: "mcp_servers[].url",
                reason: "must be an absolute HTTPS URL with a host",
            });
        }
        if !parsed.username().is_empty() || parsed.password().is_some() {
            return Err(MessagesCodecError::InvalidOption {
                field: "mcp_servers[].url",
                reason: "must not contain user information",
            });
        }
        if parsed.fragment().is_some() {
            return Err(MessagesCodecError::InvalidOption {
                field: "mcp_servers[].url",
                reason: "must not contain a fragment",
            });
        }
        if let Some(token) = &self.authorization_token {
            token.validate()?;
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct McpServerWire {
    #[serde(rename = "type")]
    kind: String,
    name: String,
    url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    authorization_token: Option<String>,
}

impl Serialize for McpServer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        McpServerWire {
            kind: "url".to_string(),
            name: self.name.clone(),
            url: self.url.clone(),
            authorization_token: self
                .authorization_token
                .as_ref()
                .map(|value| value.expose_secret().to_string()),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for McpServer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = McpServerWire::deserialize(deserializer)?;
        if wire.kind != "url" {
            return Err(D::Error::custom("MCP server type must be url"));
        }
        let mut server = Self::new(wire.name, wire.url).map_err(D::Error::custom)?;
        if let Some(token) = wire.authorization_token {
            server = server.with_authorization_token(
                McpAuthorizationToken::new(token).map_err(D::Error::custom)?,
            );
        }
        Ok(server)
    }
}

/// Typed `output_config` override for one fallback attempt.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FallbackOutputConfig {
    effort: Option<OutputEffort>,
    format: Option<StructuredOutputSpec>,
}

impl FallbackOutputConfig {
    pub const fn new() -> Self {
        Self {
            effort: None,
            format: None,
        }
    }

    pub const fn with_effort(mut self, effort: OutputEffort) -> Self {
        self.effort = Some(effort);
        self
    }

    pub fn with_format(mut self, format: StructuredOutputSpec) -> Self {
        self.format = Some(format);
        self
    }

    pub const fn effort(&self) -> Option<OutputEffort> {
        self.effort
    }

    pub const fn format(&self) -> Option<&StructuredOutputSpec> {
        self.format.as_ref()
    }

    pub(crate) fn validate(&self) -> Result<(), MessagesCodecError> {
        if let Some(format) = &self.format {
            validate_structured_output(format)?;
        }
        Ok(())
    }
}

/// One explicit server-side fallback attempt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerFallback {
    model: String,
    max_tokens: Option<u64>,
    thinking: Option<ThinkingConfig>,
    output_config: Option<FallbackOutputConfig>,
    speed: Option<InferenceSpeed>,
}

impl ServerFallback {
    pub fn new(model: impl Into<String>) -> Result<Self, MessagesCodecError> {
        let value = Self {
            model: model.into(),
            max_tokens: None,
            thinking: None,
            output_config: None,
            speed: None,
        };
        value.validate(None)?;
        Ok(value)
    }

    pub fn model(&self) -> Result<ModelId, MessagesCodecError> {
        ModelId::new(self.model.clone()).map_err(|_| MessagesCodecError::InvalidOption {
            field: "fallbacks[].model",
            reason: "must be a valid model identifier",
        })
    }

    pub fn model_name(&self) -> &str {
        &self.model
    }

    pub const fn max_tokens(&self) -> Option<u64> {
        self.max_tokens
    }

    pub const fn thinking(&self) -> Option<ThinkingConfig> {
        self.thinking
    }

    pub const fn output_config(&self) -> Option<&FallbackOutputConfig> {
        self.output_config.as_ref()
    }

    pub const fn speed(&self) -> Option<InferenceSpeed> {
        self.speed
    }

    pub const fn with_max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub const fn with_thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self
    }

    pub fn with_output_config(mut self, output_config: FallbackOutputConfig) -> Self {
        self.output_config = Some(output_config);
        self
    }

    pub const fn with_speed(mut self, speed: InferenceSpeed) -> Self {
        self.speed = Some(speed);
        self
    }

    pub(crate) fn validate(
        &self,
        request_max_tokens: Option<u64>,
    ) -> Result<(), MessagesCodecError> {
        self.model()?;
        if self.max_tokens == Some(0) {
            return Err(MessagesCodecError::InvalidOption {
                field: "fallbacks[].max_tokens",
                reason: "must be greater than zero",
            });
        }
        if let Some(output_config) = &self.output_config {
            output_config.validate()?;
        }
        let effective_max_tokens = self.max_tokens.or(request_max_tokens);
        validate_thinking(self.thinking, effective_max_tokens, "fallbacks[].thinking")?;
        Ok(())
    }
}

/// Typed server-side fallback policy.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ServerFallbacks {
    /// Ask Anthropic to choose the fallback chain for the requested model.
    Default,
    /// Try explicit fallback attempts in order.
    Explicit(Vec<ServerFallback>),
}

impl Serialize for ServerFallbacks {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Default => serializer.serialize_str("default"),
            Self::Explicit(fallbacks) => fallbacks.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for ServerFallbacks {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::String(value) if value.eq_ignore_ascii_case("default") => Ok(Self::Default),
            Value::Array(_) => {
                let fallbacks = serde_json::from_value::<Vec<ServerFallback>>(value)
                    .map_err(serde::de::Error::custom)?;
                Ok(Self::Explicit(fallbacks))
            }
            _ => Err(serde::de::Error::custom(
                "fallbacks must be \"default\" or an array of fallback objects",
            )),
        }
    }
}

impl ServerFallbacks {
    pub fn explicit(fallbacks: impl Into<Vec<ServerFallback>>) -> Result<Self, MessagesCodecError> {
        let value = Self::Explicit(fallbacks.into());
        value.validate(None)?;
        Ok(value)
    }

    pub(crate) fn validate(
        &self,
        request_max_tokens: Option<u64>,
    ) -> Result<(), MessagesCodecError> {
        let Self::Explicit(fallbacks) = self else {
            return Ok(());
        };
        if fallbacks.is_empty() {
            return Err(MessagesCodecError::InvalidOption {
                field: "fallbacks",
                reason: "explicit fallback chain must not be empty",
            });
        }
        if fallbacks.len() > MAX_FALLBACKS {
            return Err(MessagesCodecError::InvalidOption {
                field: "fallbacks",
                reason: "explicit fallback chain exceeds three entries",
            });
        }
        let mut models = std::collections::BTreeSet::new();
        for fallback in fallbacks {
            if !models.insert(fallback.model_name()) {
                return Err(MessagesCodecError::InvalidOption {
                    field: "fallbacks",
                    reason: "explicit fallback chain must not repeat a model",
                });
            }
            fallback.validate(request_max_tokens)?;
        }
        Ok(())
    }

    pub(crate) fn validate_primary_model(
        &self,
        primary_model: &ModelId,
    ) -> Result<(), MessagesCodecError> {
        if let Self::Explicit(fallbacks) = self
            && fallbacks
                .iter()
                .any(|fallback| fallback.model_name() == primary_model.as_str())
        {
            return Err(MessagesCodecError::InvalidOption {
                field: "fallbacks",
                reason: "fallback models must differ from the requested model",
            });
        }
        Ok(())
    }
}

/// A caller that is allowed to invoke an Anthropic-defined tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ToolCaller {
    Direct,
    CodeExecution20250825,
    CodeExecution20260120,
    CodeExecution20260521,
}

impl ToolCaller {
    pub(crate) const fn as_wire_str(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::CodeExecution20250825 => "code_execution_20250825",
            Self::CodeExecution20260120 => "code_execution_20260120",
            Self::CodeExecution20260521 => "code_execution_20260521",
        }
    }
}

/// User location hint for the Anthropic web-search tool.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UserLocation {
    city: Option<String>,
    country: Option<String>,
    region: Option<String>,
    timezone: Option<String>,
}

impl UserLocation {
    pub fn new() -> Self {
        Self {
            city: None,
            country: None,
            region: None,
            timezone: None,
        }
    }

    pub fn with_city(mut self, city: impl Into<String>) -> Self {
        self.city = Some(city.into());
        self
    }

    pub fn with_country(mut self, country: impl Into<String>) -> Self {
        self.country = Some(country.into());
        self
    }

    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    pub fn with_timezone(mut self, timezone: impl Into<String>) -> Self {
        self.timezone = Some(timezone.into());
        self
    }

    pub fn city(&self) -> Option<&str> {
        self.city.as_deref()
    }

    pub fn country(&self) -> Option<&str> {
        self.country.as_deref()
    }

    pub fn region(&self) -> Option<&str> {
        self.region.as_deref()
    }

    pub fn timezone(&self) -> Option<&str> {
        self.timezone.as_deref()
    }

    pub(crate) fn validate(&self) -> Result<(), MessagesCodecError> {
        for value in [
            self.city.as_deref(),
            self.country.as_deref(),
            self.region.as_deref(),
            self.timezone.as_deref(),
        ] {
            if let Some(value) = value
                && (value.trim().is_empty()
                    || value.len() > 128
                    || value.chars().any(char::is_control))
            {
                return Err(MessagesCodecError::InvalidOption {
                    field: "tools.web_search.user_location",
                    reason: "location fields must be 1..=128 bytes and contain no control characters",
                });
            }
        }
        if let Some(country) = &self.country
            && (country.len() != 2 || !country.bytes().all(|byte| byte.is_ascii_alphabetic()))
        {
            return Err(MessagesCodecError::InvalidOption {
                field: "tools.web_search.user_location.country",
                reason: "country must be a two-letter ISO code",
            });
        }
        Ok(())
    }
}

/// Response inclusion policy for Anthropic web tools nested in code execution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ResponseInclusion {
    #[default]
    Full,
    Excluded,
}

impl ResponseInclusion {
    pub(crate) const fn as_wire_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Excluded => "excluded",
        }
    }
}

/// Typed web-search tool settings.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WebSearchToolOptions {
    allowed_domains: Option<Vec<String>>,
    blocked_domains: Option<Vec<String>>,
    max_uses: Option<u32>,
    response_inclusion: Option<ResponseInclusion>,
    user_location: Option<UserLocation>,
}

impl WebSearchToolOptions {
    pub fn with_allowed_domains(
        mut self,
        domains: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.allowed_domains = Some(domains.into_iter().map(Into::into).collect());
        self
    }

    pub fn with_blocked_domains(
        mut self,
        domains: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.blocked_domains = Some(domains.into_iter().map(Into::into).collect());
        self
    }

    pub const fn with_max_uses(mut self, max_uses: u32) -> Self {
        self.max_uses = Some(max_uses);
        self
    }

    pub const fn with_response_inclusion(mut self, response_inclusion: ResponseInclusion) -> Self {
        self.response_inclusion = Some(response_inclusion);
        self
    }

    pub fn with_user_location(mut self, user_location: UserLocation) -> Self {
        self.user_location = Some(user_location);
        self
    }

    pub(crate) fn validate(&self) -> Result<(), MessagesCodecError> {
        validate_domains(&self.allowed_domains, "tools.web_search.allowed_domains")?;
        validate_domains(&self.blocked_domains, "tools.web_search.blocked_domains")?;
        if self.allowed_domains.is_some() && self.blocked_domains.is_some() {
            return Err(MessagesCodecError::InvalidOption {
                field: "tools.web_search",
                reason: "allowed_domains and blocked_domains are mutually exclusive",
            });
        }
        if self.max_uses == Some(0) {
            return Err(MessagesCodecError::InvalidOption {
                field: "tools.web_search.max_uses",
                reason: "must be greater than zero",
            });
        }
        if let Some(user_location) = &self.user_location {
            user_location.validate()?;
        }
        Ok(())
    }

    pub fn allowed_domains(&self) -> Option<&[String]> {
        self.allowed_domains.as_deref()
    }

    pub fn blocked_domains(&self) -> Option<&[String]> {
        self.blocked_domains.as_deref()
    }

    pub const fn max_uses(&self) -> Option<u32> {
        self.max_uses
    }

    pub const fn response_inclusion(&self) -> Option<ResponseInclusion> {
        self.response_inclusion
    }

    pub fn user_location(&self) -> Option<&UserLocation> {
        self.user_location.as_ref()
    }
}

/// Typed web-fetch tool settings.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WebFetchToolOptions {
    allowed_domains: Option<Vec<String>>,
    blocked_domains: Option<Vec<String>>,
    citations: Option<bool>,
    max_content_tokens: Option<u32>,
    max_uses: Option<u32>,
    response_inclusion: Option<ResponseInclusion>,
    use_cache: Option<bool>,
}

impl WebFetchToolOptions {
    pub fn with_allowed_domains(
        mut self,
        domains: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.allowed_domains = Some(domains.into_iter().map(Into::into).collect());
        self
    }

    pub fn with_blocked_domains(
        mut self,
        domains: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.blocked_domains = Some(domains.into_iter().map(Into::into).collect());
        self
    }

    pub const fn with_citations(mut self, enabled: bool) -> Self {
        self.citations = Some(enabled);
        self
    }

    pub const fn with_max_content_tokens(mut self, max_content_tokens: u32) -> Self {
        self.max_content_tokens = Some(max_content_tokens);
        self
    }

    pub const fn with_max_uses(mut self, max_uses: u32) -> Self {
        self.max_uses = Some(max_uses);
        self
    }

    pub const fn with_response_inclusion(mut self, response_inclusion: ResponseInclusion) -> Self {
        self.response_inclusion = Some(response_inclusion);
        self
    }

    pub const fn with_use_cache(mut self, use_cache: bool) -> Self {
        self.use_cache = Some(use_cache);
        self
    }

    pub(crate) fn validate(&self) -> Result<(), MessagesCodecError> {
        validate_domains(&self.allowed_domains, "tools.web_fetch.allowed_domains")?;
        validate_domains(&self.blocked_domains, "tools.web_fetch.blocked_domains")?;
        if self.allowed_domains.is_some() && self.blocked_domains.is_some() {
            return Err(MessagesCodecError::InvalidOption {
                field: "tools.web_fetch",
                reason: "allowed_domains and blocked_domains are mutually exclusive",
            });
        }
        if self.max_content_tokens == Some(0) || self.max_uses == Some(0) {
            return Err(MessagesCodecError::InvalidOption {
                field: "tools.web_fetch",
                reason: "token and use limits must be greater than zero",
            });
        }
        Ok(())
    }

    pub fn allowed_domains(&self) -> Option<&[String]> {
        self.allowed_domains.as_deref()
    }

    pub fn blocked_domains(&self) -> Option<&[String]> {
        self.blocked_domains.as_deref()
    }

    pub const fn citations(&self) -> Option<bool> {
        self.citations
    }

    pub const fn max_content_tokens(&self) -> Option<u32> {
        self.max_content_tokens
    }

    pub const fn max_uses(&self) -> Option<u32> {
        self.max_uses
    }

    pub const fn response_inclusion(&self) -> Option<ResponseInclusion> {
        self.response_inclusion
    }

    pub const fn use_cache(&self) -> Option<bool> {
        self.use_cache
    }
}

/// Typed advisor tool settings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdvisorToolOptions {
    model: String,
    max_tokens: Option<u64>,
    max_uses: Option<u32>,
    caching: Option<CacheControl>,
}

impl AdvisorToolOptions {
    pub fn new(model: impl Into<String>) -> Result<Self, MessagesCodecError> {
        let value = Self {
            model: model.into(),
            max_tokens: None,
            max_uses: None,
            caching: None,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn model(&self) -> Result<ModelId, MessagesCodecError> {
        ModelId::new(self.model.clone()).map_err(|_| MessagesCodecError::InvalidOption {
            field: "tools.advisor.model",
            reason: "must be a valid model identifier",
        })
    }

    pub fn model_name(&self) -> &str {
        &self.model
    }

    pub const fn with_max_uses(mut self, max_uses: u32) -> Self {
        self.max_uses = Some(max_uses);
        self
    }

    pub const fn with_max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub const fn with_caching(mut self, caching: CacheControl) -> Self {
        self.caching = Some(caching);
        self
    }

    pub(crate) fn validate(&self) -> Result<(), MessagesCodecError> {
        self.model()?;
        if self.max_tokens.is_some_and(|max_tokens| max_tokens < 1_024) || self.max_uses == Some(0)
        {
            return Err(MessagesCodecError::InvalidOption {
                field: "tools.advisor",
                reason: "max_tokens must be at least 1024 and max_uses must be greater than zero",
            });
        }
        Ok(())
    }

    pub const fn max_tokens(&self) -> Option<u64> {
        self.max_tokens
    }

    pub const fn max_uses(&self) -> Option<u32> {
        self.max_uses
    }

    pub const fn caching(&self) -> Option<CacheControl> {
        self.caching
    }
}

/// Typed MCP per-tool configuration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McpToolConfig {
    enabled: Option<bool>,
    defer_loading: Option<bool>,
}

impl McpToolConfig {
    pub const fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = Some(enabled);
        self
    }

    pub const fn with_defer_loading(mut self, defer_loading: bool) -> Self {
        self.defer_loading = Some(defer_loading);
        self
    }

    pub const fn enabled(&self) -> Option<bool> {
        self.enabled
    }

    pub const fn defer_loading(&self) -> Option<bool> {
        self.defer_loading
    }
}

/// Typed MCP toolset settings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McpToolsetOptions {
    anchor_name: String,
    server_name: String,
    configs: BTreeMap<String, McpToolConfig>,
    default_config: Option<McpToolConfig>,
}

impl McpToolsetOptions {
    pub fn new(server_name: impl Into<String>) -> Result<Self, MessagesCodecError> {
        let server_name = server_name.into();
        let value = Self {
            anchor_name: server_name.clone(),
            server_name,
            configs: BTreeMap::new(),
            default_config: None,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn with_anchor(
        anchor_name: impl Into<String>,
        server_name: impl Into<String>,
    ) -> Result<Self, MessagesCodecError> {
        let value = Self {
            anchor_name: anchor_name.into(),
            server_name: server_name.into(),
            configs: BTreeMap::new(),
            default_config: None,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn anchor_name(&self) -> &str {
        &self.anchor_name
    }

    pub fn server_name(&self) -> &str {
        &self.server_name
    }

    pub fn with_config(mut self, tool_name: impl Into<String>, config: McpToolConfig) -> Self {
        self.configs.insert(tool_name.into(), config);
        self
    }

    pub const fn with_default_config(mut self, config: McpToolConfig) -> Self {
        self.default_config = Some(config);
        self
    }

    pub fn configs(&self) -> &BTreeMap<String, McpToolConfig> {
        &self.configs
    }

    pub const fn default_config(&self) -> Option<McpToolConfig> {
        self.default_config
    }

    pub(crate) fn validate(&self) -> Result<(), MessagesCodecError> {
        validate_anchor_name(&self.anchor_name, "tools.mcp_toolset.anchor_name")?;
        validate_bounded_label(&self.server_name, "tools.mcp_toolset.mcp_server_name")?;
        if self.configs.len() > MAX_TOOL_CONFIGS {
            return Err(MessagesCodecError::InvalidOption {
                field: "tools.mcp_toolset.configs",
                reason: "must contain at most 256 tool overrides",
            });
        }
        for name in self.configs.keys() {
            validate_bounded_label(name, "tools.mcp_toolset.configs")?;
        }
        Ok(())
    }
}

/// Typed text-editor tool settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextEditorToolOptions {
    max_characters: Option<u32>,
}

impl TextEditorToolOptions {
    pub const fn new() -> Self {
        Self {
            max_characters: None,
        }
    }

    pub const fn with_max_characters(mut self, max_characters: u32) -> Self {
        self.max_characters = Some(max_characters);
        self
    }

    pub const fn max_characters(&self) -> Option<u32> {
        self.max_characters
    }
}

/// Typed computer-use tool settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComputerToolOptions {
    display_width_px: u32,
    display_height_px: u32,
    display_number: Option<u32>,
    enable_zoom: bool,
}

impl ComputerToolOptions {
    pub const fn new(display_width_px: u32, display_height_px: u32) -> Self {
        Self {
            display_width_px,
            display_height_px,
            display_number: None,
            enable_zoom: false,
        }
    }

    pub const fn with_display_number(mut self, display_number: u32) -> Self {
        self.display_number = Some(display_number);
        self
    }

    pub const fn with_enable_zoom(mut self, enable_zoom: bool) -> Self {
        self.enable_zoom = enable_zoom;
        self
    }

    pub const fn display_width_px(&self) -> u32 {
        self.display_width_px
    }

    pub const fn display_height_px(&self) -> u32 {
        self.display_height_px
    }

    pub const fn display_number(&self) -> Option<u32> {
        self.display_number
    }

    pub const fn enable_zoom(&self) -> bool {
        self.enable_zoom
    }

    pub(crate) fn validate(&self) -> Result<(), MessagesCodecError> {
        if self.display_width_px == 0 || self.display_height_px == 0 {
            return Err(MessagesCodecError::InvalidOption {
                field: "tools.computer",
                reason: "display dimensions must be greater than zero",
            });
        }
        Ok(())
    }
}

impl Default for TextEditorToolOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// Anthropic-defined tool families supported by the canonical Messages codec.
///
/// Variants include both provider-executed server tools and provider-defined
/// client tools. Execution ownership is part of each tool contract rather than
/// implied by this enum.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AnthropicTool {
    WebSearch20260318(WebSearchToolOptions),
    WebFetch20260318(WebFetchToolOptions),
    CodeExecution20260521,
    Advisor20260301(AdvisorToolOptions),
    ToolSearchRegex20251119,
    ToolSearchBm25V20251119,
    McpToolset(McpToolsetOptions),
    Memory20250818,
    Bash20250124,
    TextEditor20250728(TextEditorToolOptions),
    Computer20251124(ComputerToolOptions),
}

impl AnthropicTool {
    pub const fn web_search_20260318(options: WebSearchToolOptions) -> Self {
        Self::WebSearch20260318(options)
    }

    pub const fn web_fetch_20260318(options: WebFetchToolOptions) -> Self {
        Self::WebFetch20260318(options)
    }

    pub fn advisor_20260301(options: AdvisorToolOptions) -> Self {
        Self::Advisor20260301(options)
    }

    pub fn mcp_toolset(options: McpToolsetOptions) -> Self {
        Self::McpToolset(options)
    }

    pub const fn text_editor_20250728(options: TextEditorToolOptions) -> Self {
        Self::TextEditor20250728(options)
    }

    pub const fn computer_20251124(options: ComputerToolOptions) -> Self {
        Self::Computer20251124(options)
    }

    pub fn canonical_name(&self) -> &str {
        match self {
            Self::WebSearch20260318(_) => "web_search",
            Self::WebFetch20260318(_) => "web_fetch",
            Self::CodeExecution20260521 => "code_execution",
            Self::Advisor20260301(_) => "advisor",
            Self::ToolSearchRegex20251119 => "tool_search_tool_regex",
            Self::ToolSearchBm25V20251119 => "tool_search_tool_bm25",
            Self::McpToolset(options) => options.anchor_name(),
            Self::Memory20250818 => "memory",
            Self::Bash20250124 => "bash",
            Self::TextEditor20250728(_) => "str_replace_based_edit_tool",
            Self::Computer20251124(_) => "computer",
        }
    }

    pub(crate) const fn wire_type(&self) -> &'static str {
        match self {
            Self::WebSearch20260318(_) => "web_search_20260318",
            Self::WebFetch20260318(_) => "web_fetch_20260318",
            Self::CodeExecution20260521 => "code_execution_20260521",
            Self::Advisor20260301(_) => "advisor_20260301",
            Self::ToolSearchRegex20251119 => "tool_search_tool_regex_20251119",
            Self::ToolSearchBm25V20251119 => "tool_search_tool_bm25_20251119",
            Self::McpToolset(_) => "mcp_toolset",
            Self::Memory20250818 => "memory_20250818",
            Self::Bash20250124 => "bash_20250124",
            Self::TextEditor20250728(_) => "text_editor_20250728",
            Self::Computer20251124(_) => "computer_20251124",
        }
    }

    pub(crate) fn validate(&self) -> Result<(), MessagesCodecError> {
        match self {
            Self::WebSearch20260318(options) => options.validate(),
            Self::WebFetch20260318(options) => options.validate(),
            Self::Advisor20260301(options) => options.validate(),
            Self::McpToolset(options) => options.validate(),
            Self::TextEditor20250728(options) => {
                if options.max_characters() == Some(0) {
                    Err(MessagesCodecError::InvalidOption {
                        field: "tools.text_editor.max_characters",
                        reason: "must be greater than zero",
                    })
                } else {
                    Ok(())
                }
            }
            Self::Computer20251124(options) => options.validate(),
            Self::CodeExecution20260521
            | Self::ToolSearchRegex20251119
            | Self::ToolSearchBm25V20251119
            | Self::Memory20250818
            | Self::Bash20250124 => Ok(()),
        }
    }
}

/// Request metadata accepted by Anthropic Messages.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MessagesMetadata {
    user_id: String,
}

impl MessagesMetadata {
    pub fn new(user_id: impl Into<String>) -> Result<Self, MessagesCodecError> {
        let value = Self {
            user_id: user_id.into(),
        };
        value.validate()?;
        Ok(value)
    }

    pub fn user_id(&self) -> &str {
        &self.user_id
    }

    fn validate(&self) -> Result<(), MessagesCodecError> {
        if self.user_id.trim().is_empty()
            || self.user_id.len() > 256
            || self.user_id.chars().any(char::is_control)
        {
            return Err(MessagesCodecError::InvalidOption {
                field: "metadata.user_id",
                reason: "must be 1..=256 bytes and contain no control characters",
            });
        }
        Ok(())
    }
}

/// Protocol-owned shaping for one Anthropic token-count request.
#[derive(Debug, Clone, Default)]
pub struct MessagesTokenCountOptions {
    pub(crate) cache_control: Option<CacheControl>,
    pub(crate) thinking: Option<ThinkingConfig>,
    pub(crate) output_effort: Option<OutputEffort>,
    pub(crate) task_budget: Option<TokenTaskBudget>,
    pub(crate) speed: Option<InferenceSpeed>,
    pub(crate) context_management: Option<ContextManagement>,
    pub(crate) mcp_servers: Option<Vec<McpServer>>,
}

impl MessagesTokenCountOptions {
    pub const fn new() -> Self {
        Self {
            cache_control: None,
            thinking: None,
            output_effort: None,
            task_budget: None,
            speed: None,
            context_management: None,
            mcp_servers: None,
        }
    }

    pub const fn cache_control(&self) -> Option<CacheControl> {
        self.cache_control
    }

    pub const fn thinking(&self) -> Option<ThinkingConfig> {
        self.thinking
    }

    pub const fn output_effort(&self) -> Option<OutputEffort> {
        self.output_effort
    }

    pub const fn task_budget(&self) -> Option<TokenTaskBudget> {
        self.task_budget
    }

    pub const fn speed(&self) -> Option<InferenceSpeed> {
        self.speed
    }

    pub fn context_management(&self) -> Option<&ContextManagement> {
        self.context_management.as_ref()
    }

    pub fn mcp_servers(&self) -> Option<&[McpServer]> {
        self.mcp_servers.as_deref()
    }

    pub const fn with_cache_control(mut self, cache_control: CacheControl) -> Self {
        self.cache_control = Some(cache_control);
        self
    }

    pub const fn with_thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self
    }

    pub const fn with_output_effort(mut self, output_effort: OutputEffort) -> Self {
        self.output_effort = Some(output_effort);
        self
    }

    pub const fn with_task_budget(mut self, task_budget: TokenTaskBudget) -> Self {
        self.task_budget = Some(task_budget);
        self
    }

    pub const fn with_speed(mut self, speed: InferenceSpeed) -> Self {
        self.speed = Some(speed);
        self
    }

    pub fn with_context_management(mut self, context_management: ContextManagement) -> Self {
        self.context_management = Some(context_management);
        self
    }

    pub fn with_mcp_servers(mut self, mcp_servers: Vec<McpServer>) -> Self {
        self.mcp_servers = Some(mcp_servers);
        self
    }

    pub fn validate(&self, request: &LanguageRequest) -> Result<(), MessagesCodecError> {
        validate_thinking(self.thinking, None, "thinking")?;
        if let Some(task_budget) = self.task_budget {
            task_budget.validate()?;
        }
        if let Some(context_management) = &self.context_management {
            context_management.validate()?;
        }
        validate_mcp_servers(self.mcp_servers.as_deref())?;
        if request.tools.len() > MAX_TOOL_LIST {
            return Err(MessagesCodecError::InvalidOption {
                field: "tools",
                reason: "must contain at most 256 tools",
            });
        }
        Ok(())
    }
}

/// Protocol-owned shaping for one Anthropic Messages create request.
#[derive(Debug, Clone, Default)]
pub struct MessagesRequestOptions {
    pub(crate) stream: bool,
    pub(crate) metadata: Option<MessagesMetadata>,
    pub(crate) thinking: Option<ThinkingConfig>,
    pub(crate) output_effort: Option<OutputEffort>,
    pub(crate) task_budget: Option<TokenTaskBudget>,
    pub(crate) fallbacks: Option<ServerFallbacks>,
    pub(crate) top_k: Option<u64>,
    pub(crate) service_tier: Option<MessagesServiceTierPreference>,
    pub(crate) cache_control: Option<CacheControl>,
    pub(crate) speed: Option<InferenceSpeed>,
    pub(crate) inference_geo: Option<InferenceGeo>,
    pub(crate) container: Option<MessagesContainer>,
    pub(crate) context_management: Option<ContextManagement>,
    pub(crate) mcp_servers: Option<Vec<McpServer>>,
    pub(crate) extra: BTreeMap<String, Value>,
}

impl MessagesRequestOptions {
    pub fn new(stream: bool) -> Self {
        Self {
            stream,
            ..Self::default()
        }
    }

    pub const fn stream(&self) -> bool {
        self.stream
    }

    pub fn metadata(&self) -> Option<&MessagesMetadata> {
        self.metadata.as_ref()
    }

    pub const fn thinking(&self) -> Option<ThinkingConfig> {
        self.thinking
    }

    pub const fn output_effort(&self) -> Option<OutputEffort> {
        self.output_effort
    }

    pub const fn task_budget(&self) -> Option<TokenTaskBudget> {
        self.task_budget
    }

    pub fn fallbacks(&self) -> Option<&ServerFallbacks> {
        self.fallbacks.as_ref()
    }

    pub const fn top_k(&self) -> Option<u64> {
        self.top_k
    }

    pub const fn service_tier(&self) -> Option<MessagesServiceTierPreference> {
        self.service_tier
    }

    pub const fn cache_control(&self) -> Option<CacheControl> {
        self.cache_control
    }

    pub const fn speed(&self) -> Option<InferenceSpeed> {
        self.speed
    }

    pub const fn inference_geo(&self) -> Option<InferenceGeo> {
        self.inference_geo
    }

    pub fn container(&self) -> Option<&MessagesContainer> {
        self.container.as_ref()
    }

    pub fn context_management(&self) -> Option<&ContextManagement> {
        self.context_management.as_ref()
    }

    pub fn mcp_servers(&self) -> Option<&[McpServer]> {
        self.mcp_servers.as_deref()
    }

    pub fn extra(&self) -> &BTreeMap<String, Value> {
        &self.extra
    }

    pub fn with_metadata(mut self, metadata: MessagesMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    pub fn with_thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self
    }

    pub const fn with_output_effort(mut self, effort: OutputEffort) -> Self {
        self.output_effort = Some(effort);
        self
    }

    pub const fn with_task_budget(mut self, task_budget: TokenTaskBudget) -> Self {
        self.task_budget = Some(task_budget);
        self
    }

    pub fn with_fallbacks(mut self, fallbacks: ServerFallbacks) -> Self {
        self.fallbacks = Some(fallbacks);
        self
    }

    pub const fn with_top_k(mut self, top_k: u64) -> Self {
        self.top_k = Some(top_k);
        self
    }

    pub const fn with_service_tier(mut self, service_tier: MessagesServiceTierPreference) -> Self {
        self.service_tier = Some(service_tier);
        self
    }

    pub const fn with_cache_control(mut self, cache_control: CacheControl) -> Self {
        self.cache_control = Some(cache_control);
        self
    }

    pub const fn with_speed(mut self, speed: InferenceSpeed) -> Self {
        self.speed = Some(speed);
        self
    }

    pub const fn with_inference_geo(mut self, inference_geo: InferenceGeo) -> Self {
        self.inference_geo = Some(inference_geo);
        self
    }

    pub fn with_container(mut self, container: MessagesContainer) -> Self {
        self.container = Some(container);
        self
    }

    pub fn with_context_management(mut self, context_management: ContextManagement) -> Self {
        self.context_management = Some(context_management);
        self
    }

    pub fn with_mcp_servers(mut self, mcp_servers: Vec<McpServer>) -> Self {
        self.mcp_servers = Some(mcp_servers);
        self
    }

    pub fn with_extra(mut self, extra: BTreeMap<String, Value>) -> Self {
        self.extra = extra;
        self
    }

    pub fn validate(&self, request: &LanguageRequest) -> Result<(), MessagesCodecError> {
        if let Some(metadata) = &self.metadata {
            metadata.validate()?;
        }
        if request.generation.max_output_tokens == Some(0) {
            if self.stream {
                return Err(MessagesCodecError::InvalidOption {
                    field: "max_output_tokens",
                    reason: "zero-token Messages requests cannot stream",
                });
            }
            if matches!(
                self.thinking,
                Some(ThinkingConfig::Enabled { .. } | ThinkingConfig::Adaptive { .. })
            ) {
                return Err(MessagesCodecError::InvalidOption {
                    field: "thinking",
                    reason: "enabled or adaptive thinking requires a positive max_tokens value",
                });
            }
            if request.structured_output.is_some() {
                return Err(MessagesCodecError::InvalidOption {
                    field: "structured_output",
                    reason: "structured output requires a positive max_tokens value",
                });
            }
            if matches!(
                request.tool_choice.as_ref(),
                Some(ToolChoice::Required | ToolChoice::Named { .. })
            ) {
                return Err(MessagesCodecError::InvalidOption {
                    field: "tool_choice",
                    reason: "required tool use requires a positive max_tokens value",
                });
            }
        }
        validate_thinking(
            self.thinking,
            request.generation.max_output_tokens,
            "thinking",
        )?;
        if let Some(fallbacks) = &self.fallbacks {
            fallbacks.validate(request.generation.max_output_tokens)?;
        }
        if let Some(task_budget) = self.task_budget {
            task_budget.validate()?;
        }
        if let Some(container) = &self.container {
            container.validate()?;
        }
        if let Some(context_management) = &self.context_management {
            context_management.validate()?;
        }
        validate_mcp_servers(self.mcp_servers.as_deref())?;
        if self.top_k == Some(0) {
            return Err(MessagesCodecError::InvalidOption {
                field: "top_k",
                reason: "must be greater than zero",
            });
        }
        if request.tools.len() > MAX_TOOL_LIST {
            return Err(MessagesCodecError::InvalidOption {
                field: "tools",
                reason: "must contain at most 256 tools",
            });
        }
        validate_extra(&self.extra)
    }
}

fn validate_mcp_servers(mcp_servers: Option<&[McpServer]>) -> Result<(), MessagesCodecError> {
    let Some(mcp_servers) = mcp_servers else {
        return Ok(());
    };
    if mcp_servers.is_empty() || mcp_servers.len() > MAX_MCP_SERVERS {
        return Err(MessagesCodecError::InvalidOption {
            field: "mcp_servers",
            reason: "must contain between one and 20 servers",
        });
    }
    for server in mcp_servers {
        server.validate()?;
    }
    Ok(())
}

fn validate_thinking(
    thinking: Option<ThinkingConfig>,
    max_tokens: Option<u64>,
    field: &'static str,
) -> Result<(), MessagesCodecError> {
    if let Some(ThinkingConfig::Enabled { budget_tokens, .. }) = thinking {
        if budget_tokens < 1_024 {
            return Err(MessagesCodecError::InvalidOption {
                field,
                reason: "enabled thinking budget must be at least 1024",
            });
        }
        if max_tokens.is_some_and(|maximum| budget_tokens >= maximum) {
            return Err(MessagesCodecError::InvalidOption {
                field,
                reason: "enabled thinking budget must be less than max_tokens",
            });
        }
    }
    Ok(())
}

fn validate_structured_output(output: &StructuredOutputSpec) -> Result<(), MessagesCodecError> {
    if !output.strict {
        return Err(MessagesCodecError::Unsupported {
            feature: "non-strict structured output",
        });
    }
    if !output.schema.is_object() {
        return Err(MessagesCodecError::Unsupported {
            feature: "boolean structured-output schemas",
        });
    }
    Ok(())
}

fn validate_printable_string(
    value: &str,
    field: &'static str,
    maximum_bytes: usize,
    reason: &'static str,
) -> Result<(), MessagesCodecError> {
    if value.trim().is_empty() || value.len() > maximum_bytes || value.chars().any(char::is_control)
    {
        return Err(MessagesCodecError::InvalidOption { field, reason });
    }
    Ok(())
}

fn validate_container_id(value: &str) -> Result<(), MessagesCodecError> {
    validate_printable_string(
        value,
        "container.id",
        MAX_CONTAINER_ID_BYTES,
        "must be a non-empty printable value of at most 256 bytes",
    )
}

fn validate_positive_optional(
    value: Option<u64>,
    field: &'static str,
) -> Result<(), MessagesCodecError> {
    if value == Some(0) {
        return Err(MessagesCodecError::InvalidOption {
            field,
            reason: "must be greater than zero",
        });
    }
    Ok(())
}

fn validate_trigger(trigger: Option<ContextManagementTrigger>) -> Result<(), MessagesCodecError> {
    if matches!(
        trigger,
        Some(ContextManagementTrigger::InputTokens(0) | ContextManagementTrigger::ToolUses(0))
    ) {
        return Err(MessagesCodecError::InvalidOption {
            field: "context_management.edits[].trigger.value",
            reason: "must be greater than zero",
        });
    }
    Ok(())
}

fn validate_context_tool_names(
    names: &[String],
    field: &'static str,
) -> Result<(), MessagesCodecError> {
    if names.len() > MAX_CONTEXT_TOOL_NAMES {
        return Err(MessagesCodecError::InvalidOption {
            field,
            reason: "must contain at most 256 tool names",
        });
    }
    for name in names {
        validate_printable_string(
            name,
            field,
            MAX_TOOL_NAME_BYTES,
            "tool names must be non-empty printable values of at most 128 bytes",
        )?;
    }
    Ok(())
}

fn validate_domains(
    domains: &Option<Vec<String>>,
    field: &'static str,
) -> Result<(), MessagesCodecError> {
    let Some(domains) = domains else {
        return Ok(());
    };
    if domains.len() > MAX_DOMAIN_LIST {
        return Err(MessagesCodecError::InvalidOption {
            field,
            reason: "domain list exceeds 256 entries",
        });
    }
    for domain in domains {
        if domain.trim().is_empty()
            || domain.len() > MAX_DOMAIN_BYTES
            || domain
                .chars()
                .any(|character| character.is_control() || character.is_whitespace())
            || domain
                .chars()
                .any(|character| matches!(character, '/' | ':' | '?' | '#' | '@'))
        {
            return Err(MessagesCodecError::InvalidOption {
                field,
                reason: "domains must be bounded host names without URL delimiters",
            });
        }
    }
    Ok(())
}

fn validate_bounded_label(name: &str, field: &'static str) -> Result<(), MessagesCodecError> {
    if name.trim().is_empty()
        || name.len() > MAX_TOOL_NAME_BYTES
        || name.chars().any(char::is_control)
    {
        return Err(MessagesCodecError::InvalidOption {
            field,
            reason: "names must be 1..=128 bytes and contain no control characters",
        });
    }
    Ok(())
}

fn validate_anchor_name(name: &str, field: &'static str) -> Result<(), MessagesCodecError> {
    if name.is_empty()
        || name.len() > MAX_TOOL_NAME_BYTES
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(MessagesCodecError::InvalidOption {
            field,
            reason: "anchor names must be 1..=128 ASCII letters, digits, '-' or '_'",
        });
    }
    Ok(())
}

pub(crate) fn validate_tool_node_options(
    options: &ToolNodeOptions,
    anthropic_tool: Option<&AnthropicTool>,
) -> Result<(), MessagesCodecError> {
    let callers = options.allowed_callers();
    if callers.len() > MAX_ALLOWED_CALLERS {
        return Err(MessagesCodecError::InvalidOption {
            field: "tools.allowed_callers",
            reason: "must contain at most four callers",
        });
    }
    let mut seen = std::collections::BTreeSet::new();
    for caller in callers {
        if !seen.insert(caller.as_wire_str()) {
            return Err(MessagesCodecError::InvalidOption {
                field: "tools.allowed_callers",
                reason: "must not contain duplicate callers",
            });
        }
    }
    if matches!(anthropic_tool, Some(AnthropicTool::McpToolset(_)))
        && (!callers.is_empty() || options.strict().is_some() || options.defer_loading().is_some())
    {
        return Err(MessagesCodecError::Unsupported {
            feature: "allowed_callers, strict, or defer_loading on MCP toolsets",
        });
    }
    if let Some(anthropic_tool) = anthropic_tool {
        anthropic_tool.validate()?;
    }
    Ok(())
}

fn validate_extra(extra: &BTreeMap<String, Value>) -> Result<(), MessagesCodecError> {
    let encoded = serde_json::to_vec(extra).map_err(MessagesCodecError::JsonEncode)?;
    if encoded.len() > MAX_EXTRA_BYTES {
        return Err(MessagesCodecError::InvalidOption {
            field: "extra",
            reason: "encoded value exceeds 256 KiB",
        });
    }

    let mut fields = 0usize;
    for (name, value) in extra {
        if is_protected_option_field(name) {
            return Err(MessagesCodecError::ProtectedOptionField {
                path: safe_path(name),
            });
        }
        validate_extra_value(value, name, 1, &mut fields)?;
    }
    Ok(())
}

fn validate_extra_value(
    value: &Value,
    path: &str,
    depth: usize,
    fields: &mut usize,
) -> Result<(), MessagesCodecError> {
    if depth > MAX_EXTRA_DEPTH {
        return Err(MessagesCodecError::InvalidOption {
            field: "extra",
            reason: "JSON nesting exceeds 16 levels",
        });
    }
    match value {
        Value::Object(object) => {
            for (name, child) in object {
                *fields = fields.saturating_add(1);
                if *fields > MAX_EXTRA_FIELDS {
                    return Err(MessagesCodecError::InvalidOption {
                        field: "extra",
                        reason: "JSON object exceeds 1024 fields",
                    });
                }
                let child_path = format!("{path}.{name}");
                if is_sensitive_nested_field(name) {
                    return Err(MessagesCodecError::ProtectedOptionField {
                        path: safe_path(&child_path),
                    });
                }
                validate_extra_value(child, &child_path, depth.saturating_add(1), fields)?;
            }
        }
        Value::Array(values) => {
            for (index, child) in values.iter().enumerate() {
                validate_extra_value(
                    child,
                    &format!("{path}[{index}]"),
                    depth.saturating_add(1),
                    fields,
                )?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn is_sensitive_nested_field(name: &str) -> bool {
    matches!(
        normalize_field(name).as_str(),
        "api_key"
            | "x_api_key"
            | "authorization"
            | "auth"
            | "token"
            | "bearer"
            | "endpoint"
            | "base_url"
            | "host"
            | "headers"
            | "header"
            | "proxy"
            | "tls"
            | "audience"
    )
}

pub(crate) fn normalize_field(name: &str) -> String {
    name.trim().to_ascii_lowercase().replace('-', "_")
}

fn safe_path(path: &str) -> String {
    let mut safe = path
        .chars()
        .take(256)
        .map(|character| {
            if character.is_control() {
                '?'
            } else {
                character
            }
        })
        .collect::<String>();
    if path.chars().count() > 256 {
        safe.push_str("...");
    }
    safe
}
