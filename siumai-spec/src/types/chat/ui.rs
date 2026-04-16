use serde::de::Error as _;
use serde::ser::{Error as SerError, SerializeMap};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

use crate::types::{ProviderOptionsMap, ProviderReference};

/// AI SDK-aligned UI message role.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum UiMessageRole {
    System,
    User,
    Assistant,
}

/// Shared state for streaming text/reasoning UI parts.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "kebab-case")]
pub enum UiPartState {
    Streaming,
    Done,
}

/// Tool invocation lifecycle state carried by a UI tool part.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "kebab-case")]
pub enum UiToolPartState {
    InputStreaming,
    InputAvailable,
    ApprovalRequested,
    ApprovalResponded,
    OutputAvailable,
    OutputError,
    OutputDenied,
}

/// Nested provider metadata map used by AI SDK UI messages.
pub type UiProviderMetadata = ProviderOptionsMap;

/// Frontend-facing UI message aligned with AI SDK `UIMessage`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UiMessage {
    /// Stable UI message id.
    pub id: String,
    /// Message role.
    pub role: UiMessageRole,
    /// Optional UI-only metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    /// Renderable UI parts.
    #[serde(default)]
    pub parts: Vec<UiMessagePart>,
}

impl UiMessage {
    /// Create a new UI message.
    pub fn new(id: impl Into<String>, role: UiMessageRole, parts: Vec<UiMessagePart>) -> Self {
        Self {
            id: id.into(),
            role,
            metadata: None,
            parts,
        }
    }

    /// Create a system UI message.
    pub fn system(id: impl Into<String>, parts: Vec<UiMessagePart>) -> Self {
        Self::new(id, UiMessageRole::System, parts)
    }

    /// Create a user UI message.
    pub fn user(id: impl Into<String>, parts: Vec<UiMessagePart>) -> Self {
        Self::new(id, UiMessageRole::User, parts)
    }

    /// Create an assistant UI message.
    pub fn assistant(id: impl Into<String>, parts: Vec<UiMessagePart>) -> Self {
        Self::new(id, UiMessageRole::Assistant, parts)
    }

    /// Attach UI-only metadata.
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

/// UI message part aligned with AI SDK `UIMessagePart`.
#[derive(Debug, Clone, PartialEq)]
pub enum UiMessagePart {
    Text(UiTextPart),
    Custom(UiCustomPart),
    Reasoning(UiReasoningPart),
    SourceUrl(UiSourceUrlPart),
    SourceDocument(UiSourceDocumentPart),
    File(UiFilePart),
    ReasoningFile(UiReasoningFilePart),
    Data(UiDataPart),
    Tool(UiToolPart),
    StepStart,
}

impl UiMessagePart {
    /// Create a text UI part.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(UiTextPart::new(text))
    }

    /// Create a reasoning UI part.
    pub fn reasoning(text: impl Into<String>) -> Self {
        Self::Reasoning(UiReasoningPart::new(text))
    }

    /// Create a custom UI part.
    pub fn custom(kind: impl Into<String>) -> Self {
        Self::Custom(UiCustomPart::new(kind))
    }

    /// Create a data UI part.
    pub fn data(data_type: impl Into<String>, data: Value) -> Self {
        Self::Data(UiDataPart::new(data_type, data))
    }

    /// Create a step-start UI part.
    pub fn step_start() -> Self {
        Self::StepStart
    }

    /// Validate the structural invariants for this UI message part.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Data(part) => part.validate(),
            Self::Tool(part) => part.validate(),
            _ => Ok(()),
        }
    }
}

impl Serialize for UiMessagePart {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Text(part) => serialize_part(serializer, "text", serde_json::to_value(part)),
            Self::Custom(part) => serialize_part(serializer, "custom", serde_json::to_value(part)),
            Self::Reasoning(part) => {
                serialize_part(serializer, "reasoning", serde_json::to_value(part))
            }
            Self::SourceUrl(part) => {
                serialize_part(serializer, "source-url", serde_json::to_value(part))
            }
            Self::SourceDocument(part) => {
                serialize_part(serializer, "source-document", serde_json::to_value(part))
            }
            Self::File(part) => serialize_part(serializer, "file", serde_json::to_value(part)),
            Self::ReasoningFile(part) => {
                serialize_part(serializer, "reasoning-file", serde_json::to_value(part))
            }
            Self::Data(part) => {
                serialize_object_part(serializer, &part.type_name(), part.to_json())
            }
            Self::Tool(part) => {
                serialize_object_part(serializer, &part.type_name(), part.to_json())
            }
            Self::StepStart => serialize_object_part(serializer, "step-start", Map::new()),
        }
    }
}

impl<'de> Deserialize<'de> for UiMessagePart {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        let mut obj = as_object(value).map_err(D::Error::custom)?;
        let part_type = remove_string(&mut obj, &["type"]).ok_or_else(|| {
            D::Error::custom("UI message part must contain a string `type` field")
        })?;

        match part_type.as_str() {
            "text" => deserialize_fixed_part::<UiTextPart>(obj)
                .map(Self::Text)
                .map_err(D::Error::custom),
            "custom" => deserialize_fixed_part::<UiCustomPart>(obj)
                .map(Self::Custom)
                .map_err(D::Error::custom),
            "reasoning" => deserialize_fixed_part::<UiReasoningPart>(obj)
                .map(Self::Reasoning)
                .map_err(D::Error::custom),
            "source-url" => deserialize_fixed_part::<UiSourceUrlPart>(obj)
                .map(Self::SourceUrl)
                .map_err(D::Error::custom),
            "source-document" => deserialize_fixed_part::<UiSourceDocumentPart>(obj)
                .map(Self::SourceDocument)
                .map_err(D::Error::custom),
            "file" => deserialize_fixed_part::<UiFilePart>(obj)
                .map(Self::File)
                .map_err(D::Error::custom),
            "reasoning-file" => deserialize_fixed_part::<UiReasoningFilePart>(obj)
                .map(Self::ReasoningFile)
                .map_err(D::Error::custom),
            "step-start" => Ok(Self::StepStart),
            "dynamic-tool" => UiToolPart::from_json("dynamic-tool", obj)
                .map(Self::Tool)
                .map_err(D::Error::custom),
            value if value.starts_with("tool-") => UiToolPart::from_json(value, obj)
                .map(Self::Tool)
                .map_err(D::Error::custom),
            value if value.starts_with("data-") => UiDataPart::from_json(value, obj)
                .map(Self::Data)
                .map_err(D::Error::custom),
            other => Err(D::Error::custom(format!(
                "unsupported UI message part type `{other}`"
            ))),
        }
    }
}

/// Text UI part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UiTextPart {
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<UiPartState>,
    #[serde(
        default,
        rename = "providerMetadata",
        alias = "provider_metadata",
        skip_serializing_if = "ProviderOptionsMap::is_empty"
    )]
    pub provider_metadata: UiProviderMetadata,
}

impl UiTextPart {
    /// Create a text UI part.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            state: None,
            provider_metadata: ProviderOptionsMap::default(),
        }
    }
}

/// Reasoning UI part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UiReasoningPart {
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<UiPartState>,
    #[serde(
        default,
        rename = "providerMetadata",
        alias = "provider_metadata",
        skip_serializing_if = "ProviderOptionsMap::is_empty"
    )]
    pub provider_metadata: UiProviderMetadata,
}

impl UiReasoningPart {
    /// Create a reasoning UI part.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            state: None,
            provider_metadata: ProviderOptionsMap::default(),
        }
    }
}

/// Custom UI part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UiCustomPart {
    pub kind: String,
    #[serde(
        default,
        rename = "providerMetadata",
        alias = "provider_metadata",
        skip_serializing_if = "ProviderOptionsMap::is_empty"
    )]
    pub provider_metadata: UiProviderMetadata,
}

impl UiCustomPart {
    /// Create a custom UI part.
    pub fn new(kind: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            provider_metadata: ProviderOptionsMap::default(),
        }
    }
}

/// URL source UI part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UiSourceUrlPart {
    #[serde(rename = "sourceId", alias = "source_id")]
    pub source_id: String,
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(
        default,
        rename = "providerMetadata",
        alias = "provider_metadata",
        skip_serializing_if = "ProviderOptionsMap::is_empty"
    )]
    pub provider_metadata: UiProviderMetadata,
}

/// Document source UI part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UiSourceDocumentPart {
    #[serde(rename = "sourceId", alias = "source_id")]
    pub source_id: String,
    #[serde(rename = "mediaType", alias = "media_type")]
    pub media_type: String,
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    #[serde(
        default,
        rename = "providerMetadata",
        alias = "provider_metadata",
        skip_serializing_if = "ProviderOptionsMap::is_empty"
    )]
    pub provider_metadata: UiProviderMetadata,
}

/// File UI part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UiFilePart {
    #[serde(rename = "mediaType", alias = "media_type")]
    pub media_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    pub url: String,
    #[serde(
        default,
        rename = "providerReference",
        alias = "provider_reference",
        skip_serializing_if = "Option::is_none"
    )]
    pub provider_reference: Option<ProviderReference>,
    #[serde(
        default,
        rename = "providerMetadata",
        alias = "provider_metadata",
        skip_serializing_if = "ProviderOptionsMap::is_empty"
    )]
    pub provider_metadata: UiProviderMetadata,
}

impl UiFilePart {
    /// Create a file UI part from a URL.
    pub fn new(url: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self {
            media_type: media_type.into(),
            filename: None,
            url: url.into(),
            provider_reference: None,
            provider_metadata: ProviderOptionsMap::default(),
        }
    }
}

/// Reasoning-file UI part.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UiReasoningFilePart {
    #[serde(rename = "mediaType", alias = "media_type")]
    pub media_type: String,
    pub url: String,
    #[serde(
        default,
        rename = "providerMetadata",
        alias = "provider_metadata",
        skip_serializing_if = "ProviderOptionsMap::is_empty"
    )]
    pub provider_metadata: UiProviderMetadata,
}

impl UiReasoningFilePart {
    /// Create a reasoning-file UI part.
    pub fn new(url: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self {
            media_type: media_type.into(),
            url: url.into(),
            provider_metadata: ProviderOptionsMap::default(),
        }
    }
}

/// Data UI part with the `data-*` discriminator folded into `data_type`.
#[derive(Debug, Clone, PartialEq)]
pub struct UiDataPart {
    pub data_type: String,
    pub id: Option<String>,
    pub data: Value,
}

impl UiDataPart {
    /// Create a data UI part.
    pub fn new(data_type: impl Into<String>, data: Value) -> Self {
        Self {
            data_type: data_type.into(),
            id: None,
            data,
        }
    }

    /// Return the serialized `type` discriminator.
    pub fn type_name(&self) -> String {
        format!("data-{}", self.data_type)
    }

    /// Validate the structural invariants for this data part.
    pub fn validate(&self) -> Result<(), String> {
        if self.data_type.is_empty() {
            return Err("data part type cannot be empty".to_string());
        }

        Ok(())
    }

    fn to_json(&self) -> Map<String, Value> {
        let mut obj = Map::new();
        if let Some(id) = &self.id {
            obj.insert("id".to_string(), Value::String(id.clone()));
        }
        obj.insert("data".to_string(), self.data.clone());
        obj
    }

    fn from_json(type_name: &str, mut obj: Map<String, Value>) -> Result<Self, String> {
        let Some(data_type) = type_name.strip_prefix("data-") else {
            return Err(format!("invalid data UI part type `{type_name}`"));
        };

        Ok(Self {
            data_type: data_type.to_string(),
            id: remove_string(&mut obj, &["id"]),
            data: obj
                .remove("data")
                .ok_or_else(|| format!("data UI part `{type_name}` is missing `data`"))?,
        })
    }
}

/// Tool approval metadata carried by UI tool parts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UiToolApproval {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approved: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Approval payload for `approval-requested` UI tool invocations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UiToolApprovalRequest {
    pub id: String,
}

impl UiToolApprovalRequest {
    fn into_approval(self) -> UiToolApproval {
        UiToolApproval {
            id: self.id,
            approved: None,
            reason: None,
        }
    }
}

/// Approval payload for `approval-responded` UI tool invocations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UiToolApprovalDecision {
    pub id: String,
    pub approved: bool,
    pub reason: Option<String>,
}

impl UiToolApprovalDecision {
    fn into_approval(self) -> UiToolApproval {
        UiToolApproval {
            id: self.id,
            approved: Some(self.approved),
            reason: self.reason,
        }
    }
}

/// Optional approved-approval payload for successful tool result states.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UiToolApprovedApproval {
    pub id: String,
    pub reason: Option<String>,
}

impl UiToolApprovedApproval {
    fn into_approval(self) -> UiToolApproval {
        UiToolApproval {
            id: self.id,
            approved: Some(true),
            reason: self.reason,
        }
    }
}

/// Required denied-approval payload for `output-denied` UI tool invocations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UiToolDeniedApproval {
    pub id: String,
    pub reason: Option<String>,
}

impl UiToolDeniedApproval {
    fn into_approval(self) -> UiToolApproval {
        UiToolApproval {
            id: self.id,
            approved: Some(false),
            reason: self.reason,
        }
    }
}

/// State-discriminated UI tool invocation aligned with AI SDK `UIToolInvocation`.
#[derive(Debug, Clone, PartialEq)]
pub struct UiToolInvocation {
    pub tool_call_id: String,
    pub title: Option<String>,
    pub provider_executed: Option<bool>,
    pub call_provider_metadata: UiProviderMetadata,
    pub state: UiToolInvocationState,
}

impl UiToolInvocation {
    /// Return the current tool invocation state.
    pub const fn state(&self) -> UiToolPartState {
        self.state.state()
    }
}

/// State-specific payload for a UI tool invocation.
#[derive(Debug, Clone, PartialEq)]
pub enum UiToolInvocationState {
    InputStreaming {
        input: Option<Value>,
    },
    InputAvailable {
        input: Value,
    },
    ApprovalRequested {
        input: Value,
        approval: UiToolApprovalRequest,
    },
    ApprovalResponded {
        input: Value,
        approval: UiToolApprovalDecision,
    },
    OutputAvailable {
        input: Value,
        output: Value,
        result_provider_metadata: UiProviderMetadata,
        preliminary: Option<bool>,
        approval: Option<UiToolApprovedApproval>,
    },
    OutputError {
        input: Option<Value>,
        raw_input: Option<Value>,
        error_text: String,
        result_provider_metadata: UiProviderMetadata,
        approval: Option<UiToolApprovedApproval>,
    },
    OutputDenied {
        input: Value,
        approval: UiToolDeniedApproval,
    },
}

impl UiToolInvocationState {
    /// Return the stable state discriminator for this payload.
    pub const fn state(&self) -> UiToolPartState {
        match self {
            Self::InputStreaming { .. } => UiToolPartState::InputStreaming,
            Self::InputAvailable { .. } => UiToolPartState::InputAvailable,
            Self::ApprovalRequested { .. } => UiToolPartState::ApprovalRequested,
            Self::ApprovalResponded { .. } => UiToolPartState::ApprovalResponded,
            Self::OutputAvailable { .. } => UiToolPartState::OutputAvailable,
            Self::OutputError { .. } => UiToolPartState::OutputError,
            Self::OutputDenied { .. } => UiToolPartState::OutputDenied,
        }
    }
}

/// Static or dynamic tool discriminator for AI SDK UI tool parts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UiToolKind {
    Static { tool_name: String },
    Dynamic { tool_name: String },
}

impl UiToolKind {
    /// Return the resolved tool name.
    pub fn tool_name(&self) -> &str {
        match self {
            Self::Static { tool_name } | Self::Dynamic { tool_name } => tool_name.as_str(),
        }
    }

    fn type_name(&self) -> String {
        match self {
            Self::Static { tool_name } => format!("tool-{tool_name}"),
            Self::Dynamic { .. } => "dynamic-tool".to_string(),
        }
    }
}

/// UI tool part aligned with static `tool-*` and `dynamic-tool` parts.
#[derive(Debug, Clone, PartialEq)]
pub struct UiToolPart {
    pub kind: UiToolKind,
    pub tool_call_id: String,
    pub title: Option<String>,
    pub provider_executed: Option<bool>,
    pub state: UiToolPartState,
    pub input: Option<Value>,
    pub raw_input: Option<Value>,
    pub output: Option<Value>,
    pub error_text: Option<String>,
    pub call_provider_metadata: UiProviderMetadata,
    pub result_provider_metadata: UiProviderMetadata,
    pub preliminary: Option<bool>,
    pub approval: Option<UiToolApproval>,
}

impl UiToolPart {
    /// Create a static named tool UI part.
    pub fn named(
        tool_name: impl Into<String>,
        tool_call_id: impl Into<String>,
        state: UiToolPartState,
    ) -> Self {
        Self {
            kind: UiToolKind::Static {
                tool_name: tool_name.into(),
            },
            tool_call_id: tool_call_id.into(),
            title: None,
            provider_executed: None,
            state,
            input: None,
            raw_input: None,
            output: None,
            error_text: None,
            call_provider_metadata: ProviderOptionsMap::default(),
            result_provider_metadata: ProviderOptionsMap::default(),
            preliminary: None,
            approval: None,
        }
    }

    /// Create a dynamic tool UI part.
    pub fn dynamic(
        tool_name: impl Into<String>,
        tool_call_id: impl Into<String>,
        state: UiToolPartState,
    ) -> Self {
        Self {
            kind: UiToolKind::Dynamic {
                tool_name: tool_name.into(),
            },
            ..Self::named("", tool_call_id, state)
        }
    }

    /// Return the resolved tool name.
    pub fn tool_name(&self) -> &str {
        self.kind.tool_name()
    }

    /// Return the serialized type discriminator.
    pub fn type_name(&self) -> String {
        self.kind.type_name()
    }

    /// Return true when this part is one of the incomplete input states.
    pub fn is_incomplete_input_state(&self) -> bool {
        matches!(
            self.state,
            UiToolPartState::InputStreaming | UiToolPartState::InputAvailable
        )
    }

    /// Convert the wide serde-compatible tool part into a state-discriminated invocation view.
    pub fn invocation(&self) -> Result<UiToolInvocation, String> {
        let state = match self.state {
            UiToolPartState::InputStreaming => {
                reject_field(
                    self.output.is_some(),
                    "output is not allowed for input-streaming",
                )?;
                reject_field(
                    self.error_text.is_some(),
                    "errorText is not allowed for input-streaming",
                )?;
                reject_field(
                    self.approval.is_some(),
                    "approval is not allowed for input-streaming",
                )?;
                reject_field(
                    self.raw_input.is_some(),
                    "rawInput is not allowed for input-streaming",
                )?;
                reject_field(
                    !self.result_provider_metadata.is_empty(),
                    "resultProviderMetadata is not allowed for input-streaming",
                )?;
                reject_field(
                    self.preliminary.is_some(),
                    "preliminary is not allowed for input-streaming",
                )?;

                UiToolInvocationState::InputStreaming {
                    input: self.input.clone(),
                }
            }
            UiToolPartState::InputAvailable => {
                require_field(
                    self.input.is_some(),
                    "input is required for input-available",
                )?;
                reject_field(
                    self.output.is_some(),
                    "output is not allowed for input-available",
                )?;
                reject_field(
                    self.error_text.is_some(),
                    "errorText is not allowed for input-available",
                )?;
                reject_field(
                    self.approval.is_some(),
                    "approval is not allowed for input-available",
                )?;
                reject_field(
                    self.raw_input.is_some(),
                    "rawInput is not allowed for input-available",
                )?;
                reject_field(
                    !self.result_provider_metadata.is_empty(),
                    "resultProviderMetadata is not allowed for input-available",
                )?;
                reject_field(
                    self.preliminary.is_some(),
                    "preliminary is not allowed for input-available",
                )?;

                UiToolInvocationState::InputAvailable {
                    input: self.input.clone().expect("checked input-available input"),
                }
            }
            UiToolPartState::ApprovalRequested => {
                require_field(
                    self.input.is_some(),
                    "input is required for approval-requested",
                )?;
                reject_field(
                    self.output.is_some(),
                    "output is not allowed for approval-requested",
                )?;
                reject_field(
                    self.error_text.is_some(),
                    "errorText is not allowed for approval-requested",
                )?;
                reject_field(
                    self.raw_input.is_some(),
                    "rawInput is not allowed for approval-requested",
                )?;
                reject_field(
                    !self.result_provider_metadata.is_empty(),
                    "resultProviderMetadata is not allowed for approval-requested",
                )?;
                reject_field(
                    self.preliminary.is_some(),
                    "preliminary is not allowed for approval-requested",
                )?;
                require_approval_requested(self.approval.as_ref())?;

                let approval = self
                    .approval
                    .as_ref()
                    .expect("checked approval-requested approval");
                UiToolInvocationState::ApprovalRequested {
                    input: self
                        .input
                        .clone()
                        .expect("checked approval-requested input"),
                    approval: UiToolApprovalRequest {
                        id: approval.id.clone(),
                    },
                }
            }
            UiToolPartState::ApprovalResponded => {
                require_field(
                    self.input.is_some(),
                    "input is required for approval-responded",
                )?;
                reject_field(
                    self.output.is_some(),
                    "output is not allowed for approval-responded",
                )?;
                reject_field(
                    self.error_text.is_some(),
                    "errorText is not allowed for approval-responded",
                )?;
                reject_field(
                    self.raw_input.is_some(),
                    "rawInput is not allowed for approval-responded",
                )?;
                reject_field(
                    !self.result_provider_metadata.is_empty(),
                    "resultProviderMetadata is not allowed for approval-responded",
                )?;
                reject_field(
                    self.preliminary.is_some(),
                    "preliminary is not allowed for approval-responded",
                )?;
                require_approval_responded(self.approval.as_ref())?;

                let approval = self
                    .approval
                    .as_ref()
                    .expect("checked approval-responded approval");
                UiToolInvocationState::ApprovalResponded {
                    input: self
                        .input
                        .clone()
                        .expect("checked approval-responded input"),
                    approval: UiToolApprovalDecision {
                        id: approval.id.clone(),
                        approved: approval
                            .approved
                            .expect("checked approval-responded approved"),
                        reason: approval.reason.clone(),
                    },
                }
            }
            UiToolPartState::OutputAvailable => {
                require_field(
                    self.input.is_some(),
                    "input is required for output-available",
                )?;
                require_field(
                    self.output.is_some(),
                    "output is required for output-available",
                )?;
                reject_field(
                    self.error_text.is_some(),
                    "errorText is not allowed for output-available",
                )?;
                reject_field(
                    self.raw_input.is_some(),
                    "rawInput is not allowed for output-available",
                )?;
                require_optional_approved_true(
                    self.approval.as_ref(),
                    "output-available approval must be approved=true",
                )?;

                UiToolInvocationState::OutputAvailable {
                    input: self.input.clone().expect("checked output-available input"),
                    output: self
                        .output
                        .clone()
                        .expect("checked output-available output"),
                    result_provider_metadata: self.result_provider_metadata.clone(),
                    preliminary: self.preliminary,
                    approval: self
                        .approval
                        .as_ref()
                        .map(|approval| UiToolApprovedApproval {
                            id: approval.id.clone(),
                            reason: approval.reason.clone(),
                        }),
                }
            }
            UiToolPartState::OutputError => {
                require_field(
                    self.error_text.is_some(),
                    "errorText is required for output-error",
                )?;
                reject_field(
                    self.output.is_some(),
                    "output is not allowed for output-error",
                )?;
                reject_field(
                    self.preliminary.is_some(),
                    "preliminary is not allowed for output-error",
                )?;
                require_optional_approved_true(
                    self.approval.as_ref(),
                    "output-error approval must be approved=true",
                )?;

                UiToolInvocationState::OutputError {
                    input: self.input.clone(),
                    raw_input: self.raw_input.clone(),
                    error_text: self
                        .error_text
                        .clone()
                        .expect("checked output-error error_text"),
                    result_provider_metadata: self.result_provider_metadata.clone(),
                    approval: self
                        .approval
                        .as_ref()
                        .map(|approval| UiToolApprovedApproval {
                            id: approval.id.clone(),
                            reason: approval.reason.clone(),
                        }),
                }
            }
            UiToolPartState::OutputDenied => {
                require_field(self.input.is_some(), "input is required for output-denied")?;
                reject_field(
                    self.output.is_some(),
                    "output is not allowed for output-denied",
                )?;
                reject_field(
                    self.error_text.is_some(),
                    "errorText is not allowed for output-denied",
                )?;
                reject_field(
                    self.raw_input.is_some(),
                    "rawInput is not allowed for output-denied",
                )?;
                reject_field(
                    !self.result_provider_metadata.is_empty(),
                    "resultProviderMetadata is not allowed for output-denied",
                )?;
                reject_field(
                    self.preliminary.is_some(),
                    "preliminary is not allowed for output-denied",
                )?;
                require_output_denied_approval(self.approval.as_ref())?;

                let approval = self
                    .approval
                    .as_ref()
                    .expect("checked output-denied approval");
                UiToolInvocationState::OutputDenied {
                    input: self.input.clone().expect("checked output-denied input"),
                    approval: UiToolDeniedApproval {
                        id: approval.id.clone(),
                        reason: approval.reason.clone(),
                    },
                }
            }
        };

        Ok(UiToolInvocation {
            tool_call_id: self.tool_call_id.clone(),
            title: self.title.clone(),
            provider_executed: self.provider_executed,
            call_provider_metadata: self.call_provider_metadata.clone(),
            state,
        })
    }

    /// Build a wide serde-compatible tool part from a typed invocation payload.
    pub fn from_invocation(kind: UiToolKind, invocation: UiToolInvocation) -> Self {
        let UiToolInvocation {
            tool_call_id,
            title,
            provider_executed,
            call_provider_metadata,
            state,
        } = invocation;

        let mut tool = Self {
            kind,
            tool_call_id,
            title,
            provider_executed,
            state: state.state(),
            input: None,
            raw_input: None,
            output: None,
            error_text: None,
            call_provider_metadata,
            result_provider_metadata: ProviderOptionsMap::default(),
            preliminary: None,
            approval: None,
        };

        match state {
            UiToolInvocationState::InputStreaming { input } => {
                tool.input = input;
            }
            UiToolInvocationState::InputAvailable { input } => {
                tool.input = Some(input);
            }
            UiToolInvocationState::ApprovalRequested { input, approval } => {
                tool.input = Some(input);
                tool.approval = Some(approval.into_approval());
            }
            UiToolInvocationState::ApprovalResponded { input, approval } => {
                tool.input = Some(input);
                tool.approval = Some(approval.into_approval());
            }
            UiToolInvocationState::OutputAvailable {
                input,
                output,
                result_provider_metadata,
                preliminary,
                approval,
            } => {
                tool.input = Some(input);
                tool.output = Some(output);
                tool.result_provider_metadata = result_provider_metadata;
                tool.preliminary = preliminary;
                tool.approval = approval.map(UiToolApprovedApproval::into_approval);
            }
            UiToolInvocationState::OutputError {
                input,
                raw_input,
                error_text,
                result_provider_metadata,
                approval,
            } => {
                tool.input = input;
                tool.raw_input = raw_input;
                tool.error_text = Some(error_text);
                tool.result_provider_metadata = result_provider_metadata;
                tool.approval = approval.map(UiToolApprovedApproval::into_approval);
            }
            UiToolInvocationState::OutputDenied { input, approval } => {
                tool.input = Some(input);
                tool.approval = Some(approval.into_approval());
            }
        }

        tool
    }

    /// Validate the structural invariants for this tool part.
    pub fn validate(&self) -> Result<(), String> {
        self.invocation().map(|_| ())
    }

    fn to_json(&self) -> Map<String, Value> {
        let mut obj = Map::new();
        if let UiToolKind::Dynamic { tool_name } = &self.kind {
            obj.insert("toolName".to_string(), Value::String(tool_name.clone()));
        }
        obj.insert(
            "toolCallId".to_string(),
            Value::String(self.tool_call_id.clone()),
        );
        if let Some(title) = &self.title {
            obj.insert("title".to_string(), Value::String(title.clone()));
        }
        if let Some(provider_executed) = self.provider_executed {
            obj.insert(
                "providerExecuted".to_string(),
                Value::Bool(provider_executed),
            );
        }
        obj.insert(
            "state".to_string(),
            serde_json::to_value(&self.state).expect("serialize UI tool state"),
        );
        if let Some(input) = &self.input {
            obj.insert("input".to_string(), input.clone());
        }
        if let Some(raw_input) = &self.raw_input {
            obj.insert("rawInput".to_string(), raw_input.clone());
        }
        if let Some(output) = &self.output {
            obj.insert("output".to_string(), output.clone());
        }
        if let Some(error_text) = &self.error_text {
            obj.insert("errorText".to_string(), Value::String(error_text.clone()));
        }
        if !self.call_provider_metadata.is_empty() {
            obj.insert(
                "callProviderMetadata".to_string(),
                serde_json::to_value(&self.call_provider_metadata)
                    .expect("serialize UI call provider metadata"),
            );
        }
        if !self.result_provider_metadata.is_empty() {
            obj.insert(
                "resultProviderMetadata".to_string(),
                serde_json::to_value(&self.result_provider_metadata)
                    .expect("serialize UI result provider metadata"),
            );
        }
        if let Some(preliminary) = self.preliminary {
            obj.insert("preliminary".to_string(), Value::Bool(preliminary));
        }
        if let Some(approval) = &self.approval {
            obj.insert(
                "approval".to_string(),
                serde_json::to_value(approval).expect("serialize UI tool approval"),
            );
        }
        obj
    }

    fn from_json(type_name: &str, mut obj: Map<String, Value>) -> Result<Self, String> {
        let kind = if type_name == "dynamic-tool" {
            UiToolKind::Dynamic {
                tool_name: remove_string(&mut obj, &["toolName", "tool_name"])
                    .ok_or_else(|| "dynamic-tool UI part is missing `toolName`".to_string())?,
            }
        } else if let Some(tool_name) = type_name.strip_prefix("tool-") {
            UiToolKind::Static {
                tool_name: tool_name.to_string(),
            }
        } else {
            return Err(format!("invalid tool UI part type `{type_name}`"));
        };

        Ok(Self {
            kind,
            tool_call_id: remove_string(&mut obj, &["toolCallId", "tool_call_id"])
                .ok_or_else(|| format!("tool UI part `{type_name}` is missing `toolCallId`"))?,
            title: remove_string(&mut obj, &["title"]),
            provider_executed: remove_bool(&mut obj, &["providerExecuted", "provider_executed"])?,
            state: remove_required::<UiToolPartState>(&mut obj, &["state"], type_name, "state")?,
            input: remove_value(&mut obj, &["input"]),
            raw_input: remove_value(&mut obj, &["rawInput", "raw_input"]),
            output: remove_value(&mut obj, &["output"]),
            error_text: remove_string(&mut obj, &["errorText", "error_text"]),
            call_provider_metadata: remove_optional::<UiProviderMetadata>(
                &mut obj,
                &["callProviderMetadata", "call_provider_metadata"],
            )?
            .unwrap_or_default(),
            result_provider_metadata: remove_optional::<UiProviderMetadata>(
                &mut obj,
                &["resultProviderMetadata", "result_provider_metadata"],
            )?
            .unwrap_or_default(),
            preliminary: remove_bool(&mut obj, &["preliminary"])?,
            approval: remove_optional::<UiToolApproval>(&mut obj, &["approval"])?,
        })
    }
}

fn serialize_part<S>(
    serializer: S,
    type_name: &str,
    value: Result<Value, serde_json::Error>,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let obj = as_object(value.map_err(S::Error::custom)?).map_err(S::Error::custom)?;
    serialize_object_part(serializer, type_name, obj)
}

fn serialize_object_part<S>(
    serializer: S,
    type_name: &str,
    obj: Map<String, Value>,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let mut map = serializer.serialize_map(Some(obj.len() + 1))?;
    map.serialize_entry("type", type_name)?;
    for (key, value) in obj {
        map.serialize_entry(&key, &value)?;
    }
    map.end()
}

fn as_object(value: Value) -> Result<Map<String, Value>, String> {
    value
        .as_object()
        .cloned()
        .ok_or_else(|| "UI message part must serialize as an object".to_string())
}

fn deserialize_fixed_part<T>(obj: Map<String, Value>) -> Result<T, serde_json::Error>
where
    T: for<'de> Deserialize<'de>,
{
    serde_json::from_value(Value::Object(obj))
}

fn remove_value(obj: &mut Map<String, Value>, keys: &[&str]) -> Option<Value> {
    keys.iter().find_map(|key| obj.remove(*key))
}

fn remove_string(obj: &mut Map<String, Value>, keys: &[&str]) -> Option<String> {
    remove_value(obj, keys).and_then(|value| match value {
        Value::String(value) => Some(value),
        _ => None,
    })
}

fn remove_bool(obj: &mut Map<String, Value>, keys: &[&str]) -> Result<Option<bool>, String> {
    match remove_value(obj, keys) {
        Some(Value::Bool(value)) => Ok(Some(value)),
        Some(other) => Err(format!("expected boolean value, got `{other}`")),
        None => Ok(None),
    }
}

fn remove_optional<T>(obj: &mut Map<String, Value>, keys: &[&str]) -> Result<Option<T>, String>
where
    T: for<'de> Deserialize<'de>,
{
    match remove_value(obj, keys) {
        Some(value) => serde_json::from_value(value)
            .map(Some)
            .map_err(|err| err.to_string()),
        None => Ok(None),
    }
}

fn remove_required<T>(
    obj: &mut Map<String, Value>,
    keys: &[&str],
    type_name: &str,
    field: &str,
) -> Result<T, String>
where
    T: for<'de> Deserialize<'de>,
{
    remove_optional(obj, keys)?.ok_or_else(|| format!("UI part `{type_name}` is missing `{field}`"))
}

fn require_field(condition: bool, message: &str) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(message.to_string())
    }
}

fn reject_field(condition: bool, message: &str) -> Result<(), String> {
    if condition {
        Err(message.to_string())
    } else {
        Ok(())
    }
}

fn require_approval_requested(approval: Option<&UiToolApproval>) -> Result<(), String> {
    let approval = approval
        .ok_or_else(|| "approval metadata is required for approval-requested".to_string())?;
    reject_field(
        approval.approved.is_some(),
        "approval-requested approval must not include approved",
    )?;
    reject_field(
        approval.reason.is_some(),
        "approval-requested approval must not include reason",
    )?;
    Ok(())
}

fn require_approval_responded(approval: Option<&UiToolApproval>) -> Result<(), String> {
    let approval = approval
        .ok_or_else(|| "approval metadata is required for approval-responded".to_string())?;
    require_field(
        approval.approved.is_some(),
        "approval-responded approval must include approved",
    )?;
    Ok(())
}

fn require_optional_approved_true(
    approval: Option<&UiToolApproval>,
    message: &str,
) -> Result<(), String> {
    if let Some(approval) = approval {
        require_field(approval.approved == Some(true), message)?;
    }

    Ok(())
}

fn require_output_denied_approval(approval: Option<&UiToolApproval>) -> Result<(), String> {
    let approval =
        approval.ok_or_else(|| "approval metadata is required for output-denied".to_string())?;
    require_field(
        approval.approved == Some(false),
        "output-denied approval must be approved=false",
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        UiDataPart, UiMessagePart, UiToolApproval, UiToolApprovedApproval, UiToolInvocation,
        UiToolInvocationState, UiToolKind, UiToolPart, UiToolPartState,
    };
    use crate::types::ProviderOptionsMap;

    #[test]
    fn data_part_serializes_with_data_prefix() {
        let value = serde_json::to_value(UiMessagePart::Data(UiDataPart {
            data_type: "weather".to_string(),
            id: Some("part_1".to_string()),
            data: serde_json::json!({ "city": "Tokyo" }),
        }))
        .expect("serialize data part");

        assert_eq!(value["type"], serde_json::json!("data-weather"));
        assert_eq!(value["id"], serde_json::json!("part_1"));
    }

    #[test]
    fn static_tool_part_serializes_with_tool_prefix() {
        let mut tool = UiToolPart::named("search", "call_1", UiToolPartState::OutputAvailable);
        tool.input = Some(serde_json::json!({ "query": "rust" }));
        tool.output = Some(serde_json::json!({ "hits": 3 }));
        tool.approval = Some(UiToolApproval {
            id: "approval_1".to_string(),
            approved: Some(true),
            reason: None,
        });

        let value = serde_json::to_value(UiMessagePart::Tool(tool)).expect("serialize tool part");
        assert_eq!(value["type"], serde_json::json!("tool-search"));
        assert!(value.get("toolName").is_none());
        assert_eq!(value["toolCallId"], serde_json::json!("call_1"));
    }

    #[test]
    fn dynamic_tool_part_roundtrips() {
        let mut tool = UiToolPart::dynamic("code-runner", "call_2", UiToolPartState::OutputError);
        tool.raw_input = Some(serde_json::json!({ "code": "print('hi')" }));
        tool.error_text = Some("boom".to_string());
        tool.provider_executed = Some(true);

        let value = serde_json::to_value(UiMessagePart::Tool(tool.clone()))
            .expect("serialize dynamic tool part");
        assert_eq!(value["type"], serde_json::json!("dynamic-tool"));
        assert_eq!(value["toolName"], serde_json::json!("code-runner"));

        let roundtrip: UiMessagePart =
            serde_json::from_value(value).expect("deserialize dynamic tool part");
        let UiMessagePart::Tool(roundtrip) = roundtrip else {
            panic!("expected tool part");
        };
        assert_eq!(
            roundtrip.kind,
            UiToolKind::Dynamic {
                tool_name: "code-runner".to_string()
            }
        );
        assert_eq!(roundtrip, tool);
    }

    #[test]
    fn tool_part_validation_rejects_approval_requested_with_decision() {
        let mut tool = UiToolPart::named("search", "call_1", UiToolPartState::ApprovalRequested);
        tool.input = Some(serde_json::json!({ "query": "rust" }));
        tool.approval = Some(UiToolApproval {
            id: "approval_1".to_string(),
            approved: Some(true),
            reason: None,
        });

        let err = tool.validate().expect_err("validation should fail");
        assert!(err.contains("must not include approved"));
    }

    #[test]
    fn tool_part_validation_accepts_output_error_with_raw_input_only() {
        let mut tool = UiToolPart::dynamic("code-runner", "call_2", UiToolPartState::OutputError);
        tool.raw_input = Some(serde_json::json!({ "code": "print('hi')" }));
        tool.error_text = Some("boom".to_string());

        tool.validate().expect("validation should pass");
    }

    #[test]
    fn tool_part_validation_rejects_result_metadata_outside_output_states() {
        let mut tool = UiToolPart::named("search", "call_1", UiToolPartState::InputAvailable);
        tool.input = Some(serde_json::json!({ "query": "rust" }));
        tool.result_provider_metadata.insert(
            "openai".to_string(),
            serde_json::json!({ "itemId": "result_1" }),
        );

        let err = tool.validate().expect_err("validation should fail");
        assert!(err.contains("resultProviderMetadata is not allowed"));
    }

    #[test]
    fn tool_part_invocation_roundtrips_output_available_state() {
        let mut call_provider_metadata = ProviderOptionsMap::default();
        call_provider_metadata.insert("openai", serde_json::json!({ "itemId": "call_1" }));

        let mut result_provider_metadata = ProviderOptionsMap::default();
        result_provider_metadata.insert("openai", serde_json::json!({ "itemId": "result_1" }));

        let invocation = UiToolInvocation {
            tool_call_id: "call_1".to_string(),
            title: Some("Weather lookup".to_string()),
            provider_executed: Some(true),
            call_provider_metadata,
            state: UiToolInvocationState::OutputAvailable {
                input: serde_json::json!({ "city": "Tokyo" }),
                output: serde_json::json!({ "forecast": "sunny" }),
                result_provider_metadata,
                preliminary: Some(true),
                approval: Some(UiToolApprovedApproval {
                    id: "approval_1".to_string(),
                    reason: Some("approved".to_string()),
                }),
            },
        };

        let tool = UiToolPart::from_invocation(
            UiToolKind::Static {
                tool_name: "weather".to_string(),
            },
            invocation.clone(),
        );

        assert_eq!(tool.state, UiToolPartState::OutputAvailable);
        assert_eq!(tool.invocation().expect("typed invocation"), invocation);
    }

    #[test]
    fn tool_part_invocation_rejects_invalid_denied_approval_shape() {
        let mut tool = UiToolPart::named("search", "call_1", UiToolPartState::OutputDenied);
        tool.input = Some(serde_json::json!({ "query": "rust" }));
        tool.approval = Some(UiToolApproval {
            id: "approval_1".to_string(),
            approved: Some(true),
            reason: None,
        });

        let err = tool.invocation().expect_err("invocation should fail");
        assert!(err.contains("approved=false"));
    }
}
