//! Lossless OpenAI Responses wire types.

use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

/// A full Responses resource returned by HTTP or a terminal stream event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseWire {
    pub id: String,
    #[serde(default)]
    pub created_at: Option<i64>,
    pub model: String,
    pub status: ResponseStatus,
    #[serde(default)]
    pub output: Vec<OutputItem>,
    #[serde(default)]
    pub usage: Option<ResponseUsageWire>,
    #[serde(default)]
    pub error: Option<ResponseErrorWire>,
    #[serde(default)]
    pub incomplete_details: Option<IncompleteDetailsWire>,
    #[serde(default)]
    pub reasoning: Option<ResponseReasoningConfigWire>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// The lifecycle status carried by a Responses resource.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ResponseStatus {
    Queued,
    InProgress,
    Completed,
    Incomplete,
    Cancelled,
    Failed,
    Other(String),
}

impl ResponseStatus {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Queued => "queued",
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Incomplete => "incomplete",
            Self::Cancelled => "cancelled",
            Self::Failed => "failed",
            Self::Other(value) => value,
        }
    }
}

impl Serialize for ResponseStatus {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ResponseStatus {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(match value.as_str() {
            "queued" => Self::Queued,
            "in_progress" => Self::InProgress,
            "completed" => Self::Completed,
            "incomplete" => Self::Incomplete,
            "cancelled" => Self::Cancelled,
            "failed" => Self::Failed,
            _ => Self::Other(value),
        })
    }
}

/// The lifecycle status carried by an individual output item.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ItemStatus {
    InProgress,
    Completed,
    Incomplete,
    Failed,
    Other(String),
}

impl ItemStatus {
    pub fn as_str(&self) -> &str {
        match self {
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Incomplete => "incomplete",
            Self::Failed => "failed",
            Self::Other(value) => value,
        }
    }
}

impl Serialize for ItemStatus {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ItemStatus {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(match value.as_str() {
            "in_progress" => Self::InProgress,
            "completed" => Self::Completed,
            "incomplete" => Self::Incomplete,
            "failed" => Self::Failed,
            _ => Self::Other(value),
        })
    }
}

/// One Responses output item. Unknown and provider-hosted tool items retain their
/// complete JSON object so future wire additions remain replayable.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum OutputItem {
    Message(MessageItemWire),
    Reasoning(ReasoningItemWire),
    FunctionCall(FunctionCallItemWire),
    CustomToolCall(CustomToolCallItemWire),
    Program(ProgramItemWire),
    ProgramOutput(ProgramOutputItemWire),
    ProviderTool(ProviderToolItemWire),
    Unknown(UnknownOutputItemWire),
}

impl OutputItem {
    pub fn kind(&self) -> &str {
        match self {
            Self::Message(_) => "message",
            Self::Reasoning(_) => "reasoning",
            Self::FunctionCall(_) => "function_call",
            Self::CustomToolCall(_) => "custom_tool_call",
            Self::Program(_) => "program",
            Self::ProgramOutput(_) => "program_output",
            Self::ProviderTool(item) => item.kind(),
            Self::Unknown(item) => item.kind(),
        }
    }

    pub fn id(&self) -> Option<&str> {
        match self {
            Self::Message(item) => Some(item.id.as_str()),
            Self::Reasoning(item) => item.id.as_deref(),
            Self::FunctionCall(item) => item.id.as_deref(),
            Self::CustomToolCall(item) => item.id.as_deref(),
            Self::Program(item) => Some(item.id.as_str()),
            Self::ProgramOutput(item) => Some(item.id.as_str()),
            Self::ProviderTool(item) => item.id(),
            Self::Unknown(item) => item.id(),
        }
    }

    pub fn call_id(&self) -> Option<&str> {
        match self {
            Self::FunctionCall(item) => Some(item.call_id.as_str()),
            Self::CustomToolCall(item) => Some(item.call_id.as_str()),
            Self::Program(item) => Some(item.call_id.as_str()),
            Self::ProgramOutput(item) => Some(item.call_id.as_str()),
            Self::ProviderTool(item) => item.call_id(),
            Self::Unknown(item) => item.call_id(),
            Self::Message(_) | Self::Reasoning(_) => None,
        }
    }

    pub fn status(&self) -> Option<&str> {
        match self {
            Self::Message(item) => item.status.as_ref().map(ItemStatus::as_str),
            Self::Reasoning(item) => item.status.as_ref().map(ItemStatus::as_str),
            Self::FunctionCall(item) => item.status.as_ref().map(ItemStatus::as_str),
            Self::CustomToolCall(item) => item.status.as_ref().map(ItemStatus::as_str),
            Self::ProgramOutput(item) => Some(item.status.as_str()),
            Self::ProviderTool(item) => item.status(),
            Self::Unknown(item) => item.status(),
            Self::Program(_) => None,
        }
    }

    pub fn to_value(&self) -> Result<Value, serde_json::Error> {
        match self {
            Self::Message(item) => preserved_or_serialize(item.raw.as_ref(), item),
            Self::Reasoning(item) => preserved_or_serialize(item.raw.as_ref(), item),
            Self::FunctionCall(item) => preserved_or_serialize(item.raw.as_ref(), item),
            Self::CustomToolCall(item) => preserved_or_serialize(item.raw.as_ref(), item),
            Self::Program(item) => preserved_or_serialize(item.raw.as_ref(), item),
            Self::ProgramOutput(item) => preserved_or_serialize(item.raw.as_ref(), item),
            Self::ProviderTool(item) => Ok(Value::Object(item.raw.clone())),
            Self::Unknown(item) => Ok(Value::Object(item.raw.clone())),
        }
    }
}

impl Serialize for OutputItem {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.to_value()
            .map_err(serde::ser::Error::custom)?
            .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for OutputItem {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        let kind = value
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| serde::de::Error::custom("Responses output item omitted `type`"))?;

        macro_rules! known {
            ($variant:ident, $wire:ty) => {{
                let raw = into_object::<D::Error>(value.clone())?;
                let mut item =
                    serde_json::from_value::<$wire>(value).map_err(serde::de::Error::custom)?;
                item.raw = Some(raw);
                Ok(Self::$variant(item))
            }};
        }

        match kind {
            "message" => known!(Message, MessageItemWire),
            "reasoning" => known!(Reasoning, ReasoningItemWire),
            "function_call" => known!(FunctionCall, FunctionCallItemWire),
            "custom_tool_call" => known!(CustomToolCall, CustomToolCallItemWire),
            "program" => known!(Program, ProgramItemWire),
            "program_output" => known!(ProgramOutput, ProgramOutputItemWire),
            kind if is_provider_tool_kind(kind) => {
                let raw = into_object::<D::Error>(value)?;
                Ok(Self::ProviderTool(ProviderToolItemWire { raw }))
            }
            _ => {
                let raw = into_object::<D::Error>(value)?;
                Ok(Self::Unknown(UnknownOutputItemWire { raw }))
            }
        }
    }
}

fn preserved_or_serialize<T: Serialize>(
    raw: Option<&Map<String, Value>>,
    typed: &T,
) -> Result<Value, serde_json::Error> {
    raw.cloned()
        .map(Value::Object)
        .map(Ok)
        .unwrap_or_else(|| serde_json::to_value(typed))
}

fn is_provider_tool_kind(kind: &str) -> bool {
    matches!(
        kind,
        "apply_patch_call"
            | "apply_patch_call_output"
            | "code_interpreter_call"
            | "computer_call"
            | "computer_call_output"
            | "custom_tool_call_output"
            | "file_search_call"
            | "function_call_output"
            | "image_generation_call"
            | "local_shell_call"
            | "local_shell_call_output"
            | "mcp_approval_request"
            | "mcp_call"
            | "mcp_list_tools"
            | "shell_call"
            | "shell_call_output"
            | "tool_search_call"
            | "tool_search_output"
            | "web_search_call"
    )
}

fn into_object<E>(value: Value) -> Result<Map<String, Value>, E>
where
    E: serde::de::Error,
{
    value
        .as_object()
        .cloned()
        .ok_or_else(|| E::custom("Responses output item must be a JSON object"))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessageItemWire {
    #[serde(rename = "type")]
    pub kind: String,
    pub id: String,
    #[serde(default)]
    pub status: Option<ItemStatus>,
    pub role: String,
    #[serde(default)]
    pub content: Vec<OutputContentPart>,
    #[serde(default)]
    pub phase: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
    #[serde(skip)]
    pub raw: Option<Map<String, Value>>,
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum OutputContentPart {
    Text(OutputTextWire),
    Refusal(OutputRefusalWire),
    Unknown(UnknownContentPartWire),
}

impl Serialize for OutputContentPart {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Text(part) => part.serialize(serializer),
            Self::Refusal(part) => part.serialize(serializer),
            Self::Unknown(part) => Value::Object(part.raw.clone()).serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for OutputContentPart {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value.get("type").and_then(Value::as_str) {
            Some("output_text") => serde_json::from_value(value)
                .map(Self::Text)
                .map_err(serde::de::Error::custom),
            Some("refusal") => serde_json::from_value(value)
                .map(Self::Refusal)
                .map_err(serde::de::Error::custom),
            Some(_) => Ok(Self::Unknown(UnknownContentPartWire {
                raw: into_object::<D::Error>(value)?,
            })),
            None => Err(serde::de::Error::custom(
                "Responses content part omitted `type`",
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputTextWire {
    #[serde(rename = "type")]
    pub kind: String,
    pub text: String,
    #[serde(default)]
    pub annotations: Vec<AnnotationWire>,
    #[serde(default)]
    pub logprobs: Option<Value>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputRefusalWire {
    #[serde(rename = "type")]
    pub kind: String,
    pub refusal: String,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnknownContentPartWire {
    pub raw: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnnotationWire {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(flatten)]
    pub fields: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReasoningItemWire {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub summary: Vec<ReasoningTextWire>,
    #[serde(default)]
    pub content: Vec<ReasoningTextWire>,
    #[serde(default)]
    pub encrypted_content: Option<String>,
    #[serde(default)]
    pub status: Option<ItemStatus>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
    #[serde(skip)]
    pub raw: Option<Map<String, Value>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReasoningTextWire {
    #[serde(rename = "type")]
    pub kind: String,
    pub text: String,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallerWire {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub caller_id: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCallItemWire {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub id: Option<String>,
    pub call_id: String,
    pub name: String,
    pub arguments: String,
    #[serde(default)]
    pub namespace: Option<String>,
    #[serde(default)]
    pub caller: Option<ToolCallerWire>,
    #[serde(default)]
    pub status: Option<ItemStatus>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
    #[serde(skip)]
    pub raw: Option<Map<String, Value>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CustomToolCallItemWire {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub id: Option<String>,
    pub call_id: String,
    pub name: String,
    pub input: String,
    #[serde(default)]
    pub namespace: Option<String>,
    #[serde(default)]
    pub caller: Option<ToolCallerWire>,
    #[serde(default)]
    pub status: Option<ItemStatus>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
    #[serde(skip)]
    pub raw: Option<Map<String, Value>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgramItemWire {
    #[serde(rename = "type")]
    pub kind: String,
    pub id: String,
    pub call_id: String,
    pub code: String,
    pub fingerprint: String,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
    #[serde(skip)]
    pub raw: Option<Map<String, Value>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgramOutputItemWire {
    #[serde(rename = "type")]
    pub kind: String,
    pub id: String,
    pub call_id: String,
    pub result: String,
    pub status: ItemStatus,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
    #[serde(skip)]
    pub raw: Option<Map<String, Value>>,
}

/// Provider-hosted tool items are intentionally represented by their complete
/// object. Their schemas evolve independently and are not portable core data.
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderToolItemWire {
    pub raw: Map<String, Value>,
}

impl ProviderToolItemWire {
    pub fn kind(&self) -> &str {
        self.raw
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("unknown_provider_tool")
    }

    pub fn id(&self) -> Option<&str> {
        self.raw.get("id").and_then(Value::as_str)
    }

    pub fn call_id(&self) -> Option<&str> {
        self.raw.get("call_id").and_then(Value::as_str)
    }

    pub fn status(&self) -> Option<&str> {
        self.raw.get("status").and_then(Value::as_str)
    }

    pub fn caller(&self) -> Option<&Value> {
        self.raw.get("caller")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnknownOutputItemWire {
    pub raw: Map<String, Value>,
}

impl UnknownOutputItemWire {
    pub fn kind(&self) -> &str {
        self.raw
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
    }

    pub fn id(&self) -> Option<&str> {
        self.raw.get("id").and_then(Value::as_str)
    }

    pub fn status(&self) -> Option<&str> {
        self.raw.get("status").and_then(Value::as_str)
    }

    pub fn call_id(&self) -> Option<&str> {
        self.raw.get("call_id").and_then(Value::as_str)
    }

    pub fn caller(&self) -> Option<&Value> {
        self.raw.get("caller")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IncompleteDetailsWire {
    pub reason: String,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseReasoningConfigWire {
    #[serde(default)]
    pub effort: Option<String>,
    #[serde(default)]
    pub summary: Option<Value>,
    #[serde(default)]
    pub context: Option<String>,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseErrorWire {
    #[serde(default)]
    pub code: Option<String>,
    pub message: String,
    #[serde(default)]
    pub param: Option<String>,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseUsageWire {
    pub input_tokens: u64,
    #[serde(default)]
    pub input_tokens_details: Option<InputTokenDetailsWire>,
    pub output_tokens: u64,
    #[serde(default)]
    pub output_tokens_details: Option<OutputTokenDetailsWire>,
    pub total_tokens: u64,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputTokenDetailsWire {
    #[serde(default)]
    pub cached_tokens: Option<u64>,
    #[serde(default)]
    pub cache_write_tokens: Option<u64>,
    #[serde(default)]
    pub orchestration_input_tokens: Option<u64>,
    #[serde(default)]
    pub orchestration_input_cached_tokens: Option<u64>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputTokenDetailsWire {
    #[serde(default)]
    pub reasoning_tokens: Option<u64>,
    #[serde(default)]
    pub orchestration_output_tokens: Option<u64>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// One decoded SSE JSON event. Event-specific payloads remain available through
/// typed accessors in the stream decoder and the original JSON value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamEventWire {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub sequence_number: Option<u64>,
    #[serde(flatten)]
    pub fields: BTreeMap<String, Value>,
}

impl StreamEventWire {
    pub fn field(&self, name: &str) -> Option<&Value> {
        self.fields.get(name)
    }

    pub fn response(&self) -> Result<Option<ResponseWire>, serde_json::Error> {
        self.field("response")
            .cloned()
            .map(serde_json::from_value)
            .transpose()
    }

    pub fn item(&self) -> Result<Option<OutputItem>, serde_json::Error> {
        self.field("item")
            .cloned()
            .map(serde_json::from_value)
            .transpose()
    }

    pub fn to_value(&self) -> Result<Value, serde_json::Error> {
        serde_json::to_value(self)
    }
}
