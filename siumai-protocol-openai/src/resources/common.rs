use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

/// Provider metadata accepted by OpenAI resource APIs.
pub type OpenAiMetadata = BTreeMap<String, String>;

/// Cursor ordering shared by OpenAI resource list endpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiListOrder {
    Asc,
    Desc,
}

impl OpenAiListOrder {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Asc => "asc",
            Self::Desc => "desc",
        }
    }
}

/// A forward-compatible OpenAI cursor page.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAiCursorPage<T> {
    pub object: String,
    pub data: Vec<T>,
    #[serde(default)]
    pub first_id: Option<String>,
    #[serde(default)]
    pub last_id: Option<String>,
    pub has_more: bool,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// A provider-native resource value could not be represented safely.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum OpenAiResourceCodecError {
    #[error("OpenAI conversation input item must be a JSON object")]
    ConversationItemNotObject,
    #[error("OpenAI conversation input item must contain a non-empty string type")]
    ConversationItemTypeMissing,
    #[error(
        "OpenAI resource string must be non-empty, trimmed, bounded, and free of control characters"
    )]
    InvalidResourceString,
}
