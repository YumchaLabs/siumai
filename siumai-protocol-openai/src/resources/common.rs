use std::{collections::BTreeMap, fmt};

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
#[derive(Clone, PartialEq, Serialize, Deserialize)]
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

impl<T> fmt::Debug for OpenAiCursorPage<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiCursorPage")
            .field("data_len", &self.data.len())
            .field("first_id_present", &self.first_id.is_some())
            .field("last_id_present", &self.last_id.is_some())
            .field("has_more", &self.has_more)
            .field("extra_fields", &self.extra.len())
            .field("data", &"<redacted>")
            .finish()
    }
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn cursor_page_debug_redacts_ids_items_and_provider_extensions() {
        let sentinel = "cursor-page-debug-sentinel";
        let page = OpenAiCursorPage {
            object: sentinel.to_string(),
            data: vec![json!({"private": sentinel})],
            first_id: Some(sentinel.to_string()),
            last_id: Some(sentinel.to_string()),
            has_more: true,
            extra: BTreeMap::from([("private".to_string(), json!(sentinel))]),
        };

        let debug = format!("{page:?}");
        assert!(!debug.contains(sentinel));
        assert!(debug.contains("data_len"));
        assert!(debug.contains("<redacted>"));
    }
}
