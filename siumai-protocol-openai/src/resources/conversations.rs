use std::collections::BTreeMap;

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Value, json};

use super::{OpenAiMetadata, OpenAiResourceCodecError};

/// A role accepted by an OpenAI conversation message input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiConversationRole {
    User,
    Assistant,
    System,
    Developer,
}

impl OpenAiConversationRole {
    const fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::System => "system",
            Self::Developer => "developer",
        }
    }
}

/// A checked provider-native item accepted by Conversations create APIs.
///
/// OpenAI intentionally reuses the broad Responses input-item union here. The
/// checked opaque carrier preserves future item kinds without pretending that
/// every item is portable.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(transparent)]
pub struct OpenAiConversationInputItem(Value);

impl OpenAiConversationInputItem {
    pub fn from_value(value: Value) -> Result<Self, OpenAiResourceCodecError> {
        let Some(object) = value.as_object() else {
            return Err(OpenAiResourceCodecError::ConversationItemNotObject);
        };
        if object
            .get("type")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
        {
            return Err(OpenAiResourceCodecError::ConversationItemTypeMissing);
        }
        Ok(Self(value))
    }

    pub fn message(role: OpenAiConversationRole, content: impl Into<String>) -> Self {
        Self(json!({
            "type": "message",
            "role": role.as_str(),
            "content": content.into(),
        }))
    }

    pub fn as_value(&self) -> &Value {
        &self.0
    }

    pub fn into_value(self) -> Value {
        self.0
    }
}

impl<'de> Deserialize<'de> for OpenAiConversationInputItem {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::from_value(Value::deserialize(deserializer)?).map_err(D::Error::custom)
    }
}

/// A lossless item returned by the Conversations API.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(transparent)]
pub struct OpenAiConversationItem(Value);

impl OpenAiConversationItem {
    pub fn as_value(&self) -> &Value {
        &self.0
    }

    pub fn into_value(self) -> Value {
        self.0
    }
}

impl<'de> Deserialize<'de> for OpenAiConversationItem {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        let Some(object) = value.as_object() else {
            return Err(D::Error::custom(
                OpenAiResourceCodecError::ConversationItemNotObject,
            ));
        };
        if object
            .get("type")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
        {
            return Err(D::Error::custom(
                OpenAiResourceCodecError::ConversationItemTypeMissing,
            ));
        }
        Ok(Self(value))
    }
}

/// Request body for creating a conversation.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OpenAiConversationCreateRequest {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub items: Vec<OpenAiConversationInputItem>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: OpenAiMetadata,
}

impl OpenAiConversationCreateRequest {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_item(mut self, item: OpenAiConversationInputItem) -> Self {
        self.items.push(item);
        self
    }

    pub fn with_metadata(mut self, metadata: OpenAiMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Request body for replacing conversation metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAiConversationUpdateRequest {
    pub metadata: OpenAiMetadata,
}

impl OpenAiConversationUpdateRequest {
    pub fn new(metadata: OpenAiMetadata) -> Self {
        Self { metadata }
    }
}

/// Request body for appending conversation items.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAiConversationItemsCreateRequest {
    pub items: Vec<OpenAiConversationInputItem>,
}

impl OpenAiConversationItemsCreateRequest {
    pub fn new(items: impl IntoIterator<Item = OpenAiConversationInputItem>) -> Self {
        Self {
            items: items.into_iter().collect(),
        }
    }
}

/// A stored OpenAI conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAiConversation {
    pub id: String,
    pub object: String,
    pub created_at: i64,
    #[serde(default)]
    pub metadata: OpenAiMetadata,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Deletion acknowledgement for a conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAiConversationDeleted {
    pub id: String,
    pub object: String,
    pub deleted: bool,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversation_items_are_checked_and_unknown_output_is_preserved() {
        assert!(OpenAiConversationInputItem::from_value(json!("encoded")).is_err());
        assert!(OpenAiConversationInputItem::from_value(json!({"role": "user"})).is_err());

        let item: OpenAiConversationItem = serde_json::from_value(json!({
            "type": "future_item",
            "future": {"kept": true}
        }))
        .unwrap();
        assert_eq!(item.as_value()["future"]["kept"], true);
    }
}
