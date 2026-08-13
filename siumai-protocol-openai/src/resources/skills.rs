use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// OpenAI skill metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAiSkill {
    pub id: String,
    pub object: String,
    pub created_at: i64,
    #[serde(default)]
    pub updated_at: Option<i64>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub default_version: Option<String>,
    #[serde(default)]
    pub latest_version: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Request body for moving a skill's default-version pointer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiSkillUpdateRequest {
    pub default_version: String,
}

impl OpenAiSkillUpdateRequest {
    pub fn new(default_version: impl Into<String>) -> Self {
        Self {
            default_version: default_version.into(),
        }
    }
}

/// Immutable OpenAI skill-version metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAiSkillVersion {
    pub id: String,
    pub object: String,
    pub created_at: i64,
    pub skill_id: String,
    pub version: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Deletion acknowledgement for a skill.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAiDeletedSkill {
    pub id: String,
    pub object: String,
    pub deleted: bool,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Deletion acknowledgement for an immutable skill version.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAiDeletedSkillVersion {
    pub id: String,
    pub object: String,
    pub deleted: bool,
    pub version: String,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn skill_decoder_accepts_nullable_descriptive_fields() {
        let skill: OpenAiSkill = serde_json::from_value(json!({
            "id": "skill_123",
            "object": "skill",
            "created_at": 1,
            "name": null,
            "description": null,
            "default_version": "1",
            "latest_version": "2",
            "future": true
        }))
        .unwrap();
        assert!(skill.name.is_none());
        assert_eq!(skill.extra["future"], true);
    }
}
