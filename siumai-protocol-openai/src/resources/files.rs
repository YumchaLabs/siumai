use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::OpenAiResourceCodecError;

/// An open OpenAI file-purpose identifier.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct OpenAiFilePurpose(String);

impl OpenAiFilePurpose {
    pub const ASSISTANTS: &'static str = "assistants";
    pub const ASSISTANTS_OUTPUT: &'static str = "assistants_output";
    pub const BATCH: &'static str = "batch";
    pub const BATCH_OUTPUT: &'static str = "batch_output";
    pub const FINE_TUNE: &'static str = "fine-tune";
    pub const FINE_TUNE_RESULTS: &'static str = "fine-tune-results";
    pub const VISION: &'static str = "vision";
    pub const USER_DATA: &'static str = "user_data";
    pub const EVALUATIONS: &'static str = "evaluations";

    pub fn new(value: impl Into<String>) -> Result<Self, OpenAiResourceCodecError> {
        let value = value.into();
        if value.is_empty()
            || value.len() > 64
            || value != value.trim()
            || value.chars().any(char::is_control)
        {
            return Err(OpenAiResourceCodecError::InvalidResourceString);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// The only currently supported file-expiration anchor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiFileExpirationAnchor {
    CreatedAt,
}

/// File expiration configured relative to creation time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiFileExpiresAfter {
    pub anchor: OpenAiFileExpirationAnchor,
    pub seconds: u32,
}

impl OpenAiFileExpiresAfter {
    pub const fn after_created(seconds: u32) -> Self {
        Self {
            anchor: OpenAiFileExpirationAnchor::CreatedAt,
            seconds,
        }
    }
}

/// OpenAI File metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAiFile {
    pub id: String,
    pub object: String,
    pub bytes: u64,
    pub created_at: i64,
    #[serde(default)]
    pub expires_at: Option<i64>,
    pub filename: String,
    pub purpose: OpenAiFilePurpose,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub status_details: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Deletion acknowledgement for a file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAiFileDeleted {
    pub id: String,
    pub object: String,
    pub deleted: bool,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn file_decoder_keeps_future_fields_and_purpose_values() {
        let file: OpenAiFile = serde_json::from_value(json!({
            "id": "file_123",
            "object": "file",
            "bytes": 12,
            "created_at": 1,
            "filename": "data.jsonl",
            "purpose": "future-purpose",
            "future": true
        }))
        .unwrap();
        assert_eq!(file.purpose.as_str(), "future-purpose");
        assert_eq!(file.extra["future"], true);
    }
}
