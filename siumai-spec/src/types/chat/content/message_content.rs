use serde::{Deserialize, Serialize};

use super::ContentPart;

/// Message content - supports multimodality
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MessageContent {
    /// Plain text
    Text(String),
    /// Multimodal content
    MultiModal(Vec<ContentPart>),
    /// Structured JSON content (optional feature)
    #[cfg(feature = "structured-messages")]
    Json(serde_json::Value),
}

impl MessageContent {
    /// Extract text content if available
    pub fn text(&self) -> Option<&str> {
        match self {
            MessageContent::Text(text) => Some(text),
            MessageContent::MultiModal(parts) => {
                // Return the first text part found
                for part in parts {
                    if let ContentPart::Text { text, .. } = part {
                        return Some(text);
                    }
                }
                None
            }
            #[cfg(feature = "structured-messages")]
            MessageContent::Json(_) => None,
        }
    }

    /// Extract all text content
    pub fn all_text(&self) -> String {
        match self {
            MessageContent::Text(text) => text.clone(),
            MessageContent::MultiModal(parts) => {
                let mut result = String::new();
                for part in parts {
                    if let ContentPart::Text { text, .. } = part {
                        if !result.is_empty() {
                            result.push(' ');
                        }
                        result.push_str(text);
                    }
                }
                result
            }
            #[cfg(feature = "structured-messages")]
            MessageContent::Json(v) => serde_json::to_string(v).unwrap_or_default(),
        }
    }

    /// Get multimodal content parts if this is multimodal content
    pub fn as_multimodal(&self) -> Option<&Vec<ContentPart>> {
        match self {
            MessageContent::MultiModal(parts) => Some(parts),
            _ => None,
        }
    }
}
