//! Predicted Outputs types for OpenAI
//!
//! Learn more: https://platform.openai.com/docs/guides/predicted-outputs

use serde::{Deserialize, Serialize};

/// Predicted output content
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PredictionContent {
    /// The type of the predicted content (always "content")
    #[serde(rename = "type")]
    pub content_type: String,

    /// The content that should be matched when generating a model response
    #[serde(flatten)]
    pub content: PredictionContentData,
}

/// Prediction content data
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum PredictionContentData {
    /// Text content
    Text {
        /// The text content used for prediction
        content: String,
    },
    /// Array of content parts
    Parts {
        /// Array of content parts
        content: Vec<crate::types::ContentPart>,
    },
}

impl PredictionContent {
    /// Create a prediction with text content
    pub fn text<S: Into<String>>(content: S) -> Self {
        Self {
            content_type: "content".to_string(),
            content: PredictionContentData::Text {
                content: content.into(),
            },
        }
    }

    /// Create a prediction with content parts
    pub fn parts(content: Vec<crate::types::ContentPart>) -> Self {
        Self {
            content_type: "content".to_string(),
            content: PredictionContentData::Parts { content },
        }
    }
}

