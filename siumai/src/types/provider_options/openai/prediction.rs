//! Predicted Outputs types for OpenAI
//!
//! Configuration for Predicted Outputs, which can greatly improve response times
//! when large parts of the model response are known ahead of time.
//!
//! Learn more: https://platform.openai.com/docs/guides/predicted-outputs

use serde::{Deserialize, Serialize};

/// Predicted output content
///
/// Static predicted output content, such as the content of a text file that is
/// being regenerated with minor changes.
///
/// # Example
///
/// ```rust
/// use siumai::types::provider_options::openai::PredictionContent;
///
/// // Predict text content
/// let prediction = PredictionContent::text("The original file content...");
///
/// // Or use multimodal content parts
/// let prediction = PredictionContent::parts(vec![
///     siumai::types::ContentPart::text("Part 1"),
///     siumai::types::ContentPart::text("Part 2"),
/// ]);
/// ```
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
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::provider_options::openai::PredictionContent;
    ///
    /// let prediction = PredictionContent::text("The original file content...");
    /// ```
    pub fn text<S: Into<String>>(content: S) -> Self {
        Self {
            content_type: "content".to_string(),
            content: PredictionContentData::Text {
                content: content.into(),
            },
        }
    }

    /// Create a prediction with content parts
    ///
    /// # Example
    ///
    /// ```rust
    /// use siumai::types::provider_options::openai::PredictionContent;
    /// use siumai::types::ContentPart;
    ///
    /// let prediction = PredictionContent::parts(vec![
    ///     ContentPart::text("Part 1"),
    ///     ContentPart::text("Part 2"),
    /// ]);
    /// ```
    pub fn parts(content: Vec<crate::types::ContentPart>) -> Self {
        Self {
            content_type: "content".to_string(),
            content: PredictionContentData::Parts { content },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction_text_serialization() {
        let prediction = PredictionContent::text("Hello, world!");
        let json = serde_json::to_value(&prediction).unwrap();

        assert_eq!(json["type"], "content");
        assert_eq!(json["content"], "Hello, world!");
    }

    #[test]
    fn test_prediction_parts_serialization() {
        let prediction = PredictionContent::parts(vec![
            crate::types::ContentPart::text("Part 1"),
            crate::types::ContentPart::text("Part 2"),
        ]);
        let json = serde_json::to_value(&prediction).unwrap();

        assert_eq!(json["type"], "content");
        assert!(json["content"].is_array());
        assert_eq!(json["content"].as_array().unwrap().len(), 2);
    }
}
