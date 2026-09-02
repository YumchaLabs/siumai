use crate::usage::Usage;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use std::fmt;
use thiserror::Error;
/// A streamed structured value that has not passed final schema validation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PartialStructuredOutput {
    value: Value,
}

impl PartialStructuredOutput {
    pub fn unvalidated(value: Value) -> Self {
        Self { value }
    }

    pub fn value(&self) -> &Value {
        &self.value
    }
}

/// Why a language generation completed successfully.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum LanguageCompletionReason {
    Stop,
    ToolCalls,
    Refusal,
    Other(String),
}

/// Why a language generation ended before normal completion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum LanguageIncompleteReason {
    MaxOutputTokens,
    ContentFilter,
    Other(String),
}

/// The single provider-neutral termination axis for a successful language call.
///
/// Provider failures and cancellations are call outcomes rather than response
/// states. Queued and in-progress resources remain provider-native.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum LanguageTermination {
    Completed(LanguageCompletionReason),
    Incomplete(LanguageIncompleteReason),
}

/// Default maximum number of observational content items in one partial output.
pub const DEFAULT_PARTIAL_LANGUAGE_OUTPUT_ITEM_COUNT_LIMIT: usize = 128;
/// Default maximum encoded size of one observational partial-output item.
pub const DEFAULT_PARTIAL_LANGUAGE_OUTPUT_ITEM_BYTE_LIMIT: usize = 1024 * 1024;
/// Default maximum encoded size of one partial output, including its usage.
pub const DEFAULT_PARTIAL_LANGUAGE_OUTPUT_TOTAL_BYTE_LIMIT: usize = 16 * 1024 * 1024;

/// Aggregate limits for a failed or cancelled call's observational output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartialLanguageOutputBudget {
    max_items: usize,
    max_item_bytes: usize,
    max_total_bytes: usize,
}

impl PartialLanguageOutputBudget {
    pub const fn new(max_items: usize, max_item_bytes: usize, max_total_bytes: usize) -> Self {
        Self {
            max_items,
            max_item_bytes,
            max_total_bytes,
        }
    }

    pub const fn max_items(self) -> usize {
        self.max_items
    }

    pub const fn max_item_bytes(self) -> usize {
        self.max_item_bytes
    }

    pub const fn max_total_bytes(self) -> usize {
        self.max_total_bytes
    }
}

impl Default for PartialLanguageOutputBudget {
    fn default() -> Self {
        Self::new(
            DEFAULT_PARTIAL_LANGUAGE_OUTPUT_ITEM_COUNT_LIMIT,
            DEFAULT_PARTIAL_LANGUAGE_OUTPUT_ITEM_BYTE_LIMIT,
            DEFAULT_PARTIAL_LANGUAGE_OUTPUT_TOTAL_BYTE_LIMIT,
        )
    }
}

/// Non-executable content observed before a language call failed or was cancelled.
///
/// This deliberately cannot represent tool calls, tool results, provider-native
/// state, replay material, media, citations, or executable annotations.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PartialLanguageOutputPart {
    Text { text: String },
    Reasoning { text: String },
    Refusal { reason: Option<String> },
}

impl fmt::Debug for PartialLanguageOutputPart {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text { text } => formatter
                .debug_struct("Text")
                .field("utf8_bytes", &text.len())
                .finish_non_exhaustive(),
            Self::Reasoning { text } => formatter
                .debug_struct("Reasoning")
                .field("utf8_bytes", &text.len())
                .finish_non_exhaustive(),
            Self::Refusal { reason } => formatter
                .debug_struct("Refusal")
                .field("has_reason", &reason.is_some())
                .field("reason_utf8_bytes", &reason.as_ref().map(String::len))
                .finish_non_exhaustive(),
        }
    }
}

/// Bounded observational output retained for a failed or cancelled language call.
#[derive(Clone, PartialEq, Serialize)]
pub struct PartialLanguageOutput {
    content: Vec<PartialLanguageOutputPart>,
    usage: Usage,
    #[serde(skip)]
    encoded_json_bytes: usize,
}

impl fmt::Debug for PartialLanguageOutput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PartialLanguageOutput")
            .field("items", &self.content.len())
            .field("encoded_json_bytes", &self.encoded_json_bytes)
            .field("usage_provider_fields", &self.usage.provider.len())
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum PartialLanguageOutputError {
    #[error("partial language output contains {actual} items; maximum is {maximum}")]
    TooManyItems { actual: usize, maximum: usize },
    #[error("partial language output item {index} is {actual} encoded bytes; maximum is {maximum}")]
    ItemTooLarge {
        index: usize,
        actual: usize,
        maximum: usize,
    },
    #[error("partial language output is {actual} encoded bytes; maximum is {maximum}")]
    TooLarge { actual: usize, maximum: usize },
    #[error("failed to measure partial language output")]
    Encoding,
}

impl PartialLanguageOutput {
    pub fn new(
        content: Vec<PartialLanguageOutputPart>,
        usage: Usage,
    ) -> Result<Self, PartialLanguageOutputError> {
        Self::with_budget(content, usage, PartialLanguageOutputBudget::default())
    }

    pub fn with_budget(
        content: Vec<PartialLanguageOutputPart>,
        mut usage: Usage,
        budget: PartialLanguageOutputBudget,
    ) -> Result<Self, PartialLanguageOutputError> {
        // Provider-owned usage metadata may contain replay or tenant details.
        // Partial output intentionally retains only the stable portable counts.
        usage.provider.clear();
        let encoded_json_bytes = validate_partial_language_output(&content, &usage, budget)?;
        Ok(Self {
            content,
            usage,
            encoded_json_bytes,
        })
    }

    pub fn content(&self) -> &[PartialLanguageOutputPart] {
        &self.content
    }

    /// Borrow canonical text observations in their original order.
    ///
    /// Reasoning and refusal observations are intentionally excluded.
    pub fn text_parts(&self) -> impl Iterator<Item = &str> + '_ {
        self.content.iter().filter_map(|part| match part {
            PartialLanguageOutputPart::Text { text } => Some(text.as_str()),
            PartialLanguageOutputPart::Reasoning { .. }
            | PartialLanguageOutputPart::Refusal { .. } => None,
        })
    }

    /// Concatenate canonical text observations without separators or normalization.
    ///
    /// Returns `None` when no text observation exists and `Some("")` when one
    /// or more text observations concatenate to an empty string.
    pub fn output_text(&self) -> Option<String> {
        concatenate_text_parts(self.text_parts())
    }

    pub fn usage(&self) -> &Usage {
        &self.usage
    }

    pub fn encoded_json_bytes(&self) -> usize {
        self.encoded_json_bytes
    }

    pub fn into_parts(self) -> (Vec<PartialLanguageOutputPart>, Usage) {
        (self.content, self.usage)
    }

    pub fn validate(&self) -> Result<(), PartialLanguageOutputError> {
        self.validate_with_budget(PartialLanguageOutputBudget::default())
    }

    pub fn validate_with_budget(
        &self,
        budget: PartialLanguageOutputBudget,
    ) -> Result<(), PartialLanguageOutputError> {
        validate_partial_language_output(&self.content, &self.usage, budget).map(|_| ())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PartialLanguageOutputWire {
    content: Vec<PartialLanguageOutputPart>,
    usage: Usage,
}

#[derive(Serialize)]
struct PartialLanguageOutputWireRef<'a> {
    content: &'a [PartialLanguageOutputPart],
    usage: &'a Usage,
}

impl<'de> Deserialize<'de> for PartialLanguageOutput {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = PartialLanguageOutputWire::deserialize(deserializer)?;
        Self::new(wire.content, wire.usage).map_err(serde::de::Error::custom)
    }
}

pub(crate) fn concatenate_text_parts<'a>(
    mut parts: impl Iterator<Item = &'a str>,
) -> Option<String> {
    let first = parts.next()?;
    let mut output = String::from(first);
    for part in parts {
        output.push_str(part);
    }
    Some(output)
}

fn validate_partial_language_output(
    content: &[PartialLanguageOutputPart],
    usage: &Usage,
    budget: PartialLanguageOutputBudget,
) -> Result<usize, PartialLanguageOutputError> {
    if content.len() > budget.max_items {
        return Err(PartialLanguageOutputError::TooManyItems {
            actual: content.len(),
            maximum: budget.max_items,
        });
    }

    for (index, item) in content.iter().enumerate() {
        let actual = serde_json::to_vec(item)
            .map_err(|_| PartialLanguageOutputError::Encoding)?
            .len();
        if actual > budget.max_item_bytes {
            return Err(PartialLanguageOutputError::ItemTooLarge {
                index,
                actual,
                maximum: budget.max_item_bytes,
            });
        }
    }

    let actual = serde_json::to_vec(&PartialLanguageOutputWireRef { content, usage })
        .map_err(|_| PartialLanguageOutputError::Encoding)?
        .len();
    if actual > budget.max_total_bytes {
        return Err(PartialLanguageOutputError::TooLarge {
            actual,
            maximum: budget.max_total_bytes,
        });
    }
    Ok(actual)
}
