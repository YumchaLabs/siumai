use super::opaque::OpaqueProviderItem;
use crate::annotations::{
    ContentAnnotationTarget, ContentAnnotations, MessageAnnotationTarget, MessageAnnotations,
    ProviderAnnotationError, TypedProviderAnnotation,
};
use crate::tool::{ToolCall, ToolResult};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;
use thiserror::Error;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum MessageRole {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}

/// Owned media input or output.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum MediaData {
    Bytes(Bytes),
    Url(String),
}

impl fmt::Debug for MediaData {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bytes(bytes) => formatter
                .debug_struct("Bytes")
                .field("len", &bytes.len())
                .finish(),
            Self::Url(_) => formatter.debug_tuple("Url").field(&"<redacted>").finish(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaPart {
    pub media_type: String,
    pub data: MediaData,
    pub name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Citation {
    pub source_id: String,
    pub title: Option<String>,
    pub url: Option<String>,
    pub start: Option<u64>,
    pub end: Option<u64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub provider: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ContentPart {
    Text { text: String },
    Reasoning { text: String },
    Media(MediaPart),
    Citation(Citation),
    Refusal { reason: Option<String> },
    ToolCall(ToolCall),
    ToolResult(ToolResult),
    ProviderOpaque(OpaqueProviderItem),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    role: MessageRole,
    content: Vec<MessagePart>,
    #[serde(
        default,
        rename = "providerAnnotations",
        skip_serializing_if = "MessageAnnotations::is_empty"
    )]
    provider_annotations: MessageAnnotations,
}

/// One request content part plus provider-owned durable annotations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessagePart {
    content: ContentPart,
    #[serde(
        default,
        rename = "providerAnnotations",
        skip_serializing_if = "ContentAnnotations::is_empty"
    )]
    provider_annotations: ContentAnnotations,
}

impl MessagePart {
    pub fn new(content: ContentPart) -> Self {
        Self::from_parts(content, ContentAnnotations::default())
    }

    /// Rebuild a content node while preserving previously validated annotations.
    pub fn from_parts(content: ContentPart, provider_annotations: ContentAnnotations) -> Self {
        Self {
            content,
            provider_annotations,
        }
    }

    pub fn text(text: impl Into<String>) -> Self {
        Self::new(ContentPart::Text { text: text.into() })
    }

    pub fn content(&self) -> &ContentPart {
        &self.content
    }

    pub fn content_mut(&mut self) -> &mut ContentPart {
        &mut self.content
    }

    pub fn annotations(&self) -> &ContentAnnotations {
        &self.provider_annotations
    }

    pub fn with_provider_annotation<T>(
        mut self,
        annotation: &T,
    ) -> Result<Self, ProviderAnnotationError>
    where
        T: TypedProviderAnnotation<Target = ContentAnnotationTarget>,
    {
        self.provider_annotations.insert(annotation)?;
        Ok(self)
    }

    pub fn into_parts(self) -> (ContentPart, ContentAnnotations) {
        (self.content, self.provider_annotations)
    }
}

impl From<ContentPart> for MessagePart {
    fn from(content: ContentPart) -> Self {
        Self::new(content)
    }
}

impl Message {
    pub fn new<I, Part>(role: MessageRole, content: I) -> Self
    where
        I: IntoIterator<Item = Part>,
        Part: Into<MessagePart>,
    {
        Self::from_parts(role, content, MessageAnnotations::default())
    }

    /// Rebuild a message while preserving previously validated annotations.
    pub fn from_parts<I, Part>(
        role: MessageRole,
        content: I,
        provider_annotations: MessageAnnotations,
    ) -> Self
    where
        I: IntoIterator<Item = Part>,
        Part: Into<MessagePart>,
    {
        Self {
            role,
            content: content.into_iter().map(Into::into).collect(),
            provider_annotations,
        }
    }

    pub fn text(role: MessageRole, text: impl Into<String>) -> Self {
        Self::new(role, [MessagePart::text(text)])
    }

    /// Construct a system instruction containing portable text.
    pub fn system(text: impl Into<String>) -> Self {
        Self::text(MessageRole::System, text)
    }

    /// Construct a developer instruction containing portable text.
    pub fn developer(text: impl Into<String>) -> Self {
        Self::text(MessageRole::Developer, text)
    }

    /// Construct a user message containing portable text.
    pub fn user(text: impl Into<String>) -> Self {
        Self::text(MessageRole::User, text)
    }

    /// Construct an assistant message containing portable text.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self::text(MessageRole::Assistant, text)
    }

    /// Construct and validate a user message containing text and/or input media.
    pub fn user_parts<I, Part>(content: I) -> Result<Self, MessageValidationError>
    where
        I: IntoIterator<Item = Part>,
        Part: Into<MessagePart>,
    {
        Self::validated(MessageRole::User, content)
    }

    /// Construct and validate assistant history or replay content.
    pub fn assistant_parts<I, Part>(content: I) -> Result<Self, MessageValidationError>
    where
        I: IntoIterator<Item = Part>,
        Part: Into<MessagePart>,
    {
        Self::validated(MessageRole::Assistant, content)
    }

    /// Construct a canonical message for one tool result.
    pub fn tool_result(result: ToolResult) -> Self {
        Self::new(MessageRole::Tool, [ContentPart::ToolResult(result)])
    }

    /// Construct and validate a canonical message for one or more tool results.
    pub fn tool_results<I>(results: I) -> Result<Self, MessageValidationError>
    where
        I: IntoIterator<Item = ToolResult>,
    {
        Self::validated(
            MessageRole::Tool,
            results.into_iter().map(ContentPart::ToolResult),
        )
    }

    fn validated<I, Part>(role: MessageRole, content: I) -> Result<Self, MessageValidationError>
    where
        I: IntoIterator<Item = Part>,
        Part: Into<MessagePart>,
    {
        let message = Self::new(role, content);
        message.validate()?;
        Ok(message)
    }

    /// Validate the provider-neutral role/content contract for this message.
    pub fn validate(&self) -> Result<(), MessageValidationError> {
        for (content_index, part) in self.content.iter().enumerate() {
            if !content_allowed_for_role(self.role, part.content()) {
                return Err(MessageValidationError::ContentNotAllowed {
                    role: self.role,
                    content_index,
                    content_kind: content_kind(part.content()),
                });
            }
        }
        Ok(())
    }

    pub fn role(&self) -> MessageRole {
        self.role
    }

    pub fn content(&self) -> &[MessagePart] {
        &self.content
    }

    pub fn content_mut(&mut self) -> &mut Vec<MessagePart> {
        &mut self.content
    }

    pub fn annotations(&self) -> &MessageAnnotations {
        &self.provider_annotations
    }

    pub fn with_provider_annotation<T>(
        mut self,
        annotation: &T,
    ) -> Result<Self, ProviderAnnotationError>
    where
        T: TypedProviderAnnotation<Target = MessageAnnotationTarget>,
    {
        self.provider_annotations.insert(annotation)?;
        Ok(self)
    }

    pub fn into_parts(self) -> (MessageRole, Vec<MessagePart>, MessageAnnotations) {
        (self.role, self.content, self.provider_annotations)
    }
}

/// Invalid provider-neutral message role/content combination.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum MessageValidationError {
    #[error("{content_kind} content at index {content_index} is not allowed for role {role:?}")]
    ContentNotAllowed {
        role: MessageRole,
        content_index: usize,
        content_kind: &'static str,
    },
}

/// Response-only content omitted while projecting an assistant response into history.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AssistantHistoryOmissionKind {
    Citation,
    Refusal,
    ToolResult,
}

/// One observable omission made by assistant-history projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssistantHistoryOmission {
    pub(super) content_index: usize,
    pub(super) kind: AssistantHistoryOmissionKind,
}

impl AssistantHistoryOmission {
    pub const fn content_index(self) -> usize {
        self.content_index
    }

    pub const fn kind(self) -> AssistantHistoryOmissionKind {
        self.kind
    }
}

/// A role-safe assistant history message and its explicit projection diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct AssistantHistoryProjection {
    pub(super) message: Option<Message>,
    pub(super) omissions: Vec<AssistantHistoryOmission>,
}

impl AssistantHistoryProjection {
    pub fn message(&self) -> Option<&Message> {
        self.message.as_ref()
    }

    pub fn omissions(&self) -> &[AssistantHistoryOmission] {
        &self.omissions
    }

    pub fn into_parts(self) -> (Option<Message>, Vec<AssistantHistoryOmission>) {
        (self.message, self.omissions)
    }

    pub fn into_message(self) -> Option<Message> {
        self.message
    }
}

fn content_allowed_for_role(role: MessageRole, content: &ContentPart) -> bool {
    match role {
        MessageRole::System | MessageRole::Developer => {
            matches!(content, ContentPart::Text { .. })
        }
        MessageRole::User => matches!(content, ContentPart::Text { .. } | ContentPart::Media(_)),
        MessageRole::Assistant => matches!(
            content,
            ContentPart::Text { .. }
                | ContentPart::Reasoning { .. }
                | ContentPart::Media(_)
                | ContentPart::ToolCall(_)
                | ContentPart::ProviderOpaque(_)
        ),
        MessageRole::Tool => matches!(content, ContentPart::ToolResult(_)),
    }
}

fn content_kind(content: &ContentPart) -> &'static str {
    match content {
        ContentPart::Text { .. } => "text",
        ContentPart::Reasoning { .. } => "reasoning",
        ContentPart::Media(_) => "media",
        ContentPart::Citation(_) => "citation",
        ContentPart::Refusal { .. } => "refusal",
        ContentPart::ToolCall(_) => "tool call",
        ContentPart::ToolResult(_) => "tool result",
        ContentPart::ProviderOpaque(_) => "provider-opaque",
    }
}
