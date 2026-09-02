use super::message::{
    AssistantHistoryOmission, AssistantHistoryOmissionKind, AssistantHistoryProjection,
    ContentPart, Message, MessagePart, MessageRole,
};
use super::opaque::{OpaqueProviderBudget, OpaqueProviderItemError};
use super::output::{
    LanguageCompletionReason, LanguageIncompleteReason, LanguageTermination, PartialLanguageOutput,
    concatenate_text_parts,
};
use crate::error::{Error as SiumaiError, PublicDiagnosticText};
use crate::provider::ModelId;
use crate::usage::Usage;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;
use thiserror::Error;
/// A direct language-call failure plus optional bounded observational output.
pub struct LanguageCallError {
    inner: Box<LanguageCallErrorInner>,
}

struct LanguageCallErrorInner {
    error: SiumaiError,
    partial: Option<PartialLanguageOutput>,
}

impl LanguageCallError {
    pub fn new(error: SiumaiError, partial: Option<PartialLanguageOutput>) -> Self {
        Self {
            inner: Box::new(LanguageCallErrorInner { error, partial }),
        }
    }

    pub fn error(&self) -> &SiumaiError {
        &self.inner.error
    }

    pub fn kind(&self) -> crate::ErrorKind {
        self.inner.error.kind()
    }

    pub fn message(&self) -> &str {
        self.inner.error.message()
    }

    pub fn context(&self) -> &crate::ErrorContext {
        self.inner.error.context()
    }

    pub fn detail(&self) -> Option<&crate::ErrorDetail> {
        self.inner.error.detail()
    }

    pub fn diagnostics(&self) -> Option<&crate::ResponseDiagnostics> {
        self.inner.error.diagnostics()
    }

    pub fn sensitive_response(&self) -> Option<&crate::SensitiveResponse> {
        self.inner.error.sensitive_response()
    }

    pub fn sensitive_source(&self) -> Option<&crate::SensitiveErrorSource> {
        self.inner.error.sensitive_source()
    }

    pub fn partial(&self) -> Option<&PartialLanguageOutput> {
        self.inner.partial.as_ref()
    }

    /// Add Registry route context without changing the partial observation.
    pub fn with_route(mut self, route: crate::RouteId) -> Self {
        self.inner.error = self.inner.error.with_route(route);
        self
    }

    /// Discard the partial observation and return the underlying sanitized error.
    pub fn into_error(self) -> SiumaiError {
        self.inner.error
    }

    pub fn into_parts(self) -> (SiumaiError, Option<PartialLanguageOutput>) {
        (self.inner.error, self.inner.partial)
    }
}

impl From<SiumaiError> for LanguageCallError {
    fn from(error: SiumaiError) -> Self {
        Self::new(error, None)
    }
}

impl fmt::Display for LanguageCallError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.error.fmt(formatter)
    }
}

impl fmt::Debug for LanguageCallError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LanguageCallError")
            .field("error", &self.inner.error)
            .field("has_partial", &self.inner.partial.is_some())
            .field(
                "partial_items",
                &self
                    .inner
                    .partial
                    .as_ref()
                    .map(|partial| partial.content().len()),
            )
            .field(
                "partial_encoded_json_bytes",
                &self
                    .inner
                    .partial
                    .as_ref()
                    .map(|partial| partial.encoded_json_bytes()),
            )
            .finish()
    }
}

impl std::error::Error for LanguageCallError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.inner.error)
    }
}

/// Stable request/result warning categories emitted without changing call success semantics.
///
/// Model lifecycle and catalog advice is provider-owned introspection and is not emitted during
/// ordinary execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum WarningKind {
    UnsupportedOption,
    IgnoredOption,
    PartialResult,
    Provider { code: PublicDiagnosticText },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Warning {
    kind: WarningKind,
    message: PublicDiagnosticText,
}

impl Warning {
    pub fn new(kind: WarningKind, message: impl Into<PublicDiagnosticText>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub fn provider(
        code: impl Into<PublicDiagnosticText>,
        message: impl Into<PublicDiagnosticText>,
    ) -> Self {
        Self::new(WarningKind::Provider { code: code.into() }, message)
    }

    pub fn kind(&self) -> &WarningKind {
        &self.kind
    }

    pub fn message(&self) -> &str {
        self.message.as_str()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LanguageResponse {
    id: Option<String>,
    model: Option<ModelId>,
    termination: LanguageTermination,
    content: Vec<ContentPart>,
    usage: Usage,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    warnings: Vec<Warning>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    provider: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum LanguageResponseError {
    #[error("custom completion reason must not be empty")]
    EmptyCompletionReason,
    #[error("custom incomplete reason must not be empty")]
    EmptyIncompleteReason,
    #[error(transparent)]
    OpaqueProviderItems(#[from] OpaqueProviderItemError),
}

impl LanguageResponse {
    pub fn new(
        termination: LanguageTermination,
        content: Vec<ContentPart>,
        usage: Usage,
    ) -> Result<Self, LanguageResponseError> {
        let response = Self {
            id: None,
            model: None,
            termination,
            content,
            usage,
            warnings: Vec::new(),
            provider: BTreeMap::new(),
        };
        response.validate()?;
        Ok(response)
    }

    pub fn completed(
        content: Vec<ContentPart>,
        reason: LanguageCompletionReason,
        usage: Usage,
    ) -> Result<Self, LanguageResponseError> {
        Self::new(LanguageTermination::Completed(reason), content, usage)
    }

    pub fn incomplete(
        content: Vec<ContentPart>,
        reason: LanguageIncompleteReason,
        usage: Usage,
    ) -> Result<Self, LanguageResponseError> {
        Self::new(LanguageTermination::Incomplete(reason), content, usage)
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn with_model(mut self, model: ModelId) -> Self {
        self.model = Some(model);
        self
    }

    pub fn with_warnings(mut self, warnings: Vec<Warning>) -> Self {
        self.warnings = warnings;
        self
    }

    pub fn with_provider_metadata(mut self, provider: BTreeMap<String, Value>) -> Self {
        self.provider = provider;
        self
    }

    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    pub fn model(&self) -> Option<&ModelId> {
        self.model.as_ref()
    }

    pub fn termination(&self) -> &LanguageTermination {
        &self.termination
    }

    pub fn content(&self) -> &[ContentPart] {
        &self.content
    }

    /// Borrow canonical text parts in their original response order.
    ///
    /// This display-oriented view excludes reasoning, refusals, tool content,
    /// citations, media, and provider-native state. Use
    /// [`Self::project_assistant_history`] when constructing assistant history.
    pub fn text_parts(&self) -> impl Iterator<Item = &str> + '_ {
        self.content.iter().filter_map(|part| match part {
            ContentPart::Text { text } => Some(text.as_str()),
            ContentPart::Reasoning { .. }
            | ContentPart::Media(_)
            | ContentPart::Citation(_)
            | ContentPart::Refusal { .. }
            | ContentPart::ToolCall(_)
            | ContentPart::ToolResult(_)
            | ContentPart::ProviderOpaque(_) => None,
        })
    }

    /// Concatenate canonical text parts without separators or normalization.
    ///
    /// Returns `None` when no text part exists and `Some("")` when one or more
    /// text parts concatenate to an empty string. This display-oriented
    /// projection does not replace [`Self::project_assistant_history`].
    pub fn output_text(&self) -> Option<String> {
        concatenate_text_parts(self.text_parts())
    }

    pub fn usage(&self) -> &Usage {
        &self.usage
    }

    pub fn warnings(&self) -> &[Warning] {
        &self.warnings
    }

    pub fn provider_metadata(&self) -> &BTreeMap<String, Value> {
        &self.provider
    }

    /// Project generated output into canonical assistant history.
    ///
    /// Citations and refusals remain available on this response but are not
    /// request content. The returned diagnostics make those omissions
    /// observable, while replay-critical provider state remains in the
    /// projected message as `ProviderOpaque` content.
    pub fn project_assistant_history(&self) -> AssistantHistoryProjection {
        let mut content = Vec::with_capacity(self.content.len());
        let mut omissions = Vec::new();

        for (content_index, part) in self.content.iter().enumerate() {
            let kind = match part {
                ContentPart::Citation(_) => Some(AssistantHistoryOmissionKind::Citation),
                ContentPart::Refusal { .. } => Some(AssistantHistoryOmissionKind::Refusal),
                ContentPart::ToolResult(_) => Some(AssistantHistoryOmissionKind::ToolResult),
                ContentPart::Text { .. }
                | ContentPart::Reasoning { .. }
                | ContentPart::Media(_)
                | ContentPart::ToolCall(_)
                | ContentPart::ProviderOpaque(_) => {
                    content.push(MessagePart::new(part.clone()));
                    None
                }
            };
            if let Some(kind) = kind {
                omissions.push(AssistantHistoryOmission {
                    content_index,
                    kind,
                });
            }
        }

        let message = (!content.is_empty()).then(|| Message::new(MessageRole::Assistant, content));
        AssistantHistoryProjection { message, omissions }
    }

    pub fn validate(&self) -> Result<(), LanguageResponseError> {
        self.validate_with_opaque_budget(OpaqueProviderBudget::default())
    }

    pub fn validate_with_opaque_budget(
        &self,
        opaque_budget: OpaqueProviderBudget,
    ) -> Result<(), LanguageResponseError> {
        validate_language_termination(&self.termination)?;
        opaque_budget.validate(self.content.iter().filter_map(|part| match part {
            ContentPart::ProviderOpaque(item) => Some(item),
            _ => None,
        }))?;
        Ok(())
    }
}

#[derive(Deserialize)]
struct LanguageResponseWire {
    id: Option<String>,
    model: Option<ModelId>,
    termination: LanguageTermination,
    content: Vec<ContentPart>,
    usage: Usage,
    #[serde(default)]
    warnings: Vec<Warning>,
    #[serde(default)]
    provider: BTreeMap<String, Value>,
}

impl<'de> Deserialize<'de> for LanguageResponse {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = LanguageResponseWire::deserialize(deserializer)?;
        let mut response = Self::new(wire.termination, wire.content, wire.usage)
            .map_err(serde::de::Error::custom)?;
        response.id = wire.id;
        response.model = wire.model;
        response.warnings = wire.warnings;
        response.provider = wire.provider;
        Ok(response)
    }
}

fn validate_language_termination(
    termination: &LanguageTermination,
) -> Result<(), LanguageResponseError> {
    match termination {
        LanguageTermination::Completed(LanguageCompletionReason::Other(reason))
            if reason.trim().is_empty() =>
        {
            Err(LanguageResponseError::EmptyCompletionReason)
        }
        LanguageTermination::Incomplete(LanguageIncompleteReason::Other(reason))
            if reason.trim().is_empty() =>
        {
            Err(LanguageResponseError::EmptyIncompleteReason)
        }
        LanguageTermination::Completed(_) | LanguageTermination::Incomplete(_) => Ok(()),
    }
}
