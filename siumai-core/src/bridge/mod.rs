//! Protocol bridge contracts.
//!
//! This module defines the protocol-agnostic contract used by request,
//! response, and stream bridges. Concrete bridge implementations should live
//! in protocol crates or gateway adapters built on top of this contract.

use serde::{Deserialize, Serialize};

/// Named protocol targets that bridge implementations can produce or consume.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BridgeTarget {
    OpenAiResponses,
    OpenAiChatCompletions,
    AnthropicMessages,
    GeminiGenerateContent,
}

impl BridgeTarget {
    /// Return a stable identifier for diagnostics and metrics.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OpenAiResponses => "openai-responses",
            Self::OpenAiChatCompletions => "openai-chat-completions",
            Self::AnthropicMessages => "anthropic-messages",
            Self::GeminiGenerateContent => "gemini-generate-content",
        }
    }
}

/// Bridge strictness policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum BridgeMode {
    Strict,
    #[default]
    BestEffort,
    ProviderTolerant,
}

/// Overall bridge outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum BridgeDecision {
    #[default]
    Exact,
    Lossy,
    Rejected,
}

impl BridgeDecision {
    /// Combine two decisions, keeping the strongest outcome.
    pub const fn combine(self, other: Self) -> Self {
        match (self, other) {
            (Self::Rejected, _) | (_, Self::Rejected) => Self::Rejected,
            (Self::Lossy, _) | (_, Self::Lossy) => Self::Lossy,
            _ => Self::Exact,
        }
    }
}

/// Typed warning categories emitted during bridging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BridgeWarningKind {
    LossyField,
    DroppedField,
    UnsupportedCapability,
    ProviderMetadataCarried,
    ProviderMetadataDropped,
    SemanticMismatch,
    Custom,
}

/// A structured warning attached to a bridge report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BridgeWarning {
    pub kind: BridgeWarningKind,
    pub path: Option<String>,
    pub message: String,
}

impl BridgeWarning {
    pub fn new<M: Into<String>>(kind: BridgeWarningKind, message: M) -> Self {
        Self {
            kind,
            path: None,
            message: message.into(),
        }
    }

    pub fn with_path<P: Into<String>, M: Into<String>>(
        kind: BridgeWarningKind,
        path: P,
        message: M,
    ) -> Self {
        Self {
            kind,
            path: Some(path.into()),
            message: message.into(),
        }
    }
}

/// Structured bridge diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BridgeReport {
    pub source: Option<BridgeTarget>,
    pub target: BridgeTarget,
    pub mode: BridgeMode,
    pub decision: BridgeDecision,
    pub warnings: Vec<BridgeWarning>,
    pub lossy_fields: Vec<String>,
    pub dropped_fields: Vec<String>,
    pub unsupported_capabilities: Vec<String>,
    pub carried_provider_metadata: Vec<String>,
}

impl BridgeReport {
    pub fn new(target: BridgeTarget, mode: BridgeMode) -> Self {
        Self::with_source(None, target, mode)
    }

    pub fn with_source(
        source: Option<BridgeTarget>,
        target: BridgeTarget,
        mode: BridgeMode,
    ) -> Self {
        Self {
            source,
            target,
            mode,
            decision: BridgeDecision::Exact,
            warnings: Vec::new(),
            lossy_fields: Vec::new(),
            dropped_fields: Vec::new(),
            unsupported_capabilities: Vec::new(),
            carried_provider_metadata: Vec::new(),
        }
    }

    pub const fn is_exact(&self) -> bool {
        matches!(self.decision, BridgeDecision::Exact)
    }

    pub const fn is_lossy(&self) -> bool {
        matches!(self.decision, BridgeDecision::Lossy)
    }

    pub const fn is_rejected(&self) -> bool {
        matches!(self.decision, BridgeDecision::Rejected)
    }

    pub const fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    pub fn add_warning(&mut self, warning: BridgeWarning) {
        let impact = match warning.kind {
            BridgeWarningKind::LossyField
            | BridgeWarningKind::DroppedField
            | BridgeWarningKind::UnsupportedCapability
            | BridgeWarningKind::ProviderMetadataDropped
            | BridgeWarningKind::SemanticMismatch => BridgeDecision::Lossy,
            BridgeWarningKind::ProviderMetadataCarried | BridgeWarningKind::Custom => {
                BridgeDecision::Exact
            }
        };
        self.decision = self.decision.combine(impact);
        self.warnings.push(warning);
    }

    pub fn mark_lossy(&mut self) {
        self.decision = self.decision.combine(BridgeDecision::Lossy);
    }

    pub fn reject<M: Into<String>>(&mut self, message: M) {
        self.decision = BridgeDecision::Rejected;
        self.warnings
            .push(BridgeWarning::new(BridgeWarningKind::Custom, message));
    }

    pub fn reject_with_warning(&mut self, warning: BridgeWarning) {
        self.decision = BridgeDecision::Rejected;
        self.warnings.push(warning);
    }

    pub fn record_lossy_field<P: Into<String>, M: Into<String>>(&mut self, path: P, message: M) {
        let path = path.into();
        self.lossy_fields.push(path.clone());
        self.add_warning(BridgeWarning::with_path(
            BridgeWarningKind::LossyField,
            path,
            message,
        ));
    }

    pub fn record_dropped_field<P: Into<String>, M: Into<String>>(&mut self, path: P, message: M) {
        let path = path.into();
        self.dropped_fields.push(path.clone());
        self.add_warning(BridgeWarning::with_path(
            BridgeWarningKind::DroppedField,
            path,
            message,
        ));
    }

    pub fn record_unsupported_capability<C: Into<String>, M: Into<String>>(
        &mut self,
        capability: C,
        message: M,
    ) {
        let capability = capability.into();
        self.unsupported_capabilities.push(capability.clone());
        self.add_warning(BridgeWarning::with_path(
            BridgeWarningKind::UnsupportedCapability,
            capability,
            message,
        ));
    }

    pub fn record_carried_provider_metadata<N: Into<String>, M: Into<String>>(
        &mut self,
        namespace: N,
        message: M,
    ) {
        let namespace = namespace.into();
        self.carried_provider_metadata.push(namespace.clone());
        self.add_warning(BridgeWarning::with_path(
            BridgeWarningKind::ProviderMetadataCarried,
            namespace,
            message,
        ));
    }

    pub fn merge(&mut self, other: Self) {
        self.decision = self.decision.combine(other.decision);
        self.warnings.extend(other.warnings);
        self.lossy_fields.extend(other.lossy_fields);
        self.dropped_fields.extend(other.dropped_fields);
        self.unsupported_capabilities
            .extend(other.unsupported_capabilities);
        self.carried_provider_metadata
            .extend(other.carried_provider_metadata);
        if self.source.is_none() {
            self.source = other.source;
        }
    }
}

/// A bridge result paired with its diagnostic report.
///
/// Rejected bridges return `value = None` and `report.decision = Rejected`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BridgeResult<T> {
    pub value: Option<T>,
    pub report: BridgeReport,
}

impl<T> BridgeResult<T> {
    pub fn new(value: T, report: BridgeReport) -> Self {
        Self {
            value: Some(value),
            report,
        }
    }

    pub fn rejected(report: BridgeReport) -> Self {
        Self {
            value: None,
            report,
        }
    }

    pub const fn is_rejected(&self) -> bool {
        self.value.is_none() || self.report.is_rejected()
    }

    pub fn map<U, F>(self, f: F) -> BridgeResult<U>
    where
        F: FnOnce(T) -> U,
    {
        BridgeResult {
            value: self.value.map(f),
            report: self.report,
        }
    }

    pub fn into_parts(self) -> (Option<T>, BridgeReport) {
        (self.value, self.report)
    }

    pub fn into_result(self) -> Result<(T, BridgeReport), BridgeReport> {
        match (self.value, self.report) {
            (Some(value), report) if !report.is_rejected() => Ok((value, report)),
            (_, report) => Err(report),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_starts_exact_and_metadata_warning_keeps_exact_decision() {
        let mut report = BridgeReport::with_source(
            Some(BridgeTarget::OpenAiResponses),
            BridgeTarget::AnthropicMessages,
            BridgeMode::BestEffort,
        );

        assert!(report.is_exact());
        assert!(!report.has_warnings());

        report.record_carried_provider_metadata(
            "openai",
            "provider metadata namespace was preserved for downstream inspection",
        );

        assert!(report.is_exact());
        assert!(report.has_warnings());
        assert_eq!(report.carried_provider_metadata, vec!["openai"]);
        assert_eq!(report.warnings.len(), 1);
        assert_eq!(
            report.warnings[0].kind,
            BridgeWarningKind::ProviderMetadataCarried
        );
    }

    #[test]
    fn lossy_and_dropped_fields_mark_report_lossy() {
        let mut report = BridgeReport::with_source(
            Some(BridgeTarget::AnthropicMessages),
            BridgeTarget::OpenAiChatCompletions,
            BridgeMode::Strict,
        );

        report.record_lossy_field(
            "messages[0].thinking",
            "thinking blocks were flattened into plain assistant text",
        );
        report.record_dropped_field(
            "messages[0].cache_control",
            "target protocol does not support prompt caching on this route",
        );
        report.record_unsupported_capability(
            "computer-use",
            "target protocol cannot express interactive computer use actions",
        );

        assert!(report.is_lossy());
        assert_eq!(report.lossy_fields, vec!["messages[0].thinking"]);
        assert_eq!(report.dropped_fields, vec!["messages[0].cache_control"]);
        assert_eq!(report.unsupported_capabilities, vec!["computer-use"]);
        assert_eq!(report.warnings.len(), 3);
    }

    #[test]
    fn merge_aggregates_details_and_promotes_stronger_decision() {
        let mut aggregate =
            BridgeReport::new(BridgeTarget::OpenAiResponses, BridgeMode::BestEffort);
        aggregate.record_carried_provider_metadata(
            "anthropic",
            "request id metadata was forwarded to the target payload",
        );

        let mut lossy = BridgeReport::new(BridgeTarget::OpenAiResponses, BridgeMode::BestEffort);
        lossy.record_lossy_field(
            "output[0].tool_result",
            "tool result details were normalized before serialization",
        );

        aggregate.merge(lossy);

        assert!(aggregate.is_lossy());
        assert_eq!(aggregate.warnings.len(), 2);
        assert_eq!(aggregate.carried_provider_metadata, vec!["anthropic"]);
        assert_eq!(aggregate.lossy_fields, vec!["output[0].tool_result"]);
    }

    #[test]
    fn rejected_result_has_no_value_and_converts_into_error_result() {
        let mut report = BridgeReport::with_source(
            Some(BridgeTarget::OpenAiResponses),
            BridgeTarget::AnthropicMessages,
            BridgeMode::Strict,
        );
        report.reject("target route requires exact compatibility");

        let result = BridgeResult::<String>::rejected(report.clone());

        assert!(result.is_rejected());
        assert!(result.value.is_none());

        let err = result.into_result().expect_err("expected rejected bridge");
        assert!(err.is_rejected());
        assert_eq!(err.warnings.len(), 1);
        assert_eq!(err.warnings[0].kind, BridgeWarningKind::Custom);
    }
}
