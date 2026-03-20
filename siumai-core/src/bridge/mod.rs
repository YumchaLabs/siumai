//! Protocol bridge contracts.
//!
//! This module defines the protocol-agnostic contract used by request,
//! response, and stream bridges. Concrete bridge implementations should live
//! in protocol crates or gateway adapters built on top of this contract.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{
    error::LlmError,
    streaming::ChatStreamEvent,
    types::{ChatRequest, ChatResponse},
};

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

/// Typed bridge primitive categories exposed to remap policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BridgePrimitiveKind {
    ToolDefinition,
    ToolChoice,
    ToolCall,
    ToolResult,
}

/// Request bridge lifecycle context.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestBridgeContext {
    pub source: Option<BridgeTarget>,
    pub target: BridgeTarget,
    pub mode: BridgeMode,
    pub route_label: Option<String>,
    pub path_label: Option<String>,
}

impl RequestBridgeContext {
    pub fn new(
        source: Option<BridgeTarget>,
        target: BridgeTarget,
        mode: BridgeMode,
        route_label: Option<String>,
        path_label: Option<String>,
    ) -> Self {
        Self {
            source,
            target,
            mode,
            route_label,
            path_label,
        }
    }
}

/// Response bridge lifecycle context.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseBridgeContext {
    pub source: Option<BridgeTarget>,
    pub target: BridgeTarget,
    pub mode: BridgeMode,
    pub route_label: Option<String>,
    pub path_label: Option<String>,
}

impl ResponseBridgeContext {
    pub fn new(
        source: Option<BridgeTarget>,
        target: BridgeTarget,
        mode: BridgeMode,
        route_label: Option<String>,
        path_label: Option<String>,
    ) -> Self {
        Self {
            source,
            target,
            mode,
            route_label,
            path_label,
        }
    }
}

/// Stream bridge lifecycle context.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamBridgeContext {
    pub source: Option<BridgeTarget>,
    pub target: BridgeTarget,
    pub mode: BridgeMode,
    pub route_label: Option<String>,
    pub path_label: Option<String>,
}

impl StreamBridgeContext {
    pub fn new(
        source: Option<BridgeTarget>,
        target: BridgeTarget,
        mode: BridgeMode,
        route_label: Option<String>,
        path_label: Option<String>,
    ) -> Self {
        Self {
            source,
            target,
            mode,
            route_label,
            path_label,
        }
    }
}

/// Primitive remap context shared by request/response/stream remappers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BridgePrimitiveContext {
    pub source: Option<BridgeTarget>,
    pub target: BridgeTarget,
    pub mode: BridgeMode,
    pub route_label: Option<String>,
    pub path_label: Option<String>,
    pub primitive: BridgePrimitiveKind,
}

impl BridgePrimitiveContext {
    pub fn new(
        source: Option<BridgeTarget>,
        target: BridgeTarget,
        mode: BridgeMode,
        route_label: Option<String>,
        path_label: Option<String>,
        primitive: BridgePrimitiveKind,
    ) -> Self {
        Self {
            source,
            target,
            mode,
            route_label,
            path_label,
            primitive,
        }
    }
}

/// Request bridge customization hook.
pub trait RequestBridgeHook: Send + Sync {
    fn transform_request(
        &self,
        _ctx: &RequestBridgeContext,
        _request: &mut ChatRequest,
        _report: &mut BridgeReport,
    ) -> Result<(), LlmError> {
        Ok(())
    }

    fn transform_json(
        &self,
        _ctx: &RequestBridgeContext,
        _body: &mut serde_json::Value,
        _report: &mut BridgeReport,
    ) -> Result<(), LlmError> {
        Ok(())
    }

    fn validate_json(
        &self,
        _ctx: &RequestBridgeContext,
        _body: &serde_json::Value,
        _report: &mut BridgeReport,
    ) -> Result<(), LlmError> {
        Ok(())
    }
}

/// Response bridge customization hook.
pub trait ResponseBridgeHook: Send + Sync {
    fn transform_response(
        &self,
        _ctx: &ResponseBridgeContext,
        _response: &mut ChatResponse,
        _report: &mut BridgeReport,
    ) -> Result<(), LlmError> {
        Ok(())
    }
}

/// Stream bridge customization hook.
pub trait StreamBridgeHook: Send + Sync {
    fn map_event(
        &self,
        _ctx: &StreamBridgeContext,
        event: ChatStreamEvent,
    ) -> Vec<ChatStreamEvent> {
        vec![event]
    }
}

/// Primitive remapper for small, reusable semantic rewrites.
pub trait BridgePrimitiveRemapper: Send + Sync {
    fn remap_tool_name(&self, _ctx: &BridgePrimitiveContext, _name: &str) -> Option<String> {
        None
    }

    fn remap_tool_call_id(&self, _ctx: &BridgePrimitiveContext, _id: &str) -> Option<String> {
        None
    }
}

/// Final decision emitted by loss policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BridgeLossAction {
    Continue,
    Reject,
}

/// Policy for deciding whether a bridge report should continue or reject.
pub trait BridgeLossPolicy: Send + Sync {
    fn request_action(&self, ctx: &RequestBridgeContext, report: &BridgeReport)
    -> BridgeLossAction;

    fn response_action(
        &self,
        ctx: &ResponseBridgeContext,
        report: &BridgeReport,
    ) -> BridgeLossAction;

    fn stream_action(&self, ctx: &StreamBridgeContext, report: &BridgeReport) -> BridgeLossAction;
}

/// Default mode-aware loss policy.
#[derive(Debug, Default)]
pub struct DefaultBridgeLossPolicy;

fn default_loss_action(mode: BridgeMode, report: &BridgeReport) -> BridgeLossAction {
    if report.is_rejected() || (matches!(mode, BridgeMode::Strict) && report.is_lossy()) {
        BridgeLossAction::Reject
    } else {
        BridgeLossAction::Continue
    }
}

impl BridgeLossPolicy for DefaultBridgeLossPolicy {
    fn request_action(
        &self,
        ctx: &RequestBridgeContext,
        report: &BridgeReport,
    ) -> BridgeLossAction {
        default_loss_action(ctx.mode, report)
    }

    fn response_action(
        &self,
        ctx: &ResponseBridgeContext,
        report: &BridgeReport,
    ) -> BridgeLossAction {
        default_loss_action(ctx.mode, report)
    }

    fn stream_action(&self, ctx: &StreamBridgeContext, report: &BridgeReport) -> BridgeLossAction {
        default_loss_action(ctx.mode, report)
    }
}

/// Shared bridge configuration surface.
#[derive(Clone)]
pub struct BridgeOptions {
    pub mode: BridgeMode,
    pub route_label: Option<String>,
    pub request_hook: Option<Arc<dyn RequestBridgeHook>>,
    pub response_hook: Option<Arc<dyn ResponseBridgeHook>>,
    pub stream_hook: Option<Arc<dyn StreamBridgeHook>>,
    pub primitive_remapper: Option<Arc<dyn BridgePrimitiveRemapper>>,
    pub loss_policy: Arc<dyn BridgeLossPolicy>,
}

impl Default for BridgeOptions {
    fn default() -> Self {
        Self::new(BridgeMode::default())
    }
}

impl BridgeOptions {
    pub fn new(mode: BridgeMode) -> Self {
        Self {
            mode,
            route_label: None,
            request_hook: None,
            response_hook: None,
            stream_hook: None,
            primitive_remapper: None,
            loss_policy: Arc::new(DefaultBridgeLossPolicy),
        }
    }

    pub fn with_route_label(mut self, route_label: impl Into<String>) -> Self {
        self.route_label = Some(route_label.into());
        self
    }

    pub fn with_request_hook(mut self, hook: Arc<dyn RequestBridgeHook>) -> Self {
        self.request_hook = Some(hook);
        self
    }

    pub fn with_response_hook(mut self, hook: Arc<dyn ResponseBridgeHook>) -> Self {
        self.response_hook = Some(hook);
        self
    }

    pub fn with_stream_hook(mut self, hook: Arc<dyn StreamBridgeHook>) -> Self {
        self.stream_hook = Some(hook);
        self
    }

    pub fn with_primitive_remapper(mut self, remapper: Arc<dyn BridgePrimitiveRemapper>) -> Self {
        self.primitive_remapper = Some(remapper);
        self
    }

    pub fn with_loss_policy(mut self, policy: Arc<dyn BridgeLossPolicy>) -> Self {
        self.loss_policy = policy;
        self
    }

    pub fn merged_with(mut self, overlay: BridgeOptions) -> Self {
        self.mode = overlay.mode;
        if overlay.route_label.is_some() {
            self.route_label = overlay.route_label;
        }
        if overlay.request_hook.is_some() {
            self.request_hook = overlay.request_hook;
        }
        if overlay.response_hook.is_some() {
            self.response_hook = overlay.response_hook;
        }
        if overlay.stream_hook.is_some() {
            self.stream_hook = overlay.stream_hook;
        }
        if overlay.primitive_remapper.is_some() {
            self.primitive_remapper = overlay.primitive_remapper;
        }
        self.loss_policy = overlay.loss_policy;
        self
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

    #[test]
    fn default_bridge_options_use_best_effort_mode() {
        let options = BridgeOptions::default();

        assert_eq!(options.mode, BridgeMode::BestEffort);
        assert!(options.route_label.is_none());
        assert!(options.request_hook.is_none());
        assert!(options.response_hook.is_none());
        assert!(options.stream_hook.is_none());
        assert!(options.primitive_remapper.is_none());
    }

    #[test]
    fn default_loss_policy_rejects_strict_lossy_reports() {
        let ctx = RequestBridgeContext::new(
            Some(BridgeTarget::AnthropicMessages),
            BridgeTarget::OpenAiChatCompletions,
            BridgeMode::Strict,
            None,
            Some("via-normalized".to_string()),
        );
        let mut report = BridgeReport::with_source(ctx.source, ctx.target, ctx.mode);
        report.record_lossy_field(
            "messages[0].thinking",
            "thinking blocks were flattened into plain assistant text",
        );

        assert_eq!(
            DefaultBridgeLossPolicy.request_action(&ctx, &report),
            BridgeLossAction::Reject
        );
    }

    #[test]
    fn bridge_options_overlay_replaces_mode_and_present_hooks() {
        let base = BridgeOptions::new(BridgeMode::BestEffort).with_route_label("base");
        let overlay = BridgeOptions::new(BridgeMode::Strict);
        let merged = base.merged_with(overlay);

        assert_eq!(merged.mode, BridgeMode::Strict);
        assert_eq!(merged.route_label.as_deref(), Some("base"));
    }
}
