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

/// Request bridge lifecycle phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RequestBridgePhase {
    NormalizeSource,
    SerializeTarget,
}

/// Request bridge lifecycle context.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestBridgeContext {
    pub phase: RequestBridgePhase,
    pub source: Option<BridgeTarget>,
    pub target: BridgeTarget,
    pub mode: BridgeMode,
    pub route_label: Option<String>,
    pub path_label: Option<String>,
}

impl RequestBridgeContext {
    pub fn new(
        phase: RequestBridgePhase,
        source: Option<BridgeTarget>,
        target: BridgeTarget,
        mode: BridgeMode,
        route_label: Option<String>,
        path_label: Option<String>,
    ) -> Self {
        Self {
            phase,
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

/// Request bridge customization hook shared by source-normalization and target-serialization paths.
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

/// Unified bridge customization trait.
///
/// This is an ergonomic bundle over the lower-level hook/remapper/policy traits. Applications that
/// need coordinated request/response/stream customization can implement one object and attach it
/// with `BridgeOptions::with_customization(...)` instead of wiring several trait objects manually.
///
/// The lower-level traits remain available and are still the best fit when the customization only
/// targets one narrow bridge phase.
pub trait BridgeCustomization: Send + Sync {
    fn transform_request(
        &self,
        _ctx: &RequestBridgeContext,
        _request: &mut ChatRequest,
        _report: &mut BridgeReport,
    ) -> Result<(), LlmError> {
        Ok(())
    }

    fn transform_request_json(
        &self,
        _ctx: &RequestBridgeContext,
        _body: &mut serde_json::Value,
        _report: &mut BridgeReport,
    ) -> Result<(), LlmError> {
        Ok(())
    }

    fn validate_request_json(
        &self,
        _ctx: &RequestBridgeContext,
        _body: &serde_json::Value,
        _report: &mut BridgeReport,
    ) -> Result<(), LlmError> {
        Ok(())
    }

    fn transform_response(
        &self,
        _ctx: &ResponseBridgeContext,
        _response: &mut ChatResponse,
        _report: &mut BridgeReport,
    ) -> Result<(), LlmError> {
        Ok(())
    }

    fn map_stream_event(
        &self,
        _ctx: &StreamBridgeContext,
        event: ChatStreamEvent,
    ) -> Vec<ChatStreamEvent> {
        vec![event]
    }

    fn remap_tool_name(&self, _ctx: &BridgePrimitiveContext, _name: &str) -> Option<String> {
        None
    }

    fn remap_tool_call_id(&self, _ctx: &BridgePrimitiveContext, _id: &str) -> Option<String> {
        None
    }

    fn request_action(
        &self,
        ctx: &RequestBridgeContext,
        report: &BridgeReport,
    ) -> BridgeLossAction {
        if report.is_rejected() || (matches!(ctx.mode, BridgeMode::Strict) && report.is_lossy()) {
            BridgeLossAction::Reject
        } else {
            BridgeLossAction::Continue
        }
    }

    fn response_action(
        &self,
        ctx: &ResponseBridgeContext,
        report: &BridgeReport,
    ) -> BridgeLossAction {
        if report.is_rejected() || (matches!(ctx.mode, BridgeMode::Strict) && report.is_lossy()) {
            BridgeLossAction::Reject
        } else {
            BridgeLossAction::Continue
        }
    }

    fn stream_action(&self, ctx: &StreamBridgeContext, report: &BridgeReport) -> BridgeLossAction {
        if report.is_rejected() || (matches!(ctx.mode, BridgeMode::Strict) && report.is_lossy()) {
            BridgeLossAction::Reject
        } else {
            BridgeLossAction::Continue
        }
    }
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

#[derive(Clone)]
struct BridgeCustomizationAdapter {
    customization: Arc<dyn BridgeCustomization>,
}

impl BridgeCustomizationAdapter {
    fn new(customization: Arc<dyn BridgeCustomization>) -> Self {
        Self { customization }
    }
}

impl RequestBridgeHook for BridgeCustomizationAdapter {
    fn transform_request(
        &self,
        ctx: &RequestBridgeContext,
        request: &mut ChatRequest,
        report: &mut BridgeReport,
    ) -> Result<(), LlmError> {
        self.customization.transform_request(ctx, request, report)
    }

    fn transform_json(
        &self,
        ctx: &RequestBridgeContext,
        body: &mut serde_json::Value,
        report: &mut BridgeReport,
    ) -> Result<(), LlmError> {
        self.customization.transform_request_json(ctx, body, report)
    }

    fn validate_json(
        &self,
        ctx: &RequestBridgeContext,
        body: &serde_json::Value,
        report: &mut BridgeReport,
    ) -> Result<(), LlmError> {
        self.customization.validate_request_json(ctx, body, report)
    }
}

impl ResponseBridgeHook for BridgeCustomizationAdapter {
    fn transform_response(
        &self,
        ctx: &ResponseBridgeContext,
        response: &mut ChatResponse,
        report: &mut BridgeReport,
    ) -> Result<(), LlmError> {
        self.customization.transform_response(ctx, response, report)
    }
}

impl StreamBridgeHook for BridgeCustomizationAdapter {
    fn map_event(&self, ctx: &StreamBridgeContext, event: ChatStreamEvent) -> Vec<ChatStreamEvent> {
        self.customization.map_stream_event(ctx, event)
    }
}

impl BridgePrimitiveRemapper for BridgeCustomizationAdapter {
    fn remap_tool_name(&self, ctx: &BridgePrimitiveContext, name: &str) -> Option<String> {
        self.customization.remap_tool_name(ctx, name)
    }

    fn remap_tool_call_id(&self, ctx: &BridgePrimitiveContext, id: &str) -> Option<String> {
        self.customization.remap_tool_call_id(ctx, id)
    }
}

impl BridgeLossPolicy for BridgeCustomizationAdapter {
    fn request_action(
        &self,
        ctx: &RequestBridgeContext,
        report: &BridgeReport,
    ) -> BridgeLossAction {
        self.customization.request_action(ctx, report)
    }

    fn response_action(
        &self,
        ctx: &ResponseBridgeContext,
        report: &BridgeReport,
    ) -> BridgeLossAction {
        self.customization.response_action(ctx, report)
    }

    fn stream_action(&self, ctx: &StreamBridgeContext, report: &BridgeReport) -> BridgeLossAction {
        self.customization.stream_action(ctx, report)
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

/// Partial bridge configuration override.
///
/// This is primarily useful for route-level or gateway-level customization where callers want
/// to override only selected fields, without having to restate the base `BridgeMode`.
#[derive(Clone, Default)]
pub struct BridgeOptionsOverride {
    pub mode: Option<BridgeMode>,
    pub route_label: Option<String>,
    pub request_hook: Option<Arc<dyn RequestBridgeHook>>,
    pub response_hook: Option<Arc<dyn ResponseBridgeHook>>,
    pub stream_hook: Option<Arc<dyn StreamBridgeHook>>,
    pub primitive_remapper: Option<Arc<dyn BridgePrimitiveRemapper>>,
    pub loss_policy: Option<Arc<dyn BridgeLossPolicy>>,
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

    pub fn with_customization(mut self, customization: Arc<dyn BridgeCustomization>) -> Self {
        let adapter = Arc::new(BridgeCustomizationAdapter::new(customization));
        self.request_hook = Some(adapter.clone());
        self.response_hook = Some(adapter.clone());
        self.stream_hook = Some(adapter.clone());
        self.primitive_remapper = Some(adapter.clone());
        self.loss_policy = adapter;
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

    pub fn merged_with_override(mut self, overlay: BridgeOptionsOverride) -> Self {
        if let Some(mode) = overlay.mode {
            self.mode = mode;
        }
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
        if let Some(loss_policy) = overlay.loss_policy {
            self.loss_policy = loss_policy;
        }
        self
    }
}

impl BridgeOptionsOverride {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_mode(mut self, mode: BridgeMode) -> Self {
        self.mode = Some(mode);
        self
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
        self.loss_policy = Some(policy);
        self
    }

    pub fn with_customization(mut self, customization: Arc<dyn BridgeCustomization>) -> Self {
        let adapter = Arc::new(BridgeCustomizationAdapter::new(customization));
        self.request_hook = Some(adapter.clone());
        self.response_hook = Some(adapter.clone());
        self.stream_hook = Some(adapter.clone());
        self.primitive_remapper = Some(adapter.clone());
        self.loss_policy = Some(adapter);
        self
    }
}

impl From<BridgeOptions> for BridgeOptionsOverride {
    fn from(value: BridgeOptions) -> Self {
        Self {
            mode: Some(value.mode),
            route_label: value.route_label,
            request_hook: value.request_hook,
            response_hook: value.response_hook,
            stream_hook: value.stream_hook,
            primitive_remapper: value.primitive_remapper,
            loss_policy: Some(value.loss_policy),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatRequest, MessageContent};
    use serde_json::json;

    struct CompositeCustomization;

    impl BridgeCustomization for CompositeCustomization {
        fn transform_request(
            &self,
            _ctx: &RequestBridgeContext,
            request: &mut ChatRequest,
            report: &mut BridgeReport,
        ) -> Result<(), LlmError> {
            request.common_params.max_tokens = Some(42);
            report.add_warning(BridgeWarning::new(
                BridgeWarningKind::Custom,
                "customization transformed request",
            ));
            Ok(())
        }

        fn transform_request_json(
            &self,
            _ctx: &RequestBridgeContext,
            body: &mut serde_json::Value,
            _report: &mut BridgeReport,
        ) -> Result<(), LlmError> {
            body["metadata"] = json!({ "customized": true });
            Ok(())
        }

        fn validate_request_json(
            &self,
            _ctx: &RequestBridgeContext,
            body: &serde_json::Value,
            report: &mut BridgeReport,
        ) -> Result<(), LlmError> {
            assert_eq!(body["metadata"]["customized"], json!(true));
            report.add_warning(BridgeWarning::new(
                BridgeWarningKind::Custom,
                "customization validated request json",
            ));
            Ok(())
        }

        fn transform_response(
            &self,
            _ctx: &ResponseBridgeContext,
            response: &mut ChatResponse,
            _report: &mut BridgeReport,
        ) -> Result<(), LlmError> {
            response.content = MessageContent::Text("[customized]".to_string());
            Ok(())
        }

        fn map_stream_event(
            &self,
            _ctx: &StreamBridgeContext,
            event: ChatStreamEvent,
        ) -> Vec<ChatStreamEvent> {
            match event {
                ChatStreamEvent::ContentDelta { delta, index } => {
                    vec![ChatStreamEvent::ContentDelta {
                        delta: delta.to_uppercase(),
                        index,
                    }]
                }
                other => vec![other],
            }
        }

        fn remap_tool_name(&self, _ctx: &BridgePrimitiveContext, name: &str) -> Option<String> {
            Some(format!("gw_{name}"))
        }

        fn remap_tool_call_id(&self, _ctx: &BridgePrimitiveContext, id: &str) -> Option<String> {
            Some(format!("gw_{id}"))
        }

        fn request_action(
            &self,
            _ctx: &RequestBridgeContext,
            report: &BridgeReport,
        ) -> BridgeLossAction {
            if report.is_lossy() {
                BridgeLossAction::Reject
            } else {
                BridgeLossAction::Continue
            }
        }
    }

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
            RequestBridgePhase::SerializeTarget,
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

    #[test]
    fn bridge_options_override_only_replaces_present_fields() {
        let base = BridgeOptions::new(BridgeMode::Strict).with_route_label("base");
        let merged =
            base.merged_with_override(BridgeOptionsOverride::new().with_route_label("route"));

        assert_eq!(merged.mode, BridgeMode::Strict);
        assert_eq!(merged.route_label.as_deref(), Some("route"));
    }

    #[test]
    fn bridge_options_override_can_set_mode_without_rebuilding_full_options() {
        let base = BridgeOptions::new(BridgeMode::BestEffort).with_route_label("base");
        let merged =
            base.merged_with_override(BridgeOptionsOverride::new().with_mode(BridgeMode::Strict));

        assert_eq!(merged.mode, BridgeMode::Strict);
        assert_eq!(merged.route_label.as_deref(), Some("base"));
    }

    #[test]
    fn bridge_options_customization_bundles_all_extension_points() {
        let options = BridgeOptions::new(BridgeMode::BestEffort)
            .with_route_label("tests.customization")
            .with_customization(Arc::new(CompositeCustomization));

        assert!(options.request_hook.is_some());
        assert!(options.response_hook.is_some());
        assert!(options.stream_hook.is_some());
        assert!(options.primitive_remapper.is_some());

        let request_ctx = RequestBridgeContext::new(
            RequestBridgePhase::SerializeTarget,
            Some(BridgeTarget::AnthropicMessages),
            BridgeTarget::OpenAiResponses,
            BridgeMode::BestEffort,
            Some("tests.customization".to_string()),
            Some("via-normalized".to_string()),
        );
        let mut request = ChatRequest::new(Vec::new());
        let mut request_report =
            BridgeReport::with_source(request_ctx.source, request_ctx.target, request_ctx.mode);
        options
            .request_hook
            .as_ref()
            .expect("request hook")
            .transform_request(&request_ctx, &mut request, &mut request_report)
            .expect("transform request");
        assert_eq!(request.common_params.max_tokens, Some(42));

        let mut body = json!({});
        options
            .request_hook
            .as_ref()
            .expect("request hook")
            .transform_json(&request_ctx, &mut body, &mut request_report)
            .expect("transform request json");
        options
            .request_hook
            .as_ref()
            .expect("request hook")
            .validate_json(&request_ctx, &body, &mut request_report)
            .expect("validate request json");

        let primitive_ctx = BridgePrimitiveContext::new(
            request_ctx.source,
            request_ctx.target,
            request_ctx.mode,
            request_ctx.route_label.clone(),
            request_ctx.path_label.clone(),
            BridgePrimitiveKind::ToolCall,
        );
        assert_eq!(
            options
                .primitive_remapper
                .as_ref()
                .expect("remapper")
                .remap_tool_name(&primitive_ctx, "weather")
                .as_deref(),
            Some("gw_weather")
        );
        assert_eq!(
            options
                .primitive_remapper
                .as_ref()
                .expect("remapper")
                .remap_tool_call_id(&primitive_ctx, "call_1")
                .as_deref(),
            Some("gw_call_1")
        );

        let response_ctx = ResponseBridgeContext::new(
            request_ctx.source,
            request_ctx.target,
            request_ctx.mode,
            request_ctx.route_label.clone(),
            Some("normalized-response".to_string()),
        );
        let mut response = ChatResponse::new(MessageContent::Text("visible".to_string()));
        let mut response_report =
            BridgeReport::with_source(response_ctx.source, response_ctx.target, response_ctx.mode);
        options
            .response_hook
            .as_ref()
            .expect("response hook")
            .transform_response(&response_ctx, &mut response, &mut response_report)
            .expect("transform response");
        assert_eq!(
            response.content,
            MessageContent::Text("[customized]".to_string())
        );

        let stream_ctx = StreamBridgeContext::new(
            request_ctx.source,
            request_ctx.target,
            request_ctx.mode,
            request_ctx.route_label.clone(),
            Some("stream".to_string()),
        );
        let stream_events = options
            .stream_hook
            .as_ref()
            .expect("stream hook")
            .map_event(
                &stream_ctx,
                ChatStreamEvent::ContentDelta {
                    delta: "hello".to_string(),
                    index: None,
                },
            );
        assert_eq!(stream_events.len(), 1);
        let ChatStreamEvent::ContentDelta { delta, index } = &stream_events[0] else {
            panic!("expected content delta");
        };
        assert_eq!(delta, "HELLO");
        assert_eq!(*index, None);

        let mut lossy_report =
            BridgeReport::with_source(request_ctx.source, request_ctx.target, request_ctx.mode);
        lossy_report.record_lossy_field("messages[0].thinking", "lossy");
        assert_eq!(
            options
                .loss_policy
                .request_action(&request_ctx, &lossy_report),
            BridgeLossAction::Reject
        );
    }
}
