//! Closure-friendly adapters for bridge customization traits.

use std::sync::Arc;

use siumai::experimental::bridge::{
    BridgeCustomization, BridgeLossAction, BridgeMode, BridgePrimitiveContext,
    BridgePrimitiveRemapper, BridgeReport, RequestBridgeContext, ResponseBridgeContext,
    ResponseBridgeHook, StreamBridgeContext, StreamBridgeHook,
};
use siumai::prelude::unified::{ChatRequest, ChatResponse, ChatStreamEvent, LlmError};

type RequestBridgeClosure = dyn Fn(&RequestBridgeContext, &mut ChatRequest, &mut BridgeReport) -> Result<(), LlmError>
    + Send
    + Sync;

type RequestJsonBridgeClosure = dyn Fn(&RequestBridgeContext, &mut serde_json::Value, &mut BridgeReport) -> Result<(), LlmError>
    + Send
    + Sync;

type RequestJsonValidationClosure = dyn Fn(&RequestBridgeContext, &serde_json::Value, &mut BridgeReport) -> Result<(), LlmError>
    + Send
    + Sync;

type ResponseBridgeClosure = dyn Fn(&ResponseBridgeContext, &mut ChatResponse, &mut BridgeReport) -> Result<(), LlmError>
    + Send
    + Sync;

type StreamBridgeClosure =
    dyn Fn(&StreamBridgeContext, ChatStreamEvent) -> Vec<ChatStreamEvent> + Send + Sync;

type ToolNameRemapClosure = dyn Fn(&BridgePrimitiveContext, &str) -> Option<String> + Send + Sync;

type ToolCallIdRemapClosure = dyn Fn(&BridgePrimitiveContext, &str) -> Option<String> + Send + Sync;

type RequestLossActionClosure =
    dyn Fn(&RequestBridgeContext, &BridgeReport) -> BridgeLossAction + Send + Sync;

type ResponseLossActionClosure =
    dyn Fn(&ResponseBridgeContext, &BridgeReport) -> BridgeLossAction + Send + Sync;

type StreamLossActionClosure =
    dyn Fn(&StreamBridgeContext, &BridgeReport) -> BridgeLossAction + Send + Sync;

fn default_loss_action(mode: BridgeMode, report: &BridgeReport) -> BridgeLossAction {
    if report.is_rejected() || (matches!(mode, BridgeMode::Strict) && report.is_lossy()) {
        BridgeLossAction::Reject
    } else {
        BridgeLossAction::Continue
    }
}

/// Closure-backed response bridge hook.
#[derive(Clone)]
pub struct ClosureResponseBridgeHook {
    inner: Arc<ResponseBridgeClosure>,
}

impl ClosureResponseBridgeHook {
    /// Create a response bridge hook from a closure.
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&ResponseBridgeContext, &mut ChatResponse, &mut BridgeReport) -> Result<(), LlmError>
            + Send
            + Sync
            + 'static,
    {
        Self { inner: Arc::new(f) }
    }
}

impl ResponseBridgeHook for ClosureResponseBridgeHook {
    fn transform_response(
        &self,
        ctx: &ResponseBridgeContext,
        response: &mut ChatResponse,
        report: &mut BridgeReport,
    ) -> Result<(), LlmError> {
        (self.inner)(ctx, response, report)
    }
}

/// Create an `Arc<dyn ResponseBridgeHook>` from a closure.
pub fn response_bridge_hook<F>(f: F) -> Arc<dyn ResponseBridgeHook>
where
    F: Fn(&ResponseBridgeContext, &mut ChatResponse, &mut BridgeReport) -> Result<(), LlmError>
        + Send
        + Sync
        + 'static,
{
    Arc::new(ClosureResponseBridgeHook::new(f))
}

/// Closure-backed stream bridge hook.
#[derive(Clone)]
pub struct ClosureStreamBridgeHook {
    inner: Arc<StreamBridgeClosure>,
}

impl ClosureStreamBridgeHook {
    /// Create a stream bridge hook from a closure.
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&StreamBridgeContext, ChatStreamEvent) -> Vec<ChatStreamEvent>
            + Send
            + Sync
            + 'static,
    {
        Self { inner: Arc::new(f) }
    }
}

impl StreamBridgeHook for ClosureStreamBridgeHook {
    fn map_event(&self, ctx: &StreamBridgeContext, event: ChatStreamEvent) -> Vec<ChatStreamEvent> {
        (self.inner)(ctx, event)
    }
}

/// Create an `Arc<dyn StreamBridgeHook>` from a closure.
pub fn stream_bridge_hook<F>(f: F) -> Arc<dyn StreamBridgeHook>
where
    F: Fn(&StreamBridgeContext, ChatStreamEvent) -> Vec<ChatStreamEvent> + Send + Sync + 'static,
{
    Arc::new(ClosureStreamBridgeHook::new(f))
}

/// Closure-backed primitive remapper.
#[derive(Clone, Default)]
pub struct ClosurePrimitiveRemapper {
    tool_name: Option<Arc<ToolNameRemapClosure>>,
    tool_call_id: Option<Arc<ToolCallIdRemapClosure>>,
}

impl ClosurePrimitiveRemapper {
    /// Attach a tool-name remap closure.
    pub fn with_tool_name<F>(mut self, f: F) -> Self
    where
        F: Fn(&BridgePrimitiveContext, &str) -> Option<String> + Send + Sync + 'static,
    {
        self.tool_name = Some(Arc::new(f));
        self
    }

    /// Attach a tool-call-id remap closure.
    pub fn with_tool_call_id<F>(mut self, f: F) -> Self
    where
        F: Fn(&BridgePrimitiveContext, &str) -> Option<String> + Send + Sync + 'static,
    {
        self.tool_call_id = Some(Arc::new(f));
        self
    }
}

impl BridgePrimitiveRemapper for ClosurePrimitiveRemapper {
    fn remap_tool_name(&self, ctx: &BridgePrimitiveContext, name: &str) -> Option<String> {
        self.tool_name.as_ref().and_then(|f| f(ctx, name))
    }

    fn remap_tool_call_id(&self, ctx: &BridgePrimitiveContext, id: &str) -> Option<String> {
        self.tool_call_id.as_ref().and_then(|f| f(ctx, id))
    }
}

/// Closure-backed unified bridge customization bundle.
#[derive(Clone, Default)]
pub struct ClosureBridgeCustomization {
    request: Option<Arc<RequestBridgeClosure>>,
    request_json: Option<Arc<RequestJsonBridgeClosure>>,
    request_json_validation: Option<Arc<RequestJsonValidationClosure>>,
    response: Option<Arc<ResponseBridgeClosure>>,
    stream: Option<Arc<StreamBridgeClosure>>,
    tool_name: Option<Arc<ToolNameRemapClosure>>,
    tool_call_id: Option<Arc<ToolCallIdRemapClosure>>,
    request_loss_action: Option<Arc<RequestLossActionClosure>>,
    response_loss_action: Option<Arc<ResponseLossActionClosure>>,
    stream_loss_action: Option<Arc<StreamLossActionClosure>>,
}

impl ClosureBridgeCustomization {
    /// Attach a normalized request transform closure.
    pub fn with_request<F>(mut self, f: F) -> Self
    where
        F: Fn(&RequestBridgeContext, &mut ChatRequest, &mut BridgeReport) -> Result<(), LlmError>
            + Send
            + Sync
            + 'static,
    {
        self.request = Some(Arc::new(f));
        self
    }

    /// Attach a target JSON transform closure for request bridging.
    pub fn with_request_json<F>(mut self, f: F) -> Self
    where
        F: Fn(
                &RequestBridgeContext,
                &mut serde_json::Value,
                &mut BridgeReport,
            ) -> Result<(), LlmError>
            + Send
            + Sync
            + 'static,
    {
        self.request_json = Some(Arc::new(f));
        self
    }

    /// Attach a target JSON validation closure for request bridging.
    pub fn with_request_json_validation<F>(mut self, f: F) -> Self
    where
        F: Fn(&RequestBridgeContext, &serde_json::Value, &mut BridgeReport) -> Result<(), LlmError>
            + Send
            + Sync
            + 'static,
    {
        self.request_json_validation = Some(Arc::new(f));
        self
    }

    /// Attach a normalized response transform closure.
    pub fn with_response<F>(mut self, f: F) -> Self
    where
        F: Fn(&ResponseBridgeContext, &mut ChatResponse, &mut BridgeReport) -> Result<(), LlmError>
            + Send
            + Sync
            + 'static,
    {
        self.response = Some(Arc::new(f));
        self
    }

    /// Attach a stream-event transform closure.
    pub fn with_stream<F>(mut self, f: F) -> Self
    where
        F: Fn(&StreamBridgeContext, ChatStreamEvent) -> Vec<ChatStreamEvent>
            + Send
            + Sync
            + 'static,
    {
        self.stream = Some(Arc::new(f));
        self
    }

    /// Attach a tool-name remap closure.
    pub fn with_tool_name<F>(mut self, f: F) -> Self
    where
        F: Fn(&BridgePrimitiveContext, &str) -> Option<String> + Send + Sync + 'static,
    {
        self.tool_name = Some(Arc::new(f));
        self
    }

    /// Attach a tool-call-id remap closure.
    pub fn with_tool_call_id<F>(mut self, f: F) -> Self
    where
        F: Fn(&BridgePrimitiveContext, &str) -> Option<String> + Send + Sync + 'static,
    {
        self.tool_call_id = Some(Arc::new(f));
        self
    }

    /// Attach a request loss-policy closure.
    pub fn with_request_loss_action<F>(mut self, f: F) -> Self
    where
        F: Fn(&RequestBridgeContext, &BridgeReport) -> BridgeLossAction + Send + Sync + 'static,
    {
        self.request_loss_action = Some(Arc::new(f));
        self
    }

    /// Attach a response loss-policy closure.
    pub fn with_response_loss_action<F>(mut self, f: F) -> Self
    where
        F: Fn(&ResponseBridgeContext, &BridgeReport) -> BridgeLossAction + Send + Sync + 'static,
    {
        self.response_loss_action = Some(Arc::new(f));
        self
    }

    /// Attach a stream loss-policy closure.
    pub fn with_stream_loss_action<F>(mut self, f: F) -> Self
    where
        F: Fn(&StreamBridgeContext, &BridgeReport) -> BridgeLossAction + Send + Sync + 'static,
    {
        self.stream_loss_action = Some(Arc::new(f));
        self
    }
}

impl BridgeCustomization for ClosureBridgeCustomization {
    fn transform_request(
        &self,
        ctx: &RequestBridgeContext,
        request: &mut ChatRequest,
        report: &mut BridgeReport,
    ) -> Result<(), LlmError> {
        if let Some(f) = &self.request {
            f(ctx, request, report)?;
        }
        Ok(())
    }

    fn transform_request_json(
        &self,
        ctx: &RequestBridgeContext,
        body: &mut serde_json::Value,
        report: &mut BridgeReport,
    ) -> Result<(), LlmError> {
        if let Some(f) = &self.request_json {
            f(ctx, body, report)?;
        }
        Ok(())
    }

    fn validate_request_json(
        &self,
        ctx: &RequestBridgeContext,
        body: &serde_json::Value,
        report: &mut BridgeReport,
    ) -> Result<(), LlmError> {
        if let Some(f) = &self.request_json_validation {
            f(ctx, body, report)?;
        }
        Ok(())
    }

    fn transform_response(
        &self,
        ctx: &ResponseBridgeContext,
        response: &mut ChatResponse,
        report: &mut BridgeReport,
    ) -> Result<(), LlmError> {
        if let Some(f) = &self.response {
            f(ctx, response, report)?;
        }
        Ok(())
    }

    fn map_stream_event(
        &self,
        ctx: &StreamBridgeContext,
        event: ChatStreamEvent,
    ) -> Vec<ChatStreamEvent> {
        match &self.stream {
            Some(f) => f(ctx, event),
            None => vec![event],
        }
    }

    fn remap_tool_name(&self, ctx: &BridgePrimitiveContext, name: &str) -> Option<String> {
        self.tool_name.as_ref().and_then(|f| f(ctx, name))
    }

    fn remap_tool_call_id(&self, ctx: &BridgePrimitiveContext, id: &str) -> Option<String> {
        self.tool_call_id.as_ref().and_then(|f| f(ctx, id))
    }

    fn request_action(
        &self,
        ctx: &RequestBridgeContext,
        report: &BridgeReport,
    ) -> BridgeLossAction {
        self.request_loss_action
            .as_ref()
            .map(|f| f(ctx, report))
            .unwrap_or_else(|| default_loss_action(ctx.mode, report))
    }

    fn response_action(
        &self,
        ctx: &ResponseBridgeContext,
        report: &BridgeReport,
    ) -> BridgeLossAction {
        self.response_loss_action
            .as_ref()
            .map(|f| f(ctx, report))
            .unwrap_or_else(|| default_loss_action(ctx.mode, report))
    }

    fn stream_action(&self, ctx: &StreamBridgeContext, report: &BridgeReport) -> BridgeLossAction {
        self.stream_loss_action
            .as_ref()
            .map(|f| f(ctx, report))
            .unwrap_or_else(|| default_loss_action(ctx.mode, report))
    }
}
