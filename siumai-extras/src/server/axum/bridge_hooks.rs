//! Closure-friendly adapters for bridge customization traits.

use std::sync::Arc;

use siumai::experimental::bridge::{
    BridgePrimitiveContext, BridgePrimitiveRemapper, BridgeReport, ResponseBridgeContext,
    ResponseBridgeHook, StreamBridgeContext, StreamBridgeHook,
};
use siumai::prelude::unified::{ChatResponse, ChatStreamEvent, LlmError};

type ResponseBridgeClosure = dyn Fn(&ResponseBridgeContext, &mut ChatResponse, &mut BridgeReport) -> Result<(), LlmError>
    + Send
    + Sync;

type StreamBridgeClosure =
    dyn Fn(&StreamBridgeContext, ChatStreamEvent) -> Vec<ChatStreamEvent> + Send + Sync;

type ToolNameRemapClosure = dyn Fn(&BridgePrimitiveContext, &str) -> Option<String> + Send + Sync;

type ToolCallIdRemapClosure = dyn Fn(&BridgePrimitiveContext, &str) -> Option<String> + Send + Sync;

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
