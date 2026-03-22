//! Bridge customization helpers shared across request/response/stream paths.

use std::sync::Arc;

use siumai_core::bridge::{
    BridgeCustomization, BridgeMode, BridgePrimitiveContext, BridgePrimitiveKind,
    BridgePrimitiveRemapper, BridgeReport, BridgeTarget, BridgeWarning, BridgeWarningKind,
    RequestBridgeContext, ResponseBridgeContext, StreamBridgeContext,
};
use siumai_core::streaming::ChatStreamEvent;
use siumai_core::types::{
    ChatMessage, ChatRequest, ChatResponse, ContentPart, MessageContent, ProviderDefinedTool, Tool,
    ToolChoice,
};

type ProviderToolArgsMapper = dyn Fn(&RequestBridgeContext, &ProviderDefinedTool, &mut BridgeReport) -> serde_json::Value
    + Send
    + Sync;

/// Declarative provider-defined tool rewrite rule for request bridging.
#[derive(Clone)]
pub struct ProviderToolRewriteRule {
    from_tool_id: String,
    to_tool_id: String,
    target_name: Option<String>,
    args_mapper: Option<Arc<ProviderToolArgsMapper>>,
}

impl ProviderToolRewriteRule {
    pub fn new(from_tool_id: impl Into<String>, to_tool_id: impl Into<String>) -> Self {
        Self {
            from_tool_id: from_tool_id.into(),
            to_tool_id: to_tool_id.into(),
            target_name: None,
            args_mapper: None,
        }
    }

    pub fn with_target_name(mut self, target_name: impl Into<String>) -> Self {
        self.target_name = Some(target_name.into());
        self
    }

    pub fn with_args_mapper(mut self, args_mapper: Arc<ProviderToolArgsMapper>) -> Self {
        self.args_mapper = Some(args_mapper);
        self
    }

    fn matches(&self, tool: &ProviderDefinedTool) -> bool {
        tool.id == self.from_tool_id
    }

    fn rewrite(
        &self,
        ctx: &RequestBridgeContext,
        tool: &ProviderDefinedTool,
        report: &mut BridgeReport,
    ) -> ProviderDefinedTool {
        let args = self
            .args_mapper
            .as_ref()
            .map(|mapper| mapper(ctx, tool, report))
            .unwrap_or_else(|| tool.args.clone());

        ProviderDefinedTool::new(
            self.to_tool_id.clone(),
            self.target_name
                .clone()
                .unwrap_or_else(|| tool.name.clone()),
        )
        .with_args(args)
    }
}

/// Ready-made bridge customization that rewrites provider-defined tools before request serialization.
///
/// This is primarily useful for direct pair bridges where applications want to map an otherwise
/// unsupported provider-hosted tool into an equivalent tool on the target protocol without
/// rewriting the whole request manually.
#[derive(Clone, Default)]
pub struct ProviderToolRewriteCustomization {
    rules: Vec<ProviderToolRewriteRule>,
}

impl ProviderToolRewriteCustomization {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_rule(mut self, rule: ProviderToolRewriteRule) -> Self {
        self.rules.push(rule);
        self
    }

    pub fn map_provider_tool_id(
        mut self,
        from_tool_id: impl Into<String>,
        to_tool_id: impl Into<String>,
    ) -> Self {
        self.rules
            .push(ProviderToolRewriteRule::new(from_tool_id, to_tool_id));
        self
    }
}

impl BridgeCustomization for ProviderToolRewriteCustomization {
    fn transform_request(
        &self,
        ctx: &RequestBridgeContext,
        request: &mut ChatRequest,
        report: &mut BridgeReport,
    ) -> Result<(), siumai_core::LlmError> {
        let Some(tools) = request.tools.as_mut() else {
            return Ok(());
        };

        for tool in tools {
            let Tool::ProviderDefined(provider_tool) = tool else {
                continue;
            };

            let Some(rule) = self.rules.iter().find(|rule| rule.matches(provider_tool)) else {
                continue;
            };

            let original_id = provider_tool.id.clone();
            let rewritten = rule.rewrite(ctx, provider_tool, report);
            let rewritten_id = rewritten.id.clone();
            *provider_tool = rewritten;

            report.add_warning(BridgeWarning::new(
                BridgeWarningKind::Custom,
                format!(
                    "bridge customization rewrote provider-defined tool `{original_id}` to `{rewritten_id}`"
                ),
            ));
        }

        Ok(())
    }
}

fn primitive_context(
    source: Option<BridgeTarget>,
    target: BridgeTarget,
    mode: BridgeMode,
    route_label: Option<&String>,
    path_label: Option<&String>,
    primitive: BridgePrimitiveKind,
) -> BridgePrimitiveContext {
    BridgePrimitiveContext::new(
        source,
        target,
        mode,
        route_label.cloned(),
        path_label.cloned(),
        primitive,
    )
}

fn remap_tool_name(
    remapper: &dyn BridgePrimitiveRemapper,
    ctx: BridgePrimitiveContext,
    value: &mut String,
) {
    if let Some(mapped) = remapper.remap_tool_name(&ctx, value) {
        *value = mapped;
    }
}

fn remap_tool_call_id(
    remapper: &dyn BridgePrimitiveRemapper,
    ctx: BridgePrimitiveContext,
    value: &mut String,
) {
    if let Some(mapped) = remapper.remap_tool_call_id(&ctx, value) {
        *value = mapped;
    }
}

fn remap_message_content(
    content: &mut MessageContent,
    source: Option<BridgeTarget>,
    target: BridgeTarget,
    mode: BridgeMode,
    route_label: Option<&String>,
    path_label: Option<&String>,
    remapper: &dyn BridgePrimitiveRemapper,
) {
    let MessageContent::MultiModal(parts) = content else {
        return;
    };

    for part in parts {
        match part {
            ContentPart::ToolCall {
                tool_call_id,
                tool_name,
                ..
            } => {
                remap_tool_call_id(
                    remapper,
                    primitive_context(
                        source,
                        target,
                        mode,
                        route_label,
                        path_label,
                        BridgePrimitiveKind::ToolCall,
                    ),
                    tool_call_id,
                );
                remap_tool_name(
                    remapper,
                    primitive_context(
                        source,
                        target,
                        mode,
                        route_label,
                        path_label,
                        BridgePrimitiveKind::ToolCall,
                    ),
                    tool_name,
                );
            }
            ContentPart::ToolResult {
                tool_call_id,
                tool_name,
                ..
            } => {
                remap_tool_call_id(
                    remapper,
                    primitive_context(
                        source,
                        target,
                        mode,
                        route_label,
                        path_label,
                        BridgePrimitiveKind::ToolResult,
                    ),
                    tool_call_id,
                );
                remap_tool_name(
                    remapper,
                    primitive_context(
                        source,
                        target,
                        mode,
                        route_label,
                        path_label,
                        BridgePrimitiveKind::ToolResult,
                    ),
                    tool_name,
                );
            }
            ContentPart::ToolApprovalRequest { tool_call_id, .. } => {
                remap_tool_call_id(
                    remapper,
                    primitive_context(
                        source,
                        target,
                        mode,
                        route_label,
                        path_label,
                        BridgePrimitiveKind::ToolCall,
                    ),
                    tool_call_id,
                );
            }
            _ => {}
        }
    }
}

fn remap_message(
    message: &mut ChatMessage,
    source: Option<BridgeTarget>,
    target: BridgeTarget,
    mode: BridgeMode,
    route_label: Option<&String>,
    path_label: Option<&String>,
    remapper: &dyn BridgePrimitiveRemapper,
) {
    remap_message_content(
        &mut message.content,
        source,
        target,
        mode,
        route_label,
        path_label,
        remapper,
    );
}

pub(crate) fn apply_request_remapper(
    request: &mut ChatRequest,
    ctx: &RequestBridgeContext,
    remapper: &dyn BridgePrimitiveRemapper,
) {
    if let Some(tools) = request.tools.as_mut() {
        for tool in tools {
            match tool {
                Tool::Function { function } => remap_tool_name(
                    remapper,
                    primitive_context(
                        ctx.source,
                        ctx.target,
                        ctx.mode,
                        ctx.route_label.as_ref(),
                        ctx.path_label.as_ref(),
                        BridgePrimitiveKind::ToolDefinition,
                    ),
                    &mut function.name,
                ),
                Tool::ProviderDefined(tool) => remap_tool_name(
                    remapper,
                    primitive_context(
                        ctx.source,
                        ctx.target,
                        ctx.mode,
                        ctx.route_label.as_ref(),
                        ctx.path_label.as_ref(),
                        BridgePrimitiveKind::ToolDefinition,
                    ),
                    &mut tool.name,
                ),
            }
        }
    }

    if let Some(ToolChoice::Tool { name }) = request.tool_choice.as_mut() {
        remap_tool_name(
            remapper,
            primitive_context(
                ctx.source,
                ctx.target,
                ctx.mode,
                ctx.route_label.as_ref(),
                ctx.path_label.as_ref(),
                BridgePrimitiveKind::ToolChoice,
            ),
            name,
        );
    }

    for message in &mut request.messages {
        remap_message(
            message,
            ctx.source,
            ctx.target,
            ctx.mode,
            ctx.route_label.as_ref(),
            ctx.path_label.as_ref(),
            remapper,
        );
    }
}

pub(crate) fn apply_response_remapper(
    response: &mut ChatResponse,
    ctx: &ResponseBridgeContext,
    remapper: &dyn BridgePrimitiveRemapper,
) {
    remap_message_content(
        &mut response.content,
        ctx.source,
        ctx.target,
        ctx.mode,
        ctx.route_label.as_ref(),
        ctx.path_label.as_ref(),
        remapper,
    );
}

pub(crate) fn remap_stream_event(
    event: ChatStreamEvent,
    ctx: &StreamBridgeContext,
    remapper: &dyn BridgePrimitiveRemapper,
) -> ChatStreamEvent {
    match event {
        ChatStreamEvent::ToolCallDelta {
            mut id,
            mut function_name,
            arguments_delta,
            index,
        } => {
            remap_tool_call_id(
                remapper,
                primitive_context(
                    ctx.source,
                    ctx.target,
                    ctx.mode,
                    ctx.route_label.as_ref(),
                    ctx.path_label.as_ref(),
                    BridgePrimitiveKind::ToolCall,
                ),
                &mut id,
            );
            if let Some(name) = function_name.as_mut() {
                remap_tool_name(
                    remapper,
                    primitive_context(
                        ctx.source,
                        ctx.target,
                        ctx.mode,
                        ctx.route_label.as_ref(),
                        ctx.path_label.as_ref(),
                        BridgePrimitiveKind::ToolCall,
                    ),
                    name,
                );
            }
            ChatStreamEvent::ToolCallDelta {
                id,
                function_name,
                arguments_delta,
                index,
            }
        }
        ChatStreamEvent::StreamEnd { mut response } => {
            let response_ctx = ResponseBridgeContext::new(
                ctx.source,
                ctx.target,
                ctx.mode,
                ctx.route_label.clone(),
                ctx.path_label.clone(),
            );
            apply_response_remapper(&mut response, &response_ctx, remapper);
            ChatStreamEvent::StreamEnd { response }
        }
        other => other,
    }
}
