//! Bridge customization helpers shared across request/response/stream paths.

use siumai_core::bridge::{
    BridgeMode, BridgePrimitiveContext, BridgePrimitiveKind, BridgePrimitiveRemapper, BridgeTarget,
    RequestBridgeContext, ResponseBridgeContext, StreamBridgeContext,
};
use siumai_core::streaming::ChatStreamEvent;
use siumai_core::types::{
    ChatMessage, ChatRequest, ChatResponse, ContentPart, MessageContent, Tool, ToolChoice,
};

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
