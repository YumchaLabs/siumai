//! Request bridge inspection.

use siumai_core::bridge::{BridgeReport, BridgeTarget};
use siumai_core::types::ChatRequest;

use super::primitives::{
    inspect_cache_control_semantics, inspect_reasoning_semantics, inspect_tool_approval_semantics,
};
use super::target_caps::request_target_capabilities;

/// Inspect a normalized request bridge without serializing it.
pub fn inspect_chat_request_bridge(
    request: &ChatRequest,
    target: BridgeTarget,
    report: &mut BridgeReport,
) {
    let caps = request_target_capabilities(target);
    inspect_reasoning_semantics(request, caps, report);
    inspect_cache_control_semantics(request, caps, report);
    inspect_tool_approval_semantics(request, caps, report);
}
