//! Request bridge inspection.

use siumai_core::bridge::{BridgeReport, BridgeTarget};
use siumai_core::types::ChatRequest;

use super::primitives::{
    inspect_cache_control_semantics, inspect_reasoning_semantics, inspect_tool_approval_semantics,
};

/// Inspect a normalized request bridge without serializing it.
pub fn inspect_chat_request_bridge(
    request: &ChatRequest,
    target: BridgeTarget,
    report: &mut BridgeReport,
) {
    inspect_reasoning_semantics(request, target, report);
    inspect_cache_control_semantics(request, target, report);
    inspect_tool_approval_semantics(request, target, report);
}
