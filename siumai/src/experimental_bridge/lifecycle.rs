//! Shared bridge lifecycle helpers.

use siumai_core::bridge::{
    BridgeLossAction, BridgeMode, BridgeOptions, BridgeReport, BridgeTarget, RequestBridgeContext,
    ResponseBridgeContext, StreamBridgeContext,
};

use super::planner::{RequestBridgePath, RequestBridgePlan};

pub(crate) const NORMALIZED_RESPONSE_PATH_LABEL: &str = "normalized-response";

pub(crate) fn request_path_label(plan: &RequestBridgePlan) -> String {
    match plan.path {
        RequestBridgePath::ViaNormalized => "via-normalized".to_string(),
        RequestBridgePath::Direct(pair) => pair.as_str().to_string(),
    }
}

pub(crate) fn new_request_context(
    source: Option<BridgeTarget>,
    target: BridgeTarget,
    options: &BridgeOptions,
    path_label: Option<String>,
) -> RequestBridgeContext {
    RequestBridgeContext::new(
        source,
        target,
        options.mode,
        options.route_label.clone(),
        path_label,
    )
}

pub(crate) fn new_response_context(
    source: Option<BridgeTarget>,
    target: BridgeTarget,
    options: &BridgeOptions,
    path_label: Option<String>,
) -> ResponseBridgeContext {
    ResponseBridgeContext::new(
        source,
        target,
        options.mode,
        options.route_label.clone(),
        path_label,
    )
}

pub(crate) fn new_stream_context(
    source: Option<BridgeTarget>,
    target: BridgeTarget,
    options: &BridgeOptions,
    path_label: Option<String>,
) -> StreamBridgeContext {
    StreamBridgeContext::new(
        source,
        target,
        options.mode,
        options.route_label.clone(),
        path_label,
    )
}

pub(crate) fn new_bridge_report(
    source: Option<BridgeTarget>,
    target: BridgeTarget,
    mode: BridgeMode,
) -> BridgeReport {
    BridgeReport::with_source(source, target, mode)
}

pub(crate) fn reject_if_needed(
    report: &mut BridgeReport,
    action: BridgeLossAction,
    phase: &str,
    target: BridgeTarget,
) -> bool {
    if report.is_rejected() {
        return true;
    }

    if matches!(action, BridgeLossAction::Reject) {
        report.reject(format!(
            "bridge policy rejected {phase} conversion to {}",
            target.as_str()
        ));
        return true;
    }

    false
}
