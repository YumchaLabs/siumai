//! Framework-agnostic gateway bridge helpers shared by server adapters.

use std::time::Duration;

use siumai::experimental::bridge::{
    BridgeDecision, BridgeMode, BridgeOptions, BridgeOptionsOverride, BridgeReport, BridgeTarget,
};

use super::GatewayBridgePolicy;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GatewayHeader {
    pub name: &'static str,
    pub value: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GatewaySseRuntimePolicy {
    pub keepalive_interval: Option<Duration>,
    pub idle_timeout: Option<Duration>,
    pub passthrough_runtime_errors: bool,
}

pub(crate) fn resolve_gateway_bridge_options(
    policy: Option<&GatewayBridgePolicy>,
    route_options: Option<BridgeOptions>,
    override_options: Option<BridgeOptionsOverride>,
) -> BridgeOptions {
    let mut effective = match (policy, route_options) {
        (Some(policy), Some(route_options)) => policy.resolve_bridge_options(Some(&route_options)),
        (Some(policy), None) => policy.bridge_options.clone(),
        (None, Some(route_options)) => route_options,
        (None, None) => BridgeOptions::default(),
    };

    if let Some(override_options) = override_options {
        effective = effective.merged_with_override(override_options);
    }

    effective
}

pub(crate) fn gateway_sse_runtime_policy(
    policy: Option<&GatewayBridgePolicy>,
) -> Option<GatewaySseRuntimePolicy> {
    let policy = policy?;
    if policy.keepalive_interval.is_none() && policy.stream_idle_timeout.is_none() {
        return None;
    }

    Some(GatewaySseRuntimePolicy {
        keepalive_interval: policy.keepalive_interval,
        idle_timeout: policy.stream_idle_timeout,
        passthrough_runtime_errors: policy.passthrough_runtime_errors,
    })
}

pub(crate) fn gateway_bridge_headers(
    policy: &GatewayBridgePolicy,
    target: BridgeTarget,
    report: Option<&BridgeReport>,
    mode: BridgeMode,
) -> Vec<GatewayHeader> {
    let mut headers = Vec::new();

    if policy.emit_bridge_headers {
        push_header(
            &mut headers,
            policy,
            "x-siumai-bridge-target",
            target.as_str(),
        );
        push_header(
            &mut headers,
            policy,
            "x-siumai-bridge-mode",
            bridge_mode_label(mode),
        );
        if let Some(report) = report {
            push_header(
                &mut headers,
                policy,
                "x-siumai-bridge-decision",
                bridge_decision_label(report.decision),
            );
        }
    }

    if let Some(report) = report.filter(|_| policy.emit_bridge_warning_headers) {
        push_header(
            &mut headers,
            policy,
            "x-siumai-bridge-warnings",
            report.warnings.len().to_string(),
        );
        push_header(
            &mut headers,
            policy,
            "x-siumai-bridge-lossy-fields",
            report.lossy_fields.len().to_string(),
        );
        push_header(
            &mut headers,
            policy,
            "x-siumai-bridge-dropped-fields",
            report.dropped_fields.len().to_string(),
        );
    }

    headers
}

fn push_header(
    headers: &mut Vec<GatewayHeader>,
    policy: &GatewayBridgePolicy,
    name: &'static str,
    value: impl Into<String>,
) {
    if !policy.allows_response_header(name) {
        return;
    }

    headers.push(GatewayHeader {
        name,
        value: value.into(),
    });
}

fn bridge_mode_label(mode: BridgeMode) -> &'static str {
    match mode {
        BridgeMode::Strict => "strict",
        BridgeMode::BestEffort => "best-effort",
        BridgeMode::ProviderTolerant => "provider-tolerant",
    }
}

fn bridge_decision_label(decision: BridgeDecision) -> &'static str {
    match decision {
        BridgeDecision::Exact => "exact",
        BridgeDecision::Lossy => "lossy",
        BridgeDecision::Rejected => "rejected",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_gateway_bridge_options_applies_policy_route_and_override() {
        let policy = GatewayBridgePolicy::new(BridgeMode::BestEffort).with_route_label("policy");
        let route_options = BridgeOptions::new(BridgeMode::Strict).with_route_label("route");
        let override_options = BridgeOptionsOverride::new().with_mode(BridgeMode::ProviderTolerant);

        let resolved = resolve_gateway_bridge_options(
            Some(&policy),
            Some(route_options),
            Some(override_options),
        );

        assert_eq!(resolved.mode, BridgeMode::ProviderTolerant);
        assert_eq!(resolved.route_label.as_deref(), Some("route"));
    }

    #[test]
    fn gateway_bridge_headers_respect_policy_filters() {
        let policy = GatewayBridgePolicy::new(BridgeMode::BestEffort)
            .with_bridge_headers(true)
            .with_bridge_warning_headers(true)
            .with_response_header_allowlist([
                "x-siumai-bridge-mode",
                "x-siumai-bridge-lossy-fields",
            ]);
        let mut report =
            BridgeReport::with_source(None, BridgeTarget::AnthropicMessages, BridgeMode::Strict);
        report.decision = BridgeDecision::Lossy;
        report
            .lossy_fields
            .push("messages[0].content[0]".to_string());

        let headers = gateway_bridge_headers(
            &policy,
            BridgeTarget::AnthropicMessages,
            Some(&report),
            report.mode,
        );

        assert_eq!(
            headers,
            vec![
                GatewayHeader {
                    name: "x-siumai-bridge-mode",
                    value: "strict".to_string(),
                },
                GatewayHeader {
                    name: "x-siumai-bridge-lossy-fields",
                    value: "1".to_string(),
                },
            ]
        );
    }

    #[test]
    fn gateway_sse_runtime_policy_requires_timer_configuration() {
        let disabled = GatewayBridgePolicy::new(BridgeMode::BestEffort);
        assert!(gateway_sse_runtime_policy(Some(&disabled)).is_none());

        let enabled = GatewayBridgePolicy::new(BridgeMode::BestEffort)
            .with_keepalive_interval(Duration::from_secs(1))
            .with_passthrough_runtime_errors(false);

        assert_eq!(
            gateway_sse_runtime_policy(Some(&enabled)),
            Some(GatewaySseRuntimePolicy {
                keepalive_interval: Some(Duration::from_secs(1)),
                idle_timeout: None,
                passthrough_runtime_errors: false,
            })
        );
    }
}
