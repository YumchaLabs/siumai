//! Gateway bridge policy shared by server adapters.

use std::{fmt, time::Duration};

use std::sync::Arc;

use siumai::experimental::bridge::{
    BridgeCustomization, BridgeMode, BridgeOptions, BridgeOptionsOverride,
};

/// Stable gateway policy for bridge-backed server adapters.
#[derive(Clone)]
pub struct GatewayBridgePolicy {
    /// Default bridge customization applied by the gateway.
    pub bridge_options: BridgeOptions,
    /// Emit bridge target/mode/decision headers on downstream responses when possible.
    pub emit_bridge_headers: bool,
    /// Emit bridge warning counters on downstream responses when possible.
    pub emit_bridge_warning_headers: bool,
    /// Whether runtime errors should be surfaced verbatim from the current helper.
    pub passthrough_runtime_errors: bool,
    /// Optional downstream request body limit for route layers to consume.
    pub request_body_limit_bytes: Option<usize>,
    /// Optional upstream read limit for proxy/gateway runtimes to consume.
    pub upstream_read_limit_bytes: Option<usize>,
    /// Optional idle timeout enforced by streaming transcode helpers when supported.
    pub stream_idle_timeout: Option<Duration>,
    /// Optional keepalive interval enforced by streaming transcode helpers when supported.
    pub keepalive_interval: Option<Duration>,
    /// Optional response-header allowlist for gateway-emitted headers.
    pub response_header_allowlist: Option<Vec<String>>,
    /// Optional response-header denylist for gateway-emitted headers.
    pub response_header_denylist: Vec<String>,
}

impl Default for GatewayBridgePolicy {
    fn default() -> Self {
        Self::new(BridgeMode::BestEffort)
    }
}

impl fmt::Debug for GatewayBridgePolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GatewayBridgePolicy")
            .field("bridge_mode", &self.bridge_options.mode)
            .field("route_label", &self.bridge_options.route_label)
            .field("emit_bridge_headers", &self.emit_bridge_headers)
            .field(
                "emit_bridge_warning_headers",
                &self.emit_bridge_warning_headers,
            )
            .field(
                "passthrough_runtime_errors",
                &self.passthrough_runtime_errors,
            )
            .field("request_body_limit_bytes", &self.request_body_limit_bytes)
            .field("upstream_read_limit_bytes", &self.upstream_read_limit_bytes)
            .field("stream_idle_timeout", &self.stream_idle_timeout)
            .field("keepalive_interval", &self.keepalive_interval)
            .field("response_header_allowlist", &self.response_header_allowlist)
            .field("response_header_denylist", &self.response_header_denylist)
            .finish()
    }
}

impl GatewayBridgePolicy {
    /// Create a new gateway bridge policy with the given default bridge mode.
    pub fn new(mode: BridgeMode) -> Self {
        Self {
            bridge_options: BridgeOptions::new(mode),
            emit_bridge_headers: false,
            emit_bridge_warning_headers: false,
            passthrough_runtime_errors: true,
            request_body_limit_bytes: None,
            upstream_read_limit_bytes: None,
            stream_idle_timeout: None,
            keepalive_interval: None,
            response_header_allowlist: None,
            response_header_denylist: Vec::new(),
        }
    }

    /// Replace the default bridge options used by this policy.
    pub fn with_bridge_options(mut self, bridge_options: BridgeOptions) -> Self {
        self.bridge_options = bridge_options;
        self
    }

    /// Partially override the default bridge options used by this policy.
    pub fn with_bridge_options_override(mut self, override_options: BridgeOptionsOverride) -> Self {
        self.bridge_options = self.bridge_options.merged_with_override(override_options);
        self
    }

    /// Attach a unified bridge customization object to the default bridge options.
    pub fn with_customization(mut self, customization: Arc<dyn BridgeCustomization>) -> Self {
        self.bridge_options = self.bridge_options.with_customization(customization);
        self
    }

    /// Override only the default bridge mode used by this policy.
    pub fn with_bridge_mode(mut self, mode: BridgeMode) -> Self {
        self.bridge_options.mode = mode;
        self
    }

    /// Set a route label on the default bridge options.
    pub fn with_route_label(mut self, route_label: impl Into<String>) -> Self {
        self.bridge_options = self.bridge_options.with_route_label(route_label);
        self
    }

    /// Emit bridge target/mode/decision headers.
    pub fn with_bridge_headers(mut self, enabled: bool) -> Self {
        self.emit_bridge_headers = enabled;
        self
    }

    /// Emit bridge warning counters in response headers.
    pub fn with_bridge_warning_headers(mut self, enabled: bool) -> Self {
        self.emit_bridge_warning_headers = enabled;
        self
    }

    /// Surface runtime errors verbatim from the current helper.
    pub fn with_passthrough_runtime_errors(mut self, enabled: bool) -> Self {
        self.passthrough_runtime_errors = enabled;
        self
    }

    /// Set a request body limit hint.
    pub fn with_request_body_limit_bytes(mut self, limit: usize) -> Self {
        self.request_body_limit_bytes = Some(limit);
        self
    }

    /// Set an upstream read limit hint.
    pub fn with_upstream_read_limit_bytes(mut self, limit: usize) -> Self {
        self.upstream_read_limit_bytes = Some(limit);
        self
    }

    /// Set a stream idle timeout.
    pub fn with_stream_idle_timeout(mut self, timeout: Duration) -> Self {
        self.stream_idle_timeout = Some(timeout);
        self
    }

    /// Set a stream keepalive interval.
    pub fn with_keepalive_interval(mut self, interval: Duration) -> Self {
        self.keepalive_interval = Some(interval);
        self
    }

    /// Restrict gateway-emitted response headers to an allowlist.
    pub fn with_response_header_allowlist<I, S>(mut self, headers: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.response_header_allowlist = Some(headers.into_iter().map(Into::into).collect());
        self
    }

    /// Deny selected gateway-emitted response headers.
    pub fn with_response_header_denylist<I, S>(mut self, headers: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.response_header_denylist = headers.into_iter().map(Into::into).collect();
        self
    }

    /// Resolve the effective bridge options after overlaying route-local overrides.
    pub fn resolve_bridge_options(
        &self,
        override_options: Option<&BridgeOptions>,
    ) -> BridgeOptions {
        match override_options {
            Some(override_options) => self
                .bridge_options
                .clone()
                .merged_with(override_options.clone()),
            None => self.bridge_options.clone(),
        }
    }

    /// Resolve the effective bridge options after overlaying partial route-local overrides.
    pub fn resolve_bridge_options_override(
        &self,
        override_options: Option<&BridgeOptionsOverride>,
    ) -> BridgeOptions {
        match override_options {
            Some(override_options) => self
                .bridge_options
                .clone()
                .merged_with_override(override_options.clone()),
            None => self.bridge_options.clone(),
        }
    }

    /// Check whether a gateway-emitted response header is allowed.
    pub fn allows_response_header(&self, name: &str) -> bool {
        let lower = name.to_ascii_lowercase();
        if self
            .response_header_denylist
            .iter()
            .any(|candidate| candidate.eq_ignore_ascii_case(&lower))
        {
            return false;
        }

        match &self.response_header_allowlist {
            Some(allowlist) => allowlist
                .iter()
                .any(|candidate| candidate.eq_ignore_ascii_case(&lower)),
            None => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use siumai::experimental::bridge::BridgeCustomization;

    struct NoopCustomization;

    impl BridgeCustomization for NoopCustomization {}

    #[test]
    fn resolve_bridge_options_overlays_route_override() {
        let policy = GatewayBridgePolicy::new(BridgeMode::BestEffort).with_route_label("policy");
        let effective =
            policy.resolve_bridge_options(Some(&BridgeOptions::new(BridgeMode::Strict)));

        assert_eq!(effective.mode, BridgeMode::Strict);
        assert_eq!(effective.route_label.as_deref(), Some("policy"));
    }

    #[test]
    fn resolve_bridge_options_override_keeps_base_mode_when_not_overridden() {
        let policy = GatewayBridgePolicy::new(BridgeMode::Strict).with_route_label("policy");
        let effective = policy.resolve_bridge_options_override(Some(
            &BridgeOptionsOverride::new().with_route_label("route"),
        ));

        assert_eq!(effective.mode, BridgeMode::Strict);
        assert_eq!(effective.route_label.as_deref(), Some("route"));
    }

    #[test]
    fn allow_and_deny_lists_are_case_insensitive() {
        let policy = GatewayBridgePolicy::default()
            .with_response_header_allowlist(["x-siumai-bridge-mode", "x-siumai-bridge-target"])
            .with_response_header_denylist(["x-siumai-bridge-target"]);

        assert!(policy.allows_response_header("X-SIUMAI-BRIDGE-MODE"));
        assert!(!policy.allows_response_header("x-siumai-bridge-target"));
    }

    #[test]
    fn gateway_policy_can_attach_unified_bridge_customization() {
        let policy = GatewayBridgePolicy::new(BridgeMode::BestEffort)
            .with_customization(Arc::new(NoopCustomization));

        assert!(policy.bridge_options.request_hook.is_some());
        assert!(policy.bridge_options.response_hook.is_some());
        assert!(policy.bridge_options.stream_hook.is_some());
        assert!(policy.bridge_options.primitive_remapper.is_some());
    }
}
