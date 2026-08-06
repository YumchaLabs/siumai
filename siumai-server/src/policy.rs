use std::collections::BTreeSet;
use std::time::Duration;

use thiserror::Error;

pub const DEFAULT_REQUEST_BODY_LIMIT_BYTES: usize = 1024 * 1024;
pub const DEFAULT_UPSTREAM_BODY_LIMIT_BYTES: usize = 8 * 1024 * 1024;
pub const DEFAULT_JSON_RESPONSE_LIMIT_BYTES: usize = 8 * 1024 * 1024;
pub const DEFAULT_SSE_EVENT_LIMIT_BYTES: usize = 1024 * 1024;
pub const MIN_SERVER_RESPONSE_LIMIT_BYTES: usize = 512;
pub const MAX_SERVER_BODY_LIMIT_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_RESPONSE_HEADER_RULES: usize = 64;

/// Server behavior when a canonical value cannot be projected without loss.
///
/// Provider-owned opaque state is never serialized by this crate. `Reject`
/// stops the projection, while `Report` emits a bounded loss diagnostic and
/// continues with the portable subset.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum GatewayLossPolicy {
    Reject,
    #[default]
    Report,
}

impl GatewayLossPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Reject => "reject",
            Self::Report => "report",
        }
    }
}

/// Detail level used by HTTP-facing error messages.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum GatewayErrorDetail {
    /// Emit fixed, route-safe messages only.
    #[default]
    Minimal,
    /// Include bounded public facts such as the configured byte limit.
    Public,
}

/// Finite ingress, upstream, and projection limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatewayLimits {
    request_body_bytes: usize,
    upstream_body_bytes: usize,
    json_response_bytes: usize,
    sse_event_bytes: usize,
}

impl GatewayLimits {
    pub fn new(
        request_body_bytes: usize,
        upstream_body_bytes: usize,
        json_response_bytes: usize,
        sse_event_bytes: usize,
    ) -> Result<Self, GatewayPolicyError> {
        validate_limit("request_body_bytes", request_body_bytes)?;
        validate_limit("upstream_body_bytes", upstream_body_bytes)?;
        validate_response_limit("json_response_bytes", json_response_bytes)?;
        validate_response_limit("sse_event_bytes", sse_event_bytes)?;
        Ok(Self {
            request_body_bytes,
            upstream_body_bytes,
            json_response_bytes,
            sse_event_bytes,
        })
    }

    pub const fn request_body_bytes(self) -> usize {
        self.request_body_bytes
    }

    pub const fn upstream_body_bytes(self) -> usize {
        self.upstream_body_bytes
    }

    pub const fn json_response_bytes(self) -> usize {
        self.json_response_bytes
    }

    pub const fn sse_event_bytes(self) -> usize {
        self.sse_event_bytes
    }
}

impl Default for GatewayLimits {
    fn default() -> Self {
        Self {
            request_body_bytes: DEFAULT_REQUEST_BODY_LIMIT_BYTES,
            upstream_body_bytes: DEFAULT_UPSTREAM_BODY_LIMIT_BYTES,
            json_response_bytes: DEFAULT_JSON_RESPONSE_LIMIT_BYTES,
            sse_event_bytes: DEFAULT_SSE_EVENT_LIMIT_BYTES,
        }
    }
}

/// Timers owned by the downstream SSE projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatewayStreamPolicy {
    keep_alive_interval: Option<Duration>,
    idle_timeout: Option<Duration>,
}

impl GatewayStreamPolicy {
    pub const fn disabled() -> Self {
        Self {
            keep_alive_interval: None,
            idle_timeout: None,
        }
    }

    pub fn with_keep_alive_interval(
        mut self,
        interval: Option<Duration>,
    ) -> Result<Self, GatewayPolicyError> {
        validate_duration("keep_alive_interval", interval)?;
        self.keep_alive_interval = interval;
        Ok(self)
    }

    pub fn with_idle_timeout(
        mut self,
        timeout: Option<Duration>,
    ) -> Result<Self, GatewayPolicyError> {
        validate_duration("idle_timeout", timeout)?;
        self.idle_timeout = timeout;
        Ok(self)
    }

    pub const fn keep_alive_interval(self) -> Option<Duration> {
        self.keep_alive_interval
    }

    pub const fn idle_timeout(self) -> Option<Duration> {
        self.idle_timeout
    }
}

impl Default for GatewayStreamPolicy {
    fn default() -> Self {
        Self {
            keep_alive_interval: Some(Duration::from_secs(15)),
            idle_timeout: None,
        }
    }
}

/// Allow/deny policy for optional Siumai-owned response headers.
///
/// Protocol and credential headers are never forwarded through this policy.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GatewayHeaderPolicy {
    emit_route: bool,
    emit_projection_policy: bool,
    allowlist: Option<BTreeSet<String>>,
    denylist: BTreeSet<String>,
}

impl GatewayHeaderPolicy {
    pub fn with_route_header(mut self, enabled: bool) -> Self {
        self.emit_route = enabled;
        self
    }

    pub fn with_projection_policy_header(mut self, enabled: bool) -> Self {
        self.emit_projection_policy = enabled;
        self
    }

    pub fn with_allowlist<I, S>(mut self, names: I) -> Result<Self, GatewayPolicyError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.allowlist = Some(collect_header_names(names)?);
        Ok(self)
    }

    pub fn with_denylist<I, S>(mut self, names: I) -> Result<Self, GatewayPolicyError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.denylist = collect_header_names(names)?;
        Ok(self)
    }

    pub const fn emits_route(&self) -> bool {
        self.emit_route
    }

    pub const fn emits_projection_policy(&self) -> bool {
        self.emit_projection_policy
    }

    pub fn allows(&self, name: &str) -> bool {
        let normalized = name.to_ascii_lowercase();
        if self.denylist.contains(&normalized) {
            return false;
        }
        self.allowlist
            .as_ref()
            .is_none_or(|allowlist| allowlist.contains(&normalized))
    }
}

/// Complete server-side ingress and projection policy.
#[derive(Debug, Clone, Default)]
pub struct GatewayPolicy {
    limits: GatewayLimits,
    loss: GatewayLossPolicy,
    errors: GatewayErrorDetail,
    headers: GatewayHeaderPolicy,
    stream: GatewayStreamPolicy,
}

impl GatewayPolicy {
    pub fn with_limits(mut self, limits: GatewayLimits) -> Self {
        self.limits = limits;
        self
    }

    pub fn with_loss_policy(mut self, loss: GatewayLossPolicy) -> Self {
        self.loss = loss;
        self
    }

    pub fn with_error_detail(mut self, errors: GatewayErrorDetail) -> Self {
        self.errors = errors;
        self
    }

    pub fn with_header_policy(mut self, headers: GatewayHeaderPolicy) -> Self {
        self.headers = headers;
        self
    }

    pub fn with_stream_policy(mut self, stream: GatewayStreamPolicy) -> Self {
        self.stream = stream;
        self
    }

    pub const fn limits(&self) -> GatewayLimits {
        self.limits
    }

    pub const fn loss_policy(&self) -> GatewayLossPolicy {
        self.loss
    }

    pub const fn error_detail(&self) -> GatewayErrorDetail {
        self.errors
    }

    pub const fn headers(&self) -> &GatewayHeaderPolicy {
        &self.headers
    }

    pub const fn stream(&self) -> GatewayStreamPolicy {
        self.stream
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum GatewayPolicyError {
    #[error("gateway policy field `{field}` must be greater than zero")]
    ZeroLimit { field: &'static str },
    #[error("gateway policy field `{field}` exceeds the hard maximum of {maximum} bytes")]
    LimitTooLarge { field: &'static str, maximum: usize },
    #[error("gateway policy field `{field}` must be at least {minimum} bytes")]
    LimitTooSmall { field: &'static str, minimum: usize },
    #[error("gateway policy duration `{field}` must be greater than zero")]
    ZeroDuration { field: &'static str },
    #[error("gateway response header name is invalid")]
    InvalidHeaderName,
    #[error("gateway response header policy has more than {maximum} rules")]
    TooManyHeaderRules { maximum: usize },
}

fn validate_limit(field: &'static str, value: usize) -> Result<(), GatewayPolicyError> {
    if value == 0 {
        return Err(GatewayPolicyError::ZeroLimit { field });
    }
    if value > MAX_SERVER_BODY_LIMIT_BYTES {
        return Err(GatewayPolicyError::LimitTooLarge {
            field,
            maximum: MAX_SERVER_BODY_LIMIT_BYTES,
        });
    }
    Ok(())
}

fn validate_duration(
    field: &'static str,
    value: Option<Duration>,
) -> Result<(), GatewayPolicyError> {
    if value.is_some_and(|value| value.is_zero()) {
        return Err(GatewayPolicyError::ZeroDuration { field });
    }
    Ok(())
}

fn validate_response_limit(field: &'static str, value: usize) -> Result<(), GatewayPolicyError> {
    validate_limit(field, value)?;
    if value < MIN_SERVER_RESPONSE_LIMIT_BYTES {
        return Err(GatewayPolicyError::LimitTooSmall {
            field,
            minimum: MIN_SERVER_RESPONSE_LIMIT_BYTES,
        });
    }
    Ok(())
}

fn collect_header_names<I, S>(names: I) -> Result<BTreeSet<String>, GatewayPolicyError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut normalized = BTreeSet::new();
    for name in names {
        normalized.insert(normalize_header_name(name.as_ref())?);
        if normalized.len() > MAX_RESPONSE_HEADER_RULES {
            return Err(GatewayPolicyError::TooManyHeaderRules {
                maximum: MAX_RESPONSE_HEADER_RULES,
            });
        }
    }
    Ok(normalized)
}

fn normalize_header_name(name: &str) -> Result<String, GatewayPolicyError> {
    if name.is_empty()
        || name.len() > 128
        || !name.is_ascii()
        || !name.bytes().all(is_header_name_byte)
    {
        return Err(GatewayPolicyError::InvalidHeaderName);
    }
    Ok(name.to_ascii_lowercase())
}

fn is_header_name_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric()
        || matches!(
            byte,
            b'!' | b'#'
                | b'$'
                | b'%'
                | b'&'
                | b'\''
                | b'*'
                | b'+'
                | b'-'
                | b'.'
                | b'^'
                | b'_'
                | b'`'
                | b'|'
                | b'~'
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_finite_and_report_projection_loss() {
        let policy = GatewayPolicy::default();

        assert!(policy.limits().request_body_bytes() < usize::MAX);
        assert!(policy.limits().upstream_body_bytes() < usize::MAX);
        assert_eq!(policy.loss_policy(), GatewayLossPolicy::Report);
    }

    #[test]
    fn denylist_overrides_allowlist_case_insensitively() {
        let headers = GatewayHeaderPolicy::default()
            .with_allowlist(["X-Siumai-Route", "x-siumai-projection-policy"])
            .unwrap()
            .with_denylist(["x-SIUMAI-route"])
            .unwrap();

        assert!(!headers.allows("x-siumai-route"));
        assert!(headers.allows("X-Siumai-Projection-Policy"));
    }

    #[test]
    fn zero_and_excessive_limits_are_rejected() {
        assert!(matches!(
            GatewayLimits::new(0, 1, 1, 1),
            Err(GatewayPolicyError::ZeroLimit {
                field: "request_body_bytes"
            })
        ));
        assert!(matches!(
            GatewayLimits::new(
                1,
                1,
                MAX_SERVER_BODY_LIMIT_BYTES + 1,
                MIN_SERVER_RESPONSE_LIMIT_BYTES,
            ),
            Err(GatewayPolicyError::LimitTooLarge {
                field: "json_response_bytes",
                ..
            })
        ));
    }
}
