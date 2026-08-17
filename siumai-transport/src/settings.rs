//! Provider stateless-HTTP transport settings.

use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::transport::TransportObserver;
use crate::{RetryPolicy, TransportConfigError, TransportLimits};

const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const DEFAULT_CALL_TIMEOUT: Duration = Duration::from_secs(15 * 60);
const DEFAULT_READ_TIMEOUT: Duration = Duration::from_secs(5 * 60);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProviderHttpRoute {
    Direct,
}

#[derive(Debug, Default)]
struct NoopObserver;

impl TransportObserver for NoopObserver {
    fn observe(&self, _event: &crate::TransportEvent) {}
}

/// Immutable settings for one provider-owned stateless HTTP transport.
///
/// This value configures only [`crate::ProviderTransport`]. Provider WebSocket,
/// Realtime, media-session, external-download, and MCP transports retain their
/// own lifecycle-specific settings and never inherit this value implicitly.
/// Endpoint policy, authentication, DNS resolution, retry classification, and
/// per-request replay proof also remain explicit transport-builder inputs.
#[derive(Clone)]
pub struct ProviderHttpTransportSettings {
    limits: TransportLimits,
    retry_policy: RetryPolicy,
    connect_timeout: Duration,
    call_timeout: Duration,
    read_timeout: Duration,
    observer: Arc<dyn TransportObserver>,
    observer_configured: bool,
    route: ProviderHttpRoute,
}

impl ProviderHttpTransportSettings {
    /// Replace the bounded transport limits after validating their hard caps.
    ///
    /// Only fields applicable to provider stateless HTTP are consumed by
    /// [`crate::ProviderTransport`]. This does not propagate the value to any
    /// WebSocket, resource-download, or MCP transport.
    pub fn with_limits(mut self, limits: TransportLimits) -> Result<Self, TransportConfigError> {
        limits.validate()?;
        self.limits = limits;
        Ok(self)
    }

    /// Replace the provider-owned maximum retry policy.
    ///
    /// Replay safety and caller attempt caps can only narrow this policy when a
    /// request is executed.
    pub fn with_retry_policy(mut self, retry_policy: RetryPolicy) -> Self {
        self.retry_policy = retry_policy;
        self
    }

    /// Replace the provider HTTP connection timeout.
    pub fn with_connect_timeout(mut self, timeout: Duration) -> Result<Self, TransportConfigError> {
        validate_timeout("connect_timeout", timeout)?;
        self.connect_timeout = timeout;
        Ok(self)
    }

    /// Replace the total provider HTTP call timeout.
    pub fn with_call_timeout(mut self, timeout: Duration) -> Result<Self, TransportConfigError> {
        validate_timeout("call_timeout", timeout)?;
        self.call_timeout = timeout;
        Ok(self)
    }

    /// Replace the provider HTTP socket read timeout.
    pub fn with_read_timeout(mut self, timeout: Duration) -> Result<Self, TransportConfigError> {
        validate_timeout("read_timeout", timeout)?;
        self.read_timeout = timeout;
        Ok(self)
    }

    /// Install a synchronous, payload-free observer for provider HTTP attempts.
    pub fn with_observer(mut self, observer: Arc<dyn TransportObserver>) -> Self {
        self.observer = observer;
        self.observer_configured = true;
        self
    }

    pub fn limits(&self) -> &TransportLimits {
        &self.limits
    }

    pub fn retry_policy(&self) -> RetryPolicy {
        self.retry_policy
    }

    pub fn connect_timeout(&self) -> Duration {
        self.connect_timeout
    }

    pub fn call_timeout(&self) -> Duration {
        self.call_timeout
    }

    pub fn read_timeout(&self) -> Duration {
        self.read_timeout
    }

    pub(crate) fn observer(&self) -> &dyn TransportObserver {
        self.observer.as_ref()
    }

    pub(crate) fn route(&self) -> ProviderHttpRoute {
        self.route
    }

    pub(crate) fn validate(&self) -> Result<(), TransportConfigError> {
        self.limits.validate()?;
        for (name, timeout) in [
            ("connect_timeout", self.connect_timeout),
            ("call_timeout", self.call_timeout),
            ("read_timeout", self.read_timeout),
        ] {
            validate_timeout(name, timeout)?;
        }
        Ok(())
    }
}

impl Default for ProviderHttpTransportSettings {
    fn default() -> Self {
        Self {
            limits: TransportLimits::default(),
            retry_policy: RetryPolicy::default(),
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            call_timeout: DEFAULT_CALL_TIMEOUT,
            read_timeout: DEFAULT_READ_TIMEOUT,
            observer: Arc::new(NoopObserver),
            observer_configured: false,
            route: ProviderHttpRoute::Direct,
        }
    }
}

impl fmt::Debug for ProviderHttpTransportSettings {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderHttpTransportSettings")
            .field("limits", &self.limits)
            .field("retry_policy", &self.retry_policy)
            .field("connect_timeout", &self.connect_timeout)
            .field("call_timeout", &self.call_timeout)
            .field("read_timeout", &self.read_timeout)
            .field("route", &self.route)
            .field("observer_configured", &self.observer_configured)
            .finish()
    }
}

fn validate_timeout(name: &'static str, timeout: Duration) -> Result<(), TransportConfigError> {
    if timeout.is_zero() {
        return Err(TransportConfigError::ZeroTimeout { name });
    }
    if Instant::now().checked_add(timeout).is_none() {
        return Err(TransportConfigError::TimeoutTooLarge { name });
    }
    Ok(())
}
