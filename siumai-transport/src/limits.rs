//! Resource limits applied consistently across provider transports.

use crate::TransportConfigError;

const MAX_BODY_BYTES: usize = 1024 * 1024 * 1024;
const MAX_HEADER_COUNT: usize = 1_024;
const MAX_HEADER_VALUE_BYTES: usize = 1024 * 1024;
const MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;
const MAX_EVENT_BYTES: usize = 64 * 1024 * 1024;
const MAX_EVENTS_PER_STREAM: usize = 10_000_000;
const MAX_MULTIPART_PARTS: usize = 1_024;
const MAX_REDIRECTS: usize = 32;
const MAX_QUEUED_REQUESTS: usize = 100_000;
const MAX_CONNECTIONS: usize = 10_000;
const MAX_IN_FLIGHT_REQUESTS: usize = 10_000;

/// Hard limits for one configured provider transport.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransportLimits {
    pub max_request_bytes: usize,
    pub max_response_bytes: usize,
    pub max_header_count: usize,
    pub max_header_value_bytes: usize,
    pub max_frame_bytes: usize,
    pub max_event_bytes: usize,
    pub max_events_per_stream: usize,
    pub max_multipart_parts: usize,
    pub max_redirects: usize,
    pub max_queued_requests: usize,
    pub max_connections: usize,
    pub max_in_flight_requests: usize,
}

impl Default for TransportLimits {
    fn default() -> Self {
        Self {
            max_request_bytes: 64 * 1024 * 1024,
            max_response_bytes: 128 * 1024 * 1024,
            max_header_count: 128,
            max_header_value_bytes: 16 * 1024,
            max_frame_bytes: 2 * 1024 * 1024,
            max_event_bytes: 8 * 1024 * 1024,
            max_events_per_stream: 1_000_000,
            max_multipart_parts: 64,
            max_redirects: 4,
            max_queued_requests: 256,
            max_connections: 64,
            max_in_flight_requests: 64,
        }
    }
}

impl TransportLimits {
    /// Validate limits once when a configured provider runtime is built.
    pub fn validate(&self) -> Result<(), TransportConfigError> {
        for (name, value, maximum) in [
            ("max_request_bytes", self.max_request_bytes, MAX_BODY_BYTES),
            (
                "max_response_bytes",
                self.max_response_bytes,
                MAX_BODY_BYTES,
            ),
            ("max_header_count", self.max_header_count, MAX_HEADER_COUNT),
            (
                "max_header_value_bytes",
                self.max_header_value_bytes,
                MAX_HEADER_VALUE_BYTES,
            ),
            ("max_frame_bytes", self.max_frame_bytes, MAX_FRAME_BYTES),
            ("max_event_bytes", self.max_event_bytes, MAX_EVENT_BYTES),
            (
                "max_events_per_stream",
                self.max_events_per_stream,
                MAX_EVENTS_PER_STREAM,
            ),
            (
                "max_multipart_parts",
                self.max_multipart_parts,
                MAX_MULTIPART_PARTS,
            ),
            (
                "max_queued_requests",
                self.max_queued_requests,
                MAX_QUEUED_REQUESTS,
            ),
            ("max_connections", self.max_connections, MAX_CONNECTIONS),
            (
                "max_in_flight_requests",
                self.max_in_flight_requests,
                MAX_IN_FLIGHT_REQUESTS,
            ),
        ] {
            if value == 0 {
                return Err(TransportConfigError::ZeroLimit { name });
            }
            if value > maximum {
                return Err(TransportConfigError::LimitTooLarge { name, maximum });
            }
        }
        if self.max_redirects > MAX_REDIRECTS {
            return Err(TransportConfigError::LimitTooLarge {
                name: "max_redirects",
                maximum: MAX_REDIRECTS,
            });
        }
        if self.max_in_flight_requests > self.max_connections {
            return Err(TransportConfigError::InFlightExceedsConnections);
        }
        let admission_capacity = self
            .max_queued_requests
            .checked_add(self.max_in_flight_requests)
            .ok_or(TransportConfigError::CapacityOverflow)?;
        if admission_capacity > tokio::sync::Semaphore::MAX_PERMITS {
            return Err(TransportConfigError::AdmissionCapacityTooLarge {
                maximum: tokio::sync::Semaphore::MAX_PERMITS,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        TransportLimits::default().validate().unwrap();
    }

    #[test]
    fn zero_capacity_is_rejected() {
        let limits = TransportLimits {
            max_in_flight_requests: 0,
            ..TransportLimits::default()
        };
        assert_eq!(
            limits.validate(),
            Err(TransportConfigError::ZeroLimit {
                name: "max_in_flight_requests"
            })
        );
    }

    #[test]
    fn excessive_resource_limit_is_rejected() {
        let limits = TransportLimits {
            max_response_bytes: MAX_BODY_BYTES + 1,
            ..TransportLimits::default()
        };
        assert_eq!(
            limits.validate(),
            Err(TransportConfigError::LimitTooLarge {
                name: "max_response_bytes",
                maximum: MAX_BODY_BYTES,
            })
        );
    }

    #[test]
    fn excessive_semaphore_capacity_is_rejected_without_panicking() {
        let limits = TransportLimits {
            max_queued_requests: usize::MAX,
            max_connections: usize::MAX,
            max_in_flight_requests: usize::MAX,
            ..TransportLimits::default()
        };
        assert_eq!(
            limits.validate(),
            Err(TransportConfigError::LimitTooLarge {
                name: "max_queued_requests",
                maximum: MAX_QUEUED_REQUESTS,
            })
        );
    }

    #[test]
    fn zero_redirects_are_allowed_but_excessive_redirects_are_rejected() {
        let no_redirects = TransportLimits {
            max_redirects: 0,
            ..TransportLimits::default()
        };
        no_redirects.validate().unwrap();

        let excessive = TransportLimits {
            max_redirects: MAX_REDIRECTS + 1,
            ..TransportLimits::default()
        };
        assert_eq!(
            excessive.validate(),
            Err(TransportConfigError::LimitTooLarge {
                name: "max_redirects",
                maximum: MAX_REDIRECTS,
            })
        );
    }
}
