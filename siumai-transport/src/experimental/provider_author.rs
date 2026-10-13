//! Transport-owned assembly for provider authors.
//!
//! This module is intentionally experimental. It owns only the common
//! configured-versus-prebuilt transport resolution invariant; provider crates
//! retain credential validation, endpoint selection, replay declarations,
//! retry classification, and provider-specific error mapping.

use std::fmt;

pub use crate::error::ProviderTransportSourceError;
use crate::{
    EndpointConfig, ProviderHttpTransportSettings, ProviderTransport, ProviderTransportBuilder,
};

/// A single-use source for resolving one provider transport.
///
/// A configured source owns a complete [`ProviderTransportBuilder`]. A
/// prebuilt source returns its existing [`ProviderTransport`] after checking
/// the expected endpoint and policy. The source never performs network I/O,
/// accepts provider credentials, or interprets provider profiles.
pub struct ProviderTransportSource {
    state: SourceState,
    settings_override: Option<ProviderHttpTransportSettings>,
}

enum SourceState {
    Configured(Box<ProviderTransportBuilder>),
    Prebuilt(ProviderTransport),
}

impl ProviderTransportSource {
    /// Create a source that will consume and resolve a configured builder.
    pub fn configured(builder: ProviderTransportBuilder) -> Self {
        Self {
            state: SourceState::Configured(Box::new(builder)),
            settings_override: None,
        }
    }

    /// Create a source around an already configured transport.
    pub fn prebuilt(transport: ProviderTransport) -> Self {
        Self {
            state: SourceState::Prebuilt(transport),
            settings_override: None,
        }
    }

    /// Replace the complete HTTP settings snapshot.
    ///
    /// For a configured source, repeated calls use the last snapshot. For a
    /// prebuilt source, the override is recorded and rejected when resolution
    /// is consumed; this preserves the provider-facing delayed conflict
    /// ordering without mutating or rebuilding the prebuilt transport.
    pub fn with_http_transport_settings(mut self, settings: ProviderHttpTransportSettings) -> Self {
        self.settings_override = Some(settings);
        self
    }

    /// Consume the source and resolve it against the provider's expected
    /// typed endpoint and policy.
    pub fn resolve(
        self,
        expected_endpoint: &EndpointConfig,
    ) -> Result<ProviderTransport, ProviderTransportSourceError> {
        let Self {
            state,
            settings_override,
        } = self;

        let transport = match state {
            SourceState::Configured(builder) => {
                let builder = if let Some(settings) = settings_override {
                    builder.with_http_transport_settings(settings)
                } else {
                    *builder
                };
                builder.build()?
            }
            SourceState::Prebuilt(transport) => {
                if settings_override.is_some() {
                    return Err(ProviderTransportSourceError::SettingsConflict);
                }
                transport
            }
        };

        if !expected_endpoint.is_compatible_with(transport.endpoint()) {
            return Err(ProviderTransportSourceError::EndpointMismatch);
        }
        Ok(transport)
    }
}

impl fmt::Debug for ProviderTransportSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = match &self.state {
            SourceState::Configured(_) => "configured",
            SourceState::Prebuilt(_) => "prebuilt",
        };
        formatter
            .debug_struct("ProviderTransportSource")
            .field("state", &state)
            .field("settings_override", &self.settings_override.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::time::Duration;

    use crate::{EndpointPolicy, ProviderTransportBuilder, TransportLimits};

    fn local_endpoint(port: u16) -> EndpointConfig {
        EndpointConfig::local_explicit(format!("http://127.0.0.1:{port}/v1")).unwrap()
    }

    #[test]
    fn configured_source_is_single_use_and_applies_last_settings_snapshot() {
        let endpoint = local_endpoint(43101);
        let first = ProviderHttpTransportSettings::default()
            .with_limits(TransportLimits {
                max_response_bytes: 8 * 1024,
                ..TransportLimits::default()
            })
            .unwrap();
        let second = ProviderHttpTransportSettings::default()
            .with_limits(TransportLimits {
                max_response_bytes: 16 * 1024,
                ..TransportLimits::default()
            })
            .unwrap()
            .with_call_timeout(Duration::from_secs(42))
            .unwrap();

        let source =
            ProviderTransportSource::configured(ProviderTransportBuilder::new(endpoint.clone()))
                .with_http_transport_settings(first)
                .with_http_transport_settings(second);
        let debug = format!("{source:?}");
        assert!(debug.contains("configured"));
        let transport = source.resolve(&endpoint).unwrap();

        assert_eq!(transport.limits().max_response_bytes, 16 * 1024);
    }

    #[test]
    fn prebuilt_source_returns_the_same_runtime_and_checks_conflict_first() {
        let expected = local_endpoint(43102);
        let actual = local_endpoint(43103);
        let transport = ProviderTransportBuilder::new(actual).build().unwrap();
        let source = ProviderTransportSource::prebuilt(transport.clone());
        let conflict = source
            .with_http_transport_settings(ProviderHttpTransportSettings::default())
            .resolve(&expected)
            .unwrap_err();
        assert_eq!(conflict, ProviderTransportSourceError::SettingsConflict);

        let transport = ProviderTransportBuilder::new(expected.clone())
            .build()
            .unwrap();
        let source = ProviderTransportSource::prebuilt(transport.clone());
        let resolved = source.resolve(&expected).unwrap();
        assert!(transport.shares_runtime(&resolved));
    }

    #[test]
    fn endpoint_comparison_keeps_path_query_port_and_policy_exact() {
        let expected = EndpointConfig::new(
            "http://service.internal:43104/v1?tenant=one",
            EndpointPolicy::LocalExplicit(crate::LocalNetworkGrant::Loopback),
        )
        .unwrap();
        let same = EndpointConfig::new(
            "http://service.internal:43104/v1?tenant=one",
            EndpointPolicy::LocalExplicit(crate::LocalNetworkGrant::Loopback),
        )
        .unwrap();
        let different_path = EndpointConfig::new(
            "http://service.internal:43104/v2",
            EndpointPolicy::LocalExplicit(crate::LocalNetworkGrant::Loopback),
        )
        .unwrap();
        let different_query = EndpointConfig::new(
            "http://service.internal:43104/v1?tenant=two",
            EndpointPolicy::LocalExplicit(crate::LocalNetworkGrant::Loopback),
        )
        .unwrap();
        let different_policy = EndpointConfig::new(
            "http://service.internal:43104/v1?tenant=one",
            EndpointPolicy::LocalExplicit(crate::LocalNetworkGrant::PrivateNetwork),
        )
        .unwrap();

        assert!(expected.is_compatible_with(&same));
        assert!(!expected.is_compatible_with(&different_path));
        assert!(!expected.is_compatible_with(&different_query));
        assert!(!expected.is_compatible_with(&different_policy));
    }

    #[test]
    fn endpoint_comparison_preserves_url_normalization_and_trailing_slash_behavior() {
        let default_port = EndpointConfig::local_explicit("http://127.0.0.1/v1").unwrap();
        let explicit_default_port =
            EndpointConfig::local_explicit("http://127.0.0.1:80/v1").unwrap();
        let trailing_slash = EndpointConfig::local_explicit("http://127.0.0.1/v1/").unwrap();

        assert!(default_port.is_compatible_with(&explicit_default_port));
        assert!(!default_port.is_compatible_with(&trailing_slash));
    }

    #[test]
    fn source_debug_is_structural_and_does_not_include_endpoint_or_settings_details() {
        let canary = "127.0.0.1:43105";
        let endpoint =
            EndpointConfig::local_explicit(format!("http://{canary}/secret?token=canary")).unwrap();
        let source = ProviderTransportSource::configured(ProviderTransportBuilder::new(endpoint));
        let debug = format!("{source:?}");
        assert!(debug.contains("configured"));
        assert!(!debug.contains(canary));
        assert!(!debug.contains("canary"));
    }
}
