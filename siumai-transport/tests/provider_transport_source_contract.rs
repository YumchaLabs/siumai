use std::time::Duration;

use siumai_transport::experimental::provider_author::{
    ProviderTransportSource, ProviderTransportSourceError,
};
use siumai_transport::{
    EndpointConfig, HttpTransportRoute, ProviderHttpTransportSettings, ProviderTransport,
    ProxyEndpoint, TransportConfigError, TransportLimits,
};

fn local_endpoint(port: u16) -> EndpointConfig {
    EndpointConfig::local_explicit(format!("http://127.0.0.1:{port}/v1")).unwrap()
}

#[test]
fn configured_source_preserves_the_final_complete_settings_snapshot() {
    let endpoint = local_endpoint(43111);
    let first = ProviderHttpTransportSettings::default()
        .with_limits(TransportLimits {
            max_response_bytes: 8 * 1024,
            ..TransportLimits::default()
        })
        .unwrap();
    let final_settings = ProviderHttpTransportSettings::default()
        .with_limits(TransportLimits {
            max_response_bytes: 24 * 1024,
            ..TransportLimits::default()
        })
        .unwrap()
        .with_call_timeout(Duration::from_secs(43))
        .unwrap();

    let transport =
        ProviderTransportSource::configured(ProviderTransport::builder(endpoint.clone()))
            .with_http_transport_settings(first)
            .with_http_transport_settings(final_settings)
            .resolve(&endpoint)
            .unwrap();

    assert_eq!(
        transport.endpoint().expose_base_url(),
        endpoint.expose_base_url()
    );
    assert_eq!(transport.limits().max_response_bytes, 24 * 1024);
}

#[test]
fn prebuilt_source_reports_settings_conflict_before_endpoint_mismatch() {
    let expected = local_endpoint(43112);
    let prebuilt = ProviderTransport::builder(local_endpoint(43113))
        .build()
        .unwrap();

    let error = ProviderTransportSource::prebuilt(prebuilt)
        .with_http_transport_settings(ProviderHttpTransportSettings::default())
        .resolve(&expected)
        .unwrap_err();

    assert_eq!(error, ProviderTransportSourceError::SettingsConflict);
}

#[test]
fn source_preserves_transport_configuration_errors_without_leaking_details() {
    let endpoint = local_endpoint(43114);
    let route = HttpTransportRoute::trusted_connect(
        ProxyEndpoint::local_explicit("http://127.0.0.1:43115").unwrap(),
    );
    let settings = ProviderHttpTransportSettings::default()
        .with_route(route)
        .unwrap();

    let error = ProviderTransportSource::configured(ProviderTransport::builder(endpoint.clone()))
        .with_http_transport_settings(settings)
        .resolve(&endpoint)
        .unwrap_err();

    assert!(matches!(
        error,
        ProviderTransportSourceError::Transport(
            TransportConfigError::ProxyDestinationMustBePublicHttps
        )
    ));
    let debug = format!("{error:?}");
    assert!(!debug.contains("43114"));
    assert!(!debug.contains("43115"));
}
