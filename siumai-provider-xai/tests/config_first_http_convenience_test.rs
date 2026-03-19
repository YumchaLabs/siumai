#![cfg(feature = "xai")]

use siumai_provider_xai::builder::BuilderBase;
use siumai_provider_xai::execution::http::interceptor::LoggingInterceptor;
use siumai_provider_xai::providers::xai::{XaiBuilder, XaiConfig};
use std::sync::Arc;
use std::time::Duration;

#[test]
fn xai_builder_and_config_http_conveniences_stay_aligned() {
    let builder_config = XaiBuilder::new(BuilderBase::default())
        .api_key("test-key")
        .model("grok-2-1212")
        .timeout(Duration::from_secs(11))
        .connect_timeout(Duration::from_secs(4))
        .http_stream_disable_compression(true)
        .with_http_interceptor(Arc::new(LoggingInterceptor))
        .into_config()
        .expect("builder into_config");

    let manual_config = XaiConfig::new("test-key")
        .with_model("grok-2-1212")
        .with_timeout(Duration::from_secs(11))
        .with_connect_timeout(Duration::from_secs(4))
        .with_http_stream_disable_compression(true)
        .with_http_interceptor(Arc::new(LoggingInterceptor));

    assert_eq!(
        builder_config.common_params.model,
        manual_config.common_params.model
    );
    assert_eq!(
        builder_config.http_config.timeout,
        manual_config.http_config.timeout
    );
    assert_eq!(
        builder_config.http_config.connect_timeout,
        manual_config.http_config.connect_timeout
    );
    assert_eq!(
        builder_config.http_config.stream_disable_compression,
        manual_config.http_config.stream_disable_compression
    );
    assert_eq!(
        builder_config.http_interceptors.len(),
        manual_config.http_interceptors.len()
    );
}
