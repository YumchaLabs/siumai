#![cfg(feature = "groq")]

use siumai_provider_groq::builder::BuilderBase;
use siumai_provider_groq::execution::http::interceptor::LoggingInterceptor;
use siumai_provider_groq::providers::groq::{GroqBuilder, GroqConfig};
use std::sync::Arc;
use std::time::Duration;

#[test]
fn groq_builder_and_config_http_conveniences_stay_aligned() {
    let builder_config = GroqBuilder::new(BuilderBase::default())
        .api_key("test-key")
        .model("llama-3.3-70b-versatile")
        .timeout(Duration::from_secs(9))
        .connect_timeout(Duration::from_secs(3))
        .http_stream_disable_compression(true)
        .with_http_interceptor(Arc::new(LoggingInterceptor))
        .into_config()
        .expect("builder into_config");

    let manual_config = GroqConfig::new("test-key")
        .with_model("llama-3.3-70b-versatile")
        .with_timeout(Duration::from_secs(9))
        .with_connect_timeout(Duration::from_secs(3))
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
