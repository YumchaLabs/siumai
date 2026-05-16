use siumai_spec::types::HttpConfig;
use std::time::Duration;

#[test]
fn http_config_default_is_passive_data() {
    let config = HttpConfig::default();

    assert_eq!(config.timeout, None);
    assert_eq!(config.connect_timeout, None);
    assert!(config.headers.is_empty());
    assert_eq!(config.proxy, None);
    assert_eq!(config.user_agent, None);
    assert!(!config.stream_disable_compression);
}

#[test]
fn http_config_empty_matches_request_override_semantics() {
    let config = HttpConfig::empty();

    assert_eq!(config.timeout, None);
    assert_eq!(config.connect_timeout, None);
    assert!(config.headers.is_empty());
    assert_eq!(config.proxy, None);
    assert_eq!(config.user_agent, None);
    assert!(!config.stream_disable_compression);
}

#[test]
fn http_config_builder_does_not_apply_runtime_policy() {
    let config = HttpConfig::builder()
        .timeout(Some(Duration::from_secs(5)))
        .header("x-test", "1")
        .build();

    assert_eq!(config.timeout, Some(Duration::from_secs(5)));
    assert_eq!(config.connect_timeout, None);
    assert_eq!(config.user_agent, None);
    assert_eq!(config.headers.get("x-test").map(String::as_str), Some("1"));
    assert!(!config.stream_disable_compression);
}

#[test]
fn http_config_builder_preserves_explicit_stream_compression_policy() {
    let config = HttpConfig::builder()
        .stream_disable_compression(true)
        .build();

    assert!(config.stream_disable_compression);
}
