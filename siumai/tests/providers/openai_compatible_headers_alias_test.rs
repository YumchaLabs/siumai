//! OpenAI-Compatible provider header alias policy tests
//!
//! 仅在启用 `provider-openai-compatible-external` 时编译，直接调用外部 helpers，
//! 验证保守别名策略（例如 OpenRouter 的 HTTP-Referer 复制）。

#![cfg(all(feature = "openai-compatible", feature = "provider-openai-compatible-external"))]

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use std::collections::HashMap;

#[test]
fn openrouter_copies_referer_to_http_referer() {
    let mut extra = HashMap::new();
    // 提供标准 Referer
    extra.insert("Referer".to_string(), "https://example.com".to_string());

    let config_headers = HeaderMap::new();
    let adapter_headers = HeaderMap::new();

    let headers = siumai_provider_openai_compatible::helpers::build_json_headers_with_provider(
        "openrouter",
        "test-key",
        &extra,
        &config_headers,
        &adapter_headers,
    )
    .expect("headers build should succeed");

    // 应包含 Authorization 与 Content-Type/Accept
    assert!(headers.contains_key("authorization"));
    assert_eq!(
        headers.get("authorization").unwrap().to_str().unwrap(),
        "Bearer test-key"
    );

    // 应复制 Referer -> HTTP-Referer
    let http_referer = HeaderName::from_static("http-referer");
    assert!(headers.contains_key(&http_referer));
    assert_eq!(
        headers.get(&http_referer).unwrap().to_str().unwrap(),
        "https://example.com"
    );
}

#[test]
fn openrouter_does_not_inject_http_referer_without_referer() {
    let extra = HashMap::new();
    let config_headers = HeaderMap::new();
    let adapter_headers = HeaderMap::new();

    let headers = siumai_provider_openai_compatible::helpers::build_json_headers_with_provider(
        "openrouter",
        "test-key",
        &extra,
        &config_headers,
        &adapter_headers,
    )
    .expect("headers build should succeed");

    // 未提供 Referer 时，不应注入 HTTP-Referer
    let http_referer = HeaderName::from_static("http-referer");
    assert!(!headers.contains_key(&http_referer));
}

#[test]
fn deepseek_no_alias_injection() {
    let extra = HashMap::new();
    let config_headers = HeaderMap::new();
    let adapter_headers = HeaderMap::new();

    let headers = siumai_provider_openai_compatible::helpers::build_json_headers_with_provider(
        "deepseek",
        "test-key",
        &extra,
        &config_headers,
        &adapter_headers,
    )
    .expect("headers build should succeed");

    // 仅应包含标准头部（Authorization/Content-Type/Accept），不应注入别名头
    assert!(headers.contains_key("authorization"));
    assert_eq!(
        headers.get("authorization").unwrap().to_str().unwrap(),
        "Bearer test-key"
    );
    let http_referer = HeaderName::from_static("http-referer");
    assert!(!headers.contains_key(&http_referer));
}

#[test]
fn siliconflow_no_alias_injection() {
    let mut extra = HashMap::new();
    // 提供一个自定义头，确保透传
    extra.insert("X-Title".to_string(), "demo".to_string());
    let config_headers = HeaderMap::new();
    let adapter_headers = HeaderMap::new();

    let headers = siumai_provider_openai_compatible::helpers::build_json_headers_with_provider(
        "siliconflow",
        "test-key",
        &extra,
        &config_headers,
        &adapter_headers,
    )
    .expect("headers build should succeed");

    // 透传 X-Title，不额外注入 HTTP-Referer 等
    assert_eq!(
        headers.get("X-Title").unwrap().to_str().unwrap(),
        "demo"
    );
    let http_referer = HeaderName::from_static("http-referer");
    assert!(!headers.contains_key(&http_referer));
}

#[test]
fn groq_no_alias_injection() {
    let extra = HashMap::new();
    let config_headers = HeaderMap::new();
    let adapter_headers = HeaderMap::new();

    let headers = siumai_provider_openai_compatible::helpers::build_json_headers_with_provider(
        "groq",
        "test-key",
        &extra,
        &config_headers,
        &adapter_headers,
    )
    .expect("headers build should succeed");

    // 默认仅标准头，不注入别名
    assert!(headers.contains_key("authorization"));
    let http_referer = HeaderName::from_static("http-referer");
    assert!(!headers.contains_key(&http_referer));
}
