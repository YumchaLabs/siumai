//! Ensure ProviderHeaders::openai builds required headers

use siumai::execution::http::headers::ProviderHeaders;
use std::collections::HashMap;

#[test]
fn openai_headers_include_bearer_and_json() {
    let mut custom = HashMap::new();
    custom.insert("X-Custom".to_string(), "yes".to_string());
    let headers = ProviderHeaders::openai("test-key", Some("org"), Some("proj"), &custom).expect("ok");
    let auth = headers.get(reqwest::header::AUTHORIZATION).unwrap();
    assert_eq!(auth.to_str().unwrap(), "Bearer test-key");
    let ct = headers.get(reqwest::header::CONTENT_TYPE).unwrap();
    assert_eq!(ct.to_str().unwrap(), "application/json");
    // Organization/Project headers, if provided
    assert_eq!(
        headers.get("OpenAI-Organization").unwrap().to_str().unwrap(),
        "org"
    );
    assert_eq!(headers.get("OpenAI-Project").unwrap().to_str().unwrap(), "proj");
    assert_eq!(headers.get("X-Custom").unwrap().to_str().unwrap(), "yes");
}
