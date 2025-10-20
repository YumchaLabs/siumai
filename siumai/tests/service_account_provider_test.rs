use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use siumai::auth::TokenProvider;
use siumai::auth::service_account::{ServiceAccountCredentials, ServiceAccountTokenProvider};

// Minimal RSA private key for testing (do not use in production)
const TEST_RSA_PRIVATE_KEY: &str = r#"-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEAo1gMGQmtFbx3Vw8uGJQARk51EbqdmTYU7C3zchj3yKQ3QvI+
5qv+fY2j6qkI/8tR5L3ZI9D2p5qzV4VtqYkn1scxRkqS3wq9S4cG3V2NwKa6X1wZ
3hx7Qqv7JkPy6Wk8g0X9m7g3mN3T1v6m5Yv/O3cJmG5iZp1h8u0qGLoH7qj2n5nD
gQIDAQABAoIBAGvQ3r0cdx2oGxJRUb+7jZ8xf5G3Qx9x3b6vJk0k1J1x3mAjP0kR
V0G7vU8f2x8z37T8gP2LzvV4kxK6mQ6X7d4o8G9nsqzj5y8N3H3u6tV8m4a7G0iQ
GSkz3Qh9M2k2q8sC4ZlP4v+JzJp0j3yOfnx7Hf4w9cU0p7YqHkz2VbVb7C3q0wM9
E0ECgYEAzNVB3s+vD3Xsp6ZOkq5Zk4S9cC0V+3kzqV1m6Km2tYqjT4aYJYQz8v2m
uY2tKyt4O1R5sW2L8e9J8cHnI8hB+Qk5Lskk7mA3Yf3vF2b3xX5j22Xz6A3f8n1Y
u8JX1Hf7QqZzV3D2Zl0e0rA1yS5Yb8vZQf9Qz3r8L8C4qf8sCgYEAxk9W7wYkY3b
Jb9yGx9h3O7z6l6q8c6x5bU5xJ7yHf7Rk4w8k9s5rZ2uYb7r8t2f3n9u6v5x8y7s
W3Q2f4f7v9w3s6y5t4r8j7w7t6q8y9r4v2s5t7u9w3s0x9y7v6u5r4q2p1o9n8m7
VQ8CgYEAoV6xQGk2Z2FGz0bE5vV1bWm6oZVvZr2h4o7m6n8l5j2h3k4l5m6n7o8p
q9r8s7t6u5v4w3x2y1z0a9b8c7d6e5f4g3h2i1j0k9l8m7n6o5p4q3r2s1t0u9v8
WwECgYB4h1f2g3h4i5j6k7l8m9n0o1p2q3r4s5t6u7v8w9x0y1z2a3b4c5d6e7f8
g9h0i1j2k3l4m5n6o7p8q9r0s1t2u3v4w5x6y7z8A9B0C1D2E3F4G5H6I7J8K9L0
EQKBgQCw3v5X+testKeyForUnitTestsOnly00000000000000000000000000000
-----END RSA PRIVATE KEY-----"#;

#[tokio::test]
async fn service_account_token_provider_fetch_and_cache() {
    // Mock token endpoint
    let server = MockServer::start().await;
    let token_path = "/token";
    let token_url = format!("{}{}", &server.uri(), token_path);

    let template = ResponseTemplate::new(200).set_body_json(serde_json::json!({
        "access_token": "ya29.test-token",
        "token_type": "Bearer",
        "expires_in": 3600
    }));

    Mock::given(method("POST"))
        .and(path(token_path))
        .respond_with(template)
        .mount(&server)
        .await;

    let creds = ServiceAccountCredentials {
        client_email: "svc@test.iam.gserviceaccount.com".to_string(),
        private_key: TEST_RSA_PRIVATE_KEY.to_string(),
        token_uri: Some(token_url.clone()),
        scopes: vec!["https://www.googleapis.com/auth/cloud-platform".to_string()],
    };

    // Run blocking client and provider inside a blocking thread to avoid dropping
    // the blocking client in async context.
    tokio::task::spawn_blocking(move || {
        let http = reqwest::blocking::Client::new();
        // Use assertion override in tests to bypass RSA signing
        let provider = ServiceAccountTokenProvider::new_with_assertion_override(
            creds,
            http,
            None,
            "test-assertion".to_string(),
        );
        let t1 = provider.token().expect("token fetch should succeed");
        assert_eq!(t1, "ya29.test-token");
        let t2 = provider.token().expect("token cache should serve");
        assert_eq!(t2, "ya29.test-token");
    })
    .await
    .expect("spawn_blocking should succeed");
}
