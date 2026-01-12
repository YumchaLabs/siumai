#![cfg(feature = "gcp")]

use siumai_core::auth::TokenProvider;
use siumai_core::auth::adc::AdcTokenProvider;
use siumai_core::auth::service_account::{ServiceAccountCredentials, ServiceAccountTokenProvider};

struct EnvGuard {
    key: &'static str,
    previous: Option<String>,
}

impl EnvGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var(key).ok();
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, previous }
    }

    fn remove(key: &'static str) -> Self {
        let previous = std::env::var(key).ok();
        unsafe {
            std::env::remove_var(key);
        }
        Self { key, previous }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match &self.previous {
            Some(v) => unsafe {
                std::env::set_var(self.key, v);
            },
            None => unsafe {
                std::env::remove_var(self.key);
            },
        }
    }
}

#[tokio::test]
async fn service_account_token_provider_exchanges_assertion_and_caches() {
    let mut server = mockito::Server::new_async().await;
    let _m = server
        .mock("POST", "/token")
        .match_header(
            "content-type",
            mockito::Matcher::Regex("^application/x-www-form-urlencoded".into()),
        )
        .match_body(mockito::Matcher::AllOf(vec![
            mockito::Matcher::Regex(
                "grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer".into(),
            ),
            mockito::Matcher::Regex("assertion=dummy-assertion".into()),
        ]))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"access_token":"mocked-token","token_type":"Bearer","expires_in":3600}"#)
        .expect(1)
        .create_async()
        .await;

    let creds = ServiceAccountCredentials {
        client_email: "svc@example.com".to_string(),
        private_key: "unused".to_string(),
        token_uri: Some(format!("{}/token", server.url())),
        scopes: vec![],
    };

    let provider = ServiceAccountTokenProvider::new_with_assertion_override(
        creds,
        reqwest::Client::new(),
        None,
        "dummy-assertion".to_string(),
    );

    let t1 = provider.token().await.expect("token 1");
    let t2 = provider.token().await.expect("token 2");
    assert_eq!(t1, "mocked-token");
    assert_eq!(t2, "mocked-token");
}

#[tokio::test]
async fn adc_token_provider_prefers_env_token() {
    let _g1 = EnvGuard::set("GOOGLE_OAUTH_ACCESS_TOKEN", "env-token");
    let _g2 = EnvGuard::remove("GOOGLE_APPLICATION_CREDENTIALS");
    let _g3 = EnvGuard::set("ADC_METADATA_URL", "http://example.invalid/token");

    let provider = AdcTokenProvider::default_client();
    let tok = provider.token().await.expect("token");
    assert_eq!(tok, "env-token");
}

#[tokio::test]
async fn adc_token_provider_falls_back_to_metadata() {
    let _g1 = EnvGuard::remove("GOOGLE_OAUTH_ACCESS_TOKEN");
    let _g2 = EnvGuard::remove("GOOGLE_APPLICATION_CREDENTIALS");

    let mut server = mockito::Server::new_async().await;
    let _m = server
        .mock("GET", "/metadata")
        .match_header("metadata-flavor", "Google")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"access_token":"md-token","expires_in":600}"#)
        .expect(1)
        .create_async()
        .await;

    let _g3 = EnvGuard::set("ADC_METADATA_URL", &format!("{}/metadata", server.url()));

    let provider = AdcTokenProvider::default_client();
    let t1 = provider.token().await.expect("token 1");
    let t2 = provider.token().await.expect("token 2");
    assert_eq!(t1, "md-token");
    assert_eq!(t2, "md-token");
}
