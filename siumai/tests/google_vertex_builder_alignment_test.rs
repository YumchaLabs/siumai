#![cfg(feature = "google-vertex")]

use siumai::Provider;

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

#[test]
fn provider_vertex_builder_uses_express_base_url_when_api_key_is_set() {
    let _g1 = EnvGuard::remove("GOOGLE_VERTEX_API_KEY");
    let _g2 = EnvGuard::remove("GOOGLE_VERTEX_PROJECT");
    let _g3 = EnvGuard::remove("GOOGLE_VERTEX_LOCATION");

    let client = Provider::vertex()
        .api_key("test-api-key")
        .model("gemini-2.5-pro")
        .build()
        .expect("build vertex client");

    assert_eq!(
        client.base_url(),
        siumai::experimental::auth::vertex::GOOGLE_VERTEX_EXPRESS_BASE_URL
    );
}

#[test]
fn provider_vertex_builder_uses_env_project_location_when_no_base_url_and_no_api_key() {
    let _g1 = EnvGuard::remove("GOOGLE_VERTEX_API_KEY");
    let _g2 = EnvGuard::set("GOOGLE_VERTEX_PROJECT", "test-project");
    let _g3 = EnvGuard::set("GOOGLE_VERTEX_LOCATION", "us-central1");

    let client = Provider::vertex()
        .model("gemini-2.5-pro")
        .build()
        .expect("build vertex client");

    assert_eq!(
        client.base_url(),
        "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/test-project/locations/us-central1/publishers/google"
    );
}

#[test]
fn provider_vertex_builder_custom_base_url_overrides_defaults() {
    let _g1 = EnvGuard::set("GOOGLE_VERTEX_API_KEY", "env-key");
    let _g2 = EnvGuard::set("GOOGLE_VERTEX_PROJECT", "test-project");
    let _g3 = EnvGuard::set("GOOGLE_VERTEX_LOCATION", "us-central1");

    let client = Provider::vertex()
        .base_url("https://custom.example.com")
        .model("gemini-2.5-pro")
        .build()
        .expect("build vertex client");

    assert_eq!(client.base_url(), "https://custom.example.com");
}
